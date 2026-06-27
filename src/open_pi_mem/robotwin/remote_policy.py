from __future__ import annotations

import logging
import time
from typing import Any, Callable

import numpy as np

from open_pi_mem.robotwin.mem_controller import (
    Pi05MEMController,
    RoboTwinMEMPlanner,
    encode_pi05_observation,
)

logger = logging.getLogger(__name__)


def empty_action_response() -> dict[str, np.ndarray]:
    return {"actions": np.zeros((0, 14), dtype=np.float32)}


def actions_response(actions: Any) -> dict[str, np.ndarray]:
    return {"actions": np.asarray(actions, dtype=np.float32)}


def extract_actions(response: Any) -> np.ndarray:
    if isinstance(response, dict):
        response = response["actions"]
    return np.asarray(response, dtype=np.float32)


class RobotwinRemotePolicyClient:
    """OpenPI websocket client with RoboTwin episode reset support."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        client_factory: Callable[[str, int], Any] | None = None,
        ping_interval_s: float | None = None,
        ping_timeout_s: float | None = None,
    ) -> None:
        if client_factory is None:
            self._client = _OpenPIWebsocketClient(
                host,
                int(port),
                ping_interval_s=ping_interval_s,
                ping_timeout_s=ping_timeout_s,
            )
        else:
            self._client = client_factory(host, int(port))

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._client.infer(request)

    def reset(self) -> None:
        self._client.infer({"reset": True})


class _OpenPIWebsocketClient:
    """OpenPI websocket wire protocol with slow-inference-safe keepalive settings."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        connect_factory: Callable[..., Any] | None = None,
        packer_factory: Callable[[], Any] | None = None,
        unpackb: Callable[[bytes], Any] | None = None,
        ping_interval_s: float | None = None,
        ping_timeout_s: float | None = None,
        connect_retry_sleep_s: float = 5.0,
    ) -> None:
        self._uri = f"ws://{host}:{port}"
        if connect_factory is None:
            import websockets.sync.client

            connect_factory = websockets.sync.client.connect
        if packer_factory is None or unpackb is None:
            from openpi_client import msgpack_numpy

            packer_factory = packer_factory or msgpack_numpy.Packer
            unpackb = unpackb or msgpack_numpy.unpackb

        self._connect_factory = connect_factory
        self._packer = packer_factory()
        self._unpackb = unpackb
        self._ping_interval_s = ping_interval_s
        self._ping_timeout_s = ping_timeout_s
        self._connect_retry_sleep_s = connect_retry_sleep_s
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> dict[str, Any]:
        return self._server_metadata

    def _wait_for_server(self) -> tuple[Any, dict[str, Any]]:
        logger.info("Waiting for server at %s...", self._uri)
        while True:
            try:
                conn = self._connect_factory(
                    self._uri,
                    compression=None,
                    max_size=None,
                    ping_interval=self._ping_interval_s,
                    ping_timeout=self._ping_timeout_s,
                )
                return conn, self._unpackb(conn.recv())
            except ConnectionRefusedError:
                logger.info("Still waiting for server...")
                time.sleep(self._connect_retry_sleep_s)

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        try:
            self._ws.send(self._packer.pack(obs))
        except Exception as exc:
            if not _is_websocket_connection_closed(exc):
                raise
            self._ws, self._server_metadata = self._wait_for_server()
            self._ws.send(self._packer.pack(obs))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return self._unpackb(response)

    def reset(self) -> None:
        pass


def _is_websocket_connection_closed(exc: Exception) -> bool:
    try:
        import websockets

        if isinstance(exc, websockets.ConnectionClosed):
            return True
    except Exception:
        pass
    return exc.__class__.__name__.startswith("ConnectionClosed")


def serve_openpi_policy_forever(
    policy: Any,
    *,
    host: str = "0.0.0.0",
    port: int | None = None,
    metadata: dict[str, Any] | None = None,
    ping_interval_s: float | None = None,
    ping_timeout_s: float | None = None,
) -> None:
    """Serve an OpenPI-compatible policy without websocket keepalive timeouts.

    RoboTwin smoke tests often run pi05 on CPU when the local GPU cannot fit the
    model. A single inference can exceed the websocket library's default
    keepalive interval, so the RoboTwin server owns these connection settings.
    """

    import asyncio

    asyncio.run(
        _run_openpi_policy_server(
            policy,
            host=host,
            port=port,
            metadata=metadata or {},
            ping_interval_s=ping_interval_s,
            ping_timeout_s=ping_timeout_s,
        )
    )


async def _run_openpi_policy_server(
    policy: Any,
    *,
    host: str,
    port: int | None,
    metadata: dict[str, Any],
    ping_interval_s: float | None,
    ping_timeout_s: float | None,
) -> None:
    import http

    import websockets.asyncio.server as websocket_server

    def health_check(
        connection: websocket_server.ServerConnection,
        request: websocket_server.Request,
    ) -> websocket_server.Response | None:
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        return None

    async with websocket_server.serve(
        lambda websocket: _openpi_policy_handler(websocket, policy, metadata),
        host,
        port,
        compression=None,
        max_size=None,
        ping_interval=ping_interval_s,
        ping_timeout=ping_timeout_s,
        process_request=health_check,
    ) as server:
        await server.serve_forever()


async def _openpi_policy_handler(
    websocket: Any,
    policy: Any,
    metadata: dict[str, Any],
) -> None:
    import time as _time
    import traceback

    import websockets
    import websockets.frames
    from openpi_client import msgpack_numpy

    logger.info("Connection from %s opened", websocket.remote_address)
    packer = msgpack_numpy.Packer()

    await websocket.send(packer.pack(metadata))

    prev_total_time = None
    while True:
        try:
            start_time = _time.monotonic()
            obs = msgpack_numpy.unpackb(await websocket.recv())

            infer_time = _time.monotonic()
            action = policy.infer(obs)
            infer_time = _time.monotonic() - infer_time

            action["server_timing"] = {"infer_ms": infer_time * 1000}
            if prev_total_time is not None:
                action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

            await websocket.send(packer.pack(action))
            prev_total_time = _time.monotonic() - start_time
        except websockets.ConnectionClosed:
            logger.info("Connection from %s closed", websocket.remote_address)
            break
        except Exception:
            await websocket.send(traceback.format_exc())
            await websocket.close(
                code=websockets.frames.CloseCode.INTERNAL_ERROR,
                reason="Internal server error. Traceback included in previous frame.",
            )
            raise


class RobotwinPi05RemotePolicy:
    """Server-side pi05 baseline adapter for OpenPI's websocket server."""

    def __init__(self, pi05_policy: Any, *, action_steps: int | None = None) -> None:
        self.pi05_policy = pi05_policy
        self.action_steps = action_steps

    def reset(self) -> None:
        reset = getattr(self.pi05_policy, "reset_obsrvationwindows", None)
        if callable(reset):
            reset()

    def infer(self, request: dict[str, Any]) -> dict[str, np.ndarray]:
        if request.get("reset"):
            self.reset()
            return empty_action_response()

        if getattr(self.pi05_policy, "observation_window", None) is None:
            self.pi05_policy.set_language(str(request["goal"]))

        images, state = encode_pi05_observation(request["observation"])
        self.pi05_policy.update_observation_window(images, state)
        actions = np.asarray(self.pi05_policy.get_action(), dtype=np.float32)
        if self.action_steps is not None:
            actions = actions[: self.action_steps]
        return actions_response(actions)


class RobotwinPi06RemotePolicy:
    """Server-side open-pi-mem pi06 adapter for OpenPI's websocket server."""

    def __init__(
        self,
        pi05_policy: Any,
        *,
        planner: RoboTwinMEMPlanner | None = None,
        plan_interval_steps: int = 1,
        action_steps: int | None = None,
    ) -> None:
        self.controller = Pi05MEMController(
            pi05_policy,
            planner=planner,
            plan_interval_steps=plan_interval_steps,
            action_steps=action_steps,
        )

    def reset(self) -> None:
        self.controller.reset()

    def infer(self, request: dict[str, Any]) -> dict[str, np.ndarray]:
        if request.get("reset"):
            self.reset()
            return empty_action_response()
        return actions_response(self.controller.act_request(request))
