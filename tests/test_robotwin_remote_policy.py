from __future__ import annotations

import numpy as np

from open_pi_mem.robotwin.mem_controller import PassthroughMEMPlanner
from open_pi_mem.robotwin.remote_policy import (
    _OpenPIWebsocketClient,
    RobotwinPi05RemotePolicy,
    RobotwinPi06RemotePolicy,
    RobotwinRemotePolicyClient,
)


def _observation(value: int = 0) -> dict:
    rgb = np.zeros((3, 4, 3), dtype=np.uint8) + value
    return {
        "observation": {
            "head_camera": {"rgb": rgb},
            "right_camera": {"rgb": rgb + 1},
            "left_camera": {"rgb": rgb + 2},
        },
        "joint_action": {"vector": np.arange(14, dtype=np.float32)},
    }


class _FakePi05Policy:
    def __init__(self) -> None:
        self.languages = []
        self.updates = []
        self.reset_count = 0

    def set_language(self, instruction: str) -> None:
        self.languages.append(instruction)

    def update_observation_window(self, images, state) -> None:
        self.updates.append((images, state))

    def get_action(self):
        return np.array([[0.1] * 14, [0.2] * 14], dtype=np.float32)

    def reset_obsrvationwindows(self) -> None:
        self.reset_count += 1


def test_pi05_remote_policy_infers_action_chunk_and_resets() -> None:
    pi05 = _FakePi05Policy()
    policy = RobotwinPi05RemotePolicy(pi05, action_steps=1)

    response = policy.infer({"goal": "pick can", "observation": _observation()})
    reset_response = policy.infer({"reset": True})

    np.testing.assert_allclose(response["actions"], np.array([[0.1] * 14], dtype=np.float32))
    assert reset_response["actions"].shape == (0, 14)
    assert pi05.languages == ["pick can"]
    assert len(pi05.updates) == 1
    assert pi05.reset_count == 1


def test_pi06_remote_policy_runs_mem_controller_on_server_side() -> None:
    pi05 = _FakePi05Policy()
    policy = RobotwinPi06RemotePolicy(
        pi05,
        planner=PassthroughMEMPlanner(),
        plan_interval_steps=1,
        action_steps=1,
    )

    response = policy.infer({"goal": "place can", "observation": _observation()})
    reset_response = policy.infer({"reset": True})

    np.testing.assert_allclose(response["actions"], np.array([[0.1] * 14], dtype=np.float32))
    assert reset_response["actions"].shape == (0, 14)
    assert pi05.languages == ["place can"]
    assert pi05.reset_count == 1


def test_remote_policy_client_sends_reset_request_to_websocket_server() -> None:
    class FakeWebsocketClient:
        def __init__(self, host: str, port: int) -> None:
            self.host = host
            self.port = port
            self.requests = []

        def infer(self, request):
            self.requests.append(request)
            return {"actions": np.zeros((0, 14), dtype=np.float32)}

    created = []

    def factory(host: str, port: int):
        client = FakeWebsocketClient(host, port)
        created.append(client)
        return client

    client = RobotwinRemotePolicyClient("127.0.0.1", 8123, client_factory=factory)
    response = client.infer({"goal": "g", "observation": _observation()})
    client.reset()

    assert response["actions"].shape == (0, 14)
    assert created[0].host == "127.0.0.1"
    assert created[0].port == 8123
    assert created[0].requests[0]["goal"] == "g"
    np.testing.assert_array_equal(
        created[0].requests[0]["observation"]["joint_action"]["vector"],
        _observation()["joint_action"]["vector"],
    )
    assert created[0].requests[1] == {"reset": True}


def test_openpi_websocket_client_disables_keepalive_for_slow_cpu_inference() -> None:
    class FakePacker:
        def pack(self, request):
            return {"packed": request}

    class FakeConnection:
        def __init__(self) -> None:
            self.sent = []
            self.responses = [b"metadata", b"response"]

        def recv(self):
            return self.responses.pop(0)

        def send(self, payload):
            self.sent.append(payload)

    captured = {}
    connection = FakeConnection()

    def connect(uri, **kwargs):
        captured["uri"] = uri
        captured["kwargs"] = kwargs
        return connection

    def unpackb(payload):
        if payload == b"metadata":
            return {"server": "ready"}
        if payload == b"response":
            return {"actions": np.zeros((1, 14), dtype=np.float32)}
        raise AssertionError(f"unexpected payload: {payload!r}")

    client = _OpenPIWebsocketClient(
        "127.0.0.1",
        8105,
        connect_factory=connect,
        packer_factory=FakePacker,
        unpackb=unpackb,
        connect_retry_sleep_s=0,
    )
    response = client.infer({"goal": "pick", "observation": _observation()})

    assert captured["uri"] == "ws://127.0.0.1:8105"
    assert captured["kwargs"]["compression"] is None
    assert captured["kwargs"]["max_size"] is None
    assert captured["kwargs"]["ping_interval"] is None
    assert captured["kwargs"]["ping_timeout"] is None
    assert client.get_server_metadata() == {"server": "ready"}
    assert connection.sent[0]["packed"]["goal"] == "pick"
    assert response["actions"].shape == (1, 14)


def test_openpi_websocket_client_reconnects_when_socket_is_closed_before_send() -> None:
    class ConnectionClosedError(Exception):
        pass

    class FakePacker:
        def pack(self, request):
            return {"packed": request}

    class FakeConnection:
        def __init__(self, *, fail_send: bool) -> None:
            self.fail_send = fail_send
            self.sent = []
            self.responses = [b"metadata", b"response"]

        def recv(self):
            return self.responses.pop(0)

        def send(self, payload):
            if self.fail_send:
                raise ConnectionClosedError("closed")
            self.sent.append(payload)

    connections = [
        FakeConnection(fail_send=True),
        FakeConnection(fail_send=False),
    ]

    def connect(_uri, **_kwargs):
        return connections.pop(0)

    def unpackb(payload):
        if payload == b"metadata":
            return {"server": "ready"}
        if payload == b"response":
            return {"actions": np.zeros((1, 14), dtype=np.float32)}
        raise AssertionError(f"unexpected payload: {payload!r}")

    client = _OpenPIWebsocketClient(
        "127.0.0.1",
        8105,
        connect_factory=connect,
        packer_factory=FakePacker,
        unpackb=unpackb,
        connect_retry_sleep_s=0,
    )
    response = client.infer({"goal": "pick", "observation": _observation()})

    assert response["actions"].shape == (1, 14)
    assert connections == []
