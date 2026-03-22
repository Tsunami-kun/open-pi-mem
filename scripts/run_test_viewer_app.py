from __future__ import annotations

import argparse
import functools
import http.server
import subprocess
import sys
from pathlib import Path


def _build_site(repo_root: Path, site_dir: Path) -> None:
    build_script = repo_root / "scripts" / "build_github_pages_site.py"
    subprocess.run(
        [sys.executable, str(build_script), "--output", str(site_dir)],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the static GitHub Pages viewer bundle locally.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument(
        "--site-dir",
        default="dist/github_pages",
        help="Directory containing the staged static viewer site.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the static site before serving.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    site_dir = (repo_root / args.site_dir).resolve()
    if args.rebuild or not site_dir.exists():
        _build_site(repo_root, site_dir)

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(site_dir))
    with http.server.ThreadingHTTPServer((args.host, args.port), handler) as httpd:
        print(f"Serving static viewer at http://{args.host}:{args.port}")
        print("Bundled reports will appear automatically on the home page.")
        print("You can also open a specific report with ?report=reports/<model>/<task>/report.json")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
