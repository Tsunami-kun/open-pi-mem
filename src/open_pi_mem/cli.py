from __future__ import annotations

import argparse

from open_pi_mem import __version__


def main() -> None:
    parser = argparse.ArgumentParser(prog="open-pi-mem")
    parser.add_argument("--version", action="store_true")
    args = parser.parse_args()
    if args.version:
        print(f"open-pi-mem {__version__}")
        return
    parser.print_help()


if __name__ == "__main__":
    main()
