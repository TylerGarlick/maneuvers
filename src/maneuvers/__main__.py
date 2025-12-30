"""Simple CLI for maneuvers"""

from __future__ import annotations
import argparse


def main(argv: list[str] | None = None) -> int:
    """Entry point for the maneuvers CLI.

    Usage:
        maneuvers --name Alice
        python -m maneuvers --name Bob
    """
    parser = argparse.ArgumentParser(prog="maneuvers", description="A tiny demo CLI")
    parser.add_argument("--name", "-n", default="world", help="Name to greet")
    args = parser.parse_args(argv)

    print(f"Hello, {args.name}! Welcome to maneuvers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
