#!/usr/bin/env python3
"""Fetch the official NCRL repository without redistributing its source."""

import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    target = args.target.resolve()
    if target.exists() and any(target.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty directory: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "clone",
            "https://github.com/vivian1993/NCRL.git",
            str(target),
        ],
        check=True,
    )
    print(f"NCRL fetched to {target}")


if __name__ == "__main__":
    main()
