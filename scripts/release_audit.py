#!/usr/bin/env python3
"""Audit a PPENet release tree for privacy and packaging mistakes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


BANNED_NAMES = {
    "train.json",
    "valid.json",
    "test.json",
    "prediction.json",
    "metrics.txt",
}
BANNED_SUFFIXES = {".pt", ".bin", ".safetensors", ".ckpt", ".xlsx", ".xls"}
GLOBAL_BANNED_TEXT = {"/home/tju", "123.xls"}
EXECUTABLE_BANNED_TEXT = {
    "drkgc",
    "graph_model.bin",
    "graphenhancer",
    "disentangledgraphencoder",
    "simplerellayer",
}
DATA_BANNED_TEXT = {
    "sample_manifest.tsv",
    "fire_process",
    "xfdwdm",
    "xzqydm",
}


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()

    failures = []
    warnings = []
    files = [path for path in root.rglob("*") if path.is_file()]
    for path in files:
        rel = path.relative_to(root).as_posix()
        lower_name = path.name.lower()
        if lower_name in BANNED_NAMES or path.suffix.lower() in BANNED_SUFFIXES:
            failures.append(f"forbidden generated artifact: {rel}")
        if "candidate" in lower_name and path.suffix.lower() == ".json":
            failures.append(f"forbidden candidate artifact: {rel}")
        if path.suffix.lower() in {".py", ".md", ".txt", ".tsv", ".json"}:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
            if rel != "scripts/release_audit.py":
                for token in GLOBAL_BANNED_TEXT:
                    if token.lower() in text.lower():
                        failures.append(f"sensitive/source token {token!r}: {rel}")
                if path.suffix.lower() == ".py":
                    for token in EXECUTABLE_BANNED_TEXT:
                        if token.lower() in text.lower():
                            failures.append(
                                f"legacy executable identifier {token!r}: {rel}"
                            )
            if rel.startswith("data/EFKG-Public-Subset/"):
                for token in DATA_BANNED_TEXT:
                    if token.lower() in text.lower():
                        failures.append(f"source-linkage token {token!r}: {rel}")
            if re.search(r"(?<!\d)1[3-9]\d{9}(?!\d)", text):
                failures.append(f"phone-like number: {rel}")
            if rel.startswith("data/EFKG-Public-Subset/") and (
                "/e/" in text or "/r/" in text
            ):
                failures.append(f"internal EFKG identifier namespace: {rel}")

    required = [
        "README.md",
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
        "DATA_AVAILABILITY.md",
        "requirements.txt",
        "train.py",
        "evaluate.py",
        "scripts/smoke_test.py",
        "data/EFKG-Public-Subset/validation_report.json",
    ]
    for name in required:
        if not (root / name).exists():
            failures.append(f"missing required file: {name}")

    subset = root / "data" / "EFKG-Public-Subset"
    checksum_file = subset / "checksums.sha256"
    if checksum_file.exists():
        for line in checksum_file.read_text(encoding="utf-8").splitlines():
            expected, name = line.split("\t", 1)
            observed = sha256(subset / name)
            if expected != observed:
                failures.append(f"checksum mismatch: data/EFKG-Public-Subset/{name}")

    readme = (root / "README.md").read_text(encoding="utf-8")
    if "<YOUR_GITHUB_ACCOUNT>" in readme:
        warnings.append("GitHub account placeholder remains in release checklist")

    report = {
        "status": "PASS" if not failures else "FAIL",
        "files_scanned": len(files),
        "failures": failures,
        "warnings": warnings,
    }
    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
