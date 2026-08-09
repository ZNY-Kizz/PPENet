#!/usr/bin/env python3
"""Prepare a PPENet training graph for NCRL rule mining.

The generated NCRL dataset deliberately mines rules from train.tsv only.
NCRL's train.txt, valid.txt, and test.txt are left empty so that validation and
test triples cannot enter the rule-mining graph through NCRL's data loader.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the canonical EFKG split into NCRL input files."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Directory containing train.tsv, entity_map.tsv, and relation_map.tsv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Target directory, for example NCRL/datasets/efkg.",
    )
    parser.add_argument(
        "--relation-token",
        choices=("id", "label"),
        default="id",
        help=(
            "Use relation IDs (default, matching the canonical EFKG triples) or "
            "human-readable relation labels in NCRL files."
        ),
    )
    return parser.parse_args()


def read_tsv_map(path: Path, id_field: str, label_field: str) -> tuple[list[str], dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or id_field not in reader.fieldnames:
            raise ValueError(f"{path} does not contain column {id_field!r}")
        if label_field not in reader.fieldnames:
            raise ValueError(f"{path} does not contain column {label_field!r}")
        ids: list[str] = []
        labels: dict[str, str] = {}
        for row in reader:
            item_id = row[id_field].strip()
            item_label = row[label_field].strip()
            if not item_id or not item_label:
                raise ValueError(f"Blank ID or label in {path}: {row}")
            if item_id in labels:
                raise ValueError(f"Duplicate ID in {path}: {item_id}")
            ids.append(item_id)
            labels[item_id] = item_label
    return ids, labels


def read_triples(path: Path) -> list[tuple[str, str, str]]:
    triples: list[tuple[str, str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 3:
                raise ValueError(
                    f"{path}:{line_number} has {len(fields)} fields; expected 3"
                )
            triples.append((fields[0], fields[1], fields[2]))
    return triples


def write_lines(path: Path, lines: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for line in lines:
            handle.write(line)
            handle.write("\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()

    entity_ids, _ = read_tsv_map(
        dataset_dir / "entity_map.tsv", "entity_id", "entity_label"
    )
    relation_ids, relation_labels = read_tsv_map(
        dataset_dir / "relation_map.tsv", "relation_id", "relation_label"
    )
    triples = read_triples(dataset_dir / "train.tsv")

    entity_set = set(entity_ids)
    relation_set = set(relation_ids)
    unknown_entities = sorted(
        {entity for h, _, t in triples for entity in (h, t)} - entity_set
    )
    unknown_relations = sorted({r for _, r, _ in triples} - relation_set)
    if unknown_entities:
        raise ValueError(f"Unknown entity IDs in train.txt: {unknown_entities[:10]}")
    if unknown_relations:
        raise ValueError(f"Unknown relation IDs in train.txt: {unknown_relations[:10]}")
    if len(set(triples)) != len(triples):
        raise ValueError("train.tsv contains duplicate triples")

    relation_counts = Counter(r for _, r, _ in triples)
    missing_train_relations = [r for r in relation_ids if relation_counts[r] == 0]
    if missing_train_relations:
        raise ValueError(
            "NCRL requires every declared relation to occur in the mining graph; "
            f"missing from train.tsv: {missing_train_relations}"
        )

    relation_token = (
        {r: r for r in relation_ids}
        if args.relation_token == "id"
        else relation_labels
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    write_lines(output_dir / "entities.txt", entity_ids)
    write_lines(
        output_dir / "relations.txt",
        [relation_token[r] for r in relation_ids],
    )

    facts: list[str] = []
    facts_with_inverse: list[str] = []
    for head, relation_id, tail in triples:
        relation = relation_token[relation_id]
        fact = f"{head}\t{relation}\t{tail}"
        inverse_fact = f"{tail}\tinv_{relation}\t{head}"
        facts.append(fact)
        facts_with_inverse.extend((fact, inverse_fact))

    write_lines(output_dir / "facts.txt", facts)
    write_lines(output_dir / "facts.txt.inv", facts_with_inverse)

    # NCRL's train() concatenates facts + train + valid. Keeping these files
    # empty guarantees that only the canonical EFKG training split is mined.
    for split_name in ("train.txt", "valid.txt", "test.txt"):
        (output_dir / split_name).write_text("", encoding="utf-8")

    generated_files = (
        "entities.txt",
        "relations.txt",
        "facts.txt",
        "facts.txt.inv",
        "train.txt",
        "valid.txt",
        "test.txt",
    )
    metadata = {
        "source_dataset": str(dataset_dir),
        "source_split": "train.tsv only",
        "relation_token": args.relation_token,
        "inverse_relation_prefix": "inv_",
        "counts": {
            "entities": len(entity_ids),
            "base_relations": len(relation_ids),
            "relations_with_inverse": 2 * len(relation_ids),
            "training_facts": len(triples),
            "facts_with_inverse": 2 * len(triples),
            "ncrl_train_lines": 0,
            "ncrl_valid_lines": 0,
            "ncrl_test_lines": 0,
        },
        "relation_training_fact_counts": {
            relation_token[r]: relation_counts[r] for r in relation_ids
        },
        "sha256": {
            name: sha256(output_dir / name) for name in generated_files
        },
        "leakage_control": (
            "Only the PPENet train.tsv triples are stored in facts.txt(.inv); "
            "validation and test triples are not used for NCRL training."
        ),
    }
    (output_dir / "ncrl_dataset_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(metadata["counts"], ensure_ascii=False, indent=2))
    print(f"status: PASS")
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
