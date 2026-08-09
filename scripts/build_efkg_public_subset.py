#!/usr/bin/env python3
"""Create a privacy-filtered, structurally representative EFKG subset.

The public subset is not the full experimental EFKG graph. It samples event
nodes deterministically, removes geographic relations, excludes fine-grained
topic relations, re-encodes every identifier, and retains coarse attributes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path


SAFE_RELATIONS = {
    "fire_at_place_group",
    "fire_has_area_bin",
    "fire_has_cause_group",
    "fire_has_loss_bin",
    "fire_has_material_group",
    "fire_has_structure",
    "fire_in_area_type",
    "fire_on_holiday",
}


def read_map(path: Path, id_field: str):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return {row[id_field]: row for row in reader}


def read_triples(path: Path):
    triples = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 3:
                raise ValueError(f"{path}:{line_number}: expected three fields")
            triples.append(tuple(fields))
    return triples


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_tsv(path: Path, rows, header=None):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        if header:
            handle.write("\t".join(header) + "\n")
        for row in rows:
            handle.write("\t".join(map(str, row)) + "\n")


def repair_cold_start(splits):
    train = list(splits["train"])
    held_out = {name: list(splits[name]) for name in ("valid", "test")}
    train_entities = {x for h, _, t in train for x in (h, t)}
    train_relations = {r for _, r, _ in train}
    moved = []

    for split_name in ("valid", "test"):
        kept = []
        for triple in held_out[split_name]:
            h, r, t = triple
            if h not in train_entities or t not in train_entities or r not in train_relations:
                train.append(triple)
                train_entities.update((h, t))
                train_relations.add(r)
                moved.append((split_name, triple))
            else:
                kept.append(triple)
        held_out[split_name] = kept
    return {"train": train, **held_out}, moved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--event-count", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    entities = read_map(source / "entity_map.tsv", "entity_id")
    relations = read_map(source / "relation_map.tsv", "relation_id")
    relation_by_label = {
        row["relation_label"]: relation_id for relation_id, row in relations.items()
    }
    missing = SAFE_RELATIONS - set(relation_by_label)
    if missing:
        raise ValueError(f"Missing required relations: {sorted(missing)}")
    safe_relation_ids = {relation_by_label[label] for label in SAFE_RELATIONS}

    event_ids = sorted(
        entity_id
        for entity_id, row in entities.items()
        if row.get("entity_type") in {"fire", "fire_event"}
    )
    if args.event_count > len(event_ids):
        raise ValueError("Requested more public events than are available")
    selected_events = set(random.Random(args.seed).sample(event_ids, args.event_count))

    raw_splits = {
        split: read_triples(source / f"{split}.txt")
        for split in ("train", "valid", "test")
    }
    filtered = {}
    for split, triples in raw_splits.items():
        filtered[split] = [
            (h, r, t)
            for h, r, t in triples
            if r in safe_relation_ids and (h in selected_events or t in selected_events)
        ]

    used_entities = {
        entity for triples in filtered.values() for h, _, t in triples for entity in (h, t)
    }
    used_relations = {
        relation for triples in filtered.values() for _, relation, _ in triples
    }

    event_order = sorted(selected_events)
    attribute_order = sorted(used_entities - selected_events)
    entity_order = event_order + attribute_order
    relation_order = sorted(
        used_relations, key=lambda rid: relations[rid]["relation_label"]
    )
    entity_new = {old: idx for idx, old in enumerate(entity_order)}
    relation_new = {old: idx for idx, old in enumerate(relation_order)}

    mapped_splits = {
        split: [
            (entity_new[h], relation_new[r], entity_new[t])
            for h, r, t in triples
        ]
        for split, triples in filtered.items()
    }
    mapped_splits, moved = repair_cold_start(mapped_splits)

    entity_rows = []
    for old in entity_order:
        row = entities[old]
        if old in selected_events:
            label = f"event_{entity_new[old] + 1:06d}"
            entity_type = "fire_event"
        else:
            label = row["entity_label"]
            entity_type = row["entity_type"]
        entity_rows.append((entity_new[old], label, entity_type))

    relation_rows = [
        (relation_new[old], relations[old]["relation_label"])
        for old in relation_order
    ]

    for split, triples in mapped_splits.items():
        write_tsv(output / f"{split}.tsv", triples)
    write_tsv(
        output / "entity_map.tsv",
        entity_rows,
        ("entity_id", "entity_label", "entity_type"),
    )
    write_tsv(
        output / "relation_map.tsv",
        relation_rows,
        ("relation_id", "relation_label"),
    )

    all_triples = sum(mapped_splits.values(), [])
    all_text = "\n".join(
        [str(row) for row in entity_rows]
        + [str(row) for row in relation_rows]
        + [str(row) for row in all_triples]
    )
    privacy_checks = {
        "no_geographic_relations": not any(
            token in all_text.lower()
            for token in ("city_in_province", "fire_in_city", "province")
        ),
        "no_source_field_names": not any(
            token in all_text.lower()
            for token in ("fire_process", "xfdwdm", "xzqydm", "address")
        ),
        "no_email_addresses": re.search(
            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", all_text
        ) is None,
        "no_phone_like_numbers": re.search(r"(?<!\d)1[3-9]\d{9}(?!\d)", all_text)
        is None,
        "event_labels_reencoded": all(
            label.startswith("event_") for _, label, typ in entity_rows if typ == "fire_event"
        ),
    }

    train_entities = {
        x for h, _, t in mapped_splits["train"] for x in (h, t)
    }
    train_relations = {r for _, r, _ in mapped_splits["train"]}
    eval_entities = {
        x
        for split in ("valid", "test")
        for h, _, t in mapped_splits[split]
        for x in (h, t)
    }
    eval_relations = {
        r
        for split in ("valid", "test")
        for _, r, _ in mapped_splits[split]
    }
    validation = {
        "status": "PASS"
        if all(privacy_checks.values())
        and eval_entities <= train_entities
        and eval_relations <= train_relations
        else "FAIL",
        "privacy_checks": privacy_checks,
        "duplicates": {
            split: len(triples) - len(set(triples))
            for split, triples in mapped_splits.items()
        },
        "split_overlap": {
            "train_valid": len(
                set(mapped_splits["train"]) & set(mapped_splits["valid"])
            ),
            "train_test": len(
                set(mapped_splits["train"]) & set(mapped_splits["test"])
            ),
            "valid_test": len(
                set(mapped_splits["valid"]) & set(mapped_splits["test"])
            ),
        },
        "unseen_eval_entities": len(eval_entities - train_entities),
        "unseen_eval_relations": len(eval_relations - train_relations),
        "cold_start_repair_moves": len(moved),
    }
    (output / "validation_report.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if validation["status"] != "PASS":
        raise RuntimeError(json.dumps(validation, ensure_ascii=False, indent=2))

    stats = {
        "dataset": "EFKG-Public-Subset-v1.0",
        "relationship_to_paper_dataset": (
            "Privacy-filtered subset; not structurally identical to the restricted "
            "EFKG-v1.0 used for the paper's reported metrics."
        ),
        "sampling": {
            "source_events": len(event_ids),
            "released_events": len(selected_events),
            "seed": args.seed,
        },
        "graph": {
            "entities": len(entity_rows),
            "relations": len(relation_rows),
            "train": len(mapped_splits["train"]),
            "valid": len(mapped_splits["valid"]),
            "test": len(mapped_splits["test"]),
            "total": len(all_triples),
        },
        "relation_distribution": dict(
            sorted(Counter(r for _, r, _ in all_triples).items())
        ),
        "removed_information": [
            "all raw incident rows and source linkage",
            "addresses, roads, organization and administrative codes",
            "free-text incident descriptions",
            "province/city entities and geographic relations",
            "fine-grained place, material, and cause topics",
            "continuous measurements and timestamps",
        ],
    }
    (output / "dataset_card.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    files = [
        "train.tsv",
        "valid.tsv",
        "test.tsv",
        "entity_map.tsv",
        "relation_map.tsv",
        "dataset_card.json",
        "validation_report.json",
    ]
    write_tsv(
        output / "checksums.sha256",
        [(sha256(output / name), name) for name in files],
    )
    print(json.dumps(stats["graph"], indent=2))
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
