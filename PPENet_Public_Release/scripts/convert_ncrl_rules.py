#!/usr/bin/env python3
"""Convert NCRL text rules into PPENet's relation-sequence JSON format."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert and count NCRL rules for PPENet's evidence retriever."
    )
    parser.add_argument(
        "--rule-file",
        type=Path,
        action="append",
        required=True,
        help="NCRL rule text file; repeat this option for lengths 2 and 3.",
    )
    parser.add_argument(
        "--relation-map",
        type=Path,
        required=True,
        help="relation_map.tsv used to validate base relation IDs or labels.",
    )
    parser.add_argument(
        "--relation-field",
        choices=("relation_id", "relation_label"),
        default="relation_label",
        help="Relation-map column used by the NCRL input (default: labels).",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--stats-json",
        type=Path,
        required=True,
        help="Output counts and filtering details for the experiment report.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Discard rules below this NCRL confidence (default: keep all).",
    )
    parser.add_argument(
        "--top-per-head",
        type=int,
        default=0,
        help="Keep at most this many rules per normalized head; 0 keeps all.",
    )
    return parser.parse_args()


def read_relation_ids(path: Path, relation_field: str) -> set[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or relation_field not in reader.fieldnames:
            raise ValueError(f"{path} does not contain {relation_field!r}")
        return {row[relation_field].strip() for row in reader}


def normalize_relation(token: str) -> tuple[str, bool]:
    token = token.strip()
    if token.startswith("inv_"):
        return token[len("inv_") :], True
    return token, False


def parse_rule_line(line: str, path: Path, line_number: int) -> tuple[float, str, tuple[str, ...]]:
    try:
        score_text, rule_text = line.rstrip("\r\n").split("\t", maxsplit=1)
        confidence = float(score_text.split(maxsplit=1)[0])
        # Official NCRL writes ``head <-- body``. Splitting on the operator
        # itself also accepts harmless whitespace differences between versions.
        head_text, body_text = rule_text.split("<--", maxsplit=1)
        body = tuple(item.strip() for item in body_text.split(","))
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Cannot parse {path}:{line_number}: {line!r}") from exc
    if not head_text.strip() or not body or any(not item for item in body):
        raise ValueError(f"Incomplete rule at {path}:{line_number}: {line!r}")
    return confidence, head_text.strip(), body


def main() -> None:
    args = parse_args()
    relation_ids = read_relation_ids(
        args.relation_map.resolve(), args.relation_field
    )

    # Each normalized (head, body) retains the greatest confidence among
    # original/inverse duplicates.
    best_rules: dict[tuple[str, tuple[str, ...]], float] = {}
    raw_by_length: Counter[int] = Counter()
    below_threshold_by_length: Counter[int] = Counter()
    unknown_by_length: Counter[int] = Counter()
    inverse_tokens_normalized = 0

    for rule_file in args.rule_file:
        path = rule_file.resolve()
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                confidence, raw_head, raw_body = parse_rule_line(
                    line, path, line_number
                )
                rule_length = len(raw_body)
                raw_by_length[rule_length] += 1
                if confidence < args.min_confidence:
                    below_threshold_by_length[rule_length] += 1
                    continue

                head, head_was_inverse = normalize_relation(raw_head)
                normalized_body: list[str] = []
                inverse_tokens_normalized += int(head_was_inverse)
                for item in raw_body:
                    normalized_item, was_inverse = normalize_relation(item)
                    normalized_body.append(normalized_item)
                    inverse_tokens_normalized += int(was_inverse)

                if head not in relation_ids or any(
                    item not in relation_ids for item in normalized_body
                ):
                    unknown_by_length[rule_length] += 1
                    continue

                key = (head, tuple(normalized_body))
                if confidence > best_rules.get(key, float("-inf")):
                    best_rules[key] = confidence

    grouped: dict[str, list[tuple[float, tuple[str, ...]]]] = defaultdict(list)
    for (head, body), confidence in best_rules.items():
        grouped[head].append((confidence, body))

    rules_json: dict[str, list[list[str]]] = {}
    kept_by_length: Counter[int] = Counter()
    kept_by_head: dict[str, int] = {}
    for head in sorted(grouped):
        ranked = sorted(grouped[head], key=lambda item: (-item[0], item[1]))
        if args.top_per_head > 0:
            ranked = ranked[: args.top_per_head]
        rules_json[head] = [list(body) for _, body in ranked]
        kept_by_head[head] = len(ranked)
        kept_by_length.update(len(body) for _, body in ranked)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.stats_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(rules_json, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    stats = {
        "input_rule_files": [str(path.resolve()) for path in args.rule_file],
        "min_confidence": args.min_confidence,
        "top_per_head_after_normalization": args.top_per_head,
        "raw_rule_lines_by_body_length": {
            str(k): raw_by_length[k] for k in sorted(raw_by_length)
        },
        "below_confidence_threshold_by_body_length": {
            str(k): below_threshold_by_length[k]
            for k in sorted(below_threshold_by_length)
        },
        "unknown_relation_rules_by_body_length": {
            str(k): unknown_by_length[k] for k in sorted(unknown_by_length)
        },
        "valid_unique_rules_by_body_length": {
            str(k): kept_by_length[k] for k in sorted(kept_by_length)
        },
        "valid_unique_rules_total": sum(kept_by_length.values()),
        "rules_by_normalized_head_relation": kept_by_head,
        "inverse_relation_tokens_normalized": inverse_tokens_normalized,
        "normalization": (
            "The inv_ prefix is removed because PPENet's evidence builder uses "
            "an undirected MultiGraph and its relation dictionary contains only "
            "base relation tokens. Duplicate normalized rules retain "
            "the greatest NCRL confidence."
        ),
    }
    args.stats_json.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(stats["valid_unique_rules_by_body_length"], indent=2))
    print(f"valid_unique_rules_total: {stats['valid_unique_rules_total']}")
    print(f"status: PASS")
    print(f"output_json: {args.output_json.resolve()}")
    print(f"stats_json: {args.stats_json.resolve()}")


if __name__ == "__main__":
    main()
