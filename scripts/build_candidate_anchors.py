#!/usr/bin/env python3
"""Build confidence-aware candidate anchors from a trained prior scorer.

The script does not ship intermediate candidate JSON files. It reconstructs
them from raw integer-ID triples and the serialized PyKEEN prior model.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch


def read_triples(path: Path) -> list[tuple[int, int, int]]:
    triples = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 3:
                raise ValueError(f"{path}:{line_number}: expected three fields")
            triples.append(tuple(map(int, fields)))
    return triples


def read_labels(path: Path, id_column: str, label_column: str) -> dict[int, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or id_column not in reader.fieldnames:
            raise ValueError(f"Missing {id_column!r} in {path}")
        labels = {}
        for row in reader:
            labels[int(row[id_column])] = row[label_column]
    return labels


def load_model(path: Path, device: torch.device):
    try:
        model = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        model = torch.load(path, map_location=device)
    model = model.to(device)
    model.eval()
    return model


def build_filters(triples):
    tails = defaultdict(set)
    heads = defaultdict(set)
    for h, r, t in triples:
        tails[(h, r)].add(t)
        heads[(r, t)].add(h)
    return tails, heads


def filtered_topk(scores, blocked, gold, top_k):
    scores = scores.clone()
    for entity in blocked:
        if entity != gold:
            scores[entity] = float("-inf")
    gold_score = scores[gold]
    rank = int((scores > gold_score).sum().item()) + 1
    candidates = torch.topk(scores, k=min(top_k, scores.numel())).indices.tolist()
    return rank, candidates


@torch.no_grad()
def build_examples(
    model,
    triples,
    entity_labels,
    relation_labels,
    known_tails,
    known_heads,
    top_k,
    device,
    training_split,
):
    examples = []
    for h, r, t in triples:
        tail_scores = model.score_t(torch.tensor([[h, r]], device=device)).squeeze(0)
        tail_rank, tail_candidates = filtered_topk(
            tail_scores, known_tails[(h, r)], t, top_k
        )
        if training_split and t not in tail_candidates:
            tail_candidates[-1] = t
        examples.append(
            {
                "triple": [entity_labels[h], relation_labels[r], entity_labels[t]],
                "triple_id": [h, r, t],
                "type": "predicted_tail",
                "query_entity": entity_labels[h],
                "query_entity_id": h,
                "rank_entities": [entity_labels[x] for x in tail_candidates],
                "rank_entities_id": tail_candidates,
                "rank": tail_rank,
            }
        )

        head_scores = model.score_h(torch.tensor([[r, t]], device=device)).squeeze(0)
        head_rank, head_candidates = filtered_topk(
            head_scores, known_heads[(r, t)], h, top_k
        )
        if training_split and h not in head_candidates:
            head_candidates[-1] = h
        examples.append(
            {
                "triple": [entity_labels[h], relation_labels[r], entity_labels[t]],
                "triple_id": [h, r, t],
                "type": "predicted_head",
                "query_entity": entity_labels[t],
                "query_entity_id": t,
                "rank_entities": [entity_labels[x] for x in head_candidates],
                "rank_entities_id": head_candidates,
                "rank": head_rank,
            }
        )
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--prior-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    split_triples = {
        split: read_triples(dataset_dir / f"{split}.tsv")
        for split in ("train", "valid", "test")
    }
    all_triples = sum(split_triples.values(), [])
    known_tails, known_heads = build_filters(all_triples)
    entity_labels = read_labels(
        dataset_dir / "entity_map.tsv", "entity_id", "entity_label"
    )
    relation_labels = read_labels(
        dataset_dir / "relation_map.tsv", "relation_id", "relation_label"
    )
    device = torch.device(args.device)
    model = load_model(args.prior_model.resolve(), device)

    for split, triples in split_triples.items():
        examples = build_examples(
            model=model,
            triples=triples,
            entity_labels=entity_labels,
            relation_labels=relation_labels,
            known_tails=known_tails,
            known_heads=known_heads,
            top_k=args.top_k,
            device=device,
            training_split=(split == "train"),
        )
        output_path = output_dir / f"{split}_candidates.json"
        output_path.write_text(
            json.dumps(examples, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"{split}: {len(examples)} examples -> {output_path}")


if __name__ == "__main__":
    main()
