#!/usr/bin/env python3
"""Assemble PPENet prompts and progressive evidence subgraphs.

Candidate-to-query shortest paths are added first. NCRL relation sequences are
then used for rule-grounded expansion until the configured evidence budget is
reached. The resulting JSON files are consumed by train.py and evaluate.py.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import networkx as nx
from tqdm import tqdm


def read_triples(path: Path):
    triples = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 3:
                raise ValueError(f"{path}:{line_number}: expected three fields")
            triples.append(tuple(map(int, fields)))
    return triples


def read_map(path: Path, id_field: str, label_field: str):
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
    id_to_label = {int(row[id_field]): row[label_field] for row in rows}
    label_to_id = {label: item_id for item_id, label in id_to_label.items()}
    return id_to_label, label_to_id


def load_questions(path: Path | None):
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def add_prompt(example, tail_questions, head_questions):
    relation = example["triple"][1]
    query = example["query_entity"]
    candidates = example["rank_entities"]
    answer_options = "(" + ", ".join(repr(x) for x in candidates) + ")"
    references = [f"'{query}': [QUERY]"]
    references.extend(f"'{name}': [ENTITY]" for name in candidates)

    if example["type"] == "predicted_tail":
        template = tail_questions.get(relation, "What is related to {}?")
        answer = example["triple"][2]
    else:
        template = head_questions.get(relation, "What is related to {}?")
        answer = example["triple"][0]

    example["input"] = (
        "Select exactly one answer entity from "
        + answer_options
        + ".\nUse the supplied graph-aware entity representations: "
        + ", ".join(references)
        + ".\n\nQuestion: "
        + template.format(query)
        + "\nAnswer: "
    )
    example["output"] = answer


def apply_rule_sequence(graph, start, relations, target=None):
    current = start
    path = []
    for relation in relations:
        next_node = None
        for neighbor, edge_dict in graph[current].items():
            if any(data.get("relation") == relation for data in edge_dict.values()):
                next_node = neighbor
                path.append([current, relation, neighbor])
                break
        if next_node is None:
            return []
        current = next_node
    if target is not None and current != target:
        return []
    return path


def assemble_subgraph(example, graph, rules, graph_size):
    query_id = int(example["query_entity_id"])
    candidate_ids = [int(x) for x in example["rank_entities_id"]]
    evidence = []

    # Query-independent graph connectivity evidence.
    for candidate_id in candidate_ids:
        try:
            nodes = nx.shortest_path(graph, source=candidate_id, target=query_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        for src, dst in zip(nodes, nodes[1:]):
            first_edge = next(iter(graph[src][dst].values()))
            evidence.append([int(src), int(first_edge["relation"]), int(dst)])
            if len(evidence) >= graph_size:
                return evidence

    rule_sequences = rules.get(int(example["triple_id"][1]), [])

    # Candidate-verifying rule paths.
    for candidate_id in candidate_ids:
        for sequence in rule_sequences:
            path = apply_rule_sequence(graph, query_id, sequence, candidate_id)
            for triple in path:
                evidence.append(triple)
                if len(evidence) >= graph_size:
                    return evidence

    # Additional rule-grounded expansion from query and anchor entities.
    for start in [query_id] + candidate_ids:
        for sequence in rule_sequences:
            path = apply_rule_sequence(graph, start, sequence)
            for triple in path:
                evidence.append(triple)
                if len(evidence) >= graph_size:
                    return evidence

    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--rules-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tail-questions", type=Path)
    parser.add_argument("--head-questions", type=Path)
    parser.add_argument("--graph-size", type=int, default=50)
    parser.add_argument(
        "--graph-scope",
        choices=("train", "train_valid"),
        default="train_valid",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    candidate_dir = args.candidate_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _, relation_to_id = read_map(
        dataset_dir / "relation_map.tsv", "relation_id", "relation_label"
    )
    raw_rules = json.loads(args.rules_json.read_text(encoding="utf-8"))
    rules = {
        relation_to_id[head]: [
            [relation_to_id[relation] for relation in body]
            for body in bodies
            if all(relation in relation_to_id for relation in body)
        ]
        for head, bodies in raw_rules.items()
        if head in relation_to_id
    }

    graph_triples = read_triples(dataset_dir / "train.tsv")
    if args.graph_scope == "train_valid":
        graph_triples += read_triples(dataset_dir / "valid.tsv")
    graph = nx.MultiGraph()
    for head, relation, tail in graph_triples:
        graph.add_edge(head, tail, relation=relation)

    tail_questions = load_questions(args.tail_questions)
    head_questions = load_questions(args.head_questions)
    for split in ("train", "valid", "test"):
        input_path = candidate_dir / f"{split}_candidates.json"
        examples = json.loads(input_path.read_text(encoding="utf-8"))
        for example in tqdm(examples, desc=f"evidence:{split}"):
            add_prompt(example, tail_questions, head_questions)
            example["subgraph"] = assemble_subgraph(
                example, graph, rules, args.graph_size
            )
        output_path = output_dir / f"{split}.json"
        output_path.write_text(
            json.dumps(examples, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"{split}: {len(examples)} examples -> {output_path}")


if __name__ == "__main__":
    main()
