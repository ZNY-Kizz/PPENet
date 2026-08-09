#!/usr/bin/env python3
"""Lightweight PPENet runtime test that does not download an LLM checkpoint."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppenet.network import PPENet
from ppenet.prior_evidence import PriorEvidenceEncoder


class DummyTokenizer:
    pad_token_id = 0

    def __init__(self) -> None:
        self.tokens = {"[QUERY]": 1, "[ENTITY]": 2}

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.tokens[token]


class DummyCausalLM(nn.Module):
    """Minimal LLM interface used only to validate PPENet tensor plumbing."""

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, inputs_embeds, attention_mask=None, labels=None):
        del attention_mask, labels
        hidden = self.self_attn.q_proj(inputs_embeds)
        return SimpleNamespace(loss=hidden.float().square().mean())


def check_dependencies() -> None:
    required = (
        "torch",
        "transformers",
        "peft",
        "bitsandbytes",
        "datasets",
        "networkx",
        "pykeen",
        "sentencepiece",
        "google.protobuf",
    )
    versions = {}
    for package in required:
        module = importlib.import_module(package)
        versions[package] = getattr(module, "__version__", "available")

    print("Dependency check:")
    for package, version in versions.items():
        print(f"  {package}: {version}")
    print(f"  CUDA available: {torch.cuda.is_available()}")


def build_subgraph(offset: int, num_relations: int) -> list[list[int]]:
    edges = []
    for index in range(12):
        src = offset + (index % 6)
        dst = offset + ((index + 1) % 6)
        edges.append([src, index % num_relations, dst])
    return edges


def test_prior_evidence_encoder() -> PriorEvidenceEncoder:
    torch.manual_seed(42)
    num_entities = 32
    num_relations = 6
    prior_dim = 16
    hidden_size = 32

    encoder = PriorEvidenceEncoder(
        kge_embedding=torch.randn(num_entities, prior_dim),
        input_size=prior_dim,
        num_rels=num_relations,
        evidence_hidden_dim=8,
        evidence_num_hidden_layers=1,
        adapter_size=24,
        output_size=hidden_size,
        hidden_act="silu",
        num_components=4,
        mi_weight=0.01,
        component_dropout=0.1,
    )

    query_ids = torch.tensor([0, 8], dtype=torch.long)
    relation_ids = torch.tensor([1, 2], dtype=torch.long)
    entity_ids = torch.tensor([[1, 2, 3], [9, 10, 11]], dtype=torch.long)
    subgraphs = [build_subgraph(0, num_relations), []]

    query_out, entity_out, auxiliary = encoder(
        query_ids,
        relation_ids,
        entity_ids,
        subgraphs,
        return_aux=True,
    )
    assert query_out.shape == (2, hidden_size)
    assert entity_out.shape == (6, hidden_size)
    assert torch.isfinite(query_out).all()
    assert torch.isfinite(entity_out).all()
    assert torch.isfinite(auxiliary["mi_loss"])
    return encoder


def test_full_wrapper(encoder: PriorEvidenceEncoder) -> None:
    tokenizer = DummyTokenizer()
    llm = DummyCausalLM(vocab_size=64, hidden_size=32)
    model = PPENet(tokenizer, llm, encoder)

    input_ids = torch.tensor(
        [
            [10, 1, 11, 2, 12, 2, 13, 2],
            [20, 1, 21, 2, 22, 2, 23, 2],
        ],
        dtype=torch.long,
    )
    query_ids = torch.tensor([0, 8], dtype=torch.long)
    relation_ids = torch.tensor([1, 2], dtype=torch.long)
    entity_ids = torch.tensor([[1, 2, 3], [9, 10, 11]], dtype=torch.long)
    subgraphs = [build_subgraph(0, 6), []]

    outputs = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=input_ids.clone(),
        query_ids=query_ids,
        relation_ids=relation_ids,
        entity_ids=entity_ids,
        subgraph=subgraphs,
    )
    assert torch.isfinite(outputs.loss)
    outputs.loss.backward()

    trainable_gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert trainable_gradients, "No trainable gradient was produced."
    assert all(torch.isfinite(gradient).all() for gradient in trainable_gradients)


def main() -> None:
    check_dependencies()
    encoder = test_prior_evidence_encoder()
    test_full_wrapper(encoder)
    print("PPENet forward/backward smoke test: PASS")


if __name__ == "__main__":
    main()
