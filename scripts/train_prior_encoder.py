import argparse
import os
import random
from typing import Tuple

import numpy as np
import torch
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tsv(path: str) -> np.ndarray:
    triples = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_id, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"{path} line {line_id} is invalid: {line}")
            triples.append(parts)
    if not triples:
        raise ValueError(f"No triples found in {path}")
    return np.asarray(triples, dtype=str)


def build_factories(
    train_triples: np.ndarray,
    valid_triples: np.ndarray,
    num_entities: int,
    num_relations: int,
    model_name: str,
) -> Tuple[TriplesFactory, TriplesFactory]:
    # Preserve integer identifiers so row i in the exported table represents
    # entity i. This invariant is required by the downstream PPENet loader.
    entity_to_id = {str(i): i for i in range(num_entities)}
    relation_to_id = {str(i): i for i in range(num_relations)}

    # PyKEEN's R-GCN handles inverse triples internally.
    create_inverse_for_train = model_name != "RGCN"

    training = TriplesFactory.from_labeled_triples(
        triples=train_triples,
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
        create_inverse_triples=create_inverse_for_train,
    )
    validation = TriplesFactory.from_labeled_triples(
        triples=valid_triples,
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
        create_inverse_triples=False,
    )
    return training, validation


def get_model_kwargs(model_name: str, embedding_dim: int) -> dict:
    if model_name in {"RotatE", "DistMult", "ComplEx"}:
        return {"embedding_dim": embedding_dim}
    if model_name == "RGCN":
        return {"embedding_dim": embedding_dim, "num_layers": 2}
    if model_name == "CompGCN":
        # Keep encoder settings minimal for compatibility across PyKEEN 1.10.x.
        return {"embedding_dim": embedding_dim}
    raise ValueError(f"Unsupported model: {model_name}")


def extract_entity_embeddings(model, num_entities: int) -> torch.Tensor:
    """Export a two-dimensional entity table across supported PyKEEN models."""
    representations = getattr(model, "entity_representations", None)
    if not representations:
        raise RuntimeError("Model does not expose entity_representations.")

    representation = representations[0]
    indices = torch.arange(num_entities, device=model.device)
    embedding = None

    # Representation call signatures vary slightly across PyKEEN models.
    try:
        embedding = representation(indices=indices)
    except Exception:
        pass

    if embedding is None:
        try:
            embedding = representation(indices)
        except Exception:
            pass

    if embedding is None:
        try:
            embedding = representation()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to extract entity embeddings from representation: {exc}"
            ) from exc
        if embedding.shape[0] != num_entities:
            raise RuntimeError(
                f"Representation returned shape {tuple(embedding.shape)}, "
                f"but expected first dimension {num_entities}."
            )

    embedding = embedding.detach().cpu()
    if embedding.dim() > 2:
        embedding = embedding.reshape(num_entities, -1)
    if embedding.dim() != 2:
        raise RuntimeError(
            "Expected a two-dimensional entity embedding matrix, "
            f"but received shape {tuple(embedding.shape)}."
        )
    return embedding


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--valid_path", type=str, required=True)
    parser.add_argument(
        "--model",
        type=str,
        default="RotatE",
        choices=["RotatE", "DistMult", "ComplEx", "RGCN", "CompGCN"],
    )
    parser.add_argument("--num_entities", type=int, required=True)
    parser.add_argument("--num_relations", type=int, required=True)
    parser.add_argument("--embedding_dim", type=int, default=200)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--model_output_path",
        type=str,
        required=True,
        help="Serialized PyKEEN model used by build_candidate_anchors.py.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    train_triples = load_tsv(args.train_path)
    valid_triples = load_tsv(args.valid_path)

    print(f"Loaded train triples: {len(train_triples)}")
    print(f"Loaded valid triples: {len(valid_triples)}")
    print(f"Model: {args.model}")
    print(f"num_entities: {args.num_entities}")
    print(f"num_relations: {args.num_relations}")
    print(f"embedding_dim: {args.embedding_dim}")

    training, validation = build_factories(
        train_triples=train_triples,
        valid_triples=valid_triples,
        num_entities=args.num_entities,
        num_relations=args.num_relations,
        model_name=args.model,
    )

    result = pipeline(
        training=training,
        validation=validation,
        model=args.model,
        model_kwargs=get_model_kwargs(args.model, args.embedding_dim),
        training_kwargs={"num_epochs": args.num_epochs, "batch_size": args.batch_size},
        optimizer="Adam",
        optimizer_kwargs={"lr": args.lr},
        device=args.device,
        random_seed=args.seed,
    )

    embedding = extract_entity_embeddings(result.model, args.num_entities)
    ensure_parent(args.output_path)
    torch.save(embedding, args.output_path)

    ensure_parent(args.model_output_path)
    result.model.to("cpu")
    torch.save(result.model, args.model_output_path)

    print(f"Saved entity embeddings to: {args.output_path}")
    print(f"Embedding shape: {tuple(embedding.shape)}")
    print(f"Saved prior scoring model to: {args.model_output_path}")


if __name__ == "__main__":
    main()
