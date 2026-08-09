import os
import json
import numpy as np
from tqdm import tqdm
import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, HfArgumentParser, set_seed
from peft import PeftModel

from ppenet import (
    Arguments,
    DataModule,
    GenerationArguments,
    PPENet,
    PriorEvidenceEncoder,
)

if torch.cuda.is_available():
    torch.cuda.empty_cache()


def infer_num_relations(dataset_path: str) -> int:
    max_rel = -1
    for split in ["train.json", "valid.json", "test.json"]:
        file_path = os.path.join(dataset_path, split)
        if not os.path.exists(file_path):
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not data:
            continue
        split_max = max(ex["triple_id"][1] for ex in data if "triple_id" in ex)
        max_rel = max(max_rel, split_max)

    if max_rel < 0:
        raise ValueError(f"Failed to infer num_relations from dataset_path={dataset_path}")
    return max_rel + 1


class Evaluator:
    def __init__(self, args, tokenizer, model, data_module, generation_config):
        self.args = args
        self.generation_config = generation_config
        self.tokenizer = tokenizer
        self.model = model
        self.data_module = data_module
        self.device = next(model.parameters()).device

        self.output_dir = os.path.dirname(args.checkpoint_dir)
        self.log_file_path = os.path.join(self.output_dir, "metrics.txt")

    @torch.no_grad()
    def ranking_metrics(self, dataset):
        self.model.eval()

        preds = []
        ranks = np.array([])
        generated = []

        for ex in tqdm(dataset):
            prompt = ex["input"]
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs.input_ids.to(self.device)
            self.generation_config.eos_token_id = self.tokenizer.eos_token_id

            subgraph = [ex["subgraph"]] if "subgraph" in ex else None

            output = self.model.generate(
                input_ids=input_ids,
                query_ids=torch.LongTensor([ex["query_entity_id"]]).to(input_ids.device),
                relation_ids=torch.LongTensor([ex["triple_id"][1]]).to(input_ids.device),
                entity_ids=torch.LongTensor([ex["rank_entities_id"]]).to(input_ids.device),
                subgraph=subgraph,
                generation_config=self.generation_config,
            )
            generated.append(output.sequences[0].cpu().numpy().tolist())
            ex.pop("input")

        batch_preds = self.tokenizer.batch_decode(generated, skip_special_tokens=True)

        for ex_idx, ex in enumerate(dataset):
            target = ex.pop("output")
            rank = ex["rank"]
            pred = str(batch_preds[ex_idx]).strip()

            topk_names = ex["rank_entities"]
            if target == pred:
                rank = 1
            else:
                if pred not in set(topk_names) or topk_names.index(pred) >= rank:
                    rank += 1

            ex["target"] = target
            ex["pred_rank"] = rank
            ex["pred"] = pred
            preds.append(ex)
            ranks = np.append(ranks, rank)

        metrics = {
            "mrr": np.mean(1.0 / ranks),
            "mr": np.mean(ranks),
            "hits1": np.mean(ranks <= 1),
            "hits3": np.mean(ranks <= 3),
            "hits10": np.mean(ranks <= 10),
        }
        metrics = {k: round(v, 8) for k, v in metrics.items()}

        print("ranking metrics:")
        print(metrics)

        with open(self.log_file_path, "w", encoding="utf-8") as log_file:
            log_file.write(f"ranking metrics: {metrics}\n")

        return preds


if __name__ == "__main__":
    set_seed(3407)

    hfparser = HfArgumentParser((Arguments, GenerationArguments))
    (data_args, generation_args, _) = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    generation_config = GenerationConfig(**vars(generation_args))
    args = argparse.Namespace(**vars(data_args))

    print(f"Load LLM: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    if tokenizer.bos_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.bos_token = tokenizer.eos_token

    tokenizer.add_tokens(["[QUERY]", "[ENTITY]", "[RELATION]"])
    generation_config.bos_token_id = tokenizer.bos_token_id

    if not torch.cuda.is_available():
        raise RuntimeError("PPENet evaluation requires a CUDA-capable GPU.")
    device = torch.device("cuda:0")

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    base_model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base_model, args.checkpoint_dir)
    model = model.half()

    if args.num_relations is None:
        args.num_relations = infer_num_relations(args.dataset_path)
    print("num_relations:", args.num_relations)

    kge_embedding = torch.load(args.kge_embedding_path, map_location="cpu")
    kge_embedding_dim = kge_embedding.shape[1]
    llm_config = model.config

    prior_evidence_encoder = PriorEvidenceEncoder(
        kge_embedding=kge_embedding,
        input_size=kge_embedding_dim,
        num_rels=args.num_relations,
        evidence_hidden_dim=args.evidence_hidden_dim,
        evidence_num_hidden_layers=args.evidence_num_hidden_layers,
        adapter_size=args.adapter_size,
        output_size=llm_config.hidden_size,
        hidden_act=llm_config.hidden_act,
        num_components=args.num_components,
        mi_weight=args.mi_weight,
        component_dropout=args.component_dropout,
    )

    ckpt_dir = Path(args.checkpoint_dir)
    state = torch.load(
        ckpt_dir / "prior_evidence_encoder.bin", map_location="cpu"
    )
    prior_evidence_encoder.load_state_dict(state)

    model = PPENet(tokenizer, model, prior_evidence_encoder)
    model = model.half()
    model.to(device)
    model.eval()

    data_module = DataModule(args, tokenizer, load_test=True)
    evaluator = Evaluator(args, tokenizer, model, data_module, generation_config)

    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else nullcontext()
    )
    with autocast_context:
        preds = evaluator.ranking_metrics(data_module.test_ds)

    output = {
        "args": vars(args),
        "generation_config": vars(generation_config),
        "prediction": preds,
    }
    output_path = os.path.join(os.path.dirname(args.checkpoint_dir), "prediction.json")
    json.dump(output, open(output_path, "w", encoding="utf-8"), ensure_ascii=False, indent=4)
