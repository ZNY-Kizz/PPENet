from dataclasses import dataclass
from typing import Sequence, Dict

import torch
from torch.nn.utils.rnn import pad_sequence
import transformers
from .dataset import DataModule


@dataclass
class QueryCollator:
    args: None
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id

        if eos_id is None:
            raise ValueError("tokenizer.eos_token_id is None, cannot build training samples")

        if pad_id is None:
            pad_id = eos_id

        if bos_id is None:
            bos_id = eos_id

        sources = [ex["input"] for ex in instances]
        targets = [ex["output"] for ex in instances]

        src_max = max(1, self.source_max_len - 1)
        tgt_max = max(1, self.target_max_len - 1)

        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=src_max,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=tgt_max,
            truncation=True,
            add_special_tokens=False,
        )

        source_input_ids = tokenized_sources_with_prompt["input_ids"]
        target_input_ids = tokenized_targets["input_ids"]

        input_ids = []
        labels = []

        for src_ids, tgt_ids in zip(source_input_ids, target_input_ids):
            src_ids = list(src_ids)
            tgt_ids = list(tgt_ids)

            seq = [bos_id] + src_ids + tgt_ids + [eos_id]
            seq = [eos_id if x is None else x for x in seq]
            input_ids.append(torch.tensor(seq, dtype=torch.long))

            lab = torch.full((len(seq),), -100, dtype=torch.long)
            start = len(src_ids) + 1
            tgt_label_ids = tgt_ids + [eos_id]
            tgt_label_ids = [eos_id if x is None else x for x in tgt_label_ids]
            lab[start:] = torch.tensor(tgt_label_ids, dtype=torch.long)
            labels.append(lab)

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        query_ids = torch.tensor([ex["query_entity_id"] for ex in instances], dtype=torch.long)
        relation_ids = torch.tensor([ex["triple_id"][1] for ex in instances], dtype=torch.long)
        entity_ids = torch.tensor([ex["rank_entities_id"] for ex in instances], dtype=torch.long)
        subgraph = [ex.get("subgraph", []) for ex in instances]

        data_dict = {
            "input_ids": input_ids,
            "attention_mask": (input_ids != pad_id).long(),
            "labels": labels,
            "query_ids": query_ids,
            "relation_ids": relation_ids,
            "entity_ids": entity_ids,
            "subgraph": subgraph,
        }

        return data_dict


def make_data_module(args, tokenizer: transformers.PreTrainedTokenizer):
    # Training deliberately does not load the held-out test split. Ranking
    # metrics are computed only by evaluate.py after training has completed.
    data_module = DataModule(args, tokenizer, load_test=False)
    data_collator = QueryCollator(
        args=args,
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
    )

    return {
        "train_dataset": data_module.train_ds,
        "eval_dataset": data_module.eval_ds,
        "data_collator": data_collator,
    }
