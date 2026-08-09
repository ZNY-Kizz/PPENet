import os
import json
import argparse

import torch

import transformers
from transformers import AutoConfig, GenerationConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Seq2SeqTrainer, HfArgumentParser, set_seed, BitsAndBytesConfig

from peft.tuners.lora import LoraLayer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from ppenet import (
    Arguments,
    FinetuningArguments,
    GenerationArguments,
    PPENet,
    PriorEvidenceEncoder,
    make_data_module,
)


def infer_num_relations(dataset_path: str) -> int:
    max_rel = -1
    # Infer schema information without opening the held-out test split.
    for split in ["train.json", "valid.json"]:
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


def get_accelerate_model(args, config, pretrained_model_class):
    if args.use_quant:
        compute_dtype = torch.bfloat16
        model = pretrained_model_class.from_pretrained(
            args.model_name_or_path,
            config=config,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=args.bits == 4,
                load_in_8bit=args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=args.double_quant,
                bnb_4bit_quant_type=args.quant_type,
            ),
            torch_dtype=torch.bfloat16,
        )
    else:
        model = pretrained_model_class.from_pretrained(
            args.model_name_or_path,
            config=config,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
        )

    if args.use_quant:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )

    if args.model_type == "llama":
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
    elif args.model_type == "mistral":
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",
            ],
        )
    elif args.model_type == "qwen":
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
    else:
        raise ValueError(f"Unsupported model_type: {args.model_type}")

    model = get_peft_model(model, peft_config)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            module = module.to(torch.bfloat16)

        if "norm" in name:
            module = module.to(torch.float32)

        # keep embed_tokens in bf16 if possible
        if "embed_tokens" in name:
            if hasattr(module, "weight") and module.weight.dtype == torch.float32:
                module = module.to(torch.bfloat16)

        # IMPORTANT:
        # final hidden_states coming out of float32 norms may stay float32,
        # so keep lm_head in float32 to avoid float != bf16 mismatch
        if "lm_head" in name:
            if hasattr(module, "weight"):
                module = module.to(torch.float32)

    return model


class SavePeftModelCallback(transformers.TrainerCallback):
    KEEP_FILES = {
        "adapter_model.bin",
        "adapter_model.safetensors",
        "adapter_config.json",
        "prior_evidence_encoder.bin",
        "README.md",
    }

    def on_save(self, args, state, control, **kwargs):
        if state.best_model_checkpoint is not None:
            checkpoint_folder = state.best_model_checkpoint
            print(f"Saving the best checkpoint to: {checkpoint_folder}")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            print(f"Saving checkpoint at step {state.global_step} to: {checkpoint_folder}")

        kwargs["model"].save_pretrained(checkpoint_folder)

        for file_name in os.listdir(checkpoint_folder):
            file_path = os.path.join(checkpoint_folder, file_name)
            if file_name not in self.KEEP_FILES and os.path.isfile(file_path):
                os.remove(file_path)

    def on_train_end(self, args, state, control, **kwargs):
        checkpoint_folder = os.path.join(args.output_dir, "checkpoint-final")
        print(f"Saving the final checkpoint to: {checkpoint_folder}")
        kwargs["model"].save_pretrained(checkpoint_folder)


def train():
    set_seed(3407)

    hfparser = HfArgumentParser((Arguments, FinetuningArguments, GenerationArguments))
    (data_args, training_args, generation_args, _) = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    training_args.generation_config = GenerationConfig(**vars(generation_args))
    args = argparse.Namespace(**vars(data_args), **vars(training_args))

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Load LLM: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(data_args.model_name_or_path, use_fast=False)

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    if tokenizer.bos_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.bos_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"bos_token": "<|bos|>"})

    num_new_tokens = tokenizer.add_tokens(["[QUERY]", "[ENTITY]", "[RELATION]"])

    model_config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = get_accelerate_model(args, model_config, AutoModelForCausalLM)
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id

    if num_new_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))

    print("pad_token:", tokenizer.pad_token)
    print("pad_token_id:", tokenizer.pad_token_id)
    print("bos_token:", tokenizer.bos_token)
    print("bos_token_id:", tokenizer.bos_token_id)
    print("eos_token:", tokenizer.eos_token)
    print("eos_token_id:", tokenizer.eos_token_id)

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

    model = PPENet(tokenizer, model, prior_evidence_encoder)

    data_module = make_data_module(args, tokenizer)

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )

    trainer.add_callback(SavePeftModelCallback)

    # No test split or ranking evaluator is available in this process. The
    # training console therefore reports optimization loss only; final KGC
    # metrics are produced separately by evaluate.py.
    trainer.train()
    trainer.save_state()


if __name__ == "__main__":
    train()
