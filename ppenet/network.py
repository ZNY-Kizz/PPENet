from pathlib import Path
from contextlib import nullcontext

import torch
from torch import nn
from transformers import GenerationConfig

__all__ = ["PPENet"]


class PPENet(nn.Module):
    """Prior-grounded progressive evidence network with an LLM reasoner."""

    def __init__(self, tokenizer, llm_model, prior_evidence_encoder):
        super().__init__()
        self.tokenizer = tokenizer
        self.llm_model = llm_model
        self.prior_evidence_encoder = prior_evidence_encoder

        self.query_token_id = self.tokenizer.convert_tokens_to_ids("[QUERY]")
        self.entity_token_id = self.tokenizer.convert_tokens_to_ids("[ENTITY]")

    def _replace_placeholders(
        self,
        input_ids: torch.Tensor,
        query_ids: torch.Tensor,
        relation_ids: torch.Tensor,
        entity_ids: torch.Tensor,
        subgraph=None,
        need_aux=False,
    ):
        if need_aux:
            query_embeds, entity_embeds, aux = self.prior_evidence_encoder(
                query_ids, relation_ids, entity_ids, subgraph, return_aux=True
            )
        else:
            query_embeds, entity_embeds = self.prior_evidence_encoder(
                query_ids, relation_ids, entity_ids, subgraph, return_aux=False
            )
            aux = None

        clean_ids = input_ids.clone()
        clean_ids[clean_ids == self.query_token_id] = self.tokenizer.pad_token_id
        clean_ids[clean_ids == self.entity_token_id] = self.tokenizer.pad_token_id

        input_embed_layer = self.llm_model.get_input_embeddings()
        inputs_embeds = input_embed_layer(clean_ids).clone()

        # Align graph representations with the LLM embedding dtype and device.
        query_embeds = query_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        entity_embeds = entity_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)

        query_pos = torch.nonzero(input_ids == self.query_token_id, as_tuple=False)
        entity_pos = torch.nonzero(input_ids == self.entity_token_id, as_tuple=False)

        if query_pos.size(0) != query_embeds.size(0):
            raise ValueError(
                f"Mismatch between [QUERY] placeholder count ({query_pos.size(0)}) "
                f"and query embeddings ({query_embeds.size(0)})."
            )

        if entity_pos.size(0) != entity_embeds.size(0):
            raise ValueError(
                f"Mismatch between [ENTITY] placeholder count ({entity_pos.size(0)}) "
                f"and entity embeddings ({entity_embeds.size(0)})."
            )

        inputs_embeds[query_pos[:, 0], query_pos[:, 1]] = query_embeds
        inputs_embeds[entity_pos[:, 0], entity_pos[:, 1]] = entity_embeds

        return inputs_embeds, aux

    def _infer_llm_compute_dtype(self):
        for name, param in self.llm_model.named_parameters():
            if "self_attn.q_proj" in name and param.is_floating_point():
                return param.dtype
        for _, param in self.llm_model.named_parameters():
            if param.is_floating_point():
                return param.dtype
        return torch.float32
    
    def forward(self, input_ids, attention_mask, labels, query_ids, relation_ids, entity_ids, subgraph):
        inputs_embeds, aux = self._replace_placeholders(
            input_ids=input_ids,
            query_ids=query_ids,
            relation_ids=relation_ids,
            entity_ids=entity_ids,
            subgraph=subgraph,
            need_aux=True,
        )

        compute_dtype = self._infer_llm_compute_dtype()
        inputs_embeds = inputs_embeds.to(dtype=compute_dtype)

        use_autocast = inputs_embeds.device.type == "cuda" and compute_dtype in (torch.float16, torch.bfloat16)

        autocast_context = (
            torch.autocast(device_type="cuda", dtype=compute_dtype)
            if use_autocast
            else nullcontext()
        )
        with autocast_context:
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )

        mi_loss = aux["mi_loss"] if aux is not None else outputs.loss.new_tensor(0.0)
        total_loss = outputs.loss + self.prior_evidence_encoder.mi_weight * mi_loss
        outputs.loss = total_loss
        return outputs

    def save_pretrained(self, save_dir):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.llm_model.save_pretrained(save_dir)
        torch.save(
            self.prior_evidence_encoder.state_dict(),
            save_dir / "prior_evidence_encoder.bin",
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        query_ids,
        relation_ids,
        entity_ids,
        subgraph=None,
        generation_config: GenerationConfig = None,
    ):
        inputs_embeds, _ = self._replace_placeholders(
            input_ids=input_ids,
            query_ids=query_ids,
            relation_ids=relation_ids,
            entity_ids=entity_ids,
            subgraph=subgraph,
            need_aux=False,
        )

        if generation_config is None:
            generation_config = GenerationConfig()

        attention_mask = torch.ones(inputs_embeds.size()[:2], dtype=torch.long, device=inputs_embeds.device)

        return self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
        )
