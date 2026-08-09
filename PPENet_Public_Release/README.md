# PPENet

Official implementation package for **PPENet: A Prior-Grounded Progressive
Evidence Network for Interpretable Knowledge Graph Completion**.

PPENet combines a frozen graph-wide relational prior, confidence-aware anchor
selection, NCRL-grounded progressive evidence retrieval, a disentangled
evidence encoder, prior-evidence alignment, and an LLM reasoning module.

## Release scope

This archive contains the model source code and all scripts required to
reconstruct prior embeddings, candidate-anchor JSON, NCRL rules, evidence
subgraphs, training data, and evaluation outputs. Generated JSON files,
embedding tensors, model checkpoints, and reported metric files are
intentionally not distributed.

The archive includes `data/EFKG-Public-Subset`, a privacy-filtered subset for
pipeline verification. The complete EFKG-v1.0 graph and its raw incident source
remain restricted; see `DATA_AVAILABILITY.md`.

## Repository layout

```text
PPENet/
|-- ppenet/                         # PPENet model and data modules
|-- scripts/                        # prior, anchor, NCRL, and evidence builders
|-- resources/                      # relation question templates
|-- data/EFKG-Public-Subset/        # public privacy-filtered subset
|-- third_party/NCRL/               # NCRL integration and attribution
|-- train.py                        # LoRA/QLoRA training
|-- evaluate.py                     # filtered ranking evaluation
|-- requirements.txt
|-- LICENSE
`-- THIRD_PARTY_NOTICES.md
```

## Installation

Python 3.11 is recommended. Install the CUDA-enabled PyTorch build appropriate
for the target server, followed by the remaining dependencies:

```bash
pip install -r requirements.txt
```

No model weights are included. Download Qwen2.5-7B, DeepSeek-R1-7B, or
Mistral-7B from their official model repositories and pass the local directory
through `--model_name_or_path`.

Before downloading a backbone, verify the installed dependencies and the
PPENet forward/backward tensor path with the lightweight runtime test:

```bash
python scripts/smoke_test.py
```

This test uses a small synthetic prior and a dummy causal-language-model
interface; it does not access benchmark test data or download model weights.

## Input format

Each dataset directory must contain integer-ID triples and mappings:

```text
train.tsv
valid.tsv
test.tsv
entity_map.tsv       # entity_id, entity_label, entity_type
relation_map.tsv     # relation_id, relation_label
```

The included EFKG subset already follows this format. WN18RR and FB15k-237
should be obtained from their standard public benchmark distributions and
converted to contiguous integer IDs without changing the established splits.

## 1. Train the graph-wide relational prior

The following command trains RotatE only on the training graph and exports the
frozen entity table and a scorer used for candidate-anchor construction:

```bash
python scripts/train_prior_encoder.py \
  --train_path data/wn18rr/train.tsv \
  --valid_path data/wn18rr/valid.tsv \
  --model RotatE \
  --num_entities 40943 \
  --num_relations 11 \
  --embedding_dim 200 \
  --num_epochs 200 \
  --batch_size 512 \
  --output_path work/wn18rr/entity_embeddings.pt \
  --model_output_path work/wn18rr/rotate_model.pt \
  --device cuda
```

`train_prior_encoder.py` does not evaluate on or optimize against the held-out
test split.

## 2. Mine NCRL rules

NCRL is a third-party tool and is fetched from its official repository:

```bash
python scripts/fetch_ncrl.py --target third_party/NCRL-src
python scripts/prepare_ncrl_dataset.py \
  --dataset-dir data/wn18rr \
  --output-dir third_party/NCRL-src/datasets/wn18rr \
  --relation-token label
```

Run NCRL from its `code` directory. The following pattern trains with observed
paths up to length 3 and exports the Top-500 rules of body lengths 2 and 3:

```bash
cd third_party/NCRL-src/code
python main.py --train --data wn18rr --max_path_len 3 \
  --anchor 10000 --model wn18rr_ncrl_l3 --gpu 0
python main.py --test --get_rule --data wn18rr \
  --model wn18rr_ncrl_l3 --learned_path_len 2 --topk 500 \
  --output_file ../../../work/wn18rr/wn18rr_ncrl
python main.py --test --get_rule --data wn18rr \
  --model wn18rr_ncrl_l3 --learned_path_len 3 --topk 500 \
  --output_file ../../../work/wn18rr/wn18rr_ncrl
cd ../../..
```

Convert the exported rules to PPENet JSON:

```bash
python scripts/convert_ncrl_rules.py \
  --rule-file work/wn18rr/wn18rr_ncrl_500_2.txt \
  --rule-file work/wn18rr/wn18rr_ncrl_500_3.txt \
  --relation-map data/wn18rr/relation_map.tsv \
  --relation-field relation_label \
  --output-json work/wn18rr/rules.json \
  --stats-json work/wn18rr/rule_statistics.json
```

The reported FB15k-237 configuration exports length-2 NCRL rules only. WN18RR
and EFKG use length-2 and length-3 NCRL rule files. Longer paths may still
occur in the final evidence subgraphs because shortest-path evidence is added
before NCRL-guided expansion.

## 3. Construct candidate anchors

```bash
python scripts/build_candidate_anchors.py \
  --dataset-dir data/wn18rr \
  --prior-model work/wn18rr/rotate_model.pt \
  --output-dir work/wn18rr/candidates \
  --top-k 20 \
  --device cuda
```

## 4. Assemble progressive evidence

```bash
python scripts/build_evidence_subgraphs.py \
  --dataset-dir data/wn18rr \
  --candidate-dir work/wn18rr/candidates \
  --rules-json work/wn18rr/rules.json \
  --tail-questions resources/wn18rr_tail_questions.json \
  --head-questions resources/wn18rr_head_questions.json \
  --output-dir work/wn18rr/ppenet_data \
  --graph-size 50 \
  --graph-scope train_valid
```

This command generates `train.json`, `valid.json`, and `test.json`. These files
are intermediate products and are excluded from the release archive.

## 5. Train PPENet

Training loads only the training and validation JSON files. It does not load
the held-out test split or compute ranking metrics. The console reports
optimization loss and checkpoint progress only.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset_path work/wn18rr/ppenet_data \
  --kge_embedding_path work/wn18rr/entity_embeddings.pt \
  --model_name_or_path /path/to/Qwen2.5-7B \
  --model_type qwen \
  --output_dir results/wn18rr/qwen25_7b \
  --use_quant true --bits 4 --double_quant true --quant_type nf4 \
  --bf16 true --num_train_epochs 13 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-4 \
  --lora_r 32 --lora_alpha 32 --lora_dropout 0.1 \
  --num_components 4 --mi_weight 0.01 \
  --evidence_hidden_dim 128 --evidence_num_hidden_layers 1 \
  --adapter_size 1024 --component_dropout 0.1 \
  --remove_unused_columns false \
  --save_strategy steps --save_steps 500 --save_total_limit 1
```

DeepSeek-R1-7B uses `--model_type qwen` when the distilled Qwen architecture is
selected. Mistral-7B uses `--model_type mistral`. Keep all evidence-side settings
unchanged for controlled backbone comparisons.

## 6. Evaluate PPENet

Ranking metrics are computed only in this separate evaluation process:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --dataset_path work/wn18rr/ppenet_data \
  --kge_embedding_path work/wn18rr/entity_embeddings.pt \
  --checkpoint_dir results/wn18rr/qwen25_7b/checkpoint-final \
  --model_name_or_path /path/to/Qwen2.5-7B \
  --model_type qwen \
  --num_components 4 --mi_weight 0.01 \
  --evidence_hidden_dim 128 --evidence_num_hidden_layers 1 \
  --adapter_size 1024 --component_dropout 0.1 \
  --num_return_sequences 1
```

The command writes `prediction.json` and `metrics.txt` beside the checkpoint.
No precomputed predictions or target performance values are included here.

## EFKG-Public-Subset

To verify the pipeline on the included subset, replace `data/wn18rr` with
`data/EFKG-Public-Subset`, set `--num_entities 1608` and `--num_relations 8`,
and use generic prompts by omitting `--tail-questions` and `--head-questions`.
Results obtained on this subset are not directly comparable with the complete
EFKG-v1.0 results reported in the manuscript.

## Reproducibility

- Random seeds are fixed in prior training, data release construction, and LLM
  fine-tuning.
- The training process is isolated from test ranking metrics.
- Public data files include SHA-256 checksums and validation reports.
- Intermediate JSON, embeddings, checkpoints, and metric files are regenerated
  locally rather than distributed.
- Third-party provenance and licenses are documented in
  `THIRD_PARTY_NOTICES.md`.

## Citation

The PPENet citation will be added after publication. When using the inherited
DrKGC components or NCRL, please also cite the original works listed in
`THIRD_PARTY_NOTICES.md`.
