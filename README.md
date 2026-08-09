# 🧠 PPENet

<p align="center">

**PPENet: A Prior-Grounded Progressive Evidence Network for Interpretable Knowledge Graph Completion**

</p>

<p align="center">

<img src="https://img.shields.io/badge/Python-3.11-blue.svg">
<img src="https://img.shields.io/badge/PyTorch-CUDA-orange.svg">
<img src="https://img.shields.io/badge/Task-Knowledge%20Graph%20Completion-green.svg">
<img src="https://img.shields.io/badge/License-MIT-lightgrey.svg">

</p>


## 📖 Overview

This repository provides the official implementation of:

**PPENet: A Prior-Grounded Progressive Evidence Network for Interpretable Knowledge Graph Completion.**

PPENet is designed for **interpretable knowledge graph completion (KGC)** by integrating:

- 🧩 **Frozen graph-wide relational prior**
- 🎯 **Confidence-aware anchor selection**
- 🔍 **NCRL-grounded progressive evidence retrieval**
- 🧬 **Disentangled evidence encoder**
- 🔗 **Prior-evidence alignment module**
- 🤖 **LLM reasoning module**

The framework jointly models:

> Stable structural priors + Query-specific progressive evidence + Language-model reasoning

to achieve transparent and interpretable knowledge graph completion.


---

# 📦 Installation


## 🔧 Environment

Recommended environment:

```text
Python 3.11
```


Before running PPENet, download one of the following language models from their official repositories:

- 🤗 Qwen2.5-7B
- 🤗 DeepSeek-R1-7B
- 🤗 Mistral-7B


Specify the local model directory through:

```bash
--model_name_or_path
```


---

# ✅ Runtime Check

Before training, we recommend running the lightweight runtime test:

```bash
python scripts/smoke_test.py
```


This test verifies:

- ✅ PPENet forward propagation
- ✅ Backward propagation
- ✅ Model interface compatibility


The test uses:

- A small synthetic prior representation
- A dummy causal language model interface


It does **not**:

- ❌ Download model weights
- ❌ Access benchmark test data


---

# 📂 Dataset Format


Each dataset directory should contain:

```text
dataset/
│
├── train.tsv
├── valid.tsv
├── test.tsv
│
├── entity_map.tsv
└── relation_map.tsv
```


where:


```text
entity_map.tsv

entity_id
entity_label
entity_type
```


```text
relation_map.tsv

relation_id
relation_label
```


All triples are represented with integer IDs:

```text
head_id    relation_id    tail_id
```


---

## 📚 Supported Datasets


PPENet supports:


| Dataset | Description |
|---|---|
| WN18RR | WordNet knowledge graph benchmark |
| FB15k-237 | Freebase knowledge graph benchmark |
| EFKG-Public-Subset | Privacy-filtered verification subset |


The provided:

```text
data/EFKG-Public-Subset
```

already follows the required format.


For WN18RR and FB15k-237:

- Obtain datasets from their official benchmark releases;
- Keep the original train/validation/test splits;
- Convert entities and relations into contiguous integer IDs.


⚠️ Do **not** re-split the original datasets.


---

# 1️⃣ Train Graph-wide Relational Prior


PPENet uses **RotatE** to learn the graph-wide relational prior.


RotatE:

- Is trained offline using only training triples;
- Does not access the test split;
- Outputs frozen entity embeddings;
- Provides the scoring model for candidate anchor construction.


Example:


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


Note:

`train_prior_encoder.py`

does **not**:

- Load the test set;
- Optimize on test data;
- Use test metrics during training.


---

# 2️⃣ NCRL Rule Mining


PPENet adopts the third-party neural compositional rule learning method:


> Neural Compositional Rule Learning for Knowledge Graph Reasoning


NCRL is used for offline logical rule mining.


## Download NCRL

```bash
python scripts/fetch_ncrl.py \
--target third_party/NCRL-src
```


Prepare NCRL data:

```bash
python scripts/prepare_ncrl_dataset.py \
  --dataset-dir data/wn18rr \
  --output-dir third_party/NCRL-src/datasets/wn18rr \
  --relation-token label
```


Enter NCRL directory:

```bash
cd third_party/NCRL-src/code
```


---

## Generate NCRL Rules


Configuration:

- Maximum path length: 3
- Top-500 rules per relation
- Export rule bodies of length 2 and 3


Train NCRL:

```bash
python main.py --train \
 --data wn18rr \
 --max_path_len 3 \
 --anchor 10000 \
 --model wn18rr_ncrl_l3 \
 --gpu 0
```


Generate length-2 rules:

```bash
python main.py --test \
 --get_rule \
 --data wn18rr \
 --model wn18rr_ncrl_l3 \
 --learned_path_len 2 \
 --topk 500 \
 --output_file ../../../work/wn18rr/wn18rr_ncrl
```


Generate length-3 rules:

```bash
python main.py --test \
 --get_rule \
 --data wn18rr \
 --model wn18rr_ncrl_l3 \
 --learned_path_len 3 \
 --topk 500 \
 --output_file ../../../work/wn18rr/wn18rr_ncrl
```


Return:

```bash
cd ../../..
```


---

## Convert NCRL Rules


```bash
python scripts/convert_ncrl_rules.py \
  --rule-file work/wn18rr/wn18rr_ncrl_500_2.txt \
  --rule-file work/wn18rr/wn18rr_ncrl_500_3.txt \
  --relation-map data/wn18rr/relation_map.tsv \
  --relation-field relation_label \
  --output-json work/wn18rr/rules.json \
  --stats-json work/wn18rr/rule_statistics.json
```


Notes:

- FB15k-237 uses only length-2 NCRL rules;
- WN18RR and EFKG use both length-2 and length-3 NCRL rules.


⚠️ Note:

> NCRL rule length is not equivalent to the hop length of the final evidence subgraph.


Longer paths may appear because shortest paths between candidate entities and query entities are additionally incorporated during evidence construction.


---

# 3️⃣ Train PPENet


During training:

✅ Loads:

- Training data
- Validation data


❌ Does not load:

- Test data


❌ Does not compute:

- MRR
- Hits@K


Example:


```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset_path work/wn18rr/ppenet_data \
  --kge_embedding_path work/wn18rr/entity_embeddings.pt \
  --model_name_or_path /path/to/Qwen2.5-7B \
  --model_type qwen \
  --output_dir results/wn18rr/qwen25_7b \
  --use_quant true \
  --bits 4 \
  --double_quant true \
  --quant_type nf4 \
  --bf16 true \
  --num_train_epochs 13 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-4 \
  --lora_r 32 \
  --lora_alpha 32 \
  --lora_dropout 0.1 \
  --num_components 4 \
  --mi_weight 0.01 \
  --evidence_hidden_dim 128 \
  --evidence_num_hidden_layers 1 \
  --adapter_size 1024 \
  --component_dropout 0.1 \
  --remove_unused_columns false \
  --save_strategy steps \
  --save_steps 500 \
  --save_total_limit 1
```


---

## 🔄 Backbone Configuration


DeepSeek-R1-7B:

```bash
--model_type qwen
```


Mistral-7B:

```bash
--model_type mistral
```


For controlled backbone comparison:

> Keep all evidence-side configurations unchanged.


---

# 4️⃣ Evaluate PPENet


Ranking metrics are computed only in the independent evaluation stage.


```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --dataset_path work/wn18rr/ppenet_data \
  --kge_embedding_path work/wn18rr/entity_embeddings.pt \
  --checkpoint_dir results/wn18rr/qwen25_7b/checkpoint-final \
  --model_name_or_path /path/to/Qwen2.5-7B \
  --model_type qwen \
  --num_components 4 \
  --mi_weight 0.01 \
  --evidence_hidden_dim 128 \
  --evidence_num_hidden_layers 1 \
  --adapter_size 1024 \
  --component_dropout 0.1 \
  --num_return_sequences 1
```


Output:

```text
prediction.json
metrics.txt
```


---

# 🔥 EFKG-Public-Subset


To verify PPENet on the public EFKG subset:


Replace:

```text
data/wn18rr
```


with:


```text
data/EFKG-Public-Subset
```


Set:


```text
--num_entities 1608
--num_relations 8
```


Remove:


```text
--tail-questions
--head-questions
```


and use generic prompts.


⚠️ Note:

Results on the public subset:

> Are not directly comparable with the complete EFKG-v1.0 results reported in the manuscript.


---

# 🔬 Reproducibility


PPENet provides:


✅ Fixed random seeds  
✅ Data preprocessing scripts  
✅ NCRL integration  
✅ Prior generation pipeline  
✅ Candidate anchor construction  
✅ Evidence subgraph generation  
✅ LoRA/QLoRA training  
✅ Independent evaluation pipeline  


Intermediate files:

- JSON files
- Embeddings
- Checkpoints
- Metrics


are regenerated locally rather than distributed.


---

# 📜 Citation


The official PPENet citation will be updated after publication.


When using:

- DrKGC components;
- NCRL;


please also cite the original works listed in:


```text
THIRD_PARTY_NOTICES.md
```


---

# ⭐ Acknowledgement


We thank the following open-source projects and datasets:

- RotatE
- NCRL
- Qwen
- DeepSeek
- Mistral
- WN18RR
- FB15k-237
