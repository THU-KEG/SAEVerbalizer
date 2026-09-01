# SAEVerbalizer: Generating Explanations for Sparse Autoencoder Features via Representation Verbalization

[[arXiv](https://arxiv.org/abs/2608.13538)]
[[PDF](https://arxiv.org/pdf/2608.13538)]

SAEVerbalizer is a framework that injects Sparse Autoencoder (SAE) decoder
directions into a large language model's representations and fine-tunes the
LLM's downstream layers to generate natural-language explanations of the
injected features. Once trained, the resulting **verbalizer** explains unseen
SAE features directly from their decoder directions, without requiring
per-feature activation-example retrieval at inference time.

The learned verbalization capability generalizes to unseen features and
transfers across separately trained SAE dictionaries. With a lightweight
representation-space adapter, it also extends to SAE features from different
LLMs. See our [arXiv paper](https://arxiv.org/abs/2608.13538) for the method,
experiments, and analysis.

This repository contains the code for running and evaluating the paper's
27B verbalizer. Model weights and datasets are downloaded separately from
Hugging Face.

## Repository layout

```text
SAEVerbalizer/
├── .gitignore
├── README.md
├── inference.py                 # SAE-injected generation
├── judge.py                     # Reference Agreement evaluation
├── run_inference.sh             # inference entry point
├── run_judge.sh                 # evaluation entry point
├── requirements-inference.txt
└── requirements-judge.txt
```

The Quick Start commands create the following ignored runtime paths:

```text
checkpoints/SAEVerbalizer-27B/   # downloaded verbalizer checkpoint
sae/gemma-scope-2-27b-it/        # downloaded SAE weights
data/evaluation/                 # downloaded GTS, LIG, and GG sets
outputs/predictions/             # generated explanations
outputs/judged.json              # RA judgments and aggregate scores
```

## Models and released artifacts

Large model files should not be committed to GitHub. The verbalizer checkpoint
and released datasets are hosted on the Hugging Face Hub:

| Artifact | Source | Usage |
|---|---|---|
| 27B verbalizer checkpoint | [`THU-KEG/SAEVerbalizer-27B`](https://huggingface.co/THU-KEG/SAEVerbalizer-27B) | Pass its local directory with `CHECKPOINT_PATH` |
| SAEVerbalizer datasets | [`THU-KEG/SAEVerbalizer-Data`](https://huggingface.co/datasets/THU-KEG/SAEVerbalizer-Data) | Evaluation sets are under `evaluation/`; later releases may add training data |
| Gemma Scope 2 width-262k SAE | [`google/gemma-scope-2-27b-it`](https://huggingface.co/google/gemma-scope-2-27b-it/tree/main/resid_post/layer_16_width_262k_l0_medium) | Pass the downloaded `params.safetensors` with `SAE_PATH` |
| Reference Agreement judge | [`Qwen/Qwen3-30B-A3B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) | Pass its local directory or Hub ID with `JUDGE_MODEL_PATH` |
| Verbalizer backbone | [`google/gemma-3-27b-it`](https://huggingface.co/google/gemma-3-27b-it) | Included in the full verbalizer checkpoint |

GitHub hosts the code and documentation only. Model checkpoints and SAE
weights belong in Hugging Face model repositories; test and training data
belong in Hugging Face dataset repositories. The local `data/` directory is
ignored by Git and is populated after downloading the released test sets.

## Main evaluation protocol

The paper defines three mutually disjoint test sets. This release uses the
paper's names and abbreviations directly.

The reference feature explanations in these sets are sourced from
[Neuronpedia](https://www.neuronpedia.org/).

| Paper name | Abbreviation | File | Samples | Qualification and sampling |
|---|---|---|---:|---|
| Global Train-Standard | GTS | `data/evaluation/global_train_standard_1000.json` | 1000 | Globally sampled; passes the training qualification standard |
| Low-Index Gold | LIG | `data/evaluation/low_index_gold_200.json` | 200 | Low-index sample; passes the stricter gold qualification standard |
| Global Gold | GG | `data/evaluation/global_gold_1000.json` | 1000 | Globally sampled; passes the stricter gold qualification standard |

GTS and GG share a global index distribution but differ in qualification
strictness. LIG and GG share the stricter gold standard but differ in index
distribution. This is why the paper reports results in the order
`GTS / LIG / GG`.

The 27B verbalizer inference configuration is:

- `gemma-3-27b-it` verbalizer backbone;
- Gemma Scope 2 layer-16 width-262k `l0_medium` SAE;
- normalized SAE decoder direction;
- norm-matched additive injection at transformer layer 16;
- injection strength `alpha=0.2`;
- injection into the final four tokens of the chat prompt;
- greedy generation with at most 15 new tokens.

## Quick start

Run the following commands from the repository root. First create the
inference environment:

```bash
python -m venv .venv-inference
source .venv-inference/bin/activate
pip install --upgrade pip
pip install -r requirements-inference.txt
```

Log in to Hugging Face if required, then download the model, SAE, and
evaluation sets:

```bash
hf auth login
hf download THU-KEG/SAEVerbalizer-27B \
  --local-dir checkpoints/SAEVerbalizer-27B
hf download google/gemma-scope-2-27b-it \
  resid_post/layer_16_width_262k_l0_medium/params.safetensors \
  --local-dir sae/gemma-scope-2-27b-it
hf download THU-KEG/SAEVerbalizer-Data \
  --repo-type dataset \
  --include "evaluation/*.json" \
  --local-dir data
```

Run prediction:

```bash
export CHECKPOINT_PATH=checkpoints/SAEVerbalizer-27B
export SAE_PATH=sae/gemma-scope-2-27b-it/resid_post/layer_16_width_262k_l0_medium/params.safetensors
CUDA_VISIBLE_DEVICES=0 bash run_inference.sh
```

Append `--limit 2` for a quick smoke test. Predictions are written to
`outputs/predictions/predictions.json`.

Create a separate judge environment with vLLM, then run Reference Agreement
evaluation. The judge model may be supplied as a Hugging Face model ID:

```bash
deactivate
python -m venv .venv-judge
source .venv-judge/bin/activate
pip install --upgrade pip
pip install -r requirements-judge.txt
export JUDGE_MODEL_PATH=Qwen/Qwen3-30B-A3B-Instruct-2507
CUDA_VISIBLE_DEVICES=0 bash run_judge.sh
```

Judged examples and per-set Reference Agreement (RA) are written to
`outputs/judged.json`. RA is the proportion of test features for which an
independent LLM judge determines that the generated explanation agrees with
the reference explanation. The paper uses `Qwen3-30B-A3B-Instruct-2507` as
the judge with temperature 0.

## Hardware and scope

The released inference and judging pipelines were tested separately on one
NVIDIA A100 80GB GPU. `device_map="auto"` also supports distributing the 27B
verbalizer across multiple visible GPUs. The released checkpoint targets the
specified layer-16 width-262k Gemma Scope 2 SAE and fixed verbalization prompt;
other SAE layers, widths, and model families require separate validation or
the adapters described in the paper.

## Citation

If you use SAEVerbalizer, please cite:

```bibtex
@article{meng2026saeverbalizer,
  title   = {SAEVerbalizer: Generating Explanations for Sparse Autoencoder Features via Representation Verbalization},
  author  = {Meng, Weihan and Guo, Hongzhu and Jing, Yi and Liu, Dewen and Yao, Zijun and Wang, Xiaozhi and Hou, Lei and Li, Juanzi},
  journal = {arXiv preprint arXiv:2608.13538},
  year    = {2026}
}
```

## Open-source release checklist

### 1. 27B verbalizer inference and evaluation

- [x] Inference and Reference Agreement evaluation code
- [x] GTS, LIG, and GG test sets
- [x] Full evaluation reproduction
- [ ] Complete Hugging Face verbalizer checkpoint upload

### 2. 27B verbalizer training

- [ ] Training code and configuration
- [ ] Feature–explanation training pairs

### 3. 1B-to-27B adapter inference

- [ ] Adapter checkpoint and inference code
- [ ] Source-SAE test sets

### 4. Adapter training

- [ ] Adapter training code and configuration
- [ ] Representation-pair training data

### 5. Data selection

- [ ] Candidate construction and LLM-based filtering code
- [ ] Split-construction code
