<h1 align="center">SAEVerbalizer</h1>

<p align="center">
  <strong>Generating Explanations for Sparse Autoencoder Features<br>
  via Representation Verbalization</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2608.13538">Paper</a> ·
  <a href="https://arxiv.org/pdf/2608.13538">PDF</a> ·
  <a href="https://huggingface.co/THU-KEG/SAEVerbalizer-27B">Model</a> ·
  <a href="https://huggingface.co/datasets/THU-KEG/SAEVerbalizer-Data">Data</a>
</p>

SAEVerbalizer generates natural-language explanations of Sparse Autoencoder
(SAE) features directly from their decoder directions. It injects a feature
direction into an LLM's hidden states and trains the downstream layers to
verbalize the resulting internal representation. At inference time, unseen
features can be explained without retrieving per-feature activation examples.

The learned verbalization capability also transfers across separately trained
SAEs. A lightweight representation-space adapter further extends it to SAE
features from different language models. See the
[paper](https://arxiv.org/abs/2608.13538) for full details.

This repository currently provides inference and Reference Agreement (RA)
evaluation for the paper's **27B verbalizer**.

## Resources

| Resource | Link |
|---|---|
| 27B verbalizer | [THU-KEG/SAEVerbalizer-27B](https://huggingface.co/THU-KEG/SAEVerbalizer-27B) |
| Evaluation data | [THU-KEG/SAEVerbalizer-Data](https://huggingface.co/datasets/THU-KEG/SAEVerbalizer-Data) |
| Gemma Scope 2 SAE | [Layer 16, width 262k, `l0_medium`](https://huggingface.co/google/gemma-scope-2-27b-it/tree/main/resid_post/layer_16_width_262k_l0_medium) |
| RA judge | [Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) |
| Backbone | [Gemma-3-27B-IT](https://huggingface.co/google/gemma-3-27b-it) |

Model weights and datasets are hosted on Hugging Face rather than committed to
this repository.

## Evaluation sets

The paper evaluates the 27B verbalizer on three mutually disjoint feature
sets. Reference explanations are sourced from
[Neuronpedia](https://www.neuronpedia.org/).

| Set | Abbreviation | Samples | Selection |
|---|:---:|---:|---|
| Global Train-Standard | GTS | 1,000 | Global sample passing the training qualification standard |
| Low-Index Gold | LIG | 200 | Low-index sample passing the stricter gold standard |
| Global Gold | GG | 1,000 | Global sample passing the stricter gold standard |

The released files are:

```text
data/evaluation/
├── global_train_standard_1000.json
├── low_index_gold_200.json
└── global_gold_1000.json
```

## Quick start

### 1. Install inference dependencies

```bash
python -m venv .venv-inference
source .venv-inference/bin/activate
pip install --upgrade pip
pip install -r requirements-inference.txt
```

### 2. Download the artifacts

Log in to Hugging Face first if any gated resource requires authentication.

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

### 3. Run inference

```bash
export CHECKPOINT_PATH=checkpoints/SAEVerbalizer-27B
export SAE_PATH=sae/gemma-scope-2-27b-it/resid_post/layer_16_width_262k_l0_medium/params.safetensors

CUDA_VISIBLE_DEVICES=0 bash run_inference.sh
```

Predictions are saved to `outputs/predictions/predictions.json`. For a quick
smoke test, evaluate two examples from each split:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_inference.sh --limit 2
```

### 4. Run Reference Agreement evaluation

Use a separate environment for vLLM and the judge model:

```bash
deactivate
python -m venv .venv-judge
source .venv-judge/bin/activate
pip install --upgrade pip
pip install -r requirements-judge.txt

export JUDGE_MODEL_PATH=Qwen/Qwen3-30B-A3B-Instruct-2507
CUDA_VISIBLE_DEVICES=0 bash run_judge.sh
```

Per-example judgments and aggregate RA scores are saved to
`outputs/judged.json`. RA measures the proportion of generated explanations
that the independent judge considers semantically consistent with their
reference explanations. Following the paper, judging uses
`Qwen3-30B-A3B-Instruct-2507` with temperature 0.

## Inference configuration

The released setup follows the paper's 27B-L16 configuration:

| Setting | Value |
|---|---|
| Backbone | Gemma-3-27B-IT |
| SAE | Gemma Scope 2, residual stream, layer 16, width 262k, `l0_medium` |
| Direction | Normalized SAE decoder vector |
| Intervention | Norm-matched additive injection |
| Injection layer | 16 |
| Injection strength | `0.2` |
| Injection span | Final four prompt tokens |
| Decoding | Greedy, up to 15 new tokens |

## Hardware and scope

Inference and judging were tested separately on one NVIDIA A100 80GB GPU.
`device_map="auto"` can distribute the 27B verbalizer across multiple visible
GPUs. The released checkpoint targets the SAE and prompt configuration above;
other layers, widths, or model families require separate validation or the
adapters described in the paper.

## Repository structure

```text
SAEVerbalizer/
├── .gitignore
├── inference.py                 # SAE-injected generation
├── judge.py                     # Reference Agreement evaluation
├── run_inference.sh             # inference entry point
├── run_judge.sh                 # evaluation entry point
├── requirements-inference.txt
├── requirements-judge.txt
└── README.md
```

Downloaded artifacts and generated outputs are placed in Git-ignored
`checkpoints/`, `sae/`, `data/`, and `outputs/` directories.

## Release roadmap

- [x] 27B verbalizer inference and RA evaluation
- [x] GTS, LIG, and GG evaluation sets
- [x] Full evaluation reproduction
- [ ] Complete 27B verbalizer checkpoint upload
- [ ] 27B verbalizer training code and data
- [ ] 1B-to-27B adapter checkpoint and inference
- [ ] Adapter training code and data
- [ ] Data selection and split-construction code

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
