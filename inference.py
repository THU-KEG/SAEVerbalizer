#!/usr/bin/env python3
"""Generate concept predictions from a trained SAE verbalizer checkpoint."""

import argparse
import json
from contextlib import AbstractContextManager
from pathlib import Path

import torch
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = (
    "You are an expert in concept interpretation.\n"
    "I will inject an internal intervention into your hidden states.\n"
    "Please complete the final sentence based on the intervention you experience.\n"
)
INJECTION_PROMPT = "The target concept is: "
DEFAULT_SPLITS = (
    ("global_train_standard", "global_train_standard_1000.json"),
    ("low_index_gold", "low_index_gold_200.json"),
    ("global_gold", "global_gold_1000.json"),
)


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="save_pretrained checkpoint directory")
    parser.add_argument("--sae", required=True, help="SAE params.safetensors containing w_dec")
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "evaluation")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs" / "predictions")
    parser.add_argument("--inject-layer", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--injection-tokens", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=15)
    parser.add_argument("--limit", type=int, default=None, help="Optional per-split smoke-test limit")
    return parser.parse_args()


def text_layers(model):
    text_model = model.model
    if hasattr(text_model, "language_model"):
        text_model = text_model.language_model
    return text_model.layers


class ResidualInjector(AbstractContextManager):
    def __init__(self, layer, span, direction, alpha):
        self.layer = layer
        self.span = span
        self.direction = direction
        self.alpha = alpha
        self.handle = None

    def hook(self, _module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        # During cached autoregressive decoding L=1. Inject only into the prompt prefill.
        if hidden.ndim != 3 or hidden.shape[1] <= 1:
            return output
        start, end = self.span
        end = min(end, hidden.shape[1])
        if start < 0 or end <= start or self.direction.numel() != hidden.shape[-1]:
            raise RuntimeError(
                f"Invalid injection: span={self.span}, hidden={tuple(hidden.shape)}, "
                f"direction={tuple(self.direction.shape)}"
            )
        modified = hidden.clone()
        direction = self.direction.to(device=hidden.device, dtype=hidden.dtype)
        target_norm = hidden[0, start:end].norm(dim=-1).mean()
        modified[0, start:end] = hidden[0, start:end] + self.alpha * direction * target_norm
        return (modified,) + output[1:] if isinstance(output, tuple) else modified

    def __enter__(self):
        self.handle = self.layer.register_forward_hook(self.hook)
        return self

    def __exit__(self, *_args):
        self.handle.remove()


def load_split(path, limit=None):
    records = json.loads(path.read_text(encoding="utf-8"))
    cleaned = []
    for item in records:
        feature_id = item.get("id", item.get("idx"))
        concept = item.get("concept", item.get("My_Label"))
        if feature_id is None or concept is None:
            raise ValueError(f"Invalid record in {path}: expected id/concept or idx/My_Label")
        cleaned.append({"id": str(feature_id), "target": str(concept)})
    return cleaned[:limit] if limit is not None else cleaned


def main():
    args = parse_args()
    if args.injection_tokens <= 0:
        raise ValueError("--injection-tokens must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive when provided")
    if not args.data_dir.is_dir():
        raise FileNotFoundError(
            f"Evaluation directory not found: {args.data_dir}. "
            "Download the evaluation files as described in README.md."
        )
    if not Path(args.sae).is_file():
        raise FileNotFoundError(f"SAE weights not found: {args.sae}")
    torch.manual_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, dtype=torch.bfloat16, device_map="auto"
    ).eval()

    weights = load_file(args.sae)
    if "w_dec" not in weights:
        raise KeyError(f"{args.sae} does not contain w_dec")
    decoder = weights["w_dec"].to(dtype=torch.bfloat16)
    config = getattr(model.config, "text_config", model.config)
    hidden_size = int(config.hidden_size)
    if decoder.shape[-1] != hidden_size:
        if decoder.shape[0] == hidden_size:
            decoder = decoder.T
        else:
            raise ValueError(f"Cannot orient w_dec {tuple(decoder.shape)} for hidden size {hidden_size}")

    layers = text_layers(model)
    if not 0 <= args.inject_layer < len(layers):
        raise ValueError(f"--inject-layer must be in [0, {len(layers) - 1}]")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": INJECTION_PROMPT},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
    input_device = model.get_input_embeddings().weight.device
    prompt_ids = prompt_ids.to(input_device)
    prompt_length = prompt_ids.shape[1]
    span = (max(0, prompt_length - args.injection_tokens), prompt_length)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "sae": str(Path(args.sae).resolve()),
        "injection": {
            "mode": "direct_add",
            "layer": args.inject_layer,
            "alpha": args.alpha,
            "prompt_token_span": list(span),
            "injection_tokens": args.injection_tokens,
        },
        "generation": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
    }

    for split_name, filename in DEFAULT_SPLITS:
        path = args.data_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing evaluation split: {path}")
        records = load_split(path, args.limit)
        results = []
        for item in tqdm(records, desc=split_name):
            feature_id = int(item["id"])
            if not 0 <= feature_id < decoder.shape[0]:
                raise IndexError(f"Feature {feature_id} is outside SAE width {decoder.shape[0]}")
            direction = decoder[feature_id]
            direction = direction / (direction.norm() + 1e-8)
            with torch.inference_mode(), ResidualInjector(
                layers[args.inject_layer], span, direction, args.alpha
            ):
                generated = model.generate(
                    input_ids=prompt_ids,
                    attention_mask=torch.ones_like(prompt_ids),
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            prediction = tokenizer.decode(
                generated[0, prompt_length:], skip_special_tokens=True
            ).strip()
            results.append({**item, "prediction": prediction})
        payload[split_name] = {
            "data_path": str(path.resolve()),
            "num_samples": len(results),
            "results": results,
        }

    output_path = args.output_dir / "predictions.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
