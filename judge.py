#!/usr/bin/env python3
"""Judge verbalizer predictions with an independent vLLM model."""

import argparse
import json
from pathlib import Path

from tqdm import tqdm


SYSTEM_PROMPT = (
    "You are an expert semantic judge for concept extraction. Compare Concept B, the model "
    "prediction, with Concept A, the reference. Answer YES if they are synonymous, strongly "
    "overlapping, one is a subset or superset of the other, or they have a strong direct "
    "logical/contextual association. Repetition and harmless wording differences must still "
    "be YES. Answer NO only when the concepts are meaningfully unrelated. Output exactly YES or NO."
)
SPLITS = (
    "global_train_standard",
    "low_index_gold",
    "global_gold",
)


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=root / "outputs/predictions/predictions.json")
    parser.add_argument("--output", type=Path, default=root / "outputs/judged.json")
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.tensor_parallel_size <= 0:
        raise ValueError("--tensor-parallel-size must be positive")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("--gpu-memory-utilization must be in (0, 1]")
    if not args.predictions.is_file():
        raise FileNotFoundError(f"Predictions file not found: {args.predictions}")
    from vllm import LLM, SamplingParams

    source = json.loads(args.predictions.read_text(encoding="utf-8"))
    missing_splits = [name for name in SPLITS if name not in source]
    if missing_splits:
        raise KeyError(f"Predictions file is missing splits: {missing_splits}")
    engine = LLM(
        model=args.judge_model,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )
    tokenizer = engine.get_tokenizer()
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    output = {"format_version": 1, "judge_model": args.judge_model, "source": str(args.predictions)}

    for split_name in SPLITS:
        records = source[split_name]["results"]
        judged = []
        for start in tqdm(range(0, len(records), args.batch_size), desc=split_name):
            batch = records[start : start + args.batch_size]
            prompts = []
            for item in batch:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": (
                        f"Concept A: {item['target']!r}\nConcept B: {item['prediction']!r}\n"
                        "Is Concept B a semantic hit for Concept A?"
                    )},
                ]
                prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            responses = engine.generate(prompts, sampling, use_tqdm=False)
            for item, response in zip(batch, responses):
                reply = response.outputs[0].text.strip() if response.outputs else ""
                judged.append({**item, "hit": reply.upper().startswith("YES"), "judge_reply": reply})
        hits = sum(item["hit"] for item in judged)
        output[split_name] = {
            "num_samples": len(judged),
            "hit_count": hits,
            "reference_agreement": hits / max(1, len(judged)),
            "results": judged,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    for name in SPLITS:
        result = output[name]
        print(
            f"{name}: {result['hit_count']}/{result['num_samples']} "
            f"= {result['reference_agreement']:.2%} RA"
        )
    print(f"Saved judged results to {args.output}")


if __name__ == "__main__":
    main()
