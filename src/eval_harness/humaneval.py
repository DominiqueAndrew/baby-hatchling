from __future__ import annotations

import argparse

from datasets import load_dataset

from .common import greedy_generate, load_model_and_tokenizer
from ..utils.sandbox import run_tests


def main() -> None:
    parser = argparse.ArgumentParser(description="HumanEval/EvalPlus mini evaluation")
    parser.add_argument("--config", default="configs/hatchling_xs.yaml")
    parser.add_argument("--load", required=True)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    _, model, tokenizer = load_model_and_tokenizer(args.config, args.load)
    dataset = load_dataset("openai/openai_humaneval", split="test")
    total = min(args.limit, len(dataset))
    passed = 0
    for row in dataset.select(range(total)):
        prompt = row["prompt"]
        completion = greedy_generate(model, tokenizer, prompt, max_new_tokens=128)
        if run_tests(prompt + completion, row.get("test") or row.get("tests", "")):
            passed += 1
    pass_rate = passed / total if total else 0.0
    print(f"HumanEval pass@1: {pass_rate:.3f} ({passed}/{total})")


if __name__ == "__main__":
    main()
