from __future__ import annotations

import argparse
import re

from datasets import load_dataset

from .common import greedy_generate, load_model_and_tokenizer


def extract_number(text: str) -> str | None:
    matches = re.findall(r"-?\d+[\d,\.]*", text)
    if matches:
        return matches[-1].replace(",", "")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="GSM8K-mini evaluation")
    parser.add_argument("--config", default="configs/hatchling_xs.yaml")
    parser.add_argument("--load", required=True)
    parser.add_argument("--limit", type=int, default=64)
    args = parser.parse_args()

    _, model, tokenizer = load_model_and_tokenizer(args.config, args.load)
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    total = min(args.limit, len(dataset))
    correct = 0
    for row in dataset.select(range(total)):
        prompt = row["question"] + "\nAnswer:"
        completion = greedy_generate(model, tokenizer, prompt, max_new_tokens=64)
        guess = extract_number(completion)
        target = extract_number(row["answer"])
        if guess == target:
            correct += 1
    accuracy = correct / total if total else 0.0
    print(f"GSM8K exact match: {accuracy:.3f} ({correct}/{total})")


if __name__ == "__main__":
    main()
