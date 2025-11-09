from __future__ import annotations

import argparse

from datasets import load_dataset

from .common import load_model_and_tokenizer, option_score


def main() -> None:
    parser = argparse.ArgumentParser(description="HellaSwag evaluation")
    parser.add_argument("--config", default="configs/hn_xs.yaml")
    parser.add_argument("--load", required=True)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    _, model, tokenizer = load_model_and_tokenizer(args.config, args.load)
    dataset = load_dataset("Rowan/hellaswag", split="validation")
    total = min(args.limit, len(dataset))
    correct = 0
    for row in dataset.select(range(total)):
        context = row["ctx_a"] + " " + row["ctx_b"]
        endings = row["endings"]
        scores = [option_score(model, tokenizer, context, ending) for ending in endings]
        pred = int(max(range(len(scores)), key=lambda i: scores[i]))
        if pred == int(row["label"]):
            correct += 1
    accuracy = correct / total if total else 0.0
    print(f"HellaSwag accuracy: {accuracy:.3f} ({correct}/{total})")


if __name__ == "__main__":
    main()
