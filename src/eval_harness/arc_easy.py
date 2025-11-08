from __future__ import annotations

import argparse

from datasets import load_dataset

from .common import load_model_and_tokenizer, option_score


def main() -> None:
    parser = argparse.ArgumentParser(description="ARC-Easy evaluation")
    parser.add_argument("--config", default="configs/hatchling_xs.yaml")
    parser.add_argument("--load", required=True)
    parser.add_argument("--limit", type=int, default=64)
    args = parser.parse_args()

    _, model, tokenizer = load_model_and_tokenizer(args.config, args.load)
    dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation")
    total = min(args.limit, len(dataset))
    correct = 0
    for row in dataset.select(range(total)):
        choices = row["choices"]
        texts = choices["text"]
        labels = choices["label"]
        context = f"Question: {row['question']}\nAnswer:".strip()
        scores = [option_score(model, tokenizer, context, f" {label}) {text}") for label, text in zip(labels, texts)]
        pred = labels[int(max(range(len(scores)), key=lambda i: scores[i]))]
        if pred == row["answerKey"]:
            correct += 1
    accuracy = correct / total if total else 0.0
    print(f"ARC-Easy accuracy: {accuracy:.3f} ({correct}/{total})")


if __name__ == "__main__":
    main()
