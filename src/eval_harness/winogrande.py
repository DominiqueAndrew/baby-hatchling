from __future__ import annotations

import argparse

from datasets import load_dataset

from .common import load_model_and_tokenizer, option_score


def main() -> None:
    parser = argparse.ArgumentParser(description="WinoGrande evaluation")
    parser.add_argument("--config", default="configs/hn_xs.yaml")
    parser.add_argument("--load", required=True)
    parser.add_argument("--limit", type=int, default=128)
    args = parser.parse_args()

    _, model, tokenizer = load_model_and_tokenizer(args.config, args.load)
    dataset = load_dataset("allenai/winogrande", "winogrande_s", split="validation")
    total = min(args.limit, len(dataset))
    correct = 0
    for row in dataset.select(range(total)):
        sentence = row["sentence"]
        options = [row["option1"], row["option2"]]
        scores = [option_score(model, tokenizer, sentence.replace("_", ""), option) for option in options]
        prediction = int(max(range(len(scores)), key=lambda i: scores[i])) + 1
        if prediction == row["answer"]:
            correct += 1
    accuracy = correct / total if total else 0.0
    print(f"WinoGrande accuracy: {accuracy:.3f} ({correct}/{total})")


if __name__ == "__main__":
    main()
