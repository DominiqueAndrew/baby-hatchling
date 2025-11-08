from __future__ import annotations

import argparse
import random
from typing import List

from .common import greedy_generate, load_model_and_tokenizer


def build_palindrome(length: int) -> tuple[str, str]:
    half = ''.join(random.choice('0123456789') for _ in range(length // 2))
    if random.random() < 0.5:
        s = half + half[::-1]
        answer = "YES"
    else:
        other = ''.join(random.choice('0123456789') for _ in range(length // 2))
        s = half + other
        answer = "NO"
    prompt = f"Is this string a palindrome? {s}\nAnswer YES or NO:"
    return prompt, answer


def build_mqar(length: int) -> tuple[str, str]:
    numbers = [random.randint(1, 9) for _ in range(length)]
    prompt = " + ".join(map(str, numbers)) + " ="
    return f"Compute the sum: {prompt}", str(sum(numbers))


def build_stack(length: int) -> tuple[str, str]:
    stack: List[int] = []
    cmds = []
    for _ in range(length):
        if stack and random.random() < 0.5:
            val = stack.pop()
            cmds.append(f"pop -> {val}")
        else:
            val = random.randint(1, 9)
            stack.append(val)
            cmds.append(f"push {val}")
    prompt = "Simulate stack operations and output the final stack as a space-separated list.\n" + "\n".join(cmds)
    answer = " ".join(map(str, stack)) or "EMPTY"
    return prompt, answer


def evaluate(tasks, model, tokenizer, max_tokens=16):
    hits = 0
    for prompt, answer in tasks:
        completion = greedy_generate(model, tokenizer, prompt, max_new_tokens=max_tokens).strip().upper()
        if completion.startswith(answer.upper()):
            hits += 1
    return hits / len(tasks) if tasks else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic probe evaluation")
    parser.add_argument("--config", default="configs/hatchling_xs.yaml")
    parser.add_argument("--load", required=True)
    parser.add_argument("--examples", type=int, default=32)
    args = parser.parse_args()

    cfg, model, tokenizer = load_model_and_tokenizer(args.config, args.load)
    pal_lengths = cfg.get("probes", {}).get("palindrome", {}).get("lengths", [32, 64])
    mqar_lengths = cfg.get("probes", {}).get("mqar", {}).get("lengths", [64])

    random.seed(0)
    for length in pal_lengths:
        tasks = [build_palindrome(length) for _ in range(args.examples)]
        acc = evaluate(tasks, model, tokenizer)
        print(f"Palindrome@{length}: {acc:.3f}")
    for length in mqar_lengths:
        tasks = [build_mqar(length // 8) for _ in range(args.examples)]
        acc = evaluate(tasks, model, tokenizer)
        print(f"MQAR@{length}: {acc:.3f}")
    tasks = [build_stack(16) for _ in range(args.examples)]
    acc = evaluate(tasks, model, tokenizer)
    print(f"Stack probe: {acc:.3f}")


if __name__ == "__main__":
    main()
