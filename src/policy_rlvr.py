"""Micro-RLVR fine-tuning with verifiable rewards."""
from __future__ import annotations

import argparse
import copy
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset

from .model import build_model
from .tokenizer import SentencePieceTokenizer
from .utils.config import load_config
from .utils.logging import CSVLogger
from .utils.sandbox import run_tests


@dataclass
class TaskSample:
    kind: str  # "math" or "code"
    prompt: str
    answer: str | None = None
    tests: str | None = None


def load_tasks(specs: Sequence[dict], kind: str) -> List[TaskSample]:
    tasks: List[TaskSample] = []
    for spec in specs:
        ds = load_dataset(spec["hf_id"], split=spec.get("split", "train"))
        limit = spec.get("limit")
        iterator = ds if limit is None else ds.select(range(limit))
        for row in iterator:
            if kind == "math":
                tasks.append(TaskSample(kind="math", prompt=row["question"], answer=row["answer"].strip()))
            else:
                tests = row.get("test") or row.get("tests")
                tasks.append(TaskSample(kind="code", prompt=row["prompt"], tests=tests))
    return tasks


def extract_number(text: str) -> str | None:
    matches = re.findall(r"-?\d+[\d,\.]*", text)
    if matches:
        return matches[-1].replace(",", "")
    return None


def evaluate_math(prediction: str, answer: str) -> float:
    guess = extract_number(prediction)
    return 1.0 if guess == extract_number(answer) else 0.0


def evaluate_code(solution: str, tests: str | None) -> float:
    if not tests:
        return 0.0
    return 1.0 if run_tests(solution, tests) else 0.0


def generate_completion(
    model,
    tokenizer: SentencePieceTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    use_amp: bool = False,
) -> Tuple[str, List[int], int]:
    tokens = tokenizer.encode(prompt)
    prompt_len = len(tokens)
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    for _ in range(max_new_tokens):
        input_ids = torch.tensor(tokens[-model.cfg.max_seq :], dtype=torch.long).unsqueeze(0).to(device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred, _ = model(input_ids, use_memory=False)
        logits = pred.logits[0, -1]
        probs = torch.softmax(logits / temperature, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        tokens.append(next_id)
        if next_id == tokenizer.eos_id:
            break
    if was_training:
        model.train()
    text = tokenizer.decode(tokens[prompt_len:])
    return text, tokens, prompt_len


def sequence_logprob(model, token_ids: List[int], prompt_len: int, use_amp: bool = False) -> torch.Tensor:
    device = next(model.parameters()).device
    input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)
    with torch.cuda.amp.autocast(enabled=use_amp):
        pred, _ = model(input_ids, use_memory=False)
    logits = pred.logits[:, :-1, :]
    log_probs = F.log_softmax(logits, dim=-1)
    targets = input_ids[:, 1:]
    seq_logp = torch.gather(log_probs, 2, targets.unsqueeze(-1)).squeeze(-1)
    return seq_logp[:, prompt_len - 1 :].sum()


def rlvr_loop(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    rl_cfg = cfg["rlvr"]
    loss_cfg = cfg["loss"]

    tokenizer = SentencePieceTokenizer()
    model = build_model(cfg["model"])
    ref_model = build_model(cfg["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Optimize for GPU if available
    use_gpu = device.type == "cuda"
    if use_gpu:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        # Compile models for faster execution
        try:
            model = torch.compile(model, mode="reduce-overhead")
            ref_model = torch.compile(ref_model, mode="reduce-overhead")
            print("Models compiled with torch.compile() for optimization")
        except Exception as e:
            print(f"torch.compile() not available or failed: {e}")
    
    model.to(device)
    ref_model.to(device)
    
    # Use mixed precision for GPU
    use_amp = use_gpu
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print("Using mixed precision training (FP16) for faster GPU training")

    state = torch.load(args.load, map_location="cpu")
    model.load_state_dict(state, strict=False)
    ref_model.load_state_dict(state, strict=False)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["optim"]["lr"] * 0.3)

    math_tasks = load_tasks(cfg.get("datasets", {}).get("math", []), kind="math")
    code_tasks = load_tasks(cfg.get("datasets", {}).get("code", []), kind="code")
    tasks = math_tasks + code_tasks
    if not tasks:
        raise ValueError("No RLVR tasks were loaded; please configure datasets.math/code in the YAML file.")
    random.shuffle(tasks)

    logger = CSVLogger(Path("logs") / "rlvr.csv", ["step", "reward", "math", "code", "kl"])
    steps = rl_cfg.get("steps", 20)
    epsilon = rl_cfg.get("epsilon_clip", 0.2)
    rho_max = rl_cfg.get("rho_max", 5.0)
    samples_per_prompt = rl_cfg.get("samples_per_prompt", 2)
    temperature = rl_cfg.get("temperature", 0.8)

    for step in range(steps):
        batch = random.sample(tasks, k=min(samples_per_prompt, len(tasks)))
        logps: List[torch.Tensor] = []
        ref_logps: List[torch.Tensor] = []
        rewards: List[float] = []
        math_hits = 0.0
        code_hits = 0.0
        math_total = 0.0
        code_total = 0.0
        for task in batch:
            completion, seq, prompt_len = generate_completion(model, tokenizer, task.prompt, 128, temperature, use_amp)
            seq_logp = sequence_logprob(model, seq, prompt_len, use_amp)
            ref_logp = sequence_logprob(ref_model, seq, prompt_len, use_amp).detach()
            if task.kind == "math" and task.answer:
                reward = evaluate_math(completion, task.answer)
                math_hits += reward
                math_total += 1
            else:
                reward = evaluate_code(task.prompt + completion, task.tests)
                code_hits += reward
                code_total += 1
            rewards.append(reward)
            logps.append(seq_logp)
            ref_logps.append(ref_logp)
        if not rewards:
            continue
        logp_tensor = torch.stack(logps).to(device)
        ref_tensor = torch.stack(ref_logps).to(device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        policy_loss, ratio = ppo_objective(logp_tensor, ref_tensor, rewards_tensor, epsilon, rho_max)
        kl = torch.mean(logp_tensor - ref_tensor)
        loss = policy_loss + loss_cfg.get("lambda_kl", 0.02) * kl
        optimizer.zero_grad()
        
        # Mixed precision backward pass
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        avg_reward = sum(rewards) / len(rewards)
        logger.log(
            {
                "step": step,
                "reward": avg_reward,
                "math": math_hits / max(1.0, math_total),
                "code": code_hits / max(1.0, code_total),
                "kl": float(kl.item()),
            }
        )

    if args.save:
        torch.save(model.state_dict(), args.save)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Micro-RLVR trainer")
    parser.add_argument("--config", required=True)
    parser.add_argument("--load", required=True)
    parser.add_argument("--save", default="out/rlvr.pt")
    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()
    rlvr_loop(args)
def ppo_objective(
    logp: torch.Tensor,
    ref_logp: torch.Tensor,
    rewards: torch.Tensor,
    epsilon: float,
    rho_max: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    advantage = rewards - rewards.mean()
    ratio = torch.exp(logp - ref_logp)
    ratio = torch.clamp(ratio, max=rho_max)
    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    policy_loss = -torch.mean(torch.min(ratio * advantage, clipped * advantage))
    return policy_loss, ratio
