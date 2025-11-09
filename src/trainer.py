"""Training entry-point for Baby-Hatchling."""
from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .model import BabyHatchlingModel, build_model
from .tokenizer import SentencePieceTokenizer
from .utils.config import load_config
from .utils.data import contamination_report, load_text_splits
from .utils.logging import CSVLogger


@dataclass
class Batch:
    tokens: torch.Tensor  # [B,T]
    targets: torch.Tensor  # [B,T]


class ChunkedDataset(Dataset):
    def __init__(self, texts: Sequence[str], tokenizer: SentencePieceTokenizer, seq_len: int) -> None:
        self.seq_len = seq_len
        token_stream: List[int] = []
        for text in texts:
            token_stream.extend(tokenizer.encode(text))
        window = seq_len + 1
        self.examples: List[List[int]] = []
        for idx in range(0, len(token_stream) - window, window):
            chunk = token_stream[idx : idx + window]
            if len(chunk) == window:
                self.examples.append(chunk)
        if not self.examples:
            raise ValueError("Not enough tokens to create training chunks. Provide more data.")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Batch:
        seq = torch.tensor(self.examples[idx], dtype=torch.long)
        return Batch(tokens=seq[:-1], targets=seq[1:])


def collate_fn(batch: Sequence[Batch]) -> Batch:
    tokens = torch.stack([item.tokens for item in batch], dim=0)
    targets = torch.stack([item.targets for item in batch], dim=0)
    return Batch(tokens=tokens, targets=targets)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_texts(specs: Sequence[dict]) -> List[str]:
    samples = load_text_splits(specs)
    report = contamination_report(samples[:512])
    if report:
        report_path = Path("logs") / "contamination.csv"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf8") as handle:
            handle.write("sample_a,sample_b\n")
            for a, b in report:
                handle.write(f"{a},{b}\n")
    return [sample.text for sample in samples]


def train(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    train_cfg = cfg["train"]
    optim_cfg = cfg["optim"]
    loss_cfg = cfg["loss"]

    if args.stage == "gridworld":
        try:
            from .gridworld import run_gridworld_stage
            run_gridworld_stage(cfg, args)
        except ImportError:
            raise ImportError(
                "gridworld module not found. The gridworld functionality has been removed. "
                "Please use 'pretrain' or 'sft' stages instead."
            )
        return

    set_seed(cfg.get("seed", 0))
    torch.set_num_threads(cfg.get("threads") or os.cpu_count() or 4)

    tokenizer = SentencePieceTokenizer()
    texts = load_texts(cfg.get("datasets", {}).get(args.stage, []))
    dataset = ChunkedDataset(texts, tokenizer, train_cfg["seq_len"])

    batch_size = max(1, train_cfg["batch_tokens"] // train_cfg["seq_len"])
    
    model = build_model(cfg["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Optimize for GPU if available
    use_gpu = device.type == "cuda"
    if use_gpu:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        # Compile model for faster execution (PyTorch 2.0+)
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("Model compiled with torch.compile() for optimization")
        except Exception as e:
            print(f"torch.compile() not available or failed: {e}")
    
    # Use multiple CPU workers for parallel data loading (CPU-GPU parallelism)
    num_workers = min(4, os.cpu_count() or 1) if use_gpu else 0
    print(f"DataLoader using {num_workers} worker(s) for parallel data loading")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=use_gpu,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0
    )
    model.to(device)

    if args.load and Path(args.load).exists():
        state = torch.load(args.load, map_location="cpu")
        model.load_state_dict(state, strict=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optim_cfg["lr"],
        betas=tuple(optim_cfg.get("betas", [0.9, 0.95])),
        weight_decay=optim_cfg.get("weight_decay", 0.0),
        eps=optim_cfg.get("eps", 1e-8),
    )

    grad_clip = train_cfg.get("grad_clip", 1.0)
    grad_accum = train_cfg.get("grad_accum", 1)
    max_steps = train_cfg.get("max_steps", 1000)

    logger = CSVLogger(Path("logs") / f"train_{args.stage}.csv", ["step", "loss", "lm", "pc", "epi", "bonus"])

    model.train()
    optimizer.zero_grad()
    step = 0
    loss_meter = []
    micro_step = 0
    
    # Use mixed precision training for GPU (faster and uses less memory)
    use_amp = use_gpu
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print("Using mixed precision training (FP16) for faster GPU training")
    
    for epoch in range(10):  # small curriculum loop
        for batch in tqdm(dataloader, desc=f"{args.stage}-epoch{epoch}"):
            # Non-blocking transfer for better CPU-GPU parallelism
            tokens = batch.tokens.to(device, non_blocking=use_gpu)
            targets = batch.targets.to(device, non_blocking=use_gpu)
            
            # Mixed precision forward pass
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred_out, _ = model(tokens)
            logits = pred_out.logits[:, :-1, :]
            lm_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets[:, : logits.size(1)].reshape(-1), ignore_index=0
            )
            lambda_pc = loss_cfg["lambda_pc"] if getattr(model.pred_head, "enabled", True) else 0.0
            total_loss = lm_loss + lambda_pc * pred_out.pc_loss
            epi_penalty = torch.tensor(model.episodic_gate_penalty(), device=device)
            total_loss = total_loss + loss_cfg.get("lambda_epi", 0.0) * epi_penalty
            
            # Mixed precision backward pass
            if use_amp:
                scaler.scale(total_loss / grad_accum).backward()
            else:
                (total_loss / grad_accum).backward()
            micro_step += 1

            if micro_step % grad_accum == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                optimizer.zero_grad()
                bonus = model.pred_head.curiosity_bonus(pred_out.error_trace)
                logger.log(
                    {
                        "step": step,
                        "loss": float(total_loss.item()),
                        "lm": float(lm_loss.item()),
                        "pc": float(pred_out.pc_loss.item()),
                        "epi": float(epi_penalty.item()),
                        "bonus": bonus,
                    }
                )
                if args.save:
                    save_path = Path(args.save)
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), save_path)
                step += 1
                if step >= max_steps:
                    return
        if step >= max_steps:
            break


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Baby-Hatchling Trainer")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=["pretrain", "sft", "gridworld"], default="pretrain")
    parser.add_argument("--load", default=None)
    parser.add_argument("--save", default="out/latest.pt")
    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)
