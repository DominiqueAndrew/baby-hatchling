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
        # Do CPU-side tensor operations to utilize CPU workers
        seq = torch.tensor(self.examples[idx], dtype=torch.long)
        # Pre-split on CPU (moves work from GPU to CPU)
        tokens = seq[:-1].clone()  # Clone to ensure memory is allocated on CPU
        targets = seq[1:].clone()
        return Batch(tokens=tokens, targets=targets)


def collate_fn(batch: Sequence[Batch]) -> Batch:
    # Do CPU-side batching operations to utilize CPU workers
    # Stack tensors on CPU before transfer to GPU
    tokens = torch.stack([item.tokens for item in batch], dim=0)
    targets = torch.stack([item.targets for item in batch], dim=0)
    # Ensure contiguous memory layout for efficient GPU transfer
    tokens = tokens.contiguous()
    targets = targets.contiguous()
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
    train_block = cfg.get("train", {})
    train_cfg = train_block.get(args.stage, train_block)
    if not train_cfg:
        raise ValueError(f"No training config found for stage '{args.stage}'.")
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
    use_gpu = torch.cuda.is_available()
    # Set CPU threads for PyTorch operations (helps with CPU utilization)
    num_cpu_threads = cfg.get("threads") or max(1, (os.cpu_count() or 4) - 2)
    torch.set_num_threads(num_cpu_threads)
    # Enable CPU parallelism for operations
    torch.set_num_interop_threads(max(1, num_cpu_threads // 2))
    if use_gpu:
        print(f"PyTorch CPU threads: {num_cpu_threads} (for CPU-side operations)")

    tokenizer = SentencePieceTokenizer()
    texts = load_texts(cfg.get("datasets", {}).get(args.stage, []))
    dataset = ChunkedDataset(texts, tokenizer, train_cfg["seq_len"])

    # Calculate initial batch size
    initial_batch_size = max(1, train_cfg["batch_tokens"] // train_cfg["seq_len"])
    
    model = build_model(cfg["model"])
    device = torch.device("cuda" if use_gpu else "cpu")
    print(f"Using device: {device}")
    
    # Optimize for GPU if available
    if use_gpu:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        # Get GPU memory info
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Total Memory: {total_memory:.2f} GB")
        
        # Compile model for faster execution (PyTorch 2.0+)
        # Skip compilation on Python 3.12+ as it's not supported
        try:
            import sys
            if sys.version_info < (3, 12):
                model = torch.compile(model, mode="reduce-overhead")
                print("Model compiled with torch.compile() for optimization")
            else:
                print("Skipping torch.compile() (not supported on Python 3.12+)")
        except Exception as e:
            print(f"torch.compile() not available or failed: {e}")
        
        # Adjust batch size based on available GPU memory and sequence length
        # RTX 3090 has 24GB, but we'll be conservative to avoid OOM
        # The KDA attention mechanism and NoPE global attention can be memory-intensive
        # Sequence length has quadratic impact on attention memory
        seq_len = train_cfg["seq_len"]
        
        # More aggressive reduction for longer sequences
        # Also consider model size (d_model) for memory estimation
        model_d_model = cfg["model"].get("d_model", 512)
        model_n_layers = cfg["model"].get("n_layers", 16)
        
        # Base reduction on sequence length
        if seq_len >= 2048:
            # For very long sequences, use 25% of calculated batch size
            batch_size = max(1, initial_batch_size // 4)
        elif seq_len >= 1024:
            # For medium sequences, use 50% of calculated batch size
            batch_size = max(1, initial_batch_size // 2)
        elif seq_len >= 512:
            # For medium-short sequences, use 60% of calculated batch size
            batch_size = max(1, int(initial_batch_size * 0.6))
        else:
            # For shorter sequences, use 75% of calculated batch size
            batch_size = max(1, int(initial_batch_size * 0.75))
        
        # Additional reduction for large models
        if model_d_model >= 640 or model_n_layers >= 24:
            batch_size = max(1, batch_size // 2)
            print(f"⚠️  Further reduced batch size to {batch_size} due to large model size (d_model={model_d_model}, layers={model_n_layers})")
        
        # Additional reduction based on total GPU memory
        if total_memory < 20:
            batch_size = max(1, batch_size // 2)
            print(f"⚠️  Further reduced batch size to {batch_size} due to limited GPU memory ({total_memory:.1f}GB)")
        
        if batch_size != initial_batch_size:
            print(f"⚠️  Reduced batch size from {initial_batch_size} to {batch_size} for memory safety")
            print(f"   (seq_len={seq_len}, KDA + NoPE attention can be memory-intensive)")
        
        print(f"Effective batch size: {batch_size} (effective tokens: {batch_size * train_cfg['seq_len']})")
    else:
        batch_size = initial_batch_size
    
    # Use multiple CPU workers for parallel data loading (CPU-GPU parallelism)
    # Reduce workers if memory is tight to avoid OOM
    if use_gpu:
        cpu_count = os.cpu_count() or 1
        # Reduce workers and prefetch for memory-constrained scenarios
        # With seq_len >= 1024, reduce workers to save GPU memory
        if train_cfg["seq_len"] >= 1024:
            num_workers = min(4, max(2, cpu_count // 4))  # Fewer workers for long sequences
            prefetch_factor = 2  # Less prefetching to save memory
        else:
            num_workers = min(8, max(4, cpu_count // 2))
            prefetch_factor = 4
        print(f"DataLoader using {num_workers} worker(s) for parallel data loading (out of {cpu_count} CPU cores)")
        print(f"Prefetch factor: {prefetch_factor} (keeps GPU fed with pre-loaded batches)")
        print(f"💡 CPU workers will handle tensor operations and batching to maximize CPU-GPU parallelism")
    else:
        num_workers = 0
        prefetch_factor = None
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=use_gpu,
        prefetch_factor=prefetch_factor,
        persistent_workers=num_workers > 0,
        timeout=60  # Timeout for worker processes (prevents hanging)
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
    
    if use_gpu:
        print(f"Gradient accumulation: {grad_accum} (effective batch: {batch_size * grad_accum})")

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
    
    # Print parallelism summary
    if use_gpu:
        print("\n" + "="*60)
        print("CPU-GPU Parallelism Configuration:")
        print(f"  • GPU: {torch.cuda.get_device_name(0)}")
        print(f"  • CPU Workers: {num_workers} (prefetching {prefetch_factor} batches)")
        print(f"  • Batch Size: {batch_size} (GPU processes)")
        print(f"  • Mixed Precision: Enabled (FP16)")
        print("="*60 + "\n")
    
    for epoch in range(10):  # small curriculum loop
        try:
            for batch in tqdm(dataloader, desc=f"{args.stage}-epoch{epoch}"):
                try:
                    # Clear cache before each batch to free up memory
                    if use_gpu:
                        torch.cuda.empty_cache()
                    
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
                        # Clear cache before optimizer step to free memory
                        if use_gpu:
                            torch.cuda.empty_cache()
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
                        
                        # Clear cache after logging to free memory
                        if use_gpu:
                            torch.cuda.empty_cache()
                        
                        step += 1
                        if step >= max_steps:
                            print(f"\n✅ Reached max_steps ({max_steps}). Training complete.")
                            return
                except torch.cuda.OutOfMemoryError as e:
                    if use_gpu:
                        torch.cuda.empty_cache()
                        print(f"\n❌ CUDA Out of Memory Error at step {step}, micro_step {micro_step}")
                        print(f"   Attempting to recover...")
                        # Try to save current state
                        if args.save:
                            try:
                                save_path = Path(args.save)
                                save_path.parent.mkdir(parents=True, exist_ok=True)
                                torch.save(model.state_dict(), save_path)
                                print(f"   Saved checkpoint to {save_path}")
                            except Exception as save_err:
                                print(f"   Failed to save checkpoint: {save_err}")
                        raise e
                    else:
                        raise e
                except Exception as e:
                    print(f"\n❌ Error during training at step {step}, micro_step {micro_step}")
                    print(f"   Error type: {type(e).__name__}")
                    print(f"   Error message: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    raise e
        except KeyboardInterrupt:
            print(f"\n⚠️  Training interrupted by user at step {step}")
            if args.save:
                try:
                    save_path = Path(args.save)
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), save_path)
                    print(f"   Checkpoint saved to {save_path}")
                except Exception as e:
                    print(f"   Failed to save checkpoint: {e}")
            return
        except Exception as e:
            print(f"\n❌ Fatal error in training loop at epoch {epoch}, step {step}")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error message: {str(e)}")
            import traceback
            traceback.print_exc()
            # Try to save checkpoint before exiting
            if args.save:
                try:
                    save_path = Path(args.save)
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), save_path)
                    print(f"   Emergency checkpoint saved to {save_path}")
                except Exception as save_err:
                    print(f"   Failed to save emergency checkpoint: {save_err}")
            raise e
        if step >= max_steps:
            print(f"\n✅ Reached max_steps ({max_steps}). Training complete.")
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
