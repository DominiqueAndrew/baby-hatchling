"""Training entry-point for Baby-Hatchling with curriculum + quick-win optimizations."""
from __future__ import annotations

import argparse
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info
from tqdm import tqdm

from .model import BabyHatchlingModel, build_model
from .optim import AdaPM
from .tokenizer import SentencePieceTokenizer
from .utils.config import load_config
from .utils.data import contamination_report, load_text_splits, stream_dataset_texts
from .utils.loss import make_next_token_labels
from .utils.logging import CSVLogger
from .utils.sparsity import MSTSparsifier, SparsitySchedule, collect_sparse_modules


@dataclass
class Batch:
    tokens: torch.Tensor  # [B,T]
    targets: torch.Tensor  # [B,T]


class CudaPrefetcher:
    """Prefetches batches to GPU on a separate CUDA stream for overlapping H2D transfers."""
    
    def __init__(self, loader: DataLoader, device: torch.device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream() if device.type == "cuda" else None
        self.preload = None
        self.preload_iter = None
    
    def __iter__(self):
        self.preload_iter = iter(self.loader)
        if self.stream is not None:
            # Preload first batch on CUDA stream
            try:
                with torch.cuda.stream(self.stream):
                    batch = next(self.preload_iter)
                    self.preload = Batch(
                        tokens=batch.tokens.to(self.device, non_blocking=True),
                        targets=batch.targets.to(self.device, non_blocking=True),
                    )
            except StopIteration:
                self.preload = None
        else:
            # CPU case: just yield batches directly (no prefetching benefit)
            for batch in self.preload_iter:
                yield Batch(
                    tokens=batch.tokens.to(self.device),
                    targets=batch.targets.to(self.device),
                )
            return
        
        while True:
            if self.preload is None:
                break
            
            # Swap current and preload
            current = self.preload
            self.preload = None
            
            if self.stream is not None:
                # Wait for current batch to be ready
                torch.cuda.current_stream().wait_stream(self.stream)
                
                # Preload next batch in background
                try:
                    with torch.cuda.stream(self.stream):
                        batch = next(self.preload_iter)
                        self.preload = Batch(
                            tokens=batch.tokens.to(self.device, non_blocking=True),
                            targets=batch.targets.to(self.device, non_blocking=True),
                        )
                except StopIteration:
                    pass
            
            yield current


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
        tokens = seq[:-1].clone()
        targets = seq[1:].clone()
        return Batch(tokens=tokens, targets=targets)


class StreamingChunkedDataset(IterableDataset):
    """Iterable dataset that tokenizes streaming samples on the fly."""

    def __init__(
        self,
        specs: Sequence[dict],
        tokenizer: SentencePieceTokenizer,
        seq_len: int,
        *,
        seed: int,
    ) -> None:
        if not specs:
            raise ValueError("StreamingChunkedDataset requires at least one dataset spec.")
        self.specs = list(specs)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.seed = seed

    def __iter__(self):
        worker = get_worker_info()
        worker_seed = self.seed + (worker.id if worker else 0)
        rng = random.Random(worker_seed)
        token_buffer: list[int] = []
        while True:
            for spec in rng.sample(self.specs, len(self.specs)):
                for text in stream_dataset_texts(spec):
                    ids = self.tokenizer.encode(text)
                    if not ids:
                        continue
                    token_buffer.extend(ids)
                    while len(token_buffer) > self.seq_len:
                        window = token_buffer[: self.seq_len + 1]
                        token_buffer = token_buffer[self.seq_len :]
                        seq = torch.tensor(window, dtype=torch.long)
                        yield Batch(tokens=seq[:-1], targets=seq[1:])


def parameter_checksum(model: nn.Module, sample_size: int = 2048) -> float:
    remaining = sample_size
    checksum = 0.0
    for param in model.parameters():
        if not param.requires_grad or remaining <= 0:
            continue
        flat = param.detach().view(-1)
        if flat.numel() == 0:
            continue
        take = min(remaining, flat.numel())
        checksum += float(flat[:take].abs().sum().item())
        remaining -= take
    return checksum


def grad_flow_summary(model: nn.Module) -> tuple[float, float, dict]:
    total = 0
    active = 0
    global_sq = 0.0
    per_block: dict[str, float] = defaultdict(float)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        total += 1
        if param.grad is None:
            continue
        active += 1
        grad = param.grad.detach()
        global_sq += float(torch.sum(grad * grad).item())
        block = name.split(".", 1)[0]
        per_block[block] += float(grad.norm().item())
    percent = 100.0 * active / max(1, total)
    global_norm = math.sqrt(max(global_sq, 0.0))
    return percent, global_norm, dict(per_block)


def assert_training_mode(model: nn.Module) -> None:
    if not model.training:
        raise RuntimeError("Model left training mode; ensure model.train() is active.")
    for name, module in model.named_modules():
        if hasattr(module, "training") and not module.training:
            raise RuntimeError(f"Submodule '{name}' switched to eval() during training.")


def collate_fn(batch: Sequence[Batch]) -> Batch:
    tokens = torch.stack([item.tokens for item in batch], dim=0)
    targets = torch.stack([item.targets for item in batch], dim=0)
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


def build_curriculum(train_cfg: dict, max_steps: int) -> List[dict]:
    curriculum_cfg = train_cfg.get("curriculum")
    default_seq = train_cfg.get("seq_len", curriculum_cfg[-1].get("seq_len") if curriculum_cfg else None)
    default_btoks = train_cfg.get("batch_tokens", curriculum_cfg[-1].get("batch_tokens") if curriculum_cfg else None)
    default_accum = train_cfg.get("grad_accum", curriculum_cfg[-1].get("grad_accum") if curriculum_cfg else 1)
    if default_seq is None or default_btoks is None:
        raise KeyError("Training config must set 'seq_len' and 'batch_tokens' when using curriculum.")
    if not curriculum_cfg:
        return [
            {
                "seq_len": default_seq,
                "batch_tokens": default_btoks,
                "grad_accum": default_accum,
                "steps": max_steps,
            }
        ]
    stages: List[dict] = []
    consumed = 0
    for stage in curriculum_cfg:
        stage_steps = stage.get("steps")
        if stage_steps is None:
            raise ValueError("Each curriculum stage must define 'steps'.")
        stages.append(
            {
                "seq_len": stage.get("seq_len", default_seq),
                "batch_tokens": stage.get("batch_tokens", default_btoks),
                "grad_accum": stage.get("grad_accum", default_accum),
                "steps": stage_steps,
            }
        )
        consumed += stage_steps
    if consumed < max_steps:
        stages.append(
            {
                "seq_len": default_seq,
                "batch_tokens": default_btoks,
                "grad_accum": default_accum,
                "steps": max_steps - consumed,
            }
        )
    # Trim excess
    total = 0
    trimmed: List[dict] = []
    for stage in stages:
        if total >= max_steps:
            break
        stage_steps = min(stage["steps"], max_steps - total)
        trimmed.append({**stage, "steps": stage_steps})
        total += stage_steps
    return trimmed


def _gpu_mem_util(device: torch.device | None) -> tuple[float, float] | tuple[None, None]:
    if device is None or device.type != "cuda" or not torch.cuda.is_available():
        return None, None
    try:
        idx = device.index if device.index is not None else torch.cuda.current_device()
        free_mem, total_mem = torch.cuda.mem_get_info(idx)
        util = 1 - (free_mem / max(1, total_mem))
        return util, total_mem
    except (RuntimeError, AssertionError):
        return None, None


def auto_workers(use_gpu: bool, seq_len: int = 1024, device: torch.device | None = None) -> tuple[int, int | None]:
    if not use_gpu:
        return 0, None
    cpu_count = os.cpu_count() or 4
    # Reduce workers for shorter sequences to save GPU memory
    # Shorter sequences = more batches in memory = need less CPU workers
    if seq_len <= 256:
        num_workers = min(4, max(2, cpu_count // 4))
        prefetch = 2  # Less prefetching for memory-constrained scenarios
    elif seq_len <= 512:
        num_workers = min(6, max(3, cpu_count // 3))
        prefetch = 3
    else:
        num_workers = min(12, max(4, cpu_count // 2))
        prefetch = 4
    util_info = _gpu_mem_util(device)
    if util_info[0] is not None:
        util = util_info[0]
        if util < 0.3:
            num_workers = min(cpu_count, max(num_workers * 2, 2))
            prefetch = (prefetch or 2) * 2
        elif util > 0.85:
            num_workers = max(1, num_workers // 2)
            prefetch = max(2, (prefetch or 2) // 2)
    return num_workers, prefetch


def find_max_batch_tokens(
    model: nn.Module,
    seq_len: int,
    device: torch.device,
    start: int = 4096,
    step: int = 2048,
    floor: int = 2048,
    max_tokens: int = 16384,
) -> int:
    """Find maximum safe batch_tokens via binary search with OOM detection.
    
    Parameters
    ----------
    model: Model to test with
    seq_len: Sequence length to use
    device: GPU device
    start: Starting batch_tokens to test
    step: Initial step size for binary search
    floor: Minimum step size before stopping
    max_tokens: Maximum batch_tokens to test
    
    Returns
    -------
    Maximum safe batch_tokens that doesn't cause OOM
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return start
    
    was_training = model.training
    model.eval()
    batch_size = max(1, start // seq_len)
    ok = start
    current = start
    
    print(f"[batch-finder] Starting search for max batch_tokens (seq_len={seq_len})...")
    
    with torch.no_grad():
        while True:
            try:
                # Clear cache before test
                torch.cuda.empty_cache()
                
                # Create dummy batch
                tokens = torch.randint(0, 1000, (batch_size, seq_len), device=device)
                
                # Try forward pass
                with autocast():
                    _ = model(tokens)
                
                # Success - try larger
                ok = current
                if current >= max_tokens:
                    break
                if step <= floor:
                    current = min(current + step, max_tokens)
                    if current == ok:
                        break
                else:
                    current = min(current + step, max_tokens)
                    if current == ok:
                        step = max(floor, step // 2)
                        if step <= floor:
                            break
                
                batch_size = max(1, current // seq_len)
                
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if step <= floor:
                    break
                step = max(floor, step // 2)
                current = ok + step if current != ok + step else ok
                if current == ok:
                    break
                batch_size = max(1, current // seq_len)
            except Exception as e:
                # Non-OOM error - be conservative
                print(f"[batch-finder] Error during test: {e}")
                break
    
    torch.cuda.empty_cache()
    if was_training:
        model.train()
    
    # Align to seq_len
    aligned = max(seq_len, (ok // seq_len) * seq_len)
    print(f"[batch-finder] Found max safe batch_tokens: {aligned}")
    return aligned


def maybe_scale_batch_tokens(
    batch_tokens: int, seq_len: int, device: torch.device | None, cfg: dict | None
) -> int:
    if cfg is None or not cfg.get("enabled"):
        return batch_tokens
    util_info = _gpu_mem_util(device)
    if util_info[0] is None:
        return batch_tokens
    util = util_info[0]
    target = cfg.get("target_util", 0.7)
    floor = cfg.get("min_util", 0.15)
    max_factor = cfg.get("max_factor", 2.0)
    max_tokens = cfg.get("max_tokens", batch_tokens)
    if util >= target:
        return batch_tokens
    scale = min(max_factor, target / max(util, floor))
    new_tokens = min(int(batch_tokens * scale), max_tokens)
    if new_tokens <= batch_tokens:
        return batch_tokens
    aligned = max(seq_len, (new_tokens // seq_len) * seq_len)
    print(
        f"[auto-batch] GPU util {util:.1%} < target {target:.0%}. "
        f"Increasing batch_tokens {batch_tokens} -> {aligned}"
    )
    return aligned


def build_stage_loader(
    data_source,
    tokenizer: SentencePieceTokenizer,
    stage_cfg: dict,
    use_gpu: bool,
    device: torch.device,
    *,
    streaming: bool,
    seed: int,
) -> tuple[DataLoader, int, int, int, int, int | None, int]:
    seq_len = stage_cfg["seq_len"]
    batch_tokens = stage_cfg["batch_tokens"]
    batch_tokens = maybe_scale_batch_tokens(batch_tokens, seq_len, device if use_gpu else None, stage_cfg.get("dynamic_batch"))
    batch_size = max(1, batch_tokens // seq_len)
    if streaming:
        dataset = StreamingChunkedDataset(data_source, tokenizer, seq_len, seed=seed)
        num_workers = stage_cfg.get("num_workers", 8 if use_gpu else 0)
        prefetch = stage_cfg.get("prefetch_factor", 4 if num_workers > 0 else None)
        loader_kwargs = dict(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=use_gpu,
            persistent_workers=num_workers > 0,
        )
        if num_workers > 0 and prefetch:
            loader_kwargs["prefetch_factor"] = prefetch
        dataloader = DataLoader(**loader_kwargs)
    else:
        dataset = ChunkedDataset(data_source, tokenizer, seq_len)
        num_workers, prefetch = auto_workers(use_gpu, seq_len, device if use_gpu else None)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=use_gpu,
            prefetch_factor=prefetch,
            persistent_workers=num_workers > 0,
            timeout=60,
        )
    grad_accum = stage_cfg.get("grad_accum", 1)
    return dataloader, batch_size, grad_accum, seq_len, num_workers, prefetch, batch_tokens


def build_scheduler(optimizer: torch.optim.Optimizer, lr_cfg: dict, total_steps: int):
    if not lr_cfg:
        return None
    warmup = lr_cfg.get("warmup_steps", 1000)
    min_factor = lr_cfg.get("min_factor", 0.1)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup:
            return max(1e-3, current_step / max(1, warmup))
        progress = min(1.0, (current_step - warmup) / max(1, total_steps - warmup))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_factor + (1 - min_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def build_optimizer(model: BabyHatchlingModel, optim_cfg: dict) -> torch.optim.Optimizer:
    opt_type = optim_cfg.get("type", "adamw").lower()
    if opt_type == "adapm":
        return AdaPM(
            model.parameters(),
            lr=optim_cfg.get("lr", 2e-4),
            beta=optim_cfg.get("beta", 0.9),
            gamma=optim_cfg.get("gamma", 0.3),
            eps=optim_cfg.get("eps", 1e-8),
            weight_decay=optim_cfg.get("weight_decay", 0.0),
        )
    return torch.optim.AdamW(
        model.parameters(),
        lr=optim_cfg.get("lr", 2e-4),
        betas=tuple(optim_cfg.get("betas", [0.9, 0.95])),
        eps=optim_cfg.get("eps", 1e-8),
        weight_decay=optim_cfg.get("weight_decay", 0.0),
    )


def train(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    train_block = cfg["train"]
    stage_cfg = train_block.get(args.stage) or train_block.get("pretrain" if args.stage == "pretrain" else args.stage)
    if stage_cfg is None:
        raise KeyError(f"No training config found for stage '{args.stage}'.")
    optim_cfg = cfg["optim"]
    loss_cfg = cfg["loss"]
    torch.set_float32_matmul_precision("high")

    if args.stage == "gridworld":
        try:
            from .gridworld import run_gridworld_stage

            run_gridworld_stage(cfg, args)
        except ImportError as exc:  # pragma: no cover - optional feature
            raise ImportError(
                "gridworld module not found. The gridworld functionality has been removed. "
                "Please use 'pretrain' or 'sft' stages instead."
            ) from exc
        return

    set_seed(cfg.get("seed", 0))
    # Set environment variables to avoid thread oversubscription with dataloader workers
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    use_gpu = torch.cuda.is_available()
    num_cpu_threads = cfg.get("threads") or max(1, (os.cpu_count() or 4) - 2)
    torch.set_num_threads(num_cpu_threads)
    torch.set_num_interop_threads(max(1, num_cpu_threads // 2))
    if use_gpu:
        print(f"PyTorch CPU threads: {num_cpu_threads}")

    # Get vocab_size from model config to ensure tokenizer matches
    model_vocab_size = cfg["model"].get("vocab_size", 32000)
    tokenizer = SentencePieceTokenizer(vocab_size=model_vocab_size)
    pad_token_id = tokenizer.pad_id
    if tokenizer.eos_id == pad_token_id:
        raise ValueError("Tokenizer PAD and EOS IDs must differ for stable training.")
    
    # Use the actual tokenizer vocab size (may be less than requested for synthetic corpora)
    actual_vocab_size = tokenizer.vocab_size
    if actual_vocab_size != model_vocab_size:
        print(
            f"Note: Using tokenizer vocab_size={actual_vocab_size} instead of "
            f"config vocab_size={model_vocab_size}. Updating model config."
        )
        # Update model config to match tokenizer
        cfg["model"]["vocab_size"] = actual_vocab_size
    
    dataset_specs = cfg.get("datasets", {}).get(args.stage, [])
    streaming_loader = train_block.get("use_streaming_loader", False)
    if streaming_loader:
        data_source = dataset_specs
        texts = None
    else:
        texts = load_texts(dataset_specs)
        data_source = texts

    model = build_model(cfg["model"])
    device = torch.device("cuda" if use_gpu else "cpu")
    model.to(device)
    if model.embedding.num_embeddings != tokenizer.vocab_size:
        raise ValueError(
            f"Tokenizer vocab ({tokenizer.vocab_size}) != model embeddings ({model.embedding.num_embeddings})"
        )
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable_params == 0:
        raise RuntimeError("No trainable parameters were found.")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    checksum_steps = {0, 10}
    logged_checksums: dict[int, bool] = {}

    def maybe_log_checksum(step_idx: int) -> None:
        if step_idx in checksum_steps and step_idx not in logged_checksums:
            print(f"[param] step={step_idx} checksum={parameter_checksum(model):.4e}")
            logged_checksums[step_idx] = True

    maybe_log_checksum(0)

    if use_gpu:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        
        # Enable TF32 for Ampere+ GPUs (faster matmuls)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TF32 enabled for matmuls and cuDNN")

    if args.load and Path(args.load).exists():
        state = torch.load(args.load, map_location="cpu")
        model.load_state_dict(state, strict=False)
    
    # Compile model for faster execution (after loading weights)
    if use_gpu:
        try:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
            print("Model compiled with torch.compile() for optimization")
        except Exception as e:
            print(f"torch.compile() not available or failed: {e}")
            print("Continuing without compilation")

    optimizer = build_optimizer(model, optim_cfg)

    grad_clip = stage_cfg.get("grad_clip", 1.0)
    max_steps = stage_cfg.get("max_steps", 1000)
    curriculum = build_curriculum(stage_cfg, max_steps)
    
    # Find max safe batch_tokens for first stage if GPU available
    if use_gpu and curriculum:
        first_seq_len = curriculum[0]["seq_len"]
        first_batch_tokens = curriculum[0]["batch_tokens"]
        dynamic_cfg = stage_cfg.get("dynamic_batch", {})
        max_tokens_cap = dynamic_cfg.get("max_tokens", 16384)
        
        try:
            found_max = find_max_batch_tokens(
                model, first_seq_len, device,
                start=first_batch_tokens,
                max_tokens=max_tokens_cap
            )
            # Update first stage and use as reference for others
            curriculum[0]["batch_tokens"] = found_max
            # Update max_tokens in dynamic_batch config if it's lower
            if found_max < max_tokens_cap:
                dynamic_cfg["max_tokens"] = found_max
                print(f"[batch-finder] Updated dynamic_batch.max_tokens to {found_max}")
        except Exception as e:
            print(f"[batch-finder] Failed to find max batch_tokens: {e}")
            print(f"[batch-finder] Using config value {first_batch_tokens} instead")

    logger = CSVLogger(
        Path("logs") / f"train_{args.stage}.csv",
        ["step", "loss", "lm", "pc", "epi", "bonus", "lr", "grad_pct", "grad_norm"],
    )
    scheduler = build_scheduler(optimizer, stage_cfg.get("lr_schedule", {}), max_steps)

    sparsifier = None
    sparsity_cfg = stage_cfg.get("sparsity", {})
    if sparsity_cfg.get("enabled"):
        sparse_modules = collect_sparse_modules(model)
        if sparse_modules:
            schedule = SparsitySchedule(
                warmup=sparsity_cfg.get("warmup", 0),
                prune=sparsity_cfg.get("prune", 0),
                restore=sparsity_cfg.get("restore", 0),
                target=sparsity_cfg.get("target", 0.0),
                update_every=sparsity_cfg.get("update_every", 1000),
            )
            sparsifier = MSTSparsifier(sparse_modules, schedule)
            sparsifier.apply_masks()

    use_amp = use_gpu
    scaler = GradScaler(enabled=use_amp)
    if use_amp:
        print("Using mixed precision training (FP16)")

    early_cfg = stage_cfg.get("early_stopping", {})
    use_early = early_cfg.get("enabled", False)
    patience = early_cfg.get("patience", 0)
    min_delta = early_cfg.get("min_delta", 0.0)
    best_loss = float("inf")
    steps_since_improve = 0

    stage_index = 0
    seed = cfg.get("seed", 0)
    dataloader, batch_size, grad_accum, seq_len, workers, prefetch, stage_batch_tokens = build_stage_loader(
        data_source,
        tokenizer,
        curriculum[stage_index],
        use_gpu,
        device,
        streaming=streaming_loader,
        seed=seed,
    )
    
    # Optionally wrap with CUDA prefetcher for overlapping H2D transfers
    use_prefetcher = use_gpu and stage_cfg.get("use_cuda_prefetcher", False)
    if use_prefetcher:
        dataloader = CudaPrefetcher(dataloader, device)
        print("Using CUDA prefetcher for overlapping data transfers")
    
    dataloader_iter = iter(dataloader)
    stage_step = 0
    frozen_batch: Batch | None = None

    if use_gpu:
        effective_tokens = batch_size * seq_len
        print(
            f"Stage 0 ⇒ seq_len={seq_len}, batch_tokens={stage_batch_tokens}, batch_size={batch_size}, grad_accum={grad_accum}, "
            f"effective tokens={effective_tokens}, workers={workers}, prefetch={prefetch}"
        )

    step = 0
    micro_step = 0
    # Initialize progress bar with smoothing and mininterval to avoid inflated ETA from slow first step
    # smoothing=0.1 helps average out the first few slow steps
    # mininterval=10 means update at most every 10 seconds to avoid flickering
    progress = tqdm(total=max_steps, desc=args.stage, smoothing=0.1, miniters=1, mininterval=10)

    model.train()
    optimizer.zero_grad()

    try:
        while step < max_steps:
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)
            if args.overfit_one_batch:
                if frozen_batch is None:
                    frozen_batch = batch
                else:
                    batch = frozen_batch

            # If using prefetcher, batches are already on GPU
            if use_prefetcher:
                tokens = batch.tokens
            else:
                tokens = batch.tokens.to(device, non_blocking=use_gpu)
            labels = make_next_token_labels(tokens, pad_token_id)
            assert_training_mode(model)

            with autocast(enabled=use_amp):
                pred_out, _ = model(tokens)

            logits = pred_out.logits
            lm_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
            lambda_pc = loss_cfg["lambda_pc"] if getattr(model.pred_head, "enabled", True) else 0.0
            total_loss = lm_loss + lambda_pc * pred_out.pc_loss
            epi_penalty = torch.tensor(model.episodic_gate_penalty(), device=device)
            total_loss = total_loss + loss_cfg.get("lambda_epi", 0.0) * epi_penalty
            bonus = 0.0
            if model.use_curiosity and getattr(model.pred_head, "enabled", True):
                bonus = model.pred_head.curiosity_bonus(pred_out.error_trace)
            
            loss = total_loss / grad_accum
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            micro_step += 1

            if micro_step % grad_accum == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                # Only clear cache every 100 steps to avoid slowdown from synchronous calls
                if use_gpu and step % 100 == 0:
                    torch.cuda.empty_cache()

                current_lr = optimizer.param_groups[0]["lr"]
                if step % 20 == 0:
                    print(f"[lr] step={step} lr={current_lr:.3e}")
                if use_amp and step % 20 == 0:
                    print(f"[amp] step={step} scale={scaler.get_scale():.2e}")
                if step % 10 == 0:
                    grad_pct, grad_norm, block_norms = grad_flow_summary(model)
                    logger.log(
                        {
                            "step": step,
                            "loss": float(total_loss.item()),
                            "lm": float(lm_loss.item()),
                            "pc": float(pred_out.pc_loss.item()),
                            "epi": float(epi_penalty.item()),
                            "bonus": float(bonus),
                            "lr": float(current_lr),
                            "grad_pct": float(grad_pct),
                            "grad_norm": float(grad_norm),
                        }
                    )
                    if step % 50 == 0 and block_norms:
                        top = ", ".join(f"{name}:{val:.2e}" for name, val in list(block_norms.items())[:3])
                        print(f"[grad] step={step} active={grad_pct:.1f}% norm={grad_norm:.2e} {top}")
                if step % 100 == 0 and getattr(model.pred_head, "enabled", True):
                    mean = float(model.pred_head.ema_bonus_mean.item())
                    std = float(model.pred_head.ema_bonus_var.sqrt().item())
                    print(f"[bonus] step={step} mean={mean:.3e} std={std:.3e}")

                # Save checkpoint every 1000 steps instead of every step to avoid massive slowdown
                if args.save and (step % 1000 == 0 or step == max_steps - 1):
                    save_path = Path(args.save)
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), save_path)
                    if step % 5000 == 0:  # Print confirmation every 5000 steps
                        print(f"\n💾 Checkpoint saved to {save_path}")

                step += 1
                maybe_log_checksum(step)
                if sparsifier is not None:
                    sparsifier.maybe_update(step)
                stage_step += 1
                progress.update(1)

                # Early stopping check - only every 100 steps to avoid sync overhead
                if use_early and step % 100 == 0:
                    current = float(total_loss.item())
                    if current < best_loss - min_delta:
                        best_loss = current
                        steps_since_improve = 0
                    else:
                        steps_since_improve += 100
                        if steps_since_improve >= patience:
                            print("\n⏹ Early stopping triggered.")
                            break

                if stage_step >= curriculum[stage_index]["steps"]:
                    # Save checkpoint at stage transition
                    if args.save:
                        save_path = Path(args.save)
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(model.state_dict(), save_path)
                        print(f"\n💾 Stage {stage_index} complete - checkpoint saved to {save_path}")
                    
                    stage_index += 1
                    if stage_index >= len(curriculum):
                        break
                    dataloader, batch_size, grad_accum, seq_len, workers, prefetch, stage_batch_tokens = build_stage_loader(
                        data_source,
                        tokenizer,
                        curriculum[stage_index],
                        use_gpu,
                        device,
                        streaming=streaming_loader,
                        seed=seed,
                    )
                    # Re-wrap with prefetcher if enabled
                    if use_prefetcher:
                        dataloader = CudaPrefetcher(dataloader, device)
                    dataloader_iter = iter(dataloader)
                    stage_step = 0
                    micro_step = 0
                    if use_gpu:
                        eff_tokens = batch_size * seq_len
                        print(
                            f"\n→ Stage {stage_index} transition: seq_len={seq_len}, batch_tokens={stage_batch_tokens}, "
                            f"batch_size={batch_size}, grad_accum={grad_accum}, effective tokens={eff_tokens}, workers={workers}"
                        )

            if step >= max_steps or (use_early and steps_since_improve >= patience):
                break

    except torch.cuda.OutOfMemoryError as exc:
        if use_gpu:
            torch.cuda.empty_cache()
            print("\n❌ CUDA OOM — saving checkpoint before exiting.")
            if args.save:
                save_path = Path(args.save)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_path)
                print(f"   Emergency checkpoint saved to {save_path}")
        raise exc
    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted at step {step}")
        if args.save:
            save_path = Path(args.save)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"   Checkpoint saved to {save_path}")
        return
    finally:
        progress.close()

    print(f"\n✅ Training complete at step {step}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Baby-Hatchling Trainer")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=["pretrain", "sft", "gridworld"], default="pretrain")
    parser.add_argument("--load", default=None)
    parser.add_argument("--save", default="out/latest.pt")
    parser.add_argument(
        "--overfit_one_batch",
        action="store_true",
        help="Repeat the first batch every step (sanity check for labels/grad paths).",
    )
    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)
