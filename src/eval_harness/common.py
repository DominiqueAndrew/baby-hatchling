from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch

from ..model import build_model
from ..tokenizer import SentencePieceTokenizer
from ..utils.config import load_config
import torch.nn.functional as F


def load_model_and_tokenizer(config_path: str, ckpt_path: str):
    cfg = load_config(config_path)
    model = build_model(cfg["model"])
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Compile model for faster inference on GPU
    if device.type == "cuda":
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print(f"Evaluation model compiled for GPU: {torch.cuda.get_device_name(0)}")
        except Exception as e:
            print(f"torch.compile() not available or failed: {e}")
    
    tokenizer = SentencePieceTokenizer()
    return cfg, model, tokenizer


def greedy_generate(model, tokenizer, prompt: str, max_new_tokens: int = 128) -> str:
    tokens = tokenizer.encode(prompt)
    prompt_len = len(tokens)
    device = next(model.parameters()).device
    use_amp = device.type == "cuda"
    for _ in range(max_new_tokens):
        input_ids = torch.tensor(tokens[-model.cfg.max_seq :], dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred, _ = model(input_ids, use_memory=False)
            next_id = int(torch.argmax(pred.logits[0, -1]))
        tokens.append(next_id)
        if next_id == tokenizer.eos_id:
            break
    return tokenizer.decode(tokens[prompt_len:])


def option_score(model, tokenizer, context: str, option: str) -> float:
    context_ids = tokenizer.encode(context, add_eos=False)
    option_ids = tokenizer.encode(option, add_bos=False)
    tokens = context_ids + option_ids
    device = next(model.parameters()).device
    use_amp = device.type == "cuda"
    input_ids = torch.tensor(tokens[:-1], dtype=torch.long).unsqueeze(0).to(device)
    targets = torch.tensor(tokens[1:], dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred, _ = model(input_ids, use_memory=False)
        log_probs = F.log_softmax(pred.logits, dim=-1)
        gathered = torch.gather(log_probs, 2, targets.unsqueeze(-1)).squeeze(-1)
    context_len = max(len(context_ids) - 1, 0)
    return float(gathered[:, context_len:].sum().item())
