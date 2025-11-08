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
    model.eval()
    tokenizer = SentencePieceTokenizer()
    return cfg, model, tokenizer


def greedy_generate(model, tokenizer, prompt: str, max_new_tokens: int = 128) -> str:
    tokens = tokenizer.encode(prompt)
    prompt_len = len(tokens)
    for _ in range(max_new_tokens):
        input_ids = torch.tensor(tokens[-model.cfg.max_seq :], dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
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
    input_ids = torch.tensor(tokens[:-1], dtype=torch.long).unsqueeze(0)
    targets = torch.tensor(tokens[1:], dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        pred, _ = model(input_ids, use_memory=False)
        log_probs = F.log_softmax(pred.logits, dim=-1)
        gathered = torch.gather(log_probs, 2, targets.unsqueeze(-1)).squeeze(-1)
    context_len = max(len(context_ids) - 1, 0)
    return float(gathered[:, context_len:].sum().item())
