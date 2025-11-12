"""Lightweight SentencePiece tokenizer helper."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List

import sentencepiece as spm

DEFAULT_CORPUS = """Baby-Hatchling blueprint describing hybrid KDA and NoPE attention,
unit-test RLVR, predictive coding auxiliaries, curiosity bonuses, and episodic memory.
"""


class SentencePieceTokenizer:
    def __init__(self, model_path: str = "data/tokenizer.model", vocab_size: int = 32000) -> None:
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.model_path.exists():
            self._train_default(vocab_size)
        self.processor = spm.SentencePieceProcessor()
        self.processor.load(str(self.model_path))
        self.bos_id = self.processor.bos_id()
        self.eos_id = self.processor.eos_id()
        if self.bos_id == -1:
            self.bos_id = self.processor.vocab_size()  # ensure IDs exist
        if self.eos_id == -1:
            self.eos_id = self.processor.vocab_size() + 1

    def _train_default(self, vocab_size: int) -> None:
        with tempfile.NamedTemporaryFile("w", delete=False) as handle:
            handle.write(DEFAULT_CORPUS)
            corpus_path = handle.name
        spm.SentencePieceTrainer.train(
            input=corpus_path,
            model_prefix=str(self.model_path.with_suffix("")),
            vocab_size=vocab_size,
            model_type="unigram",
            character_coverage=1.0,
            bos_id=1,
            eos_id=2,
            pad_id=0,
            unk_id=3,
            hard_vocab_limit=False,
        )

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        ids = self.processor.encode(text, out_type=int)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int]) -> str:
        return self.processor.decode(ids)

    @property
    def pad_id(self) -> int:
        pad = self.processor.pad_id()
        return pad if pad is not None and pad >= 0 else 0

    @property
    def vocab_size(self) -> int:
        return int(self.processor.vocab_size())
