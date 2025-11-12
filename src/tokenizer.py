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
        else:
            # Check if existing tokenizer matches requested vocab_size
            temp_processor = spm.SentencePieceProcessor()
            temp_processor.load(str(self.model_path))
            existing_vocab_size = int(temp_processor.vocab_size())
            if existing_vocab_size != vocab_size:
                print(
                    f"Warning: Existing tokenizer has vocab_size={existing_vocab_size}, "
                    f"but {vocab_size} was requested. Recreating tokenizer..."
                )
                # Delete existing tokenizer files
                self.model_path.unlink(missing_ok=True)
                vocab_path = self.model_path.with_suffix(".vocab")
                vocab_path.unlink(missing_ok=True)
                # Create new tokenizer with correct vocab_size
                self._train_default(vocab_size)
        self.processor = spm.SentencePieceProcessor()
        self.processor.load(str(self.model_path))
        # Verify the tokenizer was created with the correct vocab size
        actual_vocab_size = int(self.processor.vocab_size())
        if actual_vocab_size != vocab_size:
            raise ValueError(
                f"Failed to create tokenizer with vocab_size={vocab_size}. "
                f"Created tokenizer has vocab_size={actual_vocab_size}. "
                f"This usually means the training corpus is too small. "
                f"Please delete {self.model_path} and ensure you have sufficient training data."
            )
        self.bos_id = self.processor.bos_id()
        self.eos_id = self.processor.eos_id()
        if self.bos_id == -1:
            self.bos_id = self.processor.vocab_size()  # ensure IDs exist
        if self.eos_id == -1:
            self.eos_id = self.processor.vocab_size() + 1

    def _train_default(self, vocab_size: int) -> None:
        import os
        import string
        
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as handle:
            # Create a large synthetic corpus with enough diversity to support the vocab size
            # This includes various character combinations, numbers, and text patterns
            expanded_corpus = []
            
            # Base corpus
            expanded_corpus.append(DEFAULT_CORPUS)
            
            # Add character-level diversity: all printable ASCII characters
            expanded_corpus.append(" ".join(string.printable))
            
            # Add number sequences
            for i in range(1000):
                expanded_corpus.append(f"Number {i} is {i * 2} and {i * 3}.")
            
            # Add word combinations with various patterns
            words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
            for i in range(500):
                expanded_corpus.append(" ".join([f"{w}{i}" for w in words]))
            
            # Add character n-grams
            chars = string.ascii_lowercase + string.digits
            for i in range(0, len(chars), 3):
                expanded_corpus.append(" ".join(chars[i:i+10]))
            
            # Add repeated patterns with variations
            base_text = DEFAULT_CORPUS
            for i in range(200):
                expanded_corpus.append(f"{base_text} Variation {i}. " + " ".join([chr(ord('a') + (j % 26)) for j in range(20)]))
            
            # Write the expanded corpus
            handle.write("\n".join(expanded_corpus))
            corpus_path = handle.name
        
        try:
            # Use hard_vocab_limit=True to force creation of exactly vocab_size tokens
            # This ensures we get the full vocabulary even if the corpus is small
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
                hard_vocab_limit=True,  # Force exact vocab size
                split_by_unicode_script=True,
                split_by_number=True,
                split_by_whitespace=True,
            )
        finally:
            # Clean up temp file
            try:
                os.unlink(corpus_path)
            except OSError:
                pass

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
