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
            # We need extensive character-level diversity to allow SentencePiece to create many tokens
            expanded_corpus = []
            
            # Base corpus
            expanded_corpus.append(DEFAULT_CORPUS)
            
            # Add all printable ASCII characters in various combinations
            expanded_corpus.append(" ".join(string.printable))
            expanded_corpus.append("".join(string.printable))
            
            # Add extensive number sequences with various formats
            for i in range(5000):
                expanded_corpus.append(f"Number {i} is {i * 2} and {i * 3}.")
                expanded_corpus.append(f"Value {i:05d} equals {i * 1.5:.2f}.")
                expanded_corpus.append(f"Count {i} {i+1} {i+2} {i+3} {i+4}")
            
            # Add word combinations with extensive variations
            words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "cat", "bird"]
            for i in range(2000):
                expanded_corpus.append(" ".join([f"{w}{i}" for w in words]))
                expanded_corpus.append(" ".join([f"{w}_{i}" for w in words]))
                expanded_corpus.append(" ".join([f"{w}-{i}" for w in words]))
            
            # Add character n-grams with all combinations
            chars = string.ascii_lowercase + string.ascii_uppercase + string.digits + string.punctuation
            for i in range(0, len(chars), 2):
                expanded_corpus.append(" ".join(chars[i:i+15]))
                expanded_corpus.append("".join(chars[i:i+15]))
            
            # Add all possible 2-character combinations (for more token diversity)
            for i, c1 in enumerate(string.ascii_lowercase[:26]):
                for j, c2 in enumerate(string.ascii_lowercase[:26]):
                    if i * 26 + j < 1000:  # Limit to avoid too many
                        expanded_corpus.append(f"{c1}{c2} {c1}{c2}{i} {c2}{c1}{j}")
            
            # Add repeated patterns with extensive variations
            base_text = DEFAULT_CORPUS
            for i in range(1000):
                expanded_corpus.append(f"{base_text} Variation {i}. " + " ".join([chr(ord('a') + (j % 26)) for j in range(30)]))
                expanded_corpus.append(f"{base_text} Version {i:04d}. " + "".join([chr(ord('A') + (j % 26)) for j in range(30)]))
            
            # Add Unicode character sequences for additional diversity
            for i in range(500):
                expanded_corpus.append(" ".join([chr(0x0020 + (i * 7 + j) % 0x007F) for j in range(20) if 0x0020 <= (0x0020 + (i * 7 + j) % 0x007F) <= 0x007E]))
            
            # Write the expanded corpus
            handle.write("\n".join(expanded_corpus))
            corpus_path = handle.name
        
        try:
            # Use byte_fallback to allow creation of full vocab_size
            # This enables byte-level encoding for tokens that can't be represented
            # Otherwise, we're limited by the corpus diversity
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
                byte_fallback=True,  # Allow byte-level fallback to reach vocab_size
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
