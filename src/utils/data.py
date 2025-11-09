"""Dataset helpers including contamination checks."""
from __future__ import annotations

import glob
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

from datasketch import MinHash, MinHashLSH
from datasets import DatasetDict, load_dataset


@dataclass
class Sample:
    text: str
    source: str


def _render_template(template: str, example: Mapping[str, object]) -> str:
    class DefaultDict(dict):
        def __missing__(self, key: str) -> str:
            return ""

    safe_map = DefaultDict({k: example.get(k, "") for k in example})
    return template.format_map(safe_map)


def _load_local_samples(path_pattern: str, name: str | None = None) -> List[Sample]:
    samples: List[Sample] = []
    for path in sorted(glob.glob(path_pattern)):
        file_path = Path(path)
        source = name or file_path.name
        if file_path.suffix == ".jsonl":
            with file_path.open("r", encoding="utf8") as handle:
                for line in handle:
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = data.get("text") or data.get("content") or ""
                    if text:
                        samples.append(Sample(text=text, source=source))
        else:
            text = file_path.read_text(encoding="utf8")
            if text:
                samples.append(Sample(text=text, source=source))
    return samples


def load_text_splits(specs: Sequence[dict], text_field: str = "text") -> List[Sample]:
    """Loads a list of text samples from HF datasets with parallel processing.

    Parameters
    ----------
    specs: sequence of dataset spec dictionaries with keys ``hf_id``, ``split``,
        and an optional ``limit`` or ``name``.
    text_field: column to read from, defaults to ``text``.
    """
    import concurrent.futures
    import os
    
    def _load_single_dataset(spec: dict) -> List[Sample]:
        """Load a single dataset spec."""
        local_path = spec.get("path")
        if local_path:
            try:
                return _load_local_samples(local_path, spec.get("name"))
            except Exception as e:
                print(f"Warning: Failed to load local dataset '{spec.get('name', local_path)}': {e}")
                return []
        
        try:
            config_name = spec.get("config")
            trust_remote_code = spec.get("trust_remote_code", False)
            dataset_name = spec.get("name", spec.get("hf_id", "unknown"))
            print(f"Loading dataset: {dataset_name}...")
            
            # Use streaming=False for better caching, but don't use num_proc in threaded context
            dataset = load_dataset(
                spec["hf_id"], 
                config_name, 
                split=spec.get("split", "train"), 
                trust_remote_code=trust_remote_code,
                streaming=False  # Enable caching
            ) if config_name else load_dataset(
                spec["hf_id"], 
                split=spec.get("split", "train"), 
                trust_remote_code=trust_remote_code,
                streaming=False  # Enable caching
            )
            
            limit = spec.get("limit")
            iterable = dataset if limit is None else dataset.select(range(limit))
            
            samples = []
            for example in iterable:
                template = spec.get("template")
                if template:
                    text = _render_template(template, example)
                else:
                    text = example.get(spec.get("field", text_field)) or example.get("content") or ""
                if text:  # Only add non-empty texts
                    samples.append(Sample(text=text, source=dataset_name))
            
            print(f"Loaded {len(samples)} samples from {dataset_name}")
            return samples
        except Exception as e:
            print(f"Warning: Failed to load dataset '{spec.get('name', spec.get('hf_id', 'unknown'))}': {e}")
            print(f"  Skipping this dataset and continuing with others...")
            return []

    # Load datasets in parallel (up to 4 concurrent downloads)
    all_samples: List[Sample] = []
    max_workers = min(4, len(specs), os.cpu_count() or 1)
    
    print(f"Loading {len(specs)} datasets with up to {max_workers} parallel workers...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_spec = {executor.submit(_load_single_dataset, spec): spec for spec in specs}
        for future in concurrent.futures.as_completed(future_to_spec):
            try:
                samples = future.result()
                all_samples.extend(samples)
            except Exception as e:
                spec = future_to_spec[future]
                print(f"Error loading {spec.get('name', spec.get('hf_id', 'unknown'))}: {e}")
    
    print(f"Total samples loaded: {len(all_samples)}")
    return all_samples


def minhash_signatures(samples: Iterable[Sample], num_perm: int = 64) -> List[MinHash]:
    sigs: List[MinHash] = []
    for sample in samples:
        mh = MinHash(num_perm=num_perm)
        tokens = sample.text.lower().split()
        for token in tokens:
            mh.update(token.encode("utf8"))
        sigs.append(mh)
    return sigs


def contamination_report(samples: Sequence[Sample], threshold: float = 0.8) -> List[tuple]:
    """Computes possible overlaps via MinHash LSH."""

    lsh = MinHashLSH(threshold=threshold, num_perm=64)
    sigs = minhash_signatures(samples)
    matches = []
    for idx, sig in enumerate(sigs):
        key = f"sample-{idx}"
        lsh.insert(key, sig)
        for dup in lsh.query(sig):
            if dup == key:
                continue
            matches.append((key, dup))
    return matches
