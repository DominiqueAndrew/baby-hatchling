# Baby-Hatchling

Baby-Hatchling is a CPU-friendly research playground that reproduces the core ideas from the "Project Baby-Hatchling" design brief: a hybrid Transformer with Kimi Delta Attention (KDA) linear blocks, periodic NoPE global attention, predictive-coding auxiliaries, a tiny episodic memory, and a micro-RLVR fine-tuning loop with verifiable rewards. The codebase is intentionally compact so it can run end-to-end on a single desktop CPU with 32 GB RAM.

## Key features
- **Hybrid attention stack** – three KDA layers followed by one NoPE global layer using multi-query KV. KDA keeps per-head fast state constant while global layers preserve quality every few blocks.
- **Predictive coding helpers** – next-token LM head (tied embeddings) and a next-state forecasting head with a stop-gradient teacher. Their errors drive a curiosity signal.
- **Tiny episodic memory** – a SQLite + FAISS store that learns to write surprising states and retrieve useful context when the model is uncertain.
- **Micro-RLVR** – PPO-lite RL with truncated importance weights, KL penalty, and unit-test / numeric rewards for GSM8K-mini and HumanEval/EvalPlus-mini.
- **Evaluation harness** – ARC-Easy, WinoGrande, HellaSwag, GSM8K, HumanEval/EvalPlus, and synthetic long-range probes (palindrome, MQAR, stack).
- **CPU pragmatics** – gradient accumulation, activation checkpoint toggles, INT8 dynamic quantization, deterministic seeds, contamination checks via MinHash.

## Repo layout
```
baby-hatchling/
  README.md
  requirements.txt
  configs/
    hatchling_xs.yaml
    hatchling_s.yaml
  scripts/
    run_pretrain.sh
    run_sft.sh
    run_rlvr.sh
    eval_all.sh
  src/
    __init__.py
    model.py
    trainer.py
    policy_rlvr.py
    attn_kda.py
    global_attn_nope.py
    predictive_head.py
    episodic_mem.py
    eval_harness/
      arc_easy.py
      winogrande.py
      hellaswag.py
      gsm8k.py
      humaneval.py
      probes.py
  tests/
    test_kda.py
    test_nope.py
    test_episodic.py
    test_sandbox.py
    test_rlvr_math.py
```

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Optional: pin PyTorch threads for reproducibility
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Pretrain Hatchling-XS on WikiText-2 + OpenWebText-10k shards
bash scripts/run_pretrain.sh

# Supervised fine-tuning (alapca-like + math/code rationales)
bash scripts/run_sft.sh

# Micro-RLVR on GSM8K-mini + HumanEval/EvalPlus-mini
bash scripts/run_rlvr.sh

# Evaluation suite
bash scripts/eval_all.sh

# Optional Stage-D curiosity gridworld
bash scripts/run_gridworld.sh configs/hatchling_xs.yaml out/xs_sft.pt
# Script prints automatically a curiosity summary; manual inspection:
head logs/gridworld_xs.csv
ls logs/gridworld_transcripts | head

# Ablation sweep example (baseline vs. no memory)
bash scripts/run_ablation_eval.sh configs/hatchling_xs.yaml out/xs_rlvr.pt baseline
cp configs/hatchling_xs.yaml /tmp/no_memory.yaml && yq -Y '.model.use_episodic_memory=false' -i /tmp/no_memory.yaml
bash scripts/run_ablation_eval.sh /tmp/no_memory.yaml out/xs_rlvr.pt no_memory
```

## Ablation toggles
Each config exposes boolean switches under `model` to simplify the ablations described in the design brief:

- `use_predictive_head`: disable the next-state predictor / curiosity teacher while keeping the LM head active.
- `use_episodic_memory`: remove the SQLite/FAISS store along with its gate penalty.
- `use_curiosity_bonus`: zeroes out the intrinsic reward tracker (gridworld summaries still run but report 0 bonuses).

Flip these flags to `false` (and optionally adjust the corresponding loss weights) before re-running `run_sft.sh`, `run_rlvr.sh`, or `run_gridworld.sh` to collect ablation metrics without editing code.

## Quantization
After Stage-C RLVR, export an INT8 dynamic quantized checkpoint:
```bash
python - <<'PY'
import torch, torch.nn as nn
from torch.ao.quantization import quantize_dynamic
from src.model import build_model
from src.utils.config import load_config
cfg = load_config("configs/hatchling_xs.yaml")
model = build_model(cfg["model"])
model.load_state_dict(torch.load("out/xs_rlvr.pt", map_location="cpu"))
model.eval()
quantized = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
torch.save(quantized.state_dict(), "out/xs_rlvr_int8.pt")
PY
```

## Tests
Run the targeted unit tests with:
```bash
pytest -q
```

The tests focus on:
- Numerical sanity for the KDA recurrence
- Shape/causality for global NoPE attention
- Read/write behaviour of the episodic memory store
- Sandbox safety for verifiable unit tests
- PPO clipping and truncated importance math
- Gridworld episode sanity checks (greedy policy reaches the goal)

## References
Key design references are linked inside the README comments and module docstrings. Notably: Kimi Linear (KDA) [arXiv:2510.26692], NoPE attention [arXiv:2404.12224], RLVR (Ring-1T) inspiration, curiosity via predictive coding, and EvalPlus for robust HumanEval tests.
