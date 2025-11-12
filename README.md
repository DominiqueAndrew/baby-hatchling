# Hatchling-NEURO (Baby-Hatchling v2)

Hatchling-NEURO keeps the 3:1 Kimi Delta Attention (KDA) : NoPE-global recipe but adds event-driven spiking gates, predictive-coding auxiliaries, episodic memory, and a verifiable-reward RL stage that all run end-to-end on a single RTX 3090 (24 GB) or equivalent cloud GPU. The repo now ships the full “crawler → pretrain → SFT → micro-RLVR → eval” loop plus configs for the XS (~38 M) and S (~100 M) variants that stay within a 24 GB VRAM envelope via linear attention and grouped KV caches.

## Highlights
- **Spike-KDA stacks** – every three Spike-modulated KDA blocks are followed by a NoPE global attention block (MQA/GQA friendly). LIF spikes modulate the per-channel decay `α_t` and write step `β_t`, enabling compute skipping when heads are silent.
- **Parallel-Scan Spike-KDA** – log-depth upsweep/downsweep fast weight accumulation via `model.kda_mode: "scan"` keeps training fully parallel on GPU while autoregressive decoding automatically falls back to the sequential kernel.
- **Predictive coding + curiosity** – tied-embedding LM head plus a stop-grad next-state head. Errors feed a curiosity bonus and optional episodic writes.
- **Tiny episodic memory** – SQLite + FAISS store with write-on-surprise / read-on-uncertainty gates and a differentiable penalty to keep accesses controlled.
- **Verifiable micro-RLVR** – PPO-lite with truncated importance weights, dynamic KL penalties, and sandboxed unit tests for GSM8K-mini and HumanEval/EvalPlus-mini.
- **Crawler-in-the-loop** – polite English-only crawler (robots-aware, license keywords, Simhash dedup, near-dup filters) that emits JSONL shards consumable by the trainer.
- **Long-context pragmatics** – KDA keeps constant fast weights (`dk × dv` per head) and NoPE global layers use grouped KV, so 4k ctx fits on 24 GB with gradient accumulation.
- **Quick-win efficiency stack** – curriculum schedule (256 → 1024), cosine LR with warmup, early stopping, gradient checkpointing for global layers, and stochastic token dropping (ScTD-inspired) inside mid-stack KDAs. Together they cut wall-clock time ~70 % on a single 3090 without hurting quality.
- **Phase-2 optimizations** – Mixed Sparsity Training (MST) gradually prunes + regrows 70 % of dense linear weights (4× FLOP drop) while AdaPM replaces AdamW with a low-rank momentum optimizer so we can run much larger effective batches without the 24 GB limit getting in the way.

## Repo layout (trimmed)
```
baby-hatchling/
  configs/
    hn_xs.yaml            # default 38 M config
    hn_s.yaml             # 100 M config
    crawler_english.yaml  # polite crawler template
  scripts/
    run_crawler.sh        # ingest new English-only data
    run_pretrain.sh       # Stage-A
    run_sft.sh            # Stage-B
    run_rlvr.sh           # Stage-C verifiable RL
    eval_all.sh           # ARC-E, Wino, HellaSwag, GSM8K, HumanEval, probes
    cloud_*.sh            # remote orchestration on RunPod/AWS/etc.
  src/
    attn_kda.py           # Spike-KDA recurrent/chunkwise kernels
    global_attn_nope.py   # NoPE global attention (MQA/GQA)
    crawler/pipeline.py   # respectful crawler emitting shards
    predictive_head.py    # LM + next-state + curiosity bonus
    episodic_mem.py       # SQLite/FAISS episodic store
    policy_rlvr.py        # PPO-lite w/ truncated IS + dynamic KL
    trainer.py            # unified pretrain/SFT loop with AMP + grad accum
    eval_harness/         # ARC-E, Wino, HellaSwag, GSM8K, HumanEval, probes
  tests/
    test_kda.py           # streaming vs. full pass equivalence
    test_nope.py, test_episodic.py, test_rlvr_math.py, test_data_loader.py …
```

## Parallel-Scan Spike-KDA
- `model.kda_mode` switches between `"sequential"`, `"chunked"`, `"scan"`, and `"auto"` (auto = try scan on GPU w/ `seq_len ≥ model.kda_scan_min_len`, otherwise fall back).
- `model.kda_scan_min_len` (default 64) avoids scan overhead on very short contexts.
- Scan mode parallelizes the Spike-KDA fast-weight recurrence with a Blelloch prefix algorithm (`precompute_updates → scan_emit_outputs`). Outputs stream out in tiles of `model.kda_chunk_size` tokens so we never materialize the entire `[B,T,H,dk,dv]` state (keeps VRAM roughly flat vs. sequence length). Autoregressive decoding still uses the sequential path; set `model.kda_mode: "auto"` if you want CPU batches to fall back automatically.
- Full derivation plus kernel notes live in `docs/parallel_scan_spike_kda.md`.

Quick benchmark recipe (PyTorch 2.2+):

```python
import torch, torch.utils.benchmark as bench
from src.attn_kda import KDABlock

block = KDABlock(512, 8, 64, 64, 2048, 256, kda_mode="scan").cuda()
x = torch.randn(2, 1024, 512, device="cuda", dtype=torch.float16, requires_grad=True)

def run(mode):
    block.kda_mode = mode
    y, _ = block(x)
    (y.sum()).backward()

for mode in ["sequential", "scan"]:
    print(bench.Timer(stmt="run(mode)", globals={"run": run, "mode": mode}).blocked_autorange())
```

Expect the scan path to close most of the gap between sequential KDA and full attention for long sequences while preserving the same numerics (validated in `tests/test_kda.py`).

## Environment
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 0. Crawl & prep English-only shards (optional but recommended)
```bash
# Respect robots.txt, license keywords, per-domain quotas, Simhash dedup
bash scripts/run_crawler.sh configs/crawler_english.yaml
# Output: data/crawler/shards/shard_0000.jsonl, …
```
Any JSONL/TXT glob can be referenced from the training configs via `datasets.*.path`, so your crawler shards are picked up automatically.

## 1. Pretrain → SFT → RLVR on one GPU (sequential scripts)
```bash
# Stage A – Pretrain (LM + predictive coding) on hn_xs config
bash scripts/run_pretrain.sh configs/hn_xs.yaml        # saves out/hn_xs_pretrain.pt

# Stage B – SFT on instructions/math/code rationales
bash scripts/run_sft.sh configs/hn_xs.yaml out/hn_xs_pretrain.pt   # -> out/hn_xs_sft.pt

# Stage C – Micro-RLVR (unit-test math/code rewards, dynamic KL)
bash scripts/run_rlvr.sh configs/hn_xs.yaml out/hn_xs_sft.pt       # -> out/hn_xs_rlvr.pt

# Unified evaluation suite + long-range probes
bash scripts/eval_all.sh out/hn_xs_rlvr.pt

# Optional: curiosity gridworld sanity
bash scripts/run_gridworld.sh configs/hn_xs.yaml out/hn_xs_sft.pt
```

## 2. Config matrix (fits 24 GB VRAM with grad accumulation)
| Config | Layers (Spike-KDA / NoPE) | d_model | Heads | d_k=d_v | Seq | Grad accum | Notes |
| ------ | ------------------------ | ------- | ----- | ------- | --- | ---------- | ----- |
| `hn_xs` | 24 (18 / 6) | 640 | 8 | 64 | 2k | 16 | ~38 M params, 18–20 GB w/ bf16 & accum=16 |
| `hn_s`  | 32 (24 / 8) | 768 | 12 | 64 | 4k | 32 | ~100 M params, ~22–24 GB |

Both configs keep 3:1 Spike-KDA:NoPE, grouped KV for globals, and BF16/FP16 mixed precision with gradient accumulation so a single 3090 sustains the curriculum.

## 3. Verifiable RL knobs
- Truncated importance sampling via `rho_max`
- PPO clipping (`epsilon_clip`)
- **Dynamic KL**: `kl_coeff` auto-tunes toward `rlvr.kl_target`, logged in `logs/rlvr.csv`
- Sandbox tests live in `src/utils/sandbox.py` (resource-capped)

## 4. Quick-win optimizations (enabled by default in `configs/hn_xs.yaml`)
- **Sequence-length curriculum** – training automatically progresses through four stages (256 → 512 → 768 → 1024 tokens). Each stage re-builds the dataloader with an appropriate `batch_tokens` / `grad_accum` combo so the 3090 stays in the safe VRAM envelope while still seeing long contexts in the final 50 k steps.
- **Cosine LR schedule + warmup** – `train.pretrain.lr_schedule` applies linear warmup for 3 k steps then cosine decay to 10 % of the peak LR. Works with any optimizer thanks to the `LambdaLR` hook in `src/trainer.py`.
- **Early stopping** – when the moving average of the total loss doesn’t improve by `min_delta` within `patience` optimizer steps the trainer exits gracefully (after saving the checkpoint). Disable by flipping `train.pretrain.early_stopping.enabled=false`.
- **Gradient checkpointing** – NoPE global layers run under `torch.utils.checkpoint` when `model.use_gradient_checkpointing` is true. This recovers ~1.5× VRAM headroom, letting the curriculum keep larger micro-batches without OOMs.
- **Token dropping (ScTD-lite)** – mid-stack KDABlocks randomly reuse previous outputs for ~25 % of tokens (configurable via `model.token_drop`). The block skips state updates for dropped tokens, lowering FLOPs while keeping the semantics consistent.
- **Mixed Sparsity Training (MST)** – configure via `train.pretrain.sparsity`. During warmup the model stays dense, then MST prunes weights magnitude-wise until the `target` sparsity; during restoration, masks regrow so final layers re-densify before convergence. Masks refresh every `update_every` steps and apply to every large linear layer, giving ≈4× FLOP savings in practice.
- **AdaPM optimizer** – select with `optim.type: adapm`. AdaPM stores momentum per output channel instead of per-weight, so optimizer state shrinks by >90 % while retaining the benefits of momentum. Combined with gradient checkpointing we can safely push larger effective batches, improving convergence speed.

Together these give ~70–80 % wall-clock savings on a single RTX 3090: ≈45–55 hours to hit 60 k steps vs. ~150+ hours previously for 120 k steps at constant 1 k ctx. Extend the final curriculum stage or bump `seq_len` when you need the full 2 k context; just watch VRAM when editing `configs/hn_xs.yaml`.

## 5. Quantization / export
```bash
python - <<'PY'
import torch, torch.nn as nn
from torch.ao.quantization import quantize_dynamic
from src.model import build_model
from src.utils.config import load_config
cfg = load_config("configs/hn_xs.yaml")
model = build_model(cfg["model"])
model.load_state_dict(torch.load("out/hn_xs_rlvr.pt", map_location="cpu"))
model.eval()
quantized = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
torch.save(quantized.state_dict(), "out/hn_xs_rlvr_int8.pt")
PY
```

## 6. Cloud workflow (RunPod/AWS/GCP/etc.)
```bash
# One-time setup
bash scripts/cloud_setup.sh

# Training on remote box (defaults to configs/hn_xs.yaml)
bash scripts/cloud_train.sh pretrain configs/hn_xs.yaml out/hn_xs_pretrain.pt
bash scripts/cloud_train.sh sft      configs/hn_xs.yaml out/hn_xs_sft.pt out/hn_xs_pretrain.pt
bash scripts/cloud_train.sh rlvr     configs/hn_xs.yaml out/hn_xs_rlvr.pt out/hn_xs_sft.pt

# Monitor / sync
bash scripts/cloud_monitor.sh watch
bash scripts/cloud_sync.sh     # or cloud_autosync.sh for continuous rsync
```
RunPod helper scripts (`runpod_setup.sh`, `runpod_web_terminal_*`) print the exact commands wired to the new configs/paths.

## 7. Tests
```bash
pytest -q
```
Coverage:
- Spike-KDA streaming vs. full pass equivalence (`tests/test_kda.py`)
- KDA stability / gradient flow over long sequences (`tests/test_kda_stability.py`)
- Right-shifted label builder (`tests/test_labels.py`)
- Predictive head + curiosity normalization (`tests/test_pc_head.py`)
- Episodic memory k-NN gating and penalties (`tests/test_episode_knn.py`)
- NoPE causality, episodic memory read/write gates, sandbox safety
- PPO objective clipping + truncated IS
- Local-shard loader (`tests/test_data_loader.py`) to ensure crawler output is ingestible

## 8. Tips
- **Configs:** Copy `configs/hn_xs.yaml` or `configs/hn_s.yaml`, tweak batch tokens / grad accum if you switch hardware.
- **Crawler:** Adjust `configs/crawler_english.yaml` allow/deny lists to stay license-safe. Shards auto-plug into the training mix via the `datasets.pretrain[].path` entry.
- **Ablations:** Flip `model.use_predictive_head`, `model.use_episodic_memory`, or `model.use_curiosity_bonus` in YAML before re-running scripts. `scripts/run_ablation_eval.sh` logs each sweep.
- **Long context:** Increase `model.max_seq` and `train.*.seq_len` together; linear-state KDA keeps memory flat, but watch VRAM when raising `batch_tokens`.

## 9. Toddler sanity check (20 min)
Use the tiny `hn_toddler` config to sanity-check new data or training changes with episodic memory still enabled:

```bash
# Stage A – pretrain
bash scripts/run_toddler.sh configs/hn_toddler.yaml

# Stage B – SFT (loads Stage A checkpoint unless you override path)
bash scripts/run_toddler_sft.sh configs/hn_toddler_sft.yaml

# Stage C – micro-RLVR (loads Stage B checkpoint)
bash scripts/run_toddler_rlvr.sh configs/hn_toddler_rlvr.yaml
```

- Stage A trains an 8-layer, 256d model on a streaming Wikitext-2 shard (`seq_len=256`, `batch_tokens≈2k`, fixed 8 workers/prefetch 4) for 600 steps with the SQLite episodic store active (4 MiB cap).
- Stage B runs ~600 supervised steps on ~11 k instruction traces (1 k Alpaca + 10 k sampled OpenHermes) with the same Toddler backbone.
- Stage C runs ~200 micro-RLVR updates on GSM8K-mini + HumanEval/EvalPlus-mini with truncated importance sampling, ratio clipping, and a tiny PTX mix.
- Pretrain/SFT toddler configs now auto-expand `batch_tokens` and dataloader workers whenever the GPU is <70 % utilized, so expect the first few steps to print `[auto-batch]` messages as they ramp to fill VRAM.

### Quick smoke tests
- **Overfit-one-batch (pretrain sanity):**
  ```bash
  python -m src.trainer \
    --config configs/hn_toddler.yaml \
    --stage pretrain \
    --overfit_one_batch \
    --save out/hn_toddler_overfit.pt
  ```
  Expect the loss to drop below 1.0 on the frozen batch within ~50 updates; if it does not, labels/gradients are miswired.
- **Smoke RLVR:** after Stage B, run
  ```bash
  python -m src.policy_rlvr \
    --config configs/hn_toddler_rlvr.yaml \
    --load out/hn_toddler_sft.pt \
    --save out/hn_toddler_rlvr_smoke.pt
  ```
  The CSV (`logs/rlvr.csv`) should show positive math/code reward trends and an adaptive KL coefficient hovering near 0.02.

All three stages finish within roughly an hour on a 24 GB GPU, giving you an end-to-end but tiny checkpoint (`out/hn_toddler_rlvr.pt`) you can chat with or evaluate before scaling configs back up. Tune the dataset `limit` fields if you want even faster iterations.

Everything needed to run the sequential training stages on your cloud GPU (crawler → pretrain → SFT → RLVR → eval) now ships in-tree with spiking gates, dynamic RL controls, curriculum-aware training, and scripts wired to the new Hatchling-NEURO configs.
