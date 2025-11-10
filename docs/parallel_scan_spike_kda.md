# Parallel-Scan Spike-KDA

This note captures the math derivation, operator factoring, and implementation details for the Spike-modulated Kimi Delta Attention (KDA) parallel-scan path that now powers the `model.kda_mode: "scan"` option.

## 1. Recurrence with spikes, drop masks, and fast weights

For each head we keep a fast-weight matrix \(S_t \in \mathbb{R}^{d_k \times d_v}\) updated per token:

\[
\begin{aligned}
S_t &= \operatorname{diag}(\alpha_t) S_{t-1} - \beta_t \, k_t ( (\alpha_t \odot k_t)^\top S_{t-1} ) + \beta_t \, k_t v_t^\top, \\
o_t &= S_t^\top q_t
\end{aligned}
\]

where:

- \(q_t, k_t, v_t \in \mathbb{R}^{d_k}\) / \(\mathbb{R}^{d_v}\) are the per-head projections.
- \(\alpha_t = \sigma(\alpha^{\text{base}}_t + W_\alpha \cdot \text{spike}_t)\) is the spike-modulated per-channel decay.
- \(\beta_t = \sigma(\beta^{\text{base}}_t + w_\beta^\top \text{spike}_t)\) is the spike-modulated write/forget gain (scalar per head).
- `drop_mask[b,t]=1` turns the update into identity: \(\alpha_t = 1,\ \beta_t = 0,\ o_t = o_{t-1}\).

This recurrence assumes diagonal decays, rank-1 forget/write (via the \(k_t\) outer products), and elementwise spike modulation.

## 2. Associative operator factorization

The recurrence can be written as a linear operator plus bias:

\[
S_t = A_t S_{t-1} + B_t,
\]

with

\[
A_t = \operatorname{diag}(\alpha_t) - \beta_t \, k_t (\alpha_t \odot k_t)^\top,\qquad
B_t = \beta_t \, k_t v_t^\top.
\]

Each token contributes \(F_t = (A_t, B_t)\) and the fast-weight state after \(t\) tokens is:

\[
S_t = \left( \bigoplus_{i=1}^t F_i \right) (S_0), \qquad
(A_2, B_2) \oplus (A_1,B_1) = (A_2 A_1,\ A_2 B_1 + B_2).
\]

Because \(\oplus\) is associative we can evaluate all prefixes with a Blelloch scan. We never materialize dense \(A_t\) matrices; instead we implicitly build them from \(\alpha_t, \beta_t, k_t\) (diagonal + rank‑1 structure) only when combining subtrees.

## 3. Implementation layout

The GPU path is broken into three clearly separated passes:

1. **`precompute_updates`**  
   - Inputs: \(k, v, \alpha, \beta, \text{active}, \text{drop\_mask}\).  
   - Outputs: `transitions = diag_embed(alpha) - β k (α⊙k)^T` and `writes = β k v^T`.  
   - Head-silent (`active=0`) or dropped tokens force identity transitions and zero writes.

2. **`parallel_scan`**  
   - Runs a Blelloch upsweep/downsweep over `(transitions, writes)` to produce exclusive prefix operators for every token in \(O(\log L)\) depth.  
   - Applies each prefix operator to the initial state \(S_0\) and then applies the token’s own `(A_t, B_t)` to obtain all `S_t`.  
   - Implemented with batched `torch.matmul`, padded to the next power-of-two, and works on CUDA/MPS/XPU. CPU automatically falls back to the sequential core.

3. **`scan_emit_outputs`**  
   - Performs the downsweep in tiles of `chunk_size` tokens: `(prefix_m, prefix_b)` are sliced, converted into local states, immediately contracted with `q`, then released.  
   - Keeps peak memory close to `chunk_size × H × d_k × d_v` regardless of the full sequence length while still emitting all tokens in parallel from the caller’s perspective.  
   - Token-drop forward-fills outputs using a cummax-based gather so dropped tokens reuse the last emitted vector.

The scan path sits behind `KDABlock._forward_parallel_scan` and is selected when `model.kda_mode in {"scan","auto"}` *and* the device is CUDA/XPU/MPS with `seq_len ≥ model.kda_scan_min_len`. Autoregressive streaming automatically reuses the sequential kernel.

## 4. Assumptions & mapping to the paper

- Per-head LIF spikes are still evaluated sequentially (membrane dynamics are inherently recurrent) before we enter the scan.  
- Decay matrices remain diagonal and the forget/write rules are rank-1, exactly matching the Spike-KDA derivation in the paper.  
- Token dropping (ScTD-lite) zeroes `β_t` and sets `A_t = I`, while the emitted output is forward-filled to preserve semantics.  
- Mixed Sparsity Training (MST) masks and checkpointing work unchanged because the scan path is pure PyTorch and differentiable.

## 5. Backward compatibility & fallbacks

- **CPU / very short sequences:** `kda_mode: "auto"` or `"sequential"` falls back to the original kernel, while `"scan"` can be forced for debugging/tests (expect slower runtimes on pure CPU).  
- **Autoregressive decoding:** `KDABlock.stream()` keeps using the sequential core so incremental decoding stays O(L).  
- **Custom autograd:** the scan is written with standard tensor ops, so PyTorch differentiates it end-to-end; wrapping it in a `torch.autograd.Function` for checkpointed reverse scans is straightforward if we need lower memory later.

## 6. Testing & benchmarking hooks

- `tests/test_kda.py` now compares sequential vs. scan outputs/states on random tensors.  
- The README outlines a minimal `torch.utils.benchmark` harness that times `KDABlock` in `scan` vs. `sequential` mode to validate the expected <log(L)> depth speedup and VRAM reduction.

## 7. Notes on custom kernels

The current reference path is implemented in idiomatic PyTorch 2.2+ so it works everywhere and can be wrapped with `torch.compile`/`nvFuser` immediately. For teams that need a Triton/CUDA specialization:

1. Port `precompute_updates` into a fused kernel that writes `(decay_diag, forget_vec, write_mat)` to shared memory.  
2. Implement the Blelloch upsweep/downsweep as a hierarchy of block scans (sequence → chunk → block).  
3. Reuse the same math for backward by running a reverse scan over saved `(A_t, B_t, q_t, drop_mask)`.  
4. Bind the kernel via a `torch.autograd.Function` exactly matching the reference API so the high-level code path does not change.

Until that kernel lands, `model.kda_mode: "scan"` already delivers full-parallel emissions on GPU with the math guaranteed to match the sequential Spike-KDA recurrence.
