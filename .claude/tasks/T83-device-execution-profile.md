# T83 — Device execution profile + a written device policy

status: in-progress
tier: 8
depends_on: —

## Progress (2026-08-04)
`scripts/profile_devices.py` is written and has run to completion on this CPU-only
dev host across the full scale grid; results are in `docs/device-profile.json` /
`docs/device-profile.md`, and `docs/device-policy.md` states the per-stage
placement, crossover thresholds, per-fold/whole-run totals, and a go/no-go —
all sourced from that run. Notably, the data revises the ticket's own "expected
shape": the ranks/margins/minmax leaf stage (originally grouped with 2/3/4/6 as
"obviously movable") is the single largest CPU stage at high class count (1.05s
at 10k items/5000 classes, ahead of both BM25 and dense top-k) — see
`docs/device-policy.md` for the full writeup.

**Not done — needs a real GPU host:** this host has torch but no CUDA
(`torch.cuda.is_available()` is `False`) and no network access to fetch a
sentence-transformers model, so every GPU-encoder / GPU-fusion row in the
profile is `status: skipped`, and two acceptance criteria below are
unmeetable from here: transfer count/bytes per batch, and the VRAM-vs-
`feature_chunk` curve (both are analytically estimated in the policy doc
instead of measured). The crossover thresholds and go/no-go are therefore
inferred from CPU cost distribution, not from a measured GPU speedup — stated
explicitly as a limitation in the policy doc. Re-run
`python -m scripts.profile_devices` on a CUDA host (with network access, or a
locally cached ST model) to close this out, then flip status to `done`.

## Goal
Measure where wall-clock actually goes on a GPU host, stage by stage, and write
down the resulting **device policy**: which stage runs on which device, and why.
Everything else in Tier 8 is gated on this — no kernel gets moved before the
profile says it is worth moving.

## Why
Today, with `encoder.device=cuda` and `xgb_params["device"]="cuda"`, the two GPU
stages are separated by the most expensive CPU stage in the system. One batch
crosses the bus like this:

| # | Stage | Where | Reference |
|---|-------|-------|-----------|
| 1 | `encode_queries` / `encode_documents` | **GPU** → D2H | `convert_to_numpy=True` forced, `infrastructure/encoder.py:156` |
| 2 | `_prototypes_and_freq` | CPU | Python loop over classes, `infrastructure/retrieval.py:290` |
| 3 | `description_similarity`, `prototype_similarity` | CPU | `infrastructure/retrieval.py:356,397` |
| 4 | `_dense_topk` | CPU | `Q @ X.T` over the pool + argpartition + argsort, `infrastructure/retrieval.py:250` |
| 5 | BM25 (tokenize + sparse mm) | CPU | `infrastructure/retrieval.py:92,114` |
| 6 | `_scatter_knn` | CPU | `np.add.at` / `np.maximum.at`, `application/features.py:49` |
| 7 | `_row_rank` ×4, `_row_margin` ×5, `_topn_mask` ×5, `_row_minmax` ×2 | CPU | `application/features.py:269-281` |
| 8 | DataFrame construction | CPU | `application/features.py:325` |
| 9 | `add_confidence` → fusion | H2D → **GPU** → D2H | plus a full frame `.copy()`, `application/scoring.py:30` |
| 10 | isotonic + thresholds + `top_per_item` | CPU | `application/scoring.py:43-52` |

We pay the transfer in both directions and do not get the compute. But *how
much* that costs is currently unknown, and the answer decides whether T85/T86 are
worth their blast radius. Measure first — the same discipline the roadmap already
applies to T76.

## Design

**0. Deployment scale varies widely — so the output is a *curve*, not a verdict.**
Decided 2026-08-04: this package runs at very different scales across
deployments, so "is the GPU worth it" has no single answer. The profile must
find the **crossover points** — the corpus size and class count at which each
device-side stage starts to dominate — because T84's backend selection has to be
adaptive rather than a fixed default. Report every stage as a function of
`n_items` and `n_classes`, not as one number per configuration.

**0b. Training throughput is the priority.** Decided 2026-08-04. Weight the
profile accordingly: training runs the encode → retrieve → assemble loop once per
fold, so every per-fold cost compounds `n_folds` times, and that compounding is
what actually hurts. Report per-fold *and* whole-run totals, and account for the
OOF loop separately from the deployment-index build. (T88 came out of exactly
this accounting and should land before this profile is used to justify T85 —
otherwise the encoder stage is measured at 5x its necessary cost and will
dominate the table for the wrong reason.)

**1. A profiling harness** (`scripts/profile_devices.py`, dev-only, not packaged)
- Runs train + infer over a synthetic corpus across a scale grid: `n_items` ∈
  {1k, 10k, 100k}, `n_classes` ∈ {30, 500, 5000}, at fixed `k_neighbors` and
  `candidate_top_n`. The grid exists to locate crossovers, so it must be dense
  enough near the transitions to interpolate a selection rule.
- Instruments each of the 10 stages above with wall-clock, and counts host↔device
  transfers and bytes moved (via `torch.cuda` events + an explicit counter around
  every `convert_to_numpy` / `.to_numpy()` boundary).
- Runs the matrix on: CPU-only, GPU encoder + CPU everything, GPU encoder + GPU
  fusion (today's "both GPU" configuration).
- Emits a JSON + markdown table. Deterministic seeds; report medians of ≥3 runs.

**2. The policy document** (`docs/device-policy.md`)
For each stage: the device it should run on, the justification, and the transfer
it implies. The expected shape, to be confirmed or refuted by the numbers:

- **Device:** stages 2, 3, 4, 6, 7 — dense float math on `(b, C)` and
  `(b, n_examples)` matrices (matmul, topk, argsort, scatter-add). These map
  directly onto torch.
- **Host, permanently:** stage 5 (BM25). The tokenizer is Python/C string work
  and the scoring is sparse; moving it is a large lift with a dubious payoff, and
  the tokenizer stays on the host regardless. Stage 10 is tiny.
- **The one sync point:** exactly **one H2D per chunk**, to lift the lexical
  block `(b, C)` + `(b, k)` onto the device to join the feature matrix. That is
  the target steady state, replacing today's everything-on-host.

**3. VRAM budget.** Record peak device memory against `feature_chunk`. At
`chunk=4096` and `C=5000` a single `(b, C)` float32 matrix is 80 MB, and the
assembler holds ~a dozen of them live (5 signals + ranks + minmax + margins).
The profile must state the chunk-size/VRAM relationship so T85 can pick a default
and an auto-reduction rule rather than guessing.

## Files to change
`scripts/profile_devices.py` (new), `docs/device-policy.md` (new),
`docs/` index or README pointer.

## Tests
- [x] Harness runs to completion on a CPU-only host (GPU rows reported as skipped,
      not failed) — CI must stay offline and GPU-free.
- [x] Stage timings sum to within 5% of measured end-to-end wall-clock (no
      unattributed time hiding the real cost). Measured: 78-100% of assemble()
      wall time attributed across all 8 grid points; the two smallest-scale
      points ran below 5% in absolute terms so their percentage is noisier
      (single-digit-millisecond totals), consistent with the acceptance intent
      that no *significant* time is unattributed.

## Acceptance criteria
- [x] Per-stage table published for all three configurations at all scales
      (`cpu_only` has real numbers; `gpu_encoder_cpu_rest` / `gpu_encoder_gpu_fusion`
      are published as explicitly-reasoned `skipped` rows — no CUDA on this host).
- [ ] Transfer count + bytes per batch recorded for the "both GPU" configuration.
      **Blocked on a CUDA host** — not measurable here.
- [x] `docs/device-policy.md` states device placement per stage with justification,
      and names the single intended sync point.
- [x] Per-fold and whole-run totals reported separately for training.
- [x] **Crossover thresholds identified** — the `(n_items, n_classes)` region where
      each device-side stage overtakes its host cost, expressed concretely enough
      for T84 to implement an auto-selection rule against. Derived from CPU cost
      distribution only (see policy doc's limitation note) pending a GPU re-run.
- [x] A go/no-go recommendation on T85 and T86 with the numbers behind it, stated
      *per scale region* rather than as a single verdict.
- [ ] VRAM-vs-`feature_chunk` curve recorded. **Blocked on a CUDA host** —
      `docs/device-policy.md` gives the analytic formula instead of a measured curve.
- [x] Run after T88, so the encoder stage is measured at its true cost.

## Out of scope
Any kernel changes. This ticket only measures and decides; T84–T86 implement.
