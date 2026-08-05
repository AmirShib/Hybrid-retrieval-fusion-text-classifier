# Device policy (T83, T85)

> **T85 status (2026-08-05).** The device-resident path described below is
> implemented: `array_backend="torch"` runs every dense-side kernel on the
> configured device, `dense_kind="torch"` keeps the example embeddings,
> prototypes and description matrix resident there, and a device encoder hands
> its tensors straight to retrieval. What is **not** done is the GPU
> measurement: this host still has no CUDA device, so the speedup numbers the
> T85 acceptance criterion asks for are still owed, and the crossover
> thresholds below are still inferred from CPU cost distribution rather than
> measured device speedups. `scripts/profile_devices.py --array-backend torch`
> is the harness for that re-run. See "Determinism across backends" below for
> what changes about reproducibility once a device is in play.

Written from `scripts/profile_devices.py`'s measured run, committed alongside this
doc as `docs/device-profile.json` / `docs/device-profile.md`. Run after T88 (the
corpus is encoded once per training run, not once per fold), so the encoder stage
below is measured at its true cost, not 5x it.

**Host limitation, stated up front.** This profile was produced on a CPU-only
host: `torch.cuda.is_available()` is `False` here, and there is no network access
to download a real `sentence-transformers` model offline. Every `gpu_encoder_*`
and `gpu_encoder_gpu_fusion` row in the profile is `status: skipped` for that
reason — not because the harness failed, but because there is nothing to measure.
Everything below about GPU stages is therefore **inferred** from which CPU stages
are expensive and embarrassingly vectorizable (the T83 ticket's own reasoning),
not measured directly. Re-running `scripts/profile_devices.py` on an actual CUDA
host — with network access, or a locally cached `sentence-transformers` model —
is required before treating the go/no-go below as final. That re-run should also
fill the two gaps this document cannot close from a CPU-only host: transfer
count/bytes per batch, and the VRAM-vs-`feature_chunk` curve (both estimated
analytically below instead).

## Per-stage device placement

| # | Stage | Placement | Why |
|---|-------|-----------|-----|
| 1 | `encode_queries` / `encode_documents` | GPU (when a GPU encoder is configured) | Transformer forward pass; the canonical GPU workload. Measured cost here is small in every scale tested (≤0.4s even at 100k items) because T88 already removed the 5x per-fold repetition — this stage is *not* where the "both GPU" configuration's payoff comes from. |
| 2 | `_prototypes_and_freq` | GPU | Dense `(n_examples, dim)` reduction to `(C, dim)` — grows with class count; 0.36s at C=5000/100k items, from ~0 at C=30. Matmul-shaped, maps directly onto torch. |
| 3 | `description_similarity`, `prototype_similarity` | GPU | `(b, dim) @ (dim, C)` matmuls. Cheapest measured stage at every scale (≤0.05s) — moving it alone would not pay for a sync point, but it is free to carry along once 2/4/6/7 justify one. |
| 4 | `_dense_topk` | GPU | `(b, dim) @ (dim, n_examples)` + partition/argsort. One of the two largest stages at scale: 3.3–3.7s at 100k items, at *any* class count tested (30, 500, or 5000) — this stage scales with corpus size, not class count. |
| 5 | BM25 (tokenize + sparse mat-mul) | **Host, permanently** | Confirmed the largest single stage at low-to-mid class count (7.1s at 100k items/30 classes) and stays the largest or second-largest everywhere. Sparse + Python/C tokenizer work; T32 (BM25 memory + throughput) is the lever here, not a device move. **This is the hard floor T83's own design section predicted**: once 2/4/6/7 move to GPU, BM25 is what a batch waits for. |
| 6 | `_scatter_knn` | GPU | `np.add.at`/`np.maximum.at` scatter over `(b, C)`. Small in absolute terms (≤0.38s) but scatter-add is exactly the op class that GPUs accelerate hardest relative to CPU (numpy's `.at` methods are not vectorized internally); expect the *relative* win here to be larger than the absolute CPU number suggests. |
| 7 | `_row_rank` ×4, `_row_margin` ×5, `_topn_mask` ×5, `_row_minmax` ×2 | GPU | **Revises the ticket's expected shape.** These were grouped with 2/3/4/6 as "dense float math, maps onto torch" but not flagged as a likely bottleneck. At C=5000 they are: 1.05s at 10k items (the single *largest* stage at that point, ahead of both BM25 and dense top-k) and 2.36s at 100k items. Cost scales with class count, not corpus size (near-zero at C=30, dominant at C=5000) — eleven full `(b, C)` sorts/partitions per chunk is the reason. Any deployment with a large label space should weight this stage's migration at least as high as dense top-k. |
| 8 | DataFrame construction | Host | Cheap everywhere (≤0.06s). Pandas object construction; not worth a device round-trip on its own. Relevant to T86 (drop pandas from the hot path), not T85. |
| 9 | `add_confidence` → fusion predict | Host (today), maybe GPU (unmeasured) | `predict_proba` on the assembled matrix stayed under 0.03s at every scale tested — inference-time fusion is not the bottleneck this profile finds. **Caveat:** this harness times `predict_proba` only, not `fit` (fit happens once per fold on a fixed small `n_estimators=50` stand-in and was not separately timed); a production `n_estimators=600` fit is the more plausible GPU-relevant cost on the fusion side and needs its own measurement on a GPU host. |
| 10 | isotonic + thresholds + `top_per_item` | Host | Trivial everywhere (≤0.02s). No device case. |

**The one intended sync point,** per the ticket's design, is confirmed rather
than revised by this data: exactly one H2D transfer per chunk to lift the
lexical block `(b, C)` + `(b, k)` from BM25 (host-permanent) onto the device to
join the dense-computed feature matrix, before stage 7's leaf ops run. Stage 7's
newly-surfaced cost at high C is, if anything, a stronger argument for keeping
everything from stage 2 through stage 7 device-resident once the sync happens,
rather than staging multiple round-trips.

## Crossover thresholds (CPU-only data — see limitation above)

Read as: below this region, GPU residency is very unlikely to pay for its own
transfer/kernel-launch overhead; above it, the CPU cost is large enough that
even a partially-efficient GPU port should win.

- **n_items ≲ 1,000, any class count tested:** total per-chunk cost is 3–9ms.
  No stage individually exceeds ~4ms. A GPU round trip's latency floor (kernel
  launch + PCIe transfer, typically hundreds of µs to low ms) is not small
  relative to this. **No-go** at this scale regardless of class count — T85
  should have a CPU-resident fast path for small deployments, not force every
  call through a device.
- **n_items ~10,000, n_classes ≤ 500:** combined dense-side stages (1+2+3+4+6+7)
  sum to roughly half of total per-chunk cost (~0.06–0.08s of ~0.15–0.16s);
  BM25 alone is 30–45% of the total. **Marginal** — a GPU port helps but BM25's
  host-bound cost caps the win; T32 matters as much as T85 in this band.
- **n_items ≳ 100,000 or n_classes ≳ 500:** dense-side stages sum to several
  seconds (e.g. 100k items/5000 classes: stages 2+3+4+6+7 = 6.8s of 11.5s
  total, versus BM25's 4.2s). **Go** — this is the region T85/T86 should target
  first; it is also where per-fold cost compounds hardest under `n_folds`
  training (see below), so the training-throughput priority and the
  large-deployment priority point at the same region.
- **High class count specifically (C ≳ 5,000), independent of corpus size:**
  stage 7 (leaf ops) becomes competitive with or larger than BM25/dense top-k
  (1.05s at 10k items — the largest single stage there). A deployment with a
  large label space and a modest corpus is a **go** case even where corpus size
  alone would suggest "marginal."

## Per-fold vs. whole-run training totals

(`n_folds=3`, `TrainingPipeline.run` with the offline `HashingEncoder`, T88 already
landed so the pool is encoded once per run and sliced per fold — this table is
what T85 should discount against, not a naive `assemble()`-only estimate.)

| n_items | n_classes | whole run (s) | per fold (s) |
|---|---|---|---|
| 1,000 | 30 | 0.07 | 0.02 |
| 10,000 | 30 | 0.95 | 0.32 |
| 100,000 | 30 | 71.9 | 24.0 |
| 1,000 | 500 | 0.49 | 0.16 |
| 10,000 | 500 | 1.29 | 0.43 |
| 100,000 | 500 | 55.7 | 18.6 |
| 10,000 | 5,000 | 29.6 | 9.9 |
| 100,000 | 5,000 | 94.7 | 31.6 |

Per-fold cost compounds `n_folds` times in the OOF loop (Tier 8's stated
training-throughput priority): at 100k items the whole-run cost is dominated by
exactly the stages flagged "Go" above, run once per fold.

## VRAM budget (analytic estimate — not measured; no CUDA on this host)

Using the ticket's own worked example: a single `(chunk, C)` float32 matrix at
`chunk=4096, C=5000` is `4096 * 5000 * 4 bytes ≈ 80 MB`. The assembler holds
roughly a dozen such matrices live per chunk (5 signals + 4 ranks + 2 minmax +
5 margins + masks, some overlapping in lifetime) — call it 8–12 live at once,
i.e. **~0.6–1.0 GB peak** at this `(chunk, C)` shape. Peak scales linearly in
both `chunk` and `C`:

```
peak_bytes ≈ n_live_matrices * chunk * C * 4
```

For a fixed VRAM budget, the auto-reduction rule T84 needs is therefore
`chunk ≤ budget_bytes / (n_live_matrices * C * 4)` — i.e. `chunk` must shrink
roughly linearly as `C` grows. This is a formula, not a curve: producing the
actual curve (and validating `n_live_matrices`) requires `torch.cuda.max_memory_allocated()`
instrumentation around a real device-resident run, which needs the GPU host
this document already flags as required follow-up.

## Go/no-go on T85 and T86

- **T85 (device-resident dense retrieval):** Go, scoped to the "n_items ≳
  100,000 or n_classes ≳ 500" region identified above; defer or make optional
  for smaller deployments where the sync-point overhead is not clearly repaid.
  The single sync point design (one H2D per chunk, joining the BM25 block) is
  confirmed as the right shape by this data, not revised.
- **T86 (zero-copy feature matrix into fusion, drop pandas from the hot path):**
  Stage 8 (DataFrame construction) is cheap at every scale measured (≤0.06s) —
  this profile does not find urgency for T86 on its own. Its value is more about
  removing a stage from the H2D/D2H round trip T85 introduces than about its own
  standalone cost; sequence it after T85 lands, as the roadmap already orders it.
- **Both are gated on a GPU-host re-run** of this harness (and, ideally, a
  `predict_proba`-and-`fit` timing pass on the fusion side, and the VRAM curve)
  before either lands — the crossover thresholds above are inferred from CPU
  cost distribution, not measured device speedups.

## Determinism across backends (T85)

The decision, taken 2026-08-04 and implemented here: **CPU is the reference
implementation and the benchmark/CI baseline; a device backend is an
accelerator whose results agree within float tolerance, not bit for bit.**
Concretely, on a run with `array_backend="torch"`:

- **Continuous features move in the last ulps.** float32 reductions (the kNN
  matmul, the prototype and description similarity products) are summed in a
  different order, so `d_desc_sim` and friends differ by ~1e-7 on the values
  we measured. Nothing downstream cares about that magnitude *directly*.
- **Ordinal features can flip on a near-tie, and the candidate set with them.**
  This is the part that matters: `rank_*`, `is_*_top1` and the top-n candidate
  mask are comparisons, so two candidates within ~1e-7 can swap, and a
  candidate sitting exactly at the top-n cut can move in or out. The frame is
  then not merely a different number — it is a different *row*. Feature values
  of the rows that stay are unchanged within tolerance.
- **`rank_*` on a candidate a signal did not retrieve is arbitrary on either
  backend.** All such candidates tie at `-inf`, and the numpy path breaks that
  tie with an unstable sort. It was never a reproducible value; it just never
  had a second implementation to disagree with before.
- **Same host + same device + same seed is reproducible.** Cross-device is
  not — do not compare a GPU run's metrics against a CPU run's at more than
  tolerance. One caveat inside that promise: the two scatter kernels use
  `index_put_(accumulate=True)` / `scatter_reduce_`, whose CUDA implementations
  accumulate with atomics and so do not fix a summation order. For *bitwise*
  repeatability on CUDA, set `torch.use_deterministic_algorithms(True)`, which
  selects torch's deterministic implementations of exactly those ops.
- **Every run records what produced it.** `evaluation.json`'s manifest carries
  an `execution` block (`array_backend`, `device`, `dense_kind`), so a metric
  can always be traced back to the arithmetic that produced it.

This qualifies T26's determinism invariant rather than repealing it: identical
runs on identical hardware remain identical, and the numpy backend — the
default, and what the quality benchmark and golden fixtures use — is unchanged
bit for bit by T85.

`tests/integration/test_device_parity.py` is where these claims are checked:
continuous columns with `allclose`, ordinal columns exactly *except* on rows
that are unretrieved or within tolerance of a tie, with the exemption itself
verified against the frame's own values rather than assumed.

## What T85 actually moved

| Stage | Before | After |
|---|---|---|
| Encoder output | forced to numpy on the host | stays a device tensor under the torch backend (`convert_to_tensor`), handed straight to retrieval |
| `example_emb` / `prototypes` / `description_emb` | numpy, rebuilt per fold | uploaded once per run, sliced per fold on the device |
| `_dense_topk` | `argpartition` + gather + `argsort` | one `ArrayOps.topk` |
| `_scatter_knn` | scattered on the backend, then **three `to_host` calls per signal per chunk** | stays resident |
| `_prototypes_and_freq` | scatter on the backend, `to_host`, finish on the host | stays resident |
| `n_signal_agreement` | per-row Python `set` loop on host arrays | vectorized sort/count on the backend |
| Frame construction | one `np.asarray` per column | one `to_host` of the stacked block + one of the candidate grid |

Per chunk, that leaves exactly one host→device crossing — the BM25 block: the
`(b, k)` neighbour labels, the `(b, k)` neighbour scores and the `(b, C)`
description scores, BM25 being permanently host-side per the table above — and
two device→host crossings, both at the fusion handoff at the end of the chunk.
A host-side encoder (TF-IDF/hashing) adds one more upload per chunk for the
query block; a device encoder adds none. This is asserted, not asserted-ish:
see `test_transfers_per_chunk_are_the_bm25_block_and_the_handoff`.

## Measured effect (CPU-only host — the GPU number is still owed)

Same harness, same shapes, `--array-backend numpy` vs `--array-backend torch`
on this CPU-only host. **This is not the measurement the T85 acceptance
criterion asks for** — it compares numpy's kernels with torch's *CPU* kernels,
with no device residency involved — but it does show that the seam itself does
not cost anything, and where the work is concentrated:

| Stage (10k items, 5000 classes) | numpy | torch-CPU |
|---|---|---|
| `_dense_topk` | 5.75s | 0.32s |
| `_scatter_knn` | 7.42s | 2.37s |
| leaf ops (ranks/margins/minmax/topn) | 12.72s | 6.80s |
| BM25 (host-side in both) | 2.21s | 1.46s |
| **`assemble()` wall** | **32.5s** | **14.7s** |

The BM25 row is the caution: that stage is byte-identical host work in both
runs, so its 1.5x "improvement" is measurement noise (thread contention on a
shared host), and the same noise is inside every other row. Read the table as
"the same order of magnitude or better, concentrated in the stages T83
predicted", not as a precise speedup.

Cross-process noise is why the headline number below comes from an
**interleaved, same-process A/B** instead (4 alternating runs per backend,
10,858 items / 2,000 classes / 1,000 queries, `assemble()` wall):

```
numpy  1.68  1.30  1.19  1.24   median 1.27s
torch  1.01  0.57  0.51  0.48   median 0.54s   -> 2.34x
```

**What this does and does not license.** It licenses "the T85 seam does not
tax the pipeline, and torch's kernels are at least competitive on the stages
T83 flagged." It does **not** license any claim about the GPU configuration
the ticket is actually about — device residency, one H2D per chunk, and the
encoder→retrieval handoff are *implemented and asserted structurally* (transfer
counts, parity), but their speedup is unmeasured. Since `assemble()` runs once
per fold in the OOF loop, whatever the device factor turns out to be, it
compounds `n_folds` times against the per-fold column of the table above —
that is the number the GPU-host re-run should report, alongside end-to-end.

## Reproducing / extending this profile

```
python -m scripts.profile_devices                 # full grid (~3 min on this host)
python -m scripts.profile_devices --quick          # small smoke grid
python -m scripts.profile_devices --n-items 10000,100000 --n-classes 500,5000
```

Raw output: `docs/device-profile.json`. Markdown table: `docs/device-profile.md`.
