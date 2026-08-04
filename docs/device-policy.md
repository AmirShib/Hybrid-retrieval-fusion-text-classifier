# Device policy (T83)

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

## Reproducing / extending this profile

```
python -m scripts.profile_devices                 # full grid (~3 min on this host)
python -m scripts.profile_devices --quick          # small smoke grid
python -m scripts.profile_devices --n-items 10000,100000 --n-classes 500,5000
```

Raw output: `docs/device-profile.json`. Markdown table: `docs/device-profile.md`.
