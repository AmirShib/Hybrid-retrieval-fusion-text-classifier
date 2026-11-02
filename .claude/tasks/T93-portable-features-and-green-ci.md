# T93 — CPU-independent rank features; CI green again

status: done
tier: 0

## What landed (2026-11-01/02, on `dev`)

- **Rank features depended on the CPU.** `ArrayOps.argsort` used numpy's default
  sort, which is neither stable nor portable (numpy 2.x dispatches to
  CPU-specific SIMD kernels), so tied scores ranked differently on the training
  and deployment hosts. Now stable on both backends; the port contract requires it.
- **CI had failed on every run, `main` included**, for three reasons: the golden
  feature test compared float32 BLAS output bit-exactly across machines (now:
  discrete columns exact, continuous to 1e-6); `package-smoke`/`quality-benchmark`
  installed a drifted hand-copied dependency list (no `tqdm`; now installs the
  package's own deps); mypy reported 28 errors (fixed without blanket ignores).
- The mypy work turned two latent `AttributeError`s into clear errors: a signal
  provider computing with no index built, and saving/updating/re-tuning with a
  custom retriever kind that lacks the needed method (duck-typed check).
- **macOS training segfault** (torch↔xgboost OpenMP, see T85): the package now
  imports xgboost first. Not reproducible on Linux.

First green CI run: 123 (`3b0322e`).

## Follow-up
- `ArrayOps.topk` selects with `argpartition`, so *which* tied element survives at
  the k-th boundary is still platform-dependent. Making it exact needs a full
  sort, costly on the kNN path; only matters when tied neighbours carry different
  labels.
