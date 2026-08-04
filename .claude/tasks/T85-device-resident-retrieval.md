# T85 — Device-resident dense retrieval + encoder handoff (no D2H mid-pipeline)

status: todo
tier: 8
depends_on: T63, T34 (phase 1), T83, T84, T88

## Sequencing decided 2026-08-04
**T63 (torch-optional core) lands first.** This ticket adds a torch-dependent
array backend, and T63 is the ticket that establishes where torch is allowed to
live. Doing T63 first means the dependency boundary already exists and the torch
backend slots into a `gpu` extra cleanly; doing it after would deepen the very
entanglement T63 exists to remove and leave the README's air-gapped/torch-free
claim unmet for longer. T84's "no torch import reachable from a numpy-backend
run" test is the shared guard.

**T88 also lands first.** It cuts encoder work ~5x on the shared-encoder path,
which is both the larger training-throughput win and a prerequisite for this
ticket being measured honestly — device residency of the example-pool embeddings
is only worth it once those embeddings are computed once rather than per fold.

**Training throughput is the stated priority** (decided 2026-08-04), so scope this
ticket's measurement at the OOF loop first: every per-fold cost compounds
`n_folds` times, and the deployment index is a single extra pass.

## Goal
When the encoder runs on GPU, keep the embeddings *there*: dense retrieval,
prototype/description similarity and the whole feature-assembly kernel set run on
the same device, and the only host↔device crossing per chunk is the one that
lifts the BM25 block onto the device.

## Why
Today the encoder computes on GPU and immediately copies to host
(`convert_to_numpy=True` is forced, `encoder.py:156`), after which the single
largest FLOP outside the encoder — `_dense_topk`'s `Q @ X.T` over the whole
example pool (`retrieval.py:250`) — runs on CPU numpy, followed by ~11 more
`(b, C)` sorts/partitions/scatters, and only then does the feature matrix get
copied *back* to the device for XGBoost. We pay both transfers and get neither
side's compute. T83 quantifies it; this ticket closes it.

## Design

**1. The encoder invariant is reworded, not relaxed.**
`SentenceTransformerEncoder._encode` currently forces both
`normalize_embeddings=True` and `convert_to_numpy=True`, and `CLAUDE.md` calls the
numpy part an invariant. The part that is scientifically load-bearing is
**L2-normalization** (dot == cosine); the container type is not. So:

- `CLAUDE.md`: "Embeddings are L2-normalized" stays; the implied "…and are numpy"
  becomes "…in whatever array type the configured backend uses".
- `normalize_embeddings=True` stays forced and non-configurable.
- `convert_to_numpy` becomes backend-driven: `True` for the numpy backend
  (byte-identical to today), `False` + `convert_to_tensor=True` for torch, with
  the tensor handed straight to the retriever.
- `TextEncoder`'s docstring updates to say the return type is the backend's array
  type. `TfidfEncoder` / `HashingEncoder` keep returning numpy — they are
  host-side by construction and must stay torch-free.

**2. A torch dense retriever, plugged in via T34 phase 1.**
`RetrievalConfig.dense_kind: str = "exact"` gains `"torch"`. This is *why* T34
phase 1 is a prerequisite: without the retriever registry this is a fork of
`DenseRetrieverAdapter` rather than a plug-in.

- `DenseState`'s arrays become backend arrays in memory. `example_emb`,
  `prototypes` and `description_emb` are uploaded **once at build time** and stay
  resident for every query batch and every fold.
- `knn_example_labels`, `prototype_similarity`, `description_similarity`,
  `loo_prototype_similarity` are already written against `ArrayOps` after T84, so
  they need no per-method change — only the state's residence changes.
- `_dense_topk`'s argpartition+argsort pair collapses to a single
  `ArrayOps.topk`, which is what the port exists for.

**3. Persistence stays numpy, always.**
`to_state`/`from_state` call `ArrayOps.to_host` on the way out and upload on the
way in. A model directory remains npz + json + native formats — the portability
invariant is untouched, and a GPU-trained model still loads on an air-gapped
CPU-only host.

**4. Chunking against VRAM.** `feature_chunk` defaults stay as-is for numpy. For
torch, apply T83's VRAM curve: pick a default and auto-halve the chunk on an OOM
retry rather than dying, logging each reduction.

## Determinism — the accepted cost
**Decision taken: tolerance-based parity.** CPU is the reference implementation
and stays the CI/benchmark baseline. GPU results will not be bit-identical:
float32 reduction order differs, so `d_desc_sim` and friends move in the last
ulps, and — the part that actually matters — **near-ties can flip `rank_*`,
`is_*_top1` and the top-n candidate mask**, which changes the candidate *set*,
not merely a value. Consequences, all of which this ticket must deliver:

- Same host + same GPU + same seed stays reproducible. Cross-device does not, and
  `docs/` must say so plainly rather than leaving T26's determinism invariant
  reading as an absolute.
- The parity test asserts `allclose` at a stated tolerance on the continuous
  columns, and on the *ordinal* columns asserts agreement **excluding rows whose
  underlying signal values are within tolerance of a tie** — an honest test, not
  one tuned until it passes.
- The `evaluation.json` manifest records the backend + device the run used, so a
  metric can always be traced to the arithmetic that produced it.

## Files to change
`infrastructure/encoder.py`, `infrastructure/retrieval.py`, `infrastructure/array_ops.py`
(torch backend), `infrastructure/registry.py`, `infrastructure/persistence.py`,
`config.py`, `domain/ports.py` (docstrings), `CLAUDE.md`, `docs/device-policy.md`,
`pyproject.toml` (a `gpu` extra), `tests/unit/test_retrieval.py`,
`tests/integration/test_device_parity.py` (new).

## Tests
- [ ] Numpy backend: byte-identical to pre-T85 output (the default path is untouched).
- [ ] Torch-CPU backend vs numpy: parity within tolerance — gives the parity test
      real coverage in CI, which is GPU-free.
- [ ] Persistence round-trip: train on torch backend → save → load on a host with
      the torch backend unavailable → identical predictions (within tolerance).
- [ ] `to_host` call count per chunk is asserted to be exactly the lexical block
      (a regression test against a re-introduced ping-pong).
- [ ] Encoder returns tensors on the torch backend and numpy on the numpy backend;
      both L2-normalized to within float32 tolerance.
- [ ] Leakage regression (T06) green on both backends.

## Acceptance criteria
- [ ] With encoder + fusion both on GPU, exactly one H2D per chunk, and no D2H
      between encode and the fusion handoff.
- [ ] `dense_kind="torch"` is selected by config alone; no pipeline edits.
- [ ] Model dirs stay portable and pickle-free; GPU-trained loads CPU-only.
- [ ] Measured against T83's baseline (post-T88), with the improvement recorded in
      the ticket, reported for the OOF loop specifically as well as end to end.
- [ ] CPU-only and torch-free installs are entirely unaffected; the torch backend
      lives behind the `gpu` extra established by T63.
- [ ] With T88 landed, the example-pool embeddings are uploaded to the device
      **once per run**, not once per fold.

## Out of scope
BM25 on device (permanently host-side per T83's policy). The fusion handoff (T86).
Which features get computed (T87).
