# T85 — Device-resident dense retrieval + encoder handoff (no D2H mid-pipeline)

status: in-review
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
- [x] Numpy backend: byte-identical to pre-T85 output (the default path is untouched).
      Verified two ways: the checked-in T34 golden fixtures, and a 466-array
      before/after capture of `assemble()` (labels / no-labels / `requested=` /
      leave-one-out paths, prototypes, and a full `TrainingPipeline.run`'s
      headline metrics) taken on this host before the refactor and re-checked
      after every step — byte-identical, `rtol=0`/`atol=0`, throughout.
- [x] Torch-CPU backend vs numpy: parity within tolerance
      (`tests/integration/test_device_parity.py`), with the ordinal exemption
      spelled out and checked rather than assumed.
- [x] Persistence round-trip: train on the torch backend → save → load with
      `torch_installed()` patched False → identical predicted keys, confidences
      within 1e-4.
- [x] Transfer counts per chunk asserted: 3 uploads (the BM25 block: `(b, k)`
      labels, `(b, k)` scores, `(b, C)` description scores) and 2 `to_host`
      calls, both at the fusion handoff; one extra upload per chunk when the
      encoder is host-side.
- [x] Encoder returns tensors on the torch backend and numpy on the numpy
      backend; both L2-normalized to float32 tolerance.
- [x] Leakage regression (T06) green on both backends
      (`TestLeakageOnTheTorchBackend`: fold disjointness + the singleton
      prototype canary).
- [x] OOM chunk-halving retry.

## Acceptance criteria
- [x] With encoder + fusion both on GPU, exactly one H2D per chunk, and no D2H
      between encode and the fusion handoff. **Asserted structurally** (transfer
      counting with a device-resident encoder stub and a device-resident index),
      not measured on a GPU — see the measurement gap below.
- [x] `dense_kind="torch"` is selected by config alone; no pipeline edits.
      `PipelineConfig.validate` rejects the contradictory
      `dense_kind="torch"` + `array_backend="numpy"` pairing, and `"auto"`
      resolves *to* torch when the dense kind asks for it.
- [x] Model dirs stay portable and pickle-free; GPU-trained loads CPU-only.
- [ ] **Measured against T83's baseline (post-T88) — PARTIAL, and the one
      criterion this ticket does not close.** This host has no CUDA device
      (`torch.cuda.is_available()` is False), the same limitation T83 recorded,
      so the GPU speedup is unmeasured. What *was* measured, and written into
      `docs/device-policy.md`: numpy vs torch-**CPU** through the same seam, at
      10k items / 5k classes (`assemble()` 32.5s → 14.7s) and as an interleaved
      same-process A/B at 10.8k items / 2k classes (1.27s → 0.54s median, 2.34x).
      That establishes the seam costs nothing and the hot stages are the ones
      T83 predicted; it says nothing about device residency. The GPU-host re-run
      (`scripts/profile_devices.py --array-backend torch`, which this ticket
      added) still owes the OOF-loop and end-to-end numbers.
- [x] CPU-only and torch-free installs are entirely unaffected; the torch backend
      lives behind the `gpu` extra established by T63. The "no torch import
      reachable from a numpy-backend run" guard still holds with the torch
      backend registered — registration stores a factory, and every import-free
      check in `resolve_array_backend` runs before the CUDA probe.
- [x] With T88 landed, the example-pool embeddings are uploaded to the device
      **once per run**, not once per fold: `_shared_document_embeddings` adopts
      into the backend once and each fold takes an on-device slice
      (`ArrayOps.take`), and the T88 sharing path was widened to cover
      `dense_kind="torch"` (same adapter, different backend).

## What landed (files)
`domain/ports.py` (ArrayOps grew from 18 to 33 methods — every addition is a
call the kernels were already making *around* the port; `TextEncoder
.set_array_ops`; `ArrayOps.free_memory`), `infrastructure/array_ops.py`
(`TorchArrayOps`, `available_array_backend`, `dense_kind` in the resolver),
`infrastructure/device.py` (`torch_installed`, via `find_spec` — no import),
`infrastructure/retrieval.py` (backend-resident `DenseState`, `with_array_ops`,
hoisted contiguous transpose, kernels through the port),
`infrastructure/signals.py` (no more `to_host` mid-chunk; the BM25 lift),
`infrastructure/registry.py` (`torch` array-ops + dense-retriever kinds),
`infrastructure/persistence.py` (backend resolution + re-adoption at load;
`DeployedArtifacts.array_ops`), `application/features.py` (kernels through the
port, vectorized `n_signal_agreement`, single-block handoff, OOM retry),
`application/training.py` / `application/inference.py` (thread the backend),
`application/evaluation.py` (manifest `execution` block), `config.py`,
`CLAUDE.md`, `README.md`, `CHANGELOG.md`, `docs/device-policy.md`,
`pyproject.toml` (`gpu` extra), `scripts/profile_devices.py`
(`--array-backend`), `tests/unit/test_array_ops.py`,
`tests/integration/test_leakage.py`, `tests/integration/test_device_parity.py`
(new).

## Follow-ups this surfaced
- **The GPU measurement** (above) — the only open acceptance criterion.
- `rank_*` on a candidate a signal did not retrieve was never a reproducible
  value on *any* backend (all such candidates tie at `-inf` and the numpy path
  breaks the tie with an unstable sort). T85 only made it visible. If that
  column is worth anything to the model, it should be defined deliberately
  (e.g. NaN for unretrieved) rather than left to the sort — a separate ticket.
- A custom `FeatureProvider` still receives host numpy (`FeatureContext`'s
  documented contract), so a configured provider costs one extra D2H per chunk.
  Giving providers a backend-aware context is T79/T86 territory, not this one.

## Out of scope
BM25 on device (permanently host-side per T83's policy). The fusion handoff (T86).
Which features get computed (T87).
