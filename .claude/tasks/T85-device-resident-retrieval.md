# T85 — Device-resident dense retrieval + encoder handoff (no D2H mid-pipeline)

status: in-progress
tier: 8
depends_on: T63, T34 (phase 1), T83, T84, T88

## Progress (2026-08-06, still CPU-only host)

**Landed, verified on CPU-torch (GPU-free parity, per the ticket's own Tests
section):**

- `infrastructure/array_ops_torch.py::TorchArrayOps` — the full `ArrayOps`
  port on torch tensors. Kept in its own module (not `array_ops.py`, which
  `infrastructure/__init__.py` imports unconditionally) so importing the
  package never imports torch. Registered in `registry.py` under `"torch"`
  unconditionally, but the registration is metadata only — `import torch`
  happens inside `build_array_ops("torch")`'s deferred build callable, never
  at registration time.
- `SentenceTransformerEncoder.array_backend` (`"numpy"` default, every
  existing caller unchanged) / `set_array_backend()`. `"torch"` forces
  `convert_to_numpy=False, convert_to_tensor=True`, still with
  `normalize_embeddings=True` forced — the L2-norm invariant stays absolute,
  only the container changed, per the ticket's own design point 1.
- `DenseState`'s arrays are uploaded once at build time
  (`DenseRetrieverAdapter.build`/`build_from_embeddings`, fed by `_prototypes_and_freq`,
  which now flows torch-tensor embeddings through `ops.asarray(...).reshape(-1)`
  instead of a raw `np.asarray` that would crash on a CUDA tensor) and stay
  resident for every query the run makes afterward — the corpus is not
  re-uploaded per chunk or per fold, which is the actual T83-identified cost.
  `_dense_topk`'s matmul/argpartition/argsort/gather all route through the
  array port with a single `to_host` per chunk (the small `(chunk, k)`
  result), not a `np.take_along_axis`/`np.ascontiguousarray` pair that would
  silently force a CUDA tensor through numpy.
- **Scope boundary, deliberate, not an oversight:** every public
  `DenseRetrieverAdapter` query method (`knn_example_labels`,
  `prototype_similarity`, `loo_prototype_similarity`, `description_similarity`)
  explicitly `to_host`s its return value. `application/features.py` is *not*
  in this ticket's file list, and its kernels (`_topn_mask`/`_row_rank`/
  `_row_margin`/`_row_minmax`) mix raw numpy indexing with the array port in
  ways T84 deliberately left non-polymorphic ("not an array-API
  reimplementation"). Feeding them a device tensor would silently round-trip
  through numpy on CPU torch and outright crash on CUDA. `TrainingPipeline`
  therefore carries two `ArrayOps` instances: `self._ops` (the run's resolved
  backend — encoder + dense retriever) and `self._assembler_ops` (always
  `NumpyArrayOps`, feeding `FeatureAssembler`/`build_signal_providers`).
  Concretely, this means T85 delivers "corpus resident, uploaded once" and
  "dense-index compute is device-side," but *not* literally zero D2H per
  chunk — one `to_host` per public method call remains, at a `(chunk, C)`-or-
  smaller result, not the corpus. Removing that is T86's stated job
  ("zero-copy feature matrix into fusion"), not a T85 shortfall.
- `dense_kind` stays `"exact"` — no new `"torch"` retriever kind. Assessed and
  rejected: `DenseRetrieverAdapter` is already `ArrayOps`-polymorphic via
  T34's registry + T84's port, so a `dense_kind` fork would only add a
  redundant selector with no behavioral difference from `array_backend`
  alone doing the selection. Deviates from the ticket's original design
  sketch; the outcome (config-only selection, no pipeline edits) is what the
  acceptance criterion actually asked for.
- No new `gpu` extra. Torch is already exactly one install away via T63's
  `sentence-transformers` extra; a second extra pinning the same dependency
  would only be a second name for it (documented in `pyproject.toml`).
- `resolve_array_backend`'s auto-selection reordered: the crossover-scale
  check now runs *before* `torch_installed()`/`cuda_available()`, not after.
  Necessary, not cosmetic — see "A second, unplanned fix" below.
- Tests: `tests/unit/test_array_ops_torch.py` (full port parity vs
  `NumpyArrayOps`), `tests/unit/test_encoder_array_backend.py`,
  `tests/unit/test_retrieval.py::TestDenseRetrieverAdapterTorchBackend`,
  `tests/integration/test_device_parity.py` (a real `TrainingPipeline.run`
  through both backends on identical data, checking coverage/accuracy land
  within tolerance and that a torch-trained dense index persists as numpy).
  All torch-gated tests skip cleanly when torch isn't installed. Full suite
  (794 tests) green alongside them.

**A second, unplanned fix, found while implementing this ticket.** T84's
`resolve_array_backend` short-circuited on registry membership ("torch" not
registered ⇒ numpy, skip every probe) specifically to keep an ordinary numpy
run from ever importing torch. Once T85 registers "torch" unconditionally,
that check stops being a usable install proxy; swapping it for
`torch_installed()` (a non-importing `find_spec` probe) alone would have made
`cuda_available()` — a *real* `import torch` — run on every auto-resolved run
on any host where torch happens to be installed for an unrelated reason
(this repo's own dev environment, via the `sentence-transformers` extra),
including a two-item smoke test. Reproduced a live segfault
(`OMP: Error #179`) from exactly this before catching it and reordering the
checks — crossover-scale first, then `torch_installed()`, then
`cuda_available()`. See `docs/device-policy.md`'s T85 addendum for the full
account, including a related, code-independent torch↔xgboost OpenMP conflict
found on this dev host and worked around in `tests/conftest.py`.

**Not done — needs a real GPU host** (same limitation T83 already flagged):
transfer count/bytes per batch and the VRAM-vs-`feature_chunk` curve are
unmeasured from here; the acceptance criteria below that depend on them are
checked `[ ]`, matching T83's own honesty convention rather than claiming a
verification that didn't happen. Also unmeasured: whether this dev host's
torch↔xgboost OpenMP conflict reproduces on the reference platform (Linux
x86_64) — flagged, not assumed either way.

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
      Full suite (794 tests) green with no behavior change on `array_backend="numpy"`
      (the default); `_dense_topk`'s numpy `gather`-based rewrite verified equivalent
      to the `take_along_axis` it replaced.
- [x] Torch-CPU backend vs numpy: parity within tolerance — gives the parity test
      real coverage in CI, which is GPU-free. `tests/unit/test_array_ops_torch.py`
      (full port parity) + `tests/integration/test_device_parity.py` (real
      `TrainingPipeline.run`, both backends, coverage/accuracy within tolerance).
- [x] Persistence round-trip: train on torch backend → save → load on a host with
      the torch backend unavailable → identical predictions (within tolerance).
      `DenseRetrieverAdapter.to_state`/`from_state` covered directly; the load path
      never touches torch regardless (persistence defaults to `NumpyArrayOps`).
- [ ] `to_host` call count per chunk is asserted to be exactly the lexical block.
      **Not true as delivered, so not asserted as true.** Every public
      `DenseRetrieverAdapter` query method `to_host`s its own return value (up to 4
      per chunk: desc/proto/knn/loo), not one shared lexical-block crossing —
      `application/features.py` isn't backend-polymorphic yet (T86), so the
      dense-retrieval and lexical results must both already be numpy by the time
      they reach it. Closing this to "exactly one" is T86's job.
- [x] Encoder returns tensors on the torch backend and numpy on the numpy backend;
      both L2-normalized to within float32 tolerance. `tests/unit/test_encoder_array_backend.py`.
- [ ] Leakage regression (T06) green on both backends. Green on numpy (the existing
      suite, unmodified, is part of the 794 passing). Not separately re-run
      parametrized over the torch backend in this session — a reasonable, cheap
      follow-up, not attempted here.

## Acceptance criteria
- [ ] With encoder + fusion both on GPU, exactly one H2D per chunk, and no D2H
      between encode and the fusion handoff. **Partially met, honestly scoped:**
      the corpus is uploaded once per run (not per chunk/fold) and dense-index
      compute is device-resident; the small per-chunk *result* still crosses back
      to host at each public query method (see the `to_host`-count test note
      above) — not literally zero, and unverified for "fusion" specifically (T86's
      handoff is out of this ticket's scope by design). No GPU host to measure the
      real transfer count on either.
- [x] `dense_kind="torch"` is selected by config alone; no pipeline edits. Delivered
      differently than sketched: `dense_kind` stays `"exact"`; `array_backend`
      alone selects the backend (see the Progress note — a fork was assessed and
      rejected as redundant, not skipped).
- [x] Model dirs stay portable and pickle-free; GPU-trained loads CPU-only.
      `DenseRetrieverAdapter.to_state` forces `to_host` on every array explicitly.
- [ ] Measured against T83's baseline (post-T88), with the improvement recorded in
      the ticket, reported for the OOF loop specifically as well as end to end.
      **Not measured — no GPU host available**, same limitation T83 itself flagged.
- [x] CPU-only and torch-free installs are entirely unaffected; the torch backend
      lives behind the `sentence-transformers` extra established by T63 (not a
      separate `gpu` extra — see the Progress note on why a second extra for the
      same dependency was rejected).
- [x] With T88 landed, the example-pool embeddings are uploaded to the device
      **once per run**, not once per fold. `_shared_document_embeddings`'s existing
      once-per-run cache (T88) now caches a resident tensor when the encoder's
      backend is torch, so this falls out of T88's design without extra plumbing.

## Out of scope
BM25 on device (permanently host-side per T83's policy). The fusion handoff (T86).
Which features get computed (T87).
