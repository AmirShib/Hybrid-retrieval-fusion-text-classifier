# T34 — Pluggable retrieval signals (and retrievers) behind the registry

status: in-review (phase 1 + phase 2 landed 2026-08-05)
tier: 3
depends_on: T23, T03

## Progress (2026-08-05) — Phase 2 landed
`SignalProvider` (domain/ports.py) plus `SignalContext`/`SignalMatrix` — a
provider contributes one or more `(b, C)` matrices that join the top-n
candidate union, one layer below the existing `FeatureProvider` (T70), which
only appends post-candidate-selection fusion columns. The two built-ins
(`infrastructure/signals.py`'s `DenseSignalProvider`/`LexicalSignalProvider`)
wrap an already-built `DenseRetriever`/`LexicalRetriever` (T34 phase 1) rather
than re-implementing retrieval, reproducing exactly the five signal matrices
`FeatureAssembler._assemble_chunk` used to compute inline.

The five signals are *not* symmetric in `FEATURE_NAMES` (`d_proto_sim` has no
rank/norm; only some signals get a missing flag or `q_gap_*`), so each
`SignalMatrix` declares, per matrix, which generic derivations apply (raw,
rank, min-max norm, missing, margin+gap, is-top1) and what each derived
column is named — not inferred from a fixed naming convention. Cross-signal
features (`desc_proto_gap`, `n_signal_agreement`, `class_log_freq`) stay
hardcoded in the assembler over the default two providers' matrices, looked
up by node name, exactly as the ticket scopes it — not generalized to N
providers.

`PipelineConfig.signals: List[str] = ["dense", "lexical"]` (registry keys, via
new `register_signal_provider`/`build_signal_providers`/`load_signal_providers`
in `infrastructure/registry.py`). Each provider persists to `signals/<name>/`
via its own `save()`; the two built-ins are a no-op there since their state
already lives in `dense.npz`/`lexical.npz`. `meta.json`'s `components.signals`
records the provider list, defaulting to `["dense", "lexical"]` for model dirs
saved before this field existed (mirrors `dense_kind`/`lexical_kind`'s legacy
default). The schema check on load builds a *schema-only* provider list
(`dense=lexical=None`, safe since `column_names()` never touches the
retriever) so a corrupt/incompatible dir fails on schema drift before other
files are opened.

Found and fixed a real bug in the process: `DeployedArtifacts.with_added_classes`
(the no-retrain add-classes path) rebuilds extended `dense`/`lexical` retrievers
but previously left nothing to update a wrapping signal provider's stale
reference — `rewrap_signal_providers` (`infrastructure/signals.py`) now
rebuilds the built-in providers onto the new retrievers whenever `dense`/
`lexical` are replaced.

**Byte-identical, verified via golden fixture, not just asserted.** Before
refactoring `_assemble_chunk`, three `.npz` fixtures
(`tests/unit/t34_phase2_golden*.npz`) were captured from the *unmodified* code
(labels / no-labels / T87-narrowed-request paths) and are checked in;
`tests/unit/test_features_golden.py` asserts the refactored assembler
reproduces them exactly (`assert_array_equal`, no tolerance). Leakage
regression (`tests/integration/test_leakage.py`) and the T52 benchmark-floor
tests stay green. Full suite: green except the one pre-existing flaky failure
(`test_meta_and_calibrator_change_other_artifacts_untouched`, unrelated).

`tests/unit/test_signal_providers.py` adds a toy stateless third-party
provider (`_TextLengthSignalProvider`) registered purely via
`register_signal_provider` + `cfg.signals` — zero edits to
`FeatureAssembler`/`TrainingPipeline`/`ArtifactRepository` — that joins the
candidate union, round-trips through train → save → load, and produces
identical predictions/confidences/explanations on reload. Also covers the
unregistered-signal-kind schema-drift error and the legacy-default path.

**Scoping decisions carried over from the design, not deviations:**
`desc_proto_gap`/`n_signal_agreement` remain hardcoded to the default two
providers (the ticket's own "not generalized" note); `SignalProvider` has no
`fit()` lifecycle like `FeatureProvider` does — a stateful, non-wrapping
custom signal needing per-fold leakage-free fitting is future work, not
covered by the toy provider (which is deliberately stateless).

## Progress (2026-08-05) — Phase 1 landed
`RetrievalConfig` gained `dense_kind: str = "exact"` / `lexical_kind: str =
"bm25"`. `infrastructure/registry.py` gained `DenseRetrieverSpec`/
`LexicalRetrieverSpec` (build/filename/load, same shape as `FusionSpec`) plus
`register_dense_retriever`/`register_lexical_retriever`,
`dense_retriever_spec`/`lexical_retriever_spec` lookups, and
`build_dense_retriever`/`build_lexical_retriever` factories. Built-ins register
the current adapters under `"exact"`/`"bm25"` with the current filenames
(`dense.npz`, `lexical.npz`+`.json`) — old model dirs load unchanged, and
`_components_from_meta` defaults `dense`/`lexical` to `"exact"`/`"bm25"` for
dirs written before this change (mirroring `encoder`/`fusion`/`calibrator`).

`DenseRetrieverAdapter` gained `to_state()`/`from_state()` (byte-identical
`dense.npz` layout) so its spec's load/save is generic, symmetric with
`LexicalRetrieverAdapter`'s pre-existing `to_state()`/`from_state()`.
`persistence.py` now saves/loads dense and lexical through their specs
(filenames come from the spec, not a hardcoded string), and records
`components.dense`/`components.lexical` in `meta.json`. The legacy
`lexical.pkl` pickle fallback is preserved.

In `application/training.py`, the whole-corpus deployment-index build
(`_build_deployment_index`) and the per-fold OOF loop (`_build_oof`) now go
through `build_dense_retriever`/`build_lexical_retriever` for any non-default
kind. The T88/T32 sharing optimizations (`build_from_embeddings`,
`build_from_counts`/`build_with_shared_descriptions`) are built-in-adapter
internals, not part of the generic retriever-port contract, so they stay
gated behind `dense_kind == "exact"` / `lexical_kind == "bm25"` — a non-default
kind loses the cross-fold sharing (an accepted phase-1 tradeoff, documented
inline, not a TODO) but is otherwise fully pluggable. Default-config behavior
is unchanged: the full test suite is green except the one pre-existing flaky
failure (`test_meta_and_calibrator_change_other_artifacts_untouched`,
unrelated to this ticket), and `python -m scripts.demo` runs unchanged.

Added `tests/unit/test_registry.py::test_register_and_build_custom_dense_retriever`,
`test_register_and_build_custom_lexical_retriever`,
`test_dense_and_lexical_kind_selected_persisted_and_loaded_end_to_end` (train +
persist + reload with a custom dense/lexical kind, own filenames, own
`components` entry) and the unknown-kind error-contract tests, mirroring the
existing fusion/calibrator registry tests.

(Phase 2, described as still open in this section as originally written, has
since landed — see "Progress (2026-08-05) — Phase 2 landed" above.)

## Re-prioritized 2026-08-04 (Tier 8)
Both phases are now on the critical path and the two halves have **different**
consumers, so they should be scheduled separately:

- **Phase 1 (retriever registry) is a prerequisite for T85** and should be done
  first — it is small, already fully specified below, and without it a
  device-resident dense retriever is a fork of `DenseRetrieverAdapter` rather
  than a `dense_kind="torch"` plug-in.
- **Phase 2 (signal providers) is the signal-level answer to feature
  decoupling.** T87 makes *column*-level pruning demand-driven, but a whole
  expensive signal (a cross-encoder, an external scoring service) can only be
  skipped by not enabling it — and skipping one changes the candidate set, so it
  must stay an explicit config decision. Phase 2 therefore depends on T87's
  dependency graph: a signal provider is a node in it, and `candidate_features`
  is what marks a node as unprunable.

Sequencing note added to the existing one below: land T87 before phase 2, so
providers plug into a graph that already exists rather than one invented twice.

## Goal
Open the closed heart of the system: make retrieval *signals* pluggable — a user
adds a sixth signal (domain lexicon, char-ngram BM25, metadata prior, in-house
embedding service) via registration + config, without forking `FeatureAssembler`
or breaking schema compatibility. Along the way, put the two retriever *builders*
behind registry keys (the seam T31/FAISS needs anyway).

## Why
T23 made encoder/fusion/calibrator swappable, but the five signals are hardcoded:
`FeatureAssembler._assemble_chunk` computes exactly five matrices, the candidate
set is a hardcoded union of five `_topn_mask` calls (`features.py:138-144`), and
`FEATURE_NAMES` is a frozen module-level list (`domain/services.py`). Likewise
`TrainingPipeline` calls `DenseRetrieverAdapter.build` / `LexicalRetrieverAdapter
.build` directly (`training.py:174-175`) — retrievers are the one component pair
*not* selectable by config. Every professional deployment eventually has one
extra signal; today that's a fork.

## Design — two phases, one ticket

### Phase 1: retriever builders behind the registry (small, do first)
- `RetrievalConfig` gains `dense_kind: str = "exact"` and
  `lexical_kind: str = "bm25"`. New registry maps (`register_dense_retriever`,
  `register_lexical_retriever`) whose specs carry `build(...)`, a persistence
  `filename`, and `load(...)` — same shape as `FusionSpec`.
- `TrainingPipeline` and `ArtifactRepository` go through the registry instead of
  the concrete classes. Built-ins register the current adapters with the current
  filenames (`dense.npz`, `lexical.pkl`) → existing model dirs load unchanged.
- This alone unblocks T31 (FAISS) as a pure plug-in.

### Phase 2: signal providers
- New port `SignalProvider` (domain): given a query batch it yields one or more
  named `(b, C)` matrices plus how each participates:

      class SignalProvider(ABC):
          name: str                      # unique prefix for its feature columns
          def build(items, labels, label_space, encoder, cfg) -> fitted state
          def score(texts, q_emb, k) -> Dict[str, np.ndarray]   # (b, C) each
          candidate_features: List[str]  # which outputs join the top-n union
          def save(dir) / load(dir)

  NaN keeps meaning "did not retrieve" (CLAUDE.md invariant) — providers must
  emit NaN, never 0, for misses.
- `FeatureAssembler` consumes an ordered provider list: builds each provider's
  matrices, takes the top-n union over the declared candidate features, then
  derives the *generic* per-signal features it already computes (raw value, rank,
  minmax-norm, is-top1, missing flag) uniformly per matrix. The five built-in
  signals become two built-in providers (dense: desc/proto/knn; lexical:
  desc/knn) producing **exactly today's columns**.
- **Schema**: `FEATURE_NAMES` becomes derived — `feature_names(providers)` — with
  the module-level list kept as the value for the default provider set (single
  source of truth still holds; the persisted `meta.json` schema check in
  `persistence.py` already compares full lists, so drift stays a load-time error).
  Cross-signal features (`desc_proto_gap`, `n_signal_agreement`) stay owned by
  the assembler over the default providers; a provider can additionally declare
  agreement participation via its top-1 argmax.
- **Config/persistence**: `PipelineConfig.signals: List[str] = ["dense",
  "lexical"]`; each provider persists under `signals/<name>/` via its spec.
  `meta.json` `components` records the provider list, so load reconstructs the
  same assembler. With T29, third-party providers resolve via entry points.

### The hard requirement
With the default config, the assembled feature table must be **byte-identical**
to today's (same columns, same order, same values) — the leakage regression test
and T52's benchmark floors are the net. Phase 2 is a refactor with a
golden-output guarantee, not a behaviour change.

## Files to change
Phase 1: `config.py`, `infrastructure/registry.py`, `application/training.py`,
`infrastructure/persistence.py`, `tests/unit/test_registry.py`.
Phase 2: `domain/ports.py`, `domain/services.py`, `application/features.py`,
`infrastructure/retrieval.py` (providers), `infrastructure/persistence.py`,
`tests/unit/test_features.py` (golden-frame test), `tests/integration/test_e2e.py`.

## Tests
- [ ] Phase 1: default kinds round-trip a model dir byte-identically; a dummy
      registered dense retriever is selected by config and persisted/loaded.
- [ ] Phase 2 golden test: default providers reproduce today's feature frame
      exactly on a fixed corpus (store expected values, not just shape).
- [ ] A toy third-party provider (e.g. text-length prior) adds its columns, joins
      the candidate union, round-trips through save/load, and inference on a dir
      trained with it produces identical confidences.
- [ ] Loading a model whose provider list ≠ code's registered set fails with the
      schema-drift message naming the missing provider.
- [ ] Leakage regression test still green (providers built per fold in OOF).

## Acceptance criteria
- [ ] Adding a signal = implement port + register + config; zero edits to
      assembler/pipelines/persistence.
- [ ] Default config: byte-identical features, unchanged `meta.json` for the
      built-in set (older dirs load).
- [ ] T33's cross-encoder is implementable as a provider (update that ticket's
      notes when this lands); T31 needs only Phase 1.

## Out of scope
The DAG orchestration question (T71 — this ticket's provider dependency shape is
input to that spike); custom *fusion-layer* features not tied to retrieval (T70);
non-text inputs into providers (metadata plumbing is T70/T72 territory).
