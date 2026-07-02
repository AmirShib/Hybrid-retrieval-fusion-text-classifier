# T34 — Pluggable retrieval signals (and retrievers) behind the registry

status: todo
tier: 3
depends_on: T23, T03

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
