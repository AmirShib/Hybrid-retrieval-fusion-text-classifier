# T33 — Cross-encoder reranker as an optional 6th retrieval signal

status: todo
tier: 3
depends_on: T03

## Goal
Add a **sixth retrieval signal** — a cross-encoder reranker score — to the feature
matrix as two new columns in `FEATURE_NAMES`. The signal is **entirely optional**:
when no cross-encoder is configured (the default), the system produces identical
output to today, the new columns are absent (or NaN-filled and masked), and no
torch dependency is introduced.

## Why
The five current signals (prototype cosine, KNN label, BM25, BM25-description, KNN
density) are all bi-encoder or lexical. A cross-encoder jointly encodes
`(item_text, candidate_description)` pairs and typically achieves higher ranking
quality for the top candidates — it is the standard reranking step in production
retrieval systems. Adding it as a feature (not a hard replacement) lets the fusion
model learn to weight it against the existing signals.

It must be optional because:
- It requires `sentence-transformers` (torch), unavailable on air-gapped hosts.
- Cross-encoders are 10–100× slower than bi-encoders; many deployments skip it.
- The existing architecture and all existing tests must be unaffected when absent.

## Architecture: 6th signal, not a fusion swap

The cross-encoder is a **retrieval signal at the feature-assembly layer**, NOT a
replacement for the fusion model. It produces a scalar score per `(item, candidate)`
pair, which feeds into `FeatureAssembler.assemble` alongside the five existing signals.

New port in `domain/ports.py`:

```python
class PairwiseReranker(ABC):
    @abstractmethod
    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Return float32 array of shape (n,) — relevance score per pair."""
    @abstractmethod
    def save(self, directory: str) -> None: ...
    @classmethod
    @abstractmethod
    def load(cls, path: str, **kwargs) -> "PairwiseReranker": ...
```

## Optionality contract

- `PipelineConfig` gains an optional `reranker: Optional[RerankerConfig] = None`.
- When `reranker is None` (the default), `FeatureAssembler` is constructed without a
  reranker. All existing tests pass unchanged; no new columns appear.
- When a reranker is configured, two new columns are appended to `FEATURE_NAMES`:
  `ce_score` and `ce_missing` (binary, 1 if the reranker was not called for this pair).
- `FEATURE_NAMES` is updated **only** when the reranker is present. The persisted
  `meta.json` records which feature set was used; inference loads the same set.
- Ports, registry, pipeline, and persistence all treat `Optional[PairwiseReranker]` as
  first-class — `None` means absent, not a default implementation.

## New config

```python
@dataclass
class RerankerConfig:
    kind: str = "cross-encoder"           # extensible via registry
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    batch_size: int = 32
    top_k: int = 10                       # rerank only top-k candidates
    params: dict = field(default_factory=dict)
```

`top_k` is critical: the cross-encoder is only applied to the candidates that
passed the retrieval stage, limited to the `top_k` highest-scoring ones. Pairs
outside `top_k` get `ce_score = NaN`, `ce_missing = 1`.

## Files to add/change
- `text_classifier/domain/ports.py` — new `PairwiseReranker(ABC)`
- `text_classifier/domain/services.py` — conditional `FEATURE_NAMES` extension
  (keep base list unchanged; append `ce_score`, `ce_missing` only when reranker present)
- `text_classifier/config.py` — `RerankerConfig` + `PipelineConfig.reranker`
- `text_classifier/infrastructure/reranker.py` — NEW: `CrossEncoderReranker` +
  `register_reranker("cross-encoder")`; lazy-import `sentence_transformers`; skip
  gracefully if not installed
- `text_classifier/application/features.py` — `FeatureAssembler` accepts
  `Optional[PairwiseReranker] = None`; computes CE pair scores when present
- `text_classifier/application/training.py` — build reranker from config when
  `cfg.reranker is not None`; pass to `FeatureAssembler`
- `text_classifier/infrastructure/persistence.py` — save/load reranker artifact when
  present; record in `meta.json` as `components.reranker`
- `pyproject.toml` — cross-encoder is a subset of the existing `sentence-transformers`
  optional dep; no new package needed (same torch path)
- `tests/unit/test_reranker.py` — NEW (skipif sentence-transformers absent)
- `tests/integration/test_e2e.py` — confirm no-reranker path unchanged; add
  optional reranker e2e test (skipif)

## Tests

### Unit (`test_reranker.py`, skipif sentence-transformers absent)
- [ ] `score` returns float32 `(n,)` in `[-∞, +∞]` (cross-encoders are logits, not
      probabilities — the fusion model learns the scale).
- [ ] More-relevant pair scores higher than irrelevant pair.
- [ ] `save` → `load` round-trip: identical scores on the same pairs.
- [ ] Registered under `"cross-encoder"`; `build_reranker(RerankerConfig())` returns it.

### Unit (offline / always run)
- [ ] `FeatureAssembler` with `reranker=None` produces identical output to current
      (same columns, same values) — regression guard.
- [ ] `FeatureAssembler` with a `MockReranker` stub produces `ce_score` and
      `ce_missing` columns with correct values; `ce_missing=1` for pairs outside
      `top_k`.
- [ ] `PipelineConfig(reranker=None)` round-trips `to_dict() / from_dict()` correctly.
- [ ] `PipelineConfig(reranker=RerankerConfig())` round-trips correctly.

### Integration
- [ ] No-reranker end-to-end: all existing T07 tests pass without modification.
- [ ] With-reranker end-to-end (skipif): `TrainingPipeline` with
      `reranker=RerankerConfig(...)` trains, saves, loads, predicts; `meta.json`
      records `components.reranker == "cross-encoder"`; `FEATURE_NAMES` in loaded
      artifacts includes `ce_score` and `ce_missing`.
- [ ] Leakage guard for reranker: cross-encoder is applied to retrieved candidates
      only; item's own text does not appear as a candidate in its OOF fold.

## Acceptance criteria
- [ ] With no `reranker` config, the system is **byte-for-byte identical** to today.
- [ ] Adding a reranker is config-only: no edits to `TrainingPipeline` or
      `ArtifactRepository` (beyond the initial plumbing in this ticket).
- [ ] `sentence-transformers` / torch absent → cross-encoder tests skipped;
      no-reranker path fully functional.
- [ ] `MockReranker` offline stub covers the feature-assembly path without torch.
- [ ] `ce_missing` correctly distinguishes "not retrieved" from "retrieved but not
      reranked" (when `top_k` is smaller than the candidate set).

## Out of scope
Distilling the cross-encoder into a bi-encoder (T24/T31). Benchmarking quality
improvement vs. the five-signal baseline (T40 ablation). Hyperparameter search.
Fine-tuning the cross-encoder model. Using the cross-encoder as a standalone
ranker instead of a feature.
