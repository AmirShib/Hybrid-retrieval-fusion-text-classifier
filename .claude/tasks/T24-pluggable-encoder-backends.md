# T24 — Pluggable encoder backends behind the `TextEncoder` port

status: todo
tier: 2
depends_on: T23

## Goal
Prove the `TextEncoder` port is genuinely swappable by shipping a **second,
real, dependency-light encoder backend** alongside `SentenceTransformerEncoder`,
selectable via `EncoderConfig.kind`, and trainable/persistable end-to-end.

## Why
Right now the only production encoder is `SentenceTransformerEncoder`, which
needs torch + a model download. Some hosts can't run torch, and some users want
a classical baseline. A second backend (a) validates that T23's registry is real,
(b) gives an air-gapped, torch-free option, and (c) provides a baseline to compare
embedding quality against.

## Candidate backend (pick one; TF-IDF recommended)
- **`TfidfEncoder`** (recommended): `sklearn.feature_extraction.text.TfidfVectorizer`
  → dense, L2-normalized rows (so dot == cosine, honoring the core invariant).
  Pure sklearn/scipy, no torch, no download. `save`/`load` via pickle of the fitted
  vectorizer. Must implement a `fit`/`build` step over the training corpus since
  TF-IDF vocabulary is corpus-dependent (unlike a pretrained ST model).
- Alternative: a hashing-vectorizer encoder (stateless, no vocab) if you want to
  avoid the fit step entirely.

## Files to add/change
- `text_classifier/infrastructure/encoder.py` (or a new `encoder_tfidf.py`)
  — implement `TfidfEncoder(TextEncoder)` and `register_encoder("tfidf")`
- `text_classifier/config.py` — document `EncoderConfig.kind="tfidf"` + its params
- `tests/unit/test_encoder_tfidf.py` — NEW
- `tests/integration/test_e2e.py` — parametrize the e2e test over encoder kind

## Implementation notes
- **L2-normalization is mandatory** — assert unit-norm rows (dot product == cosine
  is relied on across retrieval and feature assembly).
- TF-IDF needs the vocabulary fitted on the training corpus. Decide where the fit
  happens: most naturally a `TfidfEncoder.fit(texts)` called by the training
  pipeline's deploy step (and per-fold if `use_per_fold_encoder`), mirroring how
  `train_encoder` works for sentence-transformers. Keep the per-fold leakage rule:
  a fold's vectorizer is fit on **train** rows only (this is itself a leakage
  surface — cover it in the test).
- Output must be `float32` and dense `(n, d)` to match `encode`'s contract.

## Tests

### Unit (`test_encoder_tfidf.py`, offline)
- [ ] `encode` returns `(n, d)` float32 with unit-norm rows.
- [ ] Shared tokens → higher cosine than disjoint tokens (retrieval is meaningful).
- [ ] Empty / OOV text → finite vector (zero vector normalized to zeros, not NaN).
- [ ] `save` → `load` round-trips: identical embeddings on the same input.
- [ ] Registered under `"tfidf"`; `build_encoder(EncoderConfig(kind="tfidf"))` returns it.

### Integration (extend e2e)
- [ ] Parametrize `TestTrainReport` / persistence round-trip over
      `encoder_kind ∈ {"hashing-double", "tfidf"}` so the **whole pipeline trains,
      saves, loads, and predicts** with the TF-IDF encoder, fully offline.
- [ ] **Per-fold leakage guard:** in the OOF loop the fold's TF-IDF vocabulary is
      fit only on training rows (extend the T06 canary or add a focused assertion
      that a held-out item's unique token is absent from its fold vectorizer vocab).

## Acceptance criteria
- [ ] A second, torch-free encoder trains + infers end-to-end via config alone.
- [ ] Unit-norm invariant asserted for the new backend.
- [ ] Persistence round-trip identical within ~1e-6.
- [ ] No change to default (sentence-transformers) behaviour or existing tests.

## Out of scope
Encoder fine-tuning for the TF-IDF backend (it has no trainable weights beyond
vocab). Benchmarking embedding quality across backends (T40 ablation harness).
