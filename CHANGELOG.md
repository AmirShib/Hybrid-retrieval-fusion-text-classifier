# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [Semantic Versioning](https://semver.org/): the version
lives in one place, `text_classifier/_version.py` (see `RELEASING.md`).

## [Unreleased]

### Added
- **Device-resident dense retrieval + encoder handoff (T85)** — when the
  encoder runs on a GPU, the embeddings now stay there. A torch `ArrayOps`
  backend (`array_backend="torch"`, behind the new `gpu` extra) runs every
  dense-side kernel on the device, and a torch dense retriever
  (`retrieval.dense_kind="torch"`, registered through T34 phase 1's registry —
  no pipeline edits) keeps `example_emb`/`prototypes`/`description_emb`
  resident: uploaded once per run, sliced per fold, reused for every query
  batch. `SentenceTransformerEncoder` hands its tensors straight to retrieval
  (`convert_to_tensor`) instead of forcing numpy. Per query chunk that leaves
  exactly one host→device crossing — the BM25 block, which T83's policy keeps
  permanently host-side — and two device→host crossings, both at the fusion
  handoff; the three `to_host` calls `_scatter_knn` used to make per signal per
  chunk are gone, and a regression test asserts the counts. `array_backend`
  stays `"auto"` by default and still resolves to numpy unless torch is
  installed, a device is visible, and the run clears T83's crossover.
  - **Persistence and portability are untouched.** `to_state` lowers every
    array to numpy, so `dense.npz` from a GPU run is byte-comparable with a CPU
    run's and a GPU-trained model loads and scores on an air-gapped, torch-free
    host (the backend downgrades to numpy with a warning; the model dir never
    dictates execution).
  - **Determinism is qualified, not repealed.** The numpy backend is unchanged
    bit for bit and remains the reference, the CI baseline and the golden
    fixtures. A device backend agrees within float tolerance: float32 reduction
    order differs, so continuous columns move in the last ulps and a near-tie
    can flip `rank_*`/`is_*_top1` and with it the candidate set. Same host +
    same device + same seed stays reproducible; `evaluation.json`'s manifest
    now records an `execution` block (backend, device, dense kind) so a metric
    can be traced to the arithmetic that produced it. See
    `docs/device-policy.md`.
  - **Chunking survives VRAM pressure.** An out-of-memory failure halves
    `retrieval.feature_chunk` and retries (logging each reduction) instead of
    killing a run mid-flight.
  - Also in this ticket, as a consequence of routing every kernel through the
    port: `n_signal_agreement` lost its per-row Python `set` loop for a
    vectorized sort/count, `_dense_topk`'s argpartition/gather/argsort trio
    collapsed into one `ArrayOps.topk`, and the feature frame is materialized
    from one stacked block instead of one `np.asarray` per column.
- **Torch-optional install via extras (T63)** — `sentence-transformers` (and the
  torch it pulls in) moves out of core `dependencies` into an opt-in
  `sentence-transformers` extra: `pip install text-classifier[sentence-transformers]`.
  Plain `pip install text-classifier` stays torch-free and trains/infers with
  `--encoder-kind tfidf` or `--encoder-kind hashing`. The default encoder kind
  stays `sentence-transformers` for out-of-the-box quality; selecting it
  without the extra installed now raises a clear, actionable `ImportError`
  (pointing at the extra and the torch-free alternatives) instead of failing
  deep in the pipeline. `requirements.lock` is regenerated with
  `--extra sentence-transformers --python-platform x86_64-unknown-linux-gnu` so
  the air-gapped bundle is unaffected. CI gained two jobs: one asserting the
  core install has no torch and fails clearly on the sentence-transformers
  kind, one asserting the extra actually installs torch and the suite stays
  green with it present.
- **BM25 at scale: bounded memory and throughput (T32)** — four independent
  fixes to the lexical retrieval path:
  - **Tokenize/build once per training run, not once per fold.** The example
    corpus is tokenized once (`BM25Index.tokenize_corpus` + `fit_from_counts`,
    row-sliced per fold — IDF/length-norm are legitimately fold-local, only
    the tokenization is shared) and the class-description BM25 index is built
    once and reused verbatim (it is never row-sliced, so nothing about the
    per-fold concern applies to it). `bm25_token_kwargs` that prune vocabulary
    by corpus statistics (`min_df`/`max_df`/`max_features`) fall back to the
    ordinary per-fold path automatically — full-corpus and per-fold
    vocabularies genuinely differ then.
  - **`top_k` never densifies.** The `(chunk, n_docs)` dense block used to be
    materialized and then argpartitioned; the sparse `Qbin @ Wt` product's
    positive-only explicit nonzeros now go straight through a vectorized
    sparse row-top-k (one lexsort, no per-row Python loop, no memory blow-up
    at scale).
  - **`RetrievalConfig.bm25_max_df_ratio`** (opt-in, `None` by default):
    drops terms above a document-frequency ratio before building the weight
    matrix, shrinking it for near-zero ranking cost — the one knob here that
    can change scores, so it is opt-in and persisted in `meta.json`.
  - **`RetrievalConfig.bm25_max_block_elems`** (opt-in, `None` by default):
    `BM25Index.score_matrix` (the small class-description path; the example
    pool must go through `top_k`) now raises rather than silently allocating
    a block over the configured cap.
  Byte-identical on the default config (empty `bm25_token_kwargs`,
  `bm25_max_df_ratio=None`, `bm25_max_block_elems=None`); legacy model
  directories load unchanged via `.get`-based defaults on the new persisted
  fields.
- **Encode the corpus once, not once per fold (T88)** — on the shared-encoder
  training path, `_build_oof` now encodes the full example pool and every
  class description once per run and slices per fold
  (`DenseRetrieverAdapter.build_from_embeddings`), instead of paying
  `encode_documents` again for every fold; `_build_deployment_index` reuses
  the same cached embeddings rather than encoding a second time. At
  `n_folds=5` this cuts document-encode work from `5n + 6C` to `n + C` —
  roughly a 5x reduction on the dominant cost of a sentence-transformer
  training run, and it compounds directly with `n_folds`. `build()` still
  encodes internally and delegates to the new classmethod, so every existing
  caller is unaffected. The per-fold-encoder path (`use_per_fold_encoder`) and
  corpus-dependent encoders (e.g. TF-IDF, which must refit per fold to stay
  leakage-free) are untouched — the cache only engages for a frozen, shared
  encoder. No feature value changes: the OOF frame, fusion model, thresholds
  and evaluation are unaffected on a fixed corpus.
- **Feature dependency graph + demand-driven computation (T87)** — the
  assembler now computes only the columns a caller actually requests
  (`FeatureAssembler.assemble(..., requested=...)`), resolved through a
  declared dependency graph (`domain.FEATURE_DEPS` / `feature_closure`)
  instead of unconditionally building the full ~36-column schema every call.
  Training and scoring (`predict`, `predict_topk`, fit) request
  `fusion_feature_names(...)` — the model's own columns; `explain`,
  `explain_records`, `signal_report` and the ablation report request
  `composed_feature_names(...)` — everything, because they read core columns
  by name. The candidate mask is never pruned (every signal that feeds it
  still runs on every call); what's skipped is downstream leaf work nothing
  reads — the `rank_*`/`margin_*`/`norm_*`/`n_signal_agreement` sorts and
  partitions, and, sharpest of all, a custom `FeatureProvider`'s `compute()`
  when every one of its declared columns is dropped (previously it ran
  unconditionally, the exact cost `drop_features` claimed to avoid for a
  provider calling an external service or a reranker). **Behaviour change,
  accepted:** a model trained with `FusionConfig.drop_features` now gets a
  correspondingly narrower `signal_report` on its training out-of-fold data
  (it names the skipped signals rather than reporting on columns that were
  never assembled); diagnostics computed fresh at inference time
  (`explain`/`explain_records`/`importance_report`) are unaffected, since they
  always request the full schema. Pruning is value-preserving by construction
  and fuzz-tested: for any requested subset, every surviving column's value is
  identical to the unpruned computation. Empty `drop_features` (the default)
  is byte-for-byte unchanged.
- **Retrain-based feature ablation (T82)** — `FusionConfig.drop_features` (plus
  `--drop-features` on the train CLI) withholds named columns from the fusion
  model, and `text-classifier-retrain-ablate` trains each arm plus a paired
  no-drop baseline once per seed to report the effect with error bars. This is
  the counterpart to T40's masking ablation: masking a column on an
  already-trained model answers "what if this signal fails at inference?", while
  training without it answers "should this column be in the schema?" — a model
  fitted with a column has splits on it either way. Deltas are paired by seed
  (removing the fold-split variance both arms saw), reported against
  `accepted_correct` (coverage x accuracy, which does not move when the tuned
  threshold slides without the model changing), and turned into an
  `earns_place` / `redundant` / `inconclusive` verdict only when the effect
  exceeds the spread of its own per-seed differences. Dropping narrows the model
  only: the assembler still computes every column, so `explain`, `signal_report`
  and the masking ablation keep working on a subset-trained model. The drop list
  rides in `meta.json` and is re-applied at load, so inference rebuilds the exact
  column list the model was fitted on. Empty (the default) is byte-for-byte the
  existing behaviour.

- **Competition features for the fusion model (T81)** — eight new core columns
  (`FEATURE_NAMES` grows 28 → 36) giving the pointwise fusion model information
  about how a candidate compares to the rest of *its own query*, which it
  previously could not infer. `margin_d_desc`, `margin_d_proto`, `margin_d_knn`,
  `margin_b_desc`, `margin_b_knn` hold each candidate's signal value minus the
  best *other* candidate's value for that signal — positive only for the
  signal's leader, where it is the top1−top2 gap, and negative elsewhere as a
  deficit behind the leader. `q_gap_d_desc`, `q_gap_d_knn`, `q_gap_b_desc` carry
  that top1−top2 gap as a per-query column, so trailing candidates also see how
  contested the lead is. Previously a leader at `0.85` over a `0.84` rival and
  one at `0.85` over `0.40` were identical in every column — one a coin flip,
  the other decided — which matters most for calibrated abstention, where
  top1−top2 is the classic confidence signal. `NaN` discipline is preserved
  throughout: a signal that did not retrieve a candidate yields a `NaN` margin
  and does not compete for the top-2, and a candidate with no rival at all gets
  `NaN` (undefined) rather than `0.0` (a tie). Pure transforms of the `(b, C)`
  signal matrices already computed per chunk (`_row_margin` in
  `application/features.py`, top-2 by `argpartition`): no new retrieval, no new
  persisted state, and no leakage surface. **Effect not established:** a single
  run showed +1.3pp accuracy-on-accepted for the hashing encoder, but the seeded
  retrain-ablation added in T82 puts the benchmark's own seed-to-seed spread at
  +/-1.6pp and finds dropping all eight columns indistinguishable from keeping
  them on both encoders. The columns are retained on the design argument (a
  pointwise model cannot otherwise see the top1-top2 gap that drives abstention),
  not on a measured gain; the offline benchmark is too small to resolve
  feature-level effects below ~2-3pp, so the question is open pending a run on
  real data.

- **Best-epoch selection for encoder fine-tuning (T80)** — a multi-epoch
  fine-tune no longer returns whatever the last epoch happened to produce. With
  `encoder.train_epochs > 1`, a stratified `encoder.train_holdout_ratio`
  (default `0.1`) of the fine-tuning items is withheld from the gradient updates
  and re-scored after every epoch; the epoch scoring best on
  `encoder.train_select_metric` is the one returned and saved.
  `encoder.train_early_stopping_patience` stops the run once the metric stalls
  (default `0` = run every epoch), and `encoder.train_select_min_delta` sets how
  much an epoch must improve by to count. New train-CLI flags:
  `--encoder-epochs` (the epoch count was previously reachable only via
  `--config`), `--encoder-epoch-holdout`, `--encoder-select-metric`,
  `--encoder-patience`. Selectable metrics (`domain/services.py`:
  `ENCODER_SELECTION_METRICS`, computed by `encoder_retrieval_metrics`) are
  `desc_acc@1` (default), `desc_mrr`, `desc_pos_sim`, and `knn_acc@1`. The
  per-epoch table travels with the encoder and is written to
  `<model_dir>/encoder/encoder_training.json` alongside which epoch won.
  Decision logic lives in the framework-free `EpochSelectionPolicy` (ties go to
  the earlier epoch); the per-epoch loop is `EncoderEpochTracker`, driven by one
  evaluator call per epoch from `SentenceTransformer.fit`. Leakage-neutral: the
  holdout comes out of the caller's own items (in the out-of-fold loop, one
  fold's training rows), so withheld-from-the-gradient is still in-fold, and the
  fusion model's rows are untouched. Defaults are behaviour-preserving —
  `train_epochs` is still `1`, where there is nothing to select between; a
  holdout too small to rank epochs by (< 4 items) disables selection with a
  warning rather than picking on noise.
- **Add classes/examples to a deployed model without retraining (T68)** — a
  `text-classifier-update` console script (+ `application/updating.py::update`)
  that rebuilds only the cheap, class-indexed retrieval state (dense
  prototypes/description embeddings, BM25) while reusing the trained fusion
  model and calibrator verbatim — the fusion model is class-agnostic by
  construction (every feature is a per-candidate retrieval signal, not a
  per-class weight). `--classes` is the full taxonomy (new keys appended,
  edited descriptions re-embedded; a file missing an existing key is rejected
  — update never removes or reorders a class). `--items` adds labeled
  examples for a new or existing class: only the new texts are re-encoded
  (old example embeddings are reused as-is), while the BM25 example index is
  refit over the merged corpus (its IDF is corpus-global, so it can't be
  updated incrementally) — this needs the original training corpus, which
  `text-classifier-train` now persists as `corpus.jsonl.gz` by default
  (`--no-store-corpus` to opt out; `--base-items` supplies it for a dir that
  predates the flag). Without `--tune-with`, thresholds are left as-is and the
  persisted `evaluation.json`/`model_card.md` are carried forward marked
  stale; with it, `retune` (T66) runs in the same step and reports candidate
  recall for the newly added classes. `meta.json` gains an `updates`
  provenance list; `--in-place` overwrites `--model` instead of writing a new
  directory.
- **Re-tune the operating point without retraining (T66)** — a
  `text-classifier-tune` console script (+ `application/tuning.py::retune`) that
  refits the calibrator and re-tunes the global + per-class abstention
  thresholds against a fresh labeled set and a chosen `--target-precision`,
  reusing the model's existing encoder, retrieval indices, and fusion model
  verbatim. Updates `calibrator.pkl` and `meta.json`'s abstention block in
  place (recording a `retunes` provenance entry) and writes a fresh
  `evaluation.json`/`model_card.md`; `--dry-run` previews the new coverage/
  accuracy/thresholds and writes nothing. The threshold-tuning logic is shared
  with `TrainingPipeline` via a new `fit_calibration_and_abstention` helper, not
  duplicated. The tune set must be disjoint from the training data — an
  overlapping item retrieves itself as a perfect match and inflates its own
  confidence — so the CLI warns (a best-effort embedding-similarity proxy; see
  README) when a tune-set item looks like a near-duplicate of an indexed
  training example.
- **Prediction explanations for reviewers (T69)** — answer "why did it call this
  that, and how close was it to the threshold" from one record, built from a
  single feature pass (the plain `predict` path is untouched):
  - `InferencePipeline.explain_records(texts, top_k=3, include_contributions=False)`
    returns a JSON-clean payload per item: the decision plus the abstention
    `threshold_applied`/`threshold_scope`, each top candidate's per-signal
    `features` (`NaN` → `null`), which signals ranked it first (`signals_top1`),
    the matched class `description`, and the nearest dense/lexical example
    neighbors (`{label_key, score}`; neighbor *texts* await a persisted corpus, so
    `texts_available` is `False`).
  - **Optional per-feature SHAP contributions** via a new additive
    `FusionModel.predict_contribs(X) -> Optional[np.ndarray]` port method
    (default `None`; implemented for the XGBoost and LightGBM backends, `None` for
    the XGBRanker whose isotonic head breaks additivity). Each row sums to the raw
    margin; surfaced per candidate as `contributions` with
    `contributions_space: "raw_margin"`.
  - Infer CLI: `--explain-json PATH` writes the payloads as JSONL; add
    `--explain-contribs` to include the SHAP contributions.
- **Per-signal insight for data scientists** — see the individual retrieval
  signals' scores *before* the fusion model, and how each technique performs
  alone on your data:
  - `InferencePipeline.explain(texts, top_k=None)` returns the full
    per-(item, candidate) table `predict` computes and then discards: one row per
    candidate class with every raw signal feature plus the calibrated `conf`,
    ranked per item. `NaN` stays "this signal did not retrieve this class"
    (distinct from a true 0), exactly as the fusion model sees it. The infer CLI
    exposes it as `--explain PATH` (a CSV sidecar, bounded by `--top-k`).
  - A **per-signal diagnostics report** (`application/signal_report.py`) computed
    over the leakage-free out-of-fold rows: each signal's standalone top-1
    accuracy, how often it fires, its precision when it fires, and how much the
    signals agree. It is persisted into `evaluation.json` (key `signal_report`)
    and summarized in `model_card.md` at train time, and printed / persisted by
    the `eval` CLI for a labeled set — the evidence for which techniques carry a
    given dataset. No model internals, no change to any existing output.
- **Pluggable custom fusion features** via a `FeatureProvider` port
  (`config.features.providers`): contribute columns beyond the built-in ~28 (text
  length, a domain lexicon hit, an external score) that reach the fusion model at
  **train and inference in the identical order**. The effective schema is composed
  at runtime (core columns + each provider's, in order) and persisted into
  `meta.json`, so a loaded model rebuilds the exact column order. Providers with
  training-derived state are fit **per fold on out-of-fold rows** (leakage-free,
  like prototypes/indices) and persist portable artifacts that run air-gapped with
  no labels; "did not fire" is emitted as `NaN` (XGBoost missing), never a true 0.
  Ships one sample provider, `class-keyword` (per-class learned keyword overlap).
  With no providers configured the schema and outputs are byte-for-byte unchanged.
- Widen a trained model's label space **without retraining**
  (`DeployedArtifacts.with_added_classes` / `InferencePipeline.with_added_classes`):
  add classes that appear after a model ships, or evaluate against a label space
  larger than the training one. New classes are appended (existing class indices
  are preserved, so the trained fusion model/calibrator are reused verbatim);
  only the class-indexed retrieval state grows (dense description embeddings +
  the description BM25, refit for corpus-global IDF). Added classes are
  *description-only* — retrievable from their description, but with no example
  support they draw low calibrated confidence and typically abstain under a
  precision-tuned threshold until seeded and retrained. The `eval` CLI gains
  `--classes` to score a labeled set whose labels exceed the trained taxonomy.
- `--config`/`--dump-config` on the train CLI: every `PipelineConfig` field
  (fusion kind + hyperparameters, calibration kind, BM25/encoder kwargs, ...)
  is now reachable from the command line via a JSON file, with precedence
  `defaults < --config < explicit flags`. A trained model's `meta.json`
  `config` block is directly reusable as `--config` input.
- `--top-k` on the infer CLI and `InferencePipeline.predict_topk`:
  `Prediction.runner_up_key` is populated, and up to `k` ranked
  `(class_key, confidence)` suggestions are available per item for a human
  review queue, at no extra scoring cost over the existing top-1 pass.
- `CHANGELOG.md`, `CONTRIBUTING.md`, `SECURITY.md`, and GitHub issue/PR
  templates.
- Single-sourced package version (`text_classifier/_version.py`); `pyproject.toml`
  reads it via `[tool.setuptools.dynamic]` instead of duplicating the string.
- `--bm25-stop-words` on the train CLI to opt into stopword filtering (e.g.
  `english`) or explicitly disable it (`none`).
- Asymmetric query/document encoding for instruction-tuned embedding models
  (E5/BGE/GTE...): `EncoderConfig` gains `query_prompt`/`document_prompt`
  (literal prefixes), `query_prompt_name`/`document_prompt_name` (model-card
  prompts), and `encode_kwargs` (merged into every
  `SentenceTransformer.encode` call; `normalize_embeddings`/`convert_to_numpy`
  stay forced so embeddings remain L2-normalized). The `TextEncoder` port
  gains default `encode_queries`/`encode_documents` methods and the pipelines
  route every encode call by role — existing encoders and configs are
  unaffected (defaults are byte-identical to previous behavior).

### Changed
- **Model directories trained before T81 must be retrained.** The core feature
  schema grew from 28 to 36 columns, so a `meta.json` written before it no
  longer matches the schema the code composes and loading raises the existing
  `_check_feature_schema` error. That is the guard working as designed — the
  alternative is silently feeding XGBoost mislabelled columns.
- **Behavior change:** `RetrievalConfig.bm25_token_kwargs` now defaults to
  `{}` (no stopword removal) instead of `{"stop_words": "english"}`. BM25
  silently applied English stopword removal to every corpus regardless of
  language; that filter is now opt-in via `--bm25-stop-words english` or
  `--config`. This shifts BM25 scores slightly for English corpora trained
  from now on; retraining existing model directories is not required (the
  fitted vectorizer is already persisted, so only new training runs are
  affected).

## [0.1.0] - 2026-07

Initial release.

### Added
- Hybrid retrieval-fusion architecture: five retrieval signals (dense kNN,
  BM25, description similarity, ...) assembled into per-(item, candidate)
  features, fused by a pointwise XGBoost model, isotonic-calibrated, with a
  tuned abstention threshold for target-precision serving.
- Hexagonal/DDD layering (`domain` / `infrastructure` / `application`) with a
  pluggable component registry: swappable encoder (sentence-transformers,
  TF-IDF, hashing), fusion model (XGBoost, LightGBM, XGBRanker), and
  calibrator (isotonic, Platt, beta) behind ports, selected purely by config.
- Leakage-free training and calibration: out-of-fold scoring throughout, with
  a regression test pinning the invariant.
- Console scripts `text-classifier-train`, `text-classifier-infer`,
  `text-classifier-eval`, each installed via `pip install .`.
- Persisted `evaluation.json` and `model_card.md` per trained model directory;
  package version recorded in `meta.json` and checked on load.
- Deterministic training (seeded fusion backends, config validation at
  pipeline entry) and an offline quality-regression benchmark with metric
  floors, run in CI.
- Hash-locked dependency pins (`requirements.lock`) for air-gapped,
  reproducible installs.
- ruff + mypy + pre-commit gates; a type-clean, `py.typed` package.
