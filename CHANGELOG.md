# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [Semantic Versioning](https://semver.org/): the version
lives in one place, `text_classifier/_version.py` (see `RELEASING.md`).

## [Unreleased]

### Added
- **Training-data cleanup round (`--cleanup`, config section `cleanup`).** Before
  training, an extra out-of-fold pass flags suspicious training rows: `junk`
  (placeholders, no letters) is dropped; exact-text `label_conflict`s take the
  majority label (ties kept); `likely_mislabel` (a fusion model cross-fitted by
  item is confident in another class, gives the label <= 5%, and a kNN signal
  agrees), `ambiguous` and `not_shortlisted` rows are flagged. Every flagged row
  is written to `suspicious_rows.csv` (UTF-8 with BOM, most actionable first)
  with the suggested label, the scores and the reason; `evaluation.json` and the
  model card get a summary. Likely mislabels are **kept by default**
  (`mislabel_action = "keep"`): on 20k Hebrew COICOP items, dropping or
  relabelling them moved accuracy on an untouched test set by -0.23 to -0.07
  points (within noise), because the flagged rows mix real label errors with
  hard-but-correct rows. No action may push a class below `n_folds` rows. Only
  training rows are touched; external validation/test sets never are.
- **Training progress (`--progress`, `training.progress`).** One log line per
  stage (each fold, the final model, fusion + calibration + evaluation) with the
  elapsed time, the estimated time remaining and the estimated finish time; an
  overall progress bar; per-step bars inside the encoder, cross-encoder and SPLADE
  fine-tunes; and a live `progress.json` in the output directory, rewritten after
  every stage and marked `done` or `failed` at the end. The estimate averages the
  fold-like stages (the final model build costs about one fold); the last step is
  reported as not estimated. Off by default; display only.
- **`evaluation.json` → `signal_contributions` and a model-card section "Which
  signals drive the fusion model's decisions".** Share of the fusion model's SHAP
  attribution per signal (every configured signal, plus `cross-signal` columns),
  the most and least used features, and the full per-feature table, on the
  held-out rows.
- **`evaluation.json` → `manifest.models` and a model-card "Models" table.** Per
  component: base model, whether it was trained on this run's data, and where the
  deployed weights are. The config alone names only the base checkpoint, e.g.
  `cross_encoder.model_name` after a fine-tune.

### Fixed
- **The candidate shortlist could contain every class.** A signal's top-n kept
  every class tied with the n-th score; when a signal scores most classes
  identically (measured: TF-IDF item vectors vs label-name descriptions, cosine
  exactly 0 for 100% of pairs) that admitted all classes, so a 55-class model
  scored 55 candidates per item. New `candidate_tie_cap` (default 2.0): ties
  are kept unless they would admit more than `2 x candidate_top_n`, then the
  signal is cut to exactly `candidate_top_n` by class order. Measured on 20k
  Hebrew COICOP items: 55 -> 19 candidates per item, training 267 s -> 92 s
  (with the change below), candidate recall 100% -> 99.3%, accuracy 90.45% ->
  90.57%. Models saved without the setting keep uncapped shortlists on load.
- **Class prototypes were computed with an n x dim scatter.** Now one sparse
  matrix product on the numpy backend: 39x faster on a 20k x 3,000 matrix,
  identical results.
- **Per-signal diagnostics covered only dense + BM25.** `signal_report` (and the
  model card) now include SPLADE and every cross-encoder document (marked as round
  two: its pick is the best within the shortlist), and discover any other signal by
  its `is_<name>_top1` column. The agreement line reads against the real number of
  voting signals instead of a hardcoded 5.
- **XGBoost on GPU warned "Falling back to prediction using DMatrix due to
  mismatched devices" on every prediction.** Features were handed over as host
  arrays to a CUDA booster. Prediction now moves them to the GPU first when CuPy is
  installed, and otherwise builds the `DMatrix` explicitly (the same data path,
  without the mismatch). CPU prediction is unchanged and numerically identical.
- **Ties in calibrated confidence picked the alphabetically-first class.**
  Isotonic calibration is a step function, so an item's top candidates often share
  one calibrated value; `top_key`, ranks and top-k then fell back to class order.
  They are now ordered by the raw fusion score. Measured on the Hebrew COICOP data:
  0.5% of items changed, accuracy 89.79% → 89.96%.
- **Loading a model with the cross-encoder signal no longer loads the base
  model.** The feature-schema check in `ArtifactRepository.load` built every
  signal provider just to read its column names, and for
  `retrieval.cross_encoder.kind = "cross-encoder"` that loaded
  `cross_encoder.model_name` (the untrained base model) on top of the
  fine-tuned one in `signals/cross-encoder/`. Predictions were unaffected, but
  each load paid for a second model and logged Hugging Face warnings pointing at
  the base model's path ("Some weights of BertForSequenceClassification were not
  initialized from the model checkpoint at ..."), and a host without the base
  model could not load at all. `SignalProviderSpec` gained an optional `schema`
  builder for providers whose `build` is expensive; the cross-encoder uses it.

## [0.3.0] - 2026-11-02

### Added
- **Learned-sparse (SPLADE) retrieval signal (T94).** Add `"splade"` to
  `signals` to get a sixth, round-one signal: a masked-language model's
  per-word weights (`retrieval.splade.kind = "splade"`, any checkpoint with its
  MLM head, e.g. a Hebrew BERT) scored as a sparse dot product against class
  descriptions (`s_desc_*`) and the example pool (`s_knn_*`). It joins candidate
  selection, so unlike the cross-encoder it can raise candidate recall.
  `retrieval.splade.train` fine-tunes a fresh copy per fold (contrastive loss +
  FLOPS sparsity penalty), leakage-free. Weights ship in the model dir as
  safetensors. `kind = "hashing"` is the offline stand-in. Off by default; the
  default config is unchanged. New `splade` install extra. `SignalProvider` gained
  an optional `fit` hook so stateful signals are fitted per fold.

### Changed
- **`n_signal_agreement` now counts every shortlist-building signal (T94).** It
  used to require exactly the five dense + BM25 picks and vanished otherwise; now
  every round-one signal with a top-1 pick votes (SPLADE included; the
  cross-encoder does not), and the column is kept whenever at least two round-one
  signals are configured. Values for the default dense + BM25 config are
  unchanged, so existing models load and score identically.
- **New feature `n_top1_votes`: how many shortlist-building signals picked
  *this* candidate as their top class (T94).** The per-candidate counterpart of
  `n_signal_agreement` (which is one value per item). Same voters, same
  presence rule. Models saved before it existed load as if trained with it in
  `fusion.drop_features` (which is how they were trained) and score unchanged.

## [0.2.0] - 2026-11-02

### Added
- **A real cross-encoder backend for the rerank signal, with optional per-fold
  fine-tuning (T33 phase 3).** `retrieval.cross_encoder.kind = "cross-encoder"`
  wraps a sentence-transformers `CrossEncoder` (`model_name` required; no default,
  since the right model depends on your language; extra constructor args via
  `params`, e.g. `max_length`). Scores are the model's raw logits, forced
  explicitly because `CrossEncoder.predict`'s default activation differs between
  versions (5.6+ applies a sigmoid, 5.2 does not). The weights are saved into the
  model dir as safetensors and reloaded from there, so deployed models work
  offline. `device` follows the encoder's auto-detection.

  With `cross_encoder.train = true`, a fresh copy of the model is fine-tuned per
  fold on that fold's training items only, and once more on all items for the
  shipped model, so no item is scored by a model that saw its label. Training
  lists are mined from each item's own round-one shortlist, computed leave-one-out
  (true class positive, other shortlisted classes negative, hardest first by
  `prescore_node`), rendered with `train_document` (default `pos`). Losses:
  `binary_cross_entropy` (positives up-weighted), `lambda`, `listnet`. New fields:
  `train`, `train_document`, `train_negatives`, `train_loss`, `train_epochs`,
  `train_batch_size`, `learning_rate`, `weight_decay`, `warmup_ratio`. Validation
  rejects training on the leave-one-out path (`n_folds = 1`, which would leak),
  without the signal configured, or with the token-overlap stand-in. Fine-tuning
  needs sentence-transformers >= 4.0. Tested offline on 6.2.0, 5.6.0 and 5.2.0.

  Multi-epoch fine-tuning selects its best epoch, as the encoder does: with
  `train_epochs > 1` a class-stratified `train_holdout_ratio` (default 0.1) of the
  training lists is held out and scored after each epoch by a
  `CrossEncoderRerankingEvaluator`; the Trainer reloads the best epoch on
  `train_select_metric` (`mrr`/`ndcg`/`map`), with optional early stopping
  (`train_early_stopping_patience`, `train_select_min_delta`). The shipped model's
  per-epoch table is saved as `signals/cross-encoder/cross_encoder_training.json`.
  Off at the default single epoch.

  Verified end to end with a real multilingual cross-encoder
  (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`) on a Hebrew sample of the COICOP
  example data, on sentence-transformers 5.2.0 + transformers 4.57.3 and on 6.2.0 +
  5.22: training, saving, offline reload with identical predictions, and every
  post-training CLI (evaluate, tune, importance, update with new examples or a new
  class -- a class added after training is reranked like any other).
- **Encoder fine-tune and model knobs, all defaulting to the old behaviour.**
  `encoder.learning_rate` (`2e-5`) and `encoder.weight_decay` (`0.01`) — the
  values `model.fit` used — are now configurable. `encoder.max_seq_length` sets the
  truncation length (a model property that `encoder.params` could not set, so
  longer texts were silently cut at the model default); it is persisted with the
  saved model. `encoder.pooling_mode` (`mean`/`cls`/`max`/`lasttoken`/
  `weightedmean`/`mean_sqrt_len_tokens`) rebuilds the model as its Transformer plus
  a fresh `Pooling`, warning when that replaces the model's own pooling or drops a
  post-pooling `Dense`/`Normalize` module; the model's prompts are carried over.
  Overrides apply when an encoder is built or fine-tuned and are not re-applied
  when a saved model dir is reloaded. New train-CLI flags: `--encoder-lr`,
  `--encoder-weight-decay`, `--encoder-max-seq-length`, `--encoder-pooling`. The
  loss choice and scale (`train_loss`, `train_loss_params={"scale": ...}`) were
  already configurable.
- **Cross-encoder rerank signal (T33, phase 1) — opt-in, off by default.** Adds a
  second signal *round* to `FeatureAssembler`: a `SignalProvider` may now declare
  `needs_candidates = True` and run after candidate selection, receiving the
  shortlist (`SignalContext.candidates`) instead of scoring all `C` classes. That
  is what a reranker needs and what the assembler could not express — every
  provider previously ran before the mask existed. A second-round provider must
  declare no `candidate_features` (it cannot select what it consumes; the
  assembler rejects the combination), and that restriction is what makes it safe
  to skip such a provider entirely when every one of its columns is pruned — so
  dropping the `ce_*` columns costs *zero* cross-encoder calls, asserted as a
  call count.

  Enabled with `"cross-encoder"` in `PipelineConfig.signals`. It is the first
  consumer of the structured taxonomy fields: each candidate is scored against
  documents rendered from `core`/`examples`/`inclusions`/`exclusions`/`siblings`
  views, with the entry most like the query selected per item. The default plan
  scores positive identity and negative evidence (`exclusions` +
  `sibling_distinctions`) **separately**, plus an explicit `ce_pos_neg_gap`
  difference column — a stock relevance cross-encoder cannot tell that a span is
  a prohibition, so composing negatives into one document would invert the
  signal; an instructed LLM judge or a purpose-trained model can, and reaching
  that is a config change (`retrieval.cross_encoder.documents`), not a rewrite.
  Backends plug in behind the new `PairwiseReranker` port; `token-overlap` (no
  model, no network, deterministic) is the only one registered so far.

  A class with no exclusions yields `NaN`, never `0.0` — an absent view is
  absent, not evidence of a poor match (the `NaN` invariant, at the text layer).
  With the signal off, output, schema and on-disk layout are byte-for-byte
  unchanged; no torch is imported either way.

### Fixed
- **Fine-tuning no longer prints a `{'train_runtime': ...}` line per run.** With
  progress bars off, transformers' `PrinterCallback` printed the end-of-training
  summary to stdout even with logging disabled; it is removed from both the encoder
  and the cross-encoder trainers, which report through `logging` instead.
- **Possible segfault when training on macOS.** On Macs using Homebrew's libomp,
  importing torch before xgboost makes xgboost's first `fit()` crash
  (`OMP: Error #179`). A normal training run does exactly that, because the
  sentence-transformers encoder loads torch before the fusion model loads xgboost;
  only the test suite was protected. The package now imports xgboost when it loads.
  This can't help if your own code imports torch before `text_classifier`. Linux is
  unaffected (checked on two torch/xgboost version pairs).
- **Rank features no longer depend on the CPU.** `ArrayOps.argsort` used numpy's
  default sort, which is neither stable nor portable: numpy 2.x dispatches it to
  CPU-specific SIMD kernels (AVX2/AVX-512, NEON), so tied scores — common with
  BM25 — came back in a machine-dependent order and the `rank_*` features of the
  same model differed between the training host and the deployment host. The
  port's `argsort` is now stable on both backends (ties go to the lower class
  index), and the port contract says so. Values change only where scores tie.
- **CI was red on every run, `main` included; it is green again.** Three
  independent causes:
  - The golden feature-frame test compared float32 dense similarities
    bit-exactly against fixtures generated on another machine; BLAS libraries sum
    in different orders, so CI always failed by an ulp. Discrete columns (ranks,
    flags, counts) are still compared exactly — the stable sort above is what made
    them portable — and continuous columns to within `1e-6`.
  - `package-smoke` and `quality-benchmark` installed a hand-copied dependency
    list with `--no-deps`, which had drifted (no `tqdm`), so the package failed to
    import. Both now install the package's own declared (torch-free) dependencies.
  - mypy (28 errors): missing stubs for the optional `torch`/`transformers`/`tqdm`
    are now ignored like the other ML libraries; the numpy 2.x stubs' typing of
    `savez_compressed(**arrays)` is satisfied; stale ignores are removed; and the
    `Optional` retriever handling is made explicit (below).
- **Clear errors instead of `AttributeError`s for inconsistent retriever state.**
  A signal provider asked to compute with no index built (e.g. a lexical provider
  when no BM25 index exists) now raises a `ValueError` naming the signal. Saving,
  updating or re-tuning a model whose custom retriever kind lacks a method that
  operation needs (`to_state`, `with_added_classes`, ...) now raises a
  `TypeError` naming the kind, the operation and the missing method. The check is
  duck-typed, so custom kinds that do implement those methods keep working.
  `DeployedArtifacts.dense`/`lexical` are typed against the retrieval ports.

- **`_topn_mask` (candidate selection) reached past the `ArrayOps` port and
  raised under a torch backend.** It called `M.astype` (which torch tensors do
  not have) and `np.partition` on what may be a tensor, then compared that numpy
  result against the tensor it came from. Both are now port calls (`asarray`,
  `topk` — the nth-best value is the same threshold `partition`'s pivot gave).
  Latent rather than user-visible: `TrainingPipeline._assembler_ops` pins the
  whole feature-assembly layer to numpy until T86, so nothing reached it. Numpy
  output is unchanged — verified bit-identical over 6000 randomized cases
  covering NaN, exact ties, all-NaN rows, all-non-positive rows and `k > C`, on
  top of the golden-output benchmarks.
- **`fusion.drop_features` could not name a provider column.** `TrainingPipeline`
  validated the drop list at construction against the *core-only* schema, before
  any `FeatureProvider`/`SignalProvider` existed, so dropping a custom provider's
  column raised "cannot drop unknown feature(s)" before the run started — even
  though the per-fold and deployment paths both compute the correct composed
  schema. Pre-existing since custom providers landed; surfaced by T33's `ce_*`
  columns. The eager computation is removed (building providers early just to
  read `column_names()` would load a real cross-encoder to validate a string);
  the real schema is still computed and validated before any consumer reads it.

### Changed
- **Encoder fine-tuning now runs through `SentenceTransformerTrainer`, and the
  `sentence-transformers` extra requires `>=3.0` plus `datasets`/`accelerate`.**
  `model.fit` and the hand-written epoch loop are gone. The batch sampler is
  `BatchSamplers.NO_DUPLICATES` (same job as the old `NoDuplicatesDataLoader`),
  each epoch is scored by an `InformationRetrievalEvaluator` over the stratified
  holdout (`desc_acc@1` is its `accuracy@1`, `desc_mrr` its full-rank `mrr@C`,
  `knn_acc@1` a second evaluator with the fine-tuning pool as the corpus), and
  best-epoch selection is the Trainer's own `load_best_model_at_end`, with
  `train_early_stopping_patience`/`train_select_min_delta` mapped onto
  `EarlyStoppingCallback`. Ties still go to the earlier epoch. The per-epoch table
  in `encoder_training.json` keeps its schema. Query/document prompts are passed
  to the evaluator so what is scored is what the deployed encoder computes.
  Checkpoints go to a temporary directory (weights only, `save_only_model`,
  nothing reported to any tracker) and are removed; the saved model directory is
  still written only by `model.save()`, so it stays pickle-free. Training is now
  seeded by the Trainer (default seed), so a rerun reproduces.
  - **`pyproject.toml`:** the extra is `sentence-transformers[train]>=3.0`
    (was `>=2.2`). From sentence-transformers 4.x, `datasets` and `accelerate` sit
    behind a `train` extra, so a locked install previously lacked them; the
    hash-locked `requirements.lock` is regenerated and gains 18 packages (no
    existing pin moved).
  - **Compatibility, tested:** sentence-transformers 6.2.0, 5.6.0 (the locked
    version) and 5.2.0, the last on both transformers 4.57 and 5.22. 5.2.0 on
    transformers 5 needed a shim: its Trainer reads `args.save_safetensors` when
    writing a checkpoint, which transformers 5 removed, so every best-epoch
    fine-tune died with an `AttributeError`; the argument is now supplied when
    absent.
  - **Default fine-tuning loss is now `multiple_negatives_ranking`** (one-directional,
    item→description), replacing `multiple_negatives_symmetric_ranking`, which
    sentence-transformers deprecates from 5.6. **This changes what an untouched
    config trains**: the symmetric loss also scored description→item. Set
    `encoder.train_loss="multiple_negatives_symmetric_ranking"` (or
    `--encoder-loss`) to keep the old objective; it remains available. A config
    that already names a loss is unaffected, and `encoder.train_loss` only matters
    when an encoder is fine-tuned, so models already trained are not touched.
  - **Device handling differs:** the Trainer places the model itself.
    `encoder.device="cpu"` still pins to CPU, but a CUDA index (`"cuda:1"`) is not
    honoured and several visible GPUs are all used (`DataParallel`). Both warn;
    use `CUDA_VISIBLE_DEVICES`.
  - **Removed:** `EpochSelectionPolicy`, `encoder_retrieval_metrics`,
    `EncoderEpochTracker` and the `desc_pos_sim` selection metric (the IR evaluator
    has no equivalent, and it was documented as a weak selection target). A config
    naming `desc_pos_sim` now fails validation, listing the valid metrics.

- **Retrieval-index construction extracted into `RetrievalIndexBuilder`
  (`application/indexing.py`)** — `TrainingPipeline` stated the same index-build
  policy twice, once in the out-of-fold fold loop and once in the deployment
  build: the same T88 embedding-cache, T32 A1/A2 tokenization-cache, and T34
  phase-1 built-in-kind gating, spelled out in two places that had to be kept in
  agreement by hand. A backend that changed one and not the other would have
  trained the fusion model against a differently-built index than the one it
  ships with. Both sites now call `build(encoder, rows)` on one builder that owns
  the caches, so "shared once per run" is an object invariant rather than a
  convention held by comments, and `TrainingPipeline` drops ~190 lines and six
  cache fields. No behavioral change: the T88/T32/T89 call-count tests and the
  golden feature vectors are unchanged.
- **The shared encoder is built once per training run, not twice** — the
  out-of-fold loop and the deployment build each called `_load_shared_encoder()`,
  which constructed a fresh encoder every time. For a pretrained backend that was
  a second load of identical weights off disk. It is memoized now; the per-fold
  path still fits one encoder per fold, as intended.
- `AbstentionPolicy.accept` no longer does a Python-level dict lookup per scored
  row (against the package's own "no per-row Python loops on the hot path"
  convention). The new `AbstentionPolicy.thresholds_for` resolves a whole batch
  with a sorted `searchsorted` probe; `threshold_for` is unchanged for the scalar
  case, and out-of-range/sentinel class indices still fall back to the global
  threshold.
- `InferencePipeline`'s five public entry points (`predict`, `predict_topk`,
  `explain`, `importance_report`, `explain_records`) shared an identical 13-line
  encode-and-assemble block that differed only in the requested schema; it is now
  one `_featurize` call each.
- `LabelSpace.unknown_keys` and `_messages.format_preview` replace six
  re-implementations of "reject labels this model does not know" and eight of the
  "show the first ten offenders" message idiom, so every such error now reads the
  same way to an operator.
- `application.evaluation._json_safe` is now public as `json_safe`: it was
  imported under its private name by five modules across two layers.
- CLI presentation helpers (`pct`/`signed_pct`/`num`/`signed_num`,
  `write_json_report`) moved to `cli/_common.py`, deduplicating four copies of the
  percentage formatter and four of the `--output` writer. `retrain-ablate` now
  loads `--config` through the shared `load_pipeline_config` (friendly errors, no
  leaked file handle) and gained `--dump-config` for parity with `train`.
- **Query embeddings are reused instead of recomputed (T89)** — on the
  shared-encoder path (the default), training encoded every item twice: once as
  a document into T88's once-per-run pool cache, and again as a *query* when the
  fold loop reached it. Because the folds' held-out sets partition the item list,
  that was one extra full pass over the corpus per run — flat in `n_folds`, not
  proportional to it, and pure recomputation whenever the encoder treats both
  roles identically (which it does unless a query/document prompt is configured).
  Held-out items now slice the cache. Measured on Apple Silicon (M5/MPS) with
  the default MiniLM encoder over 3,000 items: **55-61% less encoder time**
  (6.9s → 3.1s uniform-length, 10.5s → 4.1s variable-length) and 47-55% less
  out-of-fold wall-clock.
  **On feature parity, read this carefully.** With a deterministic host-side
  encoder the output is bit-identical (all 40 columns, NaN placement included).
  With a real transformer on GPU it is bit-identical for uniform-length inputs
  but *not* for variable-length ones: 12 of 40 continuous columns moved by up to
  ~2e-06 (ordinal `rank_*`/`is_*_top1` columns were unaffected in that run,
  though a near-tie could in principle flip one). The cause is that the old path
  re-encoded held-out items in different *batches* than the pooled pass had, and
  padding changes float reduction order — meaning the pre-T89 pipeline computed
  two slightly different vectors for the same text, one indexed and one used to
  query. Reuse removes that inconsistency rather than introducing one, but the
  numbers are not byte-for-byte what a pre-T89 GPU run produced.
  Reuse is refused whenever it would be wrong: a per-fold fine-tuned encoder
  never populates a cache in the first place (its pre-training embeddings would
  be stale), an asymmetric E5/BGE-style encoder (T28) keeps re-encoding, and a
  custom `TextEncoder` that does not advertise the new `roles_share_encoding`
  capability is left alone rather than assumed symmetric. Override with
  `encoder.reuse_query_embeddings` / `--reuse-query-embeddings`:
  `auto` (default, detect), `always` (force, warns when it overrides a detected
  asymmetry), `never` (byte-for-byte the pre-T89 path).

### Added
- **Loss/objective is a config choice, not a hardcoded default, for both the
  encoder fine-tune and the fusion models.** `EncoderConfig.train_loss`
  (`--encoder-loss`) picks among `domain.services.ENCODER_LOSSES`
  (`multiple_negatives_symmetric_ranking`, the previous hardcoded default;
  `multiple_negatives_ranking`; `cached_multiple_negatives_ranking`) — all
  three train on the same (item, description) pairs `train_encoder` already
  builds, so switching is a config change, not a data-plumbing one. Beyond
  those three, `train_loss` also accepts any other class name under
  `sentence_transformers.losses` (e.g. `"CosineSimilarityLoss"`,
  `"TripletLoss"`) for direct access to the rest of the package's built-in
  losses, resolved by name at fine-tune time; unlike the three aliases those
  are not verified to match the (item, description) *pair* shape this package
  builds (some expect a label, a triplet, or a different batch structure), so
  picking one is the caller's call, and an unresolvable name fails loudly
  with a clear error rather than mis-training silently.
  `FusionConfig.objective` merges into `xgb_params`/`params` as `"objective"`
  for whichever fusion backend is selected (e.g. `binary:hinge` for xgboost,
  `cross_entropy` for lightgbm, `rank:ndcg` for xgb-ranker); unset, each
  backend keeps its previous implicit default (xgboost/lightgbm's own binary
  log-loss, `rank:pairwise` for xgb-ranker). Not validated against a fixed
  list — valid objectives are backend-specific and pluggable third-party
  fusion kinds may support ones this package doesn't know about; an
  unrecognized value surfaces as a clear error from the backend library
  itself. An `"objective"` key already present in `xgb_params`/`params` wins
  over the new field.
- **Device-resident dense retrieval + encoder handoff (T85)** — a torch
  `ArrayOps` backend (`array_backend="torch"`/`"auto"` over T83's crossover)
  keeps the dense index's embeddings and query compute on-device across the
  whole out-of-fold loop instead of re-uploading the corpus on every chunk:
  `SentenceTransformerEncoder.array_backend="torch"` hands back a resident
  tensor instead of forcing a numpy conversion, and `DenseState`'s arrays
  (uploaded once per run, not once per fold) stay there for every query
  afterward. Every public `DenseRetrieverAdapter` method still returns numpy
  at its boundary — `application/features.py`'s kernels aren't
  backend-polymorphic yet (T86), so `FeatureAssembler` always gets a plain
  `NumpyArrayOps` regardless of the run's resolved backend. Registered
  lazily: listing "torch" as a kind never imports it, and a below-crossover
  run never even asks whether torch is installed. Persistence stays numpy
  always — a torch-trained model directory loads on an air-gapped, torch-free
  host unchanged. No new install extra: the existing `sentence-transformers`
  extra already pulls in torch. CPU-only, torch-free, and CUDA-visible-but-
  below-crossover runs are all byte-identical to before. Verified with a
  torch-CPU parity suite (`tests/unit/test_array_ops_torch.py`,
  `tests/integration/test_device_parity.py`) — GPU-specific acceptance
  criteria (transfer byte counts, the VRAM-vs-chunk curve) remain unverified
  pending a CUDA host, same limitation T83 already flagged; see
  `docs/device-policy.md`'s T85 addendum.
- **Multi-threaded BM25 kNN (`RetrievalConfig.bm25_n_jobs`)** — `BM25Index.top_k`'s
  per-chunk sparse mat-mul (`Qbin_chunk @ Wt`) previously ran serially even
  though scipy's sparse `@` releases the GIL during the C-level multiply, so
  on a large example pool (hundreds of thousands of rows) one core sat at
  100% while the rest of the machine idled. `bm25_n_jobs` (default `1`,
  byte-for-byte unchanged; `-1` uses all CPU cores) runs the chunk loop
  across a thread pool instead — no cross-process pickling of the
  `(vocab, n_docs)` weight matrix. Plumbed through
  `LexicalRetrieverAdapter`/persisted in its `to_state`/`from_state`, and
  exposed as `--bm25-n-jobs` on `text-classifier-train`.
- **`PipelineConfig.signals` can now actually drop a built-in signal** —
  previously `signals=["dense"]` (or `["lexical"]`) crashed at `fusion.fit`
  with a `KeyError`: `FEATURE_NAMES` was a fixed ~36-column constant covering
  both built-in signals unconditionally, so excluding one from `signals` never
  actually removed its columns from the schema. `composed_feature_names`/
  `fusion_feature_names` now narrow the schema (via the new
  `core_feature_names`) to whichever of `"dense"`/`"lexical"` are actually
  present in the active `SignalProvider`s, derived from `FEATURE_DEPS` rather
  than a second hardcoded list; `n_signal_agreement` (needs both) drops when
  either is disabled. Training also skips *building* the excluded index
  entirely (no corpus tokenization, no per-fold BM25 fit) rather than building
  one nothing queries — `DeployedArtifacts.lexical` is `Optional` for this.
  `explain()`'s neighbor evidence, `update`, and `retune` all handle the
  missing index. The default `signals=["dense", "lexical"]` is unaffected.
- **Clear error on a feature-schema mismatch, instead of a bare `KeyError`** —
  every place that indexed an assembled frame by the expected feature-column
  list (`_fit_fusion`, `fit_calibration_and_abstention`, `add_confidence`,
  `ablation_report`, `explain_records`'s contributions, `importance_report`)
  now goes through a new `select_feature_columns` (`application/scoring.py`),
  which raises a `ValueError` naming exactly which columns are missing and why
  (a `SignalProvider`/`FeatureProvider` declared a column it didn't produce,
  or `signals`/`drop_features` don't match what actually ran) instead of
  pandas' `KeyError: "[...] not in index"` deep inside `fusion.fit`/`predict`.
- **Progress bars on BM25's CPU-bound stages** — `tqdm` (new core dependency,
  pure Python, no transitive deps) now brackets `BM25Index.tokenize_corpus`
  (corpus tokenization), `fit_from_counts` (weight-matrix construction), and
  `top_k`'s per-chunk query-scoring loop (real fractional progress, since that
  loop is already chunked). Every bar passes `disable=None`, tqdm's "silence
  on a non-TTY" mode, so piped/CI/log output and the test suite are unaffected
  — nothing prints unless stdout is an interactive terminal. Purely observational:
  scoring output is unchanged (verified byte-identical with/without).

## [0.1.1] - 2026-08-05

### Added
- **Per-class calibration behind the `ConfidenceCalibrator` port (T45)** — a
  `PerClassCalibrator` fits a separate inner calibrator (isotonic, platt, or
  beta) per class, falling back to a single global inner calibrator for
  classes whose out-of-fold support is below `min_support`. A raw score of
  0.8 does not mean the same thing for a common class and a rare one; a
  global curve averages the two and is wrong for both, and thresholds
  (`AbstentionPolicy`) already go per-class — calibration was the remaining
  global stage. Selected via `CalibrationConfig(kind="per-class", inner=...,
  min_support=...)`; `ConfidenceCalibrator.fit`/`transform` both gained an
  optional `classes=None` keyword so the existing isotonic/platt/beta
  calibrators are unaffected when it's absent. Persists as a directory (a
  manifest plus one file per class calibrator and one for the global),
  registered in the same registry as the other calibrator kinds. Class-blind
  callers (`classes=None`) reproduce the prior global-only behaviour exactly.
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
