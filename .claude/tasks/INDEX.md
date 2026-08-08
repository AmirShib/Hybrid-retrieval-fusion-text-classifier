# Task backlog

SWE-style tickets for this package. Each `TNN-*.md` is self-contained: read it
top-to-bottom and you have everything needed to do the work.

**Status values:** `todo` · `in-progress` · `in-review` · `done`
When you start a ticket, set its `status:` field (top of the file) and update the table here.
When a ticket reaches `done` and the work is merged: move its file from `.claude/tasks/` into
`.claude/tasks/done/` — the row stays in this table for history.

Priority is top-down. **Tier 8 (T83–T88) is the current top priority** — see
"Tier 8" below. It addresses two structural problems raised 2026-08-04: the
pipeline ping-pongs between GPU and CPU when both the encoder and the fusion
model are on GPU, and feature computation is welded into the assembler so
`drop_features` discards work it already paid for. Scoping it turned up a third
(T88): the corpus is re-encoded once per fold on the default path.

**Phase 0 of the execution order is complete** (T26, T27,
T52, T51, T50 — determinism, config validation, the quality-benchmark net, lint/
type gates, hash-locked pins). **Phase 1 (cheap unlocks) is complete: T74, T65,
T64, T35, T28 all landed 2026-07-05. Tier 1 has one open item (T08, docs).
Tier 2's remaining item is T29 (plugin discovery), which unblocks cross-host
custom backends.** Tier 3 (T30–T35) is the active feature tier; T30 and T35 are done,
T31–T34 remain. Tier 4 remains stubs
except T41/T42/T44 (done). **Tier 6 is the packaging / production-readiness
tier**: T60–T62, T63 (torch-optional install), T64 (release process + project
hygiene), and T65 (top-k suggestions) and T67 (pickle-free artifacts) are done;
T69 remains and T66/T68 are in-review. T63 closed the gap between the README's
air-gapped/torch-free claim and what `pip install .` actually delivers, and
was promoted to a hard T85 prerequisite (Tier 8) — done 2026-08-05.
Tier 7's T74 (`--config`) is done — the cheapest unlock in the backlog; T75
(DataFrame API + streaming) remains.

## Suggested execution order (2026-07 roadmap review)

Ordering logic: correctness-of-measurement first (nothing can be evaluated until
runs are reproducible), then cheap user-facing wins, then the big refactors last —
by which point the safety nets exist. Phases 0 and 3 are internally ordered;
items within the other phases are parallelizable. The one non-negotiable edge:
**T52 lands before T34** — the benchmark floors + golden-output tests are what
make the signal-provider refactor a safe refactor instead of a rewrite-and-pray.

| Phase | Theme | Order |
|-------|-------|-------|
| 0 | Foundations (strictly ordered) — **done 2026-07-02** | T26 → T27 → T52 → T51 → T50 |
| 1 | Cheap unlocks | T74, T65, T64 → T35 (changelog before the behavior change), T28 |
| 2 | Trust & packaging | T63, T67, T08 (docs after the Phase-1 surface settles) |
| 3 | Operational capabilities (strictly ordered) | T66 → T43 → T68 → T69, then T29 |
| 4 | Architecture & retrieval | T34 (phase 1 → 2) → T31, T33; T32 when corpus size demands |
| 5 | Science & long-tail | T45, T40; T70 → T71; T72, T75; T76 last (gated, measure-first) |
| 8 | **Execution model & feature decoupling (current priority)** | T88 → T87 → T83, T34-p1, T63 → T84 → T85 → T86; T34-p2 last. T89 → T90 slot in anywhere after T88 (independent of the device track) |

## Tier 1 — Tests (detailed, do first)

| ID  | Title                                              | Status | Depends on |
|-----|----------------------------------------------------|--------|------------|
| T01 | Test harness, fixtures, CI, determinism            | done   | —          |
| T02 | Domain unit tests (LabelSpace, policies, tuner)    | done   | T01        |
| T03 | Feature-assembly tests (numpy helpers + assemble)  | done   | T01        |
| T04 | Retrieval tests (BM25 + dense adapters)            | done   | T01        |
| T05 | Fusion + calibration tests                         | done   | T01        |
| T06 | Leakage regression test (the scientific claim)     | done   | T01, T03   |
| T07 | End-to-end pipeline + persistence round-trip       | done   | T01–T05    |

| T08 | Code comments and professional documentation       | todo   | T01        |

## Tier 2 — Hardening + pluggability (T20–T28 complete; T29 open)

| ID  | Title                                                                    | Status | Depends on       |
|-----|--------------------------------------------------------------------------|--------|------------------|
| T20 | Input validation: labels∈classes, empty/dup keys, empty text, clear errors | done | T01, T07        |
| T21 | Deterministic test double: replace `hash()` in HashingEncoder with `hashlib` | done | T01            |
| T22 | Edge cases: single class, class with no examples, k>n_docs, empty batch    | done | T01, T04, T05, T07 |
| T23 | Pluggable component registry + factory DI (encoder/fusion/calibrator)      | done | T01, T07        |
| T24 | Pluggable encoder backends behind `TextEncoder` (e.g. TF-IDF, torch-free)  | done | T23             |
| T25 | Expose `--encoder-kind` in the train CLI (reach the torch-free backend)    | done | T23, T24        |
| T26 | Seed the fusion backends: identical runs → identical models               | done | T01             |
| T27 | Validate the config at pipeline entry (n_folds ≥ 3 and friends)            | done | T01             |
| T28 | Encode-time kwargs + asymmetric query/document encoding (E5/BGE prompts)   | done | T24             |
| T29 | Plugin discovery via entry points (custom kinds load on any host)          | todo | T23             |

**Pluggability chain:** T23 is the prerequisite seam — it makes encoder, fusion,
and calibrator selectable by config. Once it lands, T24 (alt encoders), T41
(LightGBM fusion), T42 (alt calibrators), T44 (XGBRanker fusion), and T31 (FAISS
retrieval) each plug a real backend into that seam without touching the pipeline
or persistence.

**Optionality note (T33):** The cross-encoder is a 6th retrieval *signal*, not a
fusion swap. When absent (default), the system is byte-for-byte identical to
today. T33 does NOT depend on T23. Its dependencies were **restated 2026-08-08**
when the ticket was rewritten: T34 phase 2 (it is a `SignalProvider`), T87 (its
demand gating is what makes turning it off free), and d4109d9 (the structured
taxonomy fields are the texts it scores against — that commit shipped the data
layer with no consumer, and this is the consumer). The rewritten ticket does not
touch `FEATURE_NAMES`, `TrainingPipeline` or `ArtifactRepository`; its one
structural change is a second, post-candidate stage in `FeatureAssembler`.

## Tier 3 — Retrieval & signals (T30 done; rest specified, pick up in any order)

Sequencing note: T34 phase 1 (retriever registry) is the seam T31 should plug
into — do that phase before or with T31. T33's cross-encoder becomes a plug-in
signal once T34 phase 2 lands.

| ID  | Tier | Title                                                                        | Status |
|-----|------|------------------------------------------------------------------------------|--------|
| T30 | 3    | Vectorize `InferencePipeline.predict` (drop the `.iterrows()` loop)            | done |
| T31 | 3    | Optional FAISS/ANN backend behind the `DenseRetriever` port (needs T23)        | todo |
| T32 | ~~3~~ 8 | BM25 memory profile for large corpora — **re-scoped to memory + throughput, moved to Tier 8** | in-review |
| T33 | 3    | Cross-encoder rerank as a second-stage, multi-view signal (rewritten 2026-08-08) | todo |
| T34 | 3    | Pluggable retrieval signals + retrievers behind the registry (needs T23, T03)  | in-review (phase 1 + phase 2 landed 2026-08-05) |
| T35 | 3    | Language-neutral BM25 defaults (drop hidden English stopwords)                | done |
| T81 | 3    | Competition features: per-candidate margins + per-query top1−top2 gap (needs T03) | in-review |

## Tier 4+ — Stubs (T41/T42/T44 done; rest expand when picked up)

| ID  | Tier | Title                                                                        | Status |
|-----|------|------------------------------------------------------------------------------|--------|
| T40 | 4    | Feature ablation + importance reporting harness                              | done |
| T41 | 4    | Alternative fusion model (LightGBM) behind `FusionModel` port (needs T23)      | done |
| T42 | 4    | Calibration comparison (isotonic vs Platt vs beta) (needs T23)                 | done |
| T43 | 4    | Threshold tuner: add target-coverage mode alongside target-precision          | todo |
| T44 | 4    | Alternative fusion model (XGBRanker) behind `FusionModel` port (needs T23)     | done |
| T80 | 4    | Track + select the best epoch of an encoder fine-tune (needs T24)              | in-review |
| T82 | 4    | Retrain-based feature ablation: `drop_features` + seeded sweep harness (needs T40) | in-review |
| T45 | 4    | Per-class calibration behind `ConfidenceCalibrator` port (needs T42)           | done |
| T50 | 5    | Pin requirements for air-gapped reproducibility (hash-locked)                  | done |
| T51 | 5    | ruff + mypy + pre-commit; type-clean the package                               | done |
| T52 | 5    | Offline quality-regression benchmark in CI (metric floors)                     | done |

## Tier 6 — Packaging & production readiness

What turns a good repo into a package a domain expert can install and trust.
T60–T62 landed together (installable CLIs, persisted evaluation, version
provenance). T63–T64 are the remaining install/release essentials; the 2026-07
roadmap review added three operational tickets: T65 (top-k for the human review
queue — cheapest user-facing value in the backlog), T66 (re-tune thresholds
without retraining — the drift response), and T67 (pickle-free artifacts —
loading a shipped model dir must not execute code).

| ID  | Title                                                                          | Status | Depends on   |
|-----|--------------------------------------------------------------------------------|--------|--------------|
| T60 | Installable distribution + CLI ergonomics (console scripts, packaged offline encoder/datasets, friendly IO, py.typed, wheel-smoke CI) | done | T23–T25 |
| T61 | Evaluation metrics + persisted evaluation.json/model_card + `eval` CLI          | done   | T07          |
| T62 | Package version provenance (`__version__`, recorded in meta.json)               | done   | T07          |
| T63 | Torch-optional install via extras (core torch-free) — **unblocks T85**           | done   | T24, T60     |
| T64 | Release process + project hygiene (CHANGELOG/CONTRIBUTING/SECURITY, versioning) | done   | T60          |
| T65 | Top-k suggestions: populate `runner_up_key` + `--top-k` in the infer CLI        | done   | T30          |
| T66 | Re-tune calibration + abstention thresholds on a trained model (tune CLI)       | in-review | T61       |
| T67 | Pickle-free model artifacts (npz/json state; legacy `.pkl` fallback)            | done   | T60          |
| T68 | Taxonomy update without retrain: add classes/examples to a deployed model       | in-review | T61, T66  |
| T69 | Prediction explanations: per-signal evidence, neighbors, SHAP (`--explain`)     | in-review | T30, T65  |
| T78 | Larger/different label space at test & inference time (add classes, no retrain)  | in-review | T61, T77  |

_T78 delivers the description-only "add classes without retrain" path (a focused
slice of T68's scope); T68's remaining piece is seeding examples into a deployed
model, which T78 deliberately defers._

## Tier 7 — Extensibility & architecture (forward-looking)

Cross-cutting extensions that grow the system's surface rather than harden the
existing one. T70 is the high-value capability; T71 is a design spike gated on it
(land T70 first so the feature-provider dependency graph informs the DAG decision).

| ID  | Title                                                                          | Status | Depends on   |
|-----|--------------------------------------------------------------------------------|--------|--------------|
| T70 | Pluggable custom features into the fusion layer (train + inference parity)     | done | T23, T03  |
| T71 | Design spike: DAG-based pipeline orchestration with declared dependencies       | todo   | T70          |
| T72 | Pluggable input/output formats (Parquet/JSONL/SQL/cloud) behind a `RecordSource`/`RecordSink` port | todo | T23 |
| T73 | Richer labeled-evaluation analytics (confusion, aggregate scores, abstention quality, bootstrap CIs) | todo | T61 |
| T74 | `--config` file for the CLIs: reach the full PipelineConfig without Python      | done   | T60          |
| T75 | Frame-native API (interchange protocols, iterables) + streaming infer (`--chunksize`) | todo | T30    |
| T76 | Numpy-only inference path: drop pandas from the hot loop — *gated on T34*       | todo   | T34, T75     |
| T77 | User-provided validation/test splits (`--val-items` / `--test-items`)          | done   | T61          |
| T79 | Expose the fitted retrievers to custom feature providers (`ctx.dense`/`ctx.lexical`) | todo | T70    |

## Tier 8 — Execution model & feature decoupling (current priority)

Two structural problems, one shared seam. With `encoder.device=cuda` and
`xgb_params["device"]="cuda"` the two GPU stages are separated by the most
expensive CPU stage in the system (dense kNN + ~11 `(b, C)` sorts/scatters), so
we pay both transfers and get neither side's compute. Separately, the assembler
computes every feature unconditionally: `drop_features` narrows only what reaches
the model, and a fully-dropped custom provider still runs.

Both refactors want `FeatureAssembler._assemble_chunk` broken into named,
addressable units. **Land the decoupling (T87) before the backend swap (T85/T86)**
— it is pure-CPU and byte-identity-testable, so it gives the device work a stable
reference to validate against; if the GPU track stalls, the decoupling still
shipped.

**T32 moved here and was re-scoped** (2026-08-04, was Tier 3 "memory profile").
BM25 is the one stage the device policy keeps permanently on the host, so its cost
is a hard floor on what T85/T86 can deliver: once the dense side is on the GPU,
BM25 is what a batch waits for. Its throughput half also mirrors T88 — the
description index is rebuilt identically every fold, the corpus is re-tokenized
every fold, and each query batch is analyzed twice (once per BM25 index). Plus the
sparse product is densified only to have its zeros discarded. None of that needs a
device change.

**T88 goes first regardless.** Found while scoping T83: on the shared-encoder
path (the default) the example pool is re-encoded *every fold* and the class
descriptions with it, so a `n_folds=5` run performs ~`5n + 6C` document encodes
for `n + C` distinct texts. It is the largest training-throughput item in the
repo, needs no device work and no port, and until it lands any profile will
measure the encoder at 5x its necessary cost and blame the wrong stage.

Decisions taken 2026-08-04, recorded in the tickets:
- **Tolerance-based parity, not bit-identity.** CPU stays the reference and the
  CI/benchmark baseline; GPU is an accelerator. Cross-device runs are not
  bit-identical and near-ties can shift the candidate set — documented, not
  papered over (qualifies T26's determinism invariant).
- **One codepath behind an `ArrayOps` port**, not a parallel GPU implementation.
  Two implementations of the same 36 features would drift silently.
- **`drop_features` prunes automatically.** Values of remaining columns are
  provably unchanged; the diagnostic surface narrows to what was computed.
- **Deployment scale varies widely**, so `array_backend` defaults to `"auto"` and
  selects from T83's measured crossover thresholds; explicit config always wins.
  T83's deliverable is a curve and a selection rule, not a single verdict.
- **Training throughput is the optimization target**, so per-fold costs (which
  compound `n_folds` times) outrank single-pass costs everywhere in this tier.
- **T63 before T85.** The torch-optional split establishes where torch may live;
  the torch array backend then slots into the `gpu` extra instead of deepening the
  entanglement T63 exists to remove.

| ID  | Title                                                                       | Status | Depends on              |
|-----|-----------------------------------------------------------------------------|--------|-------------------------|
| T88 | Encode the corpus once, not once per fold (~5x less encoder work)            | done   | —                       |
| T32 | BM25 at scale: bounded memory + throughput (re-scoped from Tier 3)           | in-review | T04                  |
| T87 | Feature dependency graph + demand-driven computation                         | done   | T70                     |
| T83 | Device execution profile + device policy (measure-first gate; run after T88) | in-progress | T88                |
| T84 | Array-backend seam: `ArrayOps` port, `auto` selection, CPU byte-identical    | in-review | T83 (thresholds)        |
| T85 | Device-resident dense retrieval + encoder handoff (no D2H mid-pipeline)      | in-progress | T63, T34-p1, T83, T84, T88 |
| T86 | Zero-copy feature matrix into fusion; drop pandas from the hot path          | todo   | T84, T85                |
| T89 | Reuse pooled embeddings as query embeddings (2x → 1x encoder passes)          | in-review | T88, T28             |
| T90 | Overlap BM25 corpus tokenization with the up-front encoder pass               | todo   | T88, T32                |

**T89 and T90 were raised 2026-08-06** from the same question: what in training
is serialized that needn't be. T89 turned out not to be a scheduling problem at
all — it is duplicated work T88 left behind (it shared the *document* side of the
shared-encoder path and left the *query* side re-encoding every held-out item, a
flat extra full pass regardless of `n_folds`). Do **T89 before T90**: it is
bit-identity-testable against T52's golden outputs, so it has an unambiguous
pass/fail signal that a concurrency change would muddy, and it removes most of
what any later fold-level parallelism would be hiding.

**Cross-tier effects when these land:** T63 is promoted from Tier 6 to a hard
prerequisite for T85. T34 phase 1 is promoted (also a T85 prerequisite) and phase
2 becomes the signal-level half of feature decoupling, gated on T87. T71's DAG
spike is partially answered by T87 at the feature level. T86 is the first concrete
slice of T76 (numpy-only inference). `CLAUDE.md`'s embedding invariant is reworded
by T85: L2-normalization stays mandatory, the numpy container does not.
