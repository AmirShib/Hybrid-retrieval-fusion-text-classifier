# Task backlog

SWE-style tickets for this package. Each `TNN-*.md` is self-contained: read it
top-to-bottom and you have everything needed to do the work.

**Status values:** `todo` · `in-progress` · `in-review` · `done`
When you start a ticket, set its `status:` field (top of the file) and update the table here.
When a ticket reaches `done` and the work is merged: move its file from `.claude/tasks/` into
`.claude/tasks/done/` — the row stays in this table for history.

Priority is top-down. **Phase 0 of the execution order is complete** (T26, T27,
T52, T51, T50 — determinism, config validation, the quality-benchmark net, lint/
type gates, hash-locked pins). **Tier 1 has one open item (T08, docs). Tier 2's
remaining items are T28 (encode-time kwargs / asymmetric prompts) and T29
(plugin discovery), which unblock modern encoders and cross-host custom
backends.** Tier 3 (T30–T35) is the active feature tier. Tier 4 remains stubs
except T41/T42/T44 (done). **Tier 6 is the packaging / production-readiness
tier**: T60–T62 are done; T63–T69 remain — T63 (torch-optional install) is the
priority item, it closes the gap between the README's air-gapped/torch-free
claim and what `pip install .` actually delivers, and T68 (taxonomy update
without retrain) is the highest-value operational capability for adopters.
Tier 7 added T74 (`--config`) — the cheapest unlock in the backlog — and T75
(DataFrame API + streaming).

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

## Tier 2 — Hardening + pluggability (T20–T27 complete; T28–T29 open)

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
| T28 | Encode-time kwargs + asymmetric query/document encoding (E5/BGE prompts)   | todo | T24             |
| T29 | Plugin discovery via entry points (custom kinds load on any host)          | todo | T23             |

**Pluggability chain:** T23 is the prerequisite seam — it makes encoder, fusion,
and calibrator selectable by config. Once it lands, T24 (alt encoders), T41
(LightGBM fusion), T42 (alt calibrators), T44 (XGBRanker fusion), and T31 (FAISS
retrieval) each plug a real backend into that seam without touching the pipeline
or persistence.

**Optionality note (T33):** The cross-encoder is a 6th retrieval *signal*, not a
fusion swap. It depends only on T03 (feature assembly). When absent (default), the
system is byte-for-byte identical to today. T33 does NOT depend on T23.

## Tier 3 — Retrieval & signals (T30 done; rest specified, pick up in any order)

Sequencing note: T34 phase 1 (retriever registry) is the seam T31 should plug
into — do that phase before or with T31. T33's cross-encoder becomes a plug-in
signal once T34 phase 2 lands.

| ID  | Tier | Title                                                                        | Status |
|-----|------|------------------------------------------------------------------------------|--------|
| T30 | 3    | Vectorize `InferencePipeline.predict` (drop the `.iterrows()` loop)            | done |
| T31 | 3    | Optional FAISS/ANN backend behind the `DenseRetriever` port (needs T23)        | todo |
| T32 | 3    | BM25 memory profile for large corpora; chunk/sparsify as needed               | todo |
| T33 | 3    | Optional cross-encoder reranker as 6th retrieval signal (needs T03, optional)  | todo |
| T34 | 3    | Pluggable retrieval signals + retrievers behind the registry (needs T23, T03)  | todo |
| T35 | 3    | Language-neutral BM25 defaults (drop hidden English stopwords)                | todo |

## Tier 4+ — Stubs (T41/T42/T44 done; rest expand when picked up)

| ID  | Tier | Title                                                                        | Status |
|-----|------|------------------------------------------------------------------------------|--------|
| T40 | 4    | Feature ablation + importance reporting harness                               | todo |
| T41 | 4    | Alternative fusion model (LightGBM) behind `FusionModel` port (needs T23)      | done |
| T42 | 4    | Calibration comparison (isotonic vs Platt vs beta) (needs T23)                 | done |
| T43 | 4    | Threshold tuner: add target-coverage mode alongside target-precision          | todo |
| T44 | 4    | Alternative fusion model (XGBRanker) behind `FusionModel` port (needs T23)     | done |
| T45 | 4    | Per-class calibration behind `ConfidenceCalibrator` port (needs T42)           | todo |
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
| T63 | Torch-optional install via extras (core torch-free)                             | todo   | T24, T60     |
| T64 | Release process + project hygiene (CHANGELOG/CONTRIBUTING/SECURITY, versioning) | todo   | T60          |
| T65 | Top-k suggestions: populate `runner_up_key` + `--top-k` in the infer CLI        | todo   | T30          |
| T66 | Re-tune calibration + abstention thresholds on a trained model (tune CLI)       | todo   | T61          |
| T67 | Pickle-free model artifacts (npz/json state; legacy `.pkl` fallback)            | todo   | T60          |
| T68 | Taxonomy update without retrain: add classes/examples to a deployed model       | todo   | T61, T66     |
| T69 | Prediction explanations: per-signal evidence, neighbors, SHAP (`--explain`)     | todo   | T30, T65     |

## Tier 7 — Extensibility & architecture (forward-looking)

Cross-cutting extensions that grow the system's surface rather than harden the
existing one. T70 is the high-value capability; T71 is a design spike gated on it
(land T70 first so the feature-provider dependency graph informs the DAG decision).

| ID  | Title                                                                          | Status | Depends on   |
|-----|--------------------------------------------------------------------------------|--------|--------------|
| T70 | Pluggable custom features into the fusion layer (train + inference parity)     | todo   | T23, T03     |
| T71 | Design spike: DAG-based pipeline orchestration with declared dependencies       | todo   | T70          |
| T72 | Pluggable input/output formats (Parquet/JSONL/SQL/cloud) behind a `RecordSource`/`RecordSink` port | todo | T23 |
| T73 | Richer labeled-evaluation analytics (confusion, aggregate scores, abstention quality, bootstrap CIs) | todo | T61 |
| T74 | `--config` file for the CLIs: reach the full PipelineConfig without Python      | todo   | T60          |
| T75 | Frame-native API (interchange protocols, iterables) + streaming infer (`--chunksize`) | todo | T30    |
| T76 | Numpy-only inference path: drop pandas from the hot loop — *gated on T34*       | todo   | T34, T75     |
