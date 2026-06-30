# Task backlog

SWE-style tickets for this package. Each `TNN-*.md` is self-contained: read it
top-to-bottom and you have everything needed to do the work.

**Status values:** `todo` · `in-progress` · `in-review` · `done`
When you start a ticket, set its `status:` field (top of the file) and update the table here.
When a ticket reaches `done` and the work is merged: move its file from `.claude/tasks/` into
`.claude/tasks/done/` — the row stays in this table for history.

Priority is top-down. **Tiers 1–2 are done** (lone remaining Tier-1 item: T08,
docs). **Tier 3 (T30–T33) is the active tier — do these next**; all four are now
fully specified as self-contained tickets. Tier 4–5 remain stubs, except T41,
T42, and T44 (done); T50 and T51 are now fully specified. **Tier 6 is the
packaging / production-readiness tier**: T60–T62 are done (the package is now
`pip install`-able with persisted evaluation); T63–T64 remain.

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

## Tier 2 — Hardening + pluggability (complete)

| ID  | Title                                                                    | Status | Depends on       |
|-----|--------------------------------------------------------------------------|--------|------------------|
| T20 | Input validation: labels∈classes, empty/dup keys, empty text, clear errors | done | T01, T07        |
| T21 | Deterministic test double: replace `hash()` in HashingEncoder with `hashlib` | done | T01            |
| T22 | Edge cases: single class, class with no examples, k>n_docs, empty batch    | done | T01, T04, T05, T07 |
| T23 | Pluggable component registry + factory DI (encoder/fusion/calibrator)      | done | T01, T07        |
| T24 | Pluggable encoder backends behind `TextEncoder` (e.g. TF-IDF, torch-free)  | done | T23             |
| T25 | Expose `--encoder-kind` in the train CLI (reach the torch-free backend)    | done | T23, T24        |

**Pluggability chain:** T23 is the prerequisite seam — it makes encoder, fusion,
and calibrator selectable by config. Once it lands, T24 (alt encoders), T41
(LightGBM fusion), T42 (alt calibrators), T44 (XGBRanker fusion), and T31 (FAISS
retrieval) each plug a real backend into that seam without touching the pipeline
or persistence.

**Optionality note (T33):** The cross-encoder is a 6th retrieval *signal*, not a
fusion swap. It depends only on T03 (feature assembly). When absent (default), the
system is byte-for-byte identical to today. T33 does NOT depend on T23.

## Tier 3 — Next up (all four tickets are specified; pick up in any order)

| ID  | Tier | Title                                                                        | Status |
|-----|------|------------------------------------------------------------------------------|--------|
| T30 | 3    | Vectorize `InferencePipeline.predict` (drop the `.iterrows()` loop)            | done |
| T31 | 3    | Optional FAISS/ANN backend behind the `DenseRetriever` port (needs T23)        | todo |
| T32 | 3    | BM25 memory profile for large corpora; chunk/sparsify as needed               | todo |
| T33 | 3    | Optional cross-encoder reranker as 6th retrieval signal (needs T03, optional)  | todo |

## Tier 4+ — Stubs (T41/T42/T44 done; rest expand when picked up)

| ID  | Tier | Title                                                                        | Status |
|-----|------|------------------------------------------------------------------------------|--------|
| T40 | 4    | Feature ablation + importance reporting harness                               | todo |
| T41 | 4    | Alternative fusion model (LightGBM) behind `FusionModel` port (needs T23)      | done |
| T42 | 4    | Calibration comparison (isotonic vs Platt vs beta) (needs T23)                 | done |
| T43 | 4    | Threshold tuner: add target-coverage mode alongside target-precision          | todo |
| T44 | 4    | Alternative fusion model (XGBRanker) behind `FusionModel` port (needs T23)     | done |
| T45 | 4    | Per-class calibration behind `ConfidenceCalibrator` port (needs T42)           | todo |
| T50 | 5    | Pin requirements for air-gapped reproducibility (hash-locked) — *specified*    | todo |
| T51 | 5    | ruff + mypy + pre-commit; type-clean the package — *specified*                 | todo |

## Tier 6 — Packaging & production readiness

What turns a good repo into a package a domain expert can install and trust.
T60–T62 landed together (installable CLIs, persisted evaluation, version
provenance). T63–T64 are the remaining essentials.

| ID  | Title                                                                          | Status | Depends on   |
|-----|--------------------------------------------------------------------------------|--------|--------------|
| T60 | Installable distribution + CLI ergonomics (console scripts, packaged offline encoder/datasets, friendly IO, py.typed, wheel-smoke CI) | done | T23–T25 |
| T61 | Evaluation metrics + persisted evaluation.json/model_card + `eval` CLI          | done   | T07          |
| T62 | Package version provenance (`__version__`, recorded in meta.json)               | done   | T07          |
| T63 | Torch-optional install via extras (core torch-free)                             | todo   | T24, T60     |
| T64 | Release process + project hygiene (CHANGELOG/CONTRIBUTING/SECURITY, versioning) | todo   | T60          |

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
