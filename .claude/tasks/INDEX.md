# Task backlog

SWE-style tickets for this package. Each `TNN-*.md` is self-contained: read it
top-to-bottom and you have everything needed to do the work.

**Status values:** `todo` · `in-progress` · `in-review` · `done`
When you start a ticket, set its `status:` field (top of the file) and update the table here.
When a ticket reaches `done` and the work is merged: move its file from `.claude/tasks/` into
`.claude/tasks/done/` — the row stays in this table for history.

Priority is top-down: **Tier 1 is complete.** Tier 2 tickets are now fully
specified and ready to pick up. Tier 3+ remain as stubs.

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

## Tier 2 — Hardening + pluggability (fully specified, pick up now)

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

## Tier 3+ (T33 specified; T41/T42 done; the rest are stubs — expand when picked up)

| ID  | Tier | Title                                                                        | Status |
|-----|------|------------------------------------------------------------------------------|--------|
| T30 | 3    | Vectorize `InferencePipeline.predict` (drop the `.iterrows()` loop)            | todo |
| T31 | 3    | Optional FAISS/ANN backend behind the `DenseRetriever` port (needs T23)        | todo |
| T32 | 3    | BM25 memory profile for large corpora; chunk/sparsify as needed               | todo |
| T33 | 3    | Optional cross-encoder reranker as 6th retrieval signal (needs T03, optional)  | todo |
| T40 | 4    | Feature ablation + importance reporting harness                               | todo |
| T41 | 4    | Alternative fusion model (LightGBM) behind `FusionModel` port (needs T23)      | done |
| T42 | 4    | Calibration comparison (isotonic vs Platt vs beta) (needs T23)                 | done |
| T43 | 4    | Threshold tuner: add target-coverage mode alongside target-precision          | todo |
| T44 | 4    | Alternative fusion model (XGBRanker) behind `FusionModel` port (needs T23)     | done |
| T50 | 5    | Pin requirements for air-gapped reproducibility                               | todo |
| T51 | 5    | ruff + mypy + pre-commit; type-clean the package                              | todo |
