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

## Tier 2 — Hardening (fully specified, pick up now)

| ID  | Title                                                                    | Status | Depends on       |
|-----|--------------------------------------------------------------------------|--------|------------------|
| T20 | Input validation: labels∈classes, empty/dup keys, empty text, clear errors | todo | T01, T07        |
| T21 | Deterministic test double: replace `hash()` in HashingEncoder with `hashlib` | todo | T01            |
| T22 | Edge cases: single class, class with no examples, k>n_docs, empty batch    | todo | T01, T04, T05, T07 |

## Tier 3+ — Stubs (expand when Tier 2 is green)

| ID  | Tier | Title                                                                    | Status |
|-----|------|--------------------------------------------------------------------------|--------|
| T30 | 3    | Vectorize `InferencePipeline.predict` (drop the `.iterrows()` loop)        | todo |
| T31 | 3    | Optional FAISS/ANN backend behind the `DenseRetriever` port               | todo |
| T32 | 3    | BM25 memory profile for large corpora; chunk/sparsify as needed           | todo |
| T40 | 4    | Feature ablation + importance reporting harness                           | todo |
| T41 | 4    | Alternative fusion model (LightGBM) behind `FusionModel` port             | todo |
| T42 | 4    | Calibration comparison (isotonic vs Platt vs beta)                        | todo |
| T43 | 4    | Threshold tuner: add target-coverage mode alongside target-precision      | todo |
| T50 | 5    | Pin requirements for air-gapped reproducibility                           | todo |
| T51 | 5    | ruff + mypy + pre-commit; type-clean the package                          | todo |
