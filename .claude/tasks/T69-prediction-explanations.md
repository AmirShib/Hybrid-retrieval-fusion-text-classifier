# T69 — Explain predictions: per-signal evidence for reviewers and debugging

status: todo
tier: 6
depends_on: T30, T65

## Goal
An `InferencePipeline.explain(texts, top_k=3)` API and an infer-CLI flag that,
for each item and its top-k candidates, surfaces *why*: the assembled feature
values per signal, the retrieved nearest neighbors, and (optionally) per-feature
fusion contributions. Output is JSON-clean for a review UI.

## Why
The product routes low-confidence items to humans, but a reviewer sees only
key + confidence + margin (`Prediction`, `domain/models.py`). "Why did it call
this a power tool?" is unanswerable without rerunning the pipeline by hand. The
evidence already exists in memory at predict time and is thrown away: the feature
row per (item, candidate) says which signals nominated it (`d_desc_sim` vs
`b_knn_sum` vs …), and the retrievers know which neighbors matched. Explanations
are also the adoption lever: trust in the first weeks decides whether a team
keeps the tool.

## Design
**Explanation payload** (per item), assembled from one normal predict pass — no
second feature assembly:
```
{
  "text": ...,
  "decision": {top_key, confidence, abstained, threshold_applied},
  "candidates": [                      # top-k by calibrated conf (reuse T65)
    {"key": ..., "conf": ...,
     "features": {name: value|null},   # the actual assembled row, NaN -> null
     "contributions": {name: shap}?,   # optional, see below
     "signals_top1": ["d_desc", "d_knn"]},  # which signals ranked it first
  ],
  "neighbors": {
    "dense":   [{"label_key": ..., "sim": ..., "text": ...?}, ...],
    "lexical": [{"label_key": ..., "score": ..., "text": ...?}, ...]
  }
}
```
- **Features**: slice the already-built feature frame for the item's top-k rows;
  serialize via the `_json_safe` convention (`application/evaluation.py`).
- **Neighbors**: `knn_example_labels` already returns indices/labels/scores
  (`retrieval.py`); label keys via `LabelSpace`. Neighbor *texts* require the
  persisted corpus from T68's `--store-corpus`; when absent, emit labels+scores
  only and set `"texts_available": false`. (Descriptions are always available —
  include the matched class description snippet.)
- **Contributions** (`include_contributions=True`): XGBoost
  `booster.predict(DMatrix, pred_contribs=True)` gives per-feature SHAP + bias;
  LightGBM: `predict(..., pred_contrib=True)`. Expose via a new optional
  `FusionModel.predict_contribs(X) -> Optional[np.ndarray]` defaulting to `None`
  (port stays additive; XGBRanker may return None — its isotonic head breaks
  additivity anyway). Contributions are on the *raw* score, pre-calibration —
  say so in the payload (`"contributions_space": "raw_margin"`).
- **Threshold transparency**: include which threshold applied (per-class or
  global) and its value — reviewers keep asking "how close was it".
- **CLI**: `text-classifier-infer --explain explanations.jsonl [--top-k 3]
  [--explain-contribs]` writes one JSON line per input row alongside the normal
  CSV. JSONL because payloads are nested and row counts are large.
- **Performance**: explanation path may be slower (contribs pass) but must not
  slow down the plain predict path at all; assemble-once is the invariant.

## Files to change
- `text_classifier/application/inference.py` — `explain`.
- `text_classifier/domain/ports.py` — optional `predict_contribs` default.
- `text_classifier/infrastructure/fusion.py` — xgboost/lightgbm contribs.
- `text_classifier/cli/infer.py` — the flags.
- `tests/unit/test_fusion.py`, `tests/integration/test_cli.py` — extend.

## Tests
- [ ] `explain` on the hashing-encoder e2e model: payload schema validates; the
      top candidate's `features` match the feature frame values exactly; NaN
      encodes as null.
- [ ] `signals_top1` agrees with the `is_*_top1` feature flags.
- [ ] Contributions: sum(shap) + bias ≈ raw margin (xgboost identity) on a probe
      batch; `predict_contribs` returns None for the ranker without raising.
- [ ] Corpus present vs absent: neighbor texts included vs `texts_available:
      false`, no crash.
- [ ] Plain `predict` timing/behaviour untouched (no assembly duplication —
      assert assembler called once via a counting double).

## Acceptance criteria
- [ ] A reviewer can answer "which signal(s) drove this, how close to the
      threshold, what did it match against" from one JSONL line.
- [ ] JSON-clean (round-trips through `json.loads`), stable field names
      (document them in README).
- [ ] Zero overhead on the non-explain path.

## Out of scope
A UI (payload is the contract); natural-language explanation text; SHAP for the
calibration stage; explaining *training* (T40's ablation harness covers
model-level importance).
