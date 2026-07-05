# T65 — Top-k suggestions: populate `runner_up_key` and expose `--top-k` in infer

status: todo
tier: 6
depends_on: T30

## Goal
Give the human-review queue more than top-1: populate the existing (currently
always-`None`) `Prediction.runner_up_key`, and let the infer CLI emit the top-k
candidate keys with calibrated confidences per item.

## Why
The product story is "abstain and route to a human" — and what that human sees per
routed item today is a single `top_key`. The information to do better is already
computed and then thrown away: `top_per_item` (`application/scoring.py`) sorts every
scored candidate per item, keeps the runner-up's *confidence* for the `margin`, and
drops its *identity*; `Prediction.runner_up_key` exists in the domain model
(`domain/models.py`) but nothing ever sets it. A reviewer picking from 3–5 ranked
suggestions is much faster than one searching the whole label space — this is the
cheapest user-facing value in the backlog.

## Design
1. **`scoring.top_per_item`**: also carry the runner-up's candidate index. The
   `grp.nth(1)` frame already has it — select `["item_id", "candidate", "conf"]`
   and rename to `second_candidate` / `second_conf`. `second_candidate` is NaN for
   single-candidate items (merge is `how="left"`), so keep it float and let
   consumers check.
2. **New `scoring.top_k_per_item(scored, k) -> pd.DataFrame`**: long format — up to
   k rows per item with columns `item_id`, `rank` (1-based), `candidate`, `conf`.
   Implementation is the same sort + `grp.head(k)`; no new hot-path loops.
3. **`InferencePipeline.predict`**: set `runner_up_key` on each `Prediction` (map
   `second_candidate` through `label_space.keys` the same vectorized way `top_keys`
   is built; `None` when absent). `margin` semantics unchanged.
4. **`InferencePipeline.predict_topk(texts, k) -> List[List[Tuple[str, float]]]`**:
   per input, a ranked list of `(class_key, confidence)`, length ≤ k (shorter when
   fewer candidates surfaced; empty for items that retrieved nothing). Reuses the
   feature table from a single assembly pass — do not assemble twice.
5. **CLI (`cli/infer.py`)**: `--top-k N` (default 1 = today's output, byte-identical).
   With N > 1, add wide columns `top2_key`, `top2_conf`, … `topN_key`, `topN_conf`
   (empty string / empty cell when absent). `predicted_key` / `abstained` semantics
   are untouched — abstention stays a top-1 decision; the extra columns are
   suggestions for the human, not extra accepted predictions.

## Files to change
- `text_classifier/application/scoring.py` — runner-up identity + `top_k_per_item`.
- `text_classifier/application/inference.py` — populate `runner_up_key`; `predict_topk`.
- `text_classifier/cli/infer.py` — `--top-k`.
- `tests/unit/test_features.py` or a new `tests/unit/test_scoring.py` — scoring units.
- `tests/integration/test_cli.py` — CLI output-shape cases.

## Tests
- [ ] `top_per_item`: runner-up candidate matches the second-highest-conf row;
      single-candidate item → NaN second_candidate, margin = conf (unchanged).
- [ ] `predict`: `runner_up_key` is the right key, `None` when only one candidate;
      existing fields byte-identical to before (regression guard on top-1 path).
- [ ] `predict_topk`: ranks strictly by conf desc; k > n_candidates truncates;
      empty-retrieval item → empty list; k=1 agrees with `predict`'s top pick.
- [ ] CLI: default output identical to today (golden-column check);
      `--top-k 3` adds exactly the four new columns with correct values.

## Acceptance criteria
- [ ] `runner_up_key` populated in both pipelines' decision paths (training-side
      evaluation may ignore it).
- [ ] Default CLI output unchanged; `--top-k 1` == no flag.
- [ ] No second feature-assembly or fusion pass for top-k; still vectorized.

## Out of scope
Top-k-aware abstention (accept if truth in top-k) — that changes the decision
policy and the tuner, and belongs with T43. Exposing all candidates' scores
(unbounded width); k is capped by `candidate_top_n`'s union anyway.
