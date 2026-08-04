# T87 — Feature dependency graph + demand-driven computation

status: todo
tier: 8
depends_on: T70

## Goal
Stop computing features nobody asked for. Declare what each feature column depends
on, resolve the transitive closure of the *requested* columns, and evaluate only
that. `drop_features` stops meaning "compute, then discard".

## Why
The current design says it out loud. `FusionConfig.drop_features` is documented as
*"The assembler still **computes** the full schema … this narrows only what the
model is fitted on"* (`config.py:113`), and T82 recorded that boundary as a
deliberate decision, because `explain`, `signal_report` and the masking ablation
all read core columns by name.

That trade was defensible when every column was cheap. It is not defensible now:

- **Custom providers pay full price when fully dropped.** `_provider_columns`
  (`features.py:337`) runs *every* configured provider unconditionally. A provider
  that calls an external service or a reranker, whose columns are entirely in
  `drop_features`, still executes. This is the sharpest form of the problem: the
  decoupling T70 delivered is cosmetic at the cost boundary.
- **Leaf columns are pure waste.** `_row_rank` runs 4×, `_row_margin` 5×,
  `_row_minmax` 2× (`features.py:269-281`), each a full `(b, C)` sort or partition,
  whether or not the model was fitted on the result.

## The distinction that makes this tractable
The core features are **not** independent units of work. They cluster:

1. **Leaf columns** — `rank_*`, `norm_*`, `margin_*`, `q_gap_*`, `*_missing`,
   `d_knn_max`/`count`, `b_knn_max`/`count`, `abs_top_*`, `class_log_freq`,
   `desc_proto_gap`. Terminal nodes: pruning one provably changes **no other
   column's value**. Free and safe.
2. **Shared intermediates** — the five signal matrices. `desc_d` alone feeds eight
   columns *and* the candidate mask. The matmul is skippable only when every
   consumer is gone.
3. **Candidate-set participants** — all five signals feed the top-n union
   (`features.py:236`). Removing one changes the candidate set, so **every row of
   the output changes**: different candidates, different ranks, different margins.
   That is a modelling decision, not plumbing.

This ticket handles (1) and (2). Level (3) — deciding a signal should not run at
all — is **T34 phase 2**, and must stay explicit at the signal level in config;
pruning must never silently remove a candidate-set participant.

## Design

**1. Declare the graph.** Each core column names its inputs, in `domain/services.py`
next to `FEATURE_NAMES` (which stays the source of truth for *order*):

    FEATURE_DEPS: Dict[str, Tuple[str, ...]] = {
        "d_desc_sim":   ("dense.desc",),
        "rank_d_desc":  ("dense.desc", "candidates"),
        "desc_proto_gap": ("dense.desc", "dense.proto"),
        "n_signal_agreement": ("dense.desc", "dense.proto", "bm25.desc",
                               "dense.knn", "bm25.knn"),
        ...
    }

Node names are intermediates (`dense.desc`, `dense.knn`, `bm25.desc`,
`candidates`), not columns. A `FeatureProvider` declares its own dependency via
`names()` — a provider is a single node, run only if ≥1 of its columns is
requested.

**2. Resolve demand, don't configure it.** The assembler takes a `requested:
Sequence[str]` and computes `needed = closure(FEATURE_DEPS, requested) ∪
{candidates}`. Callers ask for what they need, and the right thing happens:

| Caller | Requests |
|---|---|
| `predict`, `predict_topk`, training fit/score | `fusion_feature_names(...)` — the model's columns |
| `explain`, `explain_records`, `signal_report`, `importance_report` | `composed_feature_names(...)` — everything |

No mode flag. The two-list split `InferencePipeline` already carries
(`_feature_names` vs `_assembled_names`, `inference.py:53-56`) becomes the
mechanism instead of a workaround.

**3. `candidates` is never pruned.** The mask is an unconditional dependency of
the whole frame. Any signal feeding it computes regardless of column demand — the
guard that keeps pruning value-preserving.

**4. Unrequested columns are absent, not NaN.** A pruned column must not appear as
an all-NaN column, which XGBoost would read as "signal missing" — the one encoding
this package cannot afford to blur.

**5. Diagnostics narrow honestly.** `signal_report` reports only the signals whose
columns were computed and names the ones it skipped, rather than KeyError-ing or
silently reporting zeros. Same for `explain`. **Behaviour change, accepted:** a
model trained with `drop_features` gets a narrower diagnostic surface than today.
That is the correct trade — diagnostics are a training-time concern and should not
tax every inference batch.

## The safety property
Pruning is value-preserving: for any requested subset `S`, the columns in `S`
computed under pruning are **identical** to the same columns computed under the
full schema. That is the acceptance criterion, and it is directly testable by
fuzzing `S` — it is what makes an incorrect edge in `FEATURE_DEPS` a test failure
rather than a silent train/infer disagreement.

## Files to change
`domain/services.py`, `application/features.py`, `application/inference.py`,
`application/training.py`, `application/signal_report.py`, `application/importance.py`,
`config.py` (docstring — `drop_features` semantics change), `CHANGELOG.md`,
`tests/unit/test_features.py`, `tests/integration/test_drop_features.py`,
`tests/unit/test_signal_report.py`.

## Tests
- [ ] **Pruning parity (fuzzed)**: for randomized requested subsets, every column
      in the subset matches its full-schema value exactly. The core safety net.
- [ ] Candidate mask identical under every pruning subset (bit-for-bit).
- [ ] A provider with all columns dropped has `compute` **never called** (assert
      via a spy provider, not by timing).
- [ ] Requesting the full schema is byte-identical to today's output.
- [ ] `signal_report` on a pruned frame names the skipped signals instead of raising.
- [ ] Every name in `FEATURE_NAMES` has a `FEATURE_DEPS` entry, and every node
      referenced is produced by something (a schema-completeness test, so a new
      feature cannot be added without declaring its inputs).
- [ ] T06 leakage regression + T52 floors green.

## Acceptance criteria
- [ ] `drop_features` measurably reduces assembly work — recorded on a fixed corpus.
- [ ] Dropping every column of a provider skips it entirely.
- [ ] Adding a feature requires touching `FEATURE_NAMES`, `FEATURE_DEPS` and
      `features.py` — enforced by the completeness test, not by convention.
- [ ] Empty `drop_features` (the default) is byte-for-byte unchanged.
- [ ] `CHANGELOG.md` records the diagnostic-narrowing behaviour change.

## Out of scope
Signal-level opt-out (T34 phase 2). Generalizing this into whole-pipeline DAG
orchestration — that is **T71**, which this ticket partially answers at the
feature level; update T71's notes when this lands.
