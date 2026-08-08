# T33 — Cross-encoder rerank as a second-stage, multi-view signal

status: in-progress (phase 1 + most of phase 2 landed; phase 3 open)
tier: 3
depends_on: T34 (phase 2), T87, d4109d9 (structured taxonomy fields)

> **This ticket was rewritten 2026-08-08.** The original version predates T34
> phase 2 (`SignalProvider`), T87 (demand-driven computation) and d4109d9
> (structured taxonomy fields). Most of its proposed plumbing is now redundant —
> see "What the original design got wrong" below. The scope (an optional 6th
> signal, off by default, byte-identical when absent) is unchanged.

## Goal
Score `(item, candidate class)` pairs with a **cross-encoder** — a model that
reads both texts jointly instead of comparing two independently-built vectors —
and feed the result to the fusion model as ordinary signal columns.

Two properties define the shape of the work:

1. A cross-encoder is ~100x slower than a bi-encoder, so it can only run on
   candidates the cheap signals already shortlisted. It is therefore a **second
   stage**, and the assembler currently has no such stage.
2. A class can carry more than one text (d4109d9). The cross-encoder is the
   first consumer able to use them — including `exclusions`/
   `sibling_distinctions`, which are **negative** evidence no existing signal
   can express at all.

Entirely optional: with `"cross-encoder"` absent from `PipelineConfig.signals`
(the default) the system is byte-for-byte identical to today, no columns appear,
and no torch import happens.

## Why

**The five existing signals are all single-vector or lexical.** Every one of them
compresses the class text into a representation built without knowing what it
will be compared against. A cross-encoder attends across both texts, which is the
standard reranking step in production retrieval and consistently the largest
single quality jump available once candidate recall is adequate.

**The taxonomy data for this already landed with no consumer.** d4109d9 shipped
`ClassDefinition.title/definition/examples/inclusions/exclusions/parent_path/
sibling_distinctions`, four rendered views (`domain/models.py:144-181`), CSV
ingest (`cli/_common.py:145-151`), `meta.json` round-trip
(`infrastructure/persistence.py:84-90`) and tests. `grep` finds **no production
call** to any `*_view()` — the consumer is this ticket. `boundary_view`'s own
docstring (`domain/models.py:170-181`) states the intent explicitly:

> **Not for retrieval.** Embedding models handle negation poorly: "excludes food
> manufacturing" still sits close to *food manufacturing* [...] This view exists
> for pair-scoring (a reranker sees both sides jointly) and for mining hard
> negatives when fine-tuning the encoder.

**Negative evidence is genuinely new information.** Nothing in the system today
can represent "this item looks like the thing this class explicitly rules out."
That is precisely the case where the pipeline should abstain rather than guess,
so it feeds the product's actual deliverable (the coverage/accuracy trade-off),
not just its accuracy.

## What the original design got wrong

Recorded so the diff against the old ticket is reviewable, not silent.

| Original proposal | Why it is now wrong |
|---|---|
| Conditionally extend `FEATURE_NAMES` when a reranker is configured | `composed_feature_names` (`domain/services.py:239-253`) already appends a non-builtin `SignalProvider`'s `column_names()` after the core 36 and persists that order into `meta.json`. A conditional `FEATURE_NAMES` would be a second, competing schema mechanism — the exact train/infer disagreement that constant exists to prevent. **`FEATURE_NAMES` is not touched by this ticket.** |
| Plumb `Optional[PairwiseReranker]` through `FeatureAssembler`, `TrainingPipeline`, `ArtifactRepository` | `build_signal_providers`/`load_signal_providers` (`infrastructure/registry.py:339,368`) and `_save_signal_providers` (`infrastructure/persistence.py:458-468`) already dispatch any registered signal kind. `tests/unit/test_signal_providers.py` proves a third-party signal wires up end-to-end with zero core edits. |
| `ce_missing` as a hand-written binary column | `SignalMatrix(derive={"missing"})` is exactly `isnan(gather(M))` (`application/features.py:330`). |
| A single `ce_score` column | The fusion model is **pointwise** — one row per (item, candidate), scored in isolation. A bare cross-encoder logit is close to unusable without rank/margin/gap context; this is the same argument `domain/services.py:53-81` records for the T81 competition features. |
| "Cross-encoder is a 6th *signal*" — but with no account of *when* it runs | Correct in kind, wrong in stage. Every `SignalProvider.build()` runs **before** the candidate mask exists (`application/features.py:289` vs the mask at `:302-305`). A reranker is defined by consuming that mask. This is the one real gap and the only structural change in this ticket. |

## Design

### 1. A second stage in the assembler (the only structural change)

`_assemble_chunk` gains one insertion point. Nothing is reordered:

| step | today | after |
|---|---|---|
| 1 | run every signal provider | run providers with `needs_candidates == False` |
| 2 | mask = top-n union (`:302-305`) | unchanged — **round-two signals cannot influence it** |
| 3 | `rows, cols = nonzero(mask)`; early return if empty (`:305-310`) | unchanged |
| 3b | — | **new:** run each `needs_candidates` provider, handing it the mask |
| 4 | generic derivation loop (`:335-361`) | unchanged — now iterates 6+ matrices instead of 5 |
| 5 | cross-signal columns (`:363-384`) | unchanged |
| 6 | build frame + `FeatureProvider` columns (`:392-415`) | unchanged |

- **Port change:** `SignalProvider.needs_candidates: bool = False` (class
  attribute). The default keeps both built-ins and any existing third-party
  provider bit-identical.
- **Context change:** `SignalContext` gains `label_space` (symmetric with
  `FeatureContext`, which already carries it — `domain/ports.py:447`) and
  `candidates: Optional[CandidateView]`. `CandidateView` is a frozen dataclass
  holding `mask`/`rows`/`cols` plus the round-one `{node: (b, C) value}` map.
  One field, `None` in round one, so the two stages cannot be half-read.
  `SignalContext` has exactly **two** construction sites in the repo
  (`application/features.py:286`, `tests/unit/test_signal_providers.py:257`).
- **Enforced invariant:** a `needs_candidates` provider must return an empty
  `candidate_features()`. It cannot select what it consumes. Raise a clear error
  at assembly time rather than allowing an ordering-dependent silent wrong answer.
- **Two properties fall out of that invariant, for free:**
  - The `rows.size == 0` early return precedes step 3b — an empty shortlist never
    invokes the cross-encoder.
  - A round-two provider whose every column is pruned by T87's `needed` set is
    **skipped entirely**. Safe *only* because it contributes no candidates. This
    is what makes `drop_features` on the `ce_*` columns cost zero rather than
    paying for a full cross-encoder pass whose output is then discarded.
- The duplicate-node check (`:291-296`) becomes a shared local closure rather
  than being written twice.

`_topn_mask` (`application/features.py:70`) moves to `infrastructure/signals.py`
and is re-exported, following the precedent already set for `_scatter_knn`
(`infrastructure/signals.py:1-17`) — an infrastructure provider must not import
the application layer.

### 2. Composition is data, not code

The provider's unit of work is a **document**: for each shortlisted candidate it
renders one or more texts, scores each against the item, and returns one
`(b, C)` matrix per document. Each matrix then collects the full generic
derivation set from the untouched loop in step 4:

| document `d` | columns produced |
|---|---|
| `ce.<d>` | `ce_<d>`, `ce_<d>_missing`, `rank_ce_<d>`, `norm_ce_<d>`, `margin_ce_<d>`, `q_gap_ce_<d>`, `is_ce_<d>_top1` |

Seven columns per document, **zero new arithmetic**.

**This is the ticket's central design decision.** "One composed document, one
score" and "one document per view, N scores" are the same code path with
different config. That matters because the right answer depends on how capable
the scorer is, and the scorer will change:

| backend | composing a negative into the document |
|---|---|
| stock relevance cross-encoder (`ms-marco-*`) | **unsafe** — trained on `(query, passage)` relevance; it has no way to know a span is a prohibition. Overlap reads as relevance, so pasting `Excludes: food manufacturing` will likely *raise* the score for a food-manufacturing query. The exact failure `boundary_view` warns about; joint encoding does not fix it, *instruction or training* does |
| instructed LLM judge | **safe** — can be told what each part means |
| cross-encoder fine-tuned for this taxonomy | **safe** — trained to interpret the structure |

So the default must be conservative while the config must permit the composed
form without a rewrite. Swapping to an LLM judge later is a config change.

**Default document set** (recommended, not the only reachable one):

- `pos` — the class's positive identity, composed: `core_view()` plus the
  best-matching `examples` entry. One coherent "what this class is" statement,
  in-distribution for a relevance model.
- `neg` — the best-matching `exclusions` / `sibling_distinctions` entry, scored
  **alone**. Read on its own, a high score means "this item resembles what this
  class rules out", which is unambiguous regardless of backend.
- `pos_neg_gap` — `ce_pos - ce_neg`, an explicit column. The fusion model splits
  on one feature at a time and cannot learn this difference itself; the existing
  `desc_proto_gap` (`domain/services.py:113`) is the same move for the same
  reason. Emitted via `SignalMatrix.extra_columns`, only when both documents are
  configured.

Two cross-encoder calls per reranked candidate, not five, and no negation trap.

### 3. Query-adaptive evidence selection

A class with six `examples` has six different doors in, and which one fits
depends on the query. Each piece of evidence is therefore **selected per query**,
not fixed: for tuple-valued fields (`examples`, `inclusions`, `exclusions`,
`sibling_distinctions`) pick the entry most similar to the item text; scalar
views (`description`, `core_view()`) have nothing to select.

For the negative side this is not merely an optimization — the *most* similar
exclusion is the hard negative, i.e. the one actually at risk of being confused.

**Selection is lexical (token overlap) in this ticket, not embedding-based.**
Recorded because it contradicts the obvious first instinct: embedding selection
needs the evidence texts encoded, and `SignalProviderSpec.build`
(`infrastructure/registry.py:164-168`) receives `(RetrievalConfig, dense,
lexical, ArrayOps)` — **no encoder and no label space**. Getting one there means
widening a spec signature shared with both built-ins. Lexical overlap is cheap,
stateless, needs nothing persisted, and is a defensible selector for picking
among a handful of short phrases. Embedding-based selection is a follow-up,
gated on that signature question.

Selection runs only on cells that will actually be reranked (`b x top_k`), so it
stays vectorized and small regardless of `C`.

### 4. Absent evidence is NaN, never a score

`boundary_view()` returns `""` for a class with no exclusions. Scoring against
an empty string manufactures a number where there is no information — CLAUDE.md's
NaN invariant at the text layer, which `domain/models.py:139-143` already states.
A document is built only when at least one of its evidence slots is non-empty;
everything else stays NaN and lands in `ce_<d>_missing`. A sparse taxonomy
therefore also costs proportionally less.

Within the shortlist, NaN in `ce_<d>` means **"shortlisted but not reranked"**
(outside `top_k`, or no evidence for this view) — distinct from "never
retrieved", which does not appear in the grid at all. This is the distinction the
original ticket asked `ce_missing` to carry, obtained from existing machinery.

### 5. Cost control

Pairs = `items x top_k x documents`. Mitigations in order of effect:

1. **`top_k` per document.** The shortlist may hold 40 classes; rerank the best
   ~10 by `prescore_node` (default `dense.desc`, already computed in round one
   and handed over in `CandidateView`).
2. **Internal cascade.** A document may declare a smaller `top_k` than another,
   ranked by an earlier document's cross-encoder score — e.g. `pos` on 10
   candidates, then `neg` on only the 3 survivors. This needs no third stage:
   one provider may stage its own work internally.
3. **Demand gating** (free, §1) and **absent-evidence skipping** (free, §4).

**Token budget.** Formal taxonomy definitions run to paragraphs; `core_view()`
plus an example plus an exclusion can exceed a cross-encoder's 512-token window,
and naive concatenation truncates the *tail* — which in the obvious ordering is
the negative, silently deleting the one part this design exists to add. Each
evidence slot carries an explicit character budget and the render order is
declared, so truncation is per-slot and predictable rather than tail-chopped.

### 6. Config

Sketch — field names indicative, the *shape* is what this ticket fixes. Home is
`RetrievalConfig`, because `SignalProviderSpec.build` already receives it, so no
spec-signature churn (§3's constraint, applied consistently).

```python
@dataclass
class EvidenceSpec:
    view: str                  # description | core | examples | inclusions | exclusions | siblings
    select: str = "best"       # "best" (query-adaptive) | "all" | "first"; ignored for scalar views
    label: str = ""            # rendered prefix, e.g. "Excludes: "
    max_chars: int = 300       # per-slot budget (§5)

@dataclass
class CrossEncoderDocument:
    name: str                          # -> node "ce.<name>", column suffix
    evidence: List[EvidenceSpec]
    instruction: str = ""              # prepended; the LLM-judge affordance
    join: str = "\n"
    top_k: int = 10
    cascade_from: Optional[str] = None # rerank only this document's survivors (§5.2)

@dataclass
class CrossEncoderConfig:
    kind: str = "cross-encoder"        # registry key: swap in an LLM judge here
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    batch_size: int = 32
    device: Optional[str] = None
    prescore_node: str = "dense.desc"
    documents: List[CrossEncoderDocument] = <default: pos + neg, see §2>
    gaps: List[Tuple[str, str]] = <default: [("pos", "neg")]>
    params: Dict[str, Any] = field(default_factory=dict)
```

Enabled by adding `"cross-encoder"` to `PipelineConfig.signals` (`config.py:301`).
Typed rather than a generic `signal_params` dict, to stay inside `validate()`
coverage and CLI discoverability. `to_dict`/`from_dict` must default the whole
block so a pre-T33 `meta.json` loads unchanged.

### 7. Ports and persistence

- New port `PairwiseReranker` (`domain/ports.py`): `score(pairs) -> (n,) float32`
  plus `save`/`load`. Separates *what scores a pair* from *which pairs and how
  they are shaped*, so the offline stub, a real cross-encoder and an LLM judge
  swap behind one interface. Registered via a new `RerankerSpec`.
- `CrossEncoderSignalProvider` persists to `signals/cross-encoder/` through the
  existing `_save_signal_providers` loop — **no `persistence.py` change**. Its
  state is the reranker artifact plus its own JSON config. Portable formats only
  (a `sentence-transformers` `CrossEncoder.save` writes safetensors + json, no
  pickle), so the air-gapped-portability invariant holds.
- The provider is **stateless with respect to the taxonomy** — evidence comes
  from `ctx.label_space` at build time — so `with_added_classes` and the `update`
  CLI pick up new classes automatically and `rewrap_signal_providers`
  (`infrastructure/signals.py:78`) needs no cross-encoder case.

## Leakage

**Nothing in this ticket touches the out-of-fold rule.** Every text scored is
class-side taxonomy content: static, authored in the codebook, identical in every
fold, and already shipped inside the model directory. There is no example-pool
access and no fitted state derived from labels, so no fold discipline applies —
this is a stronger statement than "we masked it correctly", and the leakage test
should assert the *structural* property (the provider never receives pool text)
rather than a numeric one.

The stronger channel — scoring against **labeled training examples** — is
deliberately excluded; see Out of scope.

## Files to add/change
`domain/ports.py` (`needs_candidates`, `SignalContext.label_space`/`candidates`,
`CandidateView`, `PairwiseReranker`), `application/features.py` (the second
stage; move `_topn_mask` out), `infrastructure/signals.py` (receives
`_topn_mask`), `infrastructure/reranker.py` (**new**: the stub + real backends +
`CrossEncoderSignalProvider`), `infrastructure/registry.py` (`RerankerSpec`,
register both kinds), `config.py`, `cli/train.py`, `tests/unit/test_reranker.py`
(**new**), `tests/unit/test_signal_providers.py`, `tests/integration/`,
`CHANGELOG.md`, `CLAUDE.md` (the signal count is stated as five in two places).

## Plan of work

**Phase 1 — seam + stub, no torch.** The second stage, the port, a deterministic
dependency-free `HashingCrossEncoder` (sha256, mirroring `HashingEncoder`
`infrastructure/encoder.py:347`), and a single-document config. Proves the whole
path offline; CI never downloads a model.

**Phase 2 — evidence and composition.** The remaining views, query-adaptive
selection, the gap column, per-slot budgets, the cascade. Pure data and config on
proven machinery.

**Phase 3 — real backend.** `SentenceTransformersCrossEncoder` behind a lazy
import, `skipif`-gated tests.

## Progress (2026-08-08) — phases 1 and 2 implemented

`tests/unit/test_reranker.py` (35 tests) is the new coverage; full suite 925
passed, 2 skipped (both pre-existing lightgbm skips). `ruff check` and `ruff
format --check` clean. `mypy` fails on this host for the pre-existing numpy-stub
vs. Python 3.14 reason T89 recorded — identical before this change.

**Scope note: phase 2 was pulled forward.** Phase 1 was specified as "the seam,
the port, the stub, and a single-document config", but shipping a `desc`-only
default and changing it later would have been churn — the default document set
*is* the design, and the evidence rendering is the natural way to write the
provider rather than extra work layered on. What landed is phases 1 + 2 minus
the cascade (`cascade_from` is specified but not implemented; a document's
`top_k` is always taken against `prescore_node`). Phase 3 (the real backend) is
untouched and remains the next step.

Notes on what deviated or was learned:

- **The negative document works, and the mechanism is visible in the numbers.**
  For the query *"retail sale of food supermarket"* against a 3-class taxonomy,
  candidate `food_mfg` scores `ce_pos = -1.69` but `ce_neg = +0.59` — its
  exclusion text *is* "retail sale of food", so it matches the query better than
  its own definition does, and `ce_pos_neg_gap = -2.27`. The class that should
  win reads `+0.90`. That is the entire argument for the feature, reproduced as
  `test_negative_evidence_fires_on_the_class_that_excludes_the_query`.
- **`_topn_mask` moved to `infrastructure/signals.py`** and is re-exported from
  `application/features.py`, following `_scatter_knn`'s precedent — the provider
  needs the identical top-n rule and must not import the application layer.
- **Found and fixed a pre-existing bug this ticket would otherwise have to work
  around:** `TrainingPipeline.__init__` eagerly computed
  `fusion_feature_names(drop=cfg.fusion.drop_features)` against the *core-only*
  schema, so any `drop_features` entry naming a provider column (a custom
  `FeatureProvider`'s, or a signal's) raised before the run could start. Broken
  since custom providers landed; `ce_*` is just what surfaced it. Fixing it by
  building providers early to read `column_names()` was rejected — a real
  cross-encoder backend would load a model purely to validate a string — so the
  eager computation is removed and the (already correct) per-fold and deployment
  computations stand. All four readers of `_feature_names` run after those.
- **`SignalContext` gained `label_space`, and it is what keeps the provider
  stateless.** Evidence is read per chunk rather than copied at construction, so
  added classes are picked up automatically and `rewrap_signal_providers` needs
  no cross-encoder case — a nicety that turned out to fall straight out of the
  context change rather than needing its own work.
- **Config coercion via `__post_init__`** on the three new dataclasses, so a
  nested JSON blob from `meta.json` rehydrates into real dataclasses on plain
  construction as well as through `from_dict`. `_build_section`'s `dc_cls(**sub)`
  would otherwise leave nested dicts in place.
- **`extra_columns` was the right home for the gap columns.** They belong to no
  single document, are gathered at the same `(rows, cols)` grid as everything
  else, and are individually `_want`-gated for free. They ride on the first
  document's matrix.
- **`CandidateView.signals` proved worth carrying.** Handing round two the
  round-one matrices is what lets `top_k` be selected against an already-computed
  ranking rather than an arbitrary slice of the shortlist, at zero cost.

## Tests

### Offline (always run)
- [x] **Byte-for-byte identical output with `"cross-encoder"` absent** — the
      regression that matters most. Same columns, same values, same `meta.json`,
      no `signals/` directory.
- [x] A `needs_candidates` provider receives the mask; a normal provider's
      context still has `candidates is None`; round-one ordering is unchanged.
- [x] A `needs_candidates` provider declaring a non-empty `candidate_features()`
      raises, naming the provider.
- [x] Candidate recall is **unchanged** by enabling the cross-encoder (it may
      reorder the shortlist, never alter it).
- [x] Empty shortlist → the reranker is never invoked (assert call count 0).
- [x] `drop_features` on every `ce_*` column → the reranker is never invoked.
      The demand-gating claim in §1, asserted as a call count, not inferred.
- [x] `ce_<d>_missing == 1` for shortlisted-but-not-reranked pairs (`top_k`
      smaller than the shortlist) **and** for a class whose view is empty; those
      pairs' `ce_<d>` is NaN, never 0.
- [x] A class with no `exclusions` yields NaN for `neg`, and its `pos_neg_gap`
      is NaN — not `ce_pos - 0`.
- [x] Evidence selection picks the query-closest entry, and the *rendered
      document* is asserted (not just the score), including per-slot truncation.
- [x] `pos_neg_gap` equals `ce_pos - ce_neg` where both are present.
- [x] Composed-vs-split config produce the documented column sets from the same
      code path (the §2 claim).
- [x] Full schema/persistence round-trip: train → save → load → predict,
      `meta.json` records `components.signals` including `"cross-encoder"`, and
      the loaded composed schema matches the trained one exactly.
- [x] Leakage: the provider is never handed example-pool text (structural).
- [x] LOO mode (`n_folds=1`) works — no self-match surface exists, but assert it.
- [x] Config round-trip both ways; a pre-T33 `meta.json` loads with the block absent.

### Skipif `sentence-transformers` absent
- [ ] `score` returns float32 `(n,)`; logits, not probabilities.
- [ ] A relevant pair outscores an irrelevant one.
- [ ] `save`/`load` round-trip reproduces identical scores.
- [ ] End-to-end train/save/load/predict with the real model.

## Acceptance criteria
- [ ] Default config is byte-for-byte identical to today, proven by T52's golden
      outputs plus a direct full-frame comparison.
- [ ] Enabling the cross-encoder is **config-only** — no edits to
      `TrainingPipeline`, `ArtifactRepository`, or `FEATURE_NAMES`.
- [ ] The offline stub exercises the entire path with no torch installed.
- [ ] Turning the columns off costs zero cross-encoder calls, asserted.
- [ ] Negative evidence reaches the fusion model as its own column *and* as an
      explicit difference, and a class lacking it is NaN rather than 0.
- [ ] Switching to an instructed/composed document set is a config change with
      no code change (the §2 claim, demonstrated by a test config).

## Out of scope

**Scoring against labeled training examples** (a cross-encoder kNN) — probably
the strongest available channel, and deliberately deferred for two independent
reasons, either of which alone would justify a separate ticket:

1. It is the only channel that touches the out-of-fold rule. Taxonomy text is
   class-owned and static; training examples are fold-dependent, so it would need
   the OOF discipline and the LOO self-mask. There is a tidy solution — rerank
   the neighbours the cheap kNN already retrieved, which are *already* masked by
   `exclude_idx`/`self_ids`, inheriting the discipline instead of reimplementing
   it — but it widens `CandidateView` to carry the per-item neighbour arrays.
2. **The model directory does not persist example text.** Dense stores embeddings
   (`infrastructure/retrieval.py:659`), BM25 stores tokenized/weight state;
   the raw texts are nowhere. This channel cannot run at inference without
   shipping the training corpus inside the artifact — a decision about artifact
   size and about sending labeled data to an air-gapped host, not an
   implementation detail.

Also out of scope: fine-tuning or distilling the cross-encoder (its `boundary`
hard negatives are the natural input, but that is an encoder-training ticket);
benchmarking the quality gain against the five-signal baseline (T40/T82);
hyperparameter search; using the cross-encoder as a standalone ranker instead of
a feature; embedding-based evidence selection (§3); raising candidate recall —
the shortlist remains the ceiling on accuracy and this ticket does not move it.
