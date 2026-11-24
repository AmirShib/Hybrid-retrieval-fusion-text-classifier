# T97 — Contrastive lexical signal (round two)

status: in-review
tier: 3
depends_on: T33 (two-round assembler), T34 (phase 2)

## Goal
Add an optional round-two signal that compares each shortlisted candidate's words
with the *other* candidates' words for the same item, and writes the result into
that candidate's row.

Once the shortlist exists, the final choice often turns on one or two short tokens:
`זירו`, `ללא סוכר`, `500 מ"ל` vs `1.5 ליטר`. No current signal is built for that:

- **BM25 weights words by corpus-wide rarity.** A word that is common overall but
  is the only thing separating the top two candidates gets a low weight. What
  matters is how rare it is *within the shortlist*.
- **BM25 only adds positive evidence.** A query word that matches competitor B
  but not candidate A does not count against A.
- **Fusion is pointwise.** A's row cannot see that B matched a word A lacks.
  The T81 margin columns compare total scores, not which words are shared.

Off by default. With `"contrastive-lexical"` absent from `PipelineConfig.signals`,
the system is byte-for-byte unchanged and no columns appear.

## Changes from the first draft (found while implementing)
- **`lex_counter_hits` is the strongest single competitor, not the total.** The
  total weight of matched words that `c` lacks is `W - lex_local_overlap`, so the
  total share always equals `1 - local / W` and adds nothing. The column is now
  `max` over rivals `c'` of the weight of words `c'` matched and `c` did not.
- **`lex_local_bm25` is named `lex_local_overlap`.** Word sets have no term
  frequency or length normalization, so it is a weighted overlap, not BM25.
- **`views: List[str]` instead of `EvidenceSpec`.** A word set has no use for
  `select`, `label` or `max_chars`. Allowed: `description`, `core`, `examples`,
  `inclusions` (positive evidence only).
- **`neighbors` is capped at 5 by validation** (default 0 = off; 2 recommended).
- **A class with no text in the chosen views is not scored** (NaN) and does not
  count toward `K`.

## Design

### Provider shape
`ContrastiveLexicalSignalProvider` in `infrastructure/contrastive.py`:
- `name = "contrastive-lexical"`, `needs_candidates = True`, empty
  `candidate_features()`. It reorders the shortlist and never extends it, so
  candidate recall is unchanged.
- Scores **every** candidate in `ctx.candidates` (the full union). There is no
  `top_k`: the work is a set operation over a few dozen classes per item.
- Reads class text from `ctx.label_space` per chunk, like the cross-encoder, so
  added classes are picked up with no `rewrap_signal_providers` case.
- Registered in `infrastructure/registry.py`; settings in
  `RetrievalConfig.contrastive_lexical` (a `ContrastiveLexicalConfig`), next to
  `cross_encoder` and `splade`.

### Tokenization
The provider builds its analyzer from `retrieval.bm25_token_kwargs`, so it always
tokenizes exactly as BM25 does. **No built-in normalization**: preprocessing is
the user's choice. Each text becomes a *set* of tokens; term frequency is ignored,
because item names and descriptions rarely repeat a word.

### A candidate's word set `T(c)`
Built from evidence sources, in this order:

| Source | Default | Notes |
|---|---|---|
| `description` | on | Always present |
| Codebook views (`core`, `examples`, `inclusions`) | off | Opt-in via `views` |
| Training-data neighbours | off (`neighbors: 0`) | Opt-in; validated to at most 5, 2 recommended |

Training neighbours are off by default because training labels can be wrong, and a
single mislabeled example can put a deciding word into the wrong class's set.

**Neighbours, when enabled:** the provider keeps its own example pool, fitted per
fold through `SignalProvider.fit`, exactly like SPLADE (T94), so the out-of-fold
rule holds. For item *q* and candidate *c*, take the `neighbors` examples of
class *c* with the largest token overlap with *q* (ties broken by pool index).
In leave-one-out mode (`ctx.self_ids`) an item's own row is excluded.
`with_added_examples` extends the pool for the no-retrain update path.

### Weights and columns
For an item with query token set `Q` and shortlist `S` of size `K`:
- `df(t)` = the number of candidates in `S` whose set contains `t`
- `w(t) = log(K / df(t)) / log(K)`: 1 for a word only one candidate has, 0 for a
  word every candidate has, comparable across shortlist sizes
- `M` = the query words matched by at least one candidate; `W = Σ_{t∈M} w(t)`

| Column | Definition (per candidate `c`) | NaN when |
|---|---|---|
| `lex_local_overlap` | `Σ_{t ∈ Q ∩ T(c)} w(t)` | `c` not in the shortlist, or `K = 1` |
| `lex_unique_hits` | `Σ_{t ∈ Q ∩ T(c), df(t)=1} w(t) / W` | as above, or `W = 0` |
| `lex_counter_hits` | `max_{c' ≠ c} Σ_{t ∈ (Q ∩ T(c')) \ T(c)} w(t) / W` | as above, or `W = 0` |
| `lex_shortlist_size` | `K` (per query, same on every row) | never |
| `lex_evidence_docs` | documents in `T(c)` (only when neighbours are on) | `c` not in the shortlist |

`lex_local_overlap` is a **true 0** when `c` is scored and shares no query word: in a
comparison within the shortlist, "matched nothing" is information. Not being
scored is what NaN means here, as everywhere else.

Words matching no candidate are ignored: a typo should not count against anyone.

`lex_local_overlap` gets the assembler's generic derivations (`rank`, `margin` and a
`q_gap`), named `rank_lex_local`, `margin_lex_local`, `q_gap_lex_local`. The share
columns and the size are plain `extra_columns`/`extra_scalars`.

### Config
```python
@dataclass
class ContrastiveLexicalConfig:
    views: List[str] = ["description"]   # subset of description, core, examples, inclusions
    neighbors: int = 0                   # training examples per candidate; 0 = off, max 5
```
Enabled with `signals: ["dense", "lexical", "contrastive-lexical"]`.

## Tests (offline, `HashingEncoder`)
- The worked example: query `קוקה קולה זירו 500 מ"ל`, candidates A `קוקה קולה 500 מ"ל`,
  B `קוקה קולה זירו 1.5 ליטר`, C `פפסי מקס 500 מ"ל`. Assert every column by hand,
  including A's counter-evidence on `זירו`.
- Weights: a word in all candidates weighs 0; a word only one candidate has
  weighs 1 at any shortlist size.
- Counter-evidence is the strongest single competitor, not the sum over rivals.
- NaN rules: outside the shortlist, `K = 1`, `W = 0`.
- Tokenization follows `bm25_token_kwargs` (a custom `token_pattern` changes
  the result).
- Neighbours: off by default; capped at 2; out-of-fold (an item never sees
  itself, in fold mode or leave-one-out mode); `with_added_examples` works.
- Pipeline: absent from `signals` → byte-identical to today (golden test);
  present → columns appear in `meta.json` and the model saves and reloads with
  no pickle.
- Vectorized: no per-candidate Python loop on the scoring path (one sparse
  incidence matrix per chunk).

## Evaluation
On `examples/coicop_hebrew`, normalized with `HebrewItemNormalizer` and the BM25
`token_pattern` recommended in T95. First, the ceiling:
1. **Near-miss rate:** errors where the true class was in the shortlist and
   ranked second. That is the most this signal can win back.

Then compare with and without the signal: top-1 accuracy, and coverage at the
tuned target precision. Report feature importance for the new columns.

### Result (2026-11-24)
`examples/coicop_hebrew/_data.csv`, normalized with `HebrewItemNormalizer` v2 and
the T95 `token_pattern`. Split grouped by cleaned name (no exact duplicate on both
sides); classes with < 3 training rows dropped. Dense signal on the `tfidf`
encoder (no torch in the run environment), 3 folds, target precision 0.95.
Class descriptions are the bare Hebrew category names (no codebook).

| 40k train / 8k test, 76 classes | Top-1 | True class in top 3 | Coverage @ precision |
|---|---|---|---|
| dense + lexical | 92.66% | 97.64% | 93.5% @ 95.4% |
| + contrastive, description only | 92.73% | 97.66% | 93.2% @ 95.6% |
| + contrastive, 2 neighbours | **93.41%** | **97.89%** | **94.3% @ 95.9%** |

Paired on the same test items: with 2 neighbours, 96 fixed / 36 broken (McNemar
p ≈ 2e-7, ~10% fewer errors); description only, 23 / 18 (noise). A 6k-train run
gave the same pattern (37 / 22 and 12 / 9).

Reading: with neighbours the signal is a clear win, and coverage and precision
both rise at the operating point. Description-only does nothing here because the
descriptions are category names that share almost no words with product names;
a codebook with `examples`/`inclusions` would give it something to contrast.
Still to measure: a real bi-encoder (its shortlist differs from TF-IDF's).

Speed: build is 0.23 s per 4,096 items x 30 candidates with neighbours off,
~4 s against a 100k-example pool with them on (the neighbour overlap search).

## Acceptance
- Off by default and byte-identical when off.
- Columns, NaN rules and weights exactly as specified above.
- No regression in the quality benchmark (T52) with it off; measured result
  with it on recorded in this ticket.

## Out of scope
- Normalization (the user's step; see T95).
- Counter-evidence from `exclusions` / `sibling_distinctions` views. Those are
  negative evidence of a different kind, and T33's cross-encoder already
  consumes them.
- Weighting competitors by their round-one rank. Revisit only if weak
  candidates turn out to dilute the weights.
