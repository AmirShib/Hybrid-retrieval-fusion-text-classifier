# T88 — Encode the corpus once, not once per fold

status: todo
tier: 8
depends_on: —

## Goal
On the shared-encoder path (the default), encode each distinct text exactly once
per training run instead of once per fold. Roughly a **5x reduction in encoder
work** at `n_folds=5`, with no GPU, no new port and no change to any feature value.

## Why
Found while scoping T83. `_build_oof` (`application/training.py:399`) loads one
shared encoder and reuses it for every fold:

    shared = None
    if not self._use_per_fold_encoder():
        shared = self._load_shared_encoder()
    for fold, (tr, va) in enumerate(skf.split(texts, y)):
        enc = self._encoder_for_split(tr, texts, y, label_space, shared)
        tr_texts = [texts[i] for i in tr]
        dense = DenseRetrieverAdapter.build(enc, tr_texts, y[tr], label_space, ...)

and `DenseRetrieverAdapter.build` (`infrastructure/retrieval.py:329-332`) then does:

    emb  = encoder.encode_documents(texts)                      # ~0.8n, every fold
    desc = encoder.encode_documents(label_space.descriptions)   # all C, every fold

For a frozen shared encoder `encode_documents` is a **pure function of the text**.
So at `n_folds=5` a training run currently performs:

| | encodes today | distinct texts |
|---|---|---|
| example pool | 4n (OOF) + n (deployment index) = **5n** | n |
| class descriptions | 5C (OOF) + C (deployment index) = **6C** | C |

Four fifths of the pool encoding and five sixths of the description encoding is
recomputation of vectors already in hand. On a sentence-transformer encoder that
is the dominant cost of a training run, and it compounds directly with `n_folds`
— exactly the axis that hurts when the complaint is training throughput.

This is cheaper than everything else in Tier 8 and independent of all of it: it
needs no device work, no `ArrayOps` port, and no change to the feature graph.

## Design

**1. Encode once, slice per fold.** In `_build_oof`, when
`_use_per_fold_encoder()` is False, encode all `texts` and all
`label_space.descriptions` up front with `shared`, then hand each fold its slice.

**2. `DenseRetrieverAdapter.build_from_embeddings(...)`** — a classmethod taking
precomputed `example_emb` / `description_emb` instead of an encoder. `build()`
stays as-is (encode, then delegate), so every existing caller and every test
double is untouched.

**3. Query embeddings too.** `enc.encode_queries(va_texts)` is already 1n total
across folds (the folds partition the items), so there is nothing to save there —
*unless* query and document prompts are identical (both `None`, the symmetric
default), in which case the query pass is also redundant with the document pass.
**Do not exploit that.** The two roles are conceptually distinct, the equality is
a coincidence of configuration, and collapsing them would silently break the
moment someone sets an E5/BGE prompt. Encode queries separately; the saving is
not worth the trap.

**4. The per-fold-encoder path is untouched.** When `use_per_fold_encoder=True`,
or the encoder is corpus-dependent (TF-IDF — `encoder_is_corpus_dependent`), each
fold has a *different* encoder and the cache is invalid. Guard on the same
`_use_per_fold_encoder()` predicate that already selects the shared path, so the
rigorous mode keeps its current semantics exactly.

**5. Leakage is unaffected, and this must be argued explicitly.** The concern
would be that sharing embeddings across folds leaks. It does not: a frozen shared
encoder already sees every item's *text* at load time regardless of folds — that
is precisely why `_use_per_fold_encoder()` exists and why a corpus-dependent
encoder is excluded from this path. What must stay fold-local is which
embeddings enter *fold k's index* (`emb[tr]`, never `emb[va]`), and slicing
preserves that exactly. The T06 leakage regression is the check.

## Expected effect
At `n_folds=5`: pool encodes 5n → n, description encodes 6C → C. Encoder cost of a
training run drops to ~1/5. Memory rises: the full `(n, d)` float32 matrix is held
for the whole OOF loop rather than a fold-sized slice — at n=100k, d=384 that is
150 MB, which is well worth it, but T83 should record the curve and the ticket
should note the crossover where it is not.

## Files to change
`application/training.py`, `infrastructure/retrieval.py`,
`tests/unit/test_retrieval.py`, `tests/integration/test_e2e.py`,
`tests/integration/test_leakage.py`, `CHANGELOG.md`.

## Tests
- [ ] **Byte-identity**: OOF frame, fusion model, thresholds and evaluation are
      bit-for-bit unchanged on a fixed corpus with the default config.
- [ ] Encoder call counting: a counting encoder double asserts exactly `n + C`
      document encodes across a full `n_folds=5` run (today: `5n + 6C`).
- [ ] `use_per_fold_encoder=True` still refits per fold and still encodes per fold
      — the rigorous path is provably unchanged.
- [ ] TF-IDF (corpus-dependent) encoder still takes the per-fold path.
- [ ] T06 leakage regression green; `emb[va]` never reaches fold `va`'s index.
- [ ] `build()` and `build_from_embeddings()` produce identical adapters.

## Acceptance criteria
- [ ] Distinct-text encodes per training run = n + C on the shared-encoder path.
- [ ] Byte-identical outputs; T52 benchmark floors green.
- [ ] Per-fold-encoder and corpus-dependent paths unchanged.
- [ ] Speedup measured and recorded on a real sentence-transformer encoder.

## Out of scope
Caching embeddings *across runs* (a persisted embedding cache is its own ticket
with its own invalidation problem). Device residency of the cached matrix — that
is T85, and this ticket's slicing is what makes it worth uploading once.
