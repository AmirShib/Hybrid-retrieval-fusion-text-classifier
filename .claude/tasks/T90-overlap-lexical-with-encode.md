# T90 — Overlap BM25 corpus tokenization with the up-front encoder pass

status: todo
tier: 8
depends_on: T88, T32

## Goal
Start the BM25 corpus tokenization while the encoder is running, instead of after
it. Two independent stages that currently run back to back, run concurrently. No
change to any feature value.

## Why
`_build_oof` does the up-front shared work strictly sequentially
(`application/training.py:543-566`):

    if not self._use_per_fold_encoder():
        shared = self._load_shared_encoder()
        if use_shared_dense:
            shared_emb, shared_desc_emb = self._shared_document_embeddings(...)   # GPU
    example_state, desc_bm25 = (
        self._shared_lexical_state(texts, label_space) if use_shared_lexical else (None, None)
    )                                                                             # CPU

There is **no data dependency** between them. Both consume raw `texts`;
`_shared_document_embeddings` needs the encoder, `_shared_lexical_state` needs
nothing but the corpus and `RetrievalConfig`. The encoder pass is the GPU-heavy
stage and the tokenization is CPU-only — the classic case for overlapping, and
precisely the "embarrassingly parallel branches" T71 names without building.

The same pairing exists on the `n_folds == 1` path, where
`_build_deployment_index` runs first and populates both caches
(`application/training.py:885` then `:908`).

This is the concurrency half of what T32 attacks structurally. T32 removes
*duplicate* lexical work; T90 hides what remains under work already happening.
They compose and neither blocks the other.

## Design

**1. Background thread for the CPU stage; GPU stays on the main thread.**
Submit `_shared_lexical_state` to a single-worker `ThreadPoolExecutor` before the
encode, run the encode on the main thread, then join before the fold loop.

The direction is deliberate: keeping torch and the CUDA context on the calling
thread avoids a class of problems not worth inviting for no gain.

**2. Why this is safe without locks.** The two stages write to **disjoint**
`self` attributes, populated once per `run()`:
- `_shared_document_embeddings` → `_shared_pool_emb`, `_shared_desc_emb`
- `_shared_lexical_state` → `_shared_desc_bm25`, `_shared_example_counts`,
  `_shared_example_vectorizer`

Neither reads the other's fields. Both are deterministic and use no RNG
(`CountVectorizer` has none), so results cannot depend on completion order.
Confirm at implementation time that neither touches a global/seeded RNG.

**3. No thread oversubscription.** `bm25_n_jobs` threads the per-chunk kNN
mat-mul (`infrastructure/retrieval.py:298`), which is on the *query* path and not
reached here. `_shared_lexical_state` runs `tokenize_corpus`
(`retrieval.py:150`, single-threaded `CountVectorizer`) plus a description-index
fit over a small corpus — one background thread total. Re-verify rather than
assume, since T32 may change what runs in here.

**4. Error handling.** A failure in either stage must surface as itself, not as
a hang or a swallowed traceback. Specifically: if the encode raises first, the
run must not block waiting on the background thread.

**5. Gating.** Only worth doing when there is an up-front encode to hide under —
i.e. the shared-dense path with the lexical signal enabled. On the
per-fold-encoder path there is no up-front encode (the GPU work is inside the
fold loop), so it stays sequential; overlapping BM25 with the *first fold's*
fine-tune is a different, larger change and is out of scope.

**6. Opt-out.** `TrainingConfig.overlap_lexical_encode: bool = True`. Setting it
`False` restores exactly today's sequential order — the debugging escape hatch a
concurrency change should always ship with. `from_dict` reads it with a default
so pre-T90 model dirs still load.

## The honest unknown: how much this actually buys
This must be **measured, not asserted**. `CountVectorizer.fit_transform`
(`retrieval.py:150`) is Python-level per-document regex analysis and holds the
GIL. The encode side releases it (HF fast tokenizers are Rust; torch releases
around ops), so the GPU stream keeps running — but the encoder's Python-side
batch preparation contends with the tokenizer. Realistic expectation is that a
good fraction of the tokenization hides, not all of it.

If measurement shows the overlap is GIL-starved, the fallback is a
`ProcessPoolExecutor`, which gets full overlap but pays pickling the corpus in
and a `csr_matrix` + vocabulary back out. **Do not build that speculatively** —
record the thread-based number first, then decide whether the cost is earned.
Threads first is the recommendation, not the conclusion.

## Files to change
`application/training.py` (`_build_oof`, `_build_deployment_index`), `config.py`,
`tests/integration/`, `CHANGELOG.md`.

## Tests
- [ ] Output identical with the overlap on and off — both stages are
      deterministic, so this is exact equality, not a tolerance.
- [ ] `overlap_lexical_encode=False` reproduces pre-T90 ordering.
- [ ] An exception raised by the lexical stage propagates to the caller intact.
- [ ] An exception raised by the encode stage does not hang on the join.
- [ ] Paths where the overlap does not apply (per-fold encoder; lexical signal
      disabled; non-default `lexical_kind`) are untouched.
- [ ] The `n_folds == 1` path gets the same treatment and the same identity check.
- [ ] Leakage regression (T06) green.

## Acceptance criteria
- [ ] Wall-clock for the up-front phase is measured with the overlap on vs off,
      on a realistic corpus, and **the number is recorded in this ticket** —
      including if it turns out to be marginal (T83's honesty convention).
- [ ] Zero change to feature values.
- [ ] A decision recorded on threads vs processes, backed by that measurement.

## Out of scope
Fold-level parallelism and overlapping `_build_deployment_index` with the OOF
loop (both real, both more invasive; revisit after T89 lands, since T89 removes
most of what fold-level parallelism would be hiding). BM25 on device — permanently
host-side per T83's device policy. Anything that changes model outputs.
