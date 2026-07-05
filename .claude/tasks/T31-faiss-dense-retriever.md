# T31 — Optional FAISS/ANN backend behind the `DenseRetriever` port

status: todo
tier: 3
depends_on: T23

## Goal
Add an **optional** approximate-nearest-neighbour backend (FAISS) for the dense
**example-kNN** signal, selectable by config behind the existing `DenseRetriever`
port. Brute-force cosine (today's `DenseRetrieverAdapter`) stays the default and
the fallback when FAISS is absent, so default behaviour and air-gapped hosts are
unaffected.

## Why
`DenseRetrieverAdapter.knn_example_labels` is an exact `Q @ X.T` + `argpartition`
over the whole example pool (`_dense_topk`), i.e. O(n·d) per query. Fine for
thousands of examples, but it does not scale to large pools. ANN indexes give
sub-linear query time at a small recall cost. Only the **example-kNN** path needs
it; `prototype_similarity` and `description_similarity` are `(b, C)` mat-muls over
the (small) class set and stay brute-force.

## Design notes (read before starting)
- **Seam:** the registry (T23) currently covers encoder/fusion/calibrator only —
  there is **no retriever seam yet**. This ticket adds one mirroring T23: a
  `RetrievalConfig.dense_kind` (default `"bruteforce"`), `register_retriever` /
  `build_retriever`, and persistence dispatch. The brute-force adapter becomes
  the default registered backend.
- **Cosine == inner product:** embeddings are already L2-normalized (core
  invariant), so a FAISS inner-product index (`IndexFlatIP`, or IVF/HNSW-IP)
  yields cosine directly. Assert unit-norm at build.
- **Portability:** the model dir must stay portable (stdlib + numpy + json +
  native XGB/ST only). Do **not** persist a native FAISS index. Keep `dense.npz`
  as the source of truth and **(re)build the ANN index in memory at load** from
  the stored `example_emb`. On-disk format is unchanged.
- **Contract parity:** `knn_example_labels` must return `(labels (b,k) int with
  -1 pad, sims (b,k) float with NaN pad)` exactly like `_dense_topk`, including
  the `k > n_examples` padding and the empty-batch `(0,k)` behaviour.
- **Optional dependency:** add `faiss-cpu` as an extra (like `lightgbm`); lazy
  import; if missing, `build_retriever("faiss")` falls back to brute force with a
  logged warning. Tests `skipif` FAISS absent.

## Files to add/change
- `text_classifier/infrastructure/retrieval.py` — `FaissDenseRetriever` (build
  ANN from `DenseState.example_emb`; reuse `DenseState` for the rest).
- `text_classifier/infrastructure/registry.py` — retriever spec + registration.
- `text_classifier/config.py` — `RetrievalConfig.dense_kind` (+ ANN params).
- `text_classifier/infrastructure/persistence.py` — build the configured dense
  retriever on load (numpy state unchanged); record `components.dense_retriever`.
- `pyproject.toml` — `faiss = ["faiss-cpu"]` optional extra.
- `tests/unit/test_retrieval_faiss.py` (skipif); `tests/integration/test_e2e.py`
  (parametrize `dense_kind`).

## Tests
- [ ] (skipif) FAISS recall@k vs brute force ≥ 0.95 on synthetic embeddings; same
      `(idx, sim)` contract incl. `-1`/`NaN` padding and `k > n`.
- [ ] Inner-product index reproduces cosine on unit-norm inputs within tolerance.
- [ ] Fallback: with FAISS import monkeypatched absent, `build_retriever("faiss")`
      returns the brute-force adapter and still works.
- [ ] e2e (skipif): train → save → load → predict with `dense_kind="faiss"`;
      `dense.npz` unchanged; ANN rebuilt on load; predictions match brute force
      within tolerance.
- [ ] Default (`bruteforce`) behaviour and all existing retrieval tests unchanged.

## Acceptance criteria
- [ ] FAISS backend selectable by config; brute force remains default + fallback.
- [ ] Model directory format unchanged (no native FAISS file); ANN built on load.
- [ ] Contract (shapes, padding, empty batch) identical to brute force.
- [ ] FAISS absent → tests skip; full system still works.

## Out of scope
ANN for prototype/description similarity (small `(b, C)` mat-muls). GPU FAISS.
Index hyper-parameter search beyond a sane default. The lexical/BM25 path (T32).
