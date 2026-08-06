# T89 — Reuse the pooled embeddings as query embeddings (stop the second full encode)

status: in-review
tier: 8
depends_on: T88, T28

## Goal
On the shared-encoder path (the default), stop re-encoding held-out texts as
*queries* when their vectors are already in T88's whole-pool cache. Cuts encoder
work on that path from **2 full passes to 1**, with no change to any feature value.

Ships with an explicit config override so the behaviour can be forced on or off
rather than only inferred.

## Why
T88 shared the **document** side and left the **query** side untouched.
`_build_oof` (`application/training.py:555`) encodes the whole pool once:

    shared_emb, shared_desc_emb = self._shared_document_embeddings(texts, label_space, shared)

Each fold's dense index is then built by slicing `shared_emb[tr]` — no re-encode,
exactly as T88 intended. But at `application/training.py:622-623` the same loop does:

    va_texts = [texts[i] for i in va]
    q_emb = enc.encode_queries(va_texts)

`va_texts` is an exact subset of `texts`, and across the folds the `va` sets
**partition** the item list — every item is held out exactly once. So the loop
performs, in aggregate, one additional full pass over the corpus.

With the default config this is pure recomputation. `EncoderConfig`'s four prompt
fields (`config.py:67-70`) all default to `None`, and with them unset
`encode_queries` and `encode_documents` reduce to the identical call
`_encode(texts, None, None)` (`infrastructure/encoder.py:188-192`) — same text,
same frozen weights, same vectors.

| | encodes today (post-T88) | distinct texts |
|---|---|---|
| example pool, as documents | n | n |
| example pool, as queries (across folds) | n | 0 new |
| **total** | **2n** | **n** |

Note the waste is **flat in `n_folds`**, not proportional to it: 5 folds and 10
folds both re-encode each item exactly once. This is a straight 2x on the
encoder, which on a sentence-transformer is the dominant cost of a training run.

`_build_loo` (`application/training.py:669`) has the same defect in starker form:
it encodes the *entire* pool as queries immediately after `_build_deployment_index`
populated the cache with that identical list.

## Where this does *not* apply (and why that falls out for free)
The reuse is only ever valid for a **frozen, shared** encoder. When the encoder
is fine-tuned per fold, embeddings computed before training are stale and reusing
them would be a correctness bug, not an optimization.

That case is already excluded structurally, not by a new check:
`shared_emb` is only populated under `if not self._use_per_fold_encoder():`
(`application/training.py:543`), and `fit_encoder` — the function that actually
trains — is reachable only from `_encoder_for_split` (`:504`) and the
`_use_per_fold_encoder()` branch of `_build_deployment_index` (`:879`). The
shared path calls `build_encoder` (`:172`), which loads and never trains. So
`shared_emb is not None` is itself the guard.

Unaffected paths, stated explicitly so a reviewer can confirm nothing is lost:
- `--per-fold-encoder` and corpus-fitted encoders (TF-IDF): each fold encodes
  `tr_texts` and `va_texts`, disjoint sets, against that fold's own weights.
  No redundancy exists there to remove.
- `fit_encoder`'s multi-epoch best-epoch loop, which re-encodes a holdout every
  epoch (`cli/train.py:101-104`). Weights change per epoch; that re-encode is
  required. This ticket does not touch `infrastructure/encoder.py`'s training loop.
- `_featurize_external` (`:720`): external val/test text was never in the pool.

## Design

**1. Reuse is gated on a capability the encoder advertises, not on config alone.**
Reading `EncoderConfig`'s prompt fields directly would be wrong for a *custom*
`TextEncoder` plugged in through the port (or injected as `shared_encoder`), which
could distinguish roles internally without using those fields — we would hand it
document vectors where it expects query vectors, silently.

Follow T85's duck-typed precedent (`application/training.py:181` probes
`hasattr(encoder, "set_array_backend")`): probe for a `roles_share_encoding`
capability and **do not reuse when it is absent**. Conservative by default;
unknown encoders keep today's behaviour.

- `SentenceTransformerEncoder.roles_share_encoding` mirrors `_encode`'s own
  precedence (`encoder.py:200-203`, where an explicit literal prompt wins over a
  `prompt_name`): compare the *effective* `(prompt, prompt_name)` pair per role,
  not the four raw fields.
- `TfidfEncoder` / `HashingEncoder`: role-agnostic by construction → `True`.

**2. The override the behaviour can be forced with.**
`EncoderConfig.reuse_query_embeddings: str = "auto"`, following the
`array_backend="auto"` precedent (`config.py:265`) where auto-detection is the
default and an explicit value always wins:

| value | meaning |
|---|---|
| `"auto"` (default) | reuse when the shared cache exists **and** the encoder advertises role-symmetry |
| `"always"` | reuse whenever the shared cache exists, even if an asymmetry was detected — the caller asserts the roles are equivalent. Logs a warning when it overrides a detected asymmetry, so it never happens silently |
| `"never"` | always re-encode: exactly today's behaviour, byte for byte. The escape hatch for a custom encoder whose role asymmetry we cannot detect |

`"always"` still cannot reuse what does not exist — on the per-fold-encoder path
there is no cache, and it stays a no-op there. Document that plainly; the flag
changes *policy*, never the fine-tuning correctness rule above.

CLI: `--reuse-query-embeddings {auto,always,never}` on the train CLI, wired
through the existing "explicit flag beats `--config` beats default" precedence.

**Home for the field:** `EncoderConfig`, because the validity condition is
entirely a property of encoder role semantics. `TrainingConfig` was the other
candidate (it only affects the training pipeline) — recorded here so the choice
is reviewable rather than silently made.

**3. Config round-trip.** `to_dict`/`from_dict` must read the new field with a
default (`data.get("reuse_query_embeddings", cls().reuse_query_embeddings)`,
matching the `array_backend` handling at `config.py:473`) so a model dir saved
before this ticket still loads.

## Files to change
`application/training.py` (`_build_oof`, `_build_loo`), `infrastructure/encoder.py`
(the `roles_share_encoding` capability), `config.py`, `cli/train.py`,
`tests/unit/test_encoder.py`, `tests/integration/` (byte-identity), `CHANGELOG.md`.

## Progress (2026-08-06) — implemented

Delivered as designed. `tests/unit/test_query_embedding_reuse.py` (18 tests) is
the new coverage; full suite 814 passed, 2 skipped (both pre-existing lightgbm
skips). Notes on what deviated or was learned:

- **The capability lives on the concrete encoders, not the port.** Adding a
  defaulting `roles_share_encoding` to `TextEncoder` was rejected: the port's
  default role methods both delegate to `encode`, so a base-class `True` would
  be inherited by any subclass that overrides *both* role methods asymmetrically
  — exactly the silent-wrongness case this ticket exists to prevent. It is a
  property on `SentenceTransformerEncoder` and a class attribute on
  `TfidfEncoder`/`HashingEncoder`; `domain/` is untouched.
- **One existing test needed its premise updated, not weakened.**
  `test_encoder_asymmetric.py::test_training_routes_examples_and_descriptions_as_documents`
  (T28) asserted role routing by counting `encode_queries` calls with a
  `HashingEncoder` double. That double is role-symmetric, so under `"auto"` the
  pipeline correctly stops calling `encode_queries` and the test could no longer
  observe what it protects. Pinned to `reuse_query_embeddings="never"` so T28's
  routing assertion stays intact on its own terms, with the reuse path covered
  separately.
- **Bit-identity verified beyond the golden tests.** Directly compared the full
  out-of-fold frame between `"never"` and `"auto"` on a 6-class/72-item run:
  identical shape, identical column list, and all 40 numeric columns equal under
  `array_equal(..., equal_nan=True)` — NaN placement included, which is the
  invariant that matters most here (`NaN` means "signal did not retrieve this
  class", never a value to be imputed).
- **Per-fold guard is asserted structurally**, not by call count: the test asserts
  `_shared_pool_emb is None` after a `use_per_fold_encoder` run, which is the
  actual reason the fine-tuning path is safe. (It uses `tfidf`, since `hashing`
  is not corpus-fittable and cannot exercise that path at all.)
- **Measured on GPU** (2026-08-06). This host is *not* CPU-only as T83/T85
  assumed — it is an Apple M5 with a working MPS backend (torch 2.12.1,
  `mps.is_available()` True, `SentenceTransformer` lands on `mps:0`). Measured
  with the default `all-MiniLM-L6-v2`, 3,000 items / 60 classes / `n_folds=5`,
  comparing `"never"` vs `"auto"` on an identical run:

  | corpus | encoder time | out-of-fold wall-clock |
  |---|---|---|
  | 3,000 x ~48 words (uniform) | 6.85s -> 3.05s (**-55%**) | 8.13s -> 4.32s (-47%) |
  | 3,000 x 6-160 words (variable) | 10.53s -> 4.10s (**-61%**) | 11.66s -> 5.22s (-55%) |

  Encoder texts drop 6,060 -> 3,060 exactly as predicted (`n + C` documents, zero
  queries). The variable-length case gains more because the removed pass carried
  its share of the long documents.

  Hitting this required the `import xgboost`-first workaround `tests/conftest.py`
  documents — the torch/xgboost OpenMP conflict (`OMP: Error #179`) reproduces
  outside the test suite, so it is a property of the host, not of pytest.
  Worth noting in `docs/device-policy.md` if anything else here starts profiling.

- **CORRECTION to this ticket's central claim: "bit-identical" does not hold
  universally on GPU.** It holds for a deterministic host-side encoder (the test
  suite: 40/40 columns) *and* on MPS for uniform-length inputs (40/40). It does
  **not** hold on MPS for variable-length inputs: 12 of 40 columns differ, all
  continuous (`d_desc_sim`, `d_proto_sim`, `d_knn_*`, the `margin_*`/`q_gap_*`
  family), max |delta| ~2e-06 — float32 magnitude, not a logic error. The ordinal
  columns (`rank_*`, `is_*_top1`) and the candidate mask were unaffected in that
  run, but a near-tie could flip one, which would change the candidate *set* —
  the same caveat T85 records for cross-device parity.

  Mechanism: under `"never"`, `encode_queries(va_texts)` batched the held-out
  items differently than the whole-pool document pass had. Transformer batches
  pad to their longest member, and padding changes float reduction order. The
  uniform-length control coming out exactly identical is what rules out a
  slicing/ordering bug on our side — with uniform lengths the batching difference
  has no padding to express itself through.

  **The direction of this matters.** It means the *pre-T89* pipeline computed two
  slightly different vectors for the same text: one stored in the index, another
  used to query it. T89 makes an item's query vector exactly the vector that
  represents it in the index, so this is a consistency *improvement*. But the
  honest statement is "not byte-for-byte what a pre-T89 GPU run produced," and
  the CHANGELOG says so rather than claiming bit-identity.
- `mypy` currently fails on this host for an unrelated, pre-existing reason (a
  numpy stub vs. Python 3.14 mismatch, identical before this change); `ruff
  check` and `ruff format --check` are clean.

## Tests
- [x] **Byte-identical output on the default path.** Not "close" — identical.
      The T52 golden-output/benchmark net is green, plus a direct full-frame
      comparison (see Progress) and `test_auto_is_bit_identical_to_never`.
- [x] A configured `query_prompt` / `query_prompt_name` (T28's asymmetric path)
      **disables** reuse under `"auto"`, and the re-encode still happens. This is
      the regression that would otherwise be silent and scientifically wrong.
      Covered both at the capability level (`roles_share_encoding` parametrized
      over prompt combinations, including the explicit-prompt-beats-prompt_name
      precedence case) and end to end via an asymmetric encoder double.
- [x] `"never"` reproduces pre-T89 behaviour exactly.
- [x] `"always"` reuses despite a configured asymmetry, and logs the warning
      (and does *not* warn for a symmetric encoder).
- [x] An injected encoder without the capability does **not** reuse under `"auto"`
      — tested with a third-party-style encoder implementing only the port.
- [x] `--per-fold-encoder` and the TF-IDF path are untouched (no cache, no reuse).
- [x] Leakage regression (T06) green — reuse must not change *which* index an
      item is scored against, only where its query vector came from.
- [ ] Torch backend (T85): `shared_emb` may be a resident torch tensor, so the
      `shared_emb[va]` slice must index correctly there too, not just on numpy.
      **Not separately asserted.** The slice deliberately uses the identical
      indexing form as the adjacent, already-torch-exercised `shared_emb[tr]`, so
      it is covered by construction rather than by a dedicated test — a cheap
      follow-up, not attempted here.
- [x] Config round-trip: a pre-T89 `meta.json` loads with the default.
- [x] Offline smoke test (`python -m scripts.demo`) and the CLI flag (help,
      `--dump-config`, invalid-value rejection) verified by hand.

## Acceptance criteria
- [x] Default-path encoder work drops from 2 full passes to 1. Asserted exactly
      as an encode *count* (query-role texts drop from `n` to 0), and measured on
      MPS: 55-61% less encoder time, 47-55% less out-of-fold wall-clock (table
      above).
- [x] Feature values are bit-identical on the default path — **qualified**: true
      for host-side encoders and for uniform-length inputs on MPS; last-ulp
      differences appear on MPS with variable-length inputs. See the correction
      in Progress; this is a pre-existing GPU batching effect that T89 removes
      rather than causes, but the unqualified claim was wrong.
- [x] The asymmetric-encoder path (T28) is provably unaffected.
- [x] `reuse_query_embeddings` is reachable from config **and** the train CLI,
      and forcing it either way works without editing Python.

## Out of scope
The per-fold-encoder path (no redundancy to remove). `fit_encoder`'s per-epoch
re-encode (required). Overlapping encode with BM25 (T90). Fold-level parallelism.
