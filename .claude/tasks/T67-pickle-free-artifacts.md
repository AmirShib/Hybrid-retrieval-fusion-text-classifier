# T67 — Pickle-free model artifacts (loading a model dir must not execute code)

status: todo
tier: 6
depends_on: T60

## Goal
Remove `pickle` from the model-directory format. Every artifact becomes inert data
(JSON / `.npz` / native XGBoost/LightGBM/SentenceTransformer formats), so loading a
model directory can never execute attacker-controlled code. Old directories keep
loading via a legacy fallback.

## Why
`pickle.load` executes arbitrary code embedded in the file. Our model directories
are exactly the artifact that gets *moved between machines* — built on a connected
host, carried to an air-gapped one (CLAUDE.md invariant: "a trained model directory
must be portable ... ships to an air-gapped host"). That transfer is the classic
spot where an operator loads a directory they did not build. Current pickle surface
(`grep pickle text_classifier/`):

- `persistence.py` — `lexical.pkl`: the whole `LexicalRetrieverAdapter` (two
  `CountVectorizer`s + BM25 weight matrices + labels).
- `fusion.py` — `calibrator.pkl` for all three calibrators (`IsotonicCalibrator`
  pickles a fitted `IsotonicRegression`; `_ParametricCalibrator` pickles a
  `LogisticRegression`), and `isotonic.pkl` inside the `XGBRankerFusionModel` dir.
- `encoder.py` — the TF-IDF encoder pickles its fitted `TfidfVectorizer`.

None of these needs pickle — each is a small set of arrays plus scalars.

## Design
Per artifact, serialize the *numeric state*, not the sklearn object; rebuild the
object on load:

- **BM25 / lexical** (`retrieval.py`, `persistence.py`): replace `lexical.pkl` with
  `lexical.npz` + `lexical.json`. Per index (examples, descriptions): the sparse
  weight matrix `_Wt` via `scipy.sparse.save_npz`-style keys packed into the npz
  (`data/indices/indptr/shape`), the vocabulary as a JSON `{term: column}` map, and
  scalars `k1`, `b`, `n_docs`, plus the analyzer config actually used
  (`cv_kwargs`). On load, rebuild a `CountVectorizer(vocabulary=...)` with those
  kwargs — with a *fixed* vocabulary it never refits, so transform behaviour is
  identical. `example_labels` goes in the npz. Give `BM25Index` explicit
  `to_state()/from_state()` instead of relying on `__dict__`.
  Note: `cv_kwargs` must stay JSON-clean for this to work — validate/document that
  custom callable analyzers are unsupported for persistence (they already can't
  round-trip safely under pickle-by-value semantics anyway).
- **Isotonic calibrator** (`fusion.py`): persist the fitted breakpoints
  (`iso.X_thresholds_`, `iso.y_thresholds_`) + `out_of_bounds` in an `.npz`; load
  by re-fitting `IsotonicRegression` on the breakpoints themselves (exact — the
  interpolant through its own breakpoints reproduces the curve). Same for the
  XGBRanker's isotonic head.
- **Parametric calibrators** (`fusion.py`): persist `coef_`, `intercept_`,
  `classes_`, and the `constant` fallback in JSON; rebuild `LogisticRegression`
  and set the fitted attributes (document the sklearn-version caveat; these three
  attributes have been stable for a decade).
- **TF-IDF encoder** (`encoder.py`): vocabulary JSON + `idf_` npz + the vectorizer
  params; rebuild `TfidfVectorizer(vocabulary=...)` and set `idf_`.
- **Legacy fallback**: each `load` first looks for the new file(s); if only the old
  `.pkl` exists, load it with a one-line `log.warning` ("legacy pickle artifact;
  re-save to upgrade"). Registry `filename` entries change to the new names;
  the fallback lives in the component `load` functions, not `ArtifactRepository`.
- **Docs**: update the `persistence.py` module docstring layout table and the
  CLAUDE.md invariant line ("stdlib pickle + numpy + json" → "numpy + json +
  native formats; no pickle").

## Files to change
- `text_classifier/infrastructure/retrieval.py` — `BM25Index.to_state/from_state`.
- `text_classifier/infrastructure/persistence.py` — lexical save/load + docstring.
- `text_classifier/infrastructure/fusion.py` — calibrators + ranker head.
- `text_classifier/infrastructure/encoder.py` — TF-IDF persistence.
- `text_classifier/infrastructure/registry.py` — new filenames.
- `CLAUDE.md`, `README.md` — invariant wording.
- `tests/integration/test_e2e.py`, `tests/unit/test_fusion.py`,
  `tests/unit/test_encoder_tfidf.py`, `tests/unit/test_retrieval.py` — round-trips.

## Tests
- [ ] Every save→load round-trip reproduces scores within exact equality (BM25,
      isotonic) or ~1e-9 (logistic) on a probe batch — parametrized over all
      calibrator kinds and encoder kinds.
- [ ] A directory saved by the *old* code (build one in-test by monkeypatching or
      keep a tiny checked-in fixture) still loads, with the warning.
- [ ] `grep -r "pickle" text_classifier/` finds no `pickle.load` on the artifact
      path (test can assert the import is gone from the three modules).
- [ ] Wheel-smoke CI path (train→infer→eval) unchanged and green.

## Acceptance criteria
- [ ] Fresh model directories contain no `.pkl` files.
- [ ] Old directories load with a warning; new inference results match old within
      the tolerances above.
- [ ] No new dependencies; still stdlib + numpy + scipy + sklearn.

## Out of scope
Signing/hashing the model directory (a possible follow-up: a manifest of file
sha256s written at save, verified at load); sandboxing sentence-transformers'
own loading (upstream format, out of our hands); safetensors migration.
