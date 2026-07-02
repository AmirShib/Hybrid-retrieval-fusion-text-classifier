# T28 — Encode-time kwargs + asymmetric query/document encoding

status: todo
tier: 2
depends_on: T24

## Goal
Let users reach `SentenceTransformer.encode(...)` kwargs from config, and give the
`TextEncoder` port an (optional, backward-compatible) query/document distinction so
instruction-tuned encoders (E5, BGE, GTE, …) can be used correctly.

## Why
`EncoderConfig.params` reaches only the **constructor**
(`SentenceTransformer(model, device=..., **params)` — `encoder.py`,
`registry.py`). Encode-time behaviour is hardcoded in
`SentenceTransformerEncoder.encode` (`encoder.py:47-55`): `batch_size` aside,
users cannot set `prompt`/`prompt_name` (E5/BGE instruction prefixes),
`truncate_dim` (Matryoshka), or `precision`. Worse, the port has a single
`encode()` used for items *and* class descriptions/examples alike, while most
current MTEB-leading models require different prefixes for queries vs documents
("query: …" vs "passage: …"). Today the package silently encodes both sides
identically, which measurably degrades the three dense signals on those models.

## Design
**Config** (`config.py`): add to `EncoderConfig`:
- `encode_kwargs: Dict[str, Any] = {}` — merged into every `model.encode(...)`
  call; user keys win over our defaults *except* the invariants (see below).
- `query_prompt: Optional[str] = None`, `document_prompt: Optional[str] = None` —
  literal prefixes prepended to texts (works on every ST version), OR
  `query_prompt_name`/`document_prompt_name` passed as `prompt_name` when set
  (newer ST versions with model-card prompts). Support both; document precedence
  (explicit prompt wins over prompt_name).

**Port** (`domain/ports.py`): add concrete default methods to `TextEncoder`:
```
def encode_queries(self, texts):   return self.encode(texts)
def encode_documents(self, texts): return self.encode(texts)
```
Purely additive — every existing adapter and test double keeps working unchanged.
`SentenceTransformerEncoder` overrides both to apply the respective prompt +
`encode_kwargs`.

**Call sites** — route by role:
- queries: `TrainingPipeline._build_oof` (`q_emb = enc.encode(va_texts)`) and
  `InferencePipeline.predict` (`a.encoder.encode(texts)`) → `encode_queries`.
- documents: `DenseRetrieverAdapter.build` (example corpus + descriptions,
  `retrieval.py`) and the fine-tune path where relevant → `encode_documents`.

**Invariants**: embeddings must stay L2-normalized (dot == cosine, CLAUDE.md).
Force `normalize_embeddings=True` and `convert_to_numpy=True` after merging user
`encode_kwargs` (ours win for these two keys; log a warning if the user tried to
override them). `truncate_dim` interacts with normalization — ST normalizes after
truncation when asked; add a test asserting unit norms whatever the kwargs.

**Persistence**: nothing new — `EncoderConfig` already serializes into
`meta.json`'s config block, and `load` already receives the config
(`registry.py`), so prompts/kwargs survive the round-trip and apply at inference.

## Files to change
- `text_classifier/config.py` — new `EncoderConfig` fields.
- `text_classifier/domain/ports.py` — default `encode_queries`/`encode_documents`.
- `text_classifier/infrastructure/encoder.py` — ST adapter overrides.
- `text_classifier/application/training.py`, `application/inference.py`,
  `infrastructure/retrieval.py` — role-routed call sites.
- `tests/unit/test_encoder_tfidf.py`, `tests/_doubles.py`, new unit tests.

## Tests
- [ ] Double-based: a recording encoder asserts queries go through
      `encode_queries` and corpus/descriptions through `encode_documents` in both
      pipelines (no torch needed).
- [ ] Prompt logic: `query_prompt="query: "` prepends exactly once; document side
      untouched; `None` == today byte-for-byte.
- [ ] `encode_kwargs` merge: user key passed through; attempts to override
      `normalize_embeddings`/`convert_to_numpy` are ignored with a warning.
- [ ] Output is unit-norm under every kwargs combination (property test on the
      adapter with a stubbed model).
- [ ] Round-trip: a config with prompts saves/loads and inference uses them.

## Acceptance criteria
- [ ] Default config → byte-identical behaviour to today (regression guard).
- [ ] An E5-style setup is expressible entirely via config, CLI-reachable once a
      `--config` flag exists (T74).
- [ ] No new required dependency; doubles/TF-IDF/hashing unaffected.

## Out of scope
Per-signal prompts (different prompt for descriptions vs examples); ONNX/quantized
encode paths; changing the fine-tune loss (candidate future ticket).
