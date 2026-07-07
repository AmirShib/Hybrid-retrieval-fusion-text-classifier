# T79 — Expose the fitted retrievers to custom feature providers

status: todo
tier: 7
depends_on: T70

## Goal
Let a custom `FeatureProvider` (T70) reuse the pipeline's already-fitted retrieval
components — the `DenseRetriever` and `LexicalRetriever` ports, and by extension the
signal providers once T34 phase 2 lands — instead of rebuilding its own index. The
provider `ctx` grows read-only handles to the fitted retrievers so a custom feature
can call `dense.prototype_similarity(...)`, `lexical.description_score(...)`,
`dense.class_freq`, etc., over the same candidate grid the assembler already built.

## Why
T70 gives users the seam to add fusion features, but its `ctx` exposes only the raw
per-(item, candidate) grid: query texts, embeddings, candidate `rows`/`cols`, label
space. The most natural class of custom features — new combinations or transforms of
retrieval evidence (a different kNN aggregation, a per-class BM25 margin, a
prototype/description interaction beyond `desc_proto_gap`) — needs the *fitted*
retrievers. Without this ticket a user must build and persist a second BM25 index or
re-embed the corpus inside their provider: duplicated state, duplicated memory, and a
second copy of the out-of-fold discipline to get wrong. The retrievers already exist
in scope at both call sites (`FeatureAssembler.assemble` takes `dense` and `lexical`
as arguments); this ticket is plumbing them one level down, not new machinery.

## Design
- Extend T70's `FeatureContext` with the port-typed handles the assembler already
  holds: `ctx.dense: DenseRetriever`, `ctx.lexical: LexicalRetriever`, and the
  configured `k_neighbors`. Ports only — providers must not see adapter internals
  (`DenseState`, `BM25Index`); anything a provider legitimately needs beyond the port
  surface is a case for widening the port, not leaking the adapter.
- Expose the signal matrices the assembler has already computed for the chunk
  (`desc_d`, `proto`, dense/lexical kNN outputs, `desc_l`) as read-only `(b, C)` /
  `(b, k)` arrays on `ctx`, so the common case — a derived feature over existing
  signals — costs zero extra retrieval calls. Direct port calls are the escape hatch
  for genuinely new queries against the fitted state.
- **Out-of-fold discipline is inherited, not re-implemented.** During training the
  assembler is invoked with fold-local retrievers; passing those same objects through
  `ctx` means provider features computed via `ctx.dense`/`ctx.lexical` are OOF-safe by
  construction. Document this as the reason providers should prefer `ctx` retrievers
  over private state (which falls under T70's constraint 3 and needs its own OOF
  handling).
- **Persistence: nothing new.** The retrievers are already persisted and rebuilt by
  the existing artifact path; at inference `ctx` carries the deployment-index
  retrievers. Providers that only consume `ctx` retrievers are stateless on disk —
  their `save`/`load` become no-ops, which is the cheapest possible custom feature.
- After T34 phase 2, `ctx` should carry the ordered provider list's fitted
  `SignalProvider`s under their registered names rather than the hardcoded pair;
  design the accessor as a name-keyed lookup (`ctx.retriever("dense")`) with `dense`/
  `lexical` as the guaranteed built-in keys so the T34 generalization is additive.

## Files to change
- `text_classifier/application/features.py` — build `ctx` with the retriever handles
  and the chunk's precomputed signal matrices.
- `text_classifier/domain/ports.py` (or wherever T70 lands `FeatureContext`) — the
  widened context type.
- docs for T70's provider-authoring guide — a worked example: a provider computing a
  BM25 description-score margin between top-1 and runner-up candidates using
  `ctx.lexical`, with no persisted state.
- tests — see below.

## Tests
- [ ] A sample provider computing a feature from `ctx.dense.prototype_similarity`
      produces identical values at train and at inference from a persisted model dir
      (parity, per T70 constraint 1) with no provider artifacts on disk.
- [ ] OOF: a `ctx`-retriever feature for item *i* computed during training never
      reflects an index containing *i* (extend the T06 leakage regression).
- [ ] Precomputed signal matrices on `ctx` match a direct port call bit-for-bit
      (no stale/reordered chunk state).
- [ ] Mutating attempts on `ctx` arrays fail (read-only views) — providers cannot
      corrupt the core features computed from the same matrices.

## Acceptance criteria
- [ ] A retrieval-derived custom feature requires implementing only
      `names()`/`compute(ctx)` — no index building, no persistence code, no OOF logic.
- [ ] Zero providers configured → byte-identical behaviour (inherits T70's guarantee;
      the ctx additions are inert when unused).
- [ ] Port-only exposure: no adapter-internal types appear in the provider API.

## Out of scope
New retrieval signals joining the candidate union (T34 phase 2). Provider-to-provider
dependencies and ordering (T71). Letting providers *modify* core signals or the
candidate set — `ctx` is read-only evidence.
