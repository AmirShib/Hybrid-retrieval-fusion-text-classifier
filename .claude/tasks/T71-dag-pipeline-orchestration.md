# T71 — Design spike: DAG-based pipeline orchestration with declared dependencies

status: todo
tier: 7
depends_on: T70

## Goal
**Decide, don't build (yet).** Produce a short design doc + a thin proof-of-concept that
evaluates modelling the pipeline as an explicit DAG of nodes with declared
`depends_on`, so stage ordering, caching, and parallelism are validated up front instead
of being implicit in the two pipeline classes.

This is a spike, not a rewrite. Its deliverable is a recommendation (adopt / adopt-lite /
defer) backed by a small prototype, not a migrated pipeline.

## Why
Ordering today (encode → retrieve [5 independent signals] → assemble features → fuse →
calibrate → tune) lives implicitly in `TrainingPipeline` / `InferencePipeline` and the
hexagonal layering. An explicit DAG could buy:
- **Validated dependencies**: detect a missing/cyclic dependency before running.
- **Caching / incremental recompute**: re-tune without re-encoding.
- **Parallelism**: the five signals (and independent feature providers from T70) are
  embarrassingly parallel branches.

It costs a meaningful rewrite of two working pipelines and a new failure surface. Doing
it speculatively now risks over-engineering. T70 is the forcing function: custom feature
providers that depend on signals and on each other form a real dependency graph, so this
spike should be done with that graph in front of you.

## Scope of the spike
1. Model the current pipeline as a node graph on paper: nodes, inputs/outputs, which
   edges are real dependencies vs incidental ordering. Mark the parallelizable branches.
2. Fold in T70's feature-provider dependencies — this is the case that actually motivates
   a DAG.
3. Prototype **one** representative subgraph (e.g. the five signals → feature assembly)
   behind a minimal internal node abstraction with topological sort + dependency
   validation. No framework adoption in the prototype.
4. Compare two end states:
   - **(a) lightweight internal**: a topo-sorted list of typed nodes + a tiny runner.
   - **(b) external DAG framework**.
   On: complexity, testability, caching/parallelism payoff, and especially the
   **air-gapped constraint** below.

## Hard constraints (carry into whatever is chosen)
- **No new heavy runtime dependency on the air-gapped path.** A model dir must stay
  portable (stdlib pickle + numpy + json + native formats). An orchestration framework
  that inference *requires* at runtime is disqualifying; train-only tooling is negotiable.
- **Leakage rule survives.** Out-of-fold construction (T06) must be expressible and
  enforceable as graph edges — making leakage *harder*, not a hidden footgun.
- **Domain stays framework-free.** Any DAG machinery lives in `application` /
  `infrastructure`, never `domain`.
- **Determinism.** Same inputs → same outputs regardless of execution order/parallelism.

## Deliverable
- A `docs/` (or ticket-appendix) design note with the graph, the (a)-vs-(b) comparison,
  and a clear recommendation.
- The throwaway prototype of one subgraph (kept out of the shipped package or clearly
  marked experimental).
- If the recommendation is "adopt", a follow-up implementation ticket with a phased,
  behaviour-preserving migration (pipelines keep passing the same tests throughout).

## Acceptance criteria
- [ ] Written recommendation: adopt / adopt-lite / defer, with reasons tied to real
      payoff (caching/parallelism) vs cost.
- [ ] Prototype demonstrates dependency validation + topo order on one real subgraph.
- [ ] Air-gapped portability and the leakage rule are explicitly addressed in the note.

## Out of scope
Migrating the full pipeline (that's the follow-up, only if adopted). Distributed
execution. A scheduler/queue. Anything that changes model outputs.
