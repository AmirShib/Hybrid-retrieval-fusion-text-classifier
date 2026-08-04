# T84 — Array-backend seam: an `ArrayOps` port behind feature assembly + retrieval

status: todo
tier: 8
depends_on: —

## Goal
Make every numeric kernel in feature assembly and dense retrieval run against a
narrow **array port** instead of calling numpy directly, with a numpy backend as
the default. Zero behaviour change: the CPU path must be byte-identical. This is
the enabling refactor for T85/T86 — nothing device-specific lands here.

## Why
`application/features.py` and `infrastructure/retrieval.py` call `np.*` directly
in ~20 places. To run those kernels on a GPU you either (a) write a second
implementation, which drifts, or (b) write them once against an abstraction.
**Decision taken: (b), a single codepath.** Two implementations of the same 36
features would eventually disagree, and the disagreement would be silent — the
worst failure mode this package has (see T82's note on `explain` silently
inheriting the wrong column list).

The dependency rule constrains where the abstraction can live: `domain` imports
no ML framework, and `application` orchestrates through ports. So the port is
declared in `domain/ports.py`, the numpy backend and the (later) torch backend
are `infrastructure/` adapters, and the assembler takes one by injection.

## Design

**The port** (`domain/ports.py`) — deliberately narrow. Only the primitives the
existing kernels actually use, not an array-API reimplementation:

    class ArrayOps(ABC):
        name: str                     # "numpy" | "torch"
        # construction / movement
        def asarray(x, dtype=None); def to_host(x) -> np.ndarray
        def zeros(shape, dtype); def full(shape, value, dtype)
        # elementwise
        def where(cond, a, b); def isnan(x); def isfinite(x); def maximum(a, b)
        def log1p(x)
        # reduction / ordering
        def matmul(a, b); def topk(x, k, axis); def argsort(x, axis)
        def argpartition(x, k, axis); def nanmin(x, axis); def nanmax(x, axis)
        # scatter / gather
        def scatter_add(target, rows, cols, values)
        def scatter_max(target, rows, cols, values)
        def gather(M, rows, cols)

`to_host` is the *only* sanctioned exit to numpy, so every transfer is one
greppable call — which is what makes T83's transfer count enforceable later.

**The numpy backend** (`infrastructure/array_ops.py::NumpyArrayOps`) is a thin
pass-through, except for two kernels that get fixed on the way past because the
port forces them to be expressed properly:

- `scatter_add` / `scatter_max` replace `np.add.at` / `np.maximum.at`
  (`features.py:49-51`) with `np.bincount`-based equivalents. `ufunc.at` is
  unbuffered and is the slowest scatter numpy offers; the bincount form is
  numerically identical for these shapes (integer-indexed sums over float64).
- `_prototypes_and_freq` (`retrieval.py:290`) loses its `for c in range(n_classes)`
  Python loop in favour of a single `scatter_add` + norm. On a 5000-class taxonomy
  that loop is 5000 masked means over the full embedding matrix, and it violates
  this repo's own "no per-row Python loops on the hot path" convention.

Both are CPU-side wins in their own right and are independently verifiable —
which is the point of doing them here, under a byte-identity test, rather than
tangled into the device work.

**Selection — adaptive, because scale varies by deployment.**
`PipelineConfig.array_backend: str = "auto"` (registry key, same shape as
`encoder.kind` / `fusion.kind`), with `"numpy"` / `"torch"` as explicit overrides
that always win — the same explicit-beats-detected rule `resolve_device` already
follows (`infrastructure/device.py:30`).

Decided 2026-08-04: deployments range from a few thousand items to 100k+, so a
fixed default is wrong for someone either way. `"auto"` picks the backend from
T83's measured crossover thresholds using quantities known *before* assembly runs
(`n_items`, `n_classes`, `k_neighbors`, device visibility), logs the choice and
the rule that produced it, and falls back to numpy whenever torch is absent, the
device is unavailable, or the corpus is below the crossover. The rule must be a
single named function with its thresholds as module constants traceable to T83's
table — not a heuristic sprinkled through the pipeline.

Recorded in `meta.json`'s config block for provenance, but **not** load-bearing at
load time: a model trained with the torch backend must load and score on a
numpy-only air-gapped host. The backend is an execution choice, never a property
of the artifact.

**The port is also the torch/no-torch boundary.** T63 (torch-optional core) lands
before T85 and shares this seam, so the numpy backend must be reachable with torch
uninstalled — enforced by test, not by convention (see below).

**Not in scope for the port:** BM25 (`LexicalRetrieverAdapter`) stays on scipy
sparse and plain numpy. Per T83's policy it is permanently host-side; routing it
through the port would buy nothing and add a layer.

## Files to change
`domain/ports.py`, `infrastructure/array_ops.py` (new), `infrastructure/registry.py`,
`infrastructure/__init__.py`, `application/features.py`, `infrastructure/retrieval.py`
(dense adapter only), `config.py`, `application/training.py` +
`application/inference.py` (inject the backend), `tests/unit/test_array_ops.py` (new),
`tests/unit/test_features.py`, `tests/unit/test_retrieval.py`.

## Tests
- [ ] **Golden-frame byte-identity**: on a fixed corpus, the assembled feature
      frame after this refactor is bit-for-bit equal to the frame before it
      (store expected values, not just shape). This is the whole safety net.
- [ ] `scatter_add`/`scatter_max` match `np.add.at`/`np.maximum.at` exactly on
      randomized inputs including duplicate indices, empty input, and all-NaN rows.
- [ ] `_prototypes_and_freq` loop-free version matches the loop version exactly,
      including the all-NaN row for a class with no examples.
- [ ] Existing leakage regression test (T06) and T52 benchmark floors stay green.
- [ ] A model dir trained under this change loads on a host with the torch backend
      unregistered.
- [ ] **No torch import is reachable from a numpy-backend run** — asserted by
      blocking `torch` in `sys.modules` and running the full offline suite. This
      is the T63 boundary and the one guard that keeps the port from quietly
      re-entrenching torch in the core.
- [ ] `array_backend="auto"` selects numpy on a CPU-only host, on a torch-free
      install, and below the crossover; the choice and its reason are logged.
- [ ] Explicit `array_backend` always overrides `"auto"`.

## Acceptance criteria
- [ ] No direct `np.*` call left in `features.py`'s kernels or the dense adapter's
      scoring paths — all go through the port.
- [ ] Default config produces byte-identical features and an unchanged `meta.json`
      schema; older model dirs load unchanged.
- [ ] `to_host` is the only numpy exit, and is called in a countable number of
      places.
- [ ] ruff + mypy clean on touched files.

## Out of scope
Any torch backend (T85). Any change to *which* features are computed (T87).
Removing pandas from the assembly path (T86).
