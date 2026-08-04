# T86 — Zero-copy feature matrix into the fusion model; drop pandas from the hot path

status: todo
tier: 8
depends_on: T84, T85

## Goal
Close the last device crossing: hand the fusion model a feature matrix that is
already on its device, and stop materializing a pandas DataFrame purely to convert
it straight back to an array.

## Why
Two separate taxes at the same boundary.

**The transfer.** `add_confidence` does `features[cols].to_numpy(np.float32)`
(`scoring.py:28`) and hands a *host* array to a booster configured with
`device="cuda"`. XGBoost then uploads it internally on every call. XGBoost ≥2.0
accepts `__cuda_array_interface__` / DLPack input, which torch tensors implement —
so after T85 the matrix is already on the right device and the copy is pure waste.

**The materialization.** `_assemble_chunk` builds a 36-column DataFrame
(`features.py:325`) that every consumer immediately converts back with
`.to_numpy()`. On a 100k-item corpus at ~30 candidates each that is a 3M × 36
float32 matrix built column-by-column into pandas, then copied out again. And
`add_confidence` does `features.copy()` (`scoring.py:30`) — a **full frame copy on
every scoring call**.

That last one compounds badly in `ablation_report` (`importance.py:126-131`),
which loops over features doing `feats.copy()` and then calls `_score_decisions`,
which copies *again*. For the 36-column core schema that is ~72 full copies of the
entire feature table to produce one report.

## Design

**1. A `FeatureBlock` return type.** `FeatureAssembler.assemble` returns a small
struct rather than a DataFrame:

    @dataclass
    class FeatureBlock:
        X: Any                 # (n_candidates, n_features) backend array
        names: List[str]       # column order — the composed schema
        item_id: np.ndarray    # (n_candidates,) host
        candidate: np.ndarray  # (n_candidates,) host
        is_true: Optional[np.ndarray]
        def to_frame(self) -> pd.DataFrame   # for diagnostics + back-compat

`item_id`/`candidate` stay on the host: they drive pandas groupbys in the decision
layer, which T83's policy keeps host-side. `X` lives wherever the backend put it.

**2. `FusionModel` gains a device-array path.** `fit`/`predict_proba`/
`predict_contribs` already take "array-like"; the XGBoost adapter routes a device
array into `QuantileDMatrix` / `inplace_predict` unchanged, and calls `to_host`
first for any backend that cannot consume it (LightGBM's CPU-only wheel). No port
signature change — the adapters absorb it, which is the point of the port.

**3. `add_confidence` stops copying.** It returns `(conf_array, frame_or_block)`
or mutates a caller-owned block; either way one full-frame copy per scoring call
disappears. `ablation_report` masks a *column of `X`* to NaN in place and restores
it, instead of copying the frame per feature — 72 full copies become zero.

**4. `to_frame()` keeps the diagnostic surface intact.** `explain`,
`explain_records`, `signal_report` and the CLI paths keep receiving a DataFrame;
they are not hot paths and their ergonomics are worth the conversion. Only
`predict` / `predict_topk` / training's fit+score path take the array route.

This is the concrete first slice of **T76** (numpy-only inference), which the
roadmap gates on T34 — worth noting there when this lands.

## Files to change
`application/features.py`, `application/scoring.py`, `application/importance.py`,
`application/inference.py`, `application/training.py`, `infrastructure/fusion.py`,
`domain/ports.py` (docstrings), `tests/unit/test_features.py`,
`tests/unit/test_scoring.py`, `tests/unit/test_importance.py`,
`tests/integration/test_e2e.py`.

## Tests
- [ ] `FeatureBlock.to_frame()` equals today's assembled frame byte-for-byte
      (columns, order, dtypes, values).
- [ ] Predictions identical (numpy backend) before/after, end to end.
- [ ] `ablation_report` output identical before/after, with the in-place mask
      provably restoring `X` (assert the matrix is unchanged after the report).
- [ ] Device array reaches XGBoost without a host round-trip — asserted via the
      `to_host` counter from T85, not by reading logs.
- [ ] LightGBM backend (no device-array support) still works via the host fallback.
- [ ] Peak memory during OOF assembly measurably lower on a fixed corpus.

## Acceptance criteria
- [ ] No DataFrame constructed on the `predict` / `predict_topk` path.
- [ ] Zero host copies of the feature matrix when backend and fusion device agree.
- [ ] `features.copy()` gone from `add_confidence`; per-feature frame copies gone
      from `ablation_report`.
- [ ] Diagnostic surface (`explain`, `signal_report`, `explain_records`) unchanged
      in output.
- [ ] T52 benchmark floors green; numbers recorded against T83's baseline.

## Out of scope
Streaming/chunked inference (T75). Dropping pandas from the *decision* layer —
`top_per_item`'s groupby/merge stays until T76 proper.
