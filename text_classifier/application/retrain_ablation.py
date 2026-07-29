"""Retrain-based feature ablation (application service).

The complement to ``application/importance.py``. That module answers a question
about a *fixed* model — mask a column to ``NaN`` and re-score, i.e. "what happens
if this signal fails at inference time?". This module answers the schema
question: **train without the column and compare.** A model fitted with a column
has splits on it; masking it sends rows down default branches, which is a
faithful simulation of a broken signal but a poor proxy for "would the model be
better if this column had never existed". Only a retrain answers that.

Two things make the answer trustworthy rather than anecdotal:

- **Seeds.** One run cannot separate a 1pp effect from noise on a few hundred
  items. Every arm is trained ``len(seeds)`` times over different fold splits
  (``TrainingConfig.random_state``), and results are reported as mean +/- std
  with the per-seed values kept, so a reader can see the spread rather than a
  point estimate that happens to flatter the change.
- **A paired baseline.** The no-drop arm is trained under the *same* seeds, so
  arms are compared seed-for-seed rather than against a single reference run.

Cost is the honest tradeoff: this trains ``(len(arms) + 1) * len(seeds)`` models.
It is a measurement tool for deciding what belongs in the schema, not something
on any hot path.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from ..config import PipelineConfig
from ..domain import LabeledItem, LabelSpace

# The headline metrics each arm is scored on. All are "higher is better" except
# that coverage and accuracy trade against each other along the threshold curve —
# which is why ``accepted_correct`` (their product, the rate of items both
# accepted *and* right) is reported alongside: it is the one number that does not
# move when the operating point slides without the model actually changing.
ABLATION_METRICS = ("candidate_recall", "coverage", "accuracy_on_accepted", "accepted_correct")


@dataclass(frozen=True)
class AblationArm:
    """One experimental condition: a named set of columns to withhold.

    ``name`` is what appears in the report (e.g. ``"t81_margins"``); ``drop`` is
    the columns that arm withholds. An arm with an empty ``drop`` is the
    baseline, which ``retrain_ablation`` adds on its own — callers pass only the
    arms they want compared against it.
    """

    name: str
    drop: Sequence[str] = field(default_factory=tuple)


def _summarize(values: Sequence[float]) -> Dict[str, Any]:
    """mean/std/min/max over an arm's per-seed values, NaNs excluded.

    A run that produced no usable value (all-NaN) yields NaN summaries rather
    than raising: one unscoreable arm should not lose the rest of the report.
    """
    arr = np.asarray([v for v in values], dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }


def _run_one(
    items: Sequence[LabeledItem],
    label_space: LabelSpace,
    config: PipelineConfig,
    drop: Sequence[str],
    seed: int,
) -> Dict[str, float]:
    """Train one model with ``drop`` withheld at fold-seed ``seed`` and return its
    headline metrics. Imported lazily so this module stays importable without
    paying the training pipeline's import cost."""
    from .training import TrainingPipeline

    cfg = copy.deepcopy(config)
    cfg.fusion.drop_features = list(drop)
    cfg.training.random_state = seed
    _, report = TrainingPipeline(cfg).run(list(items), label_space)
    coverage = float(report.coverage)
    accuracy = float(report.accuracy_on_accepted)
    return {
        "candidate_recall": float(report.candidate_recall),
        "coverage": coverage,
        "accuracy_on_accepted": accuracy,
        # Accepted *and* correct. Coverage and accuracy slide against each other
        # as the tuned threshold moves, so either alone can shift by a point
        # without the model having changed; their product does not.
        "accepted_correct": coverage * accuracy,
    }


def retrain_ablation(
    items: Sequence[LabeledItem],
    label_space: LabelSpace,
    config: PipelineConfig,
    arms: Sequence[AblationArm],
    seeds: Sequence[int] = (0, 1, 2, 3),
    progress: Optional[Any] = None,
) -> Dict[str, Any]:
    """Train each arm (plus a no-drop baseline) once per seed and compare.

    Returns a JSON-clean dict:

    - ``seeds`` / ``n_runs`` — the sample the report rests on.
    - ``baseline`` — ``{metric: {mean, std, min, max}}`` plus ``per_seed``.
    - ``arms`` — one entry per arm with the same summaries, plus ``delta``:
      the arm's mean minus the baseline's, and ``paired_delta`` — the mean of
      the *per-seed* differences, which is the number to read. Comparing means
      across independently-noisy runs understates how much of a difference is
      shared noise; pairing by seed removes the fold-split variance both arms
      saw. ``paired_delta_std`` is the spread of those per-seed differences: a
      ``paired_delta`` smaller than its own std is not evidence of anything.
    - ``verdict`` — per arm, ``"earns_place"`` when dropping the columns makes
      ``accepted_correct`` reliably *worse* (paired delta negative and larger in
      magnitude than its std), ``"redundant"`` when dropping them reliably does
      not hurt, and ``"inconclusive"`` otherwise — which, at these sample sizes,
      is the honest answer more often than not.

    ``progress`` is an optional callable ``(arm_name, seed, run_index, total)``
    invoked before each run, so a CLI can report progress on a long sweep.
    """
    arms = list(arms)
    seeds = list(seeds)
    if not seeds:
        raise ValueError("retrain_ablation needs at least one seed")
    names = [a.name for a in arms]
    if len(set(names)) != len(names):
        raise ValueError(f"arm names must be unique; got {names}")
    if "baseline" in names:
        raise ValueError("'baseline' is reserved: retrain_ablation adds the no-drop arm itself")

    total = (len(arms) + 1) * len(seeds)
    run_index = 0

    baseline_runs: List[Dict[str, float]] = []
    for seed in seeds:
        run_index += 1
        if progress is not None:
            progress("baseline", seed, run_index, total)
        baseline_runs.append(_run_one(items, label_space, config, (), seed))

    baseline: Dict[str, Any] = {
        "drop": [],
        "per_seed": baseline_runs,
        **{m: _summarize([r[m] for r in baseline_runs]) for m in ABLATION_METRICS},
    }

    arm_reports: List[Dict[str, Any]] = []
    for arm in arms:
        runs: List[Dict[str, float]] = []
        for seed in seeds:
            run_index += 1
            if progress is not None:
                progress(arm.name, seed, run_index, total)
            runs.append(_run_one(items, label_space, config, arm.drop, seed))

        entry: Dict[str, Any] = {
            "name": arm.name,
            "drop": list(arm.drop),
            "per_seed": runs,
            **{m: _summarize([r[m] for r in runs]) for m in ABLATION_METRICS},
        }
        for metric in ABLATION_METRICS:
            paired = [runs[i][metric] - baseline_runs[i][metric] for i in range(len(seeds))]
            arr = np.asarray(paired, dtype=np.float64)
            entry[f"delta_{metric}"] = float(entry[metric]["mean"] - baseline[metric]["mean"])
            entry[f"paired_delta_{metric}"] = float(np.nanmean(arr))
            entry[f"paired_delta_{metric}_std"] = float(np.nanstd(arr))
        entry["verdict"] = _verdict(
            entry["paired_delta_accepted_correct"], entry["paired_delta_accepted_correct_std"]
        )
        arm_reports.append(entry)

    return {
        "seeds": seeds,
        "n_runs": total,
        "metrics": list(ABLATION_METRICS),
        "baseline": baseline,
        "arms": arm_reports,
    }


def _verdict(paired_delta: float, paired_std: float) -> str:
    """Turn a paired delta into a word, conservatively.

    ``paired_delta`` is the change in ``accepted_correct`` from *dropping* the
    arm's columns, so negative means dropping hurt — the columns were doing
    work. The magnitude must exceed the spread of the per-seed differences
    before we call it anything: an effect smaller than its own run-to-run
    variation is not an effect, and saying so plainly is the whole point of the
    harness. A zero-variance sweep (identical differences on every seed) is
    decided by sign alone, since there is no spread to clear.
    """
    if not np.isfinite(paired_delta) or not np.isfinite(paired_std):
        return "inconclusive"
    # Zero spread means every seed agreed exactly, so there is nothing to clear
    # and sign alone decides. This covers the important degenerate case of a
    # provably inert column: delta and spread both exactly 0.0 is not "we cannot
    # tell", it is the strongest possible evidence that dropping costs nothing.
    if paired_std == 0.0:
        return "earns_place" if paired_delta < 0 else "redundant"
    if abs(paired_delta) <= paired_std:
        return "inconclusive"
    return "earns_place" if paired_delta < 0 else "redundant"


def arms_from_groups(groups: Mapping[str, Sequence[str]]) -> List[AblationArm]:
    """Build arms from a ``{name: [columns]}`` mapping, in iteration order."""
    return [AblationArm(name=name, drop=tuple(cols)) for name, cols in groups.items()]
