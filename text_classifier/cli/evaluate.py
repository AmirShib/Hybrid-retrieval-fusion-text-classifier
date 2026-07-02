#!/usr/bin/env python
r"""Evaluate a trained model against a labeled CSV.

Usage:
    text-classifier-eval \
        --model model_dir/ \
        --input labeled.csv \         # columns: text,label
        [--output report.json]

Reports coverage, accuracy on accepted, calibration (Brier / ECE), a
risk-coverage curve, and a per-class breakdown. Use it to validate a model on a
held-out set, or to monitor a deployed model for drift over time.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from .. import InferencePipeline
from ..application.evaluation import _json_safe, build_manifest, evaluate_decisions
from ._common import add_logging_arg, configure_logging, read_items


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _num(x) -> str:
    return "n/a" if x is None else f"{x:.4f}"


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True, help="labeled CSV with text + label columns")
    p.add_argument("--output", default=None, help="optional path to write the full JSON report")
    p.add_argument("--text-col", default="text")
    p.add_argument("--label-col", default="label")
    add_logging_arg(p)
    args = p.parse_args()
    configure_logging(args.log_level)

    pipeline = InferencePipeline.from_directory(args.model)
    label_space = pipeline.label_space
    keys = label_space.keys
    key_to_idx = {k: i for i, k in enumerate(keys)}

    items = read_items(args.input, args.text_col, args.label_col)
    texts = [it.text for it in items]
    true_keys = [it.label for it in items]

    unknown = sorted({k for k in true_keys if k not in key_to_idx})
    if unknown:
        shown = unknown[:10]
        suffix = " ..." if len(unknown) > 10 else ""
        raise SystemExit(
            f"error: {len(unknown)} label(s) in {args.input!r} are not in the "
            f"model's label space: {shown}{suffix}"
        )

    preds = pipeline.predict(texts)
    confidence = np.array([pr.confidence for pr in preds], dtype=np.float64)
    accepted = np.array([not pr.abstained for pr in preds], dtype=bool)
    correct = np.array([pr.top_key == tk for pr, tk in zip(preds, true_keys)], dtype=bool)
    # top_key is "" when no candidate surfaced; map that to -1 ("no prediction").
    pred_idx = np.array([key_to_idx.get(pr.top_key, -1) for pr in preds], dtype=np.intp)
    true_idx = np.array([key_to_idx[tk] for tk in true_keys], dtype=np.intp)

    evaluation = evaluate_decisions(
        confidence=confidence,
        correct=correct,
        accepted=accepted,
        pred_idx=pred_idx,
        true_idx=true_idx,
        keys=keys,
    )
    manifest = build_manifest(
        n_training_items=len(items),
        n_classes=label_space.size,
        config=pipeline.config,
        n_evaluated=len(items),
    )

    o = evaluation["overall"]
    cal = evaluation["calibration"]
    print("\n=== evaluation ===")
    print(f"items evaluated       : {o['n_items']}")
    print(
        f"coverage              : {_pct(o['coverage'])} "
        f"({o['n_accepted']} accepted, {o['n_abstained']} abstained)"
    )
    print(f"accuracy on accepted  : {_pct(o['accuracy_on_accepted'])}")
    print(f"accuracy if no abstain: {_pct(o['accuracy_if_no_abstain'])}")
    print(f"expected calib. error : {_num(cal['expected_calibration_error'])}")
    print(f"brier score           : {_num(cal['brier_score'])}")

    worst = sorted(
        (r for r in evaluation["per_class"] if r["support"] > 0),
        key=lambda r: r["coverage"] if r["coverage"] is not None else 1.0,
    )[:5]
    if worst:
        print("\nlowest-coverage classes (support>0):")
        for r in worst:
            print(
                f"  {r['key']:>12}  support={r['support']:<5} "
                f"coverage={_pct(r['coverage'])}  precision={_pct(r['precision_on_accepted'])}"
            )

    if args.output:
        with open(args.output, "w") as fh:
            json.dump(_json_safe({"manifest": manifest, **evaluation}), fh, indent=2)
        print(f"\nwrote full report to {args.output}")


if __name__ == "__main__":
    main()
