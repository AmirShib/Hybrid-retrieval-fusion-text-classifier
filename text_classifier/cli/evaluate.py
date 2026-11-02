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

With ``--classes classes.csv`` (columns: key, description) the model's label
space is first widened with any class in the file it was not trained on, so a
test set that references *new* classes can be scored end-to-end. Added
classes are description-only — retrievable from their description but, lacking
example support, low-confidence and prone to abstain — so this measures the
floor a not-yet-retrained class reaches, not trained-class performance.
"""

from __future__ import annotations

import argparse

import numpy as np

from .. import InferencePipeline
from .._messages import format_preview
from ..application.evaluation import build_manifest, evaluate_decisions
from ..application.signal_report import signal_report
from ._common import (
    add_logging_arg,
    configure_logging,
    num,
    pct,
    read_items,
    read_label_space,
    write_json_report,
)


def _extend_with_new_classes(pipeline: InferencePipeline, classes_path: str) -> InferencePipeline:
    """Widen ``pipeline``'s label space with any class in ``classes_path`` that the
    model was not trained on (description-only, no retrain). Existing classes in
    the file are ignored — their description in the model wins; changing it needs
    a retrain. Returns the original pipeline unchanged when there is nothing new."""
    file_space = read_label_space(classes_path)
    known = set(pipeline.label_space.keys)
    new = [(k, d) for k, d in zip(file_space.keys, file_space.descriptions) if k not in known]
    if not new:
        print(f"no new classes in {classes_path!r}; evaluating against the model's own label space")
        return pipeline
    print(
        f"added {len(new)} class(es) from {classes_path!r} not seen at training "
        f"(description-only, no retrain): {[k for k, _ in new][:10]}"
        f"{' ...' if len(new) > 10 else ''}"
    )
    return pipeline.with_added_classes(new)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True, help="labeled CSV with text + label columns")
    p.add_argument("--output", default=None, help="optional path to write the full JSON report")
    p.add_argument("--text-col", default="text")
    p.add_argument("--label-col", default="label")
    p.add_argument(
        "--classes",
        default=None,
        help="optional classes CSV (key, description); classes not already in the "
        "model are added description-only (no retrain) so a test set with new "
        "labels can be scored. Such classes are low-confidence and prone to abstain.",
    )
    add_logging_arg(p)
    args = p.parse_args()
    configure_logging(args.log_level)

    pipeline = InferencePipeline.from_directory(args.model)

    if args.classes:
        pipeline = _extend_with_new_classes(pipeline, args.classes)

    label_space = pipeline.label_space
    keys = label_space.keys
    key_to_idx = {k: i for i, k in enumerate(keys)}

    items = read_items(args.input, args.text_col, args.label_col)
    texts = [it.text for it in items]
    true_keys = [it.label for it in items]

    unknown = label_space.unknown_keys(true_keys)
    if unknown:
        raise SystemExit(
            f"error: {len(unknown)} label(s) in {args.input!r} are not in the "
            f"model's label space: {format_preview(unknown)}"
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
    # Per-signal diagnostics on this labeled set: reuse the same encode → assemble →
    # calibrate pass `explain` exposes, tag each candidate row with ground truth, and
    # report how each retrieval signal does alone. Cheap, and it makes the eval report
    # a drift-monitoring surface for the signals, not just the fused decision.
    detail = pipeline.explain(texts)
    if len(detail):
        item_true = np.asarray(true_keys, dtype=object)[detail["item_id"].to_numpy(dtype=np.intp)]
        detail = detail.assign(
            is_true=(detail["candidate_key"].to_numpy() == item_true).astype(int)
        )
        evaluation["signal_report"] = signal_report(detail)

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
        f"coverage              : {pct(o['coverage'])} "
        f"({o['n_accepted']} accepted, {o['n_abstained']} abstained)"
    )
    print(f"accuracy on accepted  : {pct(o['accuracy_on_accepted'])}")
    print(f"accuracy if no abstain: {pct(o['accuracy_if_no_abstain'])}")
    print(f"expected calib. error : {num(cal['expected_calibration_error'])}")
    print(f"brier score           : {num(cal['brier_score'])}")

    worst = sorted(
        (r for r in evaluation["per_class"] if r["support"] > 0),
        key=lambda r: r["coverage"] if r["coverage"] is not None else 1.0,
    )[:5]
    if worst:
        print("\nlowest-coverage classes (support>0):")
        for r in worst:
            print(
                f"  {r['key']:>12}  support={r['support']:<5} "
                f"coverage={pct(r['coverage'])}  precision={pct(r['precision_on_accepted'])}"
            )

    sig = evaluation.get("signal_report") or {}
    per_signal = sig.get("per_signal") or []
    if per_signal:
        print("\nper-signal top-1 accuracy (each retrieval signal alone):")
        for e in per_signal:
            print(
                f"  {e['signal']:>17}  top1={pct(e.get('top1_accuracy'))}  "
                f"fires={pct(e.get('fired_rate'))}  "
                f"prec_when_fired={pct(e.get('top1_precision_when_fired'))}"
            )

    write_json_report(args.output, {"manifest": manifest, **evaluation})


if __name__ == "__main__":
    main()
