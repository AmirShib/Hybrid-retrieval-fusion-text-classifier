#!/usr/bin/env python
r"""Re-tune calibration + abstention thresholds on a trained model.

Usage:
    text-classifier-tune \
        --model model_dir/ \
        --input labeled.csv \          # fresh labeled CSV: columns text,label
        --target-precision 0.97 \
        [--per-class-min-support 100] [--dry-run] [--text-col text] [--label-col label]

Refits the calibrator and re-tunes the global + per-class abstention thresholds
for ``--target-precision``, reusing the model's existing encoder, retrieval
indices, and fusion model unchanged — no retraining. Updates the calibrator file
and ``meta.json``'s abstention block in place, and writes a fresh
``evaluation.json``/``model_card.md`` reflecting the new operating point.
``--dry-run`` prints the would-be coverage / accuracy / thresholds and writes
nothing.

**The labeled set must be fresh.** An item that was in the model's original
training set sits inside the deployed retrieval indices and retrieves itself as
a perfect match, so its confidence is optimistically inflated — the retuned
threshold would then under-abstain in production. Never point ``--input`` at
the file used for ``text-classifier-train --items``. This tool warns (but
cannot, without the training corpus itself, definitively detect) when tune-set
items look like near-duplicates of an indexed training example.
"""

from __future__ import annotations

import argparse

from .. import InferencePipeline
from ..application.evaluation import build_manifest, write_evaluation_artifacts
from ..application.tuning import retune
from ..infrastructure import ArtifactRepository
from ._common import add_logging_arg, configure_logging, pct, read_items, write_json_report


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True, help="trained model directory")
    p.add_argument(
        "--input",
        required=True,
        help="fresh labeled CSV (text, label columns); must not overlap the training set",
    )
    p.add_argument("--target-precision", type=float, required=True)
    p.add_argument(
        "--per-class-min-support",
        type=int,
        default=None,
        help="minimum calibration rows for a class to get its own threshold "
        "(default: the model's own training.per_class_min_support)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print the would-be operating point and write nothing",
    )
    p.add_argument("--output", default=None, help="optional path to also write the JSON report")
    p.add_argument("--text-col", default="text")
    p.add_argument("--label-col", default="label")
    add_logging_arg(p)
    args = p.parse_args()
    configure_logging(args.log_level)

    pipeline = InferencePipeline.from_directory(args.model)
    artifacts = pipeline.artifacts
    items = read_items(args.input, args.text_col, args.label_col)

    per_class_min_support = (
        args.per_class_min_support
        if args.per_class_min_support is not None
        else artifacts.config.training.per_class_min_support
    )

    try:
        abstention, calibrator, evaluation = retune(
            artifacts,
            items,
            artifacts.label_space,
            args.target_precision,
            per_class_min_support,
        )
    except ValueError as exc:
        raise SystemExit(f"error: {exc}")

    o = evaluation["overall"]
    heading = "retuned operating point (dry run)" if args.dry_run else "retuned operating point"
    print(f"\n=== {heading} ===")
    print(f"items evaluated       : {o['n_items']}")
    print(
        f"coverage              : {pct(o['coverage'])} "
        f"({o['n_accepted']} accepted, {o['n_abstained']} abstained)"
    )
    print(f"accuracy on accepted  : {pct(o['accuracy_on_accepted'])}")
    print(f"global threshold      : {abstention.global_threshold:.4f}")
    print(f"per-class thresholds  : {len(abstention.per_class)}")

    # Reflect the new target_precision in the in-memory config so every manifest
    # built below (report and/or model-dir evaluation artifacts) is consistent
    # with what --dry-run would/does persist.
    artifacts.config.training.target_precision = args.target_precision
    manifest = build_manifest(
        n_training_items=len(items),
        n_classes=artifacts.label_space.size,
        config=artifacts.config,
        n_evaluated=len(items),
    )

    write_json_report(args.output, {"manifest": manifest, **evaluation}, label="report")

    if args.dry_run:
        print("\n(dry run: model directory left unchanged)")
        return

    ArtifactRepository().update_decision_layer(
        args.model, calibrator, abstention, args.target_precision, len(items)
    )
    manifest = {**manifest, "retuned_at": manifest["generated_at"]}
    write_evaluation_artifacts(args.model, evaluation, manifest)
    print(f"\nupdated calibrator + abstention thresholds in {args.model}")
    print("wrote fresh evaluation.json + model_card.md reflecting the new operating point")


if __name__ == "__main__":
    main()
