#!/usr/bin/env python
r"""Feature importance + per-feature ablation report for a trained model.

Usage:
    text-classifier-importance \
        --model model_dir/ \
        --input labeled.csv \         # columns: text,label
        [--output report.json]

Two reports against the labeled set, computed with the model as-is (no
retraining):

- **importance**: mean additive contribution of each feature column toward the
  raw fusion score (from the fusion backend's ``predict_contribs``, when the
  backend supports it — XGBoost/LightGBM-style backends do).
- **ablation**: for each feature, mask it to ``NaN`` (the domain's "signal did
  not retrieve this" encoding) and re-score with the unchanged model, reporting
  the resulting accuracy/coverage change. This is the actual cost of losing a
  signal, not just how much it moves scores when present — the two can rank
  features differently.

Use this to decide which of the ~28+ feature columns are pulling their weight
before adding more (see the T81 follow-ups) or to sanity-check a new signal
after it lands.
"""

from __future__ import annotations

import argparse

from .. import InferencePipeline
from ._common import (
    add_logging_arg,
    add_placement_args,
    configure_logging,
    pct,
    read_items,
    signed_pct,
    write_json_report,
)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True, help="labeled CSV with text + label columns")
    p.add_argument("--output", default=None, help="optional path to write the full JSON report")
    p.add_argument("--text-col", default="text")
    p.add_argument("--label-col", default="label")
    p.add_argument("--top", type=int, default=10, help="rows to print per table (default: 10)")
    add_placement_args(p)
    add_logging_arg(p)
    args = p.parse_args()
    configure_logging(args.log_level)

    pipeline = InferencePipeline.from_directory(
        args.model, device=args.device, array_backend=args.array_backend
    )
    items = read_items(args.input, args.text_col, args.label_col)
    texts = [it.text for it in items]
    true_keys = [it.label for it in items]

    try:
        report = pipeline.importance_report(texts, true_keys)
    except KeyError as exc:
        raise SystemExit(f"error: {exc}")

    baseline = report["ablation"]["baseline"]
    print("\n=== baseline (unablated) ===")
    print(f"items evaluated       : {baseline['n_items']}")
    print(f"coverage              : {pct(baseline['coverage'])}")
    print(f"accuracy on accepted  : {pct(baseline['accuracy_on_accepted'])}")
    print(f"accuracy if no abstain: {pct(baseline['accuracy_if_no_abstain'])}")

    importance = report["importance"]
    if importance is None:
        print("\nimportance: unavailable (fusion backend does not support predict_contribs)")
    else:
        print(f"\n=== feature importance (top {args.top} by mean |contribution|) ===")
        for row in importance[: args.top]:
            print(f"  {row['feature']:<24} share={pct(row['share'])}")

    ablations = report["ablation"]["ablations"]
    print(f"\n=== ablation (top {args.top} most damaging removals) ===")
    for row in ablations[: args.top]:
        print(
            f"  {row['feature']:<24} "
            f"Δaccuracy_if_no_abstain={signed_pct(row['delta_accuracy_if_no_abstain'])}  "
            f"Δcoverage={signed_pct(row['delta_coverage'])}  "
            f"(n={row['n_rows_masked']})"
        )

    write_json_report(args.output, report)


if __name__ == "__main__":
    main()
