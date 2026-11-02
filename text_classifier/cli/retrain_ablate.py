#!/usr/bin/env python
r"""Retrain-based feature ablation: does a column belong in the schema?

Usage:
    text-classifier-retrain-ablate \
        --items items.csv \           # columns: text,label
        --classes classes.csv \       # columns: key,description
        --group t81_margins=margin_d_desc,margin_d_proto,margin_d_knn \
        --group t81_gaps=q_gap_d_desc,q_gap_d_knn,q_gap_b_desc \
        --seeds 0,1,2,3 \
        [--output report.json]

Each ``--group NAME=col,col,...`` is one arm: a model trained with those columns
withheld from the fusion model. A no-drop baseline arm is added automatically,
and every arm is trained once per seed so results come with a spread rather than
a single flattering number.

This is the counterpart to ``text-classifier-importance``. That tool masks a
column to ``NaN`` on an already-trained model, answering "what if this signal
fails at inference?". This one retrains without the column, answering "should
this column be in the schema at all?" — a model fitted *with* a column has splits
on it, so masking cannot tell you whether the schema is better without it.

Cost: ``(groups + 1) x seeds`` full training runs. Budget accordingly.
"""

from __future__ import annotations

import argparse
import json
import sys

from ..application.retrain_ablation import AblationArm, retrain_ablation
from ._common import (
    add_config_args,
    add_logging_arg,
    configure_logging,
    load_pipeline_config,
    read_items,
    read_label_space,
    signed_num,
    write_json_report,
)


def _parse_group(spec: str) -> AblationArm:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"--group must be NAME=col1,col2 (got {spec!r}); the name labels the arm "
            "in the report and the columns are what that arm withholds"
        )
    name, _, cols = spec.partition("=")
    columns = tuple(c.strip() for c in cols.split(",") if c.strip())
    if not name.strip():
        raise argparse.ArgumentTypeError(f"--group needs a non-empty name (got {spec!r})")
    if not columns:
        raise argparse.ArgumentTypeError(f"--group {name!r} lists no columns (got {spec!r})")
    return AblationArm(name=name.strip(), drop=columns)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Retrain-based feature ablation across seeds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--items", required=True, help="labeled CSV with text + label columns")
    p.add_argument("--classes", required=True, help="CSV with key + description columns")
    p.add_argument(
        "--group",
        action="append",
        required=True,
        type=_parse_group,
        metavar="NAME=col1,col2",
        help="one ablation arm; repeatable",
    )
    p.add_argument(
        "--seeds",
        default="0,1,2,3",
        help="comma-separated fold-split seeds (default: 0,1,2,3). More seeds = "
        "tighter error bars and proportionally more training runs.",
    )
    p.add_argument("--output", default=None, help="optional path to write the full JSON report")
    p.add_argument("--text-col", default="text")
    p.add_argument("--label-col", default="label")
    p.add_argument("--key-col", default="key")
    p.add_argument("--desc-col", default="description")
    add_config_args(p)
    add_logging_arg(p)
    args = p.parse_args()
    configure_logging(args.log_level)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    cfg = load_pipeline_config(args.config)
    # No config-overriding flags on this CLI (each arm's only deviation is its
    # own `drop`), so the loaded config is already the effective one.
    if args.dump_config:
        print(json.dumps(cfg.to_dict(), indent=2))
        return
    label_space = read_label_space(args.classes, args.key_col, args.desc_col)
    items = read_items(args.items, text_col=args.text_col, label_col=args.label_col)

    def progress(arm: str, seed: int, index: int, total: int) -> None:
        print(f"[{index}/{total}] training arm={arm} seed={seed} ...", file=sys.stderr)

    report = retrain_ablation(items, label_space, cfg, args.group, seeds=seeds, progress=progress)

    base = report["baseline"]
    print(f"\nbaseline over {len(seeds)} seed(s) {seeds}:")
    for metric in report["metrics"]:
        s = base[metric]
        print(
            f"  {metric:22s} {s['mean']:.4f} +/- {s['std']:.4f}  [{s['min']:.4f}, {s['max']:.4f}]"
        )

    print("\nper-arm effect of DROPPING the listed columns (paired by seed):")
    print(f"  {'arm':22s} {'d_accepted_correct':>19s} {'+/-':>9s}  verdict")
    for arm in report["arms"]:
        delta = arm["paired_delta_accepted_correct"]
        std = arm["paired_delta_accepted_correct_std"]
        print(f"  {arm['name']:22s} {signed_num(delta):>19s} {std:9.4f}  {arm['verdict']}")
    print(
        "\n  negative delta = dropping those columns HURT, i.e. they earn their place.\n"
        "  'inconclusive' means the effect is smaller than its own seed-to-seed spread."
    )

    write_json_report(args.output, report)


if __name__ == "__main__":
    main()
