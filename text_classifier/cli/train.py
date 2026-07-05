#!/usr/bin/env python
r"""Train the classifier.

Usage:
    text-classifier-train \
        --items items.csv \           # columns: text,label
        --classes classes.csv \       # columns: key,description
        --out model_dir/ \
        [--encoder-kind tfidf] [--per-fold-encoder] [--target-precision 0.95] [--folds 5]

`label` in items.csv must match a `key` in classes.csv. Writes a portable model
directory plus `evaluation.json` and `model_card.md` summarizing held-out
performance.

Every field of `PipelineConfig` (fusion kind + xgb_params, calibration kind,
bm25_token_kwargs, encoder params, ...) is reachable via `--config config.json`
without writing Python; a trained model dir's `meta.json` `config` block is
directly usable as one. Precedence is defaults < --config < explicit flags.
Use `--dump-config` to print the effective config and exit.
"""

from __future__ import annotations

import argparse
import json
import logging

from .. import TrainingPipeline
from ..infrastructure.registry import encoder_spec
from ._common import (
    add_config_args,
    add_logging_arg,
    configure_logging,
    load_pipeline_config,
    read_items,
    read_label_space,
)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--items", default=None, help="required unless --dump-config")
    p.add_argument("--classes", default=None, help="required unless --dump-config")
    p.add_argument("--out", default=None, help="required unless --dump-config")
    p.add_argument(
        "--encoder-kind",
        default=None,
        help="encoder backend (registry key): 'sentence-transformers' "
        "(default), 'tfidf' (torch-free, air-gapped), or 'hashing' "
        "(dependency-free baseline / smoke test)",
    )
    p.add_argument(
        "--encoder",
        default=None,
        help="model name/path for the sentence-transformers encoder; "
        "ignored by corpus-fitted encoders such as tfidf "
        "(default: sentence-transformers/all-MiniLM-L6-v2)",
    )
    p.add_argument("--folds", type=int, default=None, help="default: 5")
    p.add_argument(
        "--target-precision",
        type=float,
        default=None,
        help="target accuracy on accepted items; the threshold is tuned "
        "for max coverage subject to this (default: 0.95)",
    )
    p.add_argument("--candidate-top-n", type=int, default=None, help="default: 10")
    p.add_argument("--k-neighbors", type=int, default=None, help="default: 20")
    p.add_argument(
        "--per-fold-encoder",
        action="store_true",
        default=None,
        help="rigorous (expensive): fine-tune a fresh encoder per fold",
    )
    p.add_argument("--text-col", default="text", help="items.csv text column")
    p.add_argument("--label-col", default="label", help="items.csv label column")
    p.add_argument("--key-col", default="key", help="classes.csv key column")
    p.add_argument("--desc-col", default="description", help="classes.csv description column")
    add_config_args(p)
    add_logging_arg(p)
    args = p.parse_args()
    configure_logging(args.log_level)

    cfg = load_pipeline_config(args.config)

    # Explicit flags win over the config file; a flag left at its argparse
    # default of `None` means "not set", so the file (or the built-in
    # PipelineConfig default) is left untouched.
    if args.encoder_kind is not None:
        cfg.encoder.kind = args.encoder_kind
    if args.encoder is not None:
        cfg.encoder.model_name_or_path = args.encoder
    if args.folds is not None:
        cfg.training.n_folds = args.folds
    if args.target_precision is not None:
        cfg.training.target_precision = args.target_precision
    if args.candidate_top_n is not None:
        cfg.candidate_top_n = args.candidate_top_n
    if args.k_neighbors is not None:
        cfg.retrieval.k_neighbors = args.k_neighbors
    if args.per_fold_encoder:
        cfg.training.use_per_fold_encoder = True

    try:
        cfg.validate()
    except ValueError as exc:
        p.error(str(exc))

    if args.dump_config:
        print(json.dumps(cfg.to_dict(), indent=2))
        return

    missing = [
        name
        for name, value in [("--items", args.items), ("--classes", args.classes), ("--out", args.out)]
        if value is None
    ]
    if missing:
        p.error(f"the following arguments are required: {', '.join(missing)}")

    # Validate the encoder kind up front: an unknown backend fails fast with a
    # clear message listing the registered kinds, not a deep traceback.
    try:
        enc_spec = encoder_spec(cfg.encoder.kind)
    except ValueError as exc:
        p.error(str(exc))

    label_space = read_label_space(args.classes, args.key_col, args.desc_col)
    items = read_items(args.items, args.text_col, args.label_col)

    if enc_spec.corpus_dependent:
        logging.info(
            "encoder kind %r is corpus-fitted; --encoder=%r is ignored",
            cfg.encoder.kind,
            cfg.encoder.model_name_or_path,
        )

    _, report = TrainingPipeline(cfg).run(items, label_space, output_dir=args.out)
    print("\n=== coverage report (test fold) ===")
    print(f"candidate recall      : {report.candidate_recall:.4f}")
    print(f"coverage              : {report.coverage:.4f}")
    print(f"accuracy on accepted  : {report.accuracy_on_accepted:.4f}")
    print(f"accuracy if no abstain: {report.accuracy_if_no_abstain:.4f}")
    print(f"\nwrote model + evaluation.json + model_card.md to {args.out}")


if __name__ == "__main__":
    main()
