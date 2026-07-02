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
"""

from __future__ import annotations

import argparse
import logging

from .. import PipelineConfig, TrainingPipeline
from ..infrastructure.registry import encoder_spec
from ._common import add_logging_arg, configure_logging, read_items, read_label_space


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--items", required=True)
    p.add_argument("--classes", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--encoder-kind",
        default="sentence-transformers",
        help="encoder backend (registry key): 'sentence-transformers' "
        "(default), 'tfidf' (torch-free, air-gapped), or 'hashing' "
        "(dependency-free baseline / smoke test)",
    )
    p.add_argument(
        "--encoder",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="model name/path for the sentence-transformers encoder; "
        "ignored by corpus-fitted encoders such as tfidf",
    )
    p.add_argument("--folds", type=int, default=5)
    p.add_argument(
        "--target-precision",
        type=float,
        default=0.95,
        help="target accuracy on accepted items; the threshold is tuned "
        "for max coverage subject to this",
    )
    p.add_argument("--candidate-top-n", type=int, default=10)
    p.add_argument("--k-neighbors", type=int, default=20)
    p.add_argument(
        "--per-fold-encoder",
        action="store_true",
        help="rigorous (expensive): fine-tune a fresh encoder per fold",
    )
    p.add_argument("--text-col", default="text", help="items.csv text column")
    p.add_argument("--label-col", default="label", help="items.csv label column")
    p.add_argument("--key-col", default="key", help="classes.csv key column")
    p.add_argument("--desc-col", default="description", help="classes.csv description column")
    add_logging_arg(p)
    args = p.parse_args()
    configure_logging(args.log_level)

    # Validate the encoder kind up front: an unknown backend fails fast with a
    # clear message listing the registered kinds, not a deep traceback.
    try:
        enc_spec = encoder_spec(args.encoder_kind)
    except ValueError as exc:
        p.error(str(exc))

    label_space = read_label_space(args.classes, args.key_col, args.desc_col)
    items = read_items(args.items, args.text_col, args.label_col)

    cfg = PipelineConfig(candidate_top_n=args.candidate_top_n)
    cfg.encoder.kind = args.encoder_kind
    cfg.encoder.model_name_or_path = args.encoder
    if enc_spec.corpus_dependent:
        logging.info(
            "encoder kind %r is corpus-fitted; --encoder=%r is ignored",
            args.encoder_kind,
            args.encoder,
        )
    cfg.retrieval.k_neighbors = args.k_neighbors
    cfg.training.n_folds = args.folds
    cfg.training.target_precision = args.target_precision
    cfg.training.use_per_fold_encoder = args.per_fold_encoder

    _, report = TrainingPipeline(cfg).run(items, label_space, output_dir=args.out)
    print("\n=== coverage report (test fold) ===")
    print(f"candidate recall      : {report.candidate_recall:.4f}")
    print(f"coverage              : {report.coverage:.4f}")
    print(f"accuracy on accepted  : {report.accuracy_on_accepted:.4f}")
    print(f"accuracy if no abstain: {report.accuracy_if_no_abstain:.4f}")
    print(f"\nwrote model + evaluation.json + model_card.md to {args.out}")


if __name__ == "__main__":
    main()
