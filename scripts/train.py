#!/usr/bin/env python
r"""Train the classifier.

Usage:
    python -m scripts.train \
        --items items.csv \           # columns: text,label
        --classes classes.csv \       # columns: key,description
        --out model_dir/ \
        [--per-fold-encoder] [--target-precision 0.95] [--folds 5]

`label` in items.csv must match a `key` in classes.csv.
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd

from text_classifier import (
    ClassDefinition,
    LabeledItem,
    LabelSpace,
    PipelineConfig,
    TrainingPipeline,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--items", required=True)
    p.add_argument("--classes", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--target-precision", type=float, default=0.95)
    p.add_argument("--candidate-top-n", type=int, default=10)
    p.add_argument("--k-neighbors", type=int, default=20)
    p.add_argument("--per-fold-encoder", action="store_true",
                   help="rigorous (expensive): fine-tune a fresh encoder per fold")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    classes_df = pd.read_csv(args.classes)
    label_space = LabelSpace([ClassDefinition(str(r.key), str(r.description))
                              for r in classes_df.itertuples()])

    items_df = pd.read_csv(args.items)
    items = [LabeledItem(str(r.text), str(r.label)) for r in items_df.itertuples()]

    cfg = PipelineConfig(candidate_top_n=args.candidate_top_n)
    cfg.encoder.model_name_or_path = args.encoder
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


if __name__ == "__main__":
    main()
