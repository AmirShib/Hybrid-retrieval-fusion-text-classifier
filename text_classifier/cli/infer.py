#!/usr/bin/env python
r"""Classify new items with a trained model.

Usage:
    text-classifier-infer \
        --model model_dir/ \
        --input new_items.csv \       # column: text
        --output predictions.csv

Output columns: text, predicted_key, top_key, confidence, abstained, margin.
Rows where the system abstained have an empty predicted_key (route to a human).
"""

from __future__ import annotations

import argparse

import pandas as pd

from .. import InferencePipeline
from ._common import add_logging_arg, configure_logging, read_texts


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--text-col", default="text")
    add_logging_arg(p)
    args = p.parse_args()
    configure_logging(args.log_level)

    _, texts = read_texts(args.input, args.text_col)

    pipeline = InferencePipeline.from_directory(args.model)
    preds = pipeline.predict(texts)

    out = pd.DataFrame(
        {
            "text": texts,
            "predicted_key": [pr.predicted_key or "" for pr in preds],
            "top_key": [pr.top_key for pr in preds],
            "confidence": [pr.confidence for pr in preds],
            "abstained": [pr.abstained for pr in preds],
            "margin": [pr.margin for pr in preds],
        }
    )
    out.to_csv(args.output, index=False)
    accepted = sum(not pr.abstained for pr in preds)
    print(
        f"wrote {len(out)} predictions to {args.output} "
        f"({accepted} accepted, {len(out) - accepted} abstained)"
    )


if __name__ == "__main__":
    main()
