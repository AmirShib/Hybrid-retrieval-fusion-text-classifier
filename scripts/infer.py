#!/usr/bin/env python
r"""Classify new items with a trained model.

Usage:
    python -m scripts.infer \
        --model model_dir/ \
        --input new_items.csv \       # column: text
        --output predictions.csv

Output columns: text, predicted_key, top_key, confidence, abstained, margin.
Rows where the system abstained have an empty predicted_key (route to a human).
"""
from __future__ import annotations

import argparse

import pandas as pd

from text_classifier import InferencePipeline


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--text-col", default="text")
    args = p.parse_args()

    df = pd.read_csv(args.input)
    texts = df[args.text_col].astype(str).tolist()

    pipeline = InferencePipeline.from_directory(args.model)
    preds = pipeline.predict(texts)

    out = pd.DataFrame({
        "text": texts,
        "predicted_key": [p.predicted_key or "" for p in preds],
        "top_key": [p.top_key for p in preds],
        "confidence": [p.confidence for p in preds],
        "abstained": [p.abstained for p in preds],
        "margin": [p.margin for p in preds],
    })
    out.to_csv(args.output, index=False)
    accepted = sum(not p.abstained for p in preds)
    print(f"wrote {len(out)} predictions to {args.output} "
          f"({accepted} accepted, {len(out) - accepted} abstained)")


if __name__ == "__main__":
    main()
