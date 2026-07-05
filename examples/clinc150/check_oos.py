#!/usr/bin/env python
"""Summarize how the classifier handled out-of-scope queries.

For OOS queries the *desired* behavior is to abstain (route to a human). This
reads the predictions CSV written by `text-classifier-infer` and reports the
abstention rate — the headline number for the abstention story.

Usage:
    python check_oos.py build/oos_preds.csv
"""
from __future__ import annotations

import sys

import pandas as pd


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "build/oos_preds.csv"
    df = pd.read_csv(path)
    n = len(df)
    abstained = int(df["abstained"].sum())
    print(f"out-of-scope queries: {n}")
    print(f"correctly abstained : {abstained} ({100 * abstained / n:.1f}%)")
    print(f"wrongly accepted    : {n - abstained} ({100 * (n - abstained) / n:.1f}%)")
    if n - abstained:
        leaked = df.loc[~df["abstained"], ["text", "predicted_key", "confidence"]].head(10)
        print("\nexamples wrongly accepted (the model thought these were in-scope):")
        for _, r in leaked.iterrows():
            print(f"  [{r['confidence']:.2f}] {r['predicted_key']:>20}  <- {r['text'][:60]}")


if __name__ == "__main__":
    main()
