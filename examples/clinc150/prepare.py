#!/usr/bin/env python
"""Prepare the CLINC150 (OOS) dataset for the text-classifier demo.

CLINC150 is 150 fine-grained intents across 10 domains, plus an explicit
*out-of-scope* (OOS) bucket of queries that belong to none of them. That makes it
an ideal showcase for this package: the model learns the 150 in-scope intents,
and the OOS queries are exactly what a calibrated abstaining classifier should
decline and route to a human.

This script downloads the canonical `data_full.json` (once) and writes the
package's two-CSV input shape plus the evaluation/OOS splits:

    <out>/classes.csv       key,description   (the 150 in-scope intents)
    <out>/items_train.csv   text,label        (in-scope training queries)
    <out>/items_test.csv    text,label        (in-scope test queries -> evaluate)
    <out>/items_oos.csv     text,label        (out-of-scope queries -> abstain)

The class *descriptions* are humanized from the intent key (e.g. `bill_balance`
-> "A user request about bill balance."). They are deliberately simple; richer
descriptions improve the description-similarity signal and are a good first thing
to tune for a production demo.

Usage:
    python prepare.py [--out build] [--max-classes N] [--per-class M]

`--max-classes` / `--per-class` subsample for a fast local run; omit them for the
full dataset.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import urllib.request

DATA_URL = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"


def _download(dest: str) -> None:
    if os.path.isfile(dest):
        return
    print(f"downloading CLINC150 -> {dest}")
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    # Read the whole body and validate it parses, rather than urlretrieve (whose
    # Content-Length check trips behind chunking proxies). Write only on success.
    with urllib.request.urlopen(DATA_URL, timeout=120) as resp:
        raw = resp.read()
    json.loads(raw)  # fail loudly on a truncated/garbled download
    with open(dest, "wb") as fh:
        fh.write(raw)


def _describe(intent: str) -> str:
    """Humanize an intent key into a short natural-language description."""
    phrase = intent.replace("_", " ").strip()
    return f"A user request about {phrase}."


def _write_csv(path: str, header, rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"wrote {len(rows):>6} rows -> {path}")


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=os.path.join(here, "build"), help="output directory")
    p.add_argument("--json", default=os.path.join(here, "data_full.json"),
                   help="path to data_full.json (downloaded if absent)")
    p.add_argument("--max-classes", type=int, default=None,
                   help="keep only the first N intents (alphabetical) for a fast run")
    p.add_argument("--per-class", type=int, default=None,
                   help="cap in-scope training queries per intent")
    args = p.parse_args()

    _download(args.json)
    with open(args.json, encoding="utf-8") as fh:
        data = json.load(fh)

    intents = sorted({lab for _, lab in data["train"]})
    if args.max_classes:
        intents = intents[: args.max_classes]
    keep = set(intents)
    print(f"{len(intents)} in-scope intents")

    os.makedirs(args.out, exist_ok=True)
    _write_csv(os.path.join(args.out, "classes.csv"),
               ["key", "description"], [[i, _describe(i)] for i in intents])

    # In-scope train, optionally capped per class.
    seen: dict[str, int] = {}
    train_rows = []
    for text, lab in data["train"]:
        if lab not in keep:
            continue
        if args.per_class and seen.get(lab, 0) >= args.per_class:
            continue
        seen[lab] = seen.get(lab, 0) + 1
        train_rows.append([text, lab])
    _write_csv(os.path.join(args.out, "items_train.csv"), ["text", "label"], train_rows)

    # In-scope test (for `text-classifier-eval`).
    test_rows = [[t, l] for t, l in data["test"] if l in keep]
    _write_csv(os.path.join(args.out, "items_test.csv"), ["text", "label"], test_rows)

    # Out-of-scope queries (for the abstention demonstration via `infer`).
    oos_rows = [[t, "oos"] for t, _ in data["oos_test"]]
    _write_csv(os.path.join(args.out, "items_oos.csv"), ["text", "label"], oos_rows)

    print("\nNext:")
    print(f"  text-classifier-train --items {args.out}/items_train.csv "
          f"--classes {args.out}/classes.csv --out {args.out}/model --encoder-kind tfidf")
    print(f"  text-classifier-eval  --model {args.out}/model --input {args.out}/items_test.csv "
          f"--output {args.out}/in_scope_report.json")
    print(f"  text-classifier-infer --model {args.out}/model --input {args.out}/items_oos.csv "
          f"--output {args.out}/oos_preds.csv")
    print(f"  python {os.path.join(here, 'check_oos.py')} {args.out}/oos_preds.csv")


if __name__ == "__main__":
    main()
