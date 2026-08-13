# ISCO-08 — coding job titles to an official occupation classification

The other two examples show the package on a benchmark dataset
([CLINC150](../clinc150/)) and on messy real-world text
([COICOP Hebrew](../coicop_hebrew/)). This one is the **reference benchmark for
official statistical classification**: a real taxonomy used by national
statistical offices, with a real labeled item set, both fully public.

Occupation coding — mapping a survey write-in like *"lathe operator"* to an
ISCO-08 unit group — is one of the highest-volume human-in-the-loop
classification tasks in official statistics. It is also nearly impossible to
publish reproducible results on, because the labeled data is almost always
confidential survey microdata. This example uses the ILO's own materials
instead, so anyone can reproduce the numbers below exactly.

## The data

Two workbooks from the [ILO's ISCO-08 page](https://isco-ilo.netlify.app/en/isco-08/),
downloaded automatically by `prepare.py`:

| file | becomes | what it is |
|---|---|---|
| `ISCO-08 EN Structure and definitions.xlsx` | `classes.csv` / `classes.jsonl` | All 4 hierarchy levels (10 major → 43 sub-major → 130 minor → **436 unit groups**), each with a definition, a task list, included occupations and excluded occupations. |
| `ISCO-08 -88 EN Index.xlsx` | `items.csv` | **7,018 job titles** with their ISCO-08 *and* ISCO-88 unit-group codes — the ILO's own alphabetical coding index, the list a human coder looks a write-in up in. |

Why this is a good fit for this package specifically:

- **Naturally imbalanced.** Median 13 titles per unit group, but a long tail down
  to 1 and a head up to 113. Nothing was subsampled to induce that — it is a
  property of the source.
- **Rich class descriptions.** The `Included occupations` field is a concrete
  list of example job titles per class, which is exactly the vocabulary a short
  write-in echoes. That is what makes the description-similarity signal work
  here without any labeled data at all.
- **Two revisions of the same taxonomy.** Each item carries both its ISCO-08 and
  its ISCO-88 code, so `--target isco88` relabels the same items into the
  previous revision. Taxonomy revisions are the recurring operational problem in
  official statistics, and this gives you a real one to test against.

## Build it

```bash
pip install .            # core install: torch-free, TF-IDF encoder
pip install openpyxl     # to read the ILO workbooks

python examples/isco/prepare.py --out examples/isco/build --min-examples 3
```

```
wrote .../classes.csv and .../classes.jsonl (436 classes)
  --min-examples 3: dropped 15 items across 8 class(es) with too few titles;
  those classes remain in classes.csv with no labeled examples
wrote .../items.csv (7002 items; dropped 1 malformed, 0 out of scope)
  428 classes represented; items per class: min 3, median 13, max 113
```

`--min-examples 3` exists because `StratifiedKFold` needs k examples of every
*labeled* class and eight unit groups have fewer than three titles. It drops
**items, never classes** — those eight classes stay in `classes.csv` with their
official descriptions, so they remain live candidates reachable through the
description signal alone. That is the honest shape of a real taxonomy: every
code is codeable, but not every code has training data.

The one dropped item is a data-entry slip in the ILO index (a row coded `1` with
the title `n`).

## Train and evaluate

```bash
text-classifier-train \
    --items examples/isco/build/items.csv \
    --classes examples/isco/build/classes.jsonl \
    --out examples/isco/model/ \
    --encoder-kind tfidf --folds 3 --target-precision 0.95
```

Pass `classes.jsonl`, not `classes.csv` — it keeps title, definition, inclusions,
exclusions and the hierarchy path as separate fields rather than gluing them into
one string.

**Results — 436 classes, 7,002 items, TF-IDF encoder, ~30 seconds on a laptop CPU:**

| metric | value |
|---|---|
| candidate recall (the accuracy ceiling) | 0.983 |
| accuracy, no abstention | 0.701 |
| coverage at `--target-precision 0.95` | 0.334 |
| accuracy on accepted | 0.953 |
| expected calibration error | 0.036 |
| Brier score | 0.139 |

Read that as: **a third of a 436-way coding workload auto-codes at 95%
accuracy**, and the rest routes to a human — with the threshold tuned to hold
that bar, not hoped to.

These are **floor numbers**. The TF-IDF encoder has no semantics at all; it is
here so the example runs offline with no torch and no model download. Swap in a
real bi-encoder (drop `--encoder-kind tfidf`) and every dense signal improves.

### The risk–coverage trade-off

The operating point is a knob, not a verdict. From `evaluation.json`:

| coverage | accuracy on accepted |
|---|---|
| 0.11 | 0.976 |
| 0.21 | 0.968 |
| 0.32 | 0.955 |
| 0.42 | 0.937 |
| 0.53 | 0.915 |
| 0.74 | 0.837 |
| 1.00 | 0.701 |

Re-tune without retraining:

```bash
text-classifier-tune --model examples/isco/model/ \
    --input fresh_labeled.csv --target-precision 0.99
```

### What fusion is actually buying

`evaluation.json` carries a per-signal report. On this dataset:

| signal | top-1 accuracy alone |
|---|---|
| BM25 ↔ description | 0.558 |
| dense ↔ prototype | 0.480 |
| BM25 ↔ kNN | 0.476 |
| dense ↔ kNN | 0.447 |
| dense ↔ description | 0.410 |
| **fused (no abstention)** | **0.701** |

Fusion beats the best single signal by **14 points**. The five signals put a
single class on top only 19% of the time (mean 2.33 distinct top-1 classes per
item), which is why there is something to fuse: they are reading different
evidence, not agreeing loudly.

Note that BM25-on-descriptions is the strongest individual signal here. That is
the `Included occupations` list doing the work — exact term overlap between a
write-in and an official example list.

## Other slices

```bash
# Minor groups (130 classes) instead of unit groups.
python examples/isco/prepare.py --out build --level 3

# Professionals + technicians only.
python examples/isco/prepare.py --out build --major-groups 2,3

# The same items labeled into the previous revision of the taxonomy.
python examples/isco/prepare.py --out build --target isco88
```

A word on `--level 3`: it is **not** simply easier. Same encoder, same settings,
130 classes instead of 436, and accuracy without abstention *drops* to 0.648
(candidate recall 0.999, coverage 0.261 at 0.97 accuracy-on-accepted). The minor
groups' descriptions are longer but more abstract — the concrete example-title
lists live at the unit-group level. Description quality dominates class count
here, which is a useful thing to know before assuming a coarser taxonomy is a
softer target.

## Licensing

ISCO-08 is published by the International Labour Organization. The workbooks are
downloaded into `build/` at runtime and are **not** committed to this repository.
Check the ILO's terms before redistributing them or publishing derived data.
