# COICOP demo — classifying short Hebrew item names (cross-lingual, zero-shot)

Where the [CLINC150 demo](../clinc150/) showcases **calibrated abstention** on a
labeled English dataset, this example tackles a different shape of problem:

- **Short, messy product names written in Hebrew** — `עוגיות סנדביץ`,
  `שמן שומשום`, `דלוורדה ניוקי` — the kind of strings you export from a product
  catalog or a receipt.
- Classified into the international **[COICOP 2018](https://unstats.un.org/unsd/classifications/Econ)**
  taxonomy of household consumption (the codes statistics offices use for CPI and
  household-expenditure surveys), whose labels are in **English**.
- With **no labeled training data** — only the class descriptions and a list of
  unlabeled items.

A **multilingual** sentence encoder bridges Hebrew items and English COICOP labels
in one embedding space, and the package runs in **zero-shot retrieval** mode: the
item ↔ class-description similarity signal (signal #1 of the five), with a
confidence floor for abstention. It's the same human-in-the-loop story as
CLINC150, at the floor of what's possible before any labels exist.

## ▶ Start here: the notebook

**[`coicop_hebrew_classification.ipynb`](coicop_hebrew_classification.ipynb)** is
the end-to-end walkthrough — download the COICOP taxonomy, load a Hebrew+English
encoder, classify the items, and abstain on the uncertain ones, with a chart.

```bash
pip install .                 # from the repo root (brings in sentence-transformers)
pip install pandas openpyxl matplotlib jupyter
jupyter notebook examples/coicop_hebrew/coicop_hebrew_classification.ipynb
```

> Unlike the CLINC150 demo, this one is **not** offline: a real multilingual
> bi-encoder is the whole point, so it downloads one model (~1 GB the first time,
> then cached) and needs `torch`.

## The pieces

| file | what it is |
|---|---|
| `prepare.py` | Downloads the UN SD **COICOP 2018 hierarchies & mappings** workbook and writes `build/classes.csv` (`key,description`) for one level of the hierarchy. |
| `items.csv` | The items to classify — a single **`name`** column of short Hebrew product names. Replace it with your own export (keep the `name` column). |
| `data.csv` | ~160k Hebrew item names labelled into a **COICOP-like Hebrew retail taxonomy** (186 categories; columns `lbcItemName,label`). Real supervision, used in the second half of the notebook to train the full pipeline. |
| `coicop_hebrew_classification.ipynb` | The walkthrough. |

### Building the classes from the command line

```bash
# Subclass level (01.1.1.1 …), food division only — the default.
python examples/coicop_hebrew/prepare.py --out examples/coicop_hebrew/build

# Other slices:
python examples/coicop_hebrew/prepare.py --all            # every division
python examples/coicop_hebrew/prepare.py --divisions 01,02 --level 3
```

`classes.csv` is the package's standard class input — `key` is the COICOP code
and `description` folds the official title together with the *intro* and
*includes* notes (e.g. "cornflakes, oatmeal, muesli, granola"). That concrete
product vocabulary is what an item name echoes, so richer descriptions retrieve
better; it's the first thing to tune.

## Choosing the encoder

The zero-shot half defaults to `sentence-transformers/LaBSE` — trained on
translation pairs, it's the most robust Hebrew↔English matcher of the four on
these items. Set `MODEL_NAME` in the encoder cell to swap it:

| model | size | notes |
|---|---|---|
| `sentence-transformers/LaBSE` *(default)* | ~1.8 GB | translation-pair model, very robust cross-lingual |
| `paraphrase-multilingual-mpnet-base-v2` | ~1.1 GB | strong, a bit lighter |
| `paraphrase-multilingual-MiniLM-L12-v2` | ~470 MB | faster/smaller; used for the trainable section |
| `intfloat/multilingual-e5-base` | ~1.1 GB | expects `query:`/`passage:` prefixes; was less discriminative here |

## From zero-shot to the full pipeline

Zero-shot description similarity is the **floor**. The notebook's second half
shows the ceiling: it trains the full `TrainingPipeline` on `data.csv` — real
Hebrew items labelled into a COICOP-like Hebrew taxonomy — lighting up class
prototypes, dense/BM25 kNN, the XGBoost fusion model, isotonic calibration, and a
threshold tuned for a target accuracy. On the held-out split it reaches ~0.82
coverage at ~0.89 accuracy-on-accepted, and the transliterated pasta/candy names
that *abstained* zero-shot now classify confidently. The trainable section uses
the lighter `MiniLM` encoder for speed (the OOF loop re-encodes per fold).

To target **COICOP codes** rather than the Hebrew taxonomy, either crosswalk the
Hebrew labels to COICOP (compose `item → Hebrew label → COICOP`) or fine-tune the
encoder on these `(item, description)` pairs to sharpen the zero-shot COICOP step.

## Data source

COICOP 2018 structure is taken from the UN Statistics Division's COICOP 2018 /
COICOP 1999 correspondence workbook (its `COICOP 2018` worksheet carries the full
hierarchy). See <https://unstats.un.org/unsd/classifications/Econ>. The workbook
is downloaded into `build/` and is git-ignored. `data.csv` (the labelled Hebrew
retail sample) is committed alongside the example.
