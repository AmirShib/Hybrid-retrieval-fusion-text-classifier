# Hybrid retrieval-fusion text classifier

Classifies free-text items into one of many text-described classes,
and **abstains** when it isn't confident enough —> routing those items to a human
(Built with imbalanced data and air-gapped setting in mind).

## How it works

Five retrieval signals are scored for every item against a candidate set of
classes, then a single pointwise model fuses them into a calibrated
`P(this candidate is the true class)`:

1. dense item ↔ class-**description** similarity (bi-encoder)
2. dense item ↔ class-**prototype** similarity (mean of a class's example embeddings)
3. dense **kNN** over training examples
4. BM25 item ↔ class-description
5. BM25 **kNN** over training examples

Description and prototype are kept separate on purpose: their disagreement is a
useful feature. The candidate set is the union of each signal's top-N, so
**candidate recall is the ceiling on accuracy** and is reported every run.

The fusion model is **pointwise** — one shared binary model over ~28 features per
(item, candidate). 

Confidence is isotonic-calibrated, and a threshold is tuned for a target
accuracy (max coverage subject to accuracy ≥ target), with per-class thresholds
where a class had enough calibration support.

### Leakage control

* **Out-of-fold features** — each item is scored against indices/prototypes built
  from *other* folds (`StratifiedKFold`).
* **Held-out calibration/test** — thresholds and the coverage report come from
  folds the fusion model never trained on.
* **Encoder** — by default a single shared encoder is used (cheap). For full
  rigor, set `use_per_fold_encoder=True` to fine-tune a fresh encoder per fold.
  The encoder is fine-tuned with `MultipleNegativesSymmetricRankingLoss` on
  (item, class-description) pairs.

## Layout (domain-driven / hexagonal)

```
text_classifier/
  domain/           framework-free core
    models.py         value objects + LabelSpace aggregate
    ports.py          abstract interfaces (encoder, retrievers, fusion, calibrator)
    services.py       feature schema, candidate/abstention policies, threshold tuner
  infrastructure/   adapters implementing the ports
    encoder.py        SentenceTransformer + MNR-symmetric fine-tuning
    retrieval.py      BM25 (precomputed weight matrix) + dense retriever w/ prototypes
    fusion.py         XGBoost fusion model + isotonic calibrator
    persistence.py    save/load a model directory
  application/      use cases
    features.py       vectorized (item, candidate) feature assembler
    scoring.py        confidence + per-item argmax (shared by both pipelines)
    training.py       TrainingPipeline   <-- training entry point
    inference.py      InferencePipeline  <-- inference entry point
  config.py         configuration dataclasses
scripts/
  train.py          CLI: train from CSVs -> model directory
  infer.py          CLI: model directory + CSV -> predictions
  demo.py           offline smoke test (HashingEncoder double; no model download)
```

The domain layer imports no ML framework; infrastructure depends on the domain;
the application layer orchestrates through the ports. The two pipelines are the
public entry points.

## Efficiency notes

* Every signal is computed as a `(batch × n_classes)` matrix; candidate rows are
  gathered with numpy fancy-indexing — no per-row Python loops.
* BM25 precomputes a per-(doc, term) weight matrix `W`; because query-term
  frequency is ignored, scoring a query batch is the sparse mat-mul
  `Q_binary @ W.T`.
* kNN and feature assembly are query-chunked to bound peak memory.

## Install

```bash
pip install .                 # core (includes sentence-transformers)
pip install .[lightgbm]       # + optional LightGBM fusion backend
pip install .[test]           # + pytest for the test suite
```

Installing exposes three console commands — `text-classifier-train`,
`text-classifier-infer`, and `text-classifier-eval`. From a source checkout you
can equivalently run `python -m scripts.train` / `scripts.infer` /
`text_classifier.cli.evaluate`.

## Usage

Train (writes the model directory plus `evaluation.json` and `model_card.md`):

```bash
text-classifier-train \
    --items items.csv \        # columns: text,label
    --classes classes.csv \    # columns: key,description
    --out model_dir/ \
    --target-precision 0.95
# add --per-fold-encoder for the rigorous (expensive) encoder path
```

For a torch-free, air-gapped run (no torch, no model download) use the TF-IDF
encoder backend (corpus-fitted, so `--encoder` is ignored). For a
dependency-free smoke test there is also a non-semantic `hashing` encoder:

```bash
text-classifier-train --items items.csv --classes classes.csv \
    --out model_dir/ --encoder-kind tfidf
```

Predict:

```bash
text-classifier-infer --model model_dir/ --input new_items.csv --output preds.csv
```

Evaluate a trained model on a labeled set (coverage, accuracy-on-accepted,
calibration — Brier/ECE — a risk-coverage curve, and a per-class breakdown).
Use it to validate before deploying, or to watch for drift over time:

```bash
text-classifier-eval --model model_dir/ --input labeled.csv --output report.json
```

Each trained model directory carries its own evidence: `evaluation.json` (the
full held-out report) and `model_card.md` (a human-readable summary with the
package version, dataset shape, headline metrics, and the abstention thresholds).

### Worked example

`examples/clinc150/` is a runnable, fully offline demo on CLINC150 (150 intents +
an out-of-scope set). It shows the abstention knob in action — raising the
confidence bar routes more out-of-scope queries to a human while keeping in-scope
accuracy high. Start with the notebook walkthrough,
`examples/clinc150/clinc150_abstention_demo.ipynb` (cell-by-cell, with charts);
`examples/clinc150/README.md` has the command-line equivalent.

Library:

```python
from text_classifier import (PipelineConfig, LabelSpace, LabeledItem,
                             ClassDefinition, TrainingPipeline, InferencePipeline)

label_space = LabelSpace([ClassDefinition("CLS001", "invoices and billing"), ...])
items = [LabeledItem("late fee on my bill", "CLS001"), ...]

artifacts, report = TrainingPipeline(PipelineConfig()).run(items, label_space, output_dir="model_dir/")
print(report)  # coverage / accuracy-on-accepted / candidate recall

preds = InferencePipeline.from_directory("model_dir/").predict(["where is my refund"])
```

