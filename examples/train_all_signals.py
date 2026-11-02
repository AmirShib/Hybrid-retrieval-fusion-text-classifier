"""Train the pipeline on a DataFrame with every signal switched on.

Signals: dense (bi-encoder) + lexical (BM25) + splade (learned sparse) build the
shortlist; cross-encoder reranks it. XGBoost fuses them all.

    python examples/train_all_signals.py data.csv model_dir/            # real models
    python examples/train_all_signals.py data.csv model_dir/ --offline  # no downloads (smoke test)

The CSV needs `text` and `label` columns. Each class also needs a description;
with only labels available we use the label itself (edit DESCRIPTIONS for better
results -- descriptions feed several signals directly).
"""

import sys

import pandas as pd

from text_classifier import (
    ClassDefinition,
    InferencePipeline,
    LabeledItem,
    LabelSpace,
    PipelineConfig,
    TrainingPipeline,
)

# Optional: a better description per label, e.g. {"bank": "בנקים, הלוואות וחשבונות"}.
DESCRIPTIONS: dict = {}

# Real models. Check the ids on Hugging Face (or point at local directories).
DENSE_MODEL = "intfloat/multilingual-e5-base"
SPLADE_MODEL = "dicta-il/alephbertgimmel-base"  # any checkpoint with its MLM head
CROSS_ENCODER_MODEL = "<a Hebrew/multilingual cross-encoder>"


def build_config(offline: bool) -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.signals = ["dense", "lexical", "splade", "cross-encoder"]
    cfg.training.n_folds = 5

    if offline:
        # Dependency-free stand-ins: exercises the whole pipeline, learns little.
        cfg.encoder.kind = "hashing"
        cfg.retrieval.splade.kind = "hashing"
        cfg.retrieval.cross_encoder.kind = "token-overlap"
        return cfg

    # Dense bi-encoder, fine-tuned per fold.
    cfg.encoder.model_name_or_path = DENSE_MODEL
    cfg.training.use_per_fold_encoder = True
    cfg.encoder.train_epochs = 1
    cfg.encoder.train_batch_size = 32

    # SPLADE: a plain MLM must be fine-tuned to be a useful sparse retriever.
    sp = cfg.retrieval.splade
    sp.kind, sp.model_name, sp.train = "splade", SPLADE_MODEL, True
    sp.train_epochs, sp.train_batch_size, sp.learning_rate = 2, 8, 2e-5

    # Cross-encoder reranker, fine-tuned per fold.
    ce = cfg.retrieval.cross_encoder
    ce.kind, ce.model_name, ce.train = "cross-encoder", CROSS_ENCODER_MODEL, True
    ce.train_epochs, ce.train_batch_size, ce.train_loss = 1, 16, "binary_cross_entropy"
    return cfg


def main(csv_path: str, out_dir: str, offline: bool) -> None:
    df = pd.read_csv(csv_path).dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(str)

    label_space = LabelSpace(
        [ClassDefinition(lab, DESCRIPTIONS.get(lab, lab)) for lab in sorted(df["label"].unique())]
    )
    items = [LabeledItem(t, lab) for t, lab in zip(df["text"].astype(str), df["label"])]

    cfg = build_config(offline)
    cfg.validate()
    artifacts, report = TrainingPipeline(cfg).run(items, label_space, output_dir=out_dir)
    print(report)

    # Quick check: reuse the trained artifacts directly.
    for p in InferencePipeline(artifacts).predict(df["text"].astype(str).head(3).tolist()):
        print(p.top_key, round(p.confidence, 3), "abstained" if p.abstained else "")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0], args[1], offline="--offline" in sys.argv)
