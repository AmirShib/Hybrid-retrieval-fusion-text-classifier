#!/usr/bin/env python
"""Offline smoke test of the full wiring.

This runs the training and inference pipelines end-to-end on synthetic data
WITHOUT downloading a model, by injecting a `HashingEncoder` test double in place
of a real bi-encoder. It exercises candidate generation, the vectorized feature
assembler, XGBoost fusion, calibration, threshold tuning, and abstention.

It does NOT validate embedding quality (the hashing encoder is deliberately
trivial) — only that every component fits together and runs.

    python -m scripts.demo

Requires: numpy, pandas, scikit-learn, scipy, xgboost (no network, no torch).
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from text_classifier import (
    ClassDefinition,
    LabeledItem,
    LabelSpace,
    PipelineConfig,
    TrainingPipeline,
)
from text_classifier.application import InferencePipeline
from text_classifier.domain import TextEncoder


class HashingEncoder(TextEncoder):
    """Deterministic bag-of-hashed-tokens embeddings, L2-normalized. A test
    double: shared tokens -> higher cosine, so retrieval is meaningful enough
    to smoke-test the pipeline offline."""

    def __init__(self, dim: int = 128):
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in str(t).lower().split():
                h = hash(tok)
                out[i, h % self.dim] += 1.0 if (h >> 32) & 1 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.clip(norms, 1e-8, None)


def make_synthetic(n_classes=40, per_class=60, seed=0):
    rng = np.random.default_rng(seed)
    vocab = [f"w{i}" for i in range(600)]
    definitions, items = [], []
    for c in range(n_classes):
        theme = rng.choice(vocab, size=8, replace=False)
        definitions.append(ClassDefinition(
            key=f"CLS{c:03d}",
            description=f"class about {' '.join(theme[:4])} and related {' '.join(theme[4:])}",
        ))
        # imbalance: some classes much smaller
        count = per_class if c % 7 else max(8, per_class // 5)
        for _ in range(count):
            k = rng.integers(4, 9)
            words = list(rng.choice(theme, size=min(k, len(theme)), replace=True))
            words += list(rng.choice(vocab, size=2, replace=True))  # noise
            rng.shuffle(words)
            items.append(LabeledItem(text=" ".join(words), label=f"CLS{c:03d}"))
    rng.shuffle(items)
    return LabelSpace(definitions), items


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    label_space, items = make_synthetic()
    print(f"{len(items)} items across {label_space.size} classes")

    cfg = PipelineConfig(candidate_top_n=8)
    cfg.retrieval.k_neighbors = 15
    cfg.training.n_folds = 5
    cfg.training.target_precision = 0.85
    cfg.training.per_class_min_support = 40
    cfg.training.use_per_fold_encoder = False  # use the injected shared encoder

    pipeline = TrainingPipeline(cfg, shared_encoder=HashingEncoder(dim=128))
    artifacts, report = pipeline.run(items, label_space, output_dir=None)

    print("\n=== coverage report (test fold) ===")
    print(f"candidate recall      : {report.candidate_recall:.4f}")
    print(f"coverage              : {report.coverage:.4f}")
    print(f"accuracy on accepted  : {report.accuracy_on_accepted:.4f}")
    print(f"accuracy if no abstain: {report.accuracy_if_no_abstain:.4f}")

    sample = [items[i].text for i in range(5)]
    preds = InferencePipeline(artifacts).predict(sample)
    print("\n=== sample predictions ===")
    for text, pred in zip(sample, preds):
        verdict = pred.predicted_key if not pred.abstained else "ABSTAIN"
        print(f"  {verdict:>10}  conf={pred.confidence:.3f}  top={pred.top_key}  | {text[:50]}")


if __name__ == "__main__":
    main()
