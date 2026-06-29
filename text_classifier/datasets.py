"""Synthetic dataset generation for demos, examples, and offline tests.

Ships inside the package (rather than only under ``tests/``) so the offline
demo, the CI wheel-install smoke test, and anyone evaluating the package can
generate a small, reproducible, deliberately imbalanced dataset without a real
corpus or any network access.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .domain import ClassDefinition, LabeledItem, LabelSpace


def make_synthetic(
    n_classes: int = 40, per_class: int = 60, seed: int = 0
) -> Tuple[LabelSpace, List[LabeledItem]]:
    """Build a small, reproducible, imbalanced classification dataset.

    Parameters
    ----------
    n_classes:
        Number of distinct classes to generate.
    per_class:
        Item count for the "majority" classes; minority classes get
        ``max(8, per_class // 5)`` items.
    seed:
        Seed for the NumPy RNG so the output is reproducible.

    Returns
    -------
    (LabelSpace, list[LabeledItem])
        A label space whose descriptions share a per-class vocabulary "theme",
        and items whose text is drawn mostly from their class theme plus a little
        shared-vocabulary noise — enough lexical/semantic signal for retrieval to
        be non-trivial.

    Notes
    -----
    The imbalance (``c % 7`` picks majority vs. minority) is intentional: this
    package targets imbalanced data, and the abstention/calibration behavior is
    only interesting when class frequencies differ.
    """
    rng = np.random.default_rng(seed)
    vocab = [f"w{i}" for i in range(600)]
    definitions: List[ClassDefinition] = []
    items: List[LabeledItem] = []
    for c in range(n_classes):
        theme = rng.choice(vocab, size=8, replace=False)
        definitions.append(ClassDefinition(
            key=f"CLS{c:03d}",
            description=f"class about {' '.join(theme[:4])} and related {' '.join(theme[4:])}",
        ))
        count = per_class if c % 7 else max(8, per_class // 5)
        for _ in range(count):
            k = int(rng.integers(4, 9))
            words = list(rng.choice(theme, size=min(k, len(theme)), replace=True))
            words += list(rng.choice(vocab, size=2, replace=True))
            rng.shuffle(words)
            items.append(LabeledItem(text=" ".join(words), label=f"CLS{c:03d}"))
    rng.shuffle(items)
    return LabelSpace(definitions), items
