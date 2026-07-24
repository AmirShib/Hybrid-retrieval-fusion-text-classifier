"""Update a deployed model's classes and/or labeled examples, without
retraining the fusion model or calibrator (T68).

The fusion model is class-agnostic by construction: every feature in
``FEATURE_NAMES`` is a per-*candidate* retrieval signal (a similarity, a rank,
a count, an agreement flag) — there is no per-class weight anywhere in it. So
growing the label space or adding examples only changes the class-indexed
*retrieval* state (dense prototypes/description embeddings, BM25); the trained
fusion model and calibrator are reused verbatim, and every existing class's
index (and therefore its threshold and its predictions) is untouched.

Three independent things ``update`` can do, any subset at once:

1. **Add new classes** (``classes`` includes keys not yet in the label space):
   appended at the end (existing indices never move), description-only until
   examples arrive — the same in-memory operation ``with_added_classes`` (T78)
   performs, just persisted this time.
2. **Edit an existing class's description** (``classes`` includes an existing
   key with different text): only that description is re-embedded.
3. **Add labeled examples** (``new_items``): merges with the persisted/supplied
   corpus and rebuilds the example-indexed retrieval state. The dense side only
   *encodes the delta* (old example embeddings are reused as-is — the encoder
   is frozen, so re-encoding them would reproduce the same vectors at needless
   cost); the lexical side needs a full BM25 refit because its IDF is
   corpus-global, so there is no way to append to it incrementally.

``update`` never reorders or removes a class — there is no code path for it.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..domain import ClassDefinition, LabeledItem, LabelSpace
from ..infrastructure import DeployedArtifacts, encoder_is_corpus_dependent
from ..infrastructure.retrieval import LexicalRetrieverAdapter

log = logging.getLogger(__name__)


def update(
    artifacts: DeployedArtifacts,
    corpus: Optional[Sequence[LabeledItem]],
    classes: Optional[Sequence[ClassDefinition]] = None,
    new_items: Sequence[LabeledItem] = (),
) -> Tuple[DeployedArtifacts, Optional[List[LabeledItem]]]:
    """Widen ``artifacts``' classes and/or add labeled examples.

    Returns ``(updated_artifacts, updated_corpus)``. Neither ``artifacts`` nor
    ``corpus`` is mutated. ``updated_corpus`` is ``corpus`` (unchanged) when
    ``new_items`` is empty, ``corpus + new_items`` when there are new examples
    to merge, or ``None`` when there was no corpus to begin with and no
    examples were added — the caller decides whether/what to persist.

    Parameters
    ----------
    corpus:
        The model's original raw training items (loaded from the persisted
        ``corpus.jsonl.gz``, or supplied via ``--base-items`` for a dir that
        predates it). Only required when ``new_items`` is non-empty — a
        classes-only update (new descriptions, edited descriptions) never
        touches the example-indexed state, so it needs no corpus at all.
    classes:
        The *full* set of classes this update cares about (same shape as a
        classes CSV): every key already in the label space must be present —
        an update never removes or reorders a class, so a ``classes`` that
        drops one is rejected rather than silently doing something surprising.
        A key not yet in the label space is a new class (appended). An
        existing key whose description differs from what is stored is an
        edit (only that description is re-embedded); identical text is a
        no-op. Pass ``None`` (default) to change no classes at all.
    new_items:
        Labeled examples to add. Each label must resolve in the (possibly
        just-widened) label space.
    """
    new_items = list(new_items)
    if classes is None and not new_items:
        raise ValueError("update requires at least one of `classes` or `new_items`")

    label_space = artifacts.label_space
    encoder = artifacts.encoder
    dense = artifacts.dense
    lexical = artifacts.lexical

    # ------------------------------------------------------------- 1. classes
    new_defs: List[ClassDefinition] = []
    edited: Dict[int, str] = {}
    if classes is not None:
        current_keys = set(label_space.keys)
        file_keys = {c.key for c in classes}
        missing = sorted(current_keys - file_keys)
        if missing:
            shown = missing[:10]
            suffix = " ..." if len(missing) > 10 else ""
            raise ValueError(
                f"`classes` is missing {len(missing)} class key(s) already in the model: "
                f"{shown}{suffix}. update never removes or reorders classes; retrain if "
                "you need to change the taxonomy that way."
            )
        seen_new = set()
        for c in classes:
            if c.key not in current_keys:
                if c.key in seen_new:
                    raise ValueError(f"duplicate new class key in `classes`: {c.key!r}")
                seen_new.add(c.key)
                new_defs.append(c)
            else:
                idx = label_space.index_of(c.key)
                if c.description != label_space.descriptions[idx]:
                    edited[idx] = c.description

    if new_defs:
        dense = dense.with_added_classes(encoder, [d.description for d in new_defs])
        current_defs = [
            ClassDefinition(k, d) for k, d in zip(label_space.keys, label_space.descriptions)
        ]
        label_space = LabelSpace(current_defs + new_defs)
    if edited:
        dense = dense.with_updated_descriptions(encoder, edited)
    if new_defs or edited:
        lexical = lexical.with_added_descriptions(label_space.descriptions)
        log.info(
            "classes: %d new, %d description edit(s)", len(new_defs), len(edited)
        )

    # -------------------------------------------------------- 2. new examples
    updated_corpus: Optional[List[LabeledItem]]
    if new_items:
        known = set(label_space.keys)
        unknown = sorted({it.label for it in new_items if it.label not in known})
        if unknown:
            shown = unknown[:10]
            suffix = " ..." if len(unknown) > 10 else ""
            raise ValueError(
                f"{len(unknown)} new item label(s) are not in the label space "
                f"(add them via `classes` first): {shown}{suffix}"
            )
        if corpus is None:
            raise ValueError(
                "adding examples requires the original training corpus, and this model "
                "dir has none persisted (it predates --store-corpus, or opted out). "
                "Either pass the original items via --base-items, or retrain with "
                "--store-corpus so future updates don't need it."
            )
        if encoder_is_corpus_dependent(artifacts.config.encoder):
            log.warning(
                "encoder kind %r is corpus-fitted; its vocabulary was frozen at training "
                "time and will not learn from these new examples' vocabulary. Existing "
                "terms still score normally, but if the new items introduce substantial "
                "new vocabulary, consider a full retrain instead.",
                artifacts.config.encoder.kind,
            )

        corpus = list(corpus)
        merged_items = corpus + new_items
        merged_texts = [it.text for it in merged_items]
        merged_labels = np.array(
            label_space.encode_labels([it.label for it in merged_items]), dtype=np.int64
        )
        new_texts = [it.text for it in new_items]
        new_labels = np.array(
            label_space.encode_labels([it.label for it in new_items]), dtype=np.int64
        )

        dense = dense.with_added_examples(encoder, new_texts, new_labels, label_space.size)
        lexical = LexicalRetrieverAdapter.build(
            merged_texts, merged_labels, label_space, artifacts.config.retrieval
        )
        log.info("examples: %d added (corpus now %d items)", len(new_items), len(merged_items))
        updated_corpus = merged_items
    else:
        updated_corpus = list(corpus) if corpus is not None else None

    new_artifacts = replace(artifacts, label_space=label_space, dense=dense, lexical=lexical)
    return new_artifacts, updated_corpus
