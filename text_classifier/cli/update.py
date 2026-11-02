#!/usr/bin/env python
r"""Add classes and/or labeled examples to a deployed model, without retraining.

Usage:
    text-classifier-update \
        --model model_dir/ \
        --out updated_model_dir/ \        # or --in-place to overwrite --model
        [--classes classes.csv] \         # full class list: key,description
        [--items new_items.csv] \         # new labeled examples: text,label
        [--base-items original_items.csv] \  # only if model_dir has no corpus.jsonl.gz
        [--tune-with fresh_labeled.csv] [--target-precision 0.95]

Rebuilds only the cheap, data-dependent parts of the model — the retrieval
indices, class prototypes, and description embeddings — while reusing the
trained fusion model and calibrator verbatim (the fusion model is
class-agnostic: every feature is a per-candidate retrieval signal, not a
per-class weight). No k-fold pass over the corpus, no fusion refit.

``--classes`` is the *full* set of classes this run cares about (same schema
as ``text-classifier-train --classes``): every key already in the model must
be present — update never removes or reorders a class, so a file that drops
one is rejected. A key not yet in the model is a new class (appended,
description-only until examples arrive); an existing key with different
description text gets that description re-embedded.

``--items`` adds labeled examples for new or existing classes. This needs the
original training corpus to refit the example BM25 index (its IDF is
corpus-global, so it cannot be updated incrementally): by default this comes
from ``corpus.jsonl.gz`` in ``--model`` (written by ``text-classifier-train``
unless ``--no-store-corpus`` was passed); for a dir that predates it, supply
the original items via ``--base-items``.

At least one of ``--classes``/``--items`` is required. Without ``--tune-with``
the abstention thresholds are untouched (new classes fall back to the global
threshold) and the model dir's evaluation.json/model_card.md are marked
stale — run ``text-classifier-tune`` on fresh labeled data afterward. With
``--tune-with``, thresholds are re-tuned in the same run and fresh evaluation
artifacts are written, including candidate recall for the newly added
classes.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
from dataclasses import replace

from .. import InferencePipeline
from .._version import __version__
from ..application.evaluation import build_manifest, write_evaluation_artifacts
from ..application.tuning import retune
from ..application.updating import update
from ..infrastructure import ArtifactRepository
from ._common import add_logging_arg, configure_logging, read_items, read_label_space


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _copy_and_mark_stale(source_dir: str, target_dir: str, note: str) -> None:
    """Carry the prior evaluation.json/model_card.md forward (unchanged
    content, marked stale) when this update didn't re-evaluate (no
    ``--tune-with``). A dir with no persisted evaluation to begin with is left
    alone -- there's nothing to carry forward or mark stale."""
    src_eval = os.path.join(source_dir, "evaluation.json")
    if not os.path.isfile(src_eval):
        return
    with open(src_eval) as fh:
        evaluation = json.load(fh)
    evaluation["stale"] = True
    evaluation["stale_reason"] = note
    with open(os.path.join(target_dir, "evaluation.json"), "w") as fh:
        json.dump(evaluation, fh, indent=2)

    src_card = os.path.join(source_dir, "model_card.md")
    card = ""
    if os.path.isfile(src_card):
        with open(src_card) as fh:
            card = fh.read()
    with open(os.path.join(target_dir, "model_card.md"), "w") as fh:
        fh.write(f"> **Stale after update.** {note}\n\n{card}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True, help="trained model directory")
    p.add_argument("--out", default=None, help="write the updated model to this new directory")
    p.add_argument("--in-place", action="store_true", help="overwrite --model instead of --out")
    p.add_argument(
        "--classes",
        default=None,
        help="full classes CSV (key,description); every class already in the "
        "model must be present (update never removes/reorders classes). A new "
        "key is appended; an existing key with changed description text is "
        "re-embedded.",
    )
    p.add_argument("--items", default=None, help="new labeled examples CSV (text,label) to add")
    p.add_argument(
        "--base-items",
        default=None,
        help="original training items CSV; only needed with --items when --model "
        "has no persisted corpus.jsonl.gz (predates --store-corpus, or opted out)",
    )
    p.add_argument(
        "--tune-with",
        default=None,
        help="fresh labeled CSV: re-tune calibration + abstention thresholds in "
        "the same run (see text-classifier-tune) and report candidate recall "
        "for the newly added classes. Without this, thresholds are left as-is "
        "and the persisted evaluation is marked stale.",
    )
    p.add_argument("--target-precision", type=float, default=None)
    p.add_argument("--per-class-min-support", type=int, default=None)
    p.add_argument("--text-col", default="text")
    p.add_argument("--label-col", default="label")
    p.add_argument("--key-col", default="key")
    p.add_argument("--desc-col", default="description")
    add_logging_arg(p)
    args = p.parse_args()
    configure_logging(args.log_level)

    if bool(args.out) == bool(args.in_place):
        p.error("exactly one of --out or --in-place is required")
    out_dir = args.out if args.out else args.model

    if not args.classes and not args.items:
        p.error("at least one of --classes or --items is required")

    pipeline = InferencePipeline.from_directory(args.model)
    artifacts = pipeline.artifacts
    repo = ArtifactRepository()
    # Read before any writes: for --in-place, --model and out_dir are the same
    # directory, and `save()` below rewrites meta.json from scratch.
    prior_meta = repo.read_meta(args.model)

    corpus = repo.load_corpus(args.model)
    if args.base_items:
        if corpus is not None:
            p.error(
                "--base-items was given but --model already has a persisted "
                "corpus.jsonl.gz; omit --base-items"
            )
        corpus = read_items(args.base_items, args.text_col, args.label_col)

    classes = None
    if args.classes:
        ls = read_label_space(args.classes, args.key_col, args.desc_col)
        classes = list(ls.definitions)
    new_items = read_items(args.items, args.text_col, args.label_col) if args.items else []

    try:
        new_artifacts, updated_corpus = update(
            artifacts, corpus, classes=classes, new_items=new_items
        )
    except ValueError as exc:
        raise SystemExit(f"error: {exc}")

    n_new_classes = new_artifacts.label_space.size - artifacts.label_space.size
    print(
        f"classes: {artifacts.label_space.size} -> {new_artifacts.label_space.size} "
        f"({n_new_classes} new); examples added: {len(new_items)}"
    )

    evaluation = None
    if args.tune_with:
        tune_items = read_items(args.tune_with, args.text_col, args.label_col)
        target_precision = (
            args.target_precision
            if args.target_precision is not None
            else new_artifacts.config.training.target_precision
        )
        per_class_min_support = (
            args.per_class_min_support
            if args.per_class_min_support is not None
            else new_artifacts.config.training.per_class_min_support
        )
        try:
            abstention, calibrator, evaluation = retune(
                new_artifacts,
                tune_items,
                new_artifacts.label_space,
                target_precision,
                per_class_min_support,
            )
        except ValueError as exc:
            raise SystemExit(f"error: {exc}")
        new_artifacts = replace(new_artifacts, calibrator=calibrator, abstention=abstention)
        new_artifacts.config.training.target_precision = target_precision

        o = evaluation["overall"]
        print(
            f"\nre-tuned on {o['n_items']} fresh labeled item(s): "
            f"coverage={_pct(o['coverage'])} accuracy_on_accepted={_pct(o['accuracy_on_accepted'])}"
        )
        if n_new_classes:
            new_keys = set(new_artifacts.label_space.keys[-n_new_classes:])
            new_class_rows = [r for r in evaluation["per_class"] if r["key"] in new_keys]
            if new_class_rows:
                print("\nnew-class candidate recall on the tune set:")
                for r in new_class_rows:
                    print(
                        f"  {r['key']:>20}  support={r['support']:<4} "
                        f"coverage={_pct(r['coverage'])}"
                    )
    else:
        print(
            "\nNOTE: abstention thresholds were left unchanged; new classes fall back "
            "to the global threshold. Run text-classifier-tune on fresh labeled data "
            "(or pass --tune-with here) to re-anchor them -- otherwise the persisted "
            "evaluation.json/model_card.md are now stale."
        )

    repo.save(new_artifacts, out_dir)
    if updated_corpus is not None:
        repo.save_corpus(out_dir, updated_corpus)

    provenance_entry = {
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "n_new_classes": n_new_classes,
        "n_new_items": len(new_items),
        "package_version": __version__,
    }
    repo.apply_update_provenance(out_dir, prior_meta, provenance_entry)

    if evaluation is not None:
        manifest = build_manifest(
            n_training_items=len(updated_corpus) if updated_corpus is not None else 0,
            n_classes=new_artifacts.label_space.size,
            config=new_artifacts.config,
            n_evaluated=len(tune_items),
        )
        write_evaluation_artifacts(out_dir, evaluation, manifest)
    else:
        note = (
            f"headline metrics predate this update ({n_new_classes} new class(es), "
            f"{len(new_items)} new example(s)); re-run text-classifier-eval or "
            "text-classifier-tune on fresh labeled data to refresh them."
        )
        _copy_and_mark_stale(args.model, out_dir, note)

    print(f"\nwrote updated model to {out_dir}")


if __name__ == "__main__":
    main()
