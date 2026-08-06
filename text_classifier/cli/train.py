#!/usr/bin/env python
r"""Train the classifier.

Usage:
    text-classifier-train \
        --items items.csv \           # columns: text,label
        --classes classes.csv \       # columns: key,description
        --out model_dir/ \
        [--encoder-kind tfidf] [--per-fold-encoder] [--target-precision 0.95] [--folds 5] \
        [--encoder-epochs 10] [--encoder-epoch-holdout 0.1] [--encoder-patience 3] \
        [--val-items val.csv] [--test-items test.csv]

`label` in items.csv must match a `key` in classes.csv. Writes a portable model
directory plus `evaluation.json` and `model_card.md` summarizing held-out
performance.

Bring your own split: `--val-items` calibrates + tunes thresholds on an external
validation set, `--test-items` evaluates on an external test set (each optional,
same schema as `--items`, must be disjoint from it). With both, every internal
fold trains the fusion model, so `--folds 2` suffices. This is the supported
path for a temporal split — calibrate on a later slice, evaluate on a later one
still — while keeping the persisted evaluation evidence.

Multi-epoch encoder fine-tuning picks its own best epoch: with
`--encoder-epochs 20`, a stratified `--encoder-epoch-holdout` slice of the
fine-tuning items is withheld from the gradient updates and re-scored after every
epoch, and the best-scoring epoch is the one saved (per-epoch table:
`<out>/encoder/encoder_training.json`). `--encoder-patience` stops early once the
metric stalls.

Every field of `PipelineConfig` (fusion kind + xgb_params, calibration kind,
bm25_token_kwargs, encoder params, ...) is reachable via `--config config.json`
without writing Python; a trained model dir's `meta.json` `config` block is
directly usable as one. Precedence is defaults < --config < explicit flags.
Use `--dump-config` to print the effective config and exit.
"""

from __future__ import annotations

import argparse
import json
import logging

from .. import TrainingPipeline
from ..config import REUSE_QUERY_EMBEDDINGS_MODES
from ..domain import ENCODER_SELECTION_METRICS
from ..infrastructure.registry import encoder_spec
from ._common import (
    add_config_args,
    add_logging_arg,
    configure_logging,
    load_pipeline_config,
    read_items,
    read_label_space,
)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--items", default=None, help="required unless --dump-config")
    p.add_argument("--classes", default=None, help="required unless --dump-config")
    p.add_argument("--out", default=None, help="required unless --dump-config")
    p.add_argument(
        "--encoder-kind",
        default=None,
        help="encoder backend (registry key): 'sentence-transformers' "
        "(default), 'tfidf' (torch-free, air-gapped), or 'hashing' "
        "(dependency-free baseline / smoke test)",
    )
    p.add_argument(
        "--encoder",
        default=None,
        help="model name/path for the sentence-transformers encoder; "
        "ignored by corpus-fitted encoders such as tfidf "
        "(default: sentence-transformers/all-MiniLM-L6-v2)",
    )
    p.add_argument(
        "--encoder-epochs",
        type=int,
        default=None,
        help="fine-tuning epochs for the sentence-transformers encoder "
        "(default: 1). Only used on the fine-tuning path (--per-fold-encoder, or "
        "a corpus-fitted encoder kind); >1 enables best-epoch selection, see "
        "--encoder-epoch-holdout",
    )
    p.add_argument(
        "--encoder-epoch-holdout",
        type=float,
        default=None,
        help="fraction of the fine-tuning items held out of the gradient updates "
        "and re-scored after every epoch, so the *best* epoch is the one kept "
        "instead of the last (default: 0.1). 0 disables selection: every item "
        "trains and the final epoch wins. Ignored when --encoder-epochs is 1",
    )
    p.add_argument(
        "--encoder-select-metric",
        default=None,
        choices=list(ENCODER_SELECTION_METRICS),
        help="held-out metric the best epoch is chosen by (default: desc_acc@1, "
        "nearest class description is the true class). 'knn_acc@1' scores nearest "
        "labeled example instead — closer to the d_knn_* signals, but it re-encodes "
        "the fine-tuning pool every epoch",
    )
    p.add_argument(
        "--encoder-patience",
        type=int,
        default=None,
        help="stop fine-tuning after this many consecutive epochs without an "
        "improvement in --encoder-select-metric (default: 0 = run every epoch)",
    )
    p.add_argument(
        "--reuse-query-embeddings",
        default=None,
        choices=list(REUSE_QUERY_EMBEDDINGS_MODES),
        help="whether training may reuse the once-per-run document embeddings as "
        "the query embeddings for held-out items instead of encoding them twice "
        "(default: auto -- reuse when the encoder encodes both roles identically, "
        "which is the case unless a query/document prompt is configured). "
        "'never' always re-encodes; 'always' forces reuse even against a detected "
        "asymmetry. No effect with --per-fold-encoder, where there is no shared "
        "encoder and thus nothing cached to reuse.",
    )
    p.add_argument(
        "--folds",
        type=int,
        default=None,
        help="cross-validation folds for out-of-fold feature generation (default: 5). "
        "With both --val-items and --test-items, --folds 1 selects leave-one-out "
        "featurization (each training item scored against every other, itself masked "
        "out) instead of a k-fold split.",
    )
    p.add_argument(
        "--target-precision",
        type=float,
        default=None,
        help="target accuracy on accepted items; the threshold is tuned "
        "for max coverage subject to this (default: 0.95)",
    )
    p.add_argument("--candidate-top-n", type=int, default=None, help="default: 10")
    p.add_argument("--k-neighbors", type=int, default=None, help="default: 20")
    p.add_argument(
        "--drop-features",
        default=None,
        help="comma-separated feature columns to withhold from the fusion model "
        "(they are still computed, just not fitted on). Use to retrain without a "
        "column and compare -- the retrain-based counterpart to "
        "text-classifier-importance's masking ablation. Default: drop nothing.",
    )
    p.add_argument(
        "--bm25-stop-words",
        default=None,
        help="stop_words value for BM25's tokenizer (any value sklearn's "
        "CountVectorizer accepts, e.g. 'english'); 'none' explicitly disables "
        "stopword filtering. Default: no stopword removal (language-neutral) "
        "-- pass 'english' to filter English stopwords.",
    )
    p.add_argument(
        "--bm25-n-jobs",
        type=int,
        default=None,
        help="threads for BM25's per-chunk kNN mat-mul (scipy sparse @ releases "
        "the GIL, so this scales across cores on large example pools). Default: "
        "1 (single-threaded); -1 uses all CPU cores.",
    )
    p.add_argument(
        "--per-fold-encoder",
        action="store_true",
        default=None,
        help="rigorous (expensive): fine-tune a fresh encoder per fold",
    )
    p.add_argument(
        "--val-items",
        default=None,
        help="optional external validation set (CSV, same --text-col/--label-col "
        "schema as --items): calibrate and tune abstention thresholds on it instead "
        "of an internal fold. Must be disjoint from --items. Enables a temporal "
        "split (calibrate on a later slice) and frees the calibration fold for "
        "fusion training.",
    )
    p.add_argument(
        "--test-items",
        default=None,
        help="optional external test set (CSV, same schema as --items): the "
        "held-out evaluation (evaluation.json/model_card.md) runs on it instead "
        "of an internal fold. Must be disjoint from --items. With both --val-items "
        "and --test-items, every internal fold trains the fusion model (--folds may "
        "then be 2, or 1 for leave-one-out featurization).",
    )
    p.add_argument(
        "--store-corpus",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="persist the raw training corpus (text+label) into the model dir "
        "as corpus.jsonl.gz, so `text-classifier-update` can later add examples "
        "without --base-items (default: on; --no-store-corpus opts out for "
        "privacy/size)",
    )
    p.add_argument("--text-col", default="text", help="items.csv text column")
    p.add_argument("--label-col", default="label", help="items.csv label column")
    p.add_argument("--key-col", default="key", help="classes.csv key column")
    p.add_argument("--desc-col", default="description", help="classes.csv description column")
    add_config_args(p)
    add_logging_arg(p)
    args = p.parse_args()
    configure_logging(args.log_level)

    cfg = load_pipeline_config(args.config)

    # Explicit flags win over the config file; a flag left at its argparse
    # default of `None` means "not set", so the file (or the built-in
    # PipelineConfig default) is left untouched.
    if args.encoder_kind is not None:
        cfg.encoder.kind = args.encoder_kind
    if args.encoder is not None:
        cfg.encoder.model_name_or_path = args.encoder
    if args.encoder_epochs is not None:
        cfg.encoder.train_epochs = args.encoder_epochs
    if args.encoder_epoch_holdout is not None:
        cfg.encoder.train_holdout_ratio = args.encoder_epoch_holdout
    if args.encoder_select_metric is not None:
        cfg.encoder.train_select_metric = args.encoder_select_metric
    if args.encoder_patience is not None:
        cfg.encoder.train_early_stopping_patience = args.encoder_patience
    if args.reuse_query_embeddings is not None:
        cfg.encoder.reuse_query_embeddings = args.reuse_query_embeddings
    if args.folds is not None:
        cfg.training.n_folds = args.folds
    if args.target_precision is not None:
        cfg.training.target_precision = args.target_precision
    if args.candidate_top_n is not None:
        cfg.candidate_top_n = args.candidate_top_n
    if args.k_neighbors is not None:
        cfg.retrieval.k_neighbors = args.k_neighbors
    if args.drop_features is not None:
        cfg.fusion.drop_features = [n.strip() for n in args.drop_features.split(",") if n.strip()]
    if args.bm25_n_jobs is not None:
        cfg.retrieval.bm25_n_jobs = args.bm25_n_jobs
    if args.bm25_stop_words is not None:
        if args.bm25_stop_words.lower() == "none":
            cfg.retrieval.bm25_token_kwargs.pop("stop_words", None)
        else:
            cfg.retrieval.bm25_token_kwargs["stop_words"] = args.bm25_stop_words
    if args.per_fold_encoder:
        cfg.training.use_per_fold_encoder = True
    if args.store_corpus is not None:
        cfg.training.store_corpus = args.store_corpus

    # External splits relax the fold floor to >= 2 (each retires a fold role), so
    # validate with the same external flags run() will use.
    try:
        cfg.validate(
            external_val=args.val_items is not None, external_test=args.test_items is not None
        )
    except ValueError as exc:
        p.error(str(exc))

    if args.dump_config:
        print(json.dumps(cfg.to_dict(), indent=2))
        return

    missing = [
        name
        for name, value in [
            ("--items", args.items),
            ("--classes", args.classes),
            ("--out", args.out),
        ]
        if value is None
    ]
    if missing:
        p.error(f"the following arguments are required: {', '.join(missing)}")

    # Validate the encoder kind up front: an unknown backend fails fast with a
    # clear message listing the registered kinds, not a deep traceback.
    try:
        enc_spec = encoder_spec(cfg.encoder.kind)
    except ValueError as exc:
        p.error(str(exc))

    label_space = read_label_space(args.classes, args.key_col, args.desc_col)
    items = read_items(args.items, args.text_col, args.label_col)
    val_items = (
        read_items(args.val_items, args.text_col, args.label_col) if args.val_items else None
    )
    test_items = (
        read_items(args.test_items, args.text_col, args.label_col) if args.test_items else None
    )

    if enc_spec.corpus_dependent:
        logging.info(
            "encoder kind %r is corpus-fitted; --encoder=%r is ignored",
            cfg.encoder.kind,
            cfg.encoder.model_name_or_path,
        )

    _, report = TrainingPipeline(cfg).run(
        items, label_space, output_dir=args.out, val_items=val_items, test_items=test_items
    )
    print("\n=== coverage report (test fold) ===")
    print(f"candidate recall      : {report.candidate_recall:.4f}")
    print(f"coverage              : {report.coverage:.4f}")
    print(f"accuracy on accepted  : {report.accuracy_on_accepted:.4f}")
    print(f"accuracy if no abstain: {report.accuracy_if_no_abstain:.4f}")
    print(f"\nwrote model + evaluation.json + model_card.md to {args.out}")


if __name__ == "__main__":
    main()
