"""Cross-encoder rerank signal (T33).

Three things live here, in dependency order:

1. The ``PairwiseReranker`` backends. ``TokenOverlapReranker`` needs no model and
   no network, so the whole path is exercised offline and in CI (the reranker
   analogue of ``HashingEncoder``). ``SentenceTransformersCrossEncoder`` is the
   real one (T33 phase 3), with ``train_cross_encoder`` to fine-tune it on lists
   mined by ``mine_training_lists`` from round-one shortlists.
2. Evidence rendering — turning a ``ClassDefinition`` plus an ``EvidenceSpec``
   list into the document text a pair is scored against, with per-query
   selection among a view's entries.
3. ``CrossEncoderSignalProvider`` — the *second-stage* ``SignalProvider``
   (``needs_candidates = True``) that decides which pairs are worth scoring,
   calls the reranker once per document, and shapes the result into ``(b, C)``
   matrices the assembler derives columns from exactly as it does for the five
   built-in signals.

The split matters: (1) answers "how well do these two texts go together" and
nothing else, so a stock cross-encoder, a fine-tuned one and an instructed LLM
judge all drop into the same slot; (3) owns every decision about *which* pairs,
how documents are composed, and what the columns are called.
"""

from __future__ import annotations

import json
import os
import logging
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..config import CrossEncoderConfig, CrossEncoderDocument, EvidenceSpec
from ..domain import (
    ArrayOps,
    ClassDefinition,
    PairwiseReranker,
    SignalContext,
    SignalMatrix,
    SignalProvider,
)
from .array_ops import NumpyArrayOps
from .device import resolve_device
from .signals import _argmax_or_missing, _topn_mask


def _tokens(text: str) -> frozenset:
    return frozenset(str(text).lower().split())


class TokenOverlapReranker(PairwiseReranker):
    """Offline, dependency-free stand-in for a real cross-encoder.

    Scores a pair by the smoothed log-odds of its token Jaccard overlap:
    positive when the two texts share most of their vocabulary, negative when
    they share little, unbounded on both sides — the same *shape* a real
    cross-encoder's logits have, which is what the fusion model and the
    downstream calibration are written against.

    Deterministic by construction (exact set arithmetic, no hashing and no
    learned weights), so it is byte-identical across processes, versions and
    platforms without pinning anything. It is emphatically **not** a semantic
    reranker — it cannot see that "car" and "automobile" are related, and it
    cannot interpret negation any better than BM25 can. Its job is to prove the
    plumbing, exactly as ``HashingEncoder`` does for the encoder path.
    """

    def score(self, pairs: Sequence[Tuple[str, str]]) -> np.ndarray:
        out = np.empty(len(pairs), dtype=np.float32)
        for i, (query, document) in enumerate(pairs):
            q, d = _tokens(query), _tokens(document)
            shared = len(q & d)
            # +0.5 smoothing keeps both ends finite: no overlap is a strongly
            # negative logit rather than -inf, which would be indistinguishable
            # from "not scored" once it reaches the feature matrix.
            out[i] = np.log((shared + 0.5) / (len(q | d) - shared + 0.5))
        return out

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "reranker.json"), "w") as fh:
            json.dump({"kind": "token-overlap"}, fh)

    @classmethod
    def load(cls, path: str, **kwargs) -> "TokenOverlapReranker":
        return cls()


logger = logging.getLogger(__name__)


class SchemaOnlyReranker(PairwiseReranker):
    """Placeholder backend for a ``CrossEncoderSignalProvider`` that exists only
    to report ``column_names()`` (the load-time schema check). It holds no
    model, so building one is free and touches no files; scoring or saving it
    is a bug and raises."""

    def score(self, pairs: Sequence[Tuple[str, str]]) -> np.ndarray:
        raise RuntimeError("SchemaOnlyReranker cannot score: it exists only for column_names()")

    def save(self, directory: str) -> None:
        raise RuntimeError("SchemaOnlyReranker cannot be saved: it holds no model")

    @classmethod
    def load(cls, path: str, **kwargs) -> "SchemaOnlyReranker":
        raise RuntimeError("SchemaOnlyReranker cannot be loaded: it holds no model")


# ------------------------------------------------------ real cross-encoder
_CE_MODULES = ("sentence_transformers.cross_encoder",)


def _require_cross_encoder() -> Any:
    """``sentence_transformers.cross_encoder.CrossEncoder``, or the same actionable
    install error the encoder raises (the extra is optional, T63)."""
    from .encoder import _import_first, _require_sentence_transformers

    _require_sentence_transformers()
    return _import_first(_CE_MODULES, "CrossEncoder")


def _identity_activation() -> Any:
    import torch

    return torch.nn.Identity()


class SentenceTransformersCrossEncoder(PairwiseReranker):
    """A sentence-transformers ``CrossEncoder`` behind the ``PairwiseReranker`` port.

    ``score`` returns **raw logits**, as the port requires: the fusion model learns
    the scale and the isotonic stage calibrates. That is forced explicitly with an
    identity activation, because ``CrossEncoder.predict``'s default differs by
    version -- 5.6+ applies a sigmoid to single-label models, 5.2 does not -- so
    leaving it to the default would make the same model's features depend on the
    installed library.

    Persistence writes the model itself (safetensors + config + tokenizer) into
    the model directory, and ``load`` reads only from there, so a deployed model
    works on an air-gapped host and loading it never executes embedded code.
    """

    def __init__(self, model: Any, batch_size: int = 32):
        labels = getattr(getattr(model, "config", None), "num_labels", 1)
        if labels != 1:
            raise ValueError(
                f"cross-encoder model has num_labels={labels}; the rerank signal needs a "
                "single relevance score per pair (num_labels=1), e.g. an ms-marco- or "
                "bge-reranker-style model"
            )
        self._model = model
        self._batch_size = batch_size
        # Per-epoch record of a fine-tune that selected its best epoch (see
        # ``train_cross_encoder``). Evidence, not state: written beside the
        # weights by ``save``; empty for a pretrained or reloaded model.
        self.training_history: Dict[str, Any] = {}

    TRAINING_HISTORY_NAME = "cross_encoder_training.json"

    @property
    def model(self) -> Any:
        return self._model

    @classmethod
    def from_config(cls, config: CrossEncoderConfig) -> "SentenceTransformersCrossEncoder":
        return cls._load(config.model_name, config)

    @classmethod
    def _load(cls, path: str, config: CrossEncoderConfig) -> "SentenceTransformersCrossEncoder":
        CrossEncoder = _require_cross_encoder()
        device = resolve_device(config.device, mps_ok=True)
        logger.info("loading cross-encoder %s on device=%s", path, device)
        return cls(CrossEncoder(path, device=device, **config.params), config.batch_size)

    def score(self, pairs: Sequence[Tuple[str, str]]) -> np.ndarray:
        if not pairs:
            return np.empty(0, dtype=np.float32)
        out = self._model.predict(
            [list(p) for p in pairs],
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            activation_fn=_identity_activation(),
        )
        return np.asarray(out, dtype=np.float32).reshape(-1)

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        self._model.save(directory)
        with open(os.path.join(directory, "reranker.json"), "w") as fh:
            json.dump({"kind": "cross-encoder"}, fh)
        if self.training_history:
            with open(os.path.join(directory, self.TRAINING_HISTORY_NAME), "w") as fh:
                json.dump(self.training_history, fh, indent=2)

    @classmethod
    def load(cls, path: str, **kwargs) -> "SentenceTransformersCrossEncoder":
        config = kwargs.get("config") or CrossEncoderConfig(kind="cross-encoder", model_name=path)
        return cls._load(path, config)


# ------------------------------------------------------ fine-tuning (phase 3)
# Built-in round-one nodes -> the raw column that ranks a shortlist by them.
# Used only to order hard negatives "hardest first"; any other prescore node
# falls back to a seeded shuffle, which is still drawn from the shortlist.
_PRESCORE_COLUMN = {
    "dense.desc": "d_desc_sim",
    "dense.proto": "d_proto_sim",
    "dense.knn": "d_knn_sum",
    "bm25.desc": "b_desc_sim",
    "bm25.knn": "b_knn_sum",
}


def prescore_column(node: str) -> Optional[str]:
    """The feature column that ranks candidates by round-one node ``node``."""
    return _PRESCORE_COLUMN.get(node)


def mine_training_lists(
    shortlist: pd.DataFrame,
    true_class: np.ndarray,
    n_negatives: int,
    order_column: Optional[str] = None,
    seed: int = 0,
) -> List[Tuple[int, int, List[int]]]:
    """``(item, positive class, negative classes)`` per item, from its shortlist.

    ``shortlist`` is a round-one feature frame (``item_id``/``candidate`` rows) for
    the items being trained on, computed leave-one-out so no item retrieved
    itself. The positive is the item's true class -- always, even when round one
    missed it, since the reranker should learn what the right answer looks like.
    Negatives are the other shortlisted classes: hardest first by
    ``order_column`` when given (the candidates the prescore ranked highest are
    the ones the reranker will have to beat), else a seeded shuffle; capped at
    ``n_negatives``. An item whose shortlist holds nothing but its true class has
    no negatives and is skipped."""
    rng = np.random.default_rng(seed)
    out: List[Tuple[int, int, List[int]]] = []
    if shortlist.empty:
        return out
    for item, group in shortlist.groupby("item_id", sort=True):
        item_i = int(item)
        positive = int(true_class[item_i])
        negs = group[group["candidate"].to_numpy() != positive]
        if negs.empty:
            continue
        if order_column is not None and order_column in negs.columns:
            scores = negs[order_column].to_numpy(dtype=np.float64)
            scores = np.where(np.isnan(scores), -np.inf, scores)
            order = np.argsort(-scores, kind="stable")
        else:
            order = rng.permutation(len(negs))
        cands = negs["candidate"].to_numpy()[order][:n_negatives]
        out.append((item_i, positive, [int(c) for c in cands]))
    return out


def training_examples(
    lists: Sequence[Tuple[int, int, List[int]]],
    texts: Sequence[str],
    definitions: Sequence[ClassDefinition],
    document: CrossEncoderDocument,
) -> List[Tuple[str, str, List[str]]]:
    """See ``training_examples_with_classes``; the examples alone."""
    return training_examples_with_classes(lists, texts, definitions, document)[0]


def training_examples_with_classes(
    lists: Sequence[Tuple[int, int, List[int]]],
    texts: Sequence[str],
    definitions: Sequence[ClassDefinition],
    document: CrossEncoderDocument,
) -> Tuple[List[Tuple[str, str, List[str]]], List[int]]:
    """Render mined lists into ``(query, positive doc, negative docs)`` text.

    Uses the same evidence rendering as scoring -- including query-adaptive
    selection -- so the model is trained on exactly the documents it will see. A
    class with no text for ``document`` renders ``""``; such a negative is dropped,
    and an item whose positive renders empty is skipped (nothing to learn).

    Also returns each kept example's positive class, aligned with the examples,
    so best-epoch selection can hold out a class-stratified slice."""
    out: List[Tuple[str, str, List[str]]] = []
    classes: List[int] = []
    for item, positive, negatives in lists:
        query = texts[item]
        q_tokens = _tokens(query)
        pos_doc = _render(definitions[positive], document, q_tokens)
        if not pos_doc:
            continue
        neg_docs = [
            d for d in (_render(definitions[c], document, q_tokens) for c in negatives) if d
        ]
        if neg_docs:
            out.append((query, pos_doc, neg_docs))
            classes.append(int(positive))
    return out, classes


def _train_dataset_and_loss(
    model: Any, examples: Sequence[Tuple[str, str, List[str]]], loss_name: str
) -> Tuple[Any, Any]:
    from datasets import Dataset

    from .encoder import _import_first

    losses = "sentence_transformers.cross_encoder.losses"
    if loss_name == "binary_cross_entropy":
        query, doc, label = [], [], []
        for q, pos, negs in examples:
            query.extend([q] * (1 + len(negs)))
            doc.extend([pos, *negs])
            label.extend([1.0] + [0.0] * len(negs))
        import torch

        # Each item contributes one positive and several negatives; weight the
        # positives up so the objective is not dominated by "say no".
        n_pos = sum(1 for v in label if v == 1.0)
        pos_weight = torch.tensor(max((len(label) - n_pos) / max(n_pos, 1), 1.0))
        loss = _import_first((losses,), "BinaryCrossEntropyLoss")(model, pos_weight=pos_weight)
        return Dataset.from_dict({"query": query, "doc": doc, "label": label}), loss
    cls_name = {"lambda": "LambdaLoss", "listnet": "ListNetLoss"}[loss_name]
    dataset = Dataset.from_dict(
        {
            "query": [q for q, _, _ in examples],
            "docs": [[pos, *negs] for _, pos, negs in examples],
            "labels": [[1] + [0] * len(negs) for _, _, negs in examples],
        }
    )
    return dataset, _import_first((losses,), cls_name)(model)


def train_cross_encoder(
    reranker: SentenceTransformersCrossEncoder,
    examples: Sequence[Tuple[str, str, List[str]]],
    config: CrossEncoderConfig,
    classes: Optional[Sequence[int]] = None,
) -> SentenceTransformersCrossEncoder:
    """Fine-tune ``reranker``'s model in place on ``examples`` and return it.

    The caller is responsible for *what* it trains on: in ``TrainingPipeline``
    that is one fold's training items only, on a fresh copy of ``model_name``,
    so no held-out item is ever scored by a model that saw its label. Runs a
    ``CrossEncoderTrainer`` (sentence-transformers >= 4.0) with nothing written
    outside a temporary directory and nothing reported anywhere.

    **Which epoch you get back.** With ``train_epochs == 1`` (the default) or
    ``train_holdout_ratio == 0``, the state after the last epoch. Otherwise a
    ``train_holdout_ratio`` slice of the examples -- stratified by ``classes``
    (each example's positive class) when given -- is withheld from the gradient
    and scored after every epoch by a ``CrossEncoderRerankingEvaluator`` (is the
    positive ranked first among the item's candidates?); the Trainer reloads the
    best epoch on ``train_select_metric`` and can stop early. The per-epoch table
    is kept as ``reranker.training_history`` and saved beside the weights. The
    holdout comes out of the caller's examples, i.e. out of one fold's training
    rows, so selection stays in-fold."""
    from transformers import EarlyStoppingCallback

    from .encoder import (
        MIN_EPOCH_HOLDOUT,
        _device_notes,
        _import_first,
        _quiet_trainer,
        _stratified_holdout,
        _warmup_kwargs,
        _with_save_safetensors,
    )

    if not examples:
        logger.warning("cross-encoder fine-tuning skipped: no training lists could be mined")
        return reranker
    try:
        trainer_cls = _import_first(_CE_MODULES, "CrossEncoderTrainer")
        args_cls = _import_first(_CE_MODULES, "CrossEncoderTrainingArguments")
    except ImportError as exc:
        import sentence_transformers

        raise ImportError(
            "retrieval.cross_encoder.train needs sentence-transformers>=4.0 (its "
            f"CrossEncoderTrainer); found {sentence_transformers.__version__}. Scoring "
            "with a pretrained cross-encoder works on older versions."
        ) from exc
    _device_notes(config)
    examples = list(examples)
    groups = np.asarray(classes if classes is not None else np.zeros(len(examples)), dtype=np.int64)
    ratio = config.train_holdout_ratio if config.train_epochs > 1 else 0.0
    # Fixed seed: a rerun holds out the same lists (the determinism invariant).
    fit_idx, hold_idx = _stratified_holdout(groups, ratio, 0)
    if 0 < len(hold_idx) < MIN_EPOCH_HOLDOUT:
        logger.warning(
            "cross-encoder best-epoch selection disabled: a %d-item holdout (< %d) is too "
            "small to rank epochs by; keeping the last epoch and training on everything",
            len(hold_idx),
            MIN_EPOCH_HOLDOUT,
        )
        fit_idx, hold_idx = np.arange(len(examples)), np.empty(0, dtype=np.intp)
    selecting = len(hold_idx) > 0
    holdout = [examples[int(i)] for i in hold_idx]
    examples = [examples[int(i)] for i in fit_idx]
    dataset, loss = _train_dataset_and_loss(reranker.model, examples, config.train_loss)
    logger.info(
        "fine-tuning cross-encoder on %d item(s) (%d pair(s)), loss=%s, %d epoch(s)",
        len(examples),
        sum(1 + len(n) for _, _, n in examples),
        config.train_loss,
        config.train_epochs,
    )
    work_dir = tempfile.mkdtemp(prefix="tc-cross-encoder-train-")
    try:
        kwargs: Dict[str, Any] = {
            "output_dir": work_dir,
            "num_train_epochs": config.train_epochs,
            "per_device_train_batch_size": config.train_batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "report_to": "none",
            "disable_tqdm": True,
            "logging_strategy": "no",
            "save_strategy": "no",
            **_warmup_kwargs(config.warmup_ratio),
        }
        if config.device == "cpu":
            kwargs["use_cpu"] = True
        evaluator = None
        callbacks: List[Any] = []
        if selecting:
            evaluator, select_key = _holdout_evaluator(holdout, config)
            kwargs.update(
                {
                    "eval_strategy": "epoch",
                    "save_strategy": "epoch",
                    "load_best_model_at_end": True,
                    "metric_for_best_model": select_key,
                    "greater_is_better": True,
                    "save_total_limit": 1,
                    "save_only_model": True,  # weights only: no optimizer pickles
                }
            )
            if config.train_early_stopping_patience > 0:
                callbacks.append(
                    EarlyStoppingCallback(
                        early_stopping_patience=config.train_early_stopping_patience,
                        early_stopping_threshold=config.train_select_min_delta,
                    )
                )
        args = _with_save_safetensors(args_cls(**kwargs))
        trainer = trainer_cls(
            model=reranker.model,
            args=args,
            train_dataset=dataset,
            loss=loss,
            evaluator=evaluator,
            callbacks=callbacks,
        )
        _quiet_trainer(trainer)
        trainer.train()
        if selecting:
            reranker.training_history = _ce_training_history(
                trainer, select_key, config, n_fit=len(examples), n_holdout=len(holdout)
            )
            h = reranker.training_history
            logger.info(
                "cross-encoder: kept epoch %d of %d (%s=%.4f)%s",
                h["best_epoch"],
                h["epochs_run"],
                config.train_select_metric,
                h["epochs"][h["best_epoch"] - 1][config.train_select_metric]
                if h["best_epoch"]
                else float("nan"),
                ", stopped early" if h["early_stopped"] else "",
            )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    return reranker


def _holdout_evaluator(
    holdout: Sequence[Tuple[str, str, List[str]]], config: CrossEncoderConfig
) -> Tuple[Any, str]:
    """A ``CrossEncoderRerankingEvaluator`` over the held-out lists, and the
    Trainer metric name for ``train_select_metric``. ``at_k`` is the longest list,
    so MRR/NDCG are full-list, not truncated."""
    from .encoder import _import_first

    evaluator_cls = _import_first(
        ("sentence_transformers.cross_encoder.evaluation",), "CrossEncoderRerankingEvaluator"
    )
    k = max(1 + len(negs) for _, _, negs in holdout)
    samples = [{"query": q, "positive": [pos], "negative": list(negs)} for q, pos, negs in holdout]
    evaluator = evaluator_cls(
        samples,
        at_k=k,
        always_rerank_positives=True,
        name="ce_holdout",
        batch_size=config.batch_size,
        show_progress_bar=False,
        write_csv=False,
    )
    metric = config.train_select_metric
    key = "eval_ce_holdout_map" if metric == "map" else f"eval_ce_holdout_{metric}@{k}"
    return evaluator, key


def _ce_training_history(
    trainer: Any, select_key: str, config: CrossEncoderConfig, n_fit: int, n_holdout: int
) -> Dict[str, Any]:
    """JSON-clean per-epoch record, keyed by "mrr"/"ndcg"/"map". ``best_epoch`` is
    the epoch whose checkpoint the Trainer reloaded (ties go to the earlier one)."""
    evals = [
        e for e in trainer.state.log_history if any(k.startswith("eval_ce_holdout") for k in e)
    ]
    epochs: List[Dict[str, Any]] = []
    for number, entry in enumerate(evals, start=1):
        row: Dict[str, Any] = {"epoch": number}
        for key, value in entry.items():
            if key.startswith("eval_ce_holdout_"):
                row[key[len("eval_ce_holdout_") :].split("@")[0]] = float(value)
        epochs.append(row)
    best_epoch = 0
    checkpoint = trainer.state.best_model_checkpoint
    if checkpoint:
        try:
            step = int(os.path.basename(os.path.normpath(checkpoint)).rsplit("-", 1)[-1])
            best_epoch = next((n for n, e in enumerate(evals, 1) if e.get("step") == step), 0)
        except ValueError:
            best_epoch = 0
    return {
        "select_metric": config.train_select_metric,
        "select_min_delta": config.train_select_min_delta,
        "early_stopping_patience": config.train_early_stopping_patience,
        "n_fit_items": n_fit,
        "n_holdout_items": n_holdout,
        "best_epoch": best_epoch,
        "epochs": epochs,
        "epochs_requested": config.train_epochs,
        "epochs_run": len(epochs),
        "early_stopped": len(epochs) < config.train_epochs,
    }


# --------------------------------------------------------------- evidence
# View name -> the ClassDefinition texts it offers. Tuple-valued views return
# several short strings (so `select` has something to choose between); scalar
# views return a single-entry tuple. Absent evidence is `()` — an *absent* view,
# never an empty string, so it can propagate as NaN rather than as a score
# against nothing (CLAUDE.md's NaN invariant, at the text layer).
_EVIDENCE = {
    "description": lambda d: (d.description,),
    "core": lambda d: (d.core_view(),),
    "examples": lambda d: d.examples,
    "inclusions": lambda d: d.inclusions,
    "exclusions": lambda d: d.exclusions,
    "siblings": lambda d: d.sibling_distinctions,
}


# One resolved evidence slot for one class. Either a finished string (the slot
# does not depend on the query) or the candidates to choose between at scoring
# time, pre-tokenized.
_Slot = Union[str, Tuple[EvidenceSpec, Tuple[Tuple[str, frozenset], ...]]]


def _plan_class(definition: ClassDefinition, doc: CrossEncoderDocument) -> List[_Slot]:
    """Resolve one class's slots for one document, independent of any query.

    Everything that does not vary per query is finished here — including the
    ``"first"``/``"all"`` modes, which never consult the query at all — and the
    entries of a ``"best"`` slot are tokenized once. This runs ``C`` times per
    chunk rather than once per scored *pair*, which is the difference between
    tokenizing a taxonomy's evidence a few dozen times and tokenizing it tens of
    thousands of times: the selection input depends only on the class, so doing
    it per pair was pure repeat work."""
    plan: List[_Slot] = []
    for spec in doc.evidence:
        entries = tuple(e for e in _EVIDENCE[spec.view](definition) if e)
        if not entries:
            plan.append("")  # absent view -> contributes nothing, never a bare label
        elif spec.select == "all":
            plan.append(spec.label + "; ".join(entries)[: spec.max_chars])
        elif spec.select == "first" or len(entries) == 1:
            plan.append(spec.label + entries[0][: spec.max_chars])
        else:  # "best" -- the only query-dependent mode
            plan.append((spec, tuple((e, _tokens(e)) for e in entries)))
    return plan


def _render_plan(plan: Sequence[_Slot], doc: CrossEncoderDocument, query_tokens: frozenset) -> str:
    """Finish a planned document for one query, or ``""`` when this class has
    nothing to say through any of the document's views.

    An empty return is the caller's signal to leave the cell NaN rather than
    score against a document that does not exist.

    ``"best"`` is the query-adaptive mode and the reason this step takes a query
    at all: a class with several example phrases has several different doors in,
    and which one fits depends on the item being classified. For a negative view
    it picks the *hardest* negative — the exclusion most like the query is the
    one actually at risk of being confused with it.

    Selection is lexical (token overlap) rather than embedding-based because
    ``SignalProviderSpec.build`` receives no encoder; see T33's design note. It
    is a defensible selector for picking among a handful of short phrases, and
    it keeps this provider stateless with nothing to persist."""
    parts = []
    for slot in plan:
        if isinstance(slot, str):
            if slot:
                parts.append(slot)
            continue
        spec, candidates = slot
        text, _ = max(candidates, key=lambda e: len(query_tokens & e[1]))
        parts.append(spec.label + text[: spec.max_chars])
    if not parts:
        return ""
    body = doc.join.join(parts)
    return f"{doc.instruction}{doc.join}{body}" if doc.instruction else body


def _render(definition: ClassDefinition, doc: CrossEncoderDocument, query_tokens: frozenset) -> str:
    """``_plan_class`` + ``_render_plan`` for a single class — the one-shot form,
    for callers with no chunk to amortize the plan over (and for tests)."""
    return _render_plan(_plan_class(definition, doc), doc, query_tokens)


class CrossEncoderSignalProvider(SignalProvider):
    """Reranks shortlisted candidates with a ``PairwiseReranker`` (T33).

    Runs in the assembler's *second* round (``needs_candidates = True``): it
    scores far too slowly to look at every class, so it can only run once the
    cheap signals have produced a shortlist. It therefore declares no
    ``candidate_features`` — it reorders the shortlist and can never extend it,
    which means candidate recall (the ceiling on system accuracy) is unchanged
    by switching this signal on.

    One ``(b, C)`` matrix per configured document, NaN everywhere except the
    cells actually reranked. Within the shortlist, NaN means "not reranked"
    (outside ``top_k``, or this class has no text for that document's views);
    outside it, the pair is not in the frame at all. Each matrix then collects
    the assembler's full generic derivation set — raw/missing/rank/norm/margin +
    per-query gap + is-top1 — so seven columns per document arrive with no
    arithmetic written here.

    Stateless with respect to the taxonomy: evidence is read from
    ``ctx.label_space`` per chunk, never copied at construction. Added classes
    are therefore picked up automatically, and ``rewrap_signal_providers`` needs
    no case for this provider."""

    name = "cross-encoder"
    needs_candidates = True

    def __init__(
        self,
        reranker: PairwiseReranker,
        config: Optional[CrossEncoderConfig] = None,
        ops: Optional[ArrayOps] = None,
    ):
        self._reranker = reranker
        self._cfg = config or CrossEncoderConfig()
        self._ops = ops or NumpyArrayOps()

    @property
    def reranker(self) -> PairwiseReranker:
        """The backend this provider scores with (fine-tuned in place per fold
        by ``TrainingPipeline`` when ``cross_encoder.train`` is on)."""
        return self._reranker

    @property
    def config(self) -> CrossEncoderConfig:
        return self._cfg

    # ---- schema -----------------------------------------------------------
    def candidate_features(self) -> Sequence[str]:
        """None, and structurally so — see the class docstring. The assembler
        rejects a second-stage provider that claims otherwise."""
        return ()

    @staticmethod
    def _columns_for(name: str) -> Dict[str, str]:
        return {
            "raw": f"ce_{name}",
            "missing": f"ce_{name}_missing",
            "rank": f"rank_ce_{name}",
            "norm": f"norm_ce_{name}",
            "margin": f"margin_ce_{name}",
        }

    def column_names(self) -> List[str]:
        names: List[str] = []
        for doc in self._cfg.documents:
            cols = self._columns_for(doc.name)
            names.extend(
                [
                    cols["raw"],
                    cols["missing"],
                    cols["rank"],
                    cols["norm"],
                    cols["margin"],
                    f"q_gap_ce_{doc.name}",
                    f"is_ce_{doc.name}_top1",
                ]
            )
        names.extend(f"ce_{a}_{b}_gap" for a, b in self._cfg.gaps)
        return names

    # ---- build ------------------------------------------------------------
    def build(self, ctx: SignalContext) -> List[SignalMatrix]:
        if ctx.candidates is None:
            raise ValueError(
                f"{type(self).__name__} declares needs_candidates=True but was built with "
                "SignalContext.candidates=None. It must run in the assembler's second "
                "round, after candidate selection."
            )
        cand = ctx.candidates
        shape = (len(ctx.texts), ctx.n_classes)
        # Tokenize each query once per chunk, not once per pair.
        query_tokens = [_tokens(t) for t in ctx.texts]
        definitions = ctx.label_space.definitions
        prescore = cand.signals.get(self._cfg.prescore_node)

        # One top-n selection per *distinct* top_k rather than per document:
        # documents sharing a top_k (the default pos/neg pair does) select the
        # same cells, and the selection is an O(b*C) partition over the same
        # prescore matrix. Keyed only by top_k because `prescore` is fixed for
        # the whole provider.
        grids: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        values = {}
        for doc in self._cfg.documents:
            if doc.top_k not in grids:
                grids[doc.top_k] = self._rerank_grid(doc.top_k, cand, prescore)
            values[doc.name] = self._score_document(
                doc, grids[doc.top_k], ctx.texts, query_tokens, definitions, shape
            )
        # Cross-document differences, gathered at the same (rows, cols) grid as
        # everything else. They belong to no single document, so they ride on
        # the first matrix as plain extra columns rather than being derived.
        gaps = {f"ce_{a}_{b}_gap": values[a] - values[b] for a, b in self._cfg.gaps}

        matrices: List[SignalMatrix] = []
        for i, doc in enumerate(self._cfg.documents):
            M = values[doc.name]
            matrices.append(
                SignalMatrix(
                    node=f"ce.{doc.name}",
                    value=M,
                    derive=frozenset({"raw", "missing", "rank", "norm", "margin"}),
                    columns=self._columns_for(doc.name),
                    gap_column=f"q_gap_ce_{doc.name}",
                    top1_idx=_argmax_or_missing(M),
                    top1_column=f"is_ce_{doc.name}_top1",
                    top1_check_valid=True,
                    extra_columns=gaps if i == 0 else {},
                )
            )
        return matrices

    def _score_document(
        self,
        doc: CrossEncoderDocument,
        grid: Tuple[np.ndarray, np.ndarray],
        texts: Sequence[str],
        query_tokens: List[frozenset],
        definitions: Sequence[ClassDefinition],
        shape: Tuple[int, int],
    ) -> np.ndarray:
        """One document's ``(b, C)`` matrix: NaN except where actually scored."""
        rows, cols = grid
        # Resolve each class's slots once for this chunk, not once per pair --
        # the evidence depends on the class alone (see `_plan_class`). Only the
        # classes actually selected are planned.
        plans = {int(c): _plan_class(definitions[c], doc) for c in np.unique(cols)}

        # Render the pairs for the selected cells. This is the one Python-level
        # loop in the provider, and it is the right trade: O(pairs) string
        # assembly feeding a model that costs orders of magnitude more per pair,
        # doing work (text rendering) that cannot vectorize and cannot run on a
        # device at all. Everything numeric on either side of it stays in the
        # array backend.
        pairs: List[Tuple[str, str]] = []
        scored_rows: List[int] = []
        scored_cols: List[int] = []
        for r, c in zip(rows.tolist(), cols.tolist()):
            document = _render_plan(plans[c], doc, query_tokens[r])
            if not document:
                continue  # class has nothing to say here -> stays NaN, never 0
            pairs.append((texts[r], document))
            scored_rows.append(r)
            scored_cols.append(c)

        scores: np.ndarray = (
            np.asarray(self._reranker.score(pairs), dtype=np.float64)
            if pairs
            else np.empty(0, dtype=np.float64)
        )
        if pairs and np.shape(scores) != (len(pairs),):
            raise ValueError(
                f"{type(self._reranker).__name__}.score returned shape {np.shape(scores)} for "
                f"{len(pairs)} pairs; expected exactly one score per pair, i.e. "
                f"({len(pairs)},). PairwiseReranker.score must preserve input order and "
                "length."
            )
        return self._scatter(shape, scored_rows, scored_cols, scores)

    def _scatter(self, shape, rows, cols, values) -> np.ndarray:
        """Place ``values`` at ``(rows, cols)`` in an otherwise-NaN ``(b, C)``
        matrix, through the array backend.

        Routed through ``ArrayOps.scatter_add`` rather than fancy-index
        assignment so the matrix is built wherever the configured backend lives
        (T84/T85) instead of forcing a numpy allocation into an otherwise
        device-resident pipeline. The scored-cell mask comes from a second
        scatter of ones, exactly as ``_scatter_knn`` distinguishes "summed to
        zero" from "never written" — a cell nothing wrote must stay NaN, never
        become the 0.0 the accumulator started at.

        The trailing ``to_host`` matches ``_scatter_knn``'s: the assembler's
        derivation kernels are not device-resident yet (T86), so every signal
        matrix is handed over on the host today. When that changes this call is
        the single line to revisit — which is exactly why the port makes it the
        one sanctioned exit."""
        ops = self._ops
        empty = len(rows) == 0
        total = ops.zeros(shape, dtype=np.float64)
        count = ops.zeros(shape, dtype=np.float64)
        if not empty:
            total = ops.scatter_add(total, rows, cols, values)
            count = ops.scatter_add(count, rows, cols, np.ones(len(rows), dtype=np.float64))
        return ops.to_host(ops.where(count > 0, total, np.nan))

    def _rerank_grid(
        self, top_k: int, cand, prescore: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Which shortlisted cells to rerank: the ``top_k`` best per query by
        ``prescore_node``, intersected with the shortlist.

        Falls back to the whole shortlist when the prescoring node is absent
        (a config where the ranking signal was switched off) — correct, just
        more expensive, and the alternative would be silently reranking nothing.

        The ``to_host`` here is deliberate and is the provider's *only* other
        exit from the array backend: the selection itself is computed by the
        backend (``_topn_mask`` takes ``ops``), but the resulting indices drive
        a Python loop over text, so they have to land on the host. Doing it as
        one explicit call keeps the transfer greppable rather than smuggled in
        by an implicit ``np.asarray`` somewhere in the loop."""
        if prescore is None:
            return cand.rows, cand.cols
        ops = self._ops
        selected = ops.asarray(cand.mask) & _topn_mask(prescore, top_k, ops=ops)
        rows, cols = np.nonzero(ops.to_host(selected))
        return rows, cols

    # ---- persistence ------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self._reranker.save(path)
        with open(os.path.join(path, "cross_encoder.json"), "w") as fh:
            json.dump({"kind": self._cfg.kind}, fh)

    @classmethod
    def load(cls, path: str) -> "CrossEncoderSignalProvider":
        raise NotImplementedError(
            "CrossEncoderSignalProvider is reloaded through its registered "
            "SignalProviderSpec.load, which has the RetrievalConfig needed to rebuild "
            "the document/evidence plan -- see infrastructure.registry."
        )
