"""Learned-sparse (SPLADE) retrieval signal (T94).

A SPLADE model reads a text with a masked-language model and, for every
vocabulary entry, takes ``max over tokens of log(1 + relu(logit))``. The result
is a sparse, non-negative weight per *word*: the words the text contains, the
related words the model infers (for Hebrew, the bare form of a prefixed word —
"הבנק" also lights up "בנק"), and zeros elsewhere. Query-document score is the
sparse dot product, exactly like BM25 — except the weights are learned, which is
what lets the fusion model rely on *which words* matched rather than on IDF.

Pieces, each behind the ``SparseTextEncoder`` port:

- ``HashingSparseEncoder`` — offline, dependency-free stand-in (log-tf over
  hashed tokens). What the test suite and the demo run; the sparse analogue of
  the ``hashing`` dense encoder.
- ``SpladeEncoder`` — a Hugging Face ``AutoModelForMaskedLM`` (any BERT-style
  checkpoint with its MLM head, e.g. a pretrained SPLADE model or a Hebrew BERT
  such as AlephBertGimmel). torch/transformers are imported lazily, so merely
  registering this backend never imports them. Saved with ``save_pretrained``
  (safetensors + JSON): portable, no pickle.
- ``train_splade`` — fine-tunes a ``SpladeEncoder`` on labeled items: each item
  must score its own class's description above other classes' (softmax over the
  batch's classes plus sampled negatives), plus the FLOPS penalty that keeps the
  vectors sparse.
- ``SpladeSignalProvider`` — a round-one ``SignalProvider``: ``splade.desc``
  (item vs. class description) and ``splade.knn`` (item vs. the example pool,
  aggregated per class), so it joins candidate selection and can raise candidate
  recall, unlike the round-two cross-encoder. It owns a learned model and an
  index, so it is fitted per fold through ``SignalProvider.fit``.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import zlib
from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.sparse as sp

from ..config import SpladeConfig
from ..domain import (
    ArrayOps,
    LabelSpace,
    SignalContext,
    SignalMatrix,
    SignalProvider,
    SparseTextEncoder,
)
from .array_ops import NumpyArrayOps
from .._progress import show_training_bars
from .device import resolve_device
from .signals import _argmax_or_missing, _scatter_knn

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


# ------------------------------------------------------------------ encoders
class HashingSparseEncoder(SparseTextEncoder):
    """Offline stand-in: ``1 + log(tf)`` per hashed lowercase word token.

    Deterministic across processes (``crc32``, not Python's salted ``hash``),
    needs nothing beyond numpy/scipy and learns nothing — it exists so the
    signal's plumbing (per-fold fitting, schema, persistence, inference) is
    testable offline, as CLAUDE.md requires."""

    kind = "hashing"

    def __init__(self, dim: int = 2**18):
        self._dim = int(dim)

    @property
    def vocab_size(self) -> int:
        return self._dim

    def encode(self, texts: Sequence[str]) -> sp.csr_matrix:
        indptr = [0]
        indices: list[int] = []
        data: list[float] = []
        for text in texts:
            counts: dict[int, int] = {}
            for tok in _TOKEN_RE.findall(str(text).lower()):
                h = zlib.crc32(tok.encode("utf-8")) % self._dim
                counts[h] = counts.get(h, 0) + 1
            for h in sorted(counts):
                indices.append(h)
                data.append(1.0 + math.log(counts[h]))
            indptr.append(len(indices))
        return sp.csr_matrix(
            (np.asarray(data, dtype=np.float32), np.asarray(indices, dtype=np.int64), indptr),
            shape=(len(texts), self._dim),
        )

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "sparse_encoder.json"), "w") as fh:
            json.dump({"kind": self.kind, "dim": self._dim}, fh)

    @classmethod
    def load(cls, directory: str) -> HashingSparseEncoder:
        with open(os.path.join(directory, "sparse_encoder.json")) as fh:
            return cls(int(json.load(fh)["dim"]))


def _require_transformers() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "retrieval.splade.kind='splade' needs torch and transformers "
            "(pip install 'text-classifier[splade]'). The offline 'hashing' kind "
            "needs neither."
        ) from exc
    return torch, AutoModelForMaskedLM, AutoTokenizer


class SpladeEncoder(SparseTextEncoder):
    """A masked-language model read out SPLADE-style (max-pooled
    ``log1p(relu(logits))`` over tokens).

    Special tokens ([CLS], [SEP], [PAD], ...) are zeroed: they fire on every
    text and carry no retrieval meaning. ``top_terms`` caps the non-zeros kept
    per text. Encoding is batched (``batch_size``) because the MLM head emits
    ``batch * max_length * vocab`` logits."""

    kind = "splade"

    def __init__(self, model: Any, tokenizer: Any, config: SpladeConfig):
        torch, _, _ = _require_transformers()
        self._torch = torch
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = config
        self.device = resolve_device(config.device)
        self.model.to(self.device)
        self.model.eval()
        special = set(getattr(tokenizer, "all_special_ids", []) or [])
        self._special = torch.tensor(sorted(special), dtype=torch.long) if special else None

    # ---- construction / persistence ----------------------------------------
    @classmethod
    def from_pretrained(cls, name_or_path: str, config: SpladeConfig) -> SpladeEncoder:
        _, AutoModelForMaskedLM, AutoTokenizer = _require_transformers()
        tokenizer = AutoTokenizer.from_pretrained(name_or_path)
        model = AutoModelForMaskedLM.from_pretrained(name_or_path)
        return cls(model, tokenizer, config)

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        self.model.save_pretrained(directory, safe_serialization=True)
        self.tokenizer.save_pretrained(directory)
        with open(os.path.join(directory, "sparse_encoder.json"), "w") as fh:
            json.dump({"kind": self.kind}, fh)

    @property
    def vocab_size(self) -> int:
        return int(self.model.get_output_embeddings().weight.shape[0])

    def term(self, index: int) -> str:
        return str(self.tokenizer.convert_ids_to_tokens(int(index)))

    # ---- encoding -------------------------------------------------------------
    def forward(self, texts: Sequence[str]) -> Any:
        """Differentiable ``(n, vocab)`` dense tensor of SPLADE weights."""
        torch = self._torch
        batch = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
            return_tensors="pt",
        ).to(self.device)
        logits = self.model(**batch).logits  # (n, L, V)
        weights = torch.log1p(torch.relu(logits)) * batch["attention_mask"].unsqueeze(-1)
        pooled = weights.max(dim=1).values  # (n, V)
        if self._special is not None:
            pooled = pooled.index_fill(1, self._special.to(self.device), 0.0)
        return pooled

    def encode(self, texts: Sequence[str]) -> sp.csr_matrix:
        torch = self._torch
        V = self.vocab_size
        k = min(self.cfg.top_terms, V)
        blocks: list[sp.csr_matrix] = []
        with torch.no_grad():
            for start in range(0, len(texts), self.cfg.batch_size):
                chunk = texts[start : start + self.cfg.batch_size]
                w = self.forward(chunk)
                vals, idx = torch.topk(w, k, dim=1)
                vals = vals.float().cpu().numpy()
                idx = idx.cpu().numpy()
                keep = vals > 0
                rows = np.repeat(np.arange(len(chunk)), k)[keep.ravel()]
                blocks.append(
                    sp.csr_matrix(
                        (vals[keep].astype(np.float32), (rows, idx[keep])),
                        shape=(len(chunk), V),
                    )
                )
        if not blocks:
            return sp.csr_matrix((0, V), dtype=np.float32)
        return sp.vstack(blocks).tocsr()


def build_sparse_encoder(config: SpladeConfig) -> SparseTextEncoder:
    """A *fresh* encoder from ``config`` (pretrained weights for ``splade``)."""
    if config.kind == "hashing":
        return HashingSparseEncoder(config.hashing_dim)
    if config.kind == "splade":
        return SpladeEncoder.from_pretrained(config.model_name, config)
    raise ValueError(f"unknown retrieval.splade.kind {config.kind!r}")


def load_sparse_encoder(directory: str, config: SpladeConfig) -> SparseTextEncoder:
    with open(os.path.join(directory, "sparse_encoder.json")) as fh:
        kind = json.load(fh)["kind"]
    if kind == "hashing":
        return HashingSparseEncoder.load(directory)
    if kind == "splade":
        return SpladeEncoder.from_pretrained(directory, config)
    raise ValueError(f"unknown sparse encoder kind {kind!r} in {directory!r}")


# ------------------------------------------------------------------ training
def train_splade(
    encoder: SpladeEncoder,
    texts: Sequence[str],
    labels: np.ndarray,
    class_texts: Sequence[str],
    config: SpladeConfig,
) -> dict[str, float]:
    """Fine-tune ``encoder`` in place so each item scores its own class's text
    above the other classes'.

    Per batch: the classes of the batch's items plus up to ``train_negatives``
    other classes sampled uniformly form the candidate set; the loss is softmax
    cross-entropy over the item-vs-class dot products, plus SPLADE's FLOPS
    regularizer ``sum_v (mean_i w_iv)^2`` on both sides, which pushes rarely
    useful vocabulary entries to exactly zero. Returns the last epoch's mean
    loss (for logs/tests)."""
    torch = encoder._torch
    from transformers import get_linear_schedule_with_warmup

    labels = np.asarray(labels, dtype=np.int64)
    n, C = len(texts), len(class_texts)
    if n == 0:
        return {"loss": float("nan")}
    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)
    model = encoder.model
    model.train()
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    params = [
        {
            "params": [
                p for n_, p in model.named_parameters() if not any(nd in n_ for nd in no_decay)
            ],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n_, p in model.named_parameters() if any(nd in n_ for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    opt = torch.optim.AdamW(params, lr=config.learning_rate)
    bs = config.train_batch_size
    steps = config.train_epochs * math.ceil(n / bs)
    sched = get_linear_schedule_with_warmup(opt, int(config.warmup_ratio * steps), steps)
    all_classes = np.arange(C)
    last = float("nan")
    from tqdm.auto import tqdm

    for epoch in range(config.train_epochs):
        order = rng.permutation(n)
        total, batches = 0.0, 0
        steps_bar = tqdm(
            range(0, n, bs),
            desc=f"splade epoch {epoch + 1}/{config.train_epochs}",
            unit="batch",
            disable=not show_training_bars() or None,
        )
        for start in steps_bar:
            idx = order[start : start + bs]
            pos = np.unique(labels[idx])
            others = np.setdiff1d(all_classes, pos, assume_unique=True)
            n_neg = min(config.train_negatives, len(others))
            neg = rng.choice(others, size=n_neg, replace=False) if n_neg else others[:0]
            cand = np.concatenate([pos, neg])
            target = torch.as_tensor(
                np.searchsorted(pos, labels[idx]), dtype=torch.long, device=encoder.device
            )
            q = encoder.forward([texts[i] for i in idx])
            d = encoder.forward([class_texts[c] for c in cand])
            scores = q @ d.T
            loss = torch.nn.functional.cross_entropy(scores, target)
            loss = loss + config.flops_query * (q.mean(0) ** 2).sum()
            loss = loss + config.flops_document * (d.mean(0) ** 2).sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += float(loss.detach())
            batches += 1
        last = total / max(batches, 1)
        log.info("splade: epoch %d/%d mean loss %.4f", epoch + 1, config.train_epochs, last)
    model.eval()
    return {"loss": last}


# ------------------------------------------------------------------ provider
def _scores(Q: sp.csr_matrix, D: sp.csr_matrix) -> np.ndarray:
    """Dense ``(b, n)`` float64 dot products of two sparse matrices."""
    return np.asarray((Q @ D.T).toarray(), dtype=np.float64)


class SpladeSignalProvider(SignalProvider):
    """Round-one learned-sparse signal: ``splade.desc`` and ``splade.knn``.

    - ``splade.desc``: item vs. each class description. 0 (no shared weighted
      term) becomes NaN — "did not retrieve", never a true 0 — as for BM25.
    - ``splade.knn``: the item's ``k`` nearest examples in the pool by sparse dot
      product, aggregated per class (sum / max / count), exactly like the dense
      and BM25 kNN signals. Neighbours with no overlap are dropped, not counted.

    Both join the candidate top-n union, so this signal can bring in classes the
    other retrievers missed (Hebrew prefixes, synonyms the model learned).

    State: a sparse encoder and the encoded example pool, fitted by ``fit`` per
    fold. Class descriptions are encoded lazily from ``ctx.label_space`` and
    cached by content, so added or edited classes (the no-retrain update path)
    are picked up with no extra wiring."""

    name = "splade"

    def __init__(
        self,
        config: SpladeConfig | None = None,
        ops: ArrayOps | None = None,
        encoder: SparseTextEncoder | None = None,
        pool: sp.csr_matrix | None = None,
        pool_labels: np.ndarray | None = None,
        chunk: int = 256,
    ):
        self._cfg = config or SpladeConfig()
        self._ops = ops or NumpyArrayOps()
        self._encoder = encoder
        self._pool = pool
        self._pool_labels = None if pool_labels is None else np.asarray(pool_labels, np.int64)
        self._chunk = max(int(chunk), 1)
        self._desc_key: tuple[str, ...] | None = None
        self._desc: sp.csr_matrix | None = None
        self.training_history: dict[str, float] | None = None

    @property
    def encoder(self) -> SparseTextEncoder | None:
        return self._encoder

    # ---- schema --------------------------------------------------------------
    def candidate_features(self) -> Sequence[str]:
        return ("splade.desc", "splade.knn")

    def column_names(self) -> list[str]:
        return [
            "s_desc_sim",
            "s_desc_missing",
            "rank_s_desc",
            "norm_s_desc",
            "margin_s_desc",
            "q_gap_s_desc",
            "is_s_desc_top1",
            "s_knn_sum",
            "s_knn_missing",
            "rank_s_knn",
            "margin_s_knn",
            "q_gap_s_knn",
            "is_s_knn_top1",
            "s_knn_max",
            "s_knn_count",
            "abs_top_splade",
        ]

    # ---- fitting ---------------------------------------------------------------
    def fit(self, texts: Sequence[str], labels: np.ndarray, label_space: LabelSpace) -> None:
        """Fresh encoder (fine-tuned on these rows when ``train``), then index them."""
        encoder = build_sparse_encoder(self._cfg)
        if self._cfg.train:
            assert isinstance(encoder, SpladeEncoder)
            log.info("splade: fine-tuning on %d item(s)", len(texts))
            self.training_history = train_splade(
                encoder, list(texts), np.asarray(labels), label_space.descriptions, self._cfg
            )
        self._encoder = encoder
        self._pool = encoder.encode(list(texts))
        self._pool_labels = np.asarray(labels, dtype=np.int64)
        self._desc_key, self._desc = None, None

    def with_added_examples(self, texts: Sequence[str], labels: np.ndarray) -> SpladeSignalProvider:
        """A copy whose pool also holds ``texts`` (the no-retrain update path).
        The encoder is reused as is — new examples are indexed, not trained on."""
        enc = self._require_encoder()
        assert self._pool is not None and self._pool_labels is not None
        pool = sp.vstack([self._pool, enc.encode(list(texts))]).tocsr()
        pool_labels = np.concatenate([self._pool_labels, np.asarray(labels, dtype=np.int64)])
        return SpladeSignalProvider(self._cfg, self._ops, enc, pool, pool_labels, self._chunk)

    def _require_encoder(self) -> SparseTextEncoder:
        if self._encoder is None or self._pool is None:
            raise ValueError(
                "SpladeSignalProvider was asked to score before it was fitted; "
                "TrainingPipeline fits it per fold via SignalProvider.fit"
            )
        return self._encoder

    def _descriptions(self, label_space: LabelSpace) -> sp.csr_matrix:
        key = tuple(label_space.descriptions)
        if self._desc is None or self._desc_key != key:
            self._desc = self._require_encoder().encode(list(key))
            self._desc_key = key
        return self._desc

    # ---- scoring ---------------------------------------------------------------
    def _knn(
        self, Q: sp.csr_matrix, k: int, self_ids: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(b, k)`` neighbour labels and scores, best first; label -1 / NaN
        score where fewer than ``k`` pool items share any weighted term."""
        assert self._pool is not None and self._pool_labels is not None
        b, n = Q.shape[0], self._pool.shape[0]
        kk = min(k, n)
        labels = np.full((b, k), -1, dtype=np.int64)
        scores = np.full((b, k), np.nan, dtype=np.float64)
        if kk == 0:
            return labels, scores
        for s in range(0, b, self._chunk):
            S = _scores(Q[s : s + self._chunk], self._pool)
            r = np.arange(S.shape[0])
            if self_ids is not None:
                own = np.asarray(self_ids[s : s + self._chunk])
                hit = own >= 0
                S[r[hit], own[hit]] = -np.inf
            top = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
            vals = S[r[:, None], top]
            order = np.argsort(-vals, axis=1, kind="stable")
            top = top[r[:, None], order]
            vals = vals[r[:, None], order]
            ok = vals > 0
            labels[s : s + len(r), :kk] = np.where(ok, self._pool_labels[top], -1)
            scores[s : s + len(r), :kk] = np.where(ok, vals, np.nan)
        return labels, scores

    def build(self, ctx: SignalContext) -> list[SignalMatrix]:
        enc = self._require_encoder()
        Q = enc.encode(list(ctx.texts))
        desc_raw = _scores(Q, self._descriptions(ctx.label_space))
        desc = np.where(desc_raw > 0, desc_raw, np.nan)
        lab, sco = self._knn(Q, ctx.k, ctx.self_ids)
        k_sum, k_max, k_cnt = _scatter_knn(lab, sco, ctx.n_classes, self._ops)
        with np.errstate(invalid="ignore"):
            top = np.where(np.isnan(sco[:, 0]), 0.0, sco[:, 0])
        return [
            SignalMatrix(
                node="splade.desc",
                value=desc,
                derive=frozenset({"raw", "missing", "rank", "norm", "margin"}),
                columns={
                    "raw": "s_desc_sim",
                    "missing": "s_desc_missing",
                    "rank": "rank_s_desc",
                    "norm": "norm_s_desc",
                    "margin": "margin_s_desc",
                },
                gap_column="q_gap_s_desc",
                top1_idx=_argmax_or_missing(desc, require_positive=True),
                top1_column="is_s_desc_top1",
                top1_check_valid=True,
                topn_positive_only=True,
            ),
            SignalMatrix(
                node="splade.knn",
                value=k_sum,
                derive=frozenset({"raw", "missing", "rank", "margin"}),
                columns={
                    "raw": "s_knn_sum",
                    "missing": "s_knn_missing",
                    "rank": "rank_s_knn",
                    "margin": "margin_s_knn",
                },
                gap_column="q_gap_s_knn",
                top1_idx=lab[:, 0],
                top1_column="is_s_knn_top1",
                top1_check_valid=True,
                extra_columns={"s_knn_max": k_max, "s_knn_count": k_cnt},
                extra_scalars={"abs_top_splade": top},
            ),
        ]

    # ---- explanations ------------------------------------------------------------
    def top_terms(self, text: str, n: int = 10) -> list[tuple[str, float]]:
        """The ``n`` heaviest vocabulary entries the encoder assigns ``text`` —
        which words this signal is relying on, for debugging and explanations."""
        enc = self._require_encoder()
        row = enc.encode([text]).tocsr()
        order = np.argsort(-row.data)[:n]
        return [(enc.term(int(row.indices[i])), float(row.data[i])) for i in order]

    # ---- persistence -------------------------------------------------------------
    def save(self, path: str) -> None:
        enc = self._require_encoder()
        assert self._pool is not None and self._pool_labels is not None
        os.makedirs(path, exist_ok=True)
        enc.save(os.path.join(path, "encoder"))
        pool = self._pool.tocsr()
        np.savez(
            os.path.join(path, "pool.npz"),
            data=pool.data.astype(np.float32),
            indices=pool.indices.astype(np.int64),
            indptr=pool.indptr.astype(np.int64),
            shape=np.asarray(pool.shape, dtype=np.int64),
            labels=self._pool_labels,
        )
        with open(os.path.join(path, "splade.json"), "w") as fh:
            json.dump({"kind": self._cfg.kind, "n_pool": int(pool.shape[0])}, fh)

    @classmethod
    def load_from(cls, path: str, config: SpladeConfig, chunk: int = 256) -> SpladeSignalProvider:
        encoder = load_sparse_encoder(os.path.join(path, "encoder"), config)
        with np.load(os.path.join(path, "pool.npz"), allow_pickle=False) as z:
            pool = sp.csr_matrix(
                (z["data"], z["indices"], z["indptr"]), shape=tuple(int(x) for x in z["shape"])
            )
            labels = z["labels"]
        return cls(config, None, encoder, pool, labels, chunk)

    @classmethod
    def load(cls, path: str) -> SpladeSignalProvider:
        raise NotImplementedError(
            "SpladeSignalProvider is reloaded through its registered SignalProviderSpec.load, "
            "which has the RetrievalConfig -- see infrastructure.registry."
        )
