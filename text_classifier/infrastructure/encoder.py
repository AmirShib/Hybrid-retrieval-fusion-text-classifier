"""Encoder adapters behind the TextEncoder port.

- ``SentenceTransformerEncoder``: wraps a SentenceTransformer (torch + a model
  download) and supports fine-tuning over (item, class-description) pairs with
  a pluggable loss (``EncoderConfig.train_loss``, default
  MultipleNegativesSymmetricRankingLoss; see ``domain.services.ENCODER_LOSSES``),
  scoring each epoch on a held-out slice so a multi-epoch run keeps its *best*
  epoch rather than its last (``EncoderEpochTracker``).
- ``TfidfEncoder``: a torch-free, air-gap-friendly alternative built on sklearn's
  TfidfVectorizer. Its vocabulary/IDF are corpus-dependent, so it is *fit* on a
  training corpus rather than loaded pretrained.
- ``HashingEncoder``: a dependency-free, deterministic bag-of-hashed-tokens
  encoder. It carries no learned weights, needs no network, and is reproducible
  across processes/platforms — useful for offline smoke tests, CI, and as a
  trivial baseline. It is *not* a semantic model and should not be used where
  embedding quality matters.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import shutil
import tempfile
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import EncoderConfig
from ..domain import (
    EpochSelectionPolicy,
    LabeledItem,
    LabelSpace,
    TextEncoder,
    encoder_retrieval_metrics,
)
from .device import resolve_device

if TYPE_CHECKING:  # torch-free at runtime; the type is only for checkers
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def _require_sentence_transformers() -> Any:
    """Import ``sentence_transformers``, or raise a clear, actionable error.

    ``sentence-transformers`` (and the torch it pulls in) lives behind the
    ``sentence-transformers`` extra, not core ``dependencies`` (T63) — an
    air-gapped or lightweight install may not have it. A bare ``ImportError``
    deep in a training run is unhelpful; this points at the fix.
    """
    try:
        import sentence_transformers
    except ImportError as exc:
        raise ImportError(
            "encoder kind 'sentence-transformers' requires the 'sentence-transformers' "
            "extra, which is not installed. Install it with:\n"
            "    pip install text-classifier[sentence-transformers]\n"
            "Or use a torch-free encoder instead: --encoder-kind tfidf (corpus-fitted) "
            "or --encoder-kind hashing (dependency-free, non-semantic)."
        ) from exc
    return sentence_transformers


def _encode_options(config: Optional[EncoderConfig]) -> Dict[str, Any]:
    """Extract the encode-time settings an ``EncoderConfig`` carries, as the
    keyword arguments ``SentenceTransformerEncoder`` accepts."""
    if config is None:
        return {}
    return {
        "encode_kwargs": config.encode_kwargs,
        "query_prompt": config.query_prompt,
        "document_prompt": config.document_prompt,
        "query_prompt_name": config.query_prompt_name,
        "document_prompt_name": config.document_prompt_name,
    }


class SentenceTransformerEncoder(TextEncoder):
    """Adapter producing L2-normalized float32 embeddings.

    Supports asymmetric query/document encoding for instruction-tuned models
    (E5/BGE/GTE...): a ``query_prompt``/``document_prompt`` literal prefix is
    prepended per role, or a ``*_prompt_name`` selects a model-card prompt
    (an explicit prompt wins over its prompt_name). ``encode_kwargs`` merge
    into every ``model.encode(...)`` call, with user keys winning over our
    defaults — except ``normalize_embeddings``/``convert_to_numpy``/
    ``convert_to_tensor``, which are forced: L2-normalization (dot == cosine)
    is a package-wide invariant and cannot be configured away, but *which*
    array type carries it is backend-driven (T85) — see ``array_backend``.

    ``array_backend`` (``"numpy"`` by default -- unchanged behaviour for every
    existing caller) selects the container type ``encode``/``encode_queries``/
    ``encode_documents`` return: ``"numpy"`` forces ``convert_to_numpy=True``
    exactly as before; ``"torch"`` forces ``convert_to_numpy=False,
    convert_to_tensor=True`` and returns the raw (still L2-normalized) tensor,
    resident on whatever device the underlying model lives on, so a caller
    that immediately hands it to a torch-backed retriever pays no D2H-then-H2D
    round trip. Set via ``set_array_backend`` after construction (not a
    constructor-only choice) so a shared, already-built encoder can be
    upgraded once a run resolves its array backend, without rebuilding it.

    ``training_history`` is the per-epoch record of a fine-tune that selected its
    best epoch (see ``train_encoder``); it is evidence, not state — ``save``
    writes it beside the model as ``encoder_training.json`` and ``load`` does not
    need it back. It is empty for a loaded or non-fine-tuned encoder.
    """

    _PROTECTED_ENCODE_KWARGS = ("normalize_embeddings", "convert_to_numpy", "convert_to_tensor")
    TRAINING_HISTORY_NAME = "encoder_training.json"

    def __init__(
        self,
        model: "SentenceTransformer",
        batch_size: int = 64,
        *,
        encode_kwargs: Optional[Dict[str, Any]] = None,
        query_prompt: Optional[str] = None,
        document_prompt: Optional[str] = None,
        query_prompt_name: Optional[str] = None,
        document_prompt_name: Optional[str] = None,
        training_history: Optional[Dict[str, Any]] = None,
        array_backend: str = "numpy",
    ):
        self._model = model
        self._batch_size = batch_size
        self.training_history: Dict[str, Any] = dict(training_history or {})
        self.array_backend = array_backend
        cleaned = dict(encode_kwargs or {})
        for key in self._PROTECTED_ENCODE_KWARGS:
            if key in cleaned:
                logger.warning(
                    "encode_kwargs[%r]=%r is ignored: %s=True is required so "
                    "embeddings stay L2-normalized numpy arrays (dot == cosine)",
                    key,
                    cleaned.pop(key),
                    key,
                )
        self._encode_kwargs = cleaned
        self._query_prompt = query_prompt
        self._document_prompt = document_prompt
        self._query_prompt_name = query_prompt_name
        self._document_prompt_name = document_prompt_name

    @classmethod
    def load(
        cls,
        model_name_or_path: str,
        batch_size: int = 64,
        device=None,
        config: Optional[EncoderConfig] = None,
        **kwargs,
    ) -> "SentenceTransformerEncoder":
        SentenceTransformer = _require_sentence_transformers().SentenceTransformer

        # device=None lets SentenceTransformer do its own auto-detection (which,
        # in current sentence-transformers, checks MPS too); resolve_device here
        # is only to log what that resolves to, so mps_ok=True keeps the log
        # line accurate on Apple Silicon rather than always claiming "cpu".
        logger.info(
            "loading SentenceTransformer encoder on device=%s",
            resolve_device(device, mps_ok=True),
        )
        return cls(
            SentenceTransformer(model_name_or_path, device=device, **kwargs),
            batch_size,
            **_encode_options(config),
        )

    @property
    def model(self):
        return self._model

    def set_array_backend(self, array_backend: str) -> None:
        """Switch the container type ``encode*`` return going forward. See the
        class docstring's ``array_backend`` note. ``"numpy"`` (the value every
        encoder starts with) is always safe; ``"torch"`` requires torch to
        actually be installed, which the caller is responsible for having
        established (e.g. via ``resolve_array_backend``) before calling this."""
        self.array_backend = array_backend

    @property
    def roles_share_encoding(self) -> bool:
        """Whether ``encode_queries`` and ``encode_documents`` are the same
        function of the text, so a document embedding may be reused as a query
        embedding (T89).

        True exactly when both roles resolve to the same *effective* prompt.
        Note this compares the effective pair, not the four raw fields, because
        ``_encode`` gives an explicit literal ``prompt`` precedence over a
        ``prompt_name`` — a role with both set ignores its ``prompt_name``, so
        comparing the raw fields would report a difference that does not exist
        (and, worse, could miss one that does).

        False for any instruction-tuned asymmetric setup (E5/BGE-style
        ``"query: "``/``"passage: "`` prompts, T28): there a document embedding
        is genuinely not a query embedding, and reusing it would be a
        correctness bug rather than an optimization.
        """
        return self._effective_prompt(
            self._query_prompt, self._query_prompt_name
        ) == self._effective_prompt(self._document_prompt, self._document_prompt_name)

    @staticmethod
    def _effective_prompt(
        prompt: Optional[str], prompt_name: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        """The ``(prompt, prompt_name)`` pair ``_encode`` will actually apply,
        mirroring its ``if prompt: ... elif prompt_name: ...`` precedence."""
        return (prompt, None) if prompt else (None, prompt_name)

    def encode(self, texts: Sequence[str]) -> Any:
        return self._encode(texts, prompt=None, prompt_name=None)

    def encode_queries(self, texts: Sequence[str]) -> Any:
        return self._encode(texts, self._query_prompt, self._query_prompt_name)

    def encode_documents(self, texts: Sequence[str]) -> Any:
        return self._encode(texts, self._document_prompt, self._document_prompt_name)

    def _encode(
        self, texts: Sequence[str], prompt: Optional[str], prompt_name: Optional[str]
    ) -> Any:
        texts = list(texts)
        kwargs: Dict[str, Any] = {"batch_size": self._batch_size, "show_progress_bar": False}
        kwargs.update(self._encode_kwargs)  # user keys win over the two defaults above
        if prompt:  # explicit literal prefix wins over a named prompt
            texts = [prompt + t for t in texts]
        elif prompt_name:
            kwargs["prompt_name"] = prompt_name
        kwargs["normalize_embeddings"] = True
        if self.array_backend == "torch":
            # convert_to_tensor=True keeps the embedding resident on whatever
            # device the model lives on (T85) -- no forced D2H here, unlike
            # the numpy path below.
            kwargs["convert_to_numpy"] = False
            kwargs["convert_to_tensor"] = True
            return self._model.encode(texts, **kwargs)
        kwargs["convert_to_numpy"] = True
        emb = self._model.encode(texts, **kwargs)
        return np.ascontiguousarray(emb, dtype=np.float32)

    def save(self, directory: str) -> None:
        self._model.save(directory)
        if self.training_history:
            # Ships the per-epoch table next to the weights, so a deployed model
            # dir answers "which epoch is this, and how did the others score?".
            with open(os.path.join(directory, self.TRAINING_HISTORY_NAME), "w") as fh:
                json.dump(self.training_history, fh, indent=2)


class TfidfEncoder(TextEncoder):
    """Torch-free TextEncoder: dense, L2-normalized TF-IDF vectors.

    The vocabulary and IDF weights depend on the training corpus, so the encoder
    must be fit before use (``fit`` / ``fit_on``). Rows are L2-normalized so that
    dot product == cosine, honoring the core invariant; an empty or fully-OOV
    text maps to an all-zero (finite, never NaN) vector.

    Persistence is the vocabulary + IDF weights (JSON + npz, no pickle) plus the
    vectorizer kwargs — sklearn only, no torch and no download, so a model
    directory stays portable to an air-gapped host and inert to load.
    """

    _VOCAB_NAME = "tfidf_vocab.json"
    _IDF_NAME = "tfidf_idf.npz"
    _PICKLE_NAME = "tfidf.pkl"  # legacy fallback
    # T89: role-symmetric by construction -- it defines only `encode`, so both
    # role methods inherit TextEncoder's default delegation to it.
    roles_share_encoding = True

    def __init__(self, vectorizer: Any = None, tfidf_kwargs: dict | None = None):
        self._vectorizer = vectorizer
        self._kwargs = dict(tfidf_kwargs or {})

    @classmethod
    def from_config(cls, config: EncoderConfig) -> "TfidfEncoder":
        """Build an *unfitted* encoder carrying the configured vectorizer kwargs."""
        return cls(tfidf_kwargs=config.params)

    def fit(self, texts: Sequence[str]) -> "TfidfEncoder":
        from sklearn.feature_extraction.text import TfidfVectorizer

        vec = TfidfVectorizer(**self._kwargs)
        vec.fit(list(texts))
        self._vectorizer = vec
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._vectorizer is None:
            raise RuntimeError("TfidfEncoder must be fit before encode()")
        X = np.asarray(self._vectorizer.transform(list(texts)).todense(), dtype=np.float32)
        # Enforce unit norm regardless of the vectorizer's own `norm` setting;
        # zero rows (empty/OOV) stay zero rather than becoming NaN.
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.clip(norms, 1e-8, None)

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        vec = self._vectorizer
        try:
            json.dumps(self._kwargs)
        except TypeError as exc:
            raise ValueError(
                f"TfidfEncoder's tfidf_kwargs must be JSON-serializable to persist "
                f"without pickle; got {self._kwargs!r}"
            ) from exc
        vocab = {term: int(col) for term, col in vec.vocabulary_.items()}
        with open(os.path.join(directory, self._VOCAB_NAME), "w") as fh:
            json.dump({"vocabulary": vocab, "kwargs": self._kwargs}, fh)
        with open(os.path.join(directory, self._IDF_NAME), "wb") as fh:
            np.savez_compressed(fh, idf=vec.idf_)

    @classmethod
    def load(cls, directory: str, batch_size: int = 64, device=None) -> "TfidfEncoder":
        """Signature mirrors SentenceTransformerEncoder.load for registry parity."""
        vocab_path = os.path.join(directory, cls._VOCAB_NAME)
        idf_path = os.path.join(directory, cls._IDF_NAME)
        if os.path.isfile(vocab_path) and os.path.isfile(idf_path):
            from sklearn.feature_extraction.text import TfidfVectorizer

            with open(vocab_path) as fh:
                meta = json.load(fh)
            vec = TfidfVectorizer(vocabulary=meta["vocabulary"], **meta["kwargs"])
            with open(idf_path, "rb") as fh:
                vec.idf_ = np.load(fh)["idf"]
            return cls(vectorizer=vec, tfidf_kwargs=meta["kwargs"])
        pkl_path = os.path.join(directory, cls._PICKLE_NAME)
        if os.path.isfile(pkl_path):
            logger.warning(
                "loading legacy pickle artifact %r; re-save this model directory "
                "to upgrade to the pickle-free format",
                pkl_path,
            )
            with open(pkl_path, "rb") as fh:
                return cls(vectorizer=pickle.load(fh))
        raise FileNotFoundError(
            f"no TF-IDF vectorizer found in {directory!r} "
            f"(expected {cls._VOCAB_NAME}+{cls._IDF_NAME}, or legacy {cls._PICKLE_NAME})"
        )


class HashingEncoder(TextEncoder):
    """Deterministic bag-of-hashed-tokens embeddings, L2-normalized.

    Shared tokens produce higher cosine similarity, making retrieval meaningful
    enough to exercise the pipeline offline without a real bi-encoder or network
    access. It learns nothing and depends only on numpy + the standard library,
    so it is the right encoder for the offline demo, CI, and air-gapped smoke
    tests; it is deliberately *not* a substitute for a semantic encoder.

    Token hashing uses ``hashlib.sha256`` rather than the builtin ``hash()`` so
    embeddings are byte-for-byte identical across Python processes, versions, and
    platforms — no ``PYTHONHASHSEED`` pinning required.
    """

    # T89: role-symmetric by construction -- see TfidfEncoder's note.
    roles_share_encoding = True

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    @staticmethod
    def _bucket_and_sign(token: str) -> Tuple[int, float]:
        """Map a token to a (bucket_index_seed, signed_weight) pair via SHA-256.

        First 4 bytes (little-endian) seed the bucket; bit 0 of byte 4 picks the
        sign. SHA-256 is stable everywhere, so this is fully reproducible.
        """
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        h = int.from_bytes(digest[:4], "little")
        sign = 1.0 if digest[4] & 1 else -1.0
        return h, sign

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in str(t).lower().split():
                h, sign = self._bucket_and_sign(tok)
                out[i, h % self.dim] += sign
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.clip(norms, 1e-8, None)

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "hashing_encoder.json"), "w") as fh:
            json.dump({"dim": self.dim}, fh)

    @classmethod
    def load(cls, path: str, batch_size: int = 64, device=None) -> "HashingEncoder":
        """Signature mirrors SentenceTransformerEncoder.load for registry parity."""
        try:
            with open(os.path.join(path, "hashing_encoder.json")) as fh:
                return cls(dim=json.load(fh)["dim"])
        except FileNotFoundError:
            return cls()


def fit_tfidf_encoder(
    items: Sequence[LabeledItem],
    label_space: LabelSpace,
    config: EncoderConfig,
) -> TfidfEncoder:
    """Build + fit a TfidfEncoder on the given items' text.

    Mirrors ``train_encoder``'s signature so the registry can dispatch corpus
    fitting uniformly. Crucially, this is called with a *fold's training rows
    only* in the OOF loop, which is what keeps the vocabulary leakage-free.
    """
    return TfidfEncoder.from_config(config).fit([it.text for it in items])


# Below this many held-out items the per-epoch metrics are noise: a single item
# moves desc_acc@1 by a quarter, so "best epoch" would be a coin flip. Selection
# switches itself off (with a warning) rather than pick an epoch on that basis.
MIN_EPOCH_HOLDOUT = 4


def _stratified_holdout(
    labels: np.ndarray, ratio: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Split item positions into ``(fit, holdout)``, taking ``ratio`` of each class.

    Per class: shuffle deterministically from ``seed``, then move
    ``floor(ratio * count)`` positions to the holdout — but never a class's last
    remaining example, so every class keeps at least one (item, description) pair
    to be pulled toward by the loss. Stratifying (rather than slicing the list)
    means a rare class is represented in the scored slice instead of vanishing
    into the fine-tune, which is what makes the per-epoch metrics comparable
    across epochs on imbalanced data.

    Returns index arrays into ``labels``, ascending. Same ``seed`` -> same split
    (the determinism invariant); ``ratio == 0`` -> an empty holdout.
    """
    rng = np.random.default_rng(seed)
    fit: List[int] = []
    holdout: List[int] = []
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        rng.shuffle(idx)
        n_out = min(int(len(idx) * ratio), len(idx) - 1)
        holdout.extend(idx[:n_out].tolist())
        fit.extend(idx[n_out:].tolist())
    return (
        np.sort(np.asarray(fit, dtype=np.intp)),
        np.sort(np.asarray(holdout, dtype=np.intp)),
    )


class EncoderEpochTracker:
    """Scores a fine-tune's held-out slice after each epoch and remembers the best.

    Framework-free by construction. It drives three collaborators:

    - ``encoder`` — a ``TextEncoder`` *view of the model being trained*, so the
      holdout is embedded through the same role/prompt logic used at inference
      time (and picks up the live weights on every call);
    - ``policy`` — an ``EpochSelectionPolicy``, which owns the entire "is this
      epoch better / should we stop" rule;
    - ``snapshot`` — a callback that persists the current weights wherever the
      caller can reload them from, invoked exactly on the epochs that set a new
      best.

    That split leaves the torch-specific glue in ``train_encoder`` down to "call
    ``observe`` once per epoch", and lets the whole selection loop be tested with
    a stub encoder and no ML framework installed.

    ``observe`` tolerates being called twice on unchanged weights (some
    sentence-transformers versions can invoke an evaluator at both a step
    boundary and the epoch end): the repeat is ignored, because counting it would
    shift the epoch numbering and, with patience, stop training for no reason.
    """

    def __init__(
        self,
        encoder: TextEncoder,
        holdout_texts: Sequence[str],
        holdout_labels: np.ndarray,
        descriptions: Sequence[str],
        policy: EpochSelectionPolicy,
        snapshot: Callable[[], None],
        pool_texts: Optional[Sequence[str]] = None,
        pool_labels: Optional[np.ndarray] = None,
    ) -> None:
        self._encoder = encoder
        self._holdout_texts = list(holdout_texts)
        self._holdout_labels = np.asarray(holdout_labels, dtype=np.intp)
        self._descriptions = list(descriptions)
        self.policy = policy
        self._snapshot = snapshot
        self._pool_texts = None if pool_texts is None else list(pool_texts)
        self._pool_labels = None if pool_labels is None else np.asarray(pool_labels, dtype=np.intp)
        self.history: List[Dict[str, float]] = []
        self._last_emb: Optional[np.ndarray] = None

    def observe(self) -> Optional[float]:
        """Score the current weights; snapshot them if this is the best epoch yet.

        Returns the selection metric for this epoch, or ``None`` if the call was a
        repeat on weights that have not changed since the last one.
        """
        query_emb = self._encoder.encode_queries(self._holdout_texts)
        if self._last_emb is not None and np.array_equal(query_emb, self._last_emb):
            return None
        self._last_emb = query_emb

        pool_emb = None
        if self._pool_texts is not None:
            pool_emb = self._encoder.encode_documents(self._pool_texts)
        metrics = encoder_retrieval_metrics(
            query_emb,
            self._encoder.encode_documents(self._descriptions),
            self._holdout_labels,
            pool_emb=pool_emb,
            pool_labels=self._pool_labels,
        )
        self.history.append(metrics)
        epoch = len(self.history)
        # The policy is the single source of truth for "best": ask it about the
        # whole history rather than tracking an incumbent here, so selection during
        # training and selection re-derived from a persisted history cannot drift.
        is_best = self.policy.best_epoch(self.history) == epoch
        if is_best:
            self._snapshot()
        logger.info(
            "encoder epoch %d: %s%s",
            epoch,
            ", ".join(f"{name}={value:.4f}" for name, value in metrics.items()),
            " (best so far)" if is_best else "",
        )
        return self.policy.score(metrics)

    @property
    def best_epoch(self) -> int:
        """1-based best epoch, or 0 if no epoch produced a usable score."""
        return self.policy.best_epoch(self.history)

    @property
    def stop_requested(self) -> bool:
        return self.policy.should_stop(self.history)

    def report(self) -> Dict[str, Any]:
        """JSON-clean record of the selection: what was measured, and what won."""
        return {
            "select_metric": self.policy.metric,
            "select_min_delta": self.policy.min_delta,
            "early_stopping_patience": self.policy.patience,
            "n_holdout_items": len(self._holdout_texts),
            "best_epoch": self.best_epoch,
            "epochs": [
                {"epoch": epoch, **metrics} for epoch, metrics in enumerate(self.history, start=1)
            ],
        }


class _StopFineTuning(Exception):
    """Internal signal that unwinds out of ``SentenceTransformer.fit`` to stop early.

    An evaluator has no other way to end training, and nothing in the
    sentence-transformers/transformers call chain catches it. Safe to raise: the
    best epoch's weights are already snapshotted, and the in-memory model is left
    holding the epoch that was just scored.
    """


def _epoch_evaluator(tracker: EncoderEpochTracker) -> Any:
    """Wrap ``tracker`` in a sentence-transformers evaluator.

    ``fit`` runs an evaluator once at the end of every epoch — v2 from its own
    training loop, v3+ from an ``EvaluatorCallback`` — and accepts a bare float
    return in both, so the hook is just "score the holdout, hand back the number".
    We pass ``evaluation_steps=0`` at the call site so there is no second,
    step-triggered call per epoch.

    The class is defined inside the function to keep importing this module
    torch-free. Subclassing ``SentenceEvaluator`` is required, not cosmetic: the
    v3+ trainer wraps anything that is not a ``BaseEvaluator`` in a
    ``SequentialEvaluator``. The base's own defaults for ``greater_is_better`` /
    ``primary_metric`` are left alone — with ``save_best_model=False`` and no
    ``callback``, nothing on the fit path reads them.
    """
    from sentence_transformers.evaluation import SentenceEvaluator

    class _EpochEvaluator(SentenceEvaluator):
        def __call__(
            self, model: Any, output_path: Optional[str] = None, epoch: int = -1, steps: int = -1
        ) -> float:
            score = tracker.observe()
            if score is None:  # repeat call on unchanged weights
                return float("nan")
            if tracker.stop_requested:
                raise _StopFineTuning(
                    f"no improvement in {tracker.policy.metric} for "
                    f"{tracker.policy.patience} epoch(s)"
                )
            return score

    return _EpochEvaluator()


# EncoderConfig.train_loss friendly alias -> sentence_transformers.losses class
# name. See domain.services.ENCODER_LOSSES for what each does.
_TRAIN_LOSS_ALIASES = {
    "multiple_negatives_symmetric_ranking": "MultipleNegativesSymmetricRankingLoss",
    "multiple_negatives_ranking": "MultipleNegativesRankingLoss",
    "cached_multiple_negatives_ranking": "CachedMultipleNegativesRankingLoss",
}


def _build_train_loss(name: str, model: Any, losses: Any, params: Dict[str, Any]) -> Any:
    """Build the fine-tuning loss named by ``EncoderConfig.train_loss``.

    ``name`` is either one of the three friendly aliases above -- verified
    compatible with the (item_text, description) InputExample pairs
    train_encoder builds, so those three are a pure constructor lookup -- or
    any other class name under ``sentence_transformers.losses`` (e.g.
    "CosineSimilarityLoss", "TripletLoss"), resolved directly. The package puts
    no ceiling on which of its own losses you can reach for; it only vouches
    for the three aliases actually matching the example shape this function
    builds. Picking a raw class name that expects a different shape (a label,
    a triplet, an explicit score) is on the caller -- it will fail loudly
    inside sentence-transformers' own ``fit``, not silently mis-train.
    """
    class_name = _TRAIN_LOSS_ALIASES.get(name, name)
    if class_name == "CachedMultipleNegativesRankingLoss":
        params = dict(params)
        params.setdefault("mini_batch_size", 32)
    try:
        loss_cls = getattr(losses, class_name)
    except AttributeError:
        raise ValueError(
            f"unknown encoder.train_loss {name!r}: no sentence_transformers.losses.{class_name}. "
            f"Built-in aliases: {sorted(_TRAIN_LOSS_ALIASES)}; any other "
            "sentence_transformers.losses class name also works if its installed "
            "version has one."
        ) from None
    return loss_cls(model, **params)


def train_encoder(
    items: Sequence[LabeledItem],
    label_space: LabelSpace,
    config: EncoderConfig,
    output_path: str | None = None,
) -> SentenceTransformerEncoder:
    """Fine-tune a bi-encoder so items sit near their class description.

    Uses ``config.train_loss`` (default MultipleNegativesSymmetricRankingLoss) on
    (item_text, description) pairs. NoDuplicatesDataLoader keeps two items of the
    same class out of one batch, which is what prevents the in-batch negatives
    from treating an item's own description as a negative for a same-class
    sibling.

    **Which epoch you get back.** With ``train_epochs == 1`` (the default), or
    ``train_holdout_ratio == 0``, this is the state after the final epoch: every
    item trains, nothing is measured, and there is nothing to choose between.
    Otherwise a stratified ``train_holdout_ratio`` of ``items`` is withheld from
    the gradient updates and re-scored after every epoch, and the epoch scoring
    best on ``train_select_metric`` is what is returned and saved — so raising
    ``train_epochs`` to 10 or 20 stops being a gamble on the last epoch being the
    good one. ``train_early_stopping_patience`` cuts the run short once the metric
    stalls. The per-epoch table travels with the encoder as ``training_history``
    and is written into the saved encoder directory.

    Leakage note: the holdout is carved out of the caller's ``items``, which in
    the out-of-fold loop are already one fold's *training* rows. Withheld from the
    gradient is still in-fold, so epoch selection never sees data the encoder was
    not entitled to, and the rows the fusion model trains on are untouched.
    """
    st = _require_sentence_transformers()
    InputExample, SentenceTransformer, losses = st.InputExample, st.SentenceTransformer, st.losses
    from sentence_transformers.datasets import NoDuplicatesDataLoader

    items = list(items)
    labels = np.asarray(label_space.encode_labels([it.label for it in items]), dtype=np.int64)
    descriptions = label_space.descriptions

    # Selection is only meaningful across >1 epoch; a single epoch keeps the
    # pre-selection behaviour exactly (all items train, no holdout).
    ratio = config.train_holdout_ratio if config.train_epochs > 1 else 0.0
    fit_idx, holdout_idx = _stratified_holdout(labels, ratio, config.train_holdout_seed)
    if 0 < len(holdout_idx) < MIN_EPOCH_HOLDOUT:
        logger.warning(
            "best-epoch selection disabled: train_holdout_ratio=%.3f over %d item(s) "
            "yields a %d-item holdout (< %d), too small to rank epochs by; keeping the "
            "last epoch and training on everything",
            config.train_holdout_ratio,
            len(items),
            len(holdout_idx),
            MIN_EPOCH_HOLDOUT,
        )
        fit_idx = np.arange(len(items), dtype=np.intp)
        holdout_idx = np.empty(0, dtype=np.intp)

    logger.info(
        "fine-tuning SentenceTransformer encoder on device=%s",
        resolve_device(config.device, mps_ok=True),
    )
    model = SentenceTransformer(config.model_name_or_path, device=config.device, **config.params)
    encode_options = _encode_options(config)
    encoder = SentenceTransformerEncoder(model, config.encode_batch_size, **encode_options)

    examples = [InputExample(texts=[items[i].text, descriptions[int(labels[i])]]) for i in fit_idx]
    loader = NoDuplicatesDataLoader(examples, batch_size=config.train_batch_size)
    loss = _build_train_loss(config.train_loss, model, losses, config.train_loss_params)
    steps_per_epoch = max(1, len(loader))
    warmup_steps = int(steps_per_epoch * config.train_epochs * config.warmup_ratio)
    objectives = [(loader, loss)]

    if len(holdout_idx) == 0:
        model.fit(
            train_objectives=objectives,
            epochs=config.train_epochs,
            warmup_steps=warmup_steps,
            output_path=output_path,
            show_progress_bar=False,
        )
        return encoder

    policy = EpochSelectionPolicy(
        metric=config.train_select_metric,
        min_delta=config.train_select_min_delta,
        patience=config.train_early_stopping_patience,
    )
    # The nearest-example metric needs the fine-tuning pool re-encoded every epoch;
    # only pay for that when it is what we are selecting on. The pool deliberately
    # excludes the holdout items, or each would retrieve itself as its own neighbor.
    pool_texts = pool_labels = None
    if policy.metric.startswith("knn"):
        pool_texts = [items[i].text for i in fit_idx]
        pool_labels = labels[fit_idx]

    snapshot_dir = tempfile.mkdtemp(prefix="tc-encoder-epoch-")
    try:
        tracker = EncoderEpochTracker(
            encoder,
            [items[i].text for i in holdout_idx],
            labels[holdout_idx],
            descriptions,
            policy,
            snapshot=lambda: model.save(snapshot_dir),
            pool_texts=pool_texts,
            pool_labels=pool_labels,
        )
        logger.info(
            "fine-tuning %d epoch(s) on %d item(s); selecting the best epoch by %s "
            "on a %d-item holdout",
            config.train_epochs,
            len(fit_idx),
            policy.metric,
            len(holdout_idx),
        )
        early_stopped = False
        try:
            model.fit(
                train_objectives=objectives,
                epochs=config.train_epochs,
                warmup_steps=warmup_steps,
                evaluator=_epoch_evaluator(tracker),
                evaluation_steps=0,  # exactly one evaluator call per epoch
                save_best_model=False,  # we select and persist the best epoch ourselves
                output_path=None,  # written below, after the best epoch is restored
                show_progress_bar=False,
            )
        except _StopFineTuning as stop:
            early_stopped = True
            logger.info("stopped fine-tuning after epoch %d: %s", len(tracker.history), stop)

        selected = _select_epoch(tracker, encoder, snapshot_dir, config, encode_options)
        selected.training_history = {
            **tracker.report(),
            "n_fit_items": int(len(fit_idx)),
            "epochs_requested": config.train_epochs,
            "epochs_run": len(tracker.history),
            "early_stopped": early_stopped,
        }
    finally:
        shutil.rmtree(snapshot_dir, ignore_errors=True)

    if output_path:
        selected.save(output_path)
    return selected


def _select_epoch(
    tracker: EncoderEpochTracker,
    trained: SentenceTransformerEncoder,
    snapshot_dir: str,
    config: EncoderConfig,
    encode_options: Dict[str, Any],
) -> SentenceTransformerEncoder:
    """Return the encoder for the best epoch, reloading the snapshot if needed.

    ``fit`` leaves the in-memory model on the epoch it last ran, so when that is
    also the best epoch (the common "still improving at the end" case, and the
    case where nothing was scoreable) there is nothing to do. Otherwise the best
    epoch's weights are on disk and are reloaded from there.
    """
    from sentence_transformers import SentenceTransformer

    best, last = tracker.best_epoch, len(tracker.history)
    if best == 0:
        logger.warning(
            "no epoch produced a usable %s score; keeping the last epoch",
            config.train_select_metric,
        )
        return trained
    if best == last:
        logger.info("best epoch is the last one (%d); keeping the trained model", best)
        return trained
    logger.info(
        "restoring epoch %d of %d (best %s=%.4f; last epoch scored %.4f)",
        best,
        last,
        config.train_select_metric,
        tracker.history[best - 1][config.train_select_metric],
        tracker.history[last - 1][config.train_select_metric],
    )
    return SentenceTransformerEncoder(
        SentenceTransformer(snapshot_dir, device=config.device, **config.params),
        config.encode_batch_size,
        **encode_options,
    )
