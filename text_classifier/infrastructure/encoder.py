"""Encoder adapters behind the TextEncoder port.

- ``SentenceTransformerEncoder``: wraps a SentenceTransformer (torch + a model
  download) and supports fine-tuning over (item, class-description) pairs with
  a pluggable loss (``EncoderConfig.train_loss``, default
  MultipleNegativesSymmetricRankingLoss; see ``domain.services.ENCODER_LOSSES``)
  through ``SentenceTransformerTrainer``. Each epoch is scored on a held-out slice
  by ``InformationRetrievalEvaluator``, so a multi-epoch run keeps its *best* epoch
  rather than its last. Pooling and truncation length are configurable
  (``EncoderConfig.pooling_mode`` / ``max_seq_length``).
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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import EncoderConfig
from ..domain import ENCODER_SELECTION_METRICS, LabeledItem, LabelSpace, TextEncoder
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
        apply_model_overrides: bool = False,
        **kwargs,
    ) -> "SentenceTransformerEncoder":
        """Load a model by name or from a saved directory.

        ``apply_model_overrides`` applies ``config.pooling_mode`` and
        ``config.max_seq_length`` to the loaded model. It is for *building* an
        encoder from a pretrained checkpoint, and is off by default because this
        method also reloads a model dir this package saved, which already carries
        the pooling it was trained with and the sequence length it was saved at."""
        SentenceTransformer = _require_sentence_transformers().SentenceTransformer

        # device=None lets SentenceTransformer do its own auto-detection (which,
        # in current sentence-transformers, checks MPS too); resolve_device here
        # is only to log what that resolves to, so mps_ok=True keeps the log
        # line accurate on Apple Silicon rather than always claiming "cpu".
        logger.info(
            "loading SentenceTransformer encoder on device=%s",
            resolve_device(device, mps_ok=True),
        )
        model = SentenceTransformer(model_name_or_path, device=device, **kwargs)
        if apply_model_overrides and config is not None:
            model = _apply_model_overrides(model, config)
        return cls(model, batch_size, **_encode_options(config))

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


# ------------------------------------------------------------ model construction
def _import_first(candidates: Sequence[str], name: str) -> Any:
    """``name`` from the first of ``candidates`` (module paths) that provides it.

    sentence-transformers has moved its evaluators and training arguments between
    releases (the old paths still work in 6.x but warn that they are deprecated),
    so the newest location is tried first -- no deprecation warning on a current
    install, and the old path on an older one."""
    import importlib

    for module in candidates:
        try:
            return getattr(importlib.import_module(module), name)
        except (ImportError, AttributeError):
            continue
    raise ImportError(
        f"could not import {name!r} from sentence-transformers (tried {list(candidates)}); "
        "fine-tuning needs sentence-transformers>=3.0 -- "
        "pip install -U 'text-classifier[sentence-transformers]'"
    )


def _pooling_mode_of(pooling: Any) -> str:
    """The mode string of a ``Pooling`` module, across sentence-transformers
    releases: older ones expose ``get_pooling_mode_str()`` (booleans underneath),
    6.x a plain ``pooling_mode`` attribute (a string, or a tuple when several modes
    are concatenated). ``"?"`` if neither is there."""
    getter = getattr(pooling, "get_pooling_mode_str", None)
    if callable(getter):
        return str(getter())
    mode = getattr(pooling, "pooling_mode", None)
    if isinstance(mode, (list, tuple)):
        return "+".join(str(m) for m in mode)
    return str(mode) if mode is not None else "?"


def _embedding_dim(module: Any) -> Optional[int]:
    """A Transformer module's output width, or ``None`` if it has no such notion.
    6.x renamed ``get_word_embedding_dimension`` to ``get_embedding_dimension``."""
    for name in ("get_embedding_dimension", "get_word_embedding_dimension"):
        getter = getattr(module, name, None)
        if callable(getter):
            return int(getter())
    return None


def _st_module(new: str, old: str) -> Any:
    """A sentence-transformers submodule, preferring its newest location.

    Releases since 5.x have moved ``models``/``losses`` under
    ``sentence_transformers.sentence_transformer`` and deprecate the old paths
    (still importable, but they warn on every import); 3.x/4.x only have the old
    ones."""
    import importlib

    for module in (new, old):
        try:
            return importlib.import_module(module)
        except ImportError:
            continue
    raise ImportError(f"could not import {old} (also tried {new})")


def _losses_module() -> Any:
    """``sentence_transformers.losses``, from wherever this release keeps it."""
    return _st_module(
        "sentence_transformers.sentence_transformer.losses", "sentence_transformers.losses"
    )


def _models_module() -> Any:
    """The Transformer/Pooling/Normalize module classes (``models`` in older
    releases, ``sentence_transformer.modules`` in newer ones)."""
    return _st_module(
        "sentence_transformers.sentence_transformer.modules", "sentence_transformers.models"
    )


def _with_pooling(model: "SentenceTransformer", mode: str) -> "SentenceTransformer":
    """Rebuild ``model`` as its first (Transformer) module plus ``Pooling(mode)``.

    Anything the model had *after* its pooling -- a ``Dense`` projection, a
    ``Normalize`` layer -- is dropped, because it was sized for the old pooling's
    output and has no meaning on the new one; both that and replacing a pooling
    the model already shipped with are logged, as they change what the model
    computes before any fine-tuning has had a chance to adapt it."""
    from sentence_transformers import SentenceTransformer

    models = _models_module()
    modules = list(model)
    first = modules[0]
    width = _embedding_dim(first)
    if width is None:
        raise ValueError(
            f"encoder.pooling_mode={mode!r} needs a model whose first module is a "
            f"Transformer (token embeddings to pool); this one starts with "
            f"{type(first).__name__}. Leave pooling_mode unset for it."
        )
    previous = [m for m in modules if isinstance(m, models.Pooling)]
    dropped = [type(m).__name__ for m in modules[1:] if not isinstance(m, models.Pooling)]
    if previous:
        current = _pooling_mode_of(previous[0])
        if current != mode:
            logger.warning(
                "encoder.pooling_mode=%r replaces the model's own pooling (%s); quality "
                "will not match the original until the encoder is fine-tuned",
                mode,
                current,
            )
    if dropped:
        logger.warning(
            "encoder.pooling_mode=%r rebuilds the model as Transformer + Pooling and "
            "drops its post-pooling module(s): %s",
            mode,
            ", ".join(dropped),
        )
    try:
        pooling = models.Pooling(width, pooling_mode=mode)
    except (ValueError, TypeError, AssertionError) as exc:
        import sentence_transformers

        raise ValueError(
            f"encoder.pooling_mode={mode!r} is not supported by the installed "
            f"sentence-transformers {sentence_transformers.__version__}: {exc}"
        ) from exc
    rebuilt = SentenceTransformer(modules=[first, pooling], device=str(model.device))
    # The model-level settings live outside the modules (they come from
    # config_sentence_transformers.json); carry them over so `query_prompt_name` /
    # `document_prompt_name` keep resolving against the model's own prompts.
    for attr in ("prompts", "default_prompt_name"):
        if hasattr(model, attr):
            setattr(rebuilt, attr, getattr(model, attr))
    return rebuilt


def _set_max_seq_length(model: "SentenceTransformer", max_seq_length: int) -> None:
    """Set the model's truncation length, warning when it exceeds what the
    underlying transformer's position embeddings can address."""
    limit = getattr(
        getattr(getattr(model[0], "auto_model", None), "config", None),
        "max_position_embeddings",
        None,
    )
    if isinstance(limit, int) and max_seq_length > limit:
        logger.warning(
            "encoder.max_seq_length=%d exceeds the model's max_position_embeddings=%d; "
            "encoding longer texts is likely to fail",
            max_seq_length,
            limit,
        )
    model.max_seq_length = max_seq_length


def _apply_model_overrides(model: "SentenceTransformer", config: EncoderConfig):
    """Apply ``EncoderConfig.pooling_mode`` / ``max_seq_length`` to a freshly
    loaded model. Both default to ``None``, which changes nothing."""
    if config.pooling_mode is not None:
        model = _with_pooling(model, config.pooling_mode)
    if config.max_seq_length is not None:
        _set_max_seq_length(model, config.max_seq_length)
    return model


# ------------------------------------------------------------------- fine-tuning
def _selection_key(metric: str, n_classes: int) -> str:
    """The Trainer metric name an ``ENCODER_SELECTION_METRICS`` entry is reported
    under: ``eval_`` + the evaluator's ``name`` + ``_cosine_`` + the IR metric.
    ``desc_mrr`` is read at ``k = n_classes`` so it is a full-ranking MRR, not
    one truncated at the evaluator's default depth of 10."""
    return {
        "desc_acc@1": "eval_desc_cosine_accuracy@1",
        "desc_mrr": f"eval_desc_cosine_mrr@{n_classes}",
        "knn_acc@1": "eval_knn_cosine_accuracy@1",
    }[metric]


def _build_epoch_evaluator(
    config: EncoderConfig,
    texts: Sequence[str],
    labels: np.ndarray,
    fit_idx: np.ndarray,
    holdout_idx: np.ndarray,
    descriptions: Sequence[str],
    *,
    with_knn: bool,
) -> Any:
    """The per-epoch scorer: ``InformationRetrievalEvaluator``\\ s over the holdout.

    ``desc`` -- queries are the held-out items, the corpus is the class
    descriptions, and the one relevant document is the item's own class: its
    ``accuracy@1`` is "nearest description is the true class" and its ``mrr@C`` is
    the full-rank reciprocal rank. ``knn`` (only when selecting on it, since it
    re-encodes the fine-tuning pool every epoch) uses the *fit* items as the
    corpus, with every pool item of the query's class relevant: ``accuracy@1`` is
    then "nearest labeled example shares the label". The pool excludes the
    holdout by construction, or each query would retrieve itself.

    Prompts follow the encoder's own query/document roles (queries are the items,
    documents the descriptions and examples), so what is scored is what the
    deployed encoder will actually compute.
    """
    from sentence_transformers.util import cos_sim

    IR = _import_first(
        (
            "sentence_transformers.sentence_transformer.evaluation",
            "sentence_transformers.evaluation",
        ),
        "InformationRetrievalEvaluator",
    )
    Sequential = _import_first(
        (
            "sentence_transformers.sentence_transformer.evaluation",
            "sentence_transformers.evaluation",
        ),
        "SequentialEvaluator",
    )

    extra: Dict[str, Any] = {}
    q_prompt, q_name = SentenceTransformerEncoder._effective_prompt(
        config.query_prompt, config.query_prompt_name
    )
    d_prompt, d_name = SentenceTransformerEncoder._effective_prompt(
        config.document_prompt, config.document_prompt_name
    )
    for key, value in (
        ("query_prompt", q_prompt),
        ("query_prompt_name", q_name),
        ("corpus_prompt", d_prompt),
        ("corpus_prompt_name", d_name),
        ("truncate_dim", config.encode_kwargs.get("truncate_dim")),
    ):
        if value is not None:
            extra[key] = value

    def evaluator(name, queries, corpus, relevant, mrr_k):
        return IR(
            queries,
            corpus,
            relevant,
            name=name,
            accuracy_at_k=[1],
            precision_recall_at_k=[1],
            mrr_at_k=[mrr_k],
            ndcg_at_k=[1],
            map_at_k=[1],
            score_functions={"cosine": cos_sim},
            main_score_function="cosine",
            batch_size=config.encode_batch_size,
            show_progress_bar=False,
            write_csv=False,
            **extra,
        )

    queries = {f"q{k}": texts[int(i)] for k, i in enumerate(holdout_idx)}
    holdout_labels = [int(labels[int(i)]) for i in holdout_idx]

    desc_corpus = {f"c{c}": descriptions[c] for c in range(len(descriptions))}
    desc_relevant = {f"q{k}": {f"c{lab}"} for k, lab in enumerate(holdout_labels)}
    evaluators = [evaluator("desc", queries, desc_corpus, desc_relevant, len(descriptions))]

    if with_knn:
        pool = {f"p{k}": texts[int(i)] for k, i in enumerate(fit_idx)}
        by_class: Dict[int, set] = {}
        for k, i in enumerate(fit_idx):
            by_class.setdefault(int(labels[int(i)]), set()).add(f"p{k}")
        # One shared set per class, not one per query: the evaluator only reads
        # them, and a copy per held-out item would cost O(holdout * pool / classes).
        knn_relevant = {f"q{k}": by_class.get(lab, set()) for k, lab in enumerate(holdout_labels)}
        evaluators.append(evaluator("knn", queries, pool, knn_relevant, 1))

    return evaluators[0] if len(evaluators) == 1 else Sequential(evaluators)


def _warmup_kwargs(ratio: float) -> Dict[str, Any]:
    """The Trainer argument carrying a warmup *fraction* of the total steps.

    transformers 5 deprecates ``warmup_ratio`` in favour of a float
    ``warmup_steps``; transformers 4 only understands the former."""
    try:
        import transformers

        major = int(transformers.__version__.split(".")[0])
    except Exception:
        major = 4
    return {"warmup_steps": ratio} if major >= 5 else {"warmup_ratio": ratio}


def _training_arguments(
    config: EncoderConfig, output_dir: str, select_metric_key: Optional[str]
) -> Any:
    """``SentenceTransformerTrainingArguments`` for one fine-tune.

    ``batch_sampler=NO_DUPLICATES`` keeps two items of one class out of the same
    batch (they share the description text, which is what the sampler detects), so
    an in-batch negative is never an item's own description. With
    ``select_metric_key`` the Trainer evaluates and checkpoints once per epoch and
    reloads the best one at the end; without it nothing is evaluated or written
    and the last epoch is kept."""
    import dataclasses

    st = _require_sentence_transformers()
    BatchSamplers = _import_first(
        (
            "sentence_transformers.sentence_transformer.training_args",
            "sentence_transformers.training_args",
        ),
        "BatchSamplers",
    )
    args_cls = st.SentenceTransformerTrainingArguments
    names = {f.name for f in dataclasses.fields(args_cls)}

    def first_present(*candidates: str) -> str:
        return next(c for c in candidates if c in names)

    kwargs: Dict[str, Any] = {
        "output_dir": output_dir,
        "num_train_epochs": config.train_epochs,
        "per_device_train_batch_size": config.train_batch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "batch_sampler": BatchSamplers.NO_DUPLICATES,
        "report_to": "none",  # nothing is sent anywhere: the host may be air-gapped
        "disable_tqdm": True,
        "logging_strategy": "no",
        **_warmup_kwargs(config.warmup_ratio),
    }
    if config.device == "cpu":
        kwargs["use_cpu"] = True
    if select_metric_key is None:
        kwargs[first_present("eval_strategy", "evaluation_strategy")] = "no"
        kwargs["save_strategy"] = "no"
    else:
        kwargs.update(
            {
                first_present("eval_strategy", "evaluation_strategy"): "epoch",
                "save_strategy": "epoch",
                "load_best_model_at_end": True,
                "metric_for_best_model": select_metric_key,
                "greater_is_better": True,
                "save_total_limit": 1,
                # Skip optimizer/scheduler/RNG state: those are torch pickles, and
                # the checkpoint only ever needs the weights (it lives in a temp
                # dir that is removed; the model dir is written by model.save()).
                "save_only_model": True,
            }
        )
    return args_cls(**kwargs)


def _device_notes(config: EncoderConfig) -> None:
    """Warn about the two ways the Trainer's device handling differs from
    ``SentenceTransformer(device=...)``, which this package used to rely on."""
    device = config.device
    if device and device.startswith("cuda:"):
        logger.warning(
            "encoder.device=%r: the Trainer places the model itself and does not honour a "
            "CUDA device index; set CUDA_VISIBLE_DEVICES to pin a specific GPU",
            device,
        )
    elif device != "cpu":
        try:
            import torch

            n_gpus = torch.cuda.device_count()
        except Exception:
            n_gpus = 0
        if n_gpus > 1:
            logger.warning(
                "%d CUDA devices are visible and the Trainer will use all of them "
                "(DataParallel), which shrinks each replica's batch of in-batch "
                "negatives; set CUDA_VISIBLE_DEVICES to a single GPU to avoid that",
                n_gpus,
            )


def _log_epoch_callback() -> Any:
    """A ``TrainerCallback`` that logs each epoch's held-out scores (the line the
    old per-epoch tracker used to emit). Defined inside the function so importing
    this module stays torch-free."""
    from transformers import TrainerCallback

    class _LogEpoch(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            shown = {
                k[len("eval_") :]: v
                for k, v in (metrics or {}).items()
                if k.startswith("eval_") and "cosine" in k
            }
            logger.info(
                "encoder epoch %s: %s",
                int(round(state.epoch or 0)),
                ", ".join(f"{k}={v:.4f}" for k, v in sorted(shown.items())),
            )

    return _LogEpoch()


def _training_history(
    trainer: Any, config: EncoderConfig, n_classes: int, n_fit: int, n_holdout: int
) -> Dict[str, Any]:
    """JSON-clean record of a selected fine-tune, read off the Trainer's log.

    The per-epoch table is keyed by this package's metric names
    (``ENCODER_SELECTION_METRICS``), not the evaluator's. ``best_epoch`` is the
    epoch whose checkpoint the Trainer reloaded; ties go to the earlier epoch
    because the Trainer only replaces its best on a strict improvement."""
    evals = [
        entry
        for entry in trainer.state.log_history
        if any(key.startswith("eval_") for key in entry)
    ]
    epochs: List[Dict[str, Any]] = []
    for number, entry in enumerate(evals, start=1):
        record: Dict[str, Any] = {"epoch": number}
        for metric in ENCODER_SELECTION_METRICS:
            key = _selection_key(metric, n_classes)
            if key in entry:
                record[metric] = float(entry[key])
        epochs.append(record)

    best_epoch = 0
    checkpoint = trainer.state.best_model_checkpoint
    if checkpoint:
        try:
            step = int(os.path.basename(os.path.normpath(checkpoint)).rsplit("-", 1)[-1])
            best_epoch = next(
                (n for n, entry in enumerate(evals, start=1) if entry.get("step") == step), 0
            )
        except ValueError:
            best_epoch = 0
    if best_epoch == 0 and epochs:
        scores = [e.get(config.train_select_metric, float("nan")) for e in epochs]
        best_epoch = int(np.nanargmax(scores)) + 1 if not np.all(np.isnan(scores)) else 0

    return {
        "select_metric": config.train_select_metric,
        "select_min_delta": config.train_select_min_delta,
        "early_stopping_patience": config.train_early_stopping_patience,
        "n_holdout_items": n_holdout,
        "best_epoch": best_epoch,
        "epochs": epochs,
        "n_fit_items": n_fit,
        "epochs_requested": config.train_epochs,
        "epochs_run": len(epochs),
        "early_stopped": len(epochs) < config.train_epochs,
    }


def train_encoder(
    items: Sequence[LabeledItem],
    label_space: LabelSpace,
    config: EncoderConfig,
    output_path: str | None = None,
) -> SentenceTransformerEncoder:
    """Fine-tune a bi-encoder so items sit near their class description.

    Runs a ``SentenceTransformerTrainer`` with ``config.train_loss`` (default
    MultipleNegativesSymmetricRankingLoss) on (item_text, description) pairs,
    ``config.learning_rate`` / ``weight_decay``, and the ``NO_DUPLICATES`` batch
    sampler, which keeps two items of the same class out of one batch -- what
    prevents the in-batch negatives from treating an item's own description as a
    negative for a same-class sibling.

    **Which epoch you get back.** With ``train_epochs == 1`` (the default), or
    ``train_holdout_ratio == 0``, this is the state after the final epoch: every
    item trains, nothing is measured, and there is nothing to choose between.
    Otherwise a stratified ``train_holdout_ratio`` of ``items`` is withheld from
    the gradient updates and scored after every epoch by
    ``InformationRetrievalEvaluator``; the Trainer reloads the best epoch's
    weights at the end (``load_best_model_at_end``), so raising ``train_epochs``
    to 10 or 20 stops being a gamble on the last epoch being the good one.
    ``train_early_stopping_patience`` cuts the run short once the metric stalls
    (``train_select_min_delta`` is the improvement it must beat to count as
    progress -- it does not change which epoch is *picked*, only when to give up).
    The per-epoch table travels with the encoder as ``training_history`` and is
    written into the saved encoder directory.

    Leakage note: the holdout is carved out of the caller's ``items``, which in
    the out-of-fold loop are already one fold's *training* rows. Withheld from the
    gradient is still in-fold, so epoch selection never sees data the encoder was
    not entitled to, and the rows the fusion model trains on are untouched.
    """
    st = _require_sentence_transformers()
    if not hasattr(st, "SentenceTransformerTrainer"):
        raise ImportError(
            f"fine-tuning needs sentence-transformers>=3.0 (found {st.__version__}): "
            "pip install -U 'text-classifier[sentence-transformers]'"
        )
    from datasets import Dataset
    from transformers import EarlyStoppingCallback

    losses = _losses_module()

    items = list(items)
    texts = [it.text for it in items]
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
    selecting = len(holdout_idx) > 0

    logger.info(
        "fine-tuning SentenceTransformer encoder on device=%s",
        resolve_device(config.device, mps_ok=True),
    )
    _device_notes(config)
    model = _apply_model_overrides(
        st.SentenceTransformer(config.model_name_or_path, device=config.device, **config.params),
        config,
    )
    encode_options = _encode_options(config)
    encoder = SentenceTransformerEncoder(model, config.encode_batch_size, **encode_options)

    train_dataset = Dataset.from_dict(
        {
            "anchor": [texts[int(i)] for i in fit_idx],
            "positive": [descriptions[int(labels[int(i)])] for i in fit_idx],
        }
    )
    loss = _build_train_loss(config.train_loss, model, losses, config.train_loss_params)

    evaluator = None
    callbacks: List[Any] = []
    select_key = None
    if selecting:
        select_key = _selection_key(config.train_select_metric, len(descriptions))
        evaluator = _build_epoch_evaluator(
            config,
            texts,
            labels,
            fit_idx,
            holdout_idx,
            descriptions,
            with_knn=config.train_select_metric.startswith("knn"),
        )
        callbacks.append(_log_epoch_callback())
        if config.train_early_stopping_patience > 0:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=config.train_early_stopping_patience,
                    early_stopping_threshold=config.train_select_min_delta,
                )
            )
        logger.info(
            "fine-tuning %d epoch(s) on %d item(s); selecting the best epoch by %s "
            "on a %d-item holdout",
            config.train_epochs,
            len(fit_idx),
            config.train_select_metric,
            len(holdout_idx),
        )

    work_dir = tempfile.mkdtemp(prefix="tc-encoder-train-")
    try:
        trainer = st.SentenceTransformerTrainer(
            model=model,
            args=_training_arguments(config, work_dir, select_key),
            train_dataset=train_dataset,
            loss=loss,
            evaluator=evaluator,
            callbacks=callbacks,
        )
        trainer.train()
        if selecting:
            history = _training_history(
                trainer, config, len(descriptions), len(fit_idx), len(holdout_idx)
            )
            encoder.training_history = history
            if history["early_stopped"]:
                logger.info(
                    "stopped fine-tuning after epoch %d of %d: no improvement in %s for %d "
                    "epoch(s)",
                    history["epochs_run"],
                    config.train_epochs,
                    config.train_select_metric,
                    config.train_early_stopping_patience,
                )
            logger.info(
                "kept epoch %d of %d (%s)",
                history["best_epoch"],
                history["epochs_run"],
                config.train_select_metric,
            )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    if output_path:
        encoder.save(output_path)
    return encoder
