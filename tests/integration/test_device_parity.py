"""T85 — device-resident dense retrieval: parity, transfers, portability.

Runs the torch backend on **CPU**, which is the point: it exercises the same
code path a GPU run takes (device-resident state, one implementation of every
kernel) on a CI host with no GPU. What it cannot cover is CUDA-specific
arithmetic, so read the claims here as "the backend seam is correct", not "the
GPU numbers are verified" — that needs the GPU-host re-run
``docs/device-policy.md`` already flags as required.

The parity claim is deliberately **tolerance-based, not bit-identity** (the
decision recorded in the ticket): float32 reduction order differs between
backends, so continuous columns move in the last ulps and — the part that
matters — an exact or near tie can flip an ordinal column (``rank_*``,
``is_*_top1``) and with it the candidate set. So:

  - continuous columns are compared with ``allclose`` at a stated tolerance;
  - ordinal columns are compared **excluding rows whose underlying signal is
    within tolerance of a tie**, which is checked against the frames' own raw
    values rather than assumed. A test tuned until it passes would prove
    nothing; this one states the exemption up front and verifies it.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from text_classifier.application.features import FeatureAssembler
from text_classifier.application.inference import InferencePipeline
from text_classifier.application.training import TrainingPipeline
from text_classifier.config import FusionConfig, PipelineConfig, RetrievalConfig, TrainingConfig
from text_classifier.datasets import make_synthetic
from text_classifier.domain import CandidatePolicy
from text_classifier.infrastructure import (
    ArtifactRepository,
    DenseRetrieverAdapter,
    HashingEncoder,
    LexicalRetrieverAdapter,
)
from text_classifier.infrastructure.array_ops import NumpyArrayOps

pytest.importorskip("torch", reason="the torch array backend needs the 'gpu' extra")

import torch  # noqa: E402

from text_classifier.infrastructure.array_ops import TorchArrayOps  # noqa: E402

# Tolerance for the continuous columns. float32 similarities differ in the last
# ulp or two between reduction orders; 1e-5 is comfortably above that and
# comfortably below anything that would change a decision.
TOL = 1e-5

# Signal-value gap below which two candidates count as tied for ordinal
# purposes. Same order as TOL: a pair this close can legitimately swap.
TIE_TOL = 1e-5

ORDINAL_COLUMNS = {
    "rank_d_desc": "d_desc_sim",
    "rank_b_desc": "b_desc_sim",
    "rank_d_knn": "d_knn_sum",
    "rank_b_knn": "b_knn_sum",
    "is_d_desc_top1": "d_desc_sim",
    "is_d_proto_top1": "d_proto_sim",
    "is_b_desc_top1": "b_desc_sim",
    "is_d_knn_top1": "d_knn_sum",
    "is_b_knn_top1": "b_knn_sum",
}


# --------------------------------------------------------------------- helpers
def _corpus(n_classes=6, per_class=8, seed=5):
    label_space, items = make_synthetic(n_classes=n_classes, per_class=per_class, seed=seed)
    texts = [it.text for it in items]
    y = np.array(label_space.encode_labels([it.label for it in items]), dtype=np.int64)
    return label_space, texts, y


def _assemble(ops, label_space, texts, y, emb, desc_emb, q_texts, q_emb, chunk=5):
    cfg = RetrievalConfig()
    dense = DenseRetrieverAdapter.build_from_embeddings(
        emb, y, desc_emb, label_space, cfg, ops
    )
    lexical = LexicalRetrieverAdapter.build(texts, y, label_space, cfg)
    assembler = FeatureAssembler(label_space, CandidatePolicy(top_n_per_signal=3), ops)
    return assembler.assemble(
        q_texts,
        ops.asarray(q_emb),
        dense,
        lexical,
        k_neighbors=4,
        query_ids=list(range(len(q_texts))),
        query_labels=y[: len(q_texts)],
        chunk=chunk,
    )


def _untied_embeddings(n, dim, seed):
    """L2-normalized random embeddings. Random continuous vectors have no exact
    ties and (with overwhelming probability) no near-ties either, which is what
    lets the strict comparison below be strict."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, dim)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _keyed(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.set_index(["item_id", "candidate"]).sort_index()


def _min_gap_per_query(frame: pd.DataFrame, value_col: str) -> pd.Series:
    """Smallest gap between any two candidates' values within each query — how
    close that query came to a tie on this signal."""

    def gap(values):
        v = np.sort(values[~np.isnan(values)])
        return np.min(np.diff(v)) if v.size > 1 else np.inf

    return frame.groupby("item_id")[value_col].apply(lambda s: gap(s.to_numpy()))


def _tie_exempt(base: pd.DataFrame, keys, col: str, value_col: str) -> np.ndarray:
    """Which rows of ``keys`` this ordinal column is allowed to disagree on.

    Two cases, both genuine ties rather than convenient exclusions:

    1. **The signal did not retrieve this candidate** (value is NaN). ``_row_rank``
       maps NaN to ``-inf``, so every unretrieved candidate in a query ties with
       every other, and their relative order is whatever the sort happened to
       do. That is already unspecified *within* one backend — the numpy path
       uses an unstable quicksort — so it cannot be a cross-backend guarantee.
       (A ``rank_*`` value on a NaN row carries no information: the paired
       ``*_missing`` / NaN raw column is what the model reads.)
    2. **The query has two candidates within ``TIE_TOL``** on this signal, so a
       last-ulp difference can legitimately swap them.

    Everything else must agree exactly.
    """
    raw = base.set_index(["item_id", "candidate"]).sort_index().loc[keys, value_col].to_numpy()
    gaps = _min_gap_per_query(base, value_col)
    item_ids = keys.get_level_values("item_id").to_numpy()
    near_tie = np.array([gaps.get(i, np.inf) <= TIE_TOL for i in item_ids])
    return np.isnan(raw.astype(np.float64)) | near_tie


def _assert_ordinals_agree(base: pd.DataFrame, ka: pd.DataFrame, kb: pd.DataFrame, keys) -> None:
    for col, value_col in ORDINAL_COLUMNS.items():
        left, right = ka.loc[keys, col].to_numpy(), kb.loc[keys, col].to_numpy()
        same = (left == right) | (np.isnan(left) & np.isnan(right))
        exempt = _tie_exempt(base, keys, col, value_col)
        bad = np.nonzero(~same & ~exempt)[0]
        assert bad.size == 0, (
            f"{col} differs between backends on {bad.size} row(s) that are neither "
            f"unretrieved nor within {TIE_TOL} of a tie in {value_col}: "
            f"{[tuple(keys[i]) for i in bad[:5]]}"
        )


# ------------------------------------------------------- the default path first
def test_numpy_backend_is_the_untouched_default():
    """The explicit numpy backend and the default (no ``array_ops`` argument)
    are the same object graph and the same bytes — T85 changed where kernels
    *can* run, not what the default path does."""
    label_space, texts, y = _corpus()
    enc = HashingEncoder(dim=64)
    emb, desc_emb = enc.encode(texts), enc.encode(label_space.descriptions)
    q_texts, q_emb = texts[:11], enc.encode(texts[:11])

    explicit = _assemble(NumpyArrayOps(), label_space, texts, y, emb, desc_emb, q_texts, q_emb)
    cfg = RetrievalConfig()
    dense = DenseRetrieverAdapter.build_from_embeddings(emb, y, desc_emb, label_space, cfg)
    lexical = LexicalRetrieverAdapter.build(texts, y, label_space, cfg)
    default = FeatureAssembler(label_space, CandidatePolicy(top_n_per_signal=3)).assemble(
        q_texts,
        q_emb,
        dense,
        lexical,
        k_neighbors=4,
        query_ids=list(range(len(q_texts))),
        query_labels=y[: len(q_texts)],
        chunk=5,
    )
    pd.testing.assert_frame_equal(explicit, default)


# ------------------------------------------------------------------- parity
def test_torch_cpu_matches_numpy_on_untied_data():
    """The strict half: on data with no ties, the two backends must agree on
    the candidate set exactly, on every ordinal column exactly, and on every
    continuous column to ``TOL``."""
    label_space, texts, y = _corpus(n_classes=7, per_class=9, seed=11)
    emb = _untied_embeddings(len(texts), 32, seed=1)
    desc_emb = _untied_embeddings(label_space.size, 32, seed=2)
    q_texts = texts[:12]
    q_emb = _untied_embeddings(len(q_texts), 32, seed=3)

    a = _assemble(NumpyArrayOps(), label_space, texts, y, emb, desc_emb, q_texts, q_emb)
    b = _assemble(TorchArrayOps("cpu"), label_space, texts, y, emb, desc_emb, q_texts, q_emb)

    assert list(a.columns) == list(b.columns)
    ka, kb = _keyed(a), _keyed(b)
    assert list(ka.index) == list(kb.index), "the candidate set must not move on untied data"
    for col in ka.columns:
        got, want = kb[col].to_numpy(), ka[col].to_numpy()
        np.testing.assert_array_equal(
            np.isnan(got.astype(np.float64)),
            np.isnan(want.astype(np.float64)),
            err_msg=f"{col}: NaN pattern differs -- 'did not retrieve' must not move",
        )
        if col in ORDINAL_COLUMNS:
            continue  # checked below, with the tie exemption stated
        np.testing.assert_allclose(
            got, want, rtol=TOL, atol=TOL, equal_nan=True, err_msg=f"{col} differs"
        )
    _assert_ordinals_agree(a, ka, kb, ka.index)
    # ...and guard against the tie exemption quietly hollowing this test out.
    # The dense signals are the ones the torch backend actually recomputes, and
    # on untied data they retrieve every candidate, so *no* row of their ordinal
    # columns may be exempt: those three comparisons above were exact.
    for col, value_col in (
        ("rank_d_desc", "d_desc_sim"),
        ("is_d_desc_top1", "d_desc_sim"),
        ("is_d_proto_top1", "d_proto_sim"),
    ):
        assert not _tie_exempt(a, ka.index, col, value_col).any(), (
            f"{col} should have no exempt rows on untied data"
        )


def test_ordinal_divergences_are_confined_to_near_ties():
    """The honest half, on data that *is* full of ties (the hashing encoder over
    short synthetic texts produces plenty). Every ordinal disagreement must be
    explained by a near-tie in that query's own values for the signal the column
    ranks — not waved away."""
    label_space, texts, y = _corpus(n_classes=6, per_class=8, seed=5)
    enc = HashingEncoder(dim=64)
    emb, desc_emb = enc.encode(texts), enc.encode(label_space.descriptions)
    q_texts, q_emb = texts[:16], enc.encode(texts[:16])

    a = _assemble(NumpyArrayOps(), label_space, texts, y, emb, desc_emb, q_texts, q_emb)
    b = _assemble(TorchArrayOps("cpu"), label_space, texts, y, emb, desc_emb, q_texts, q_emb)

    ka, kb = _keyed(a), _keyed(b)
    shared = ka.index.intersection(kb.index)
    assert len(shared) > 0.5 * len(ka), "the two backends must agree on most of the grid"

    _assert_ordinals_agree(a, ka, kb, shared)

    # Continuous columns still have to agree everywhere on the shared grid.
    for col in ("d_desc_sim", "d_proto_sim", "d_knn_sum", "b_desc_sim", "b_knn_sum"):
        np.testing.assert_allclose(
            kb.loc[shared, col].to_numpy(),
            ka.loc[shared, col].to_numpy(),
            rtol=TOL,
            atol=TOL,
            equal_nan=True,
            err_msg=f"{col} differs beyond tolerance",
        )


# ------------------------------------------------------------------ transfers
class _CountingTorchOps(TorchArrayOps):
    """Counts real host<->device crossings: ``to_host`` calls, and ``asarray``
    calls that actually adopt a *host* array (an already-resident tensor is a
    no-op and is not a transfer)."""

    def __init__(self):
        super().__init__("cpu")
        self.to_host_calls = 0
        self.uploads: list = []

    def to_host(self, x):
        self.to_host_calls += 1
        return super().to_host(x)

    def asarray(self, x, dtype=None):
        if not isinstance(x, torch.Tensor):
            self.uploads.append(tuple(np.shape(x)))
        return super().asarray(x, dtype)


def test_transfers_per_chunk_are_the_bm25_block_and_the_handoff():
    """The regression test against a re-introduced ping-pong.

    With a device-resident index and device-resident query embeddings, each
    chunk must cross the boundary exactly twice-and-a-bit, and every crossing
    must be nameable:

      * **up**: the BM25 block, and nothing else — three arrays, one block:
        ``(b, k)`` neighbour labels, ``(b, k)`` neighbour scores, ``(b, C)``
        description scores. BM25 is permanently host-side (T83's policy), so
        this is the one intended sync point.
      * **down**: two calls, both at the fusion handoff at the very end of the
        chunk — the candidate grid and the stacked feature block. Nothing is
        lowered *during* assembly, which is the property that used to be
        violated by ``_scatter_knn``'s three ``to_host`` calls per signal.
    """
    ops = _CountingTorchOps()
    label_space, texts, y = _corpus()
    enc = HashingEncoder(dim=64)
    cfg = RetrievalConfig()
    dense = DenseRetrieverAdapter.build_from_embeddings(
        enc.encode(texts), y, enc.encode(label_space.descriptions), label_space, cfg, ops
    )
    lexical = LexicalRetrieverAdapter.build(texts, y, label_space, cfg)
    assembler = FeatureAssembler(label_space, CandidatePolicy(top_n_per_signal=3), ops)

    q_texts = texts[:20]
    q_emb = ops.asarray(enc.encode(q_texts))  # a device-resident encoder's output
    n_chunks = 2
    ops.to_host_calls = 0
    ops.uploads.clear()

    frame = assembler.assemble(
        q_texts, q_emb, dense, lexical, 4, query_ids=list(range(20)), chunk=10
    )
    assert len(frame) > 0

    assert ops.to_host_calls == 2 * n_chunks, (
        f"expected 2 D2H per chunk (candidate grid + feature block), got "
        f"{ops.to_host_calls} over {n_chunks} chunks"
    )
    assert len(ops.uploads) == 3 * n_chunks, (
        f"expected the BM25 block (3 arrays) per chunk and nothing else, got {ops.uploads}"
    )
    b, k, C = 10, 4, label_space.size
    assert sorted(ops.uploads[:3]) == sorted([(b, k), (b, k), (b, C)]), ops.uploads[:3]


def test_a_host_encoder_costs_one_extra_upload_per_chunk():
    """The counterpart: with a *host* encoder the query block has to be lifted,
    once per chunk, and that is the only difference."""
    ops = _CountingTorchOps()
    label_space, texts, y = _corpus()
    enc = HashingEncoder(dim=64)
    cfg = RetrievalConfig()
    dense = DenseRetrieverAdapter.build_from_embeddings(
        enc.encode(texts), y, enc.encode(label_space.descriptions), label_space, cfg, ops
    )
    lexical = LexicalRetrieverAdapter.build(texts, y, label_space, cfg)
    assembler = FeatureAssembler(label_space, CandidatePolicy(top_n_per_signal=3), ops)
    q_texts = texts[:20]
    ops.uploads.clear()
    assembler.assemble(
        q_texts, enc.encode(q_texts), dense, lexical, 4, query_ids=list(range(20)), chunk=10
    )
    assert len(ops.uploads) == 4 * 2, ops.uploads  # BM25 block (3) + query block (1), per chunk


# ------------------------------------------------------------------ OOM retry
def test_chunk_is_halved_and_the_run_continues_on_out_of_memory(caplog):
    """A device OOM must reduce ``feature_chunk`` and carry on, not kill a run
    that is already minutes in (T85's VRAM-chunking design point). Simulated by
    an ops backend that raises the first time it is asked for a big allocation
    — the failure mode is identified by the message, so this covers torch's
    ``OutOfMemoryError`` and numpy's ``MemoryError`` alike."""

    class _OOMOnce(NumpyArrayOps):
        def __init__(self):
            self.failed = False
            self.freed = 0

        def nonzero(self, mask):
            if not self.failed and mask.shape[0] > 4:
                self.failed = True
                raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
            return super().nonzero(mask)

        def free_memory(self):
            self.freed += 1

    ops = _OOMOnce()
    label_space, texts, y = _corpus()
    enc = HashingEncoder(dim=64)
    emb, desc_emb = enc.encode(texts), enc.encode(label_space.descriptions)
    q_texts, q_emb = texts[:16], enc.encode(texts[:16])

    with caplog.at_level("WARNING"):
        got = _assemble(ops, label_space, texts, y, emb, desc_emb, q_texts, q_emb, chunk=8)

    assert ops.failed and ops.freed == 1
    assert "retrying at feature_chunk=4" in caplog.text
    want = _assemble(NumpyArrayOps(), label_space, texts, y, emb, desc_emb, q_texts, q_emb, chunk=4)
    pd.testing.assert_frame_equal(got, want)


# ------------------------------------------------------------- config plumbing
def _fast_cfg(**overrides) -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.encoder.kind = "hashing"
    cfg.training = TrainingConfig(
        n_folds=3, random_state=0, target_precision=0.5, per_class_min_support=1
    )
    cfg.fusion = FusionConfig(
        xgb_params={"n_estimators": 10, "max_depth": 2, "random_state": 0, "n_jobs": 1}
    )
    cfg.retrieval = RetrievalConfig(k_neighbors=5)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_dense_kind_torch_is_selected_by_config_alone(tmp_path):
    """``dense_kind="torch"`` must be enough: no pipeline edits, no explicit
    backend, and the whole run — index, assembler, encoder — lands on the
    device backend, with the manifest recording which arithmetic ran."""
    label_space, items = make_synthetic(n_classes=5, per_class=12, seed=3)
    cfg = _fast_cfg()
    cfg.retrieval.dense_kind = "torch"

    artifacts, _ = TrainingPipeline(cfg, shared_encoder=HashingEncoder(dim=64)).run(
        items, label_space, output_dir=str(tmp_path / "model")
    )

    assert artifacts.array_ops is not None and artifacts.array_ops.name == "torch"
    assert isinstance(artifacts.dense.state.example_emb, torch.Tensor)
    assert isinstance(artifacts.dense.state.prototypes, torch.Tensor)

    import json

    manifest = json.loads((tmp_path / "model" / "evaluation.json").read_text())["manifest"]
    assert manifest["execution"] == {
        "array_backend": "torch",
        "device": "cpu",
        "dense_kind": "torch",
    }


def test_torch_dense_kind_with_an_explicit_numpy_backend_is_rejected():
    cfg = PipelineConfig()
    cfg.retrieval.dense_kind = "torch"
    cfg.array_backend = "numpy"
    with pytest.raises(ValueError, match="retrieval.dense_kind"):
        cfg.validate()


# --------------------------------------------------------------- portability
def test_a_torch_trained_model_loads_and_scores_on_a_torch_free_host(tmp_path, monkeypatch):
    """The portability invariant, end to end: a model trained on the device
    backend persists as numpy, and loads and scores on a host with no torch at
    all — with predictions equal within tolerance, not merely 'similar'."""
    label_space, items = make_synthetic(n_classes=5, per_class=12, seed=3)
    cfg = _fast_cfg()
    cfg.retrieval.dense_kind = "torch"
    model_dir = tmp_path / "model"
    TrainingPipeline(cfg, shared_encoder=HashingEncoder(dim=64)).run(
        items, label_space, output_dir=str(model_dir)
    )

    for name in ("dense.npz", "lexical.npz", "meta.json"):
        assert (model_dir / name).is_file()
    dense_npz = np.load(model_dir / "dense.npz")
    assert dense_npz["example_emb"].dtype == np.float32  # numpy on disk, always
    assert not list(model_dir.rglob("*.pkl"))

    texts = [it.text for it in items[:20]]
    on_device = InferencePipeline.from_directory(str(model_dir)).predict(texts)

    # Now hide torch from the loading host entirely.
    monkeypatch.setattr(
        "text_classifier.infrastructure.array_ops.torch_installed", lambda: False
    )
    reloaded = ArtifactRepository().load(str(model_dir))
    assert reloaded.array_ops.name == "numpy"
    on_host = InferencePipeline(reloaded).predict(texts)

    assert [p.top_key for p in on_host] == [p.top_key for p in on_device]
    np.testing.assert_allclose(
        [p.confidence for p in on_host],
        [p.confidence for p in on_device],
        rtol=1e-4,
        atol=1e-4,
    )


def test_meta_records_the_dense_kind_and_reloads_it(tmp_path):
    label_space, items = make_synthetic(n_classes=5, per_class=12, seed=3)
    cfg = _fast_cfg()
    cfg.retrieval.dense_kind = "torch"
    TrainingPipeline(cfg, shared_encoder=HashingEncoder(dim=64)).run(
        items, label_space, output_dir=str(tmp_path / "m")
    )
    import json

    meta = json.loads((tmp_path / "m" / "meta.json").read_text())
    assert meta["components"]["dense"] == "torch"
    assert ArtifactRepository().load(str(tmp_path / "m")).config.retrieval.dense_kind == "torch"


# ------------------------------------------------------------------- encoder
class _StubSentenceTransformer:
    """Minimal stand-in for a SentenceTransformer: records the kwargs the
    adapter passes and honours the two that matter."""

    def __init__(self, dim=8):
        self.dim = dim
        self.calls: list = []

    def encode(self, texts, **kwargs):
        self.calls.append(kwargs)
        rng = np.random.default_rng(0)
        emb = rng.standard_normal((len(texts), self.dim)).astype(np.float32)
        if kwargs.get("normalize_embeddings"):
            emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        if kwargs.get("convert_to_tensor"):
            return torch.from_numpy(emb)
        return emb

    def save(self, directory):  # pragma: no cover - not exercised here
        raise NotImplementedError


@pytest.mark.parametrize(
    "ops,is_tensor",
    [(None, False), (NumpyArrayOps(), False), (TorchArrayOps("cpu"), True)],
)
def test_encoder_output_follows_the_array_backend(ops, is_tensor):
    """The reworded invariant: L2-normalization is mandatory on both backends,
    the container follows the backend."""
    from text_classifier.infrastructure.encoder import SentenceTransformerEncoder

    model = _StubSentenceTransformer()
    encoder = SentenceTransformerEncoder(model, batch_size=4)
    if ops is not None:
        encoder.set_array_ops(ops)

    out = encoder.encode_queries(["a", "b", "c"])
    assert isinstance(out, torch.Tensor) is is_tensor
    kwargs = model.calls[-1]
    assert kwargs["normalize_embeddings"] is True
    assert kwargs["convert_to_numpy"] is not is_tensor
    assert kwargs.get("convert_to_tensor", False) is is_tensor

    host = out.numpy() if is_tensor else out
    assert host.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(host, axis=1), 1.0, rtol=1e-6, atol=1e-6)


def test_numpy_backend_run_never_imports_torch(monkeypatch):
    """T84's guard, still standing with a torch backend registered: registering
    a kind must not import it, and a below-crossover run must not probe for a
    device."""
    for mod in list(sys.modules):
        if mod == "torch" or mod.startswith("torch."):
            monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.setitem(sys.modules, "torch", None)  # `import torch` now raises

    from text_classifier.infrastructure.array_ops import resolve_array_backend

    assert resolve_array_backend(None, n_items=500, n_classes=10) == "numpy"
    assert sys.modules["torch"] is None


def test_dense_state_survives_a_backend_round_trip():
    """``with_array_ops`` is the load-time upload; it must carry values across
    unchanged (it is a move, not a recomputation)."""
    label_space, texts, y = _corpus(n_classes=4, per_class=5, seed=9)
    enc = HashingEncoder(dim=32)
    cfg = RetrievalConfig()
    host = DenseRetrieverAdapter.build_from_embeddings(
        enc.encode(texts), y, enc.encode(label_space.descriptions), label_space, cfg
    )
    device = host.with_array_ops(TorchArrayOps("cpu"))
    assert isinstance(device.state.example_emb, torch.Tensor)
    back = device.to_state()
    for key, value in host.to_state().items():
        np.testing.assert_array_equal(back[key], value, err_msg=key)
    # and the trivial case is a no-op, not a copy
    assert host.with_array_ops(NumpyArrayOps()) is host


def test_the_diagnostic_surfaces_work_on_a_device_backed_model(tmp_path):
    """`explain` and `retune` read retriever output directly (neighbour lists,
    the overlap probe) rather than going through the assembler, so they are the
    two places a device-resident index could leak a tensor into host-only code.
    Both must work unchanged."""
    from text_classifier.application.tuning import retune

    label_space, items = make_synthetic(n_classes=5, per_class=12, seed=3)
    cfg = _fast_cfg()
    cfg.retrieval.dense_kind = "torch"
    artifacts, _ = TrainingPipeline(cfg, shared_encoder=HashingEncoder(dim=64)).run(
        items, label_space, output_dir=str(tmp_path / "m")
    )
    pipeline = InferencePipeline(artifacts)

    records = pipeline.explain_records([it.text for it in items[:3]], n_neighbors=2)
    assert len(records) == 3
    for record in records:
        neighbours = record["neighbors"]["dense"]
        assert neighbours and all(isinstance(n["score"], float) for n in neighbours)
    assert len(pipeline.explain([it.text for it in items[:3]])) > 0

    fresh = [it for it in make_synthetic(n_classes=5, per_class=4, seed=99)[1]]
    abstention, calibrator, evaluation = retune(
        artifacts, fresh, label_space, target_precision=0.5, per_class_min_support=1
    )
    assert 0.0 <= abstention.global_threshold <= 1.0
    assert evaluation["overall"]["n_items"] == len(fresh)
