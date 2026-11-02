"""T85 -- device parity: TrainingPipeline.run with array_backend="torch"
(torch-CPU, GPU-free -- this is the offline half of the ticket's stated
"tolerance-based parity" testing strategy) reaches the same operating point
as the numpy backend on the same data/seed, and every intermediate object
(dense retriever, encoder) behaves as documented along the way.

Skipped entirely when torch is not installed. Bounds/closeness only, not
exact floats, per this suite's determinism contract (conftest.py) -- pipeline
numerics depend on XGBoost internals that vary across platforms, and cross-
device float32 reduction order is explicitly *not* bit-identical (T85's
accepted cost, see CLAUDE.md and docs/device-policy.md).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from text_classifier.config import PipelineConfig  # noqa: E402
from text_classifier.application.training import TrainingPipeline  # noqa: E402
from text_classifier.infrastructure.encoder import SentenceTransformerEncoder  # noqa: E402
from text_classifier.datasets import make_synthetic  # noqa: E402


class _StubSTModel:
    """Deterministic per-text-batch stub: same texts -> same embeddings,
    whether asked for as numpy or as a tensor, so the numpy-backend run and
    the torch-backend run of this test see identical inputs at every stage."""

    def __init__(self, dim: int = 8):
        self.dim = dim

    def save(self, directory):  # SentenceTransformerEncoder.save delegates here
        import os

        os.makedirs(directory, exist_ok=True)

    def encode(self, texts, **kwargs):
        texts = list(texts)
        seed = abs(hash(tuple(texts))) % (2**31)
        rng = np.random.default_rng(seed)
        emb = rng.standard_normal((len(texts), self.dim)).astype(np.float32)
        if kwargs.get("normalize_embeddings"):
            emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        if kwargs.get("convert_to_tensor"):
            return torch.as_tensor(emb)
        assert kwargs.get("convert_to_numpy") is True
        return emb


def _run(array_backend: str, items, label_space):
    encoder = SentenceTransformerEncoder(_StubSTModel(), batch_size=16)
    cfg = PipelineConfig()
    cfg.array_backend = array_backend
    cfg.training.n_folds = 3
    cfg.candidate_top_n = 5
    cfg.retrieval.k_neighbors = 5
    pipeline = TrainingPipeline(cfg, shared_encoder=encoder)
    return pipeline.run(items, label_space)


@pytest.fixture(scope="module")
def synthetic():
    return make_synthetic(n_classes=5, per_class=12, seed=3)


class TestTorchBackendParity:
    def test_torch_backend_run_completes_and_uses_a_torch_encoder(self, synthetic):
        label_space, items = synthetic
        artifacts, report = _run("torch", items, label_space)
        assert report.coverage > 0
        assert 0.0 <= report.accuracy_on_accepted <= 1.0
        # The run's resolved backend is recorded on the deployed config
        # (provenance) -- persisted into meta.json's config block.
        assert artifacts.config.array_backend == "torch"

    def test_torch_and_numpy_backends_reach_the_same_operating_point(self, synthetic):
        label_space, items = synthetic
        _, report_numpy = _run("numpy", items, label_space)
        _, report_torch = _run("torch", items, label_space)

        # Same deterministic embeddings, same folds, same candidate pool on
        # both runs -- coverage/accuracy should match closely. Not exact
        # equality: float32 reduction order differs by device (T85's
        # accepted cost), so a small tolerance is the honest bar, not zero.
        assert report_numpy.coverage == pytest.approx(report_torch.coverage, abs=0.05)
        assert report_numpy.accuracy_on_accepted == pytest.approx(
            report_torch.accuracy_on_accepted, abs=0.05
        )

    def test_torch_backend_dense_index_persists_as_numpy(self, synthetic, tmp_path):
        """Persistence stays numpy always (T85 design point 3): the deployed
        dense index built under the torch backend still serializes to plain
        numpy arrays, and a from_state() reload (the numpy-default path
        ArtifactRepository.load always uses) round-trips correctly -- the
        model dir stays portable to an air-gapped, torch-free host regardless
        of what trained it. This checks the dense index specifically rather
        than the full ArtifactRepository round trip, which also reloads the
        (real, network-fetched) SentenceTransformer weights -- unrelated to
        T85 and out of scope for an offline stub-model test."""
        from text_classifier.infrastructure.retrieval import DenseRetrieverAdapter

        label_space, items = synthetic
        artifacts, _ = _run("torch", items, label_space)

        state = artifacts.dense.to_state()
        for key, arr in state.items():
            assert isinstance(arr, np.ndarray), f"{key} is {type(arr)}, expected numpy"

        reloaded = DenseRetrieverAdapter.from_state(
            state, chunk=artifacts.config.retrieval.dense_chunk
        )
        query = np.asarray(state["example_emb"][:2])
        lab_before, sim_before = artifacts.dense.knn_example_labels(query, k=3)
        lab_after, sim_after = reloaded.knn_example_labels(query, k=3)
        np.testing.assert_array_equal(lab_before, lab_after)
        np.testing.assert_allclose(sim_before, sim_after, atol=1e-5)
