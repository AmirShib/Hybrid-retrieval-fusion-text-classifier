"""Where inference actually runs, and what it stops recomputing.

Before this, ``ArtifactRepository.load`` resolved no array backend at all:
``--device cuda`` placed the encoder and the fusion booster, and everything
between them — the dense index and its matmuls — stayed on host numpy, with the
persisted ``config.array_backend`` silently ignored. These tests pin the wiring
that closes that, plus the recomputation the inference path used to pay for.

Offline throughout (``HashingEncoder``); the torch backend is exercised by
substituting a recording ``ArrayOps`` double, so the *placement contract* is
tested on any host. Real torch parity is ``test_device_parity.py``'s job and
skips without torch.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

from text_classifier import PipelineConfig, TrainingPipeline
from text_classifier.application.inference import InferencePipeline
from text_classifier.infrastructure import ArtifactRepository
from text_classifier.infrastructure.array_ops import NumpyArrayOps
from text_classifier.infrastructure.registry import ArrayOpsSpec, register_array_ops
from tests._doubles import HashingEncoder, make_synthetic


class RecordingArrayOps(NumpyArrayOps):
    """A numpy backend wearing another backend's name.

    Numerically identical to ``NumpyArrayOps`` — so every assertion about
    *values* still holds — while reporting a distinct ``name`` and counting
    ``asarray`` calls. That is exactly what is needed to test placement:
    "was the index handed to this backend, once, at load time" is a wiring
    question, not a numerics one, and answering it must not require a GPU."""

    name = "recording"

    def __init__(self) -> None:
        self.asarray_calls = 0

    def asarray(self, x: Any, dtype: Optional[Any] = None) -> np.ndarray:
        self.asarray_calls += 1
        return super().asarray(x, dtype=dtype)


_LAST_RECORDING: List[RecordingArrayOps] = []


def _build_recording() -> RecordingArrayOps:
    ops = RecordingArrayOps()
    _LAST_RECORDING.append(ops)
    return ops


register_array_ops("recording", ArrayOpsSpec(build=_build_recording))


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory) -> str:
    label_space, items = make_synthetic(n_classes=8, per_class=20, seed=5)
    cfg = PipelineConfig(candidate_top_n=6)
    cfg.encoder.kind = "hashing"
    cfg.training.n_folds = 3
    cfg.training.target_precision = 0.5
    cfg.training.per_class_min_support = 1
    cfg.retrieval.k_neighbors = 8
    out = str(tmp_path_factory.mktemp("model"))
    TrainingPipeline(cfg, shared_encoder=HashingEncoder(dim=64)).run(
        items, label_space, output_dir=out
    )
    return out


@pytest.fixture(scope="module")
def texts(model_dir) -> List[str]:
    _, items = make_synthetic(n_classes=8, per_class=20, seed=5)
    return [it.text for it in items[:6]]


class TestBackendResolution:
    def test_default_load_stays_on_numpy(self, model_dir):
        """The untouched path is unchanged: no request, a config that says
        "auto", and a corpus far below the crossover -> numpy."""
        artifacts = ArtifactRepository().load(model_dir)
        assert artifacts.config.array_backend == "auto"
        assert artifacts.array_backend == "numpy"
        assert artifacts.dense.array_backend == "numpy"

    def test_explicit_request_places_the_index(self, model_dir):
        """The gap this closes: an explicit backend now actually reaches the
        dense index, instead of being accepted and dropped."""
        _LAST_RECORDING.clear()
        artifacts = ArtifactRepository().load(model_dir, array_backend="recording")
        assert artifacts.array_backend == "recording"
        assert artifacts.dense.array_backend == "recording"
        # Uploaded once at load: three float arrays (examples, prototypes,
        # descriptions) through `asarray`, and nothing per query afterward.
        ops = _LAST_RECORDING[-1]
        assert ops.asarray_calls == 3
        InferencePipeline(artifacts).predict(["some text", "another"])
        assert ops.asarray_calls == 3, "the corpus must not be re-uploaded per query"

    def test_persisted_config_is_honoured_without_a_request(self, model_dir, tmp_path):
        """A model trained with a backend deploys on it with no flag repeated
        at the call site. This is the assertion that used to be impossible:
        `config.array_backend` was read into the config object and then never
        consulted again."""
        import json
        import shutil

        copied = str(tmp_path / "pinned")
        shutil.copytree(model_dir, copied)
        meta_path = f"{copied}/meta.json"
        with open(meta_path) as fh:
            meta = json.load(fh)
        meta["config"]["array_backend"] = "recording"
        with open(meta_path, "w") as fh:
            json.dump(meta, fh)

        artifacts = ArtifactRepository().load(copied)
        assert artifacts.array_backend == "recording"

    def test_explicit_request_beats_the_persisted_config(self, model_dir, tmp_path):
        import json
        import shutil

        copied = str(tmp_path / "override")
        shutil.copytree(model_dir, copied)
        meta_path = f"{copied}/meta.json"
        with open(meta_path) as fh:
            meta = json.load(fh)
        meta["config"]["array_backend"] = "recording"
        with open(meta_path, "w") as fh:
            json.dump(meta, fh)

        artifacts = ArtifactRepository().load(copied, array_backend="numpy")
        assert artifacts.array_backend == "numpy"

    def test_auto_sizes_off_the_index_not_the_batch(self, model_dir, monkeypatch):
        """The crossover must see the indexed corpus. Sizing off the incoming
        batch would put every ordinary request below it regardless of how large
        an index it is about to search."""
        seen: Dict[str, Any] = {}

        def spy(explicit, *, n_items, n_classes, **kw):
            seen.update(n_items=n_items, n_classes=n_classes)
            return "numpy"

        indexed = int(np.asarray(ArtifactRepository().load(model_dir).dense.class_freq).sum())
        monkeypatch.setattr("text_classifier.infrastructure.persistence.resolve_array_backend", spy)
        ArtifactRepository().load(model_dir, array_backend="auto")
        assert seen["n_items"] == indexed  # the whole example pool...
        assert indexed > 100  # ...which is nothing like a per-call batch size
        assert seen["n_classes"] == 8

    def test_predictions_are_unchanged_across_backends(self, model_dir, texts):
        """Placement is a performance concern; it must not move a decision."""
        on_numpy = InferencePipeline.from_directory(model_dir, array_backend="numpy")
        on_other = InferencePipeline.from_directory(model_dir, array_backend="recording")
        for a, b in zip(on_numpy.predict(texts), on_other.predict(texts)):
            assert a.top_key == b.top_key
            assert a.abstained == b.abstained
            assert a.confidence == pytest.approx(b.confidence)

    def test_missing_torch_is_an_actionable_error(self, model_dir, monkeypatch):
        """Asking for a backend that cannot be built must say so, and say what
        to do — reaching this means someone typed `--array-backend torch`, since
        auto-resolution probes `torch_installed()` and never gets here without
        it. Before, an explicit request was accepted and silently ignored, which
        is the failure mode this replaces."""
        import sys

        monkeypatch.setitem(sys.modules, "torch", None)  # forces ImportError
        monkeypatch.delitem(
            sys.modules, "text_classifier.infrastructure.array_ops_torch", raising=False
        )
        with pytest.raises(ImportError, match="torch is not installed"):
            ArtifactRepository().load(model_dir, array_backend="torch")

    def test_taxonomy_update_survives_a_placed_index(self, model_dir):
        """`with_added_classes` extends the index with numpy operations a
        resident tensor does not support, so it has to round-trip host-side and
        re-upload. Regression guard for the path a device-resident deployment
        newly reaches."""
        pipeline = InferencePipeline.from_directory(model_dir, array_backend="recording")
        widened = pipeline.with_added_classes([("brand_new", "a freshly added class")])
        assert widened.artifacts.dense.array_backend == "recording"
        assert "brand_new" in widened.label_space.keys
        assert len(widened.predict(["anything at all"])) == 1


class TestNeighborReuse:
    """`explain_records` used to run the dense and BM25 kNN searches twice: once
    inside the assembler to build the knn signals, and again for the neighbor
    evidence — the two most expensive stages in the pipeline (T83), duplicated."""

    def _counting(self, pipeline: InferencePipeline) -> Tuple[Dict[str, int], None]:
        calls = {"dense": 0, "lexical": 0}
        a = pipeline.artifacts
        dense_knn = a.dense.knn_example_labels
        lexical_knn = a.lexical.knn_example_labels

        def dense_spy(*args, **kwargs):
            calls["dense"] += 1
            return dense_knn(*args, **kwargs)

        def lexical_spy(*args, **kwargs):
            calls["lexical"] += 1
            return lexical_knn(*args, **kwargs)

        a.dense.knn_example_labels = dense_spy  # type: ignore[method-assign]
        a.lexical.knn_example_labels = lexical_spy  # type: ignore[method-assign]
        return calls, None

    def test_explain_records_searches_once_per_retriever(self, model_dir, texts):
        pipeline = InferencePipeline.from_directory(model_dir)
        calls, _ = self._counting(pipeline)
        pipeline.explain_records(texts, top_k=2)
        assert calls == {"dense": 1, "lexical": 1}

    def test_reused_neighbors_match_a_fresh_query(self, model_dir, texts):
        """Reuse must be exactly the same evidence, not merely similar: the
        sink carries the arrays the signals were built from."""
        pipeline = InferencePipeline.from_directory(model_dir)
        records = pipeline.explain_records(texts, top_k=2, n_neighbors=4)

        a = pipeline.artifacts
        k = a.config.retrieval.k_neighbors
        q = a.encoder.encode_queries(texts)
        d_lab, d_sim = a.dense.knn_example_labels(q, k)
        b_lab, b_sco = a.lexical.knn_example_labels(list(texts), k)
        expected = pipeline._neighbor_evidence(texts, q, a.label_space.keys, 4, None)

        assert [r["neighbors"] for r in records] == [
            {
                "dense": ev["dense"],
                "lexical": ev["lexical"],
                "texts_available": False,
            }
            for ev in expected
        ]
        assert d_lab.shape == (len(texts), k) and d_sim.shape == (len(texts), k)
        assert b_lab.shape == (len(texts), k) and b_sco.shape == (len(texts), k)

    def test_chunked_assembly_keeps_neighbor_rows_aligned(self, model_dir, texts):
        """The sink concatenates per chunk; a misaligned merge would shift every
        row after the first chunk boundary."""
        pipeline = InferencePipeline.from_directory(model_dir)
        whole = pipeline.explain_records(texts, top_k=1, n_neighbors=3)
        pipeline.artifacts.config.retrieval.feature_chunk = 2  # force several chunks
        chunked = pipeline.explain_records(texts, top_k=1, n_neighbors=3)
        assert [r["neighbors"] for r in whole] == [r["neighbors"] for r in chunked]
