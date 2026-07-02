"""T20 — Input validation at the public entry points.

Every public boundary of the package should reject bad input with a clear,
actionable error that names the offending value — never a cryptic numpy/pandas/
sklearn traceback from deep inside the pipeline. This module pins that contract:

  Part A  ClassDefinition / LabelSpace construction
  Part B  LabeledItem + TrainingPipeline.run training inputs
  Part C  InferencePipeline.predict inference inputs
  Part D  ArtifactRepository.load (missing dir/file, feature-schema drift)

All tests run fully offline: the validation under test fires *before* any
encoder, index, or model work, so no real model or network access is needed.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold

from text_classifier import ClassDefinition, LabeledItem, LabelSpace
from text_classifier.application.training import TrainingPipeline
from text_classifier.config import PipelineConfig, TrainingConfig
from text_classifier.domain import FEATURE_NAMES
from text_classifier.infrastructure.persistence import ArtifactRepository


# --------------------------------------------------------------------------- #
# Part A — ClassDefinition / LabelSpace construction
# --------------------------------------------------------------------------- #
class TestLabelSpaceConstruction:
    def test_duplicate_keys_raise_and_name_the_key(self):
        with pytest.raises(ValueError, match="dup") as exc:
            LabelSpace.from_pairs([("dup", "a"), ("dup", "b"), ("ok", "c")])
        assert "dup" in str(exc.value)

    def test_empty_definitions_raises_readable_message(self):
        with pytest.raises(ValueError, match="at least one"):
            LabelSpace([])

    def test_empty_key_raises_value_error(self):
        with pytest.raises(ValueError, match="key must be a non-empty"):
            ClassDefinition(key="", description="x")

    def test_whitespace_key_raises_value_error(self):
        with pytest.raises(ValueError, match="key must be a non-empty"):
            ClassDefinition(key="   ", description="x")

    def test_empty_description_raises_and_names_the_key(self):
        # Policy: an empty description is rejected — it strips the class of the
        # text the description-similarity signals depend on.
        with pytest.raises(ValueError, match="description") as exc:
            ClassDefinition(key="alpha", description="")
        assert "alpha" in str(exc.value)

    def test_non_string_key_raises_value_error(self):
        with pytest.raises(ValueError, match="key must be a non-empty"):
            ClassDefinition(key=123, description="x")  # type: ignore[arg-type]

    def test_valid_definition_constructs(self):
        cd = ClassDefinition(key="alpha", description="the alpha class")
        assert cd.key == "alpha"


# --------------------------------------------------------------------------- #
# Part B — LabeledItem + TrainingPipeline.run
# --------------------------------------------------------------------------- #
class TestLabeledItemConstruction:
    def test_empty_text_raises_value_error(self):
        with pytest.raises(ValueError, match="text must be a non-empty"):
            LabeledItem(text="", label="A")

    def test_whitespace_text_raises_value_error(self):
        with pytest.raises(ValueError, match="text must be a non-empty"):
            LabeledItem(text="   ", label="A")

    def test_empty_label_raises_value_error(self):
        with pytest.raises(ValueError, match="label must be a non-empty"):
            LabeledItem(text="some text", label="")

    def test_none_text_raises_value_error(self):
        with pytest.raises(ValueError, match="text must be a non-empty"):
            LabeledItem(text=None, label="A")  # type: ignore[arg-type]


def _space() -> LabelSpace:
    return LabelSpace.from_pairs([("A", "class a"), ("B", "class b")])


def _pipeline(n_folds: int = 3) -> TrainingPipeline:
    cfg = PipelineConfig()
    cfg.training = TrainingConfig(n_folds=n_folds, random_state=0)
    return TrainingPipeline(cfg)


class TestTrainingInputValidation:
    def test_empty_items_raise_value_error(self):
        with pytest.raises(ValueError, match="non-empty list of items"):
            _pipeline().run([], _space())

    def test_label_not_in_space_raises_and_names_it(self):
        items = [LabeledItem("hello world", "GHOST")]
        with pytest.raises(ValueError, match="not defined in the LabelSpace") as exc:
            _pipeline().run(items, _space())
        assert "GHOST" in str(exc.value)

    def test_underpopulated_class_raises_before_any_work(self):
        # Class A has 1 item but n_folds=3 -> StratifiedKFold would crash; we
        # raise a clear error first instead.
        items = [LabeledItem("a text", "A")]
        items += [LabeledItem(f"b text {i}", "B") for i in range(5)]
        with pytest.raises(ValueError, match=r"at least 3 examples per class") as exc:
            _pipeline(n_folds=3).run(items, _space())
        assert "'A'" in str(exc.value)

    def test_single_class_dataset_is_accepted(self):
        # All items in one class is acceptable as long as it clears n_folds.
        space = _space()
        items = [LabeledItem(f"only a {i}", "A") for i in range(6)]
        # _validate_inputs must not raise...
        _pipeline(n_folds=3)._validate_inputs(items, space)
        # ...and the split it guards must not crash either.
        y = np.array(space.encode_labels([it.label for it in items]))
        list(StratifiedKFold(3, shuffle=True, random_state=0).split(items, y))


# --------------------------------------------------------------------------- #
# Part C — InferencePipeline.predict
#
# predict() validates its inputs before touching the encoder, so we can drive
# the validation through a bare instance without a trained model. A tiny stub
# stands in for DeployedArtifacts; the encoder is never reached for bad input.
# --------------------------------------------------------------------------- #
class _StubArtifacts:
    class _Cfg:
        candidate_top_n = 10

    config = _Cfg()
    label_space = _space()


def _inference_pipeline():
    from text_classifier.application.inference import InferencePipeline

    pipe = InferencePipeline.__new__(InferencePipeline)  # skip heavy __init__
    pipe._a = _StubArtifacts()  # type: ignore[attr-defined]
    return pipe


class TestInferenceInputValidation:
    def test_non_string_raises_type_error_with_index(self):
        with pytest.raises(TypeError, match="index 1") as exc:
            _inference_pipeline().predict(["ok", 42])
        assert "int" in str(exc.value)

    def test_none_raises_type_error(self):
        with pytest.raises(TypeError, match="NoneType"):
            _inference_pipeline().predict([None])

    def test_empty_string_passes_validation(self):
        # Empty strings are allowed (they abstain); validation must not reject them.
        _inference_pipeline()._validate_texts(["", "still fine"])


# --------------------------------------------------------------------------- #
# Part D — ArtifactRepository.load
# --------------------------------------------------------------------------- #
class TestArtifactRepositoryValidation:
    def test_missing_directory_raises_with_path(self):
        with pytest.raises(FileNotFoundError, match="/nonexistent/model/path"):
            ArtifactRepository().load("/nonexistent/model/path")

    def test_missing_meta_json_raises_with_path(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="meta.json"):
            ArtifactRepository().load(str(tmp_path))

    def test_feature_schema_drift_raises_value_error(self, tmp_path):
        meta = {
            "feature_names": ["only_one_feature"],
            "config": {},
            "classes": [],
            "abstention": {},
        }
        (tmp_path / "meta.json").write_text(json.dumps(meta))
        with pytest.raises(ValueError, match="feature schema drift") as exc:
            ArtifactRepository().load(str(tmp_path))
        msg = str(exc.value)
        assert "only_one_feature" in msg  # unknown-to-code column is named
        assert str(len(FEATURE_NAMES)) in msg  # expected count is reported

    def test_reordered_features_raise_value_error(self, tmp_path):
        meta = {
            "feature_names": list(reversed(FEATURE_NAMES)),
            "config": {},
            "classes": [],
            "abstention": {},
        }
        (tmp_path / "meta.json").write_text(json.dumps(meta))
        with pytest.raises(ValueError, match="column order differs"):
            ArtifactRepository().load(str(tmp_path))


# --------------------------------------------------------------------------- #
# Part E — PipelineConfig.validate (T27)
# --------------------------------------------------------------------------- #
class TestConfigValidation:
    def test_default_config_is_valid(self):
        PipelineConfig().validate()  # must not raise

    @pytest.mark.parametrize("n_folds", [1, 2])
    def test_too_few_folds_rejected(self, n_folds):
        cfg = PipelineConfig()
        cfg.training.n_folds = n_folds
        with pytest.raises(ValueError, match="n_folds") as exc:
            cfg.validate()
        # The message explains the three fold roles, not just the bound.
        assert "calibration" in str(exc.value) and "test" in str(exc.value)

    def test_three_folds_is_the_boundary(self):
        cfg = PipelineConfig()
        cfg.training.n_folds = 3
        cfg.validate()  # must not raise

    @pytest.mark.parametrize(
        "mutate, field_name",
        [
            (lambda c: setattr(c.training, "target_precision", 0.0), "target_precision"),
            (lambda c: setattr(c.training, "target_precision", 1.5), "target_precision"),
            (lambda c: setattr(c.training, "per_class_min_support", 0), "per_class_min_support"),
            (lambda c: setattr(c, "candidate_top_n", 0), "candidate_top_n"),
            (lambda c: setattr(c.retrieval, "k_neighbors", 0), "k_neighbors"),
            (lambda c: setattr(c.retrieval, "dense_chunk", 0), "dense_chunk"),
            (lambda c: setattr(c.retrieval, "feature_chunk", 0), "feature_chunk"),
            (lambda c: setattr(c.encoder, "encode_batch_size", 0), "encode_batch_size"),
        ],
    )
    def test_each_bound_rejected_naming_the_field(self, mutate, field_name):
        cfg = PipelineConfig()
        mutate(cfg)
        with pytest.raises(ValueError, match=field_name) as exc:
            cfg.validate()
        assert "got" in str(exc.value)  # the received value is echoed back

    @pytest.mark.parametrize(
        "mutate",
        [
            lambda c: setattr(c.training, "target_precision", 1.0),
            lambda c: setattr(c.training, "per_class_min_support", 1),
            lambda c: setattr(c, "candidate_top_n", 1),
            lambda c: setattr(c.retrieval, "k_neighbors", 1),
            lambda c: setattr(c.retrieval, "dense_chunk", 1),
            lambda c: setattr(c.retrieval, "feature_chunk", 1),
            lambda c: setattr(c.encoder, "encode_batch_size", 1),
        ],
    )
    def test_boundary_values_pass(self, mutate):
        cfg = PipelineConfig()
        mutate(cfg)
        cfg.validate()  # must not raise

    def test_multiple_problems_reported_together(self):
        cfg = PipelineConfig()
        cfg.training.n_folds = 2
        cfg.retrieval.k_neighbors = 0
        with pytest.raises(ValueError) as exc:
            cfg.validate()
        msg = str(exc.value)
        assert "n_folds" in msg and "k_neighbors" in msg

    def test_pipeline_rejects_bad_config_before_touching_the_encoder(self):
        class _ExplodingEncoder:
            def encode(self, texts):
                raise AssertionError("encoder must not be reached with a bad config")

        cfg = PipelineConfig()
        cfg.training.n_folds = 2
        pipe = TrainingPipeline(cfg, shared_encoder=_ExplodingEncoder())
        items = [LabeledItem(f"item {i}", "a") for i in range(10)]
        space = LabelSpace.from_pairs([("a", "class a"), ("b", "class b")])
        with pytest.raises(ValueError, match="n_folds"):
            pipe.run(items, space)
