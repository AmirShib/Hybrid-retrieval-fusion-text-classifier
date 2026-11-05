"""Training progress: estimate arithmetic (fake clock), the live file, and the
pipeline wiring (stages, done/failed status, off by default)."""

from __future__ import annotations

import json
import os

import pytest

from tests._doubles import make_synthetic
from text_classifier._progress import (
    PROGRESS_FILE,
    TrainingProgress,
    show_training_bars,
    training_bars,
)
from text_classifier.application.training import TrainingPipeline
from text_classifier.config import PipelineConfig, TrainingConfig


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_estimate_averages_fold_like_stages_and_counts_the_one_running(tmp_path):
    clock = Clock()
    folds = ["fold 1/3", "fold 2/3", "fold 3/3", "final model"]
    p = TrainingProgress(folds + ["fusion"], folds, str(tmp_path), clock=clock, wall=lambda: 0.0)
    assert p.remaining_seconds() is None  # nothing finished: no estimate yet
    p.start("fold 1/3")
    clock.t = 100
    p.done("fold 1/3")
    assert p.remaining_seconds() == pytest.approx(300)  # 3 fold-like stages left x 100s
    p.start("fold 2/3")
    clock.t = 160  # 60s into the next one
    assert p.remaining_seconds() == pytest.approx(240)
    clock.t = 400  # running long: its share is capped at the average, never negative
    assert p.remaining_seconds() == pytest.approx(200)


def test_progress_file_is_rewritten_each_stage(tmp_path):
    clock = Clock()
    p = TrainingProgress(["a", "b"], ["a", "b"], str(tmp_path), clock=clock, wall=lambda: 0.0)
    p.start("a")
    clock.t = 10
    p.done("a")
    snap = json.load(open(tmp_path / PROGRESS_FILE))
    assert snap["status"] == "running" and snap["completed_stages"] == 1
    assert snap["estimated_remaining_seconds"] == pytest.approx(10)
    assert snap["stages"][0] == {"name": "a", "seconds": 10.0}
    p.finish("done")
    assert json.load(open(tmp_path / PROGRESS_FILE))["status"] == "done"


def test_disabled_progress_writes_nothing(tmp_path):
    p = TrainingProgress(["a"], ["a"], str(tmp_path), enabled=False)
    p.start("a")
    p.done("a")
    p.finish()
    assert not os.path.exists(tmp_path / PROGRESS_FILE)


def test_training_bars_flag_is_scoped():
    assert show_training_bars() is False
    with training_bars(True):
        assert show_training_bars() is True
    assert show_training_bars() is False


def _cfg(progress):
    cfg = PipelineConfig()
    cfg.encoder.kind = "hashing"
    cfg.training = TrainingConfig(
        n_folds=3, random_state=0, target_precision=0.5, per_class_min_support=1, progress=progress
    )
    return cfg


def test_pipeline_reports_every_stage_and_finishes_done(tmp_path):
    ls, items = make_synthetic(n_classes=4, per_class=12, seed=1)
    out = str(tmp_path / "model")
    TrainingPipeline(_cfg(True)).run(items, ls, output_dir=out)
    snap = json.load(open(os.path.join(out, PROGRESS_FILE)))
    assert snap["status"] == "done"
    assert [s["name"] for s in snap["stages"]] == [
        "fold 1/3",
        "fold 2/3",
        "fold 3/3",
        "final model",
        "fusion + calibration + evaluation",
    ]
    assert all(s["seconds"] is not None for s in snap["stages"])


def test_pipeline_marks_a_failed_run_failed(tmp_path, monkeypatch):
    ls, items = make_synthetic(n_classes=4, per_class=12, seed=1)
    out = str(tmp_path / "model")
    pipe = TrainingPipeline(_cfg(True))

    def boom(*a, **k):
        raise RuntimeError("fusion exploded")

    monkeypatch.setattr(pipe, "_fit_fusion", boom)
    with pytest.raises(RuntimeError):
        pipe.run(items, ls, output_dir=out)
    assert json.load(open(os.path.join(out, PROGRESS_FILE)))["status"] == "failed"


def test_progress_off_by_default_leaves_no_file(tmp_path):
    ls, items = make_synthetic(n_classes=4, per_class=12, seed=1)
    out = str(tmp_path / "model")
    TrainingPipeline(_cfg(False)).run(items, ls, output_dir=out)
    assert not os.path.exists(os.path.join(out, PROGRESS_FILE))
    assert PipelineConfig().training.progress is False
