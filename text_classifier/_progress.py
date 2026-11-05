"""Training progress: stage timing, an estimated finish time, and a live file.

Layer-neutral (no ML imports), like ``_messages``. ``TrainingPipeline`` reports
each stage -- every fold, the final model build, fusion + calibration +
evaluation -- and this module:

- logs one line per finished stage with the elapsed time, the estimated time
  remaining and the estimated finish clock time;
- draws one overall progress bar (tqdm: auto-detects terminal vs notebook and
  stays silent when output is piped to a file);
- rewrites ``progress.json`` in the output directory after every stage, so a long
  run can be watched from another terminal or a notebook;
- switches on the per-step bars inside fine-tunes (encoder, cross-encoder,
  SPLADE) for the duration of the run, via ``show_training_bars()``.

The estimate: the final model build does roughly the work of one fold (the same
index build, plus any fine-tunes on all items), so "fold-like" stages are
averaged and the remaining ones projected. The last stage (fusion fit,
calibration, evaluation) is usually short and is not projected; the estimate is
labelled accordingly. Off by default: without ``training.progress`` nothing here
runs and output is unchanged.
"""

from __future__ import annotations

import contextlib
import contextvars
import datetime
import json
import logging
import os
import time
from typing import Any, Callable, Dict, Iterator, List, Optional

log = logging.getLogger("text_classifier.progress")

_SHOW_BARS: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "show_training_bars", default=False
)

PROGRESS_FILE = "progress.json"


def show_training_bars() -> bool:
    """Whether fine-tunes should draw their own per-step progress bars."""
    return _SHOW_BARS.get()


@contextlib.contextmanager
def training_bars(enabled: bool) -> Iterator[None]:
    token = _SHOW_BARS.set(bool(enabled))
    try:
        yield
    finally:
        _SHOW_BARS.reset(token)


def _fmt(seconds: Optional[float]) -> str:
    if seconds is None:
        return "?"
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


class TrainingProgress:
    """Times named stages and projects the remaining time.

    ``stages`` is the ordered plan; ``fold_like`` names the stages of comparable
    cost that the estimate averages over. ``clock``/``wall`` are injectable for
    tests. With ``enabled=False`` every method is a no-op."""

    def __init__(
        self,
        stages: List[str],
        fold_like: List[str],
        output_dir: Optional[str] = None,
        enabled: bool = True,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
    ):
        self.enabled = enabled
        self.stages = list(stages)
        self.fold_like = set(fold_like)
        self.output_dir = output_dir
        self._clock, self._wall = clock, wall
        self._t0 = clock()
        self._started: Dict[str, float] = {}
        self.durations: Dict[str, float] = {}
        self.current: Optional[str] = None
        self._bar: Any = None
        if enabled:
            try:
                from tqdm.auto import tqdm

                self._bar = tqdm(
                    total=len(self.stages), desc="training", unit="stage", disable=None
                )
            except Exception:  # noqa: BLE001 - progress must never break training
                self._bar = None
            self._write(status="running")

    # ------------------------------------------------------------------ stages
    def start(self, stage: str) -> None:
        if not self.enabled:
            return
        self.current = stage
        self._started[stage] = self._clock()
        if self._bar is not None:
            self._bar.set_postfix_str(stage)
        log.info("progress: starting %s (%d/%d)", stage, self._index(stage) + 1, len(self.stages))
        self._write(status="running")

    def done(self, stage: str) -> None:
        if not self.enabled:
            return
        self.durations[stage] = self._clock() - self._started.get(stage, self._clock())
        self.current = None
        if self._bar is not None:
            self._bar.update(1)
        eta = self.remaining_seconds()
        pending_other = [
            s for s in self.stages if s not in self.fold_like and s not in self.durations
        ]
        log.info(
            "progress: %s done in %s (%d/%d); ~%s remaining%s%s",
            stage,
            _fmt(self.durations[stage]),
            len(self.durations),
            len(self.stages),
            _fmt(eta),
            f", finishing ~{self._finish_clock(eta)}" if eta is not None else "",
            f" (+ {', '.join(pending_other)}: not estimated)" if pending_other else "",
        )
        self._write(status="running")

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[None]:
        self.start(name)
        yield
        self.done(name)

    def finish(self, status: str = "done") -> None:
        if not self.enabled:
            return
        if self._bar is not None:
            self._bar.close()
        log.info("progress: training %s in %s", status, _fmt(self._clock() - self._t0))
        self._write(status=status)

    # ---------------------------------------------------------------- estimate
    def remaining_seconds(self) -> Optional[float]:
        """Projected time for the fold-like stages not finished yet, counting
        the elapsed part of one in progress. ``None`` until one has finished."""
        done = [d for s, d in self.durations.items() if s in self.fold_like]
        if not done:
            return None
        avg = sum(done) / len(done)
        left = [s for s in self.stages if s in self.fold_like and s not in self.durations]
        total = avg * len(left)
        if self.current in self.fold_like and self.current in self._started:
            total -= min(self._clock() - self._started[self.current], avg)
        return max(total, 0.0)

    def _finish_clock(self, eta: float) -> str:
        at = datetime.datetime.fromtimestamp(self._wall() + eta)
        return at.strftime("%H:%M")

    def _index(self, stage: str) -> int:
        return self.stages.index(stage) if stage in self.stages else len(self.durations)

    # -------------------------------------------------------------------- file
    def snapshot(self, status: str) -> Dict[str, Any]:
        eta = self.remaining_seconds() if status == "running" else 0.0
        return {
            "status": status,
            "current_stage": self.current,
            "completed_stages": len(self.durations),
            "total_stages": len(self.stages),
            "elapsed_seconds": round(self._clock() - self._t0, 1),
            "estimated_remaining_seconds": None if eta is None else round(eta, 1),
            "estimated_finish": (
                None
                if eta is None
                else datetime.datetime.fromtimestamp(self._wall() + eta).isoformat(
                    timespec="seconds"
                )
            ),
            "estimate_note": "fold-like stages only; the final fusion/calibration step is not projected",
            "stages": [
                {"name": s, "seconds": round(self.durations[s], 1) if s in self.durations else None}
                for s in self.stages
            ],
        }

    def _write(self, status: str) -> None:
        if not self.output_dir:
            return
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            path = os.path.join(self.output_dir, PROGRESS_FILE)
            tmp = path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(self.snapshot(status), fh, indent=2)
            os.replace(tmp, path)  # atomic: a reader never sees a half-written file
        except OSError as exc:  # progress must never break training
            log.debug("could not write %s: %s", PROGRESS_FILE, exc)
