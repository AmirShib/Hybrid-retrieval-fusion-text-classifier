"""T83 — device execution profile.

Dev-only script (not packaged, not imported by the library). Measures wall-clock
per pipeline stage across a synthetic-corpus scale grid, so Tier 8 (T84-T86) has
numbers instead of a hunch before moving any kernel to a device.

No source under text_classifier/ is modified: every stage is timed by calling the
same functions the real pipeline calls (imported directly, including a few
module-private numpy helpers in application/features.py), on the same shapes the
real pipeline would produce at that scale. A second, black-box measurement of
FeatureAssembler.assemble() on the identical inputs cross-checks that the summed
stage times aren't hiding unattributed cost (T83 acceptance criterion).

Usage:
    python -m scripts.profile_devices                      # default grid
    python -m scripts.profile_devices --quick               # small smoke grid
    python -m scripts.profile_devices --n-items 1000,10000 --n-classes 30,500

Output: JSON to --out-json (default docs/device-profile.json) and a markdown
table to --out-md (default docs/device-profile.md). GPU configurations are
recorded with status="skipped" (not failed) on a host with no CUDA device or no
downloadable sentence-transformers model — this host has torch but no CUDA, so
every run here reports skipped GPU rows by design, not by bug.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from text_classifier.application.features import (
    _row_margin,
    _row_minmax,
    _row_rank,
    _scatter_knn,
    _topn_mask,
    FeatureAssembler,
)
from text_classifier.config import FusionConfig, PipelineConfig, RetrievalConfig
from text_classifier.datasets import make_synthetic
from text_classifier.domain import CandidatePolicy, LabelSpace
from text_classifier.infrastructure import (
    DenseRetrieverAdapter,
    HashingEncoder,
    LexicalRetrieverAdapter,
    build_fusion,
)
from text_classifier.infrastructure.device import cuda_available

logging.basicConfig(level=logging.WARNING)  # the harness's own prints carry the signal

STAGES = [
    "1_encode",
    "2_prototypes_and_freq",
    "3_description_prototype_similarity",
    "4_dense_topk",
    "5_bm25",
    "6_scatter_knn",
    "7_ranks_margins_topn_minmax",
    "8_dataframe_construction",
    "9_fusion_predict",
    "10_calibration_thresholds",
]

N_MEDIAN_RUNS = 3


@dataclass
class DeviceConfig:
    name: str
    encoder_device: str  # "cpu" | "cuda"
    fusion_device: str  # "cpu" | "cuda"

    def available(self) -> bool:
        needs_cuda = self.encoder_device == "cuda" or self.fusion_device == "cuda"
        return (not needs_cuda) or cuda_available()

    def skip_reason(self) -> Optional[str]:
        if self.available():
            return None
        return (
            "no CUDA device visible (torch.cuda.is_available() is False on this host); "
            "a real GPU-encoder config additionally needs a downloadable "
            "sentence-transformers model, which this offline harness does not fetch"
        )


DEVICE_CONFIGS = [
    DeviceConfig("cpu_only", "cpu", "cpu"),
    DeviceConfig("gpu_encoder_cpu_rest", "cuda", "cpu"),
    DeviceConfig("gpu_encoder_gpu_fusion", "cuda", "cuda"),
]


def _median_call(fn, n=N_MEDIAN_RUNS) -> Tuple[Any, float]:
    """Call fn() n times, return (last result, median seconds)."""
    times = []
    result = None
    for _ in range(n):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return result, statistics.median(times)


def _build_dataset(n_items: int, n_classes: int, seed: int):
    per_class = max(8, n_items // n_classes)
    label_space, items = make_synthetic(n_classes=n_classes, per_class=per_class, seed=seed)
    return label_space, items


def _profile_one_config(
    label_space: LabelSpace,
    items,
    cfg: PipelineConfig,
    device: DeviceConfig,
    query_frac: float = 0.1,
) -> Dict[str, Any]:
    if not device.available():
        return {"status": "skipped", "reason": device.skip_reason()}

    texts = [it.text for it in items]
    y = np.asarray(label_space.encode_labels([it.label for it in items]), dtype=np.int64)
    n = len(texts)
    n_query = max(1, int(n * query_frac))
    rng = np.random.default_rng(0)
    q_pos = rng.choice(n, size=n_query, replace=False)
    q_texts = [texts[i] for i in q_pos]

    encoder = HashingEncoder(dim=128)  # torch-free CPU stand-in; see module docstring
    stage_times: Dict[str, float] = {}

    def _measure(key: str, fn):
        """Run fn() once for its real return value, then re-run it
        N_MEDIAN_RUNS times purely for timing (median, not first-call/cold-cache
        noise) and record the median under stage_times[key]. fn must be a pure
        read of already-built state -- safe to call repeatedly."""
        result = fn()
        _, median_t = _median_call(fn, n=N_MEDIAN_RUNS)
        stage_times[key] = median_t
        return result

    def _encode_all():
        return (
            encoder.encode_documents(texts),
            encoder.encode_documents(label_space.descriptions),
            encoder.encode_queries(q_texts),
        )

    pool_emb, desc_emb, q_emb = _measure("1_encode", _encode_all)

    dense = DenseRetrieverAdapter.build_from_embeddings(
        pool_emb, y, desc_emb, label_space, cfg.retrieval
    )
    lexical = LexicalRetrieverAdapter.build(texts, y, label_space, cfg.retrieval)

    C = label_space.size
    k = cfg.retrieval.k_neighbors
    n_top = cfg.candidate_top_n

    from text_classifier.infrastructure.retrieval import _prototypes_and_freq

    # Already computed inside DenseRetrieverAdapter.build_from_embeddings above;
    # re-time it standalone (identical inputs) so this stage's cost is isolated
    # rather than folded into index construction.
    _measure("2_prototypes_and_freq", lambda: _prototypes_and_freq(pool_emb, y, C))

    def _desc_proto_sim():
        return (
            np.asarray(dense.description_similarity(q_emb), dtype=np.float64),
            np.asarray(dense.prototype_similarity(q_emb), dtype=np.float64),
        )

    desc_d, proto = _measure("3_description_prototype_similarity", _desc_proto_sim)

    dn_lab, dn_sim = _measure("4_dense_topk", lambda: dense.knn_example_labels(q_emb, k))

    def _bm25():
        lab, sco = lexical.knn_example_labels(q_texts, k)
        raw = np.asarray(lexical.description_score(q_texts), dtype=np.float64)
        return lab, sco, np.where(raw > 0, raw, np.nan)

    bn_lab, bn_sco, bdesc = _measure("5_bm25", _bm25)

    def _scatter():
        return _scatter_knn(dn_lab, dn_sim, C), _scatter_knn(bn_lab, bn_sco, C)

    (d_sum, d_max, d_cnt), (b_sum, b_max, b_cnt) = _measure("6_scatter_knn", _scatter)

    mask = (
        _topn_mask(desc_d, n_top)
        | _topn_mask(proto, n_top)
        | _topn_mask(bdesc, n_top, positive_only=True)
        | _topn_mask(d_sum, n_top)
        | _topn_mask(b_sum, n_top)
    )
    rows, cols = np.nonzero(mask)

    def _leaf_ops():
        _row_rank(desc_d, mask)
        _row_rank(bdesc, mask)
        _row_rank(d_sum, mask)
        _row_rank(b_sum, mask)
        _row_minmax(desc_d, mask)
        _row_minmax(bdesc, mask)
        _row_margin(desc_d, mask)
        _row_margin(proto, mask)
        _row_margin(d_sum, mask)
        _row_margin(bdesc, mask)
        _row_margin(b_sum, mask)

    _measure("7_ranks_margins_topn_minmax", _leaf_ops)

    import pandas as pd

    n_cand = rows.shape[0]
    n_cols = len(cfg_feature_names(cfg))

    def _build_df():
        data = {
            f"f{i}": np.random.default_rng(i).random(n_cand).astype(np.float32)
            for i in range(n_cols)
        }
        return pd.DataFrame(data)

    _measure("8_dataframe_construction", _build_df)

    # ---- fusion + calibration (fit small, time predict/transform: the steady-state cost) ----
    fusion_cfg = FusionConfig(kind="xgboost")
    fusion_cfg.xgb_params = dict(fusion_cfg.xgb_params)
    fusion_cfg.xgb_params.update({"n_estimators": 50, "device": device.fusion_device})
    fusion = build_fusion(fusion_cfg)
    n_train = max(64, min(4096, n_cand))
    X_train = rng.random((n_train, n_cols)).astype(np.float32)
    y_train = rng.integers(0, 2, size=n_train)
    fusion.fit(X_train, y_train)
    X_score = rng.random((max(n_cand, 1), n_cols)).astype(np.float32)
    raw = _measure("9_fusion_predict", lambda: fusion.predict_proba(X_score))

    from text_classifier.infrastructure import build_calibrator
    from text_classifier.application.scoring import top_per_item

    calibrator = build_calibrator(cfg.calibration)
    calibrator.fit(raw, rng.integers(0, 2, size=raw.shape[0]))
    scored = pd.DataFrame(
        {
            "item_id": rows if rows.size else np.zeros(1, dtype=np.int64),
            "candidate": cols if cols.size else np.zeros(1, dtype=np.int64),
            "conf": calibrator.transform(raw)[: max(rows.size, 1)],
            "is_true": np.zeros(max(rows.size, 1), dtype=np.int64),
        }
    )
    _measure(
        "10_calibration_thresholds",
        lambda: (calibrator.transform(raw), top_per_item(scored)),
    )

    # ---- cross-check: black-box assemble() wall time vs the stage sum above ----
    assembler = FeatureAssembler(label_space, CandidatePolicy(n_top))
    _, assemble_wall = _median_call(
        lambda: assembler.assemble(
            q_texts, q_emb, dense, lexical, k, query_ids=np.arange(len(q_texts))
        ),
        n=N_MEDIAN_RUNS,
    )
    stage_sum = sum(stage_times.values())
    # assemble() does its own encode-free retrieval calls (stages 2-8 equivalent)
    # but not stage 1 (encode) or 9/10 (fusion/calibration, which happen after
    # assemble() returns) -- compare against that subset only.
    comparable = sum(
        stage_times[s]
        for s in STAGES
        if s not in ("1_encode", "9_fusion_predict", "10_calibration_thresholds")
    )

    return {
        "status": "ok",
        "n_items": n,
        "n_classes": C,
        "n_query": n_query,
        "n_candidates": int(n_cand),
        "stage_seconds": stage_times,
        "stage_sum_seconds": stage_sum,
        "assemble_wall_seconds": assemble_wall,
        "assemble_vs_comparable_stage_sum_ratio": (
            comparable / assemble_wall if assemble_wall > 0 else None
        ),
    }


def cfg_feature_names(cfg: PipelineConfig) -> List[str]:
    from text_classifier.domain import FEATURE_NAMES

    return [n for n in FEATURE_NAMES if n not in cfg.fusion.drop_features]


def _oof_timing(label_space: LabelSpace, items, n_folds: int) -> Dict[str, float]:
    """Real end-to-end TrainingPipeline.run() timing: per-fold average vs whole-run
    total, over the shared-encoder OOF path (HashingEncoder, torch-free, offline).
    """
    from text_classifier.application.training import TrainingPipeline

    cfg = PipelineConfig()
    cfg.encoder.kind = "hashing"
    cfg.fusion.xgb_params = dict(cfg.fusion.xgb_params)
    cfg.fusion.xgb_params["n_estimators"] = 50
    cfg.training.n_folds = n_folds
    cfg.training.store_corpus = False

    pipeline = TrainingPipeline(cfg, shared_encoder=HashingEncoder(dim=128))
    t0 = time.perf_counter()
    pipeline.run(items, label_space)
    total = time.perf_counter() - t0
    return {"n_folds": n_folds, "whole_run_seconds": total, "per_fold_seconds": total / n_folds}


def run_grid(
    n_items_grid: List[int], n_classes_grid: List[int], seed: int, oof_folds: Optional[int]
) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    for n_classes in n_classes_grid:
        for n_items in n_items_grid:
            if n_items < n_classes * 2:
                continue  # not enough items to populate the class grid meaningfully
            label_space, items = _build_dataset(n_items, n_classes, seed)
            cfg = PipelineConfig()
            cfg.retrieval = RetrievalConfig()
            entry: Dict[str, Any] = {
                "n_items_requested": n_items,
                "n_classes": n_classes,
                "configs": {},
            }
            for device in DEVICE_CONFIGS:
                entry["configs"][device.name] = _profile_one_config(label_space, items, cfg, device)
            if oof_folds:
                entry["oof_timing"] = _oof_timing(label_space, items, oof_folds)
            results.append(entry)
            print(f"done: n_items~{n_items} n_classes={n_classes}")
    return {"grid": results, "cuda_available": cuda_available()}


def _to_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Device profile (T83)",
        "",
        f"`cuda_available()` on this host: `{report['cuda_available']}`",
        "",
    ]
    for entry in report["grid"]:
        lines.append(f"## n_items~{entry['n_items_requested']}, n_classes={entry['n_classes']}")
        for name, res in entry["configs"].items():
            if res["status"] == "skipped":
                lines.append(f"- **{name}**: skipped ({res['reason']})")
                continue
            lines.append(
                f"- **{name}** (n_query={res['n_query']}, n_candidates={res['n_candidates']}):"
            )
            lines.append("")
            lines.append("  | stage | seconds |")
            lines.append("  |---|---|")
            for stage in STAGES:
                lines.append(f"  | {stage} | {res['stage_seconds'].get(stage, 0.0):.4f} |")
            lines.append(f"  | **sum** | **{res['stage_sum_seconds']:.4f}** |")
            ratio = res["assemble_vs_comparable_stage_sum_ratio"]
            lines.append(
                f"  - assemble() wall = {res['assemble_wall_seconds']:.4f}s; "
                f"comparable stage-sum / assemble() wall = {ratio:.3f}"
                if ratio
                else ""
            )
            lines.append("")
        if "oof_timing" in entry:
            oof = entry["oof_timing"]
            lines.append(
                f"- OOF run (n_folds={oof['n_folds']}): whole_run={oof['whole_run_seconds']:.3f}s, "
                f"per_fold={oof['per_fold_seconds']:.3f}s"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--n-items", type=str, default="1000,10000,100000")
    p.add_argument("--n-classes", type=str, default="30,500,5000")
    p.add_argument(
        "--quick", action="store_true", help="small smoke grid, overrides --n-items/--n-classes"
    )
    p.add_argument("--oof-folds", type=int, default=3, help="0 disables the end-to-end OOF timing")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-json", type=str, default="docs/device-profile.json")
    p.add_argument("--out-md", type=str, default="docs/device-profile.md")
    args = p.parse_args()

    if args.quick:
        n_items_grid = [200, 1000]
        n_classes_grid = [10, 30]
    else:
        n_items_grid = [int(x) for x in args.n_items.split(",")]
        n_classes_grid = [int(x) for x in args.n_classes.split(",")]

    report = run_grid(n_items_grid, n_classes_grid, args.seed, args.oof_folds or None)

    with open(args.out_json, "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    with open(args.out_md, "w") as fh:
        fh.write(_to_markdown(report))
    print(f"wrote {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
