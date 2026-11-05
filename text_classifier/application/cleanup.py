"""Training-data cleanup round (T96).

Finds training rows that may teach the model the wrong thing, applies the
*safe* automatic actions, and writes every suspicious row to a review file.

Issues, from cheapest to most model-dependent:

- ``junk``: a placeholder ("Rejected"), empty text, or no letters at all.
  Automatic action: **drop**.
- ``label_conflict``: the same text appears with different labels. Automatic
  action: relabel the minority rows to the majority label; a tie is kept and
  left for review.
- ``likely_mislabel``: an honest model is confident in another class, gives
  the row's own label little weight, and a nearest-neighbour signal agrees with
  it. Default action: **keep** (flag only) -- see below.
- ``ambiguous``: the model splits between the given label and another; usually
  a real boundary case, flagged for review, never changed.
- ``not_shortlisted``: the row's own label did not reach its own candidate
  shortlist; flagged for review.

"Honest" matters: the scores come from the out-of-fold feature rows plus a
fusion model *cross-fitted by item*, so no row is judged by a model that trained
on it. Without that, a model agrees with whatever labels it memorized.

Why mislabels are kept by default: on 20k Hebrew COICOP items the flagged rows
were a mix of real label errors and correct-but-hard rows the model got wrong,
and the model's suggested label was itself often wrong. Dropping or relabelling
them changed accuracy on an untouched test set by -0.2 to -0.07 points (within
noise). ``mislabel_action`` can be set to ``drop`` or ``relabel`` once a review
of the file supports it.

Only *training* rows are cleaned. External validation/test sets are never
touched: cleaning the evaluation data would raise the reported accuracy without
improving the model.
"""

from __future__ import annotations

import logging
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..config import CleanupConfig
from ..domain import LabeledItem, LabelSpace

log = logging.getLogger(__name__)

REVIEW_COLUMNS = [
    "row",
    "text",
    "label",
    "issue",
    "action",
    "suggested_label",
    "label_share",
    "suggested_share",
    "knn_agrees",
    "reason",
]
ISSUE_PRIORITY = ("junk", "label_conflict", "likely_mislabel", "not_shortlisted", "ambiguous")
_LETTER = re.compile("[A-Za-zא-ת]")
_PLACEHOLDERS = frozenset({"rejected"})


@dataclass
class CleanupResult:
    items: List[LabeledItem]
    review: pd.DataFrame  # one row per flagged training row, REVIEW_COLUMNS
    summary: Dict[str, Any]


def _norm_text(text: str) -> str:
    return " ".join(str(text).split())


def find_junk(items: Sequence[LabeledItem]) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    for i, it in enumerate(items):
        t = _norm_text(it.text)
        if t.lower() in _PLACEHOLDERS:
            out.append((i, "placeholder text"))
        elif not _LETTER.search(t):
            out.append((i, "no Hebrew or Latin letters"))
    return out


def find_label_conflicts(items: Sequence[LabeledItem]) -> Dict[int, Tuple[Optional[str], str]]:
    """Row -> (majority label or None on a tie, reason), for every row whose
    exact text carries more than one label."""
    groups: Dict[str, List[int]] = {}
    for i, it in enumerate(items):
        groups.setdefault(_norm_text(it.text), []).append(i)
    out: Dict[int, Tuple[Optional[str], str]] = {}
    for rows in groups.values():
        counts = Counter(items[i].label for i in rows)
        if len(counts) < 2:
            continue
        (top, n_top), (_, n_second) = counts.most_common(2)
        majority = top if n_top > n_second else None
        desc = ", ".join(f"{lab} x{k}" for lab, k in counts.most_common())
        for i in rows:
            out[i] = (majority, f"same text has labels: {desc}")
    return out


def honest_shares(
    oof: pd.DataFrame, feature_names: Sequence[str], config: CleanupConfig, seed: int = 0
) -> pd.DataFrame:
    """Per (item, candidate) share of the item's total score, from a fusion
    model cross-fitted by item over the out-of-fold rows: each item's rows are
    scored by a model trained only on other items."""
    from sklearn.model_selection import GroupKFold
    from xgboost import XGBClassifier

    X = oof[list(feature_names)].to_numpy(dtype=np.float32)
    y = oof["is_true"].to_numpy()
    groups = oof["item_id"].to_numpy()
    score = np.zeros(len(oof))
    n_splits = min(config.cross_fit_folds, int(pd.unique(groups).size))
    for trn, hold in GroupKFold(n_splits).split(X, y, groups):
        model = XGBClassifier(
            n_estimators=config.cross_fit_trees,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.8,
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
        )
        model.fit(X[trn], y[trn])
        score[hold] = model.predict_proba(X[hold])[:, 1]
    out = oof[["item_id", "candidate", "is_true"]].copy()
    out["score"] = score
    total = out.groupby("item_id")["score"].transform("sum")
    out["share"] = np.where(total > 0, out["score"] / total.where(total > 0, 1.0), 0.0)
    for col in ("is_d_knn_top1", "is_b_knn_top1", "is_s_knn_top1"):
        if col in oof.columns:
            out[col] = oof[col].to_numpy()
    return out


def run_cleanup(
    items: Sequence[LabeledItem],
    label_space: LabelSpace,
    config: CleanupConfig,
    min_per_class: int,
    oof: Optional[pd.DataFrame] = None,
    feature_names: Optional[Sequence[str]] = None,
    seed: int = 0,
) -> CleanupResult:
    """Detect issues, apply the configured actions, and build the review table.

    ``oof`` (the out-of-fold feature frame for exactly ``items``, ``item_id`` =
    row index) enables the model-based checks; without it only junk and label
    conflicts are checked. An action that would leave a class with fewer than
    ``min_per_class`` rows is skipped (the row is kept and the reason says so),
    so cleanup can never make training impossible."""
    items = list(items)
    labels = [it.label for it in items]
    flags: Dict[int, Dict[str, Any]] = {}

    def flag(i: int, issue: str, action: str, reason: str, **extra: Any) -> None:
        if i in flags:  # first (most certain) issue wins
            return
        flags[i] = {"issue": issue, "action": action, "reason": reason, **extra}

    if config.drop_junk:
        for i, why in find_junk(items):
            flag(i, "junk", "drop", why)
    for i, (majority, why) in find_label_conflicts(items).items():
        if majority is None or config.conflict_action == "keep":
            flag(i, "label_conflict", "keep", why + ("; tie" if majority is None else ""))
        elif labels[i] != majority:
            flag(i, "label_conflict", "relabel", why, suggested_label=majority)

    if oof is not None and feature_names is not None and config.detect_mislabels:
        shares = honest_shares(oof, feature_names, config, seed)
        own = shares[shares["is_true"] == 1].set_index("item_id")["share"]
        rivals = (
            shares[shares["is_true"] == 0]
            .sort_values("share", ascending=False, kind="stable")
            .drop_duplicates("item_id")
            .set_index("item_id")
        )
        knn_cols = [c for c in ("is_d_knn_top1", "is_b_knn_top1", "is_s_knn_top1") if c in rivals]
        for i in range(len(items)):
            if i not in own.index:
                flag(i, "not_shortlisted", "keep", "the row's own label was not on its shortlist")
                continue
            if i not in rivals.index:
                continue
            q_own = float(own.loc[i])
            r = rivals.loc[i]
            q_other = float(r["share"])
            other = label_space.key_at(int(r["candidate"]))
            agrees = bool(knn_cols) and bool(sum(float(r[c]) for c in knn_cols) >= 1)
            detail = dict(
                suggested_label=other, label_share=q_own, suggested_share=q_other, knn_agrees=agrees
            )
            if (
                q_other >= config.min_suggested_share
                and q_own <= config.max_label_share
                and (agrees or not config.require_knn_agreement)
            ):
                flag(
                    i,
                    "likely_mislabel",
                    config.mislabel_action,
                    f"model gives '{other}' {q_other:.0%} vs the label {q_own:.0%}",
                    **detail,
                )
            elif q_own >= config.ambiguous_share and q_other >= config.ambiguous_share:
                flag(i, "ambiguous", "keep", "the model splits between two classes", **detail)

    # Apply, never pushing a class below the minimum training needs.
    counts = Counter(labels)
    new_labels: List[Optional[str]] = list(labels)
    for i in sorted(flags):
        f = flags[i]
        if f["action"] not in ("drop", "relabel"):
            continue
        if counts[labels[i]] - 1 < min_per_class:
            f["reason"] += f"; not applied: '{labels[i]}' would fall below {min_per_class} rows"
            f["action"] = "keep"
            continue
        counts[labels[i]] -= 1
        if f["action"] == "drop":
            new_labels[i] = None
        else:
            new_labels[i] = f["suggested_label"]
            counts[f["suggested_label"]] += 1
    cleaned = [LabeledItem(it.text, lab) for it, lab in zip(items, new_labels) if lab is not None]
    applied = {"drop": "dropped", "relabel": "relabeled", "keep": "kept"}
    review = pd.DataFrame(
        [
            {
                "row": i,
                "text": items[i].text,
                "label": labels[i],
                "issue": f["issue"],
                "action": applied[f["action"]],
                "suggested_label": f.get("suggested_label", ""),
                "label_share": f.get("label_share"),
                "suggested_share": f.get("suggested_share"),
                "knn_agrees": f.get("knn_agrees"),
                "reason": f["reason"],
            }
            for i, f in sorted(flags.items())
        ],
        columns=REVIEW_COLUMNS,
    )
    # Most actionable first: certain problems, then likely mislabels by how
    # confident the model is, then review-only boundary cases.
    if len(review):
        order = {k: n for n, k in enumerate(ISSUE_PRIORITY)}
        review = (
            review.assign(_p=review["issue"].map(order), _s=-review["suggested_share"].fillna(0))
            .sort_values(["_p", "_s", "row"], kind="stable")
            .drop(columns=["_p", "_s"])
            .reset_index(drop=True)
        )
    by_action: Dict[str, int] = {k: int(v) for k, v in review["action"].value_counts().items()}
    summary: Dict[str, Any] = {
        "rows_in": len(items),
        "rows_out": len(cleaned),
        "flagged": int(len(review)),
        "by_issue": {k: int(v) for k, v in review["issue"].value_counts().items()},
        "by_action": by_action,
        "mislabel_action": config.mislabel_action,
        "model_checks": oof is not None and config.detect_mislabels,
    }
    log.info(
        "cleanup: %d of %d training rows flagged; %d dropped, %d relabeled",
        summary["flagged"],
        len(items),
        by_action.get("dropped", 0),
        by_action.get("relabeled", 0),
    )
    return CleanupResult(cleaned, review, summary)


def write_review_file(review: pd.DataFrame, directory: str, filename: str) -> str:
    """Write the suspicious rows as CSV, UTF-8 with BOM so Excel shows Hebrew."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    review.to_csv(path, index=False, encoding="utf-8-sig")
    return path
