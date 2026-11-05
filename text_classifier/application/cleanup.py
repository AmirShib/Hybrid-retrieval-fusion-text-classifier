"""Training-data cleanup round (T96).

Finds training rows that may teach the model the wrong thing, applies the
configured actions, and writes every suspicious row to a review file.

Issues, from cheapest to most model-dependent:

- ``reviewed``: a person already decided this row (``review_decisions``). Their
  decision is applied first and the row is exempt from every automatic rule.
- ``junk``: a placeholder ("Rejected"), empty text, or no letters at all.
- ``label_conflict``: the same text appears with different labels.
- ``likely_mislabel``: an honest model is confident in another class, gives
  the row's own label little weight, and enough nearest-neighbour signals agree.
- ``not_shortlisted``: the row's own label did not reach its own shortlist.
- ``ambiguous``: the model splits between the given label and another.

Actions: ``keep`` (flag only), ``drop``, ``relabel`` (to the suggested class),
``quarantine`` (the row stays in the training data -- the fusion model still
learns from it -- but is kept out of the kNN example pools and/or the fine-tunes,
per ``quarantine_from``).

"Honest" matters: the scores come from out-of-fold feature rows plus a fusion
model *cross-fitted by item*, so no row is judged by a model that trained on it.

Safety rules, applied to every automatic action:

- no action may leave a class with fewer than ``min_per_class`` usable rows;
- ``max_changed_fraction`` caps the model-based changes, most confident first;
- with ``rounds > 1`` the round repeats on the cleaned data until nothing changes.

Why likely mislabels are kept by default: on 20k Hebrew COICOP items the flagged
rows mixed real label errors with correct-but-hard rows, and dropping or
relabelling them did not improve accuracy on an untouched test set. Only
*training* rows are ever cleaned; validation/test sets are never touched.
"""

from __future__ import annotations

import logging
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from .._messages import format_preview
from ..config import CleanupConfig
from ..domain import LabeledItem, LabelSpace

log = logging.getLogger(__name__)

REVIEW_COLUMNS = [
    "row",
    "round",
    "text",
    "label",
    "issue",
    "action",
    "suggested_label",
    "label_share",
    "suggested_share",
    "knn_agreeing",
    "reason",
    "decision",
]
ISSUE_PRIORITY = (
    "reviewed",
    "junk",
    "label_conflict",
    "likely_mislabel",
    "not_shortlisted",
    "ambiguous",
)
MODEL_ISSUES = frozenset({"likely_mislabel", "not_shortlisted", "ambiguous"})
# Which kNN signal's top-1 column votes for "the rival class".
KNN_COLUMNS = {"dense": "is_d_knn_top1", "lexical": "is_b_knn_top1", "splade": "is_s_knn_top1"}
_LETTER = re.compile("[A-Za-zא-ת]")
_PLACEHOLDERS = frozenset({"rejected"})
_PAST = {"drop": "dropped", "relabel": "relabeled", "keep": "kept", "quarantine": "quarantined"}

# Out-of-fold pass over the given items -> (feature frame, feature column names).
OofFn = Callable[[List[LabeledItem]], Tuple[pd.DataFrame, List[str]]]


@dataclass
class CleanupResult:
    items: List[LabeledItem]
    quarantined: np.ndarray  # (len(items),) bool: kept, but out of pools/fine-tunes
    review: pd.DataFrame  # one row per flagged input row, REVIEW_COLUMNS
    summary: Dict[str, Any]


def _norm_text(text: str) -> str:
    return " ".join(str(text).split())


# --------------------------------------------------------------------- detection
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
    exact text (whitespace-normalized) carries more than one label."""
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


def detector_params(config: CleanupConfig, seed: int) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "n_estimators": config.cross_fit_trees,
        "max_depth": 6,
        "learning_rate": 0.08,
        "subsample": 0.8,
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": seed,
    }
    params.update(config.detector_params)
    return params


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
        model = XGBClassifier(**detector_params(config, seed))
        model.fit(X[trn], y[trn])
        score[hold] = model.predict_proba(X[hold])[:, 1]
    out = oof[["item_id", "candidate", "is_true"]].copy()
    out["score"] = score
    total = out.groupby("item_id")["score"].transform("sum")
    out["share"] = np.where(total > 0, out["score"] / total.where(total > 0, 1.0), 0.0)
    for col in KNN_COLUMNS.values():
        if col in oof.columns:
            out[col] = oof[col].to_numpy()
    return out


def _model_flags(
    items: Sequence[LabeledItem],
    label_space: LabelSpace,
    config: CleanupConfig,
    oof: pd.DataFrame,
    feature_names: Sequence[str],
    seed: int,
    skip: Set[int],
) -> Dict[int, Dict[str, Any]]:
    shares = honest_shares(oof, feature_names, config, seed)
    own = shares[shares["is_true"] == 1].set_index("item_id")["share"]
    rivals = (
        shares[shares["is_true"] == 0]
        .sort_values("share", ascending=False, kind="stable")
        .drop_duplicates("item_id")
        .set_index("item_id")
    )
    wanted = config.knn_agreement_signals or list(KNN_COLUMNS)
    knn_cols = [KNN_COLUMNS[s] for s in wanted if KNN_COLUMNS[s] in rivals.columns]
    if config.require_knn_agreement and not knn_cols:
        log.warning(
            "cleanup: no kNN signal among %s is computed, so no row can meet "
            "require_knn_agreement; likely_mislabel cannot fire",
            wanted,
        )
    flags: Dict[int, Dict[str, Any]] = {}
    for i in range(len(items)):
        if i in skip:
            continue
        if i not in own.index:
            flags[i] = dict(
                issue="not_shortlisted",
                action=config.not_shortlisted_action,
                reason="the row's own label was not on its shortlist",
            )
            continue
        if i not in rivals.index:
            continue
        q_own, r = float(own.loc[i]), rivals.loc[i]
        q_other = float(r["share"])
        other = label_space.key_at(int(r["candidate"]))
        n_agree = int(sum(float(r[c]) for c in knn_cols))
        detail = dict(
            suggested_label=other, label_share=q_own, suggested_share=q_other, knn_agreeing=n_agree
        )
        agrees = n_agree >= config.min_knn_agreeing or not config.require_knn_agreement
        if q_other >= config.min_suggested_share and q_own <= config.max_label_share and agrees:
            flags[i] = dict(
                issue="likely_mislabel",
                action=config.mislabel_action,
                reason=f"model gives '{other}' {q_other:.0%} vs the label {q_own:.0%}; "
                f"{n_agree} kNN signal(s) agree",
                **detail,
            )
        elif q_own >= config.ambiguous_share and q_other >= config.ambiguous_share:
            flags[i] = dict(
                issue="ambiguous",
                action=config.ambiguous_action,
                reason="the model splits between two classes",
                **detail,
            )
    return flags


# ----------------------------------------------------------------- human review
_DECISIONS = ("keep", "drop", "quarantine")


def read_review_decisions(path: str, label_space: LabelSpace) -> Dict[Tuple[str, str], str]:
    """``(text, label) -> decision`` from a reviewed ``suspicious_rows.csv``.

    Only rows with a non-empty ``decision`` count. A decision is ``keep``,
    ``drop``, ``quarantine`` or ``relabel:<class key>``. Rows are matched on text
    and the label they had when reviewed (not on row number, which shifts as the
    data changes), so a review keeps applying across runs."""
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    missing = {"text", "label", "decision"} - set(df.columns)
    if missing:
        raise ValueError(f"review_decisions file {path!r} lacks column(s) {sorted(missing)}")
    out: Dict[Tuple[str, str], str] = {}
    bad: List[str] = []
    for text, label, decision in zip(df["text"], df["label"], df["decision"]):
        d = decision.strip()
        if not d:
            continue
        low = d.lower()
        if low.startswith("relabel:"):
            target = d.split(":", 1)[1].strip()
            if target not in label_space.keys:
                bad.append(f"relabel target {target!r} is not a class")
                continue
            out[(_norm_text(text), label)] = f"relabel:{target}"
        elif low in _DECISIONS:
            out[(_norm_text(text), label)] = low
        else:
            bad.append(repr(d))
    if bad:
        raise ValueError(
            f"invalid decision(s) in {path!r}: {format_preview(bad)}; use keep, drop, "
            "quarantine or relabel:<class key>"
        )
    return out


# ----------------------------------------------------------------------- driver
def run_cleanup(
    items: Sequence[LabeledItem],
    label_space: LabelSpace,
    config: CleanupConfig,
    min_per_class: int,
    oof_fn: Optional[OofFn] = None,
    seed: int = 0,
    decisions: Optional[Dict[Tuple[str, str], str]] = None,
) -> CleanupResult:
    """Detect, act and repeat for up to ``config.rounds`` rounds.

    ``oof_fn(items)`` runs an out-of-fold pass over the current items and
    enables the model-based checks; without it only reviewed decisions, junk
    and label conflicts apply. Rows are reported by their position in the
    *input* (``row``), whichever round flagged them."""
    current = list(items)
    origin = list(range(len(current)))  # current position -> input row
    quarantined: Set[int] = set()  # input rows
    locked: Set[int] = set()  # input rows decided by a person or already acted on
    report: Dict[int, Dict[str, Any]] = {}  # input row -> latest flag
    budget = (
        int(np.floor(config.max_changed_fraction * len(current)))
        if config.max_changed_fraction is not None
        else None
    )
    used = 0
    rounds_run = 0
    for rnd in range(1, max(config.rounds, 1) + 1):
        rounds_run = rnd
        flags: Dict[int, Dict[str, Any]] = {}  # current position -> flag
        if rnd == 1:
            for i, it in enumerate(current):
                d = (decisions or {}).get((_norm_text(it.text), it.label))
                if d is None:
                    continue
                act, target = (d.split(":", 1) + [""])[:2] if d.startswith("relabel:") else (d, "")
                flags[i] = dict(
                    issue="reviewed",
                    action="relabel" if act == "relabel" else act,
                    reason="decided in review",
                    suggested_label=target,
                    human=True,
                )
            if config.drop_junk:
                for i, why in find_junk(current):
                    flags.setdefault(i, dict(issue="junk", action="drop", reason=why))
            for i, (majority, why) in find_label_conflicts(current).items():
                if i in flags:
                    continue
                if majority is None or config.conflict_action == "keep":
                    reason = why + ("; tie" if majority is None else "")
                    flags[i] = dict(issue="label_conflict", action="keep", reason=reason)
                elif current[i].label != majority:
                    flags[i] = dict(
                        issue="label_conflict",
                        action="relabel",
                        reason=why,
                        suggested_label=majority,
                    )
        if oof_fn is not None and config.detect_mislabels:
            oof, names = oof_fn(current)
            skip = {i for i, o in enumerate(origin) if o in locked} | set(flags)
            for i, f in _model_flags(current, label_space, config, oof, names, seed, skip).items():
                flags[i] = f
        current, origin, changes, used = _apply(
            current, origin, flags, quarantined, locked, report, rnd, min_per_class, budget, used
        )
        if changes == 0 or oof_fn is None or not config.detect_mislabels:
            break
    mask = np.array([o in quarantined for o in origin], dtype=bool)
    review = _review_table(items, report)
    summary = _summary(items, current, mask, review, config, rounds_run, oof_fn)
    log.info(
        "cleanup: %d of %d training rows flagged over %d round(s); %d dropped, %d relabeled, "
        "%d quarantined",
        summary["flagged"],
        len(items),
        rounds_run,
        summary["by_action"].get("dropped", 0),
        summary["by_action"].get("relabeled", 0),
        summary["by_action"].get("quarantined", 0),
    )
    return CleanupResult(current, mask, review, summary)


def _apply(current, origin, flags, quarantined, locked, report, rnd, min_per_class, budget, used):
    """Apply one round's flags with the class floor and the change budget.
    Human decisions bypass both (a person decided). Model-based changes are
    applied most confident first, so the budget keeps the surest ones."""
    usable = Counter(it.label for i, it in enumerate(current) if origin[i] not in quarantined)

    def rank(i: int) -> Tuple[int, float]:
        f = flags[i]
        return (ISSUE_PRIORITY.index(f["issue"]), -(f.get("suggested_share") or 0.0))

    new_labels: List[Optional[str]] = [it.label for it in current]
    changes = 0
    for i in sorted(flags, key=rank):
        f = flags[i]
        act, label, human = f["action"], current[i].label, f.get("human", False)
        if act in ("drop", "relabel", "quarantine") and not human:
            if usable[label] - 1 < min_per_class:
                f["reason"] += f"; not applied: '{label}' would fall below {min_per_class} rows"
                act = "keep"
            elif f["issue"] in MODEL_ISSUES and budget is not None and used >= budget:
                f["reason"] += "; not applied: max_changed_fraction reached"
                act = "keep"
            elif f["issue"] in MODEL_ISSUES:
                used += 1
        if act == "relabel" and not f.get("suggested_label"):
            act = "keep"  # nothing to relabel to (e.g. a not-shortlisted row)
        if act in ("drop", "relabel", "quarantine"):
            changes += 1
            usable[label] -= 1
            locked.add(origin[i])
            if act == "drop":
                new_labels[i] = None
            elif act == "relabel":
                new_labels[i] = f["suggested_label"]
                usable[f["suggested_label"]] += 1
            else:
                quarantined.add(origin[i])
        if human:
            locked.add(origin[i])  # a person decided: no automatic rule may revisit it
        f["action"] = act
        report[origin[i]] = {**f, "round": rnd}
    keep = [i for i, lab in enumerate(new_labels) if lab is not None]
    current = [LabeledItem(current[i].text, new_labels[i]) for i in keep]  # type: ignore[arg-type]
    origin = [origin[i] for i in keep]
    return current, origin, changes, used


def _review_table(items: Sequence[LabeledItem], report: Dict[int, Dict[str, Any]]) -> pd.DataFrame:
    rows = [
        {
            "row": o,
            "round": f["round"],
            "text": items[o].text,
            "label": items[o].label,
            "issue": f["issue"],
            "action": _PAST[f["action"]],
            "suggested_label": f.get("suggested_label", ""),
            "label_share": f.get("label_share"),
            "suggested_share": f.get("suggested_share"),
            "knn_agreeing": f.get("knn_agreeing"),
            "reason": f["reason"],
            "decision": "",  # for the reviewer: keep / drop / quarantine / relabel:<class>
        }
        for o, f in report.items()
    ]
    review = pd.DataFrame(rows, columns=REVIEW_COLUMNS)
    if len(review):
        order = {k: n for n, k in enumerate(ISSUE_PRIORITY)}
        review = (
            review.assign(_p=review["issue"].map(order), _s=-review["suggested_share"].fillna(0))
            .sort_values(["_p", "_s", "row"], kind="stable")
            .drop(columns=["_p", "_s"])
            .reset_index(drop=True)
        )
    return review


def _summary(items, current, mask, review, config, rounds_run, oof_fn) -> Dict[str, Any]:
    by_action: Dict[str, int] = {k: int(v) for k, v in review["action"].value_counts().items()}
    return {
        "rows_in": len(items),
        "rows_out": len(current),
        "quarantined": int(mask.sum()),
        "quarantine_from": list(config.quarantine_from) if mask.any() else [],
        "flagged": int(len(review)),
        "by_issue": {k: int(v) for k, v in review["issue"].value_counts().items()},
        "by_action": by_action,
        "rounds_run": rounds_run,
        "mislabel_action": config.mislabel_action,
        "model_checks": oof_fn is not None and config.detect_mislabels,
    }


def write_review_file(review: pd.DataFrame, directory: str, filename: str) -> str:
    """Write the suspicious rows as CSV, UTF-8 with BOM so Excel shows Hebrew.
    Fill the ``decision`` column and pass the file back as
    ``cleanup.review_decisions`` to apply the review on the next run."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    review.to_csv(path, index=False, encoding="utf-8-sig")
    return path
