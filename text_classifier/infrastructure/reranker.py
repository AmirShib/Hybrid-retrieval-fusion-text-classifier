"""Cross-encoder rerank signal (T33).

Three things live here, in dependency order:

1. ``TokenOverlapReranker`` — a ``PairwiseReranker`` that needs no model and no
   network, so the whole path is exercised offline and in CI. The reranker
   analogue of ``HashingEncoder``.
2. Evidence rendering — turning a ``ClassDefinition`` plus an ``EvidenceSpec``
   list into the document text a pair is scored against, with per-query
   selection among a view's entries.
3. ``CrossEncoderSignalProvider`` — the *second-stage* ``SignalProvider``
   (``needs_candidates = True``) that decides which pairs are worth scoring,
   calls the reranker once per document, and shapes the result into ``(b, C)``
   matrices the assembler derives columns from exactly as it does for the five
   built-in signals.

The split matters: (1) answers "how well do these two texts go together" and
nothing else, so a stock cross-encoder, a fine-tuned one and an instructed LLM
judge all drop into the same slot; (3) owns every decision about *which* pairs,
how documents are composed, and what the columns are called.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..config import CrossEncoderConfig, CrossEncoderDocument, EvidenceSpec
from ..domain import (
    ArrayOps,
    ClassDefinition,
    PairwiseReranker,
    SignalContext,
    SignalMatrix,
    SignalProvider,
)
from .array_ops import NumpyArrayOps
from .signals import _argmax_or_missing, _topn_mask


def _tokens(text: str) -> frozenset:
    return frozenset(str(text).lower().split())


class TokenOverlapReranker(PairwiseReranker):
    """Offline, dependency-free stand-in for a real cross-encoder.

    Scores a pair by the smoothed log-odds of its token Jaccard overlap:
    positive when the two texts share most of their vocabulary, negative when
    they share little, unbounded on both sides — the same *shape* a real
    cross-encoder's logits have, which is what the fusion model and the
    downstream calibration are written against.

    Deterministic by construction (exact set arithmetic, no hashing and no
    learned weights), so it is byte-identical across processes, versions and
    platforms without pinning anything. It is emphatically **not** a semantic
    reranker — it cannot see that "car" and "automobile" are related, and it
    cannot interpret negation any better than BM25 can. Its job is to prove the
    plumbing, exactly as ``HashingEncoder`` does for the encoder path.
    """

    def score(self, pairs: Sequence[Tuple[str, str]]) -> np.ndarray:
        out = np.empty(len(pairs), dtype=np.float32)
        for i, (query, document) in enumerate(pairs):
            q, d = _tokens(query), _tokens(document)
            shared = len(q & d)
            # +0.5 smoothing keeps both ends finite: no overlap is a strongly
            # negative logit rather than -inf, which would be indistinguishable
            # from "not scored" once it reaches the feature matrix.
            out[i] = np.log((shared + 0.5) / (len(q | d) - shared + 0.5))
        return out

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "reranker.json"), "w") as fh:
            json.dump({"kind": "token-overlap"}, fh)

    @classmethod
    def load(cls, path: str, **kwargs) -> "TokenOverlapReranker":
        return cls()


# --------------------------------------------------------------- evidence
# View name -> the ClassDefinition texts it offers. Tuple-valued views return
# several short strings (so `select` has something to choose between); scalar
# views return a single-entry tuple. Absent evidence is `()` — an *absent* view,
# never an empty string, so it can propagate as NaN rather than as a score
# against nothing (CLAUDE.md's NaN invariant, at the text layer).
_EVIDENCE = {
    "description": lambda d: (d.description,),
    "core": lambda d: (d.core_view(),),
    "examples": lambda d: d.examples,
    "inclusions": lambda d: d.inclusions,
    "exclusions": lambda d: d.exclusions,
    "siblings": lambda d: d.sibling_distinctions,
}


# One resolved evidence slot for one class. Either a finished string (the slot
# does not depend on the query) or the candidates to choose between at scoring
# time, pre-tokenized.
_Slot = Union[str, Tuple[EvidenceSpec, Tuple[Tuple[str, frozenset], ...]]]


def _plan_class(definition: ClassDefinition, doc: CrossEncoderDocument) -> List[_Slot]:
    """Resolve one class's slots for one document, independent of any query.

    Everything that does not vary per query is finished here — including the
    ``"first"``/``"all"`` modes, which never consult the query at all — and the
    entries of a ``"best"`` slot are tokenized once. This runs ``C`` times per
    chunk rather than once per scored *pair*, which is the difference between
    tokenizing a taxonomy's evidence a few dozen times and tokenizing it tens of
    thousands of times: the selection input depends only on the class, so doing
    it per pair was pure repeat work."""
    plan: List[_Slot] = []
    for spec in doc.evidence:
        entries = tuple(e for e in _EVIDENCE[spec.view](definition) if e)
        if not entries:
            plan.append("")  # absent view -> contributes nothing, never a bare label
        elif spec.select == "all":
            plan.append(spec.label + "; ".join(entries)[: spec.max_chars])
        elif spec.select == "first" or len(entries) == 1:
            plan.append(spec.label + entries[0][: spec.max_chars])
        else:  # "best" -- the only query-dependent mode
            plan.append((spec, tuple((e, _tokens(e)) for e in entries)))
    return plan


def _render_plan(plan: Sequence[_Slot], doc: CrossEncoderDocument, query_tokens: frozenset) -> str:
    """Finish a planned document for one query, or ``""`` when this class has
    nothing to say through any of the document's views.

    An empty return is the caller's signal to leave the cell NaN rather than
    score against a document that does not exist.

    ``"best"`` is the query-adaptive mode and the reason this step takes a query
    at all: a class with several example phrases has several different doors in,
    and which one fits depends on the item being classified. For a negative view
    it picks the *hardest* negative — the exclusion most like the query is the
    one actually at risk of being confused with it.

    Selection is lexical (token overlap) rather than embedding-based because
    ``SignalProviderSpec.build`` receives no encoder; see T33's design note. It
    is a defensible selector for picking among a handful of short phrases, and
    it keeps this provider stateless with nothing to persist."""
    parts = []
    for slot in plan:
        if isinstance(slot, str):
            if slot:
                parts.append(slot)
            continue
        spec, candidates = slot
        text, _ = max(candidates, key=lambda e: len(query_tokens & e[1]))
        parts.append(spec.label + text[: spec.max_chars])
    if not parts:
        return ""
    body = doc.join.join(parts)
    return f"{doc.instruction}{doc.join}{body}" if doc.instruction else body


def _render(definition: ClassDefinition, doc: CrossEncoderDocument, query_tokens: frozenset) -> str:
    """``_plan_class`` + ``_render_plan`` for a single class — the one-shot form,
    for callers with no chunk to amortize the plan over (and for tests)."""
    return _render_plan(_plan_class(definition, doc), doc, query_tokens)


class CrossEncoderSignalProvider(SignalProvider):
    """Reranks shortlisted candidates with a ``PairwiseReranker`` (T33).

    Runs in the assembler's *second* round (``needs_candidates = True``): it
    scores far too slowly to look at every class, so it can only run once the
    cheap signals have produced a shortlist. It therefore declares no
    ``candidate_features`` — it reorders the shortlist and can never extend it,
    which means candidate recall (the ceiling on system accuracy) is unchanged
    by switching this signal on.

    One ``(b, C)`` matrix per configured document, NaN everywhere except the
    cells actually reranked. Within the shortlist, NaN means "not reranked"
    (outside ``top_k``, or this class has no text for that document's views);
    outside it, the pair is not in the frame at all. Each matrix then collects
    the assembler's full generic derivation set — raw/missing/rank/norm/margin +
    per-query gap + is-top1 — so seven columns per document arrive with no
    arithmetic written here.

    Stateless with respect to the taxonomy: evidence is read from
    ``ctx.label_space`` per chunk, never copied at construction. Added classes
    are therefore picked up automatically, and ``rewrap_signal_providers`` needs
    no case for this provider."""

    name = "cross-encoder"
    needs_candidates = True

    def __init__(
        self,
        reranker: PairwiseReranker,
        config: Optional[CrossEncoderConfig] = None,
        ops: Optional[ArrayOps] = None,
    ):
        self._reranker = reranker
        self._cfg = config or CrossEncoderConfig()
        self._ops = ops or NumpyArrayOps()

    # ---- schema -----------------------------------------------------------
    def candidate_features(self) -> Sequence[str]:
        """None, and structurally so — see the class docstring. The assembler
        rejects a second-stage provider that claims otherwise."""
        return ()

    @staticmethod
    def _columns_for(name: str) -> Dict[str, str]:
        return {
            "raw": f"ce_{name}",
            "missing": f"ce_{name}_missing",
            "rank": f"rank_ce_{name}",
            "norm": f"norm_ce_{name}",
            "margin": f"margin_ce_{name}",
        }

    def column_names(self) -> List[str]:
        names: List[str] = []
        for doc in self._cfg.documents:
            cols = self._columns_for(doc.name)
            names.extend(
                [
                    cols["raw"],
                    cols["missing"],
                    cols["rank"],
                    cols["norm"],
                    cols["margin"],
                    f"q_gap_ce_{doc.name}",
                    f"is_ce_{doc.name}_top1",
                ]
            )
        names.extend(f"ce_{a}_{b}_gap" for a, b in self._cfg.gaps)
        return names

    # ---- build ------------------------------------------------------------
    def build(self, ctx: SignalContext) -> List[SignalMatrix]:
        if ctx.candidates is None:
            raise ValueError(
                f"{type(self).__name__} declares needs_candidates=True but was built with "
                "SignalContext.candidates=None. It must run in the assembler's second "
                "round, after candidate selection."
            )
        cand = ctx.candidates
        shape = (len(ctx.texts), ctx.n_classes)
        # Tokenize each query once per chunk, not once per pair.
        query_tokens = [_tokens(t) for t in ctx.texts]
        definitions = ctx.label_space.definitions
        prescore = cand.signals.get(self._cfg.prescore_node)

        # One top-n selection per *distinct* top_k rather than per document:
        # documents sharing a top_k (the default pos/neg pair does) select the
        # same cells, and the selection is an O(b*C) partition over the same
        # prescore matrix. Keyed only by top_k because `prescore` is fixed for
        # the whole provider.
        grids: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        values = {}
        for doc in self._cfg.documents:
            if doc.top_k not in grids:
                grids[doc.top_k] = self._rerank_grid(doc.top_k, cand, prescore)
            values[doc.name] = self._score_document(
                doc, grids[doc.top_k], ctx.texts, query_tokens, definitions, shape
            )
        # Cross-document differences, gathered at the same (rows, cols) grid as
        # everything else. They belong to no single document, so they ride on
        # the first matrix as plain extra columns rather than being derived.
        gaps = {f"ce_{a}_{b}_gap": values[a] - values[b] for a, b in self._cfg.gaps}

        matrices: List[SignalMatrix] = []
        for i, doc in enumerate(self._cfg.documents):
            M = values[doc.name]
            matrices.append(
                SignalMatrix(
                    node=f"ce.{doc.name}",
                    value=M,
                    derive=frozenset({"raw", "missing", "rank", "norm", "margin"}),
                    columns=self._columns_for(doc.name),
                    gap_column=f"q_gap_ce_{doc.name}",
                    top1_idx=_argmax_or_missing(M),
                    top1_column=f"is_ce_{doc.name}_top1",
                    top1_check_valid=True,
                    extra_columns=gaps if i == 0 else {},
                )
            )
        return matrices

    def _score_document(
        self,
        doc: CrossEncoderDocument,
        grid: Tuple[np.ndarray, np.ndarray],
        texts: Sequence[str],
        query_tokens: List[frozenset],
        definitions: Sequence[ClassDefinition],
        shape: Tuple[int, int],
    ) -> np.ndarray:
        """One document's ``(b, C)`` matrix: NaN except where actually scored."""
        rows, cols = grid
        # Resolve each class's slots once for this chunk, not once per pair --
        # the evidence depends on the class alone (see `_plan_class`). Only the
        # classes actually selected are planned.
        plans = {int(c): _plan_class(definitions[c], doc) for c in np.unique(cols)}

        # Render the pairs for the selected cells. This is the one Python-level
        # loop in the provider, and it is the right trade: O(pairs) string
        # assembly feeding a model that costs orders of magnitude more per pair,
        # doing work (text rendering) that cannot vectorize and cannot run on a
        # device at all. Everything numeric on either side of it stays in the
        # array backend.
        pairs: List[Tuple[str, str]] = []
        scored_rows: List[int] = []
        scored_cols: List[int] = []
        for r, c in zip(rows.tolist(), cols.tolist()):
            document = _render_plan(plans[c], doc, query_tokens[r])
            if not document:
                continue  # class has nothing to say here -> stays NaN, never 0
            pairs.append((texts[r], document))
            scored_rows.append(r)
            scored_cols.append(c)

        scores = np.asarray(self._reranker.score(pairs), dtype=np.float64) if pairs else []
        if pairs and np.shape(scores) != (len(pairs),):
            raise ValueError(
                f"{type(self._reranker).__name__}.score returned shape {np.shape(scores)} for "
                f"{len(pairs)} pairs; expected exactly one score per pair, i.e. "
                f"({len(pairs)},). PairwiseReranker.score must preserve input order and "
                "length."
            )
        return self._scatter(shape, scored_rows, scored_cols, scores)

    def _scatter(self, shape, rows, cols, values) -> np.ndarray:
        """Place ``values`` at ``(rows, cols)`` in an otherwise-NaN ``(b, C)``
        matrix, through the array backend.

        Routed through ``ArrayOps.scatter_add`` rather than fancy-index
        assignment so the matrix is built wherever the configured backend lives
        (T84/T85) instead of forcing a numpy allocation into an otherwise
        device-resident pipeline. The scored-cell mask comes from a second
        scatter of ones, exactly as ``_scatter_knn`` distinguishes "summed to
        zero" from "never written" — a cell nothing wrote must stay NaN, never
        become the 0.0 the accumulator started at.

        The trailing ``to_host`` matches ``_scatter_knn``'s: the assembler's
        derivation kernels are not device-resident yet (T86), so every signal
        matrix is handed over on the host today. When that changes this call is
        the single line to revisit — which is exactly why the port makes it the
        one sanctioned exit."""
        ops = self._ops
        empty = len(rows) == 0
        total = ops.zeros(shape, dtype=np.float64)
        count = ops.zeros(shape, dtype=np.float64)
        if not empty:
            total = ops.scatter_add(total, rows, cols, values)
            count = ops.scatter_add(count, rows, cols, np.ones(len(rows), dtype=np.float64))
        return ops.to_host(ops.where(count > 0, total, np.nan))

    def _rerank_grid(
        self, top_k: int, cand, prescore: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Which shortlisted cells to rerank: the ``top_k`` best per query by
        ``prescore_node``, intersected with the shortlist.

        Falls back to the whole shortlist when the prescoring node is absent
        (a config where the ranking signal was switched off) — correct, just
        more expensive, and the alternative would be silently reranking nothing.

        The ``to_host`` here is deliberate and is the provider's *only* other
        exit from the array backend: the selection itself is computed by the
        backend (``_topn_mask`` takes ``ops``), but the resulting indices drive
        a Python loop over text, so they have to land on the host. Doing it as
        one explicit call keeps the transfer greppable rather than smuggled in
        by an implicit ``np.asarray`` somewhere in the loop."""
        if prescore is None:
            return cand.rows, cand.cols
        ops = self._ops
        selected = ops.asarray(cand.mask) & _topn_mask(prescore, top_k, ops=ops)
        return np.nonzero(ops.to_host(selected))

    # ---- persistence ------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self._reranker.save(path)
        with open(os.path.join(path, "cross_encoder.json"), "w") as fh:
            json.dump({"kind": self._cfg.kind}, fh)

    @classmethod
    def load(cls, path: str) -> "CrossEncoderSignalProvider":
        raise NotImplementedError(
            "CrossEncoderSignalProvider is reloaded through its registered "
            "SignalProviderSpec.load, which has the RetrievalConfig needed to rebuild "
            "the document/evidence plan -- see infrastructure.registry."
        )
