"""Custom fusion-feature providers (T70).

This module ships the *sample* provider that exercises the ``FeatureProvider``
seam end to end. It is deliberately one provider, not a library (a library of
extra providers is explicitly out of scope for T70): its job is to prove the
contract — a train-set-derived, per-fold-fittable feature that reaches the fusion
model at train *and* inference, in a persisted, air-gapped-portable form, with
NaN-as-missing and out-of-fold leakage discipline.

``ClassKeywordOverlapProvider`` is a "domain-lexicon hit" feature: for each class
it learns the set of tokens seen in that class's training examples, and for each
(item, candidate class) it emits the fraction of the item's (in-vocabulary) tokens
that fall in that class's learned lexicon. Because the lexicon is derived from
training data, the provider is fit *per fold on the other folds' rows* — exactly
like prototypes/indices — so an item never contributes to the lexicon it is then
scored against.
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, List, Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from ..domain import FeatureContext, FeatureProvider, LabeledItem, LabelSpace


class ClassKeywordOverlapProvider(FeatureProvider):
    """Per-(item, candidate) fraction of the item's tokens in the candidate
    class's learned keyword set.

    Fitted state is a per-class token-membership matrix over a corpus vocabulary
    (``scipy`` sparse) plus the fitted ``CountVectorizer``. A class with no
    training examples in the fitted fold has an empty lexicon and the feature is
    ``NaN`` (it "did not fire"); so is an item with no in-vocabulary tokens. Any
    ``CountVectorizer`` kwarg (``ngram_range``, ``token_pattern``, ``stop_words``,
    ...) may be passed through ``params`` and is preserved across save/load."""

    DEFAULT_COLUMN = "class_kw_overlap"

    def __init__(self, column: str = DEFAULT_COLUMN, **cv_kwargs: Any):
        self._column = column
        self._cv_kwargs = cv_kwargs
        self._vectorizer: CountVectorizer | None = None
        # (C, V) sparse {0,1}: membership[c, t] == 1 iff token t appears in a
        # training example of class c. None until fit.
        self._membership: sparse.csr_matrix | None = None
        self._has_vocab: np.ndarray | None = None  # (C,) bool: class has a lexicon

    # ------------------------------------------------------------------ contract
    def names(self) -> List[str]:
        return [self._column]

    def fit(
        self, items: Sequence[LabeledItem], label_space: LabelSpace
    ) -> "ClassKeywordOverlapProvider":
        """Learn each class's keyword set from ``items`` (this fold's training
        rows). Vectorized: a class×example indicator times the token-incidence
        matrix yields per-class token membership in one matmul."""
        C = label_space.size
        texts = [it.text for it in items]
        y = np.asarray(label_space.encode_labels([it.label for it in items]), dtype=np.int64)

        counts = None
        if texts:
            vec = CountVectorizer(**self._cv_kwargs)
            try:
                counts = vec.fit_transform(texts)
            except ValueError:
                counts = None  # empty vocabulary (e.g. every token filtered out)
        if counts is None or counts.shape[1] == 0:
            # Nothing learnable: every class has an empty lexicon -> feature is
            # always NaN. Keep a well-formed (C, 0) matrix so compute stays total.
            self._vectorizer = None
            self._membership = sparse.csr_matrix((C, 0), dtype=np.float32)
            self._has_vocab = np.zeros(C, dtype=bool)
            return self

        n, V = counts.shape
        incidence = (counts > 0).astype(np.float32)  # (n, V)
        # class-by-example indicator (C, n); (C, n) @ (n, V) -> (C, V) token counts.
        indicator = sparse.csr_matrix(
            (np.ones(n, dtype=np.float32), (y, np.arange(n))), shape=(C, n)
        )
        membership = (indicator @ incidence) > 0  # (C, V) sparse bool
        self._vectorizer = vec
        self._membership = membership.astype(np.float32).tocsr()
        self._has_vocab = np.asarray(self._membership.getnnz(axis=1)).ravel() > 0
        return self

    def compute(self, ctx: FeatureContext) -> Dict[str, np.ndarray]:
        n = ctx.n_candidates
        vals = np.full(n, np.nan, dtype=np.float64)
        if self._vectorizer is None or self._membership is None or self._membership.shape[1] == 0:
            return {self._column: vals}
        assert self._has_vocab is not None

        q = self._vectorizer.transform(list(ctx.query_texts))
        q_bin = (q > 0).astype(np.float32)  # (b, V) in-vocab incidence
        tok = np.asarray(q_bin.sum(axis=1)).ravel()  # (b,) in-vocab token count
        overlap = np.asarray((q_bin @ self._membership.T).todense(), dtype=np.float64)  # (b, C)
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = overlap / tok[:, None]
        # NaN-as-missing: a class with no learned lexicon "did not fire", and a
        # query with no in-vocabulary tokens has no denominator.
        frac[:, ~self._has_vocab] = np.nan
        frac[tok == 0, :] = np.nan

        # Gather over the candidate grid. Guard against a label space widened
        # after fit (added classes, T78): an out-of-range class simply stays NaN.
        C = self._membership.shape[0]
        in_range = ctx.cols < C
        vals[in_range] = frac[ctx.rows[in_range], ctx.cols[in_range]]
        return {self._column: vals}

    # --------------------------------------------------------------- persistence
    def save(self, path: str) -> None:
        """Persist to directory ``path``. Portable: stdlib pickle over a fitted
        ``CountVectorizer`` + ``scipy`` sparse membership (same dependency surface
        the shipped ``lexical.pkl`` already relies on)."""
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "provider.pkl"), "wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load(cls, path: str) -> "ClassKeywordOverlapProvider":
        with open(os.path.join(path, "provider.pkl"), "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"{path!r} does not contain a {cls.__name__}")
        return obj
