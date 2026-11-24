"""Contrastive lexical signal (T97): a round-two ``SignalProvider``.

Once the shortlist exists, the final choice often turns on one or two short
tokens (``זירו``, ``ללא סוכר``, ``500 מ"ל``). Every other signal scores a candidate
on its own, so the fusion model -- which is pointwise -- never sees *which* words
a candidate shares with its rivals. This signal writes that into each row.

For one item with query words ``Q`` and a shortlist of ``K`` scored candidates,
each with a word set ``T(c)``:

- ``df(t)``: how many candidates' word sets contain query word ``t``
- ``w(t) = log(K / df(t)) / log(K)``: 1 for a word only one candidate has, 0
  for a word they all have, on the same scale for any ``K``
- ``W``: the total weight of query words that match at least one candidate

and per candidate ``c``:

- ``lex_local_overlap = sum of w(t)`` over ``t`` in ``Q ∩ T(c)``
- ``lex_unique_hits``: weight of query words only ``c`` has, over ``W``
- ``lex_counter_hits``: the strongest single competitor's evidence against
  ``c`` -- ``max`` over rivals ``c'`` of the weight of words ``c'`` matched and
  ``c`` did not -- over ``W``. Not the total over all rivals: that would always
  equal ``1 - lex_local_overlap / W`` and add nothing.
- ``lex_shortlist_size = K`` (per item) and, when neighbours are on,
  ``lex_evidence_docs`` (texts in ``T(c)``)

NaN means "not scored": outside the shortlist, a class with no text in the chosen
views, or ``K = 1``; the two shares are also NaN when ``W = 0``. Inside the
shortlist, sharing no query word is a true 0.

Word sets are *sets*: item names and class descriptions rarely repeat a word, so
term frequency carries nothing. Everything is computed in each chunk's own query
vocabulary, since only query words can contribute.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

from ..config import ContrastiveLexicalConfig
from ..domain import LabelSpace, SignalContext, SignalMatrix, SignalProvider
from .reranker import _EVIDENCE

_LOCAL = "lex_local_overlap"


def _analyzer(token_kwargs: Dict[str, Any]) -> Callable[[str], List[str]]:
    """BM25's own analyzer (same ``CountVectorizer`` kwargs), so both lexical
    signals always split text the same way. JSON turns tuples into lists."""
    kw = dict(token_kwargs)
    if isinstance(kw.get("ngram_range"), list):
        kw["ngram_range"] = tuple(kw["ngram_range"])
    return CountVectorizer(**kw).build_analyzer()


class _TokenSets:
    """Binary ``(n, V)`` incidence of ``n`` word sets over their own vocabulary."""

    def __init__(self, terms: List[str], matrix: sp.csr_matrix):
        self.terms = terms
        self.vocab = {t: i for i, t in enumerate(terms)}
        self.matrix = matrix.tocsr()
        self._csc: Optional[sp.csc_matrix] = None

    @classmethod
    def from_token_lists(
        cls, token_lists: Sequence[Sequence[str]], terms: Optional[List[str]] = None
    ) -> "_TokenSets":
        terms = list(terms or [])
        vocab = {t: i for i, t in enumerate(terms)}
        indptr, indices = [0], []
        for toks in token_lists:
            ids = set()
            for t in toks:
                if t not in vocab:
                    vocab[t] = len(terms)
                    terms.append(t)
                ids.add(vocab[t])
            indices.extend(sorted(ids))
            indptr.append(len(indices))
        m = sp.csr_matrix(
            (np.ones(len(indices), dtype=np.float32), np.asarray(indices, np.int64), indptr),
            shape=(len(token_lists), len(terms)),
        )
        return cls(terms, m)

    @property
    def n(self) -> int:
        return int(self.matrix.shape[0])

    def extended(self, token_lists: Sequence[Sequence[str]]) -> "_TokenSets":
        """A copy with ``token_lists`` appended as new rows (vocabulary grows)."""
        new = _TokenSets.from_token_lists(token_lists, list(self.terms))
        old = self.matrix.copy()  # resize is in place; never touch the source pool
        old.resize((old.shape[0], len(new.terms)))
        return _TokenSets(new.terms, sp.vstack([old, new.matrix]).tocsr())

    def restrict(self, terms: Sequence[str]) -> sp.csr_matrix:
        """``(n, len(terms))``: column ``j`` is word ``terms[j]`` (empty if unseen)."""
        ids = np.fromiter((self.vocab.get(t, -1) for t in terms), np.int64, len(terms))
        present = np.flatnonzero(ids >= 0)
        if self._csc is None:
            self._csc = self.matrix.tocsc()
        sub = self._csc[:, ids[present]].tocoo()
        return sp.csr_matrix((sub.data, (sub.row, present[sub.col])), shape=(self.n, len(terms)))


def _keys(m: sp.csr_matrix) -> np.ndarray:
    """Sorted ``row * n_cols + col`` codes of ``m``'s nonzeros, for membership."""
    coo = m.tocoo()
    return np.sort(coo.row.astype(np.int64) * m.shape[1] + coo.col)


def _member(keys: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Elementwise: is each code in ``query`` one of ``keys``? (``keys`` sorted)."""
    if keys.size == 0:
        return np.zeros(query.shape, dtype=bool)
    idx = np.minimum(np.searchsorted(keys, query), keys.size - 1)
    return keys[idx] == query


class ContrastiveLexicalSignalProvider(SignalProvider):
    """Round-two contrastive lexical signal (T97); see the module docstring.

    Declares no ``candidate_features``: it reorders the shortlist and never
    extends it. Class text is read from ``ctx.label_space`` (cached by content),
    so added or edited classes are picked up with no extra wiring. The only
    fitted state is the example pool, and only when ``neighbors > 0``."""

    name = "contrastive-lexical"
    needs_candidates = True

    def __init__(
        self,
        config: Optional[ContrastiveLexicalConfig] = None,
        token_kwargs: Optional[Dict[str, Any]] = None,
        pool: Optional[_TokenSets] = None,
        pool_labels: Optional[np.ndarray] = None,
        chunk: int = 256,
    ):
        self._cfg = config or ContrastiveLexicalConfig()
        self._token_kwargs = dict(token_kwargs or {})
        self._analyze = _analyzer(self._token_kwargs)
        self._pool = pool
        self._pool_labels = None if pool_labels is None else np.asarray(pool_labels, np.int64)
        self._chunk = max(int(chunk), 1)
        self._class_key: Optional[Tuple[Any, ...]] = None
        self._classes: Optional[Tuple[_TokenSets, np.ndarray]] = None

    @property
    def config(self) -> ContrastiveLexicalConfig:
        return self._cfg

    @property
    def pool_size(self) -> int:
        return 0 if self._pool is None else self._pool.n

    # ---- schema --------------------------------------------------------------
    def candidate_features(self) -> Sequence[str]:
        return ()

    def column_names(self) -> List[str]:
        names = [
            _LOCAL,
            "rank_lex_local",
            "margin_lex_local",
            "q_gap_lex_local",
            "lex_unique_hits",
            "lex_counter_hits",
            "lex_shortlist_size",
        ]
        if self._cfg.neighbors > 0:
            names.append("lex_evidence_docs")
        return names

    # ---- fitting ---------------------------------------------------------------
    def _tokens(self, texts: Sequence[str]) -> List[List[str]]:
        return [self._analyze(str(t)) for t in texts]

    def fit(self, texts: Sequence[str], labels: np.ndarray, label_space: LabelSpace) -> None:
        """Index the example pool for neighbour lookup. A no-op with neighbours off,
        so the default signal holds no training-derived state at all."""
        if self._cfg.neighbors <= 0:
            self._pool, self._pool_labels = None, None
            return
        self._pool = _TokenSets.from_token_lists(self._tokens(texts))
        self._pool_labels = np.asarray(labels, dtype=np.int64)

    def with_added_examples(
        self, texts: Sequence[str], labels: np.ndarray
    ) -> "ContrastiveLexicalSignalProvider":
        """A copy whose pool also holds ``texts`` (the no-retrain update path)."""
        if self._cfg.neighbors <= 0:
            return self
        new_labels = np.asarray(labels, dtype=np.int64)
        if self._pool is None or self._pool_labels is None:
            pool = _TokenSets.from_token_lists(self._tokens(texts))
            pool_labels = new_labels
        else:
            pool = self._pool.extended(self._tokens(texts))
            pool_labels = np.concatenate([self._pool_labels, new_labels])
        return ContrastiveLexicalSignalProvider(
            self._cfg, self._token_kwargs, pool, pool_labels, self._chunk
        )

    # ---- class side ------------------------------------------------------------
    def _class_sets(self, label_space: LabelSpace) -> Tuple[_TokenSets, np.ndarray]:
        """Each class's word set over the chosen views, plus how many texts built it."""
        defs = tuple(label_space.definitions)
        key = (defs, tuple(self._cfg.views))
        if self._classes is None or self._class_key != key:
            token_lists, n_docs = [], np.zeros(len(defs), dtype=np.int64)
            for i, d in enumerate(defs):
                toks: List[str] = []
                for view in self._cfg.views:
                    for entry in _EVIDENCE[view](d):
                        if entry:
                            toks.extend(self._analyze(entry))
                            n_docs[i] += 1
                token_lists.append(toks)
            self._classes = (_TokenSets.from_token_lists(token_lists), n_docs)
            self._class_key = key
        return self._classes

    # ---- scoring ---------------------------------------------------------------
    def build(self, ctx: SignalContext) -> List[SignalMatrix]:
        if ctx.candidates is None:
            raise ValueError(
                f"{type(self).__name__} declares needs_candidates=True but was built with "
                "SignalContext.candidates=None. It must run in the assembler's second "
                "round, after candidate selection."
            )
        cand = ctx.candidates
        b, C = len(ctx.texts), ctx.n_classes
        rows = np.asarray(cand.rows, dtype=np.int64)
        cols = np.asarray(cand.cols, dtype=np.int64)
        order = np.lexsort((cols, rows))  # row-major, which the block layout needs
        rows, cols = rows[order], cols[order]

        out = {
            name: np.full((b, C), np.nan)
            for name in (_LOCAL, "lex_unique_hits", "lex_counter_hits", "lex_evidence_docs")
        }
        size = np.zeros(b, dtype=np.float64)
        classes = self._class_sets(ctx.label_space)
        q_tokens = [list(dict.fromkeys(t)) for t in self._tokens(ctx.texts)]

        for s in range(0, b, self._chunk):
            e = min(b, s + self._chunk)
            lo, hi = np.searchsorted(rows, [s, e])
            if lo == hi:
                continue
            self_ids = None if ctx.self_ids is None else np.asarray(ctx.self_ids[s:e])
            self._score_block(
                q_tokens[s:e], rows[lo:hi] - s, cols[lo:hi], C, classes, self_ids, s, out, size
            )

        extra = {
            "lex_unique_hits": out["lex_unique_hits"],
            "lex_counter_hits": out["lex_counter_hits"],
        }
        if self._cfg.neighbors > 0:
            extra["lex_evidence_docs"] = out["lex_evidence_docs"]
        return [
            SignalMatrix(
                node="lex.local",
                value=out[_LOCAL],
                derive=frozenset({"raw", "rank", "margin"}),
                columns={"raw": _LOCAL, "rank": "rank_lex_local", "margin": "margin_lex_local"},
                gap_column="q_gap_lex_local",
                extra_columns=extra,
                extra_scalars={"lex_shortlist_size": size},
            )
        ]

    def _score_block(
        self,
        q_tokens: List[List[str]],
        r: np.ndarray,
        c: np.ndarray,
        n_classes: int,
        classes: Tuple[_TokenSets, np.ndarray],
        self_ids: Optional[np.ndarray],
        offset: int,
        out: Dict[str, np.ndarray],
        size: np.ndarray,
    ) -> None:
        """Score one block of queries. ``r`` (block-local query index) and ``c``
        (class) list its candidates, sorted by ``(r, c)``.

        Laid out densely as ``(queries, candidates-per-query, query-words)``: item
        names are short, so the padded tensor is small, and every quantity is then a
        sum or an einsum over it."""
        nb = len(q_tokens)
        terms: Dict[str, int] = {}
        for toks in q_tokens:
            for t in toks:
                terms.setdefault(t, len(terms))
        vc = max(len(terms), 1)
        L = max((len(t) for t in q_tokens), default=0)
        tok = np.full((nb, max(L, 1)), -1, dtype=np.int64)
        for i, toks in enumerate(q_tokens):
            tok[i, : len(toks)] = [terms[t] for t in toks]
        term_list = list(terms)

        # H[i, l]: does candidate i's word set contain its query's l-th word?
        class_sets, class_docs = classes
        T = tok[r]  # (n, L)
        valid = T >= 0
        H = _member(_keys(class_sets.restrict(term_list)), c[:, None] * vc + T) & valid
        n_docs = class_docs[c].astype(np.float64)
        if self._cfg.neighbors > 0 and self._pool is not None and self._pool.n > 0:
            ci, ex, pool_keys = self._neighbors(tok, term_list, vc, r, c, n_classes, self_ids)
            if ci.size:
                Hn = _member(pool_keys, ex[:, None] * vc + T[ci]) & valid[ci]
                np.logical_or.at(H, ci, Hn)
                n_docs += np.bincount(ci, minlength=len(r))
        scored = n_docs > 0

        # Dense (nb, Kmax, L) layout of the scored candidates.
        pos = np.arange(len(r)) - np.searchsorted(r, r, side="left")
        kmax = int(pos.max()) + 1
        Hd = np.zeros((nb, kmax, tok.shape[1]), dtype=bool)
        Hd[r, pos] = H & scored[:, None]
        live = np.zeros((nb, kmax), dtype=bool)
        live[r, pos] = scored

        K = live.sum(axis=1).astype(np.float64)  # (nb,)
        df = Hd.sum(axis=1).astype(np.float64)  # (nb, L)
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(
                (df > 0) & (K[:, None] >= 2),
                np.log(K[:, None] / df) / np.log(np.maximum(K[:, None], 2.0)),
                0.0,
            )
        W = w.sum(axis=1)  # (nb,)
        Hf = Hd.astype(np.float64)
        Hw = Hf * w[:, None, :]
        local = Hw.sum(axis=2)  # (nb, Kmax)
        unique = (Hd & (df == 1)[:, None, :]).sum(axis=2).astype(np.float64)
        # Against candidate k, rival j holds local[j] minus the weight they share.
        shared = np.einsum("bkl,bjl->bkj", Hw, Hf)
        against = local[:, None, :] - shared
        rival = live[:, None, :] & ~np.eye(kmax, dtype=bool)[None]
        against = np.where(rival, against, -np.inf)
        counter = against.max(axis=2) if kmax > 1 else np.full((nb, kmax), -np.inf)

        ok = scored & (K[r] >= 2)
        ok_w = ok & (W[r] > 0)
        Wr = np.where(W[r] > 0, W[r], 1.0)
        q = r + offset
        out[_LOCAL][q, c] = np.where(ok, local[r, pos], np.nan)
        out["lex_unique_hits"][q, c] = np.where(ok_w, unique[r, pos] / Wr, np.nan)
        out["lex_counter_hits"][q, c] = np.where(ok_w, counter[r, pos] / Wr, np.nan)
        out["lex_evidence_docs"][q, c] = np.where(scored, n_docs, np.nan)
        size[offset : offset + nb] = K

    def _neighbors(
        self,
        tok: np.ndarray,
        term_list: List[str],
        vc: int,
        r: np.ndarray,
        c: np.ndarray,
        n_classes: int,
        self_ids: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """For each candidate, the ``neighbors`` examples of its class sharing the
        most words with the query (ties: lower pool index). Returns parallel
        ``(candidate index, example index)`` arrays plus the pool's membership keys.
        An item's own row (``self_ids``, leave-one-out mode) is never its neighbour."""
        assert self._pool is not None and self._pool_labels is not None
        P = self._pool.restrict(term_list)  # (n_pool, vc)
        nb = tok.shape[0]
        qr, ql = np.nonzero(tok >= 0)
        Q = sp.csr_matrix((np.ones(qr.size, dtype=np.float32), (qr, tok[qr, ql])), shape=(nb, vc))
        S = (Q @ P.T).tocoo()  # (nb, n_pool) overlap counts; zero overlap is absent
        qi = S.row.astype(np.int64)
        ex = S.col.astype(np.int64)
        overlap = S.data
        if self_ids is not None:
            keep = ex != self_ids[qi]
            qi, ex, overlap = qi[keep], ex[keep], overlap[keep]
        # Keep only examples whose class is one of this query's candidates.
        cand_keys = r * n_classes + c  # sorted: candidates are row-major
        want = qi * n_classes + self._pool_labels[ex]
        idx = np.minimum(np.searchsorted(cand_keys, want), cand_keys.size - 1)
        hit = cand_keys[idx] == want
        ci, ex, overlap = idx[hit], ex[hit], overlap[hit]
        order = np.lexsort((ex, -overlap, ci))
        ci, ex = ci[order], ex[order]
        rank = np.arange(ci.size) - np.searchsorted(ci, ci, side="left")
        keep = rank < self._cfg.neighbors
        return ci[keep], ex[keep], _keys(P)

    # ---- persistence -------------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        meta: Dict[str, Any] = {"neighbors": self._cfg.neighbors, "n_pool": self.pool_size}
        if self._pool is not None and self._pool_labels is not None:
            m = self._pool.matrix.tocsr()
            np.savez(
                os.path.join(path, "pool.npz"),
                indices=m.indices.astype(np.int64),
                indptr=m.indptr.astype(np.int64),
                shape=np.asarray(m.shape, dtype=np.int64),
                labels=self._pool_labels,
            )
            with open(os.path.join(path, "pool_terms.json"), "w", encoding="utf-8") as fh:
                json.dump(self._pool.terms, fh, ensure_ascii=False)
        with open(os.path.join(path, "contrastive.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh)

    @classmethod
    def load_from(
        cls,
        path: str,
        config: ContrastiveLexicalConfig,
        token_kwargs: Optional[Dict[str, Any]] = None,
        chunk: int = 256,
    ) -> "ContrastiveLexicalSignalProvider":
        pool, labels = None, None
        pool_file = os.path.join(path, "pool.npz")
        if os.path.isfile(pool_file):
            with np.load(pool_file, allow_pickle=False) as z:
                shape = tuple(int(x) for x in z["shape"])
                m = sp.csr_matrix(
                    (np.ones(z["indices"].size, dtype=np.float32), z["indices"], z["indptr"]),
                    shape=shape,
                )
                labels = z["labels"]
            with open(os.path.join(path, "pool_terms.json"), encoding="utf-8") as fh:
                pool = _TokenSets(list(json.load(fh)), m)
        return cls(config, token_kwargs, pool, labels, chunk)

    @classmethod
    def load(cls, path: str) -> "ContrastiveLexicalSignalProvider":
        raise NotImplementedError(
            "ContrastiveLexicalSignalProvider is reloaded through its registered "
            "SignalProviderSpec.load, which has the RetrievalConfig -- see "
            "infrastructure.registry."
        )
