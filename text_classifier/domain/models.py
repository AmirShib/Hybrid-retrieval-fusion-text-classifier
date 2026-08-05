"""Domain models: framework-free value objects and the LabelSpace aggregate.

Nothing in this module imports an ML framework. These types are the shared
vocabulary the rest of the system speaks in.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence, Tuple


def _clean_str(value: object, field: str, key: str) -> str:
    """Normalize an optional scalar text field: strip, or "" when absent."""
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"ClassDefinition.{field} for key {key!r} must be a string, got {value!r}")
    return value.strip()


def _clean_tuple(value: object, field: str, key: str) -> Tuple[str, ...]:
    """Normalize an optional multi-value text field to a tuple of clean strings.

    Blank entries are dropped rather than rejected: real taxonomy exports carry
    trailing empty cells and stray separators, and an empty entry contributes no
    signal either way, so failing on one would be hostile at the boundary. A
    *wrong-typed* entry is a different matter — that is a caller bug, and it
    raises."""
    if value is None:
        return ()
    if isinstance(value, str):
        raise ValueError(
            f"ClassDefinition.{field} for key {key!r} must be a sequence of "
            f"strings, not a bare string ({value!r}); split it first"
        )
    if not isinstance(value, Sequence):
        raise ValueError(
            f"ClassDefinition.{field} for key {key!r} must be a sequence of strings, got {value!r}"
        )
    out = []
    for entry in value:
        if not isinstance(entry, str):
            raise ValueError(
                f"ClassDefinition.{field} for key {key!r} must contain only strings, "
                f"got entry {entry!r}"
            )
        entry = entry.strip()
        if entry:
            out.append(entry)
    return tuple(out)


@dataclass(frozen=True, slots=True)
class ClassDefinition:
    """A class the system can predict: a stable key, its natural-language
    description, and — optionally — the structured taxonomy fields behind it.

    ``key`` and ``description`` are required and validated: an empty key would
    collide in the key->index map, and an empty description would silently strip a
    class of the text the description-similarity signals rely on. Failing here
    turns a corrupt feature matrix into an actionable error at the boundary.

    Everything after ``description`` is **optional and additive**. A taxonomy like
    ISIC or COICOP ships more than one text per class — an official title, a
    definition, the products it includes, the ones it explicitly excludes, its
    place in the hierarchy — and gluing them into a single string dilutes each
    one: a four-word example list buried in a 200-word definition is ~2% of the
    embedded text, so a query echoing those four words matches only weakly, and
    BM25 length-normalization penalizes the long document on top of that. Keeping
    the fields apart lets each become its own short, focused retrieval document
    (a *view*), so a class can be found through whichever door fits the query.

    ``description`` keeps its exact original meaning: the single text today's
    description-similarity signals index. Nothing downstream changes when the
    structured fields are absent — ``core_view()`` degrades to ``description``,
    and every other view renders empty (an absent view, not an empty document).
    """

    key: str
    description: str
    title: str = ""
    definition: str = ""
    examples: Tuple[str, ...] = ()
    inclusions: Tuple[str, ...] = ()
    exclusions: Tuple[str, ...] = ()
    parent_path: Tuple[str, ...] = ()
    sibling_distinctions: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key.strip():
            raise ValueError(f"ClassDefinition.key must be a non-empty string, got {self.key!r}")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError(
                f"ClassDefinition.description for key {self.key!r} must be a "
                f"non-empty string, got {self.description!r}"
            )
        # frozen dataclass: normalize in place through object.__setattr__ so
        # callers may pass lists (CSV/JSON readers produce them) and still get a
        # hashable, immutable value object back.
        for name in ("title", "definition"):
            object.__setattr__(self, name, _clean_str(getattr(self, name), name, self.key))
        for name in (
            "examples",
            "inclusions",
            "exclusions",
            "parent_path",
            "sibling_distinctions",
        ):
            object.__setattr__(self, name, _clean_tuple(getattr(self, name), name, self.key))

    @property
    def has_structured_fields(self) -> bool:
        """True when this class carries anything beyond ``key``/``description``.

        The cheap corpus-level question — "is it worth building view indices for
        this taxonomy at all?" — is the fraction of classes for which this holds.
        """
        return bool(
            self.title
            or self.definition
            or self.examples
            or self.inclusions
            or self.exclusions
            or self.parent_path
            or self.sibling_distinctions
        )

    # ---- retrieval views -------------------------------------------------
    # Each view renders one short document for one retrieval purpose. A view is
    # "" when this class has nothing to say through that door — an *absent* view,
    # which a retriever must treat as "did not fire" (NaN), never as an empty
    # document scoring 0 (CLAUDE.md's NaN invariant, at the text layer).
    #
    # Note the views past ``core_view`` deliberately do NOT repeat the title.
    # Sharing a title across every view correlates their scores, which defeats
    # the point of keeping them separate — the fusion model learns most from the
    # *disagreement* between "matches the formal definition" and "matches the
    # concrete examples". A bare example list is a legitimately better retrieval
    # document than a title-prefixed one for the queries these views exist to
    # catch. Revisit if the views turn out too sparse to retrieve on their own.

    def core_view(self) -> str:
        """Formal identity: title, definition, and place in the hierarchy.

        Falls back to ``description`` when no structured fields are set, so a
        view-based retriever over a legacy ``key,description`` taxonomy indexes
        exactly the text it indexes today."""
        parts = [self.title or self.description]
        if self.definition:
            parts.append(self.definition)
        if self.parent_path:
            parts.append("Parent: " + " > ".join(self.parent_path))
        return "\n".join(parts)

    def examples_view(self) -> str:
        """Concrete instances of the class, in everyday language.

        This is the view that bridges official terminology and how people
        actually write. It is also the only class-side text that exists for a
        class with *no* labeled training examples, so it is what gives cold-start
        and rare classes something prototype-shaped to be matched against."""
        return "; ".join(self.examples)

    def inclusions_view(self) -> str:
        """Edge cases stated positively — what does belong to this class."""
        return "; ".join(self.inclusions)

    def boundary_view(self) -> str:
        """Exclusions and sibling contrasts.

        **Not for retrieval.** Embedding models handle negation poorly: "excludes
        food manufacturing" still sits close to *food manufacturing*, and sibling
        contrasts name the competing classes outright, so indexing either pulls
        queries toward exactly the class the text rules out. This view exists for
        pair-scoring (a reranker sees both sides jointly) and for mining hard
        negatives when fine-tuning the encoder."""
        parts = []
        if self.exclusions:
            parts.append("Excludes: " + "; ".join(self.exclusions))
        if self.sibling_distinctions:
            parts.append("Distinguish from: " + "; ".join(self.sibling_distinctions))
        return "\n".join(parts)


@dataclass(frozen=True, slots=True)
class LabeledItem:
    """A training example: free text with its ground-truth class key.

    Empty (or whitespace-only) text encodes to a degenerate zero vector and
    retrieves nothing meaningful, so it is rejected at construction rather than
    quietly polluting the training folds. The label must be a non-empty string;
    whether it is a *known* class is checked against the LabelSpace in
    ``TrainingPipeline.run`` (this value object has no view of the label space).
    """

    text: str
    label: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError(f"LabeledItem.text must be a non-empty string, got {self.text!r}")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError(f"LabeledItem.label must be a non-empty string, got {self.label!r}")


@dataclass(frozen=True, slots=True)
class Prediction:
    """The outcome for one item. `predicted_key` is None when the system abstains;
    `top_key` always carries the best candidate so a human queue can see it."""

    top_key: str
    confidence: float
    abstained: bool
    predicted_key: Optional[str] = None
    runner_up_key: Optional[str] = None
    margin: Optional[float] = None


@dataclass(frozen=True, slots=True)
class CoverageReport:
    """The coverage/accuracy trade-off — the actual deliverable for a
    human-in-the-loop system."""

    coverage: float
    accuracy_on_accepted: float
    accuracy_if_no_abstain: float
    candidate_recall: float
    n_items: int


class LabelSpace:
    """Aggregate over the universe of classes. Owns the canonical key<->index
    mapping so that every component agrees on what column `c` means."""

    __slots__ = ("_definitions", "_index")

    def __init__(self, definitions: Sequence[ClassDefinition]):
        if not definitions:
            raise ValueError(
                "LabelSpace requires at least one ClassDefinition (got an empty sequence)"
            )
        keys = [d.key for d in definitions]
        duplicates = sorted(k for k, n in Counter(keys).items() if n > 1)
        if duplicates:
            raise ValueError(f"class keys must be unique; duplicated key(s): {duplicates}")
        self._definitions: Tuple[ClassDefinition, ...] = tuple(definitions)
        self._index: Mapping[str, int] = {d.key: i for i, d in enumerate(self._definitions)}

    @classmethod
    def from_pairs(cls, pairs: Sequence[Tuple[str, str]]) -> "LabelSpace":
        return cls([ClassDefinition(k, d) for k, d in pairs])

    def __len__(self) -> int:
        return len(self._definitions)

    @property
    def size(self) -> int:
        return len(self._definitions)

    @property
    def keys(self) -> List[str]:
        return [d.key for d in self._definitions]

    @property
    def descriptions(self) -> List[str]:
        """The single description text per class, in canonical column order.

        Unchanged by the structured taxonomy fields: this is still the exact text
        the description-similarity signals index. Rendered views are reached
        through ``definitions`` / ``definition_at`` instead, so adding structured
        fields never silently re-indexes an existing model."""
        return [d.description for d in self._definitions]

    @property
    def definitions(self) -> Tuple[ClassDefinition, ...]:
        """The full class definitions, in canonical column order."""
        return self._definitions

    def definition_at(self, index: int) -> ClassDefinition:
        """The ClassDefinition for column ``index`` (see ``key_at``)."""
        return self._definitions[index]

    def index_of(self, key: str) -> int:
        return self._index[key]

    def key_at(self, index: int) -> str:
        return self._definitions[index].key

    def encode_labels(self, labels: Sequence[str]) -> List[int]:
        return [self._index[label] for label in labels]
