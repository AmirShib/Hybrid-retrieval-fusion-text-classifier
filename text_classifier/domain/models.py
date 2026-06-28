"""Domain models: framework-free value objects and the LabelSpace aggregate.

Nothing in this module imports an ML framework. These types are the shared
vocabulary the rest of the system speaks in.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True, slots=True)
class ClassDefinition:
    """A class the system can predict: a stable key plus its natural-language
    description (used both as a retrieval signal and for training the encoder).

    Both fields are validated at construction: an empty key would collide in the
    key->index map, and an empty description would silently strip a class of the
    text the description-similarity signals rely on. Failing here turns a corrupt
    feature matrix into an actionable error at the boundary.
    """
    key: str
    description: str

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key.strip():
            raise ValueError(
                f"ClassDefinition.key must be a non-empty string, got {self.key!r}"
            )
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError(
                f"ClassDefinition.description for key {self.key!r} must be a "
                f"non-empty string, got {self.description!r}"
            )


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
            raise ValueError(
                f"LabeledItem.text must be a non-empty string, got {self.text!r}"
            )
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError(
                f"LabeledItem.label must be a non-empty string, got {self.label!r}"
            )


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
        return [d.description for d in self._definitions]

    def index_of(self, key: str) -> int:
        return self._index[key]

    def key_at(self, index: int) -> str:
        return self._definitions[index].key

    def encode_labels(self, labels: Sequence[str]) -> List[int]:
        return [self._index[l] for l in labels]
