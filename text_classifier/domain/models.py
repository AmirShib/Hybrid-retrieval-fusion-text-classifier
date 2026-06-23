"""Domain models: framework-free value objects and the LabelSpace aggregate.

Nothing in this module imports an ML framework. These types are the shared
vocabulary the rest of the system speaks in.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True, slots=True)
class ClassDefinition:
    """A class the system can predict: a stable key plus its natural-language
    description (used both as a retrieval signal and for training the encoder)."""
    key: str
    description: str


@dataclass(frozen=True, slots=True)
class LabeledItem:
    """A training example: free text with its ground-truth class key."""
    text: str
    label: str


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
            raise ValueError("LabelSpace requires at least one class")
        keys = [d.key for d in definitions]
        if len(set(keys)) != len(keys):
            raise ValueError("class keys must be unique")
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
