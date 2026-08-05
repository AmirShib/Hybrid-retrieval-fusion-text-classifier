"""Ingest and persistence of the structured taxonomy fields.

Two invariants under test:

1. **A plain taxonomy is untouched.** A ``key,description`` CSV reads exactly as
   before, and a label space with no structured fields serializes to the same
   ``{"key", "description"}`` meta entries it always did — so re-saving an
   existing model dir with this version produces no diff.
2. **A structured taxonomy survives the round trip** through both readers and
   through ``meta.json``.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from text_classifier.cli._common import read_label_space
from text_classifier.domain import ClassDefinition
from text_classifier.infrastructure.persistence import _class_from_meta, _class_to_meta


# ------------------------------------------------------------------ CSV ingest
def test_plain_classes_csv_reads_exactly_as_before(tmp_path):
    path = tmp_path / "classes.csv"
    pd.DataFrame({"key": ["a", "b"], "description": ["alpha text", "beta text"]}).to_csv(
        path, index=False
    )
    space = read_label_space(str(path))
    assert space.keys == ["a", "b"]
    assert space.descriptions == ["alpha text", "beta text"]
    assert not any(d.has_structured_fields for d in space.definitions)


def test_csv_optional_columns_are_picked_up(tmp_path):
    path = tmp_path / "classes.csv"
    pd.DataFrame(
        {
            "key": ["4711"],
            "description": ["Retail sale, food predominating"],
            "title": ["Retail sale in non-specialized stores"],
            "examples": ["supermarket|grocery store|mini-market"],
            "exclusions": ["fruit and vegetable shops|food manufacturing"],
            "parent_path": ["Retail trade|Non-specialized retail"],
        }
    ).to_csv(path, index=False)

    d = read_label_space(str(path)).definition_at(0)
    assert d.title == "Retail sale in non-specialized stores"
    assert d.examples == ("supermarket", "grocery store", "mini-market")
    assert d.exclusions == ("fruit and vegetable shops", "food manufacturing")
    assert d.parent_path == ("Retail trade", "Non-specialized retail")
    # description keeps its original meaning — not overwritten by the title
    assert d.description == "Retail sale, food predominating"


def test_csv_blank_optional_cells_become_empty_fields(tmp_path):
    """A partially-populated taxonomy is the normal case, not an error."""
    path = tmp_path / "classes.csv"
    pd.DataFrame(
        {
            "key": ["a", "b"],
            "description": ["alpha", "beta"],
            "examples": ["x|y", None],
        }
    ).to_csv(path, index=False)

    space = read_label_space(str(path))
    assert space.definition_at(0).examples == ("x", "y")
    assert space.definition_at(1).examples == ()


def test_csv_semicolons_inside_an_entry_survive(tmp_path):
    """Why the separator is a pipe: taxonomy prose is full of semicolons."""
    path = tmp_path / "classes.csv"
    pd.DataFrame(
        {
            "key": ["a"],
            "description": ["alpha"],
            "inclusions": ["cereals; oatmeal; muesli|breads and rolls"],
        }
    ).to_csv(path, index=False)

    assert read_label_space(str(path)).definition_at(0).inclusions == (
        "cereals; oatmeal; muesli",
        "breads and rolls",
    )


# ---------------------------------------------------------------- JSONL ingest
def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_jsonl_reads_real_arrays(tmp_path):
    path = tmp_path / "classes.jsonl"
    _write_jsonl(
        path,
        [
            {
                "key": "4711",
                "description": "Retail sale, food predominating",
                "title": "Retail sale in non-specialized stores",
                "inclusions": ["non-specialized stores where food predominates"],
                "exclusions": ["specialized fruit shops"],
                "parent_path": ["Retail trade", "Non-specialized retail"],
            },
            {"key": "4719", "description": "Other retail sale"},
        ],
    )

    space = read_label_space(str(path))
    assert space.keys == ["4711", "4719"]
    first = space.definition_at(0)
    assert first.inclusions == ("non-specialized stores where food predominates",)
    assert first.parent_path == ("Retail trade", "Non-specialized retail")
    # A minimal record alongside a rich one is fine.
    assert not space.definition_at(1).has_structured_fields


def test_jsonl_blank_lines_are_skipped(tmp_path):
    path = tmp_path / "classes.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"key": "a", "description": "alpha"}) + "\n")
        fh.write("\n")
        fh.write(json.dumps({"key": "b", "description": "beta"}) + "\n")
    assert read_label_space(str(path)).keys == ["a", "b"]


def test_jsonl_malformed_line_reports_the_line_number(tmp_path):
    path = tmp_path / "classes.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"key": "a", "description": "alpha"}) + "\n")
        fh.write("{not json\n")
    with pytest.raises(SystemExit, match="line 2"):
        read_label_space(str(path))


def test_jsonl_missing_required_field_is_a_clear_error(tmp_path):
    path = tmp_path / "classes.jsonl"
    _write_jsonl(path, [{"key": "a"}])
    with pytest.raises(SystemExit, match="missing 'key' and/or 'description'"):
        read_label_space(str(path))


# ------------------------------------------------------------ meta.json shape
def test_plain_class_serializes_to_exactly_key_and_description():
    """Guards the no-diff promise for existing model dirs."""
    entry = _class_to_meta(ClassDefinition("a", "alpha"))
    assert entry == {"key": "a", "description": "alpha"}


def test_structured_class_round_trips_through_meta():
    original = ClassDefinition(
        "4711",
        "Retail sale, food predominating",
        title="Retail sale in non-specialized stores",
        definition="Broad range of goods, food predominating.",
        examples=("supermarket", "grocery store"),
        inclusions=("food predominates",),
        exclusions=("specialized fruit shops",),
        parent_path=("Retail trade",),
        sibling_distinctions=("4719 when food does not predominate",),
    )
    assert _class_from_meta(_class_to_meta(original)) == original


def test_legacy_meta_entry_without_optional_keys_still_loads():
    """Every model dir written before these fields existed must keep loading."""
    d = _class_from_meta({"key": "a", "description": "alpha"})
    assert d == ClassDefinition("a", "alpha")


def test_meta_entry_omits_empty_optional_fields():
    entry = _class_to_meta(ClassDefinition("a", "alpha", title="A", examples=()))
    assert entry == {"key": "a", "description": "alpha", "title": "A"}
    assert "examples" not in entry
