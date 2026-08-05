"""Structured taxonomy fields on ClassDefinition.

The contract these tests pin down is *additivity*: a taxonomy that carries only
``key,description`` must behave exactly as it did before the structured fields
existed — same ``descriptions``, same ``core_view`` text, same persisted
``meta.json`` entries — while a taxonomy that carries more keeps it intact
through every path that rebuilds a label space.
"""

from __future__ import annotations

import pytest

from text_classifier.domain import ClassDefinition, LabelSpace

RICH = ClassDefinition(
    key="4711",
    description="Retail sale in non-specialized stores with food predominating",
    title="Retail sale in non-specialized stores",
    definition="Stores selling a broad range of goods, food predominating.",
    examples=("supermarket", "grocery store", "mini-market"),
    inclusions=("non-specialized stores where food predominates",),
    exclusions=("specialized fruit and vegetable shops", "food manufacturing"),
    parent_path=("Retail trade", "Retail sale in non-specialized stores"),
    sibling_distinctions=("4719 when food does not predominate",),
)


# ----------------------------------------------------------------- additivity
def test_plain_definition_still_constructs_positionally():
    """Every existing call site builds ClassDefinition(key, description)."""
    d = ClassDefinition("k1", "some description")
    assert d.key == "k1"
    assert d.description == "some description"
    assert d.title == "" and d.definition == ""
    assert d.examples == () and d.inclusions == () and d.exclusions == ()
    assert d.parent_path == () and d.sibling_distinctions == ()
    assert not d.has_structured_fields


def test_plain_definition_core_view_is_exactly_the_description():
    """A view-based retriever over a legacy taxonomy must index today's text."""
    d = ClassDefinition("k1", "some description")
    assert d.core_view() == "some description"


def test_plain_definition_other_views_are_absent_not_empty_documents():
    """ "" means the view did not fire — the text-layer form of the NaN rule."""
    d = ClassDefinition("k1", "some description")
    assert d.examples_view() == ""
    assert d.inclusions_view() == ""
    assert d.boundary_view() == ""


def test_label_space_descriptions_unchanged_by_structured_fields():
    space = LabelSpace([RICH, ClassDefinition("k2", "plain text")])
    assert space.descriptions == [
        "Retail sale in non-specialized stores with food predominating",
        "plain text",
    ]


# --------------------------------------------------------------- normalization
def test_lists_are_coerced_to_tuples_so_the_value_object_stays_hashable():
    d = ClassDefinition("k", "d", examples=["a", "b"])
    assert d.examples == ("a", "b")
    assert hash(d) is not None


def test_blank_entries_are_dropped_not_rejected():
    """Real taxonomy exports carry trailing empties and stray separators."""
    d = ClassDefinition("k", "d", examples=["a", "", "  ", "b"])
    assert d.examples == ("a", "b")


def test_scalar_fields_are_stripped():
    d = ClassDefinition("k", "d", title="  Retail  ")
    assert d.title == "Retail"


def test_bare_string_for_a_multi_value_field_is_rejected():
    """Silently iterating a string into characters would be a nasty bug."""
    with pytest.raises(ValueError, match="not a bare string"):
        ClassDefinition("k", "d", examples="supermarket")


def test_non_string_entry_is_rejected():
    with pytest.raises(ValueError, match="only strings"):
        ClassDefinition("k", "d", examples=["ok", 7])


def test_key_and_description_validation_is_unchanged():
    with pytest.raises(ValueError, match="key"):
        ClassDefinition("", "d")
    with pytest.raises(ValueError, match="description"):
        ClassDefinition("k", "   ")


# ---------------------------------------------------------------------- views
def test_core_view_composes_title_definition_and_parent_path():
    view = RICH.core_view()
    assert view.startswith("Retail sale in non-specialized stores")
    assert "food predominating" in view
    assert "Parent: Retail trade > Retail sale in non-specialized stores" in view


def test_examples_view_is_the_bare_example_list():
    """No title prefix: sharing text across views correlates their scores, and
    the disagreement between views is the signal worth keeping."""
    assert RICH.examples_view() == "supermarket; grocery store; mini-market"


def test_boundary_view_carries_exclusions_and_sibling_contrasts():
    view = RICH.boundary_view()
    assert "Excludes: specialized fruit and vegetable shops; food manufacturing" in view
    assert "Distinguish from: 4719 when food does not predominate" in view


def test_has_structured_fields_detects_any_populated_field():
    assert RICH.has_structured_fields
    assert ClassDefinition("k", "d", title="t").has_structured_fields
    assert ClassDefinition("k", "d", exclusions=("x",)).has_structured_fields
    assert not ClassDefinition("k", "d").has_structured_fields


# --------------------------------------------------------------- label space
def test_definitions_and_definition_at_expose_the_full_objects():
    space = LabelSpace([RICH, ClassDefinition("k2", "plain")])
    assert space.definitions[0] is RICH
    assert space.definition_at(0) is RICH
    assert space.definition_at(1).key == "k2"
    # Column c always means key_at(c) — definition_at must agree.
    assert space.definition_at(1).key == space.key_at(1)
