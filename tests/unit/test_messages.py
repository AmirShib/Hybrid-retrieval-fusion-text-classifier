"""`_messages.format_preview` — the shared "here is what went wrong, abbreviated"
formatting behind every validation error in the package.

Eight call sites used to inline this idiom (`shown = bad[:10]` plus a hand-rolled
ellipsis suffix); the risk it carries is a message that reads differently
depending on which layer raised it, so these pin the one formatting.
"""

from __future__ import annotations

from text_classifier._messages import PREVIEW_LIMIT, format_preview


def test_short_list_is_shown_whole_with_no_ellipsis():
    assert format_preview(["a", "b"]) == "['a', 'b']"


def test_empty_input_renders_as_an_empty_list():
    assert format_preview([]) == "[]"


def test_exactly_at_the_limit_is_not_truncated():
    values = list(range(PREVIEW_LIMIT))
    out = format_preview(values)
    assert out == f"{values}"
    assert "..." not in out


def test_over_the_limit_truncates_and_marks_the_elision():
    out = format_preview(list(range(PREVIEW_LIMIT + 1)))
    assert out == f"{list(range(PREVIEW_LIMIT))} ..."


def test_limit_is_configurable():
    assert format_preview(["a", "b", "c"], limit=2) == "['a', 'b'] ..."


def test_consumes_only_what_it_needs_from_a_lazy_iterable():
    """Validation paths hand this generators over potentially large inputs, so
    it must not materialize the whole thing to print ten values."""
    consumed = []

    def counting():
        for i in range(1000):
            consumed.append(i)
            yield i

    format_preview(counting(), limit=3)
    assert len(consumed) == 4  # three shown + one lookahead proving truncation


def test_renders_tuples_the_way_the_underpopulated_class_error_expects():
    """`TrainingPipeline._validate_inputs` previews `(key, count)` pairs."""
    assert format_preview([("rare", 2), ("odd", 3)]) == "[('rare', 2), ('odd', 3)]"
