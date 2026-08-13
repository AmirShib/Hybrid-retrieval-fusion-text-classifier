"""Class keys and item labels are identifiers, not numbers.

Official statistical classifications are full of zero-padded numeric codes:
ISCO-08's armed-forces unit groups (``0110``, ``0210``, ``0310``), COICOP
divisions (``01``), NACE/ISIC, SIC. Left to pandas' type inference, a CSV column
of those reads back as int64 and the leading zero is gone — so ``0110`` in
``classes.csv`` and ``0110`` in ``items.csv`` stop matching, and the failure
surfaces far downstream as "item label is not defined in the LabelSpace".

These tests pin the fix: the reader forces the identifier columns to string.
"""

from __future__ import annotations

import pytest

from text_classifier.cli._common import read_items, read_label_space

ZERO_PADDED = ["0110", "0210", "1111"]


def _write(path, header: str, rows: list[str]) -> str:
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return str(path)


class TestZeroPaddedCodesSurviveTheReader:
    def test_class_keys_keep_leading_zeros(self, tmp_path):
        path = _write(
            tmp_path / "classes.csv",
            "key,description",
            [f"{code},description of {code}" for code in ZERO_PADDED],
        )
        assert list(read_label_space(path).keys) == ZERO_PADDED

    def test_item_labels_keep_leading_zeros(self, tmp_path):
        path = _write(
            tmp_path / "items.csv",
            "text,label",
            [f"job title {i},{code}" for i, code in enumerate(ZERO_PADDED)],
        )
        assert [item.label for item in read_items(path)] == ZERO_PADDED

    def test_labels_and_keys_still_join(self, tmp_path):
        """The end the bug actually broke: a label must find its class."""
        classes = _write(
            tmp_path / "classes.csv",
            "key,description",
            [f"{code},description of {code}" for code in ZERO_PADDED],
        )
        items = _write(
            tmp_path / "items.csv",
            "text,label",
            [f"job title {i},{code}" for i, code in enumerate(ZERO_PADDED)],
        )
        label_space = read_label_space(classes)
        for item in read_items(items):
            assert label_space.index_of(item.label) >= 0

    def test_custom_column_names_are_also_protected(self, tmp_path):
        """The dtype override has to follow ``--label-col``/``--key-col``, not a
        hardcoded name."""
        path = _write(
            tmp_path / "items.csv",
            "utterance,isco08",
            ["welder,0110"],
        )
        items = read_items(path, text_col="utterance", label_col="isco08")
        assert items[0].label == "0110"

    @pytest.mark.parametrize("code", ["0110", "01.1.1", "A0110", "0"])
    def test_identifier_shapes_round_trip_verbatim(self, tmp_path, code):
        """Numeric, dotted, alphanumeric and a bare zero all read back as written."""
        path = _write(tmp_path / "items.csv", "text,label", [f"some text,{code}"])
        assert read_items(path)[0].label == code


class TestMissingColumnErrorIsUnchanged:
    """The dtype override names columns that may not exist; pandas ignores those,
    so the actionable missing-column error must still be the one users see."""

    def test_missing_label_column_still_reports_clearly(self, tmp_path):
        path = _write(tmp_path / "items.csv", "text,not_label", ["a,0110"])
        with pytest.raises(SystemExit, match="missing required column"):
            read_items(path)

    def test_missing_key_column_still_reports_clearly(self, tmp_path):
        path = _write(tmp_path / "classes.csv", "not_key,description", ["0110,d"])
        with pytest.raises(SystemExit, match="missing required column"):
            read_label_space(path)
