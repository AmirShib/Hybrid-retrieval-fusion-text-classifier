"""T95: Hebrew item-name normalizer. Each test pins a behaviour seen in the real
160k-row file, including the traps found while auditing it."""

from __future__ import annotations

import inspect

import pandas as pd
import pytest

from text_classifier.infrastructure import hebrew_text as ht
from text_classifier.infrastructure.hebrew_text import (
    HebrewItemNormalizer,
    NormalizerConfig,
    canonicalize_units,
    clean_dataframe,
    extract_fields,
    fix_flipped_decimals,
    fix_spacing,
    is_placeholder,
    normalize_unicode,
    strip_edge_junk,
)


# ------------------------------------------------------------------ unicode
@pytest.mark.parametrize(
    "raw, expected",
    [
        ('תפו""""ע', 'תפו"ע'),  # CSV-escape debris
        ('500 מ""ל', '500 מ"ל'),
        ("ג`ל", "ג'ל"),  # backtick used as geresh
        ("ג׳ל", "ג'ל"),  # real Hebrew geresh
        ("ג’ל", "ג'ל"),
        ("שָׁלוֹם", "שלום"),  # niqqud
        ("א‏ב", "אב"),  # bidi mark
        ("א ב", "א ב"),  # NBSP
        ("2×3", "2*3"),
    ],
)
def test_normalize_unicode(raw, expected):
    assert normalize_unicode(raw) == expected


# ------------------------------------------------------------------ spacing
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("200מל", "200 מל"),
        ("טבסקו150", "טבסקו 150"),
        ("1.89ל", "1.89 ל"),  # decimal survives
        ("שוק.חלב", "שוק. חלב"),  # dot gluing two words
        ("ב.ס.ד", "ב.ס.ד"),  # initialism untouched
        ("מ.ר.ז.ל", "מ.ר.ז.ל"),
        ("פריכה+שוקולד", "פריכה + שוקולד"),
        ("מבצע 1+1", "מבצע 1+1"),
        ("עם % 37", "עם 37%"),
        ("70 %", "70%"),
        ("5 * 35", "5*35"),
        ("וונוס *8", "וונוס *8"),  # leading multiplier keeps its space
    ],
)
def test_fix_spacing(raw, expected):
    assert fix_spacing(raw) == expected


def test_flipped_decimal_repaired_only_in_spaced_form():
    assert fix_flipped_decimals("אבקת כביסה 25 . 1 קג") == "אבקת כביסה 1.25 קג"
    assert fix_flipped_decimals("1.25 קג") == "1.25 קג"


# ------------------------------------------------------------------ units
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("200 מל", '200 מ"ל'),
        ('250 מ"ל', '250 מ"ל'),
        ("200 ml", '200 מ"ל'),
        ("30 מל.", '30 מ"ל'),
        ("500 גר", "500 גרם"),
        ("500 ג'", "500 גרם"),
        ("155 ג", "155 גרם"),
        ("1 ק'ג", '1 ק"ג'),
        ("2 ל", "2 ליטר"),
        ("10 יח", "10 יח'"),
        ("25 מטר", "25 מטר"),
        # Ambiguous: meters on cling film, ml on deodorant -> left as written.
        ("ניילון נצמד 30 מ", "ניילון נצמד 30 מ"),
        ("דאודורנט 50 מ'", "דאודורנט 50 מ'"),
        # A unit word with no number before it is a normal word.
        ("חלב ל ילדים", "חלב ל ילדים"),
        ("3 גלגלים", "3 גלגלים"),
    ],
)
def test_canonicalize_units(raw, expected):
    assert canonicalize_units(raw) == expected


def test_edge_junk_keeps_abbreviation_geresh_and_percent():
    assert strip_edge_junk("`שפתון קרמי") == "שפתון קרמי"
    assert strip_edge_junk(". מברשת שיער") == "מברשת שיער"
    assert strip_edge_junk("10 יח'") == "10 יח'"
    assert strip_edge_junk("שוקולד 70%") == "שוקולד 70%"
    assert strip_edge_junk("*90 אשפתון") == "*90 אשפתון"


def test_placeholder():
    assert is_placeholder("Rejected") and is_placeholder(" rejected ")
    assert not is_placeholder("מברשת שיער")


# ------------------------------------------------------------------ fitted normalizer
def _corpus():
    rows = (
        ["צ'יפס תפוחי אדמה"] * 6
        + ["ציפס גבינה"] * 6
        + ["צלי בקר"] * 6  # real word: must not become צ'לי
        + ["צ'לי פלפל"] * 6
        + ["חטיף שוקולד חלב"] * 20
        + ["ילד שמח שמפו"] * 5  # "ילד" used mid-name: a complete word
        + ["ילדים שמפו"] * 30
        + ["רוטב עגבניות"] * 5
    )
    return rows


@pytest.fixture(scope="module")
def norm():
    return HebrewItemNormalizer().fit(_corpus())


def test_geresh_variants_merged_but_real_words_kept(norm):
    assert norm.normalize("ציפס גבינה") == "צ'יפס גבינה"
    assert norm.normalize("צלי בקר") == "צלי בקר"
    assert norm.normalize("גל גילוח") == "ג'ל גילוח"  # seeded


def test_plene_spelling_never_generalized(norm):
    assert norm.normalize("רוטב עגבניות") == "רוטב עגבניות"
    assert norm.normalize("רטוב") == "רטוב"
    assert norm.normalize("עיפרון עיניים") == "עפרון עיניים"  # whitelisted pair


def test_abbreviations_dot_glued_and_number_glued(norm):
    assert norm.normalize('עוגיות תפו""ע') == "עוגיות תפוחי עץ"
    assert norm.normalize('בורקס תפו"א') == "בורקס תפוחי אדמה"
    assert norm.normalize("שוק.חלב") == "שוקולד חלב"
    assert norm.normalize("עוג.פפרדג") == "עוגיות פפרדג"
    assert norm.normalize("50נרסיסו א.אדפ") == "50 נרסיסו אישה או דה פרפיום"
    assert norm.normalize("פרו 30 מל. א.ד.פ אישה") == 'פרו 30 מ"ל או דה פרפיום אישה'
    assert norm.normalize('כוסות ח"פ 100 יח') == "כוסות חד פעמי 100 יח'"


def test_truncated_word_completed_only_when_safe(norm):
    cut = "חטיף שוקול"
    assert norm.normalize(cut) == "חטיף שוקולד"
    # "ילד" appears mid-name in the corpus, so it is a word, not a stub.
    assert norm.normalize("שמפו ילד") == "שמפו ילד"


def test_nothing_removed(norm):
    """The cleaned name keeps every content word of the raw one."""
    raw = 'מארז חטיפי גרנולה שקדים 5*35גר בד"צ'
    clean = norm.normalize(raw)
    for word in ["מארז", "חטיפי", "גרנולה", "שקדים", "5*35", 'בד"צ']:
        assert word in clean
    assert clean == 'מארז חטיפי גרנולה שקדים 5*35 גרם בד"צ'


def test_each_step_can_be_switched_off():
    off = HebrewItemNormalizer(
        NormalizerConfig(
            unicode=False,
            spacing=False,
            flipped_decimals=False,
            units=False,
            abbreviations=False,
            geresh_variants=False,
            spelling_whitelist=False,
            complete_truncated=False,
            edge_junk=False,
        )
    ).fit(["x"])
    assert off.normalize("200מל  ג`ל") == "200מל ג`ל"  # only whitespace collapsed


def test_save_load_round_trip(norm, tmp_path):
    path = str(tmp_path / "norm.json")
    norm.save(path)
    again = HebrewItemNormalizer.load(path)
    for s in _corpus() + ["חטיף שוקול", "ציפס 200מל", "שוק.חלב"]:
        assert again.normalize(s) == norm.normalize(s)


# ------------------------------------------------------------------ fields
def test_extract_fields_pack_and_base_quantity():
    f = extract_fields("מארז חטיפי גרנולה 5*35 גרם")
    assert f["pack_count"] == 5 and f["qty"] == 35.0 and f["unit"] == "גרם"
    assert f["base_unit"] == "g" and f["base_qty"] == 175.0
    assert "מארז" in f["promo"]


def test_extract_fields_units_percent_and_missing():
    f = extract_fields('נקטר 1.5 ליטר 12% בד"צ חלבי UOY')
    assert f["base_unit"] == "ml" and f["base_qty"] == 1500.0
    assert f["percent"] == 12.0 and f["kosher"] == ['בד"צ'] and f["dietary"] == ["חלבי"]
    assert f["latin"] == ["uoy"]
    g = extract_fields("מברשת שיער")
    assert g["qty"] is None and g["base_qty"] is None and g["percent"] is None


# ------------------------------------------------------------------ dataset
def test_clean_dataframe_drops_only_placeholders_and_reports_conflicts():
    df = pd.DataFrame(
        {
            "text": ["Rejected", "ציפס 200מל", "צ'יפס 200 מ\"ל", "שמפו", "שמפו"],
            "label": ["X", "חטיפים", "ממתקים", "", "קוסמטיקה"],
        }
    )
    # Tiny table: lower the mining threshold so the two spellings are merged.
    fitted = HebrewItemNormalizer(NormalizerConfig(min_variant_count=1)).fit(df["text"].tolist())
    out, norm, rep = clean_dataframe(df, "text", "label", normalizer=fitted)
    assert rep["placeholders_dropped"] == 1 and len(out) == 4
    assert rep["unlabeled_rows"] == 1
    # The two chips spellings collapse to one name with two labels -> a conflict.
    assert rep["conflicting_names"] == 1
    assert {"text_clean", "base_qty", "pack_count", "truncated"} <= set(out.columns)


def test_module_never_uses_ascii_only_word_classes():
    """pandas 3 (pyarrow) treats \\w/\\b as ASCII-only; Hebrew must use explicit
    ranges. Guard against someone 'simplifying' a regex back to \\w."""
    src = inspect.getsource(ht)
    code = "\n".join(line for line in src.splitlines() if not line.strip().startswith("#"))
    body = code.split('"""', 2)[2]  # skip the module docstring, which explains this
    assert "\\w" not in body.replace("\\\\w", "") and "\\b" not in body.replace("\\\\b", "")
