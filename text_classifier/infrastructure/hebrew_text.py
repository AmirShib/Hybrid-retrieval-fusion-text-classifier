"""Cleaning and normalization of Hebrew supermarket item names (T95).

Every step *normalizes* or *adds*; none removes product information. Sizes,
brands, kosher marks, promotions and every content word stay in the text. What
changes is how they are written, so that the same thing is always spelled the
same way and each meaningful word is its own token.

Built from a profile of ``examples/coicop_hebrew/_data.csv`` (159,594 names):
CSV-escaped gershayim (``מ""ל``), backticks used as a geresh, numbers glued to
words, names cut at 40 characters, placeholder rows and conflicting labels.

Usage::

    norm = HebrewItemNormalizer().fit(train_texts)   # mines spelling variants + vocabulary
    norm.normalize("גילט סירייס גל גילוח 200מל")     # -> "ג'ילט סירייס ג'ל גילוח 200 מ"ל"
    norm.extract_fields(clean_text)                  # -> {"base_qty": 200.0, "base_unit": "ml", ...}
    norm.save("normalizer.json"); HebrewItemNormalizer.load("normalizer.json")

Regexes use explicit Hebrew ranges (``א-ת``), never ``\\w``/``\\b``:
pandas 3 evaluates string regexes with pyarrow, where ``\\w`` is ASCII-only and
every Hebrew letter would count as a non-word character.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

HEB = "א-ת"  # Hebrew letters א..ת (final forms included)

# ----------------------------------------------------------------- dictionaries
# Whole-token expansions. Each was checked against its contexts and labels in the
# 160k-row file (e.g. ח"פ appears on cups, plates and razors: "disposable"). Only
# unambiguous tokens are listed; single-letter abbreviations such as "מ." (מיץ /
# מברשת / מצופה ...) are deliberately absent -- they are split, not guessed.
ABBREVIATIONS: Dict[str, str] = {
    'ח"פ': "חד פעמי",
    "ח.פ": "חד פעמי",
    "ח.פ.": "חד פעמי",
    "ח.פעמי": "חד פעמי",
    'תפו"א': "תפוחי אדמה",
    'תפו"ע': "תפוחי עץ",
    "ט.שונים": "טעמים שונים",
    "א.ד.פ": "או דה פרפיום",
    "א.ד.פ.": "או דה פרפיום",
    "א.ד.ט": "או דה טואלט",
    "א.ד.ט.": "או דה טואלט",
    "א.אדפ": "אישה או דה פרפיום",
    "א.אדט": "אישה או דה טואלט",
    "ג.אדט": "גבר או דה טואלט",
    "ג.אדפ": "גבר או דה פרפיום",
    "א.ש": "אפטר שייב",
    "א.ש.": "אפטר שייב",
    "א.שייב": "אפטר שייב",
    'לל"ס': "ללא סוכר",
    "ל.סוכר": "ללא סוכר",
    "ל.גלוטן": "ללא גלוטן",
    "ל.מלח": "ללא מלח",
    "מס'": "מספר",
    "עפר'": "עפרון",
    "עוג.": "עוגיות",
    "שוק.חלב": "שוקולד חלב",
    "שוק.מריר": "שוקולד מריר",
    "מ.שיניים": "משחת שיניים",
    "ק.גוף": "קרם גוף",
    "ק.לחות": "קרם לחות",
    "ע.עיניים": "עפרון עיניים",
    "ע.שפתיים": "עפרון שפתיים",
    "ת.רחצה": "תחליב רחצה",
    "ט.שוקולד": "טעם שוקולד",
    "צ.שיער": "צבע שיער",
    "פ.יער": "פירות יער",
    "ש.שועל": "שיבולת שועל",
    "מ.אפ": "מייק אפ",
    "דאו.": "דאודורנט",
    "דאו'": "דאודורנט",
    "כשלפ": 'כשל"פ',
}

# Plene/defective spelling pairs that are verified to be the same word. NOT a
# rule: deleting ו/י in general merges different words (רוטב sauce / רטוב wet,
# ירוק green / יורק spits, גדול big / גודל size). Canonical = the commoner form.
SPELLING_WHITELIST: Dict[str, str] = {
    "עיפרון": "עפרון",
    "לשער": "לשיער",
    "מאורז": "מארז",
}

# Unit spellings -> one canonical token. Applied only directly after a number,
# so a word like "ל" ("to") elsewhere is never touched. A bare "מ"/"מ'" after a
# number is deliberately NOT mapped: in the data it is meters on foil and cling
# film ("ניילון נצמד 30 מ") and ml on cosmetics ("דאודורנט 50 מ'"), so it stays as
# written rather than being guessed.
_UNIT_FORMS: List[Tuple[str, Tuple[str, ...]]] = [
    ('מ"ל', ('מ"ל', "מ'ל", "מל", 'מ"', "ml", "cc", 'סמ"ק', "סמק")),
    ("מטר", ("מטר", "מטרים")),
    ("ליטר", ("ליטר", "ל'", "לי", "ל", "lt", "ltr", "l")),
    ('ק"ג', ('ק"ג', "ק'ג", "קג", "קילו", "kg")),
    ("גרם", ("גרם", 'ג"ר', "גר'", "גר", "ג'", "ג", "gr", "g", "gram")),
    ("יח'", ("יחידות", "יחי'", "יח'", "יח")),
    ('ס"מ', ('ס"מ', "סמ", "cm")),
]
UNIT_CANON: Dict[str, str] = {form.lower(): canon for canon, forms in _UNIT_FORMS for form in forms}
_UNIT_BASE = {
    'מ"ל': ("ml", 1.0),
    "ליטר": ("ml", 1000.0),
    "גרם": ("g", 1.0),
    'ק"ג': ("g", 1000.0),
    "יח'": ("unit", 1.0),
    'ס"מ': ("cm", 1.0),
    "מטר": ("cm", 100.0),
}

# Geresh-less forms that are real Hebrew words, so a geresh spelling of the same
# letters is a *different* word and must not absorb them (found by auditing the
# mined map on the 160k-row file): roast, grade, rehab, shades, gentile, hue, waves.
NOT_GERESH_VARIANTS = frozenset({"צלי", "דרגה", "גמילה", "גוני", "גוי", "גון", "גלי"})
# Geresh variants accepted regardless of length; "גל" is overwhelmingly "gel" in
# product names ("גל גילוח", "גל רחצה").
GERESH_SEEDS: Dict[str, str] = {"גל": "ג'ל"}

PLACEHOLDERS = frozenset({"rejected"})
PROMO_WORDS = ("מבצע", "במתנה", "חינם", "הנחה", "בונוס", "1+1", "2+1", "מארז")
KOSHER_MARKS = ('בד"צ', 'בד"ץ', 'כשל"פ', "מהדרין", "כשר")
DIETARY_MARKS = ("חלבי", "בשרי", "פרווה", "טבעוני", "ללא גלוטן", "ללא סוכר")

# ------------------------------------------------------------- compiled regexes
_NIQQUD = re.compile("[֑-ׇ]")
_INVISIBLE = re.compile("[​-‏‪-‮⁦-⁩­﻿]")
_SPACES = re.compile(r"[ \t  -  　]+")
_QUOTE_RUN = re.compile(r'"{2,}')
_GERESH_RUN = re.compile(r"'{2,}")
_DIGIT_HEB = re.compile(rf"(?<=\d)(?=[{HEB}])")
_HEB_DIGIT = re.compile(rf"(?<=[{HEB}])(?=\d)")
# A dot gluing two real words ("שוק.חלב", "עוג.פפרדג"): split after the dot when
# a run of 2+ letters follows. Initialisms ("ב.ס.ד", "מ.ר.ז.ל") are untouched
# because what follows each of their dots is a single letter.
_DOT_GLUE = re.compile(rf"(?<=[{HEB}])\.(?=[{HEB}]{{2}})")
# "+" joining words ("פריכה+שוקולד") -> spaced; "1+1" stays.
_PLUS_GLUE = re.compile(
    rf"(?<=[{HEB}A-Za-z])\s*\+\s*(?=[{HEB}A-Za-z])|(?<=[{HEB}A-Za-z])\+(?=\s)|(?<=\s)\+(?=[{HEB}A-Za-z])"
)
# "25 . 1 קג" is 1.25 kg written with its parts flipped by RTL display. Only
# repaired when spaced on both sides of the dot AND followed by a unit.
_FLIPPED_DECIMAL = re.compile(r"(?<![\d.])(\d+) \. (\d+)(?=\s*[" + HEB + r"A-Za-z])")
_PERCENT_BEFORE = re.compile(r"%\s*(\d+(?:\.\d+)?)")
_PERCENT_SPACE = re.compile(r"(\d)\s+%")
_MULT = re.compile(r"(?<=\d)\s*\*\s*(?=\d)")
_LEAD_JUNK = re.compile(r"^[\s'\".\-_,;:|\\+`]+")
_TRAIL_JUNK = re.compile(r"[\s\"\-_,;:|\\+.]+$")
_TRAIL_LONE_GERESH = re.compile(rf"(?<![{HEB}])'+$")
_UNIT_AFTER_NUMBER = re.compile(
    r"(?<![\d.])(\d+(?:[.,]\d+)?)\s*("
    + "|".join(sorted((re.escape(f) for f in UNIT_CANON), key=len, reverse=True))
    + rf")\.?(?![{HEB}A-Za-z\"'])",
    re.IGNORECASE,
)
_TOKEN_SPLIT = re.compile(r"\s+")
_FOREIGN_SOUND = re.compile("[גצזת]'")
_HEB_WORD = re.compile(rf"^[{HEB}]+$")


@dataclass
class NormalizerConfig:
    """Which steps run. Every step is on by default; each one can be switched
    off to measure it (ablation) or because it does not suit a dataset."""

    unicode: bool = True  # NFC, invisible chars, niqqud, quote/geresh variants
    spacing: bool = True  # split glued numbers / dot-joined words / "+", collapse spaces
    flipped_decimals: bool = True  # "25 . 1 קג" -> "1.25 קג"
    units: bool = True  # one spelling per unit, only after a number
    abbreviations: bool = True  # verified whole-token expansions
    geresh_variants: bool = True  # "גילט" -> "ג'ילט" (mined by fit)
    spelling_whitelist: bool = True  # verified plene/defective pairs only
    complete_truncated: bool = True  # finish a word cut off by the source's length limit
    edge_junk: bool = True  # stray punctuation at the start/end of the name
    truncation_length: int = 40  # the source system's hard cap (profiled)
    min_variant_count: int = 5  # both spellings must appear this often to merge
    completion_min_count: int = 5  # a completion must have been seen this often
    completion_dominance: float = 3.0  # ... and be this many times commoner than the runner-up
    extra_abbreviations: Dict[str, str] = field(default_factory=dict)


class HebrewItemNormalizer:
    """Fit on the training names, then normalize any name the same way.

    ``fit`` learns two things from the corpus: which geresh-dropped spellings
    are the same word (``צ'יפס``/``ציפס``), and a word-frequency vocabulary used to
    complete words cut off at the source's 40-character limit. Everything else
    is fixed rules. Persist with ``save``/``load`` (JSON) so inference normalizes
    exactly as training did."""

    VERSION = 1

    def __init__(self, config: Optional[NormalizerConfig] = None):
        self.config = config or NormalizerConfig()
        self.geresh_map: Dict[str, str] = {}
        self.vocab: Counter = Counter()
        self.mid_vocab: Counter = Counter()
        self._abbrev = {**ABBREVIATIONS, **self.config.extra_abbreviations}

    # ------------------------------------------------------------------ fitting
    def fit(self, texts: Iterable[str]) -> "HebrewItemNormalizer":
        base = [self._rule_steps(t) for t in texts if not is_placeholder(t)]
        counts = Counter(tok for s in base for tok in _TOKEN_SPLIT.split(s) if tok)
        # Geresh variants: a token with an internal geresh whose geresh-less form
        # is also common is the same word written two ways. The geresh form is the
        # correct spelling, so it is the canonical one. Only a geresh after ג/צ/ז/ת
        # counts (the foreign-sound marker: ג'ל, צ'יפס, ז'קט, ת'ור); a geresh after
        # any other letter is an abbreviation or noise ("י'ח", "תפו'א") and must
        # not become the canonical form.
        m = self.config.min_variant_count
        mined = {
            tok.replace("'", ""): tok
            for tok, k in counts.items()
            if _FOREIGN_SOUND.search(tok)
            and "'" in tok[1:-1]
            and k >= m
            and counts.get(tok.replace("'", ""), 0) >= m
            and _HEB_WORD.match(tok.replace("'", ""))
            and len(tok.replace("'", "")) >= 3
            and tok.replace("'", "") not in NOT_GERESH_VARIANTS
        }
        self.geresh_map = {**mined, **GERESH_SEEDS}
        final = [self._apply_maps(s) for s in base]
        self.vocab = Counter()
        self.mid_vocab = Counter()
        for s in final:
            toks = [t for t in _TOKEN_SPLIT.split(s) if _HEB_WORD.match(t)]
            self.vocab.update(toks)
            # Words seen *before* the end of a name are complete words: a stub cut
            # by the length limit only ever appears last. Used to keep "ילד"
            # (child) from being "completed" to "ילדים".
            self.mid_vocab.update(t for t in _TOKEN_SPLIT.split(s)[:-1] if _HEB_WORD.match(t))
        return self

    # --------------------------------------------------------------- normalizing
    def normalize(self, text: str) -> str:
        """The cleaned name. Never shorter in content: only rewrites spellings,
        adds spaces and expansions, and trims stray edge punctuation."""
        raw = "" if text is None else str(text)
        s = self._apply_maps(self._rule_steps(raw))
        if self.config.complete_truncated:
            s = self._complete_last_token(s, at_cap=len(raw) >= self.config.truncation_length)
        return s

    def normalize_many(self, texts: Sequence[str]) -> List[str]:
        return [self.normalize(t) for t in texts]

    def _rule_steps(self, text: str) -> str:
        c = self.config
        s = str(text)
        if c.unicode:
            s = normalize_unicode(s)
        if c.flipped_decimals:
            s = fix_flipped_decimals(s)
        if c.abbreviations:
            # Before spacing: dot-glued abbreviations ("שוק.חלב", "ט.שונים") must be
            # looked up whole, before the dot split separates them. They are
            # looked up again after spacing for number-glued ones ("א.אדפ50").
            s = " ".join(self._abbrev.get(t, t) for t in _TOKEN_SPLIT.split(s) if t)
        if c.spacing:
            s = fix_spacing(s)
        if c.units:
            s = canonicalize_units(s)
        if c.edge_junk:
            s = strip_edge_junk(s)
        return collapse_spaces(s)

    def _apply_maps(self, s: str) -> str:
        c = self.config
        out: List[str] = []
        for tok in _TOKEN_SPLIT.split(s):
            if not tok:
                continue
            if c.abbreviations and tok in self._abbrev:
                tok = self._abbrev[tok]
            elif c.geresh_variants and tok in self.geresh_map:
                tok = self.geresh_map[tok]
            elif c.spelling_whitelist and tok in SPELLING_WHITELIST:
                tok = SPELLING_WHITELIST[tok]
            out.append(tok)
        s = " ".join(out)
        # An expansion can create a new dot-glued or spaced form; re-run spacing.
        return collapse_spaces(fix_spacing(s) if c.spacing else s)

    def _complete_last_token(self, s: str, at_cap: bool) -> str:
        """Finish a last word cut off by the source system ("שוקול" -> "שוקולד").

        The cut word itself is in the vocabulary (truncated names repeat), so
        the test is relative: the completion must be ``completion_dominance``
        times commoner than the cut form *and* than any other completion. At the
        40-char cap a 2-letter stub qualifies; below it (names are cut shorter
        too, e.g. "...24 שעו") the stub needs 3+ letters and 3x the dominance."""
        toks = s.split(" ")
        last = toks[-1] if toks else ""
        min_len, dom = (
            (2, self.config.completion_dominance)
            if at_cap
            else (3, 3 * self.config.completion_dominance)
        )
        if len(last) < min_len or not _HEB_WORD.match(last) or self.mid_vocab.get(last, 0) >= 2:
            return s
        cands = sorted(
            ((w, k) for w, k in self.vocab.items() if w.startswith(last) and w != last),
            key=lambda wk: -wk[1],
        )
        if not cands or cands[0][1] < self.config.completion_min_count:
            return s
        runner_up = cands[1][1] if len(cands) > 1 else 0
        own = self.vocab.get(last, 0)
        if cands[0][1] < dom * max(runner_up, own, 1):
            return s
        toks[-1] = cands[0][0]
        return " ".join(toks)

    # ----------------------------------------------------------------- fields
    def extract_fields(self, clean_text: str, raw_text: Optional[str] = None) -> Dict[str, Any]:
        """Structured attributes read from an already-normalized name. Nothing is
        removed from the text; this only reads it."""
        return extract_fields(
            clean_text,
            truncated=raw_text is not None and len(raw_text) >= self.config.truncation_length,
        )

    # ------------------------------------------------------------- persistence
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.VERSION,
            "config": asdict(self.config),
            "geresh_map": self.geresh_map,
            "vocab": dict(self.vocab),
            "mid_vocab": dict(self.mid_vocab),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HebrewItemNormalizer":
        if data.get("version") != cls.VERSION:
            raise ValueError(f"unsupported normalizer version {data.get('version')!r}")
        norm = cls(NormalizerConfig(**data["config"]))
        norm.geresh_map = dict(data["geresh_map"])
        norm.vocab = Counter(data["vocab"])
        norm.mid_vocab = Counter(data["mid_vocab"])
        return norm

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "HebrewItemNormalizer":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


# --------------------------------------------------------------- rule functions
def is_placeholder(text: object) -> bool:
    """A system placeholder in the name field (e.g. "Rejected"), not a product."""
    return str(text).strip().lower() in PLACEHOLDERS


def normalize_unicode(s: str) -> str:
    """NFC; drop invisible/bidi characters and niqqud; unify look-alike quotes.

    - geresh look-alikes (`` ` ``, ´, ‘, ’, ׳) -> ``'`` (the backtick alone is
      used as a geresh 7,021 times in the profiled file)
    - gershayim look-alikes (״, “, ”) -> ``"``
    - runs of ``"`` (CSV-escaping debris: ``מ""ל``) -> one ``"``
    - dashes -> ``-``, multiplication sign -> ``*``, NBSP/tab -> space
    """
    s = unicodedata.normalize("NFC", s)
    s = _INVISIBLE.sub("", s)
    s = _NIQQUD.sub("", s)
    s = s.translate(_CHAR_MAP)
    s = _QUOTE_RUN.sub('"', s)
    s = _GERESH_RUN.sub("'", s)
    return _SPACES.sub(" ", s)


_CHAR_MAP = str.maketrans(
    {
        "`": "'",
        "´": "'",
        "‘": "'",
        "’": "'",
        "׳": "'",
        "״": '"',
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "־": "-",
        "×": "*",
    }
)


def fix_flipped_decimals(s: str) -> str:
    """``"25 . 1 קג"`` -> ``"1.25 קג"``: a decimal whose two halves were swapped
    by right-to-left display. Only the spaced ``N . M`` form followed by a word
    is touched, which is how the flip shows up in the data."""
    return _FLIPPED_DECIMAL.sub(lambda m: f"{m.group(2)}.{m.group(1)}", s)


def fix_spacing(s: str) -> str:
    """Make every meaningful piece its own token; remove nothing.

    - number glued to a word: ``200מל`` -> ``200 מל``, ``קטן5`` -> ``קטן 5``
    - dot gluing two words: ``שוק.חלב`` -> ``שוק. חלב`` (initialisms kept)
    - ``+`` joining words: ``פריכה+שוקולד`` -> ``פריכה + שוקולד`` (``1+1`` kept)
    - percent: ``% 37`` / ``37 %`` -> ``37%``
    - multipacks: ``5 * 35`` -> ``5*35``
    """
    s = _DIGIT_HEB.sub(" ", s)
    s = _HEB_DIGIT.sub(" ", s)
    s = _DOT_GLUE.sub(". ", s)
    s = _PLUS_GLUE.sub(" + ", s)
    s = _PERCENT_BEFORE.sub(r"\1%", s)
    s = _PERCENT_SPACE.sub(r"\1%", s)
    s = _MULT.sub("*", s)  # only between two numbers: "וונוס *8" keeps its space
    return collapse_spaces(s)


def canonicalize_units(s: str) -> str:
    """One spelling per unit, written after its number with a space:
    ``200מל``/``200 מ""ל``/``200ml``/``200 מ`` -> ``200 מ"ל``; ``ג'``/``גר``/``gr`` ->
    ``גרם``; ``קג``/``ק'ג`` -> ``ק"ג``; ``ל'``/``ל`` -> ``ליטר``; ``יח``/``יחידות`` ->
    ``יח'``. A unit word not preceded by a number is left alone."""

    def repl(m: "re.Match[str]") -> str:
        num = m.group(1).replace(",", ".")
        return f"{num} {UNIT_CANON[m.group(2).lower()]}"

    return _UNIT_AFTER_NUMBER.sub(repl, s)


def strip_edge_junk(s: str) -> str:
    """Stray punctuation at the very start or end of the name (``"`שפתון"``,
    ``". מברשת שיער"``). A geresh that ends an abbreviation (``יח'``) is kept;
    ``%``, ``)`` and a leading ``*`` before a number are kept."""
    s = _LEAD_JUNK.sub("", s)
    s = _TRAIL_JUNK.sub("", s)
    s = _TRAIL_LONE_GERESH.sub("", s)
    return s.strip()


def collapse_spaces(s: str) -> str:
    return _SPACES.sub(" ", s).strip()


_QTY = re.compile(r"(\d+(?:\.\d+)?)\s+(" + "|".join(re.escape(u) for u in _UNIT_BASE) + r")")
_PACK = re.compile(
    r"(?<![\d.])(\d{1,3})\*(\d+(?:\.\d+)?)(?:\s+("
    + "|".join(re.escape(u) for u in _UNIT_BASE)
    + r"))?"
)
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)%")
_LATIN = re.compile(r"[A-Za-z][A-Za-z&]+")


def extract_fields(clean_text: str, truncated: bool = False) -> Dict[str, Any]:
    """Read structured attributes from a normalized name (text unchanged).

    Returns ``qty``/``unit`` (as written), ``base_qty``/``base_unit`` (g, ml,
    unit or cm), ``pack_count``, ``percent``, ``promo``, ``kosher``, ``dietary``
    (marks such as חלבי/פרווה, which are category hints), ``latin`` tokens and
    ``truncated``. Missing values are ``None``/empty, never 0."""
    out: Dict[str, Any] = {
        "qty": None,
        "unit": None,
        "base_qty": None,
        "base_unit": None,
        "pack_count": None,
        "percent": None,
        "promo": [w for w in PROMO_WORDS if w in clean_text],
        "kosher": [w for w in KOSHER_MARKS if w in clean_text],
        "dietary": [w for w in DIETARY_MARKS if w in clean_text],
        "latin": [t.lower() for t in _LATIN.findall(clean_text)],
        "truncated": bool(truncated),
    }
    pack = _PACK.search(clean_text)
    if pack:
        out["pack_count"] = int(pack.group(1))
        if pack.group(3):
            out["qty"], out["unit"] = float(pack.group(2)), pack.group(3)
    if out["unit"] is None:
        q = _QTY.search(clean_text)
        if q:
            out["qty"], out["unit"] = float(q.group(1)), q.group(2)
    if out["unit"] is not None:
        base_unit, factor = _UNIT_BASE[out["unit"]]
        out["base_unit"], out["base_qty"] = base_unit, out["qty"] * factor
        if out["pack_count"]:
            out["base_qty"] *= out["pack_count"]
    p = _PERCENT.search(clean_text)
    if p:
        out["percent"] = float(p.group(1))
    return out


# ------------------------------------------------------------- dataset helpers
def clean_dataframe(
    df: Any,
    text_col: str = "text",
    label_col: Optional[str] = "label",
    normalizer: Optional[HebrewItemNormalizer] = None,
) -> Tuple[Any, HebrewItemNormalizer, Dict[str, Any]]:
    """Clean a whole labelled table: fit the normalizer (unless given), add
    ``text_clean`` and the structured fields as columns, and report data issues.

    Rows are only *removed* when the name is a placeholder ("Rejected"); every
    real product is kept, including unlabeled ones (useful for pretraining and
    self-training). Returns ``(cleaned_df, normalizer, report)``; the report lists
    names that carry more than one label, for human review."""
    import pandas as pd

    data = df.copy()
    raw = data[text_col].astype(str)
    placeholder = raw.map(is_placeholder)
    data = data[~placeholder].copy()
    raw = raw[~placeholder]
    norm = normalizer or HebrewItemNormalizer().fit(raw.tolist())
    data["text_clean"] = [norm.normalize(t) for t in raw]
    fields = pd.DataFrame(
        [norm.extract_fields(c, r) for c, r in zip(data["text_clean"], raw)], index=data.index
    )
    data = pd.concat([data, fields], axis=1)
    report: Dict[str, Any] = {
        "rows_in": int(len(df)),
        "placeholders_dropped": int(placeholder.sum()),
        "rows_changed_by_cleaning": int((data["text_clean"] != raw).sum()),
        "distinct_names_raw": int(raw.nunique()),
        "distinct_names_clean": int(data["text_clean"].nunique()),
    }
    if label_col is not None and label_col in data:
        labelled = data[data[label_col].astype(str).str.strip() != ""]
        n_labels = labelled.groupby("text_clean")[label_col].nunique()
        conflicts = n_labels[n_labels > 1].index
        report["unlabeled_rows"] = int(len(data) - len(labelled))
        report["conflicting_names"] = int(len(conflicts))
        report["conflicts"] = (
            labelled[labelled["text_clean"].isin(conflicts)]
            .groupby("text_clean")[label_col]
            .agg(lambda s: sorted(set(s)))
            .to_dict()
        )
    return data, norm, report
