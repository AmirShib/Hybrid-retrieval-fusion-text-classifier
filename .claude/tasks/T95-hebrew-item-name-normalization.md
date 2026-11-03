# T95 — Hebrew item-name cleaning, parsing and normalization

status: in-progress (stages 0-4 implemented as functions; stage 5 wiring + ablation open)
tier: 3
depends_on: none (feeds every signal; pairs with T94 SPLADE domain adaptation)

## Goal
Turn raw supermarket item names (`examples/coicop_hebrew/_data.csv`,
`lbcItemName`) into (a) a **clean text** every signal sees, and (b) **structured
fields** (size, unit, pack count, percent, kosher/promo marks, brand, truncation
flag) the fusion model can use. The same normalizer, with the same versioned
dictionaries, must run at training and at inference.

## Evidence (profiled 2026-11-03 on the full file: 159,594 rows)

| Finding | Size | Example |
|---|---|---|
| Unlabeled rows | 22,562 (14.1%) | `קסמי שיניים דנטלי,` |
| Placeholder names | 68 rows `Rejected`/`rejected`, some *with* labels | `Rejected → עופות` |
| **Hard truncation at 40 chars** | 1,795 rows at len 40 vs 194 at 41 | `...מועשר בלחות 200מל`, `...עם % 37 שוקול` |
| CSV-escaped gershayim | 7,800 rows with runs of `""` | `מ""ל` (3,081) vs `מ"ל` (1,625), `תפו""ע` |
| Backtick used as geresh | 7,021 `` ` `` vs 5,773 `'` | `` ג`ל `` 671 / `ג'ל` 511 / `גל` 472 |
| Geresh dropped entirely | systematic | `צ'יפס` 301 / `ציפס` 346; `ג'ילט` 90 / `גילט` 95; `ק'ג` 83 / `קג` 1,891 |
| Number glued to word | 25.6% of rows (Hebrew→digit 5.2%) | `500מל`, `קטן5`, `איח10`, `קלויים5*42גר` |
| Has a quantity + unit | 53.5% of rows | `200 מל`, `1 ליטר`, `500 גרם`, `25 . 1 קג` (=1.25, RTL-flipped) |
| Multipack / dimensions with `*`/`x` | 3.5% | `5*35גר` (pack), `30*7.60` (foil size), `*8` |
| Leading / trailing junk punctuation | 3,532 / 1,127 rows | `` `שפתון ``, `. מברשת שיער`, `*90 אשפתון` |
| Double spaces / NBSP / bidi marks | 4,697 / 402 / 8 | U+00A0, U+200E/F, soft hyphen, tab |
| Dot / geresh abbreviations | common | `ח.פ`, `א.ד.פ` (EDP), `ט.שונים`, `עוג.`, `עפר'`, `יח'`, `מס'` |
| Semantic abbreviations | must expand | `תפו"א` (potatoes) ≠ `תפו"ע` (apples), `ח"פ` (disposable) |
| Kosher / dietary marks | 1.8% | `בד"צ`, `כשל"פ`, `מהדרין`, `לל"ס`, `פרווה` |
| Latin tokens | 5.3% of rows | brands (`UOY`, `WNW`, `MAM`), hints (`SPF` 352, `EDT` 111, `EDP` 75), units (`ml`, `oz`) |
| Exact duplicate names | 9,663 | `20 באריזה home מפיות` ×72 |
| Same name, conflicting labels | 110 names / 390 rows | `LOVLIS מרכך` → cosmetics *and* laundry |
| Glued words | present | `שטוחקרם` |

**Traps found while profiling:**
- **pandas 3 string regexes run on pyarrow, where `\w` is ASCII-only:** every
  Hebrew letter is "non-word". Use explicit ranges (`א-ת`), never `\w`/`\b`.
- **Do not auto-normalize plene/defective spelling.** Deleting ו/י merges
  different words: `רוטב`/`רטוב`, `ירוק`/`יורק`, `גדול`/`גודל`, `שקית`/`שקיות`.
  Only a whitelist of verified pairs (`עפרון`/`עיפרון`, `לשער`/`לשיער`, `מאורז`/`מארז`).
- `גל` is both "gel" (geresh dropped) and "wave". Map it only in product
  context, or let the learned signals cope.

## Progress (2026-11-03)
Stages 0-4 are implemented in `text_classifier/infrastructure/hebrew_text.py`
(`HebrewItemNormalizer`, `clean_dataframe`, `extract_fields`), with 50 tests in
`tests/unit/test_hebrew_text.py`. Nothing removes product words: the cleaned text
has 3.5% *more* characters than the raw text (expansions, spaces).

On the full file: 86,225 rows changed; vocabulary 55,076 -> ~38.7k tokens;
singletons 32,446 -> ~19.2k; 68 placeholders dropped; 163 names with conflicting
labels (up from 110, because spelling variants now collapse). `base_qty` filled for
52.9% of rows.

Audit findings folded in: bare `<number> מ` is meters on foil and ml on cosmetics,
so it is left as written; mined geresh merges are limited to ג/צ/ז/ת, are ≥ 3 letters,
and exclude real words (צלי, דרגה, גוני, ...); truncation completion skips any stub
that appears mid-name elsewhere (ילד, שני). Not done: Stage 5 wiring into the
pipelines, glued-word splitting (`שטוחקרם`), and the ablation.

## Design

Output per item:

```
raw          original string (kept for audit, never modified)
text_clean   normalized string, which every signal sees
text_core    text_clean minus size/pack/promo/kosher tokens (for dense and cross-encoder)
fields       {qty, unit, base_qty (g|ml|unit), pack_count, dims, percent,
              promo, kosher, latin_tokens, brand, truncated}
```

### Stage 0: Data hygiene (dataset level, training only)
1. Drop placeholder names (`rejected`, case-insensitive; empty after cleaning).
2. Unlabeled rows are **not** dropped: they become the unlabeled pool for
   domain-adaptive MLM pretraining (T94) and self-training.
3. Conflicting labels: write `conflicts.csv` for human review. Until reviewed,
   use the majority label, or drop the item if it's a tie.
4. Duplicates: keep them (frequency is real signal), but **split folds by
   normalized name** (group k-fold) so a duplicate never sits on both sides of a
   split. Today's exact duplicates inflate metrics.

### Stage 1: Character normalization (lossless in meaning)
NFC; remove bidi marks (U+200E/F), soft hyphen, niqqud/cantillation
(U+0591–U+05C7); NBSP/tab → space; en/em dash → `-`; `״`→`"`, `׳`/`` ` ``/`´`/`‘`/`’`→`'`;
collapse runs of `"` (the CSV-escape artifact) to one; collapse whitespace;
strip leading/trailing punctuation **after** Stage 3 has read `%`/`*`/`+`.

### Stage 2: Token boundary repair
- Split digit↔Hebrew boundaries (`500מל`→`500 מל`, `קטן5`→`קטן 5`), keeping
  decimals (`7.60`) and `N*M` / `NxM` as single tokens.
- Repair RTL-flipped decimals `25 . 1 קג` → `1.25 קג`, but only when followed by a
  unit and the reversed reading is plausible; otherwise leave it and flag it.
- Dotted abbreviations stay one token (`ח.פ`, `א.ד.פ`) and go to the Stage 4
  dictionary.
- Glued words (`שטוחקרם`): split only when both halves are frequent vocabulary
  words and the joined form is rare (unigram-frequency segmentation).

### Stage 3: Structured extraction (regex with explicit Hebrew ranges)
| Field | Patterns (after Stage 1/2) | Normalized to |
|---|---|---|
| qty + unit | `מ"ל מל ml cc סמ"ק`, `ליטר ל' L`, `גרם גר ג' gr g`, `ק"ג קג kg`, `יח' יחידות`, `מטר מ' ס"מ` | `base_qty` in g / ml / units / cm |
| pack_count | `N*Q unit`, `x6`, `*8`, `מארז N`, `שלישיה`/`רביעיה` | int |
| dims | `30*7.60` (no unit, two numbers) | `(a, b)` |
| percent | `N%`, `% N` | float (fat %, alcohol %, cocoa %) |
| promo | `מבצע`, `1+1`, `מארז`, `במתנה`, `חינם`, `הנחה` | bool + token |
| kosher | `בד"צ/ץ`, `כשל"פ`, `מהדרין`, `לל"ס`; `חלבי`/`בשרי`/`פרווה` | flags (the dietary ones are category hints) |
| latin | `[A-Za-z]{2,}` | lowercased list; `SPF`/`EDT`/`EDP` kept as hint tokens |
| truncated | `len(raw) == 40` and the last token is not in the vocabulary | bool |

### Stage 4: Canonicalization dictionary (versioned JSON, shipped with the model)
1. **Units:** every spelling → one canonical token (`מ"ל`, `ל'`, `גרם`, `ק"ג`, `יח'`).
2. **Geresh variants:** map with and without geresh to one form, using mined
   pairs where both spellings are frequent (`צ'יפס`, `ג'ילט`, `ג'ל`, `צ'ילי`,
   `קפוצ'ינו`, ...).
3. **Abbreviations:** `תפו"א`→`תפוחי אדמה`, `תפו"ע`→`תפוחי עץ`, `ח"פ`/`ח.פ`→`חד פעמי`,
   `ט.שונים`→`טעמים שונים`, `א.ד.פ`→`EDP`, `א.ד.ט`→`EDT`, `עוג.`→`עוגיות`, `עפר'`→`עפרון`, `מס'`→`מספר`.
   **Mining:** list tokens containing `'`, `"` or `.` by frequency. A human fills
   the top ~200, which should cover most of the abbreviation occurrences (to be
   measured).
4. **Truncated last token:** complete by unique prefix match against the full-word
   vocabulary, weighted by frequency (`שוקול`→`שוקולד`, `שעו`→`שעות`, `חידו`→`חידוד`).
   Only when unambiguous; otherwise keep the prefix and set `truncated`.
5. **Plene/defective:** whitelist only (see Traps).

### Stage 5: Plugging into the package
- A `TextNormalizer` port with a `HebrewRetailNormalizer` adapter. It's
  configured by a new `preprocessing` config section (default: off, so
  byte-identical) and applied in `TrainingPipeline` and `InferencePipeline` at the
  single place texts enter.
- Its dictionaries and version are saved into the model dir (JSON), so inference
  uses exactly the training-time normalization.
- `fields` reach the fusion model through a `FeatureProvider`. Item-level fields
  are constant across an item's candidates, so on their own a pointwise model
  can't use them. The provider instead emits **class-conditional** columns learned
  per fold, e.g. `log P(class | unit_type)`, `log P(class | brand)`,
  `log P(class | percent bucket)`, plus `qty_log` and `truncated` as-is.
- `text_core` vs `text_clean` per signal: a config choice (default: `text_clean`
  for all).

## Evaluation
Same seeds, group k-fold by normalized name, multilingual dense + BM25 (+ SPLADE
if available). Ablate one stage at a time:

1. baseline (raw) → 2. + Stage 0 (hygiene + group folds; expect metrics to *drop*,
   and that is the honest number) → 3. + Stage 1 → 4. + Stage 2 → 5. + Stage 4
   dictionary → 6. + Stage 3 fields as features.

Report `candidate_recall`, `accuracy_on_accepted`, `coverage`, plus:
- vocabulary size and singleton-token rate (should fall sharply);
- share of rows changed by each stage;
- 50 random before/after pairs per stage, read by a human (catches over-normalization).

## Acceptance
- The normalizer is deterministic, versioned, persisted, and identical at train
  and inference (round-trip test).
- Unit tests for every pattern above, including the traps (`רוטב` ≠ `רטוב`,
  decimals preserved, `\w` not used).
- Offline: no external NLP dependency in the default path. Optional Dicta/YAP
  lemmatization behind a flag.
- Ablation table committed to this ticket, with any stage that doesn't help
  left off by default.

## Out of scope (follow-ups)
- Cleanlab-style mislabel detection on out-of-fold predictions.
- Domain-adaptive MLM pretraining of the SPLADE/cross-encoder backbone on the
  unlabeled pool (T94 follow-up).
- Character n-gram retrieval signal (robust to truncation and typos).
