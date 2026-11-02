"""Error-message formatting shared by every layer.

Deliberately layer-neutral (it imports nothing from the package, so ``domain``
may use it without acquiring a dependency): validation failures happen at the
domain boundary, in the application services, and at the CLI edge, and they all
want to name the offending values without pasting a 4000-key list into a
traceback.
"""

from __future__ import annotations

from typing import Iterable

PREVIEW_LIMIT = 10


def format_preview(values: Iterable, limit: int = PREVIEW_LIMIT) -> str:
    """The first ``limit`` of ``values`` as a list literal, with a trailing
    ``...`` when there were more.

    The single formatting of the "here is what went wrong, abbreviated" idiom
    every validation error in the package reaches for, so a 4000-bad-label
    failure reports ten of them and a count rather than an unreadable wall of
    text — and so every one of those errors reads the same way to an operator.
    Only ``limit + 1`` values are materialized, so this is safe on a large or
    lazy iterable.
    """
    shown = []
    for value in values:
        shown.append(value)
        if len(shown) > limit:
            return f"{shown[:limit]} ..."
    return f"{shown}"
