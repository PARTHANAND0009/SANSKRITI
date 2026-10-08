"""String normalisation and matching shared by the data stages and the figures."""

from __future__ import annotations

import re

WS = r"\s+"


def norm_text(s) -> str:
    """Lowercase, underscores to spaces, whitespace collapsed (state names use underscores)."""
    return re.sub(WS, " ", str(s).replace("_", " ").lower()).strip()


def word_pattern(phrase: str, flexible_space: bool = False) -> str:
    """Regex for `phrase` as a whole word or phrase; optionally any run of whitespace between words."""
    body = re.escape(phrase)
    if flexible_space:
        body = body.replace(r"\ ", WS)
    return r"(?<!\w)" + body + r"(?!\w)"
