#!/usr/bin/env python3
"""Honest OCR eval metrics: codepoint CER AND grapheme-cluster CER.

For Brahmic / combining-mark scripts (Devanagari, Limbu, Newa, Tirhuta, ...), a
single "letter" a reader perceives is an extended grapheme cluster spanning
multiple codepoints (base + vowel signs + viramas + marks). Codepoint CER mis-counts
errors on those clusters; grapheme-cluster CER (Unicode UAX#29, via regex \X) counts
at the unit a reader actually sees. We report BOTH for transparency.
"""
from __future__ import annotations
import unicodedata
import regex
from rapidfuzz.distance import Levenshtein


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")


def graphemes(s: str) -> list[str]:
    return regex.findall(r"\X", _nfc(s))


def cer_codepoint(pred: str, gt: str) -> tuple[int, int]:
    """Return (edit_distance, gt_len) over codepoints (NFC)."""
    p, g = _nfc(pred), _nfc(gt)
    return Levenshtein.distance(p, g), len(g)


def cer_grapheme(pred: str, gt: str) -> tuple[int, int]:
    """Return (edit_distance, gt_len) over grapheme clusters (the reader's units)."""
    gp, gg = graphemes(pred), graphemes(gt)
    return Levenshtein.distance(gp, gg), len(gg)


def evaluate(pairs: list[tuple[str, str]]) -> dict:
    """Aggregate corpus metrics from (pred, gt) pairs. Micro-averaged CER (total
    edits / total length) is the standard; we also give exact-line accuracy."""
    cp_ed = cp_len = gr_ed = gr_len = exact = 0
    for pred, gt in pairs:
        e, l = cer_codepoint(pred, gt); cp_ed += e; cp_len += l
        e2, l2 = cer_grapheme(pred, gt); gr_ed += e2; gr_len += l2
        if _nfc(pred) == _nfc(gt):
            exact += 1
    n = max(len(pairs), 1)
    return {
        "n_lines": len(pairs),
        "cer_codepoint": cp_ed / max(cp_len, 1),
        "cer_grapheme": gr_ed / max(gr_len, 1),
        "exact_line_acc": exact / n,
        "codepoint_chars": cp_len,
        "grapheme_clusters": gr_len,
    }
