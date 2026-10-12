"""An outline for a PDF that has no table of contents."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Heading", "headings_from_typography", "outline_from_spans"]

# A heading is short. Beyond this it is a sentence, whatever size it is set in.
_MAX_HEADING_CHARS = 90
# How much larger than the body font a line must be set to count.
_MIN_RATIO = 1.12
# Where on the page a heading sits, as a fraction of its height.
_TOP_FRACTION = 0.45
_MIN_BODY_SPANS = 20
# A heading has words. "$3,780m" has one letter, and it is the unit.
_MIN_HEADING_LETTERS = 3

_MOSTLY_DIGITS = re.compile(r"^[\s\d.,%()$\u2013\u2014+-]*$")
_SENTENCE_END = re.compile(r"[.!?]\s*$")


@dataclass(frozen=True, slots=True)
class Span:
    """One run of text at one size, as the PDF sets it."""

    page_no: int
    text: str
    size: float
    top: float
    page_height: float


@dataclass(frozen=True, slots=True)
class Heading:
    text: str
    page_no: int
    level: int


def _is_heading(span: Span, body_size: float) -> bool:
    text = span.text.strip()
    if not text or len(text) > _MAX_HEADING_CHARS:
        return False
    if span.size < body_size * _MIN_RATIO:
        return False
    # A number set in large type is a number, not a heading. These packs set their key
    # figures in display type ("$3,780m", "83¢", "12.4%"), which this would otherwise pick
    # up. The test looks at letters rather than digits, because a unit suffix defeats a
    # digits-only pattern and a heading always has words.
    if _MOSTLY_DIGITS.match(text):
        return False
    if sum(character.isalpha() for character in text) <= _MIN_HEADING_LETTERS:
        return False
    if _SENTENCE_END.search(text):
        return False
    return not (span.page_height and span.top / span.page_height > _TOP_FRACTION)


def headings_from_typography(spans: list[Span]) -> list[Heading]:
    """Headings recovered from font size and position.

    Returns an empty list instead of guessing when the document has too little text to
    establish a body size (a one-page release, a scan), because every judgement here is
    relative to that size.
    """
    sized = [span for span in spans if span.text.strip()]
    if len(sized) < _MIN_BODY_SPANS:
        return []

    # The body font is the size most characters are set in, not the most common span.
    # Otherwise a document with lots of short, large labels and a few long paragraphs would
    # take a heading size as its body size.
    weighted: Counter[float] = Counter()
    for span in sized:
        weighted[round(span.size, 1)] += len(span.text.strip())
    body_size = weighted.most_common(1)[0][0]

    candidates = [span for span in sized if _is_heading(span, body_size)]
    if not candidates:
        return []

    # Nesting by size, largest first. Capped at three levels: deeper is unreliable from
    # typography alone and nothing downstream reads past the second.
    levels = sorted({round(span.size, 1) for span in candidates}, reverse=True)[:3]
    rank = {size: index + 1 for index, size in enumerate(levels)}

    out: list[Heading] = []
    for span in sorted(candidates, key=lambda s: (s.page_no, s.top)):
        level = rank.get(round(span.size, 1))
        if level is None:
            continue
        out.append(Heading(text=span.text.strip(), page_no=span.page_no, level=level))
    return out


def outline_from_spans(spans: list[Span]) -> tuple[tuple[str, int], ...]:
    """The recovered headings in the shape ``parse_pdf`` already uses."""
    return tuple((heading.text, heading.page_no) for heading in headings_from_typography(spans))


def spans_from_pdf(path: Path, max_pages: int | None = None) -> list[Span]:
    """Text spans with their size and position, read once per document."""
    import pymupdf

    out: list[Span] = []
    with pymupdf.open(path) as document:  # type: ignore[no-untyped-call]
        limit = min(max_pages or document.page_count, document.page_count)
        for index in range(limit):
            page = document[index]
            height = abs(page.rect.height) or 1.0
            for block in page.get_text("dict").get("blocks", ()):
                for line in block.get("lines", ()):
                    text = "".join(run.get("text", "") for run in line.get("spans", ()))
                    sizes = [float(run.get("size", 0.0)) for run in line.get("spans", ())]
                    if not text.strip() or not sizes:
                        continue
                    out.append(
                        Span(
                            page_no=index + 1,
                            text=text,
                            size=max(sizes),
                            top=float(line.get("bbox", (0, 0, 0, 0))[1]),
                            page_height=height,
                        )
                    )
    return out
