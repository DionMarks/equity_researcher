"""A page's text split back into the sentences it prints."""

from __future__ import annotations

from bankresearch.retrieval.commentary import page_sentences


def _sentences(*lines: str) -> list[str]:
    return [sentence for sentence, heading in page_sentences("\n".join(lines)) if heading >= 0]


def test_a_word_that_looks_like_a_bullet_is_still_a_word():
    """CBA's FY25 pack prints its trading-income sentence one word to a line (p.30). The
    line "or" was read as an "o" bullet, so the sentence broke in two and the second half
    began "r 2% to $1,199 million"."""
    found = _sentences(
        "Trading",
        "income",
        "increased",
        "by",
        "$22 million",
        "or",
        "2%",
        "to",
        "$1,199 million, mainly driven by higher trading gains.",
    )
    assert found == [
        "Trading income increased by $22 million or 2% to $1,199 million, mainly driven by "
        "higher trading gains."
    ]


def test_a_bullet_glyph_still_starts_a_new_point():
    found = _sentences(
        "Costs rose on two things.", "o", "Wages rose on inflation.", "• Technology spend rose."
    )
    assert "Wages rose on inflation." in found
    assert "Technology spend rose." in found
