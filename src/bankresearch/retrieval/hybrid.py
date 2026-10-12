"""Retrieval over ingested chunks."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date

from .chunks import Chunk

__all__ = ["Hit", "Retriever", "tokenize"]

_TOKEN = re.compile(r"[a-z0-9][a-z0-9.%-]*", re.IGNORECASE)
_NUMBER_COMMA = re.compile(r"(?<=\d),(?=\d)")

_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "by",
        "for",
        "from",
        "had",
        "has",
        "have",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "were",
        "will",
        "with",
        "which",
        "this",
        "these",
        "those",
        "than",
        "then",
    ]
)

# Abbreviations expand to the token *sequence* their spelled-out form produces, so a
# search for "NIM" matches a document that writes "net interest margin" and the reverse.
# Expanding to one glued token would match neither.
_SYNONYMS: dict[str, tuple[str, ...]] = {
    "nim": ("net", "interest", "margin"),
    "nii": ("net", "interest", "income"),
    "cet1": ("common", "equity", "tier", "1"),
    "rwa": ("risk", "weighted", "assets"),
    "roe": ("return", "on", "equity"),
    "cti": ("cost", "to", "income"),
    "npat": ("net", "profit", "after", "tax"),
    "pcp": ("prior", "corresponding", "period"),
    "drp": ("dividend", "reinvestment", "plan"),
    "irrbb": ("interest", "rate", "risk", "banking", "book"),
    "ecl": ("expected", "credit", "loss"),
    "bps": ("basis", "points"),
}

_TABLE_BOOST = 1.25
_BOILERPLATE_PENALTY = 0.35


def tokenize(text: str) -> list[str]:
    """Tokenise for a corpus whose vocabulary is figures and fixed idiom.

    Period labels (``1H26``), capital ratios (``CET1``) and money amounts (``11,371``)
    must survive intact; a generic word splitter destroys all three.
    """
    cleaned = _NUMBER_COMMA.sub("", text.lower())
    tokens: list[str] = []
    for match in _TOKEN.finditer(cleaned):
        token = match.group(0).strip(".-%")
        if not token or token in _STOPWORDS:
            continue
        tokens.append(token)
        tokens.extend(_SYNONYMS.get(token, ()))
    return tokens


@dataclass(frozen=True, slots=True)
class Hit:
    chunk: Chunk
    score: float


@dataclass
class Retriever:
    """BM25 with structural boosting, scoped in code."""

    k1: float = 1.5
    b: float = 0.75
    chunks: list[Chunk] = field(default_factory=list)
    _tokens: list[list[str]] = field(default_factory=list)
    _tf: list[Counter[str]] = field(default_factory=list)
    _df: Counter[str] = field(default_factory=Counter)
    _total_length: int = 0

    def add(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            tokens = tokenize(chunk.text)
            if not tokens:
                continue
            self.chunks.append(chunk)
            self._tokens.append(tokens)
            counts = Counter(tokens)
            self._tf.append(counts)
            for term in counts:
                self._df[term] += 1
            self._total_length += len(tokens)

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def average_length(self) -> float:
        return self._total_length / len(self.chunks) if self.chunks else 0.0

    def _idf(self, term: str) -> float:
        n = len(self.chunks)
        df = self._df.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(
        self,
        query: str,
        bank_id: str,
        periods: tuple[str, ...],
        as_of: date,
        doc_types: tuple[str, ...] | None = None,
        limit: int = 8,
    ) -> list[Hit]:
        """Ranked chunks for a query, scoped to one issuer and two periods."""
        query_terms = Counter(tokenize(query))
        if not query_terms or not self.chunks:
            return []

        allowed_periods = {period.upper() for period in periods}
        average = self.average_length or 1.0
        hits: list[Hit] = []

        for index, chunk in enumerate(self.chunks):
            # Scope is applied here so a caller cannot widen it.
            if chunk.bank_id != bank_id:
                continue
            if chunk.period.upper() not in allowed_periods:
                continue
            if chunk.publish_date > as_of:
                continue
            if doc_types is not None and chunk.doc_type not in doc_types:
                continue

            counts = self._tf[index]
            length = len(self._tokens[index])
            score = 0.0
            for term, in_query in query_terms.items():
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (1 - self.b + self.b * length / average)
                score += (
                    self._idf(term) * (frequency * (self.k1 + 1) / denominator) * min(in_query, 3)
                )
            if score <= 0:
                continue

            if chunk.kind == "table":
                score *= _TABLE_BOOST
            elif chunk.kind == "boilerplate":
                score *= _BOILERPLATE_PENALTY
            hits.append(Hit(chunk=chunk, score=score))

        hits.sort(key=lambda hit: (-hit.score, hit.chunk.doc_id, hit.chunk.page_no))
        return hits[:limit]
