"""Reading and writing validated figures, scoped so they can't leak."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable
from datetime import date

from ..domain.models import Basis, DocMeta, Fact, HeadlineLabel, RequestSpec, Restatement
from .db import decimal_from, decimal_to

__all__ = ["FactStore"]


def _fact_id(fact: Fact) -> str:
    """A stable id for a figure: where it was printed, and which reading of it this is.

    A variant excluding notable items is often printed on the same page as the headline,
    under its own heading, so it needs an id of its own or storing it would overwrite the
    headline. The headline keeps the id it always had.
    """
    key = f"{fact.bank_id}|{fact.period}|{fact.component}|{fact.doc_id}|{fact.page_no}"
    if fact.basis.ex_notable_items:
        key += "|ex_notable_items"
    return f"f_{hashlib.sha256(key.encode()).hexdigest()[:16]}"


class FactStore:
    """The fact, restatement, headline-label and document tables."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._db = connection

    # -- documents --------------------------------------------------------

    def put_document(self, meta: DocMeta) -> None:
        self._db.execute(
            """INSERT OR REPLACE INTO documents
               (doc_id, bank_id, period, doc_type, publish_date, url, sha256, page_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                meta.doc_id,
                meta.bank_id,
                meta.period,
                meta.doc_type,
                meta.publish_date.isoformat(),
                meta.url,
                meta.sha256,
                meta.page_count,
            ),
        )

    def documents_for(self, bank_id: str, periods: Iterable[str], as_of: date) -> list[DocMeta]:
        """Documents for these periods that existed on the as-of date."""
        labels = list(periods)
        if not labels:
            return []
        placeholders = ",".join("?" * len(labels))
        rows = self._db.execute(
            f"""SELECT * FROM documents
                WHERE bank_id = ? AND period IN ({placeholders}) AND publish_date <= ?
                ORDER BY doc_id""",
            (bank_id, *labels, as_of.isoformat()),
        ).fetchall()
        return [
            DocMeta(
                doc_id=row["doc_id"],
                bank_id=row["bank_id"],
                period=row["period"],
                doc_type=row["doc_type"],
                publish_date=date.fromisoformat(row["publish_date"]),
                url=row["url"],
                sha256=row["sha256"],
                page_count=row["page_count"],
            )
            for row in rows
        ]

    # -- facts ------------------------------------------------------------

    def forget_document(self, doc_id: str) -> int:
        """Drop everything previously read from one document, and say how much.

        A document's facts depend only on the document and the config, so re-reading one
        should replace what it said rather than add to it. Without this, a parser fix leaves
        the old figures in the store. After the divisional guard was added, CBA's Retail
        Banking margin of 2.50% was still there, outranking the 2.04% group margin the fix
        had just recovered.
        """
        before = self._db.execute(
            "SELECT COUNT(*) FROM facts WHERE doc_id = ?", (doc_id,)
        ).fetchone()[0]
        self._db.execute("DELETE FROM facts WHERE doc_id = ?", (doc_id,))
        self._db.execute("DELETE FROM facts_quarantine WHERE doc_id = ?", (doc_id,))
        self._db.execute("DELETE FROM headline_labels WHERE doc_id = ?", (doc_id,))
        self._db.commit()
        return int(before)

    def remove_document(self, doc_id: str) -> int:
        """Take a document out of the corpus entirely, and say how many figures went with it.

        ``forget_document`` clears what a document said so it can be read again, and leaves
        the document listed. This also drops the listing, its pages and tables, and any
        restatement that points at one of its figures, so the corpus is as it was before the
        document was ingested. A live eval needs that: the cold path ingests what it finds
        into the shared store, and a run that left Bendigo's filing behind turned three
        no-model abstention cases into answers.
        """
        removed = self.forget_document(doc_id)
        self._db.execute(
            """DELETE FROM restatements
               WHERE original_fact_id NOT IN (SELECT fact_id FROM facts)
                  OR restated_fact_id NOT IN (SELECT fact_id FROM facts)"""
        )
        self._db.execute(
            """DELETE FROM table_entries
               WHERE page_id IN (SELECT page_id FROM pages WHERE doc_id = ?)""",
            (doc_id,),
        )
        self._db.execute("DELETE FROM pages WHERE doc_id = ?", (doc_id,))
        self._db.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        self._db.commit()
        return removed

    def document_ids(self) -> set[str]:
        """Every document in the corpus, so a caller can put it back as it found it."""
        return {row[0] for row in self._db.execute("SELECT doc_id FROM documents")}

    def put_discovered_bank(
        self,
        bank_id: str,
        name: str,
        fy_end_month: int,
        ir_domains: Iterable[str],
        aliases: Iterable[str],
        confirmed_by: str,
        discovered_at: date,
        row_labels: Iterable[tuple[str, str]] = (),
    ) -> None:
        """Remember a bank that wasn't configured, once a filing has confirmed it."""
        self._db.execute(
            "INSERT OR REPLACE INTO discovered_banks (bank_id, name, fy_end_month, ir_domains, "
            "aliases, confirmed_by, discovered_at, row_labels) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                bank_id,
                name,
                fy_end_month,
                json.dumps(sorted(set(ir_domains))),
                json.dumps(sorted(set(aliases))),
                confirmed_by,
                discovered_at.isoformat(),
                json.dumps(sorted({tuple(pair) for pair in row_labels})),
            ),
        )

    def discovered_banks(self) -> list[dict[str, object]]:
        """Every bank remembered this way, as the fields its profile is built from."""
        rows = self._db.execute(
            "SELECT bank_id, name, fy_end_month, ir_domains, aliases, confirmed_by, row_labels "
            "FROM discovered_banks ORDER BY bank_id"
        ).fetchall()
        return [
            {
                "bank_id": row["bank_id"],
                "name": row["name"],
                "fy_end_month": int(row["fy_end_month"]),
                "ir_domains": tuple(json.loads(row["ir_domains"])),
                "aliases": tuple(json.loads(row["aliases"])),
                "confirmed_by": row["confirmed_by"],
                "row_labels": tuple(
                    (str(label), str(component))
                    for label, component in json.loads(row["row_labels"])
                ),
            }
            for row in rows
        ]

    def rows_sought(self, bank_id: str) -> set[str]:
        """The figures already looked for in each of a remembered bank's filings."""
        row = self._db.execute(
            "SELECT rows_sought FROM discovered_banks WHERE bank_id = ?", (bank_id,)
        ).fetchone()
        return set(json.loads(row["rows_sought"])) if row is not None else set()

    def put_rows_sought(self, bank_id: str, sought: Iterable[str]) -> None:
        """Note figures looked for, found or not, so they aren't asked for again."""
        self._db.execute(
            "UPDATE discovered_banks SET rows_sought = ? WHERE bank_id = ?",
            (json.dumps(sorted({*self.rows_sought(bank_id), *sought})), bank_id),
        )

    def put_row_labels(self, bank_id: str, row_labels: Iterable[tuple[str, str]]) -> None:
        """Keep the row labels matched for a remembered bank, once their figures passed."""
        self._db.execute(
            "UPDATE discovered_banks SET row_labels = ? WHERE bank_id = ?",
            (json.dumps(sorted({tuple(pair) for pair in row_labels})), bank_id),
        )

    def forget_discovered_bank(self, bank_id: str) -> None:
        self._db.execute("DELETE FROM discovered_banks WHERE bank_id = ?", (bank_id,))

    def put_fact(self, fact: Fact) -> str:
        """Store a validated figure. Refuses anything that failed validation."""
        if not fact.validated:
            raise ValueError(
                f"{fact.component} for {fact.bank_id} {fact.period} is not validated; "
                "unvalidated figures belong in facts_quarantine, which get_facts cannot read"
            )
        fact_id = fact.fact_id or _fact_id(fact)
        self._db.execute(
            """INSERT OR REPLACE INTO facts
               (fact_id, bank_id, period, component, value, published_value, unit, basis,
                doc_id, doc_type, page_no, table_id, column_header, publish_date,
                validated, is_restated)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)""",
            (
                fact_id,
                fact.bank_id,
                fact.period,
                fact.component,
                decimal_to(fact.value),
                None if fact.published_value is None else decimal_to(fact.published_value),
                fact.unit,
                fact.basis.model_dump_json(),
                fact.doc_id,
                fact.doc_type,
                fact.page_no,
                fact.table_id,
                fact.column_header,
                fact.publish_date.isoformat(),
                int(fact.is_restated),
            ),
        )
        return fact_id

    def quarantine(self, fact: Fact, check_failed: str) -> str:
        """Record a figure that failed an identity check, out of reach of reads."""
        fact_id = fact.fact_id or _fact_id(fact)
        self._db.execute(
            """INSERT OR REPLACE INTO facts_quarantine
               (fact_id, bank_id, period, component, value, unit, doc_id, page_no,
                publish_date, check_failed)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                fact_id,
                fact.bank_id,
                fact.period,
                fact.component,
                decimal_to(fact.value),
                fact.unit,
                fact.doc_id,
                fact.page_no,
                fact.publish_date.isoformat(),
                check_failed,
            ),
        )
        return fact_id

    def get_facts(self, spec: RequestSpec, components: Iterable[str] | None = None) -> list[Fact]:
        """Validated figures for this request, and nothing else.

        Scoped to the request's bank and its two periods, and filtered to the as-of date.
        Quarantined figures are in another table and can't be reached from here.
        """
        sql = [
            "SELECT * FROM facts",
            "WHERE bank_id = ? AND period IN (?, ?) AND publish_date <= ? AND validated = 1",
        ]
        params: list[object] = [
            spec.bank_id,
            spec.period,
            spec.comparison_period,
            spec.as_of.isoformat(),
        ]
        wanted = list(components) if components is not None else None
        if wanted:
            sql.append(f"AND component IN ({','.join('?' * len(wanted))})")
            params.extend(wanted)
        # Latest vintage first, so a restated comparative outranks its original.
        sql.append("ORDER BY component, period, publish_date DESC, fact_id")

        rows = self._db.execute(" ".join(sql), params).fetchall()
        return [
            Fact(
                fact_id=row["fact_id"],
                bank_id=row["bank_id"],
                period=row["period"],
                component=row["component"],
                value=decimal_from(row["value"]),
                published_value=(
                    decimal_from(row["published_value"])
                    if row["published_value"] is not None
                    else None
                ),
                unit=row["unit"],
                basis=Basis.model_validate(json.loads(row["basis"])),
                doc_id=row["doc_id"],
                doc_type=row["doc_type"],
                page_no=row["page_no"],
                table_id=row["table_id"],
                column_header=row["column_header"],
                publish_date=date.fromisoformat(row["publish_date"]),
                validated=True,
                is_restated=bool(row["is_restated"]),
            )
            for row in rows
        ]

    def quarantined_count(self) -> int:
        """For ingestion reporting only. Never used to answer a question."""
        row = self._db.execute("SELECT COUNT(*) AS n FROM facts_quarantine").fetchone()
        return int(row["n"])

    # -- restatements and labels ------------------------------------------

    def put_restatement(self, restatement: Restatement) -> None:
        self._db.execute(
            """INSERT OR REPLACE INTO restatements
               (original_fact_id, restated_fact_id, footnoted, note)
               VALUES (?, ?, ?, ?)""",
            (
                restatement.original_fact_id,
                restatement.restated_fact_id,
                int(restatement.footnoted),
                restatement.note or "",
            ),
        )

    def restatements_for(self, bank_id: str, period: str) -> list[Restatement]:
        rows = self._db.execute(
            """SELECT r.* FROM restatements r
               JOIN facts f ON f.fact_id = r.restated_fact_id
               WHERE f.bank_id = ? AND f.period = ?
               ORDER BY r.original_fact_id""",
            (bank_id, period),
        ).fetchall()
        return [
            Restatement(
                original_fact_id=row["original_fact_id"],
                restated_fact_id=row["restated_fact_id"],
                footnoted=bool(row["footnoted"]),
                note=row["note"] or None,
            )
            for row in rows
        ]

    def put_headline_label(self, label: HeadlineLabel) -> None:
        self._db.execute(
            """INSERT OR REPLACE INTO headline_labels
               (bank_id, period, metric_id, label, doc_id, page_no)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                label.bank_id,
                label.period,
                label.metric_id,
                label.label,
                label.doc_id,
                label.page_no,
            ),
        )

    def headline_label(self, bank_id: str, period: str, metric_id: str) -> HeadlineLabel | None:
        row = self._db.execute(
            """SELECT * FROM headline_labels
               WHERE bank_id = ? AND period = ? AND metric_id = ?""",
            (bank_id, period, metric_id),
        ).fetchone()
        if row is None:
            return None
        return HeadlineLabel(
            bank_id=row["bank_id"],
            period=row["period"],
            metric_id=row["metric_id"],
            label=row["label"],
            doc_id=row["doc_id"],
            page_no=row["page_no"],
        )
