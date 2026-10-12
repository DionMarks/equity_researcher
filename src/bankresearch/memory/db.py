"""The SQLite schema, and the two rules it enforces."""

from __future__ import annotations

import sqlite3
from decimal import Decimal
from pathlib import Path
from typing import Final

__all__ = ["SCHEMA_VERSION", "connect", "decimal_from", "decimal_to"]

SCHEMA_VERSION: Final = 1

_DDL: Final = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,
    bank_id      TEXT NOT NULL,
    period       TEXT NOT NULL,
    doc_type     TEXT NOT NULL,
    publish_date TEXT NOT NULL,
    url          TEXT NOT NULL,
    sha256       TEXT,
    page_count   INTEGER
);
CREATE INDEX IF NOT EXISTS ix_documents_scope ON documents (bank_id, period, publish_date);

CREATE TABLE IF NOT EXISTS pages (
    page_id      TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL REFERENCES documents (doc_id),
    page_no      INTEGER NOT NULL,
    kind         TEXT NOT NULL,
    heading_path TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS ix_pages_doc ON pages (doc_id, page_no);

CREATE TABLE IF NOT EXISTS table_entries (
    table_id       TEXT PRIMARY KEY,
    page_id        TEXT NOT NULL REFERENCES pages (page_id),
    caption        TEXT NOT NULL DEFAULT '',
    units          TEXT NOT NULL DEFAULT '',
    column_periods TEXT NOT NULL DEFAULT '{}',
    restated       INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_tables_page ON table_entries (page_id);

-- Figures. `value` is TEXT so a Decimal round-trips exactly.
CREATE TABLE IF NOT EXISTS facts (
    fact_id      TEXT PRIMARY KEY,
    bank_id      TEXT NOT NULL,
    period       TEXT NOT NULL,
    component    TEXT NOT NULL,
    value        TEXT NOT NULL,
    published_value TEXT,
    unit         TEXT NOT NULL,
    basis        TEXT NOT NULL DEFAULT '{}',
    doc_id       TEXT NOT NULL REFERENCES documents (doc_id),
    doc_type     TEXT NOT NULL,
    page_no      INTEGER NOT NULL,
    table_id     TEXT,
    column_header TEXT,
    publish_date TEXT NOT NULL,
    validated    INTEGER NOT NULL DEFAULT 0,
    is_restated  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_facts_lookup
    ON facts (bank_id, period, component, publish_date);

-- Figures that failed an identity check. `get_facts` never reads this table.
CREATE TABLE IF NOT EXISTS facts_quarantine (
    fact_id      TEXT PRIMARY KEY,
    bank_id      TEXT NOT NULL,
    period       TEXT NOT NULL,
    component    TEXT NOT NULL,
    value        TEXT NOT NULL,
    unit         TEXT NOT NULL,
    doc_id       TEXT NOT NULL,
    page_no      INTEGER NOT NULL,
    publish_date TEXT NOT NULL,
    check_failed TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS restatements (
    original_fact_id TEXT NOT NULL REFERENCES facts (fact_id),
    restated_fact_id TEXT NOT NULL REFERENCES facts (fact_id),
    footnoted        INTEGER NOT NULL DEFAULT 0,
    note             TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (original_fact_id, restated_fact_id)
);

CREATE TABLE IF NOT EXISTS headline_labels (
    bank_id   TEXT NOT NULL,
    period    TEXT NOT NULL,
    metric_id TEXT NOT NULL,
    label     TEXT NOT NULL,
    doc_id    TEXT NOT NULL REFERENCES documents (doc_id),
    page_no   INTEGER NOT NULL,
    PRIMARY KEY (bank_id, period, metric_id)
);

-- A bank nobody configured, kept once a filing confirmed what its ASX listing said: its
-- code, name and financial year end. Facts about the issuer, never about its results.
CREATE TABLE IF NOT EXISTS discovered_banks (
    bank_id       TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    fy_end_month  INTEGER NOT NULL,
    ir_domains    TEXT NOT NULL,
    aliases       TEXT NOT NULL,
    confirmed_by  TEXT NOT NULL,
    discovered_at TEXT NOT NULL,
    -- Row labels in the bank's own words, each with the component it holds, kept once the
    -- figures they gave passed their identity checks.
    row_labels    TEXT NOT NULL DEFAULT '[]',
    -- The figures a model was already asked to find in each of its filings, as
    -- "doc_id component", so one the filing doesn't have isn't asked for on every run.
    rows_sought   TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS llm_cache (
    request_hash TEXT PRIMARY KEY,
    model        TEXT NOT NULL,
    effort       TEXT NOT NULL,
    prompt_hash  TEXT NOT NULL,
    schema_hash  TEXT NOT NULL,
    response     TEXT NOT NULL,
    created_at   TEXT NOT NULL
);
"""


def connect(path: Path | str) -> sqlite3.Connection:
    """Open (and if necessary create) the research database."""
    target = Path(path)
    if str(target) != ":memory:":
        target.parent.mkdir(parents=True, exist_ok=True)
    # A generous busy timeout, because eval cases can run on several threads at once, each
    # with its own connection, and the cold path writes while others read.
    connection = sqlite3.connect(target, isolation_level=None, timeout=60)
    connection.row_factory = sqlite3.Row
    connection.executescript(_DDL)
    _add_missing_columns(connection)
    connection.execute(
        "INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('version', ?)",
        (str(SCHEMA_VERSION),),
    )
    return connection


#: Columns added to a table after it was first created, so a database made before them
#: gains them in place rather than failing on its first query.
_ADDED_COLUMNS: Final = (
    ("discovered_banks", "row_labels", "TEXT NOT NULL DEFAULT '[]'"),
    ("discovered_banks", "rows_sought", "TEXT NOT NULL DEFAULT '[]'"),
)


def _add_missing_columns(connection: sqlite3.Connection) -> None:
    for table, column, declaration in _ADDED_COLUMNS:
        present = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})")}
        if column not in present:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")


def decimal_to(value: Decimal) -> str:
    """Store a Decimal as text, preserving its exact published precision."""
    return str(value)


def decimal_from(value: str) -> Decimal:
    return Decimal(value)
