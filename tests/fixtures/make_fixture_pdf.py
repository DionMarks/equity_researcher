"""Generates the synthetic PDF the ingestion tests parse."""

from __future__ import annotations

from pathlib import Path

BANNER = "SYNTHETIC TEST FIXTURE: INVENTED FIGURES, NOT A REAL DISCLOSURE"

FIXTURE = Path(__file__).parent / "synthetic_results_document.pdf"


def _table(page, top: float, left: float, rows: list[list[str]], widths: list[float]) -> float:
    """Draw a ruled table and return the y position just below it."""
    import pymupdf

    line_height = 18.0
    xs = [left]
    for width in widths:
        xs.append(xs[-1] + width)
    bottom = top + line_height * len(rows)

    for index in range(len(rows) + 1):
        y = top + line_height * index
        page.draw_line(pymupdf.Point(xs[0], y), pymupdf.Point(xs[-1], y), width=0.6)
    for x in xs:
        page.draw_line(pymupdf.Point(x, top), pymupdf.Point(x, bottom), width=0.6)

    for row_index, row in enumerate(rows):
        baseline = top + line_height * row_index + 13
        for column_index, cell in enumerate(row):
            page.insert_text(pymupdf.Point(xs[column_index] + 4, baseline), cell, fontsize=8.5)
    return bottom


def build(path: Path = FIXTURE) -> Path:
    import pymupdf

    document = pymupdf.open()

    # --- page 1: cover -------------------------------------------------
    page = document.new_page()
    page.insert_text((60, 70), BANNER, fontsize=9)
    page.insert_text((60, 120), "Example Bank Limited", fontsize=20)
    page.insert_text((60, 150), "Results document", fontsize=14)
    page.insert_text((60, 175), "Half year ended 31 March 2026", fontsize=11)
    page.insert_text((60, 200), "All figures in Australian dollars.", fontsize=9)

    # --- page 2: key metrics ------------------------------------------
    page = document.new_page()
    page.insert_text((60, 70), BANNER, fontsize=9)
    page.insert_text((60, 110), "Key metrics", fontsize=14)
    page.insert_text((60, 135), "Group performance ($M unless stated)", fontsize=9)
    _table(
        page,
        top=150,
        left=60,
        widths=[200, 80, 80, 80],
        rows=[
            ["", "Mar 26", "Sep 25", "Mar 25"],
            ["Cash profit", "4450", "4402", "4300"],
            ["Net interest income", "9600", "9310", "9000"],
            ["Other operating income", "2400", "2460", "2500"],
            ["Operating income", "12000", "11770", "11500"],
            ["Operating expenses", "5200", "5120", "5000"],
            ["Credit impairment charge", "500", "440", "400"],
            ["Income tax expense", "1850", "1808", "1800"],
            ["Average interest earning assets", "1035000", "1020000", "1003000"],
            ["Average ordinary equity", "59380", "59000", "60180"],
            ["Net interest margin (%)", "1.86", "1.82", "1.80"],
            ["Cost to income (%)", "43.33", "43.50", "43.48"],
            ["CET1 ratio (%)", "12.39", "12.30", "12.21"],
            ["Return on equity (%)", "15.03", "14.88", "14.33"],
        ],
    )
    page.insert_text(
        (60, 440),
        "Comparatives have been restated for the sale of the insurance business.",
        fontsize=8,
    )

    # --- page 3: capital ----------------------------------------------
    page = document.new_page()
    page.insert_text((60, 70), BANNER, fontsize=9)
    page.insert_text((60, 110), "Capital", fontsize=14)
    page.insert_text((60, 135), "Capital adequacy ($M unless stated)", fontsize=9)
    _table(
        page,
        top=150,
        left=60,
        widths=[200, 80, 80, 80],
        rows=[
            ["", "Mar 26", "Sep 25", "Mar 25"],
            ["Common Equity Tier 1 capital", "50799", "49803", "48840"],
            ["Credit risk weighted assets", "338000", "334000", "330000"],
            ["Interest rate risk in the banking book", "21000", "20500", "20000"],
            ["Market risk weighted assets", "11000", "10400", "10000"],
            ["Operational risk weighted assets", "40000", "40000", "40000"],
            ["Total risk weighted assets", "410000", "404900", "400000"],
            ["CET1 ratio (%)", "12.39", "12.30", "12.21"],
        ],
    )

    # --- page 4: the margin walk --------------------------------------
    page = document.new_page()
    page.insert_text((60, 70), BANNER, fontsize=9)
    page.insert_text((60, 110), "Net interest margin movement", fontsize=14)
    page.insert_text((60, 135), "Net interest margin movement, 1H25 to 1H26 (bps)", fontsize=9)
    _table(
        page,
        top=150,
        left=60,
        widths=[260, 90],
        rows=[
            ["", "bps"],
            ["Net interest margin 1H25", "180.0"],
            ["Asset pricing and competition", "-7.0"],
            ["Deposit margins and competition", "4.0"],
            ["Replicating portfolio and capital hedge", "11.0"],
            ["Wholesale and term funding costs", "-2.0"],
            ["Net interest margin 1H26", "186.0"],
        ],
    )

    # --- page 5: disclaimer -------------------------------------------
    page = document.new_page()
    page.insert_text((60, 70), BANNER, fontsize=9)
    page.insert_text((60, 110), "Disclaimer", fontsize=14)
    page.insert_text(
        (60, 140),
        "Important notice: this document is prepared for illustration only and",
        fontsize=9,
    )
    page.insert_text(
        (60, 155), "contains no representation or warranty of any kind. 12.39 1.86 5016", fontsize=9
    )

    document.set_toc(
        [
            [1, "Cover", 1],
            [1, "Key metrics", 2],
            [1, "Capital", 3],
            [1, "Net interest margin movement", 4],
            [1, "Disclaimer", 5],
        ]
    )
    # Fixed metadata and no new document id, so regenerating gives the same bytes. CI
    # regenerates the fixture and fails if it differs from the committed one, which catches
    # a generator change that was never committed.
    document.set_metadata(
        {
            "title": "Synthetic results document (test fixture)",
            "author": "",
            "subject": "",
            "keywords": "",
            "creator": "",
            "producer": "",
            "creationDate": "D:20260101000000Z",
            "modDate": "D:20260101000000Z",
            "trapped": "",
            "format": "",
            "encryption": "",
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    document.save(path, deflate=True, no_new_id=True)
    document.close()
    return path


if __name__ == "__main__":
    print(f"wrote {build()}")
