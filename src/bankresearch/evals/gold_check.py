"""A person checking the answer key against the pages, and the record that they did."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path

from ..acquisition.manifest import load_manifest
from ..ingestion.artifacts import read_processed
from ..settings import Settings
from .cases import EvalCase

__all__ = ["GoldCheck", "checks_for", "record_verification", "review_sheet"]

#: Characters of the page shown either side of the figure.
_AROUND = 160


@dataclass(frozen=True, slots=True)
class GoldCheck:
    """One figure to confirm, with what its page says around it."""

    what: str
    value: Decimal | None
    unit: str
    doc_id: str | None
    page: int | None
    excerpt: str
    printed: bool
    """Whether the figure is printed on the page as given. A driver worked out from other
    figures isn't, and its excerpt says so."""


def _forms(value: Decimal, unit: str) -> list[str]:
    """How a figure is written on the page: 17,961 or (17,961); 1.55 or 1.55%."""
    magnitude = abs(value)
    if unit in ("aud_m", "aud_bn"):
        text = f"{magnitude:,}"
        return [text, f"({text})"]
    return [f"{magnitude}"]


def _excerpt(text: str, forms: list[str]) -> tuple[str, bool]:
    body = " ".join(text.split())
    for form in forms:
        match = re.search(rf"(?<![\d.,]){re.escape(form)}(?![\d])", body)
        if match is not None:
            start, end = max(0, match.start() - _AROUND), match.end() + _AROUND
            return f"...{body[start : match.start()]}>>{form}<<{body[match.end() : end]}...", True
    return "", False


def checks_for(case: EvalCase, settings: Settings) -> list[GoldCheck]:
    """Each gold figure in the case, beside its page."""
    pages: dict[tuple[str, int], str] = {}

    def page_text(doc_id: str | None, page: int | None) -> str:
        if doc_id is None or page is None:
            return ""
        key = (doc_id, page)
        if key not in pages:
            document = read_processed(doc_id, settings.processed_dir)
            found = document.page(page) if document is not None else None
            pages[key] = (found.text or "") if found is not None else ""
        return pages[key]

    checks: list[GoldCheck] = []
    for what, gold in (("current", case.current), ("prior", case.prior)):
        if gold is None:
            continue
        excerpt, printed = _excerpt(
            page_text(gold.doc_id, gold.page), _forms(gold.value, gold.unit)
        )
        checks.append(
            GoldCheck(
                what=f"{what} {case.metric}",
                value=gold.value,
                unit=gold.unit,
                doc_id=gold.doc_id,
                page=gold.page,
                excerpt=excerpt or "the figure is not on this page as given",
                printed=printed,
            )
        )
    labelled = [("driver", driver) for driver in case.drivers]
    labelled += [("walk driver", driver) for driver in case.walk_drivers]
    for kind, driver in labelled:
        unit = "bps" if case.current is not None and case.current.unit == "pct" else "aud_m"
        forms = _forms(driver.contribution, unit) if driver.contribution is not None else []
        excerpt, printed = _excerpt(page_text(driver.doc_id, driver.page), forms)
        checks.append(
            GoldCheck(
                what=f"{kind} {driver.driver_id}",
                value=driver.contribution,
                unit=unit,
                doc_id=driver.doc_id,
                page=driver.page,
                excerpt=excerpt
                or (
                    "worked out from figures on this page rather than printed: check the "
                    "case's notes for the arithmetic and the inputs on the page"
                ),
                printed=printed,
            )
        )
    return checks


def record_verification(cases_dir: Path, case_id: str, verifier: str, on: date) -> Path:
    """Write the verifier into the case, in place, and return the file it is in.

    Edited as text rather than reloaded and dumped, so the file's comments and layout,
    which carry most of the reasoning behind each label, survive.
    """
    verifier = " ".join(verifier.split())
    if not verifier:
        raise ValueError("a verification needs the verifier's name")
    for path in sorted(cases_dir.glob("*.yaml")):
        lines = path.read_text(encoding="utf-8").split("\n")
        try:
            start = lines.index(f"  - case_id: {case_id}")
        except ValueError:
            continue
        end = next(
            (
                index
                for index in range(start + 1, len(lines))
                if lines[index].startswith("  - case_id:") or lines[index].startswith("  # ---")
            ),
            len(lines),
        )
        block = lines[start:end]
        block = [line for line in block if not line.startswith("    verified_on:")]
        written = False
        for index, line in enumerate(block):
            if line.startswith("    labeller:"):
                block[index] = f'    labeller: "{verifier}"'
            elif line.startswith("    labeller_kind:"):
                block[index] = "    labeller_kind: human"
                block.insert(index + 1, f"    verified_on: {on.isoformat()}")
                written = True
                break
        if not written:
            raise ValueError(f"{case_id} has no labeller_kind line to record against")
        lines[start:end] = block
        path.write_text("\n".join(lines), encoding="utf-8")
        return path
    raise ValueError(f"no case {case_id} under {cases_dir}")


#: The box drawn round each kind of figure on its page, as (r, g, b) and as CSS.
_BOXES: dict[str, tuple[tuple[float, float, float], str]] = {
    "current": ((0.85, 0.1, 0.1), "#d91a1a"),
    "prior": ((0.1, 0.35, 0.85), "#1a59d9"),
    "driver": ((0.1, 0.6, 0.25), "#1a993f"),
}


def _box_of(what: str) -> str:
    return what.split()[0] if what.split()[0] in _BOXES else "driver"


def _page_image(pdf: Path, page: int, marks: list[tuple[str, list[str]]]) -> str:
    """The page as a JPEG data URI, with a box round every word that is one of the figures.

    Matched word by word rather than by substring, so 9,771 doesn't also box 19,771. A
    percentage is matched with or without its sign.
    """
    import base64

    import pymupdf

    with pymupdf.open(pdf) as document:  # type: ignore[no-untyped-call]
        sheet = document[page - 1]
        words = sheet.get_text("words")
        for kind, forms in marks:
            colour = _BOXES[kind][0]
            for word in words:
                text = str(word[4]).strip()
                if text in forms or text.rstrip("%") in forms:
                    x0, y0, x1, y1 = word[:4]
                    box = pymupdf.Rect(x0 - 2, y0 - 2, x1 + 2, y1 + 2)  # type: ignore[no-untyped-call]
                    sheet.draw_rect(box, color=colour, width=1.6)
        scale = pymupdf.Matrix(1.3, 1.3)  # type: ignore[no-untyped-call]
        image = sheet.get_pixmap(matrix=scale).tobytes("jpeg", jpg_quality=72)
    return "data:image/jpeg;base64," + base64.b64encode(image).decode()


def review_sheet(cases: list[EvalCase], settings: Settings) -> str:
    """Every case's gold on one page, each figure boxed on the page it cites.

    The quick way through the check: read each case's figures against the boxes,
    tick the cases that are right, and copy the one command that records them. Ticking
    records nothing; only that command, answered yes in a terminal, does.
    """
    from html import escape

    pdfs = {
        entry.doc_id: settings.raw_dir / f"{entry.sha256}.pdf"
        for entry in load_manifest()
        if entry.sha256
    }
    sections: list[str] = []
    for case in cases:
        checks = checks_for(case, settings)
        rows = []
        pages: dict[tuple[str, int], list[tuple[str, list[str]]]] = {}
        for check in checks:
            kind = _box_of(check.what)
            if check.doc_id is not None and check.page is not None:
                forms = _forms(check.value, check.unit) if check.value is not None else []
                if check.printed:
                    pages.setdefault((check.doc_id, check.page), []).append((kind, forms))
                else:
                    pages.setdefault((check.doc_id, check.page), [])
            found = "boxed on the page" if check.printed else "worked out, not printed"
            rows.append(
                f"<tr><td><span class='swatch' style='background:{_BOXES[kind][1]}'></span>"
                f"{escape(check.what)}</td><td class='num'>{escape(str(check.value))}</td>"
                f"<td>{escape(check.unit)}</td><td>{escape(str(check.doc_id))} "
                f"p.{check.page}</td><td class='{'ok' if check.printed else 'calc'}'>{found}</td></tr>"
            )
        images = []
        for (doc_id, page), marks in pages.items():
            pdf = pdfs.get(doc_id)
            if pdf is None or not pdf.exists():
                images.append(f"<p class='missing'>{escape(doc_id)} p.{page} isn't on disk.</p>")
                continue
            images.append(
                f"<figure><figcaption><a href='{pdf.resolve().as_uri()}#page={page}'>"
                f"{escape(doc_id)} p.{page}</a> (opens the PDF at this page)</figcaption>"
                f"<img loading='lazy' alt='{escape(doc_id)} page {page}' "
                f"src='{_page_image(pdf, page, marks)}'></figure>"
            )
        sections.append(
            f"<section id='{escape(case.case_id)}'><h2><label><input type='checkbox' "
            f"class='ok-box' value='{escape(case.case_id)}'> {escape(case.case_id)}</label></h2>"
            f"<p class='meta'>{escape(case.bank)} {escape(case.period)} {escape(case.metric)}"
            f" &middot; {len(checks)} figure(s) &middot; labelled by {escape(case.labeller)}</p>"
            + (f"<p class='notes'>{escape(case.notes)}</p>" if case.notes else "")
            + "<table><thead><tr><th>figure</th><th>value</th><th>unit</th><th>cited</th>"
            "<th>on the page</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
            + "".join(images)
            + "</section>"
        )
    return _SHEET.replace("{count}", str(len(cases))).replace("{sections}", "".join(sections))


_SHEET = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gold review</title>
<style>
  body { font: 15px/1.5 system-ui, sans-serif; margin: 0 auto; max-width: 1100px;
         padding: 16px 16px 120px; color: #1d1d1f; background: #fff; }
  h1 { font-size: 1.5rem; } h2 { font-size: 1.1rem; margin: 0 0 4px; }
  section { border-top: 1px solid #ddd; padding: 18px 0; }
  .meta, .notes { color: #555; margin: 2px 0 8px; } .notes { font-size: 0.9rem; }
  table { border-collapse: collapse; margin: 8px 0; font-size: 0.9rem; }
  td, th { padding: 4px 10px; border-bottom: 1px solid #eee; text-align: left; }
  .num { text-align: right; font-variant-numeric: tabular-nums; }
  .swatch { display: inline-block; width: 10px; height: 10px; margin-right: 6px; }
  .calc { color: #8a6d00; } .missing { color: #b00; }
  figure { margin: 8px 0; } figcaption { font-size: 0.85rem; margin-bottom: 4px; }
  img { max-width: 100%; border: 1px solid #ccc; }
  footer { position: fixed; bottom: 0; left: 0; right: 0; background: #f6f6f8;
           border-top: 1px solid #ccc; padding: 10px 16px; }
  footer code { display: block; overflow-x: auto; white-space: nowrap; margin-top: 6px; }
  input[type=text] { font: inherit; padding: 4px 8px; }
</style></head><body>
<h1>Gold review: {count} case(s)</h1>
<p>Each figure the eval scores against is boxed on the page it cites: red for the current
period, blue for the prior one, green for a driver printed on the page. A driver marked
"worked out" isn't printed; the case's note gives the arithmetic and its inputs are on the
page shown. Tick a case only when every figure in it is right as the page gives it.</p>
<p>Ticking records nothing. Copy the command below and run it in a terminal; it lists the
cases and asks you once before it records your name against them.</p>
{sections}
<footer>
  <label>Your name <input type="text" id="who" placeholder="as it should be recorded"></label>
  <button id="copy">Copy the command</button> <span id="ticked">0 ticked</span>
  <code id="command"></code>
</footer>
<script>
  const boxes = Array.from(document.querySelectorAll('.ok-box'));
  const who = document.getElementById('who');
  function command() {
    const ids = boxes.filter(b => b.checked).map(b => b.value);
    document.getElementById('ticked').textContent = ids.length + ' ticked';
    const name = (who.value || 'YOUR NAME').replace(/"/g, '');
    return ids.length ? '.venv/bin/bankresearch gold-check --split all --confirm ' +
      ids.join(',') + ' --verifier "' + name + '"' : '';
  }
  function refresh() { document.getElementById('command').textContent = command(); }
  boxes.forEach(b => b.addEventListener('change', refresh));
  who.addEventListener('input', refresh);
  document.getElementById('copy').addEventListener('click', () => {
    const text = command();
    if (text && navigator.clipboard) navigator.clipboard.writeText(text);
  });
</script>
</body></html>
"""
