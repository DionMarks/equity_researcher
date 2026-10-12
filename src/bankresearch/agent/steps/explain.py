"""The narrative, and the entailment check over it."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Final

from ...domain.models import (
    Basis,
    Decomposition,
    DriverContribution,
    EvidenceCard,
    RequestSpec,
    ResolvedInputs,
)
from ...domain.registry import get_registry
from ...llm.budget import Budget
from ...llm.client import LLMClient, LLMRequest, LLMUnavailable, ModelUnreachable
from ...llm.tracing import PromptLibrary, Tracer, default_library
from ...output.render_md import (
    amount_places,
    format_change,
    format_magnitude,
    format_share,
    format_value,
)
from ...output.schema import build_change, share_of_change

__all__ = [
    "ClaimVerdict",
    "Draft",
    "NarrativeCheck",
    "check_narrative",
    "cited_sentences",
    "computed_lines",
    "name_every_material_driver",
    "numbers_in",
    "write_memo",
    "write_narrative",
]

# Any figure a reader would take as a claim: 1.53, 1,234, (42), 12.4%, 3bps.
_NUMBER = re.compile(r"\(?-?\$?\s?\d[\d,]*(?:\.\d+)?\)?%?")
_CITATION_ID = re.compile(r"\bev_\d+\b")


def numbers_in(text: str) -> set[str]:
    """Every figure in a piece of text, normalised for comparison.

    Normalised so ``1,234``, ``$1,234`` and ``(1,234)`` all compare equal to the value they
    state. The question is whether the figure was supplied, not how it was typeset.
    """
    found: set[str] = set()
    # An evidence id is a citation, not a figure. Counting the digits of "[ev_060]" as a
    # claim of 60 threw away a narrative for citing evidence that was in its brief.
    text = _CITATION_ID.sub(" ", text or "")
    for match in _NUMBER.finditer(text):
        token = match.group(0)
        cleaned = token.strip().strip("()").replace(",", "").replace("$", "").replace("%", "")
        cleaned = cleaned.strip()
        if not cleaned or cleaned in {"-", "."}:
            continue
        try:
            value = float(cleaned)
        except ValueError:
            continue
        found.add(f"{abs(value):g}")
    return found


def _supplied(
    spec: RequestSpec,
    decomposition: Decomposition,
    inputs: ResolvedInputs,
    cards: list[EvidenceCard],
    brief: str,
) -> set[str]:
    """Every figure the narrative is allowed to contain.

    Taken from the brief the writer was actually given, rather than rebuilt separately from
    the same objects. The two had drifted apart: the brief shows the decomposition's notes
    and any data conflicts, but the allowed set was built only from the figures, drivers and
    cards. So a run was given "annualised net interest income contributed 0.3bp and balance
    growth -3.4bp", repeated it faithfully, and had its whole narrative thrown away for
    inventing 0.3 and 3.4.

    Using the brief as the definition means that kind of mismatch can't happen: whatever the
    model was shown it may repeat, and anything else it made up. Adding a line to the brief
    can't quietly narrow what's allowed any more.
    """
    allowed = numbers_in(brief)
    # The same figures again, in the shapes `numbers_in` would not recover from prose:
    # a driver printed to 28 decimal places appears in the brief exactly, while the
    # narrative will sensibly round it.
    for value in (*inputs.current.values(), *inputs.prior.values()):
        allowed.add(f"{abs(float(value)):g}")
    allowed.add(f"{abs(float(decomposition.headline_change)):g}")
    allowed.add(f"{abs(float(decomposition.residual)):g}")
    for driver in decomposition.drivers:
        if driver.contribution is not None:
            allowed.add(f"{abs(float(driver.contribution)):g}")
    for card in cards:
        if card.value is not None:
            allowed.add(f"{abs(float(card.value)):g}")
    # Period labels are not claims. "1H26" contributes 26, and a year 2026.
    for token in (spec.period, spec.comparison_period):
        allowed.update(numbers_in(token))
    allowed.update({"2024", "2025", "2026", "2027"})
    return allowed


#: What the writer returns: the narrative, and one sentence per driver giving the cause the
#: issuer states for it.
_MEMO_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["narrative", "drivers"],
    "properties": {
        "narrative": {"type": "string"},
        "drivers": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["driver", "why"],
                "properties": {
                    "driver": {"type": "string"},
                    "why": {"type": "string"},
                },
            },
        },
    },
}

#: The words a cause is asked to fit in, which the prompt states.
CAUSE_WORDS = 40
#: The words past which a cause is dropped. The 25 live runs lost 8 faithful causes for
#: running past a 40-word limit nobody had told the writer about; with the limit stated,
#: this is headroom for a writer that runs a few words over, and past it a cause is a list
#: of everything in the passage rather than a cause.
CAUSE_WORD_LIMIT = 60


@dataclass
class Draft:
    """What the writer produced: the narrative, each driver's stated cause, and a note when
    something was held back."""

    narrative: str = ""
    causes: dict[str, tuple[str, list[str]]] = field(default_factory=dict)
    """``driver_id`` -> (the sentence, the evidence ids it cites)."""
    note: str = ""


def write_narrative(
    spec: RequestSpec,
    decomposition: Decomposition,
    inputs: ResolvedInputs,
    cards: list[EvidenceCard],
    client: LLMClient,
    *,
    budget: Budget | None = None,
    model: str = "claude-fable-5-1",
    effort: str = "high",
    tracer: Tracer | None = None,
    prompts: PromptLibrary | None = None,
    repair: list[tuple[str, str]] | None = None,
) -> tuple[str, str]:
    """The narrative alone, and a note when there isn't one. See :func:`write_memo`."""
    draft = write_memo(
        spec,
        decomposition,
        inputs,
        cards,
        client,
        budget=budget,
        model=model,
        effort=effort,
        tracer=tracer,
        prompts=prompts,
        repair=repair,
    )
    return draft.narrative, draft.note


def write_memo(
    spec: RequestSpec,
    decomposition: Decomposition,
    inputs: ResolvedInputs,
    cards: list[EvidenceCard],
    client: LLMClient,
    *,
    budget: Budget | None = None,
    model: str = "claude-fable-5-1",
    effort: str = "high",
    tracer: Tracer | None = None,
    prompts: PromptLibrary | None = None,
    repair: list[tuple[str, str]] | None = None,
    repair_causes: list[tuple[str, str, str]] | None = None,
) -> Draft:
    """The narrative and each driver's stated cause, or an empty draft and the reason.

    ``repair`` is the one rewrite allowed after the check: the sentences removed
    from the first draft and why. Cutting them out of the text left fragments whose subject
    had gone ("Both values are reported and the cause is unknown."), so the writer gets one
    chance to write the memo again without them.

    Returns rather than raises, because a missing narrative is a weaker answer, while a
    failed run isn't an answer at all.
    """
    if not client.available:
        return Draft(note="no model configured: the narrative is omitted rather than written blind")
    if budget is not None and not budget.may_call("explain"):
        return Draft(note="the explain step was skipped on its budget")

    library = prompts or default_library()
    prompt = library.get("explain")
    brief = _brief(spec, decomposition, inputs, cards)
    if repair:
        brief += (
            "\n\nA first draft was checked against its evidence, and these sentences were "
            "removed, with the checker's reason. Write the narrative again without these "
            "claims, keeping what was supported:\n"
            + "\n".join(f'- "{sentence}" ({reason})' for sentence, reason in repair)
        )
    if repair_causes:
        # Most causes the checker removes are stretched, not invented: the bank explains
        # credit RWA and the cause was given for all RWA, or it explains "other income" and
        # the cause was given for the whole of other operating income. Told why, the writer
        # can state the narrower cause the passage does support.
        brief += (
            "\n\nThese stated causes were left out, with the checker's reason. Where the "
            "bank's words support a narrower cause, give it again under the same driver id, "
            "saying which part it explains; otherwise leave that driver out:\n"
            + "\n".join(
                f'- {driver_id}: "{claim}" ({reason})' for driver_id, claim, reason in repair_causes
            )
        )
    request = LLMRequest(
        step="explain",
        model=model,
        effort=effort,
        system=prompt.text,
        messages=({"role": "user", "content": brief},),
        output_schema=_MEMO_SCHEMA,
    )
    try:
        response = client.complete(request)
    except LLMUnavailable as error:
        if isinstance(error, ModelUnreachable):  # the run stops; see llm.client
            raise
        return Draft(note=f"the narrative was not written: {error}")

    if budget is not None:
        budget.record("explain", model, response.usage, cached=response.cached)
    if tracer is not None:
        tracer.model_call(request, response, prompt=prompt)

    if response.stop_reason == "max_tokens":
        # Whatever came back stops mid-thought, and a memo that stops mid-sentence is worse
        # than none. The call was paid for and is recorded above.
        return Draft(note="the narrative was cut off at the step's output limit and was not used")
    answer = response.structured if isinstance(response.structured, dict) else {}
    narrative = str(answer.get("narrative") or "").strip() or (
        "" if answer else response.text.strip()
    )
    narrative = _cite_drivers(narrative, decomposition)
    if not narrative:
        return Draft(note="the explain step returned nothing")

    # A number in the prose that was never supplied was worked out by the model. The
    # arithmetic is usually right and the citation usually plausible, which is exactly why
    # it can't be let through: the whole point is that figures come from pages.
    supplied = _supplied(spec, decomposition, inputs, cards, brief)
    invented = sorted(numbers_in(narrative) - supplied)
    if invented:
        return Draft(
            note=(
                "the narrative was discarded: it contained "
                f"{len(invented)} figure(s) that were not in its inputs "
                f"({', '.join(invented[:5])}). Every number in a report must come from a page "
                "or from the decomposition."
            )
        )
    narrative, silent = _without_claims_of_silence(narrative)
    narrative, advice = _without_advice(narrative)
    narrative = _tidy_citations(narrative)
    narrative, note = _within_the_word_limit(narrative)
    if silent:
        note = "; ".join(
            part
            for part in (
                note,
                f"{silent} sentence(s) saying the bank gave no reason were taken out: the "
                "writer sees a selection of the bank's words, so an absence can't be cited",
            )
            if part
        )
    if advice:
        note = "; ".join(
            part
            for part in (
                note,
                f"{advice} sentence(s) telling the reader what to make of the figures were "
                "taken out or reduced to the fact they gave",
            )
            if part
        )
    return Draft(
        narrative=narrative,
        causes=_causes(answer.get("drivers") or (), decomposition, cards, supplied),
        note=note,
    )


#: A sentence saying the bank didn't explain something. The writer is shown a selection of
#: the bank's words, not all of them, so it can't know, and in the live runs NAB's expense
#: rise, said to come without a cause, was in fact explained on another page.
_SILENCE = re.compile(
    r"\b(?:gives? no (?:further )?(?:cause|reason|detail|explanation)"
    r"|(?:does|did) not (?:say|explain|state|give|break|disclose|attribute)"
    r"|doesn't (?:say|explain|state|give)"
    r"|without (?:giving |stating )?(?:a |any |further )?(?:cause|reason|detail|explanation)"
    r"|no (?:cause|reason) (?:is|was) (?:given|stated)"
    r"|(?:is|are|was|were) not (?:disclosed|broken down|split out)"
    r"|(?:isn't|aren't|wasn't|weren't) (?:disclosed|broken down|split out)"
    r"|not disclosed by"
    r"|only in aggregate)\b",
    re.IGNORECASE,
)


#: Advice to the reader at the start of a sentence, up to the fact it introduces. A live run
#: on Westpac's 1H24 margin closed "Do not read the 7.0 bps fall as the underlying movement:
#: the bank's Core NIM was 1.80%, down 9 basis points"; the fact stays, the advice goes.
_ADVICE_LEAD = re.compile(r"^\s*(?:do not|don't|never)\b[^:]{0,160}:\s*", re.IGNORECASE)
#: A sentence that is nothing but advice to the reader.
_ADVICE = re.compile(
    r"\b(?:investors|readers|analysts|you) should\b"
    r"|\bshould not be (?:read|taken|treated)\b"
    r"|^\s*(?:do not|don't) (?:read|conclude|treat|take|assume)\b",
    re.IGNORECASE,
)


def _without_advice(text: str) -> tuple[str, int]:
    """The text with advice to the reader taken out: a leading instruction is cut back to the
    fact after it, and a sentence that only advises is dropped. The memo states what the
    figures are; what to make of them is the analyst's."""
    kept: list[str] = []
    changed = 0
    for paragraph in text.split("\n\n"):
        survivors: list[str] = []
        for sentence in (s for s in _SENTENCE_END.split(paragraph.strip()) if s.strip()):
            lead = _ADVICE_LEAD.match(sentence)
            if lead is not None and sentence[lead.end() :].strip():
                rest = sentence[lead.end() :].strip()
                survivors.append(rest[0].upper() + rest[1:])
                changed += 1
            elif _ADVICE.search(sentence):
                changed += 1
            else:
                survivors.append(sentence)
        if survivors:
            kept.append(" ".join(survivors))
    return "\n\n".join(kept), changed


def _without_claims_of_silence(text: str) -> tuple[str, int]:
    """The text with every sentence that claims the bank was silent taken out, and how many."""
    kept: list[str] = []
    removed = 0
    for paragraph in text.split("\n\n"):
        survivors: list[str] = []
        for sentence in (s for s in _SENTENCE_END.split(paragraph.strip()) if s.strip()):
            if not _SILENCE.search(sentence):
                survivors.append(sentence)
                continue
            removed += 1
            # Only the clause that makes the claim goes: "Tax took A$208m off [ev_031]; the
            # bank gives no cause." keeps its figure.
            clauses = [c for c in re.split(r";\s+", sentence.rstrip(".")) if not _SILENCE.search(c)]
            if clauses:
                survivors.append("; ".join(clauses).rstrip() + ".")
        if survivors:
            kept.append(" ".join(survivors))
    return "\n\n".join(kept), removed


def _tidy_citations(text: str) -> str:
    """Each evidence id cited once per sentence.

    The writer often cites a passage after the figure and again after the cause in the same
    sentence ("took 5.0 bps off [ev_055], which the bank attributes to ... [ev_055]"), and
    adjacent groups ("[ev_045] [ev_021, ev_045]") repeat ids. The repeats go; the first
    citation of each id stays where it was.
    """
    out: list[str] = []
    for paragraph in text.split("\n\n"):
        sentences = _SENTENCE_END.split(paragraph.strip())
        tidied: list[str] = []
        for sentence in sentences:
            seen: set[str] = set()

            def keep(match: re.Match[str], seen: set[str] = seen) -> str:
                ids = [part.strip() for part in match.group(1).split(",")]
                fresh = [i for i in ids if i not in seen]
                seen.update(fresh)
                return f"[{', '.join(fresh)}]" if fresh else ""

            cleaned = _GROUP.sub(keep, sentence)
            cleaned = re.sub(r"\s+([.,;:])", r"\1", re.sub(r"[ \t]{2,}", " ", cleaned))
            tidied.append(cleaned.strip())
        out.append(" ".join(part for part in tidied if part))
    return "\n\n".join(out)


#: A bracketed group of evidence ids: [ev_017] or [ev_017, ev_018].
_GROUP = re.compile(r"\[(ev_\d+(?:\s*,\s*ev_\d+)*)\]")


def _cite_drivers(text: str, decomposition: Decomposition) -> str:
    """The text with a driver cited by its id ("[cti.income_growth]") cited by its evidence.

    The brief gives each driver an id so a cause can name it, and the writer sometimes
    cites that id where an evidence id belongs. Such a citation escaped the checker, which
    only reads evidence ids, and printed as a raw id in the memo. It's replaced by the
    driver's own evidence, which is what it meant.
    """
    evidence = {d.driver_id: d.evidence_ids for d in decomposition.drivers}

    def replace(match: re.Match[str]) -> str:
        ids = [part.strip() for part in match.group(1).split(",")]
        if not all(i in evidence for i in ids):
            return match.group(0)
        cited = list(dict.fromkeys(e for i in ids for e in evidence[i]))
        return f"[{', '.join(cited)}]" if cited else ""

    return re.sub(
        r"\[([a-z0-9_]+\.[a-z0-9_.]+(?:\s*,\s*[a-z0-9_]+\.[a-z0-9_.]+)*)\]", replace, text
    )


def _causes(
    answers: Any,
    decomposition: Decomposition,
    cards: list[EvidenceCard],
    supplied: set[str],
) -> dict[str, tuple[str, list[str]]]:
    """Each driver's stated cause, kept only if it cites the issuer's own words.

    A cause with no citation of a passage the issuer wrote is the writer's opinion, and one
    with a figure nobody supplied is arithmetic; both are dropped here, before the checker
    sees them.
    """
    worded = {card.evidence_id for card in cards if card.passage}
    known = {driver.driver_id for driver in decomposition.drivers}
    out: dict[str, tuple[str, list[str]]] = {}
    for item in answers:
        if not isinstance(item, dict):
            continue
        driver_id = str(item.get("driver") or "").strip()
        why = _cite_drivers(" ".join(str(item.get("why") or "").split()), decomposition)
        if driver_id not in known or not why or driver_id in out:
            continue
        why = _tidy_citations(why)
        cited = list(dict.fromkeys(_CITATION.findall(why)))
        if not cited or not set(cited) & worded or _SILENCE.search(why):
            continue
        if numbers_in(why) - supplied or len(why.split()) > CAUSE_WORD_LIMIT:
            continue
        out[driver_id] = (why, cited)
    return out


#: ``ResearchResult`` enforces it with a validator that raises.
NARRATIVE_WORD_LIMIT = 150


def _within_the_word_limit(narrative: str) -> tuple[str, str]:
    """The narrative, trimmed to whole sentences within the limit.

    The schema validator raises on a narrative that's too long, and nothing caught it: a
    model that wrote 160 good words brought the whole run down with a ``ValidationError``
    from ``build_result``, after every figure had been established and every call paid for.
    A step shouldn't be able to hand the renderer something the renderer will reject.

    It's trimmed at a sentence boundary, never mid-clause or mid-number: a narrative cut
    after "the margin fell 3" is worse than one sentence shorter. Trimming is reported,
    because a reader should know the memo was cut.
    """
    words = narrative.split()
    if len(words) <= NARRATIVE_WORD_LIMIT:
        return narrative, ""

    kept: list[str] = []
    for sentence in re.findall(r"[^.!?]*[.!?]+|[^.!?]+$", narrative):
        candidate = [*kept, sentence]
        if len(" ".join(candidate).split()) > NARRATIVE_WORD_LIMIT:
            break
        kept = candidate
    trimmed = "".join(kept).strip()
    if not trimmed:
        # A single sentence longer than the whole limit, so there's no clean place to cut.
        return "", (
            f"the narrative was discarded: {len(words)} words in one sentence, against a "
            f"limit of {NARRATIVE_WORD_LIMIT}. Cutting it would have truncated a claim."
        )
    return trimmed, (
        f"the narrative was trimmed from {len(words)} to {len(trimmed.split())} words at a "
        f"sentence boundary, against a limit of {NARRATIVE_WORD_LIMIT}"
    )


#: A driver at least this share of the largest one is material, and the memo names it. A
#: walk of seven one- and two-point steps shouldn't have every step appended to its memo.
_MATERIAL_SHARE: Final = Decimal("0.25")


def name_every_material_driver(
    narrative: str, decomposition: Decomposition, places: int = 0
) -> tuple[str, list[str]]:
    """The checked narrative, with a plain sentence for each material driver it doesn't name,
    and the labels of those it added.

    The checker takes out a sentence whose cause it can't support, and the figure goes with
    it: Westpac's 1H26 memo explained a A$97m rise without its A$420m of net interest income
    or its A$193m impairment charge, because both causes were stated on the other basis. A
    driver's figure is exact arithmetic over cited figures, so it's stated as it stands,
    with its evidence and no cause, as long as the narrative stays within its word limit.
    """
    if not narrative.strip():
        return narrative, []
    top = [
        driver
        for driver in decomposition.drivers
        if driver.parent_id is None and driver.contribution and driver.evidence_ids
    ]
    if not top:
        return narrative, []
    largest = max(abs(driver.contribution or Decimal(0)) for driver in top)
    cited = set(_CITATION_ID.findall(narrative))
    said = _plainly(narrative)
    named: list[str] = []
    for driver in sorted(top, key=lambda d: abs(d.contribution or Decimal(0)), reverse=True):
        contribution = driver.contribution or Decimal(0)
        if abs(contribution) < largest * _MATERIAL_SHARE or cited & set(driver.evidence_ids):
            continue
        # Named in words is named: "capital and other cost 3.0 bps" for "Capital & other".
        label = driver.label or driver.driver_id
        if any(name and name in said for name in (_plainly(label), _plainly(label.split(" (")[0]))):
            continue
        size = format_magnitude(contribution, decomposition.change_unit, places)
        moved = f"added {size}" if contribution > 0 else f"took {size} off"
        sentence = f"{label} {moved} [{', '.join(driver.evidence_ids[:3])}]."
        if len(f"{narrative} {sentence}".split()) > NARRATIVE_WORD_LIMIT:
            break
        narrative = f"{narrative.rstrip()} {sentence}"
        named.append(label)
    return narrative, named


def _plainly(text: str) -> str:
    """Text as compared for a name: lower case, "&" as "and", single spaces."""
    return " ".join(text.lower().replace("&", " and ").split())


def movement_line(
    spec: RequestSpec, decomposition: Decomposition, inputs: ResolvedInputs
) -> str | None:
    """The headline movement in words, as the writer is told it: "fell A$236m (2.3%), from
    A$10,072m to A$9,836m".

    The checker is told the same line. It used to see only the drivers' contributions, so
    CBA's FY24 memo lost a correct "fell A$236m (2.3%)" as a figure printed nowhere: the
    page says 2%, and the 2.3% the writer had been given was never shown to the checker.
    """
    metric = get_registry().metrics[spec.metric_id]
    pair = inputs.pair(spec.metric_id)
    if pair is None:
        return None
    unit = inputs.units.get(spec.metric_id, metric.unit)
    change_unit = decomposition.change_unit
    prior, current = pair
    change = build_change(prior, current, change_unit, show_pct=metric.show_pct)
    if decomposition.headline_change == 0:
        movement = "unchanged"
    else:
        verb = "rose" if decomposition.headline_change > 0 else "fell"
        size = format_magnitude(decomposition.headline_change, change_unit, amount_places(*pair))
        movement = f"{verb} {size}"
        if change.pct is not None:
            movement += f" ({abs(change.pct)}%)"
    return f"{movement}, from {format_value(prior, unit)} to {format_value(current, unit)}"


def _brief(
    spec: RequestSpec,
    decomposition: Decomposition,
    inputs: ResolvedInputs,
    cards: list[EvidenceCard],
) -> str:
    """What the writer is given: the movement, the drivers in words, the issuer's own words,
    and the figures, each with its citation.

    Every figure is formatted the way the report prints it. The brief used to pass raw
    values, so the writer saw a driver of "0.048744145930878809219515831 bps" and evidence
    of "1035226" with no unit, and either repeated the noise or rounded it, and a rounded
    figure isn't in the brief, so the guard threw the whole narrative away as invented.

    Direction is written in words ("took 3.0 bps off"), because a writer handed "-3.0 bps"
    writes "fell -3.0 bps" or "rose +A$313m". Shares are given only where every driver
    moves the same way: with drivers that offset, a driver is "320.1% of the movement",
    which is true and tells a reader nothing.
    """
    from ...output.schema import describe_basis

    registry = get_registry()
    metric = registry.metrics[spec.metric_id]
    unit = inputs.units.get(spec.metric_id, metric.unit)
    change_unit = decomposition.change_unit
    # The bank's own name for the measure. Westpac reports net profit, not cash earnings,
    # and a memo that calls its statutory figure "cash earnings" states the wrong basis.
    name = spec.headline_label or metric.display_name
    noun = name if name[:2].isupper() else name[:1].lower() + name[1:]
    bank = registry.banks.get(spec.bank_id)
    basis = describe_basis(inputs.bases.get(f"{spec.period}:{spec.metric_id}", Basis()))
    lines = [
        f"Bank: {bank.name if bank else spec.bank_id}",
        f"Metric: {name}"
        + (f" ({basis})" if basis else "")
        + (
            ""
            if name == metric.display_name
            else f", the bank's own measure of {metric.display_name.lower()}"
        ),
        f"Comparison: {spec.period} against {spec.comparison_period}, the same period a year earlier",
    ]
    pair = inputs.pair(spec.metric_id)
    places = amount_places(*pair) if pair is not None else 0
    movement = movement_line(spec, decomposition, inputs)
    if pair is not None and movement is not None:
        prior, current = pair
        lines += [
            f"{spec.comparison_period}: {format_value(prior, unit)} "
            f"[{', '.join(inputs.citations_for(spec.metric_id, spec.comparison_period))}]",
            f"{spec.period}: {format_value(current, unit)} "
            f"[{', '.join(inputs.citations_for(spec.metric_id, spec.period))}]",
            f"Movement: {movement}",
        ]
    source = {
        "A": "the issuer's own walk",
        "B": "the accounting identity, computed exactly from the issuer's published figures",
    }.get(decomposition.path, "nowhere: no attribution was possible")
    lines += [f"Drivers come from: {source}", ""]

    top = [d for d in decomposition.drivers if d.parent_id is None]
    shares_mean_something = bool(top) and all(
        d.contribution is not None
        and d.contribution != 0
        and (d.contribution > 0) == (decomposition.headline_change > 0)
        for d in top
    )
    if top:
        lines.append(
            "Drivers, largest first, each with its id. Contributions are exact. A computed one "
            "cites the figures it was computed from:"
        )
        for driver in sorted(
            top,
            key=lambda d: abs(d.contribution) if d.contribution is not None else 0,
            reverse=True,
        ):
            lines.append(
                "- "
                + _driver_line(
                    driver, decomposition, change_unit, noun, shares_mean_something, places
                )
            )
            for child in decomposition.drivers:
                if child.parent_id == driver.driver_id:
                    lines.append(
                        "  - of which "
                        + _driver_line(child, decomposition, change_unit, noun, False, places)
                    )
        remainder = format_magnitude(decomposition.residual, change_unit, places)
        within = abs(decomposition.residual) <= decomposition.tolerance
        lines.append(
            f"Left unexplained: {remainder}, "
            + ("inside the rounding tolerance" if within else "beyond the rounding tolerance")
        )
    else:
        lines.append("Drivers: none were established")
    if decomposition.notes:
        lines += [
            "",
            "Notes on how the drivers were derived (for your understanding; the memo shows "
            "them to the reader separately):",
            *(f"- {note}" for note in decomposition.notes),
        ]
    # Only the disagreements that bear on this metric. Every conflict in the run used to be
    # listed, so a capital-ratio memo spent a sentence on a disagreement about earnings that
    # no figure in it used.
    relevant = _relevant_components(spec, decomposition)
    conflicts = [c for c in inputs.conflicts if c.component in relevant]
    if conflicts:
        lines += [
            "",
            "Where the issuer's documents disagree:",
            *(
                f"- {conflict.component}: {conflict.classification} ({conflict.rule_applied})"
                for conflict in conflicts
            ),
        ]
    worded = [card for card in cards if card.passage]
    if worded:
        lines += [
            "",
            "The issuer's own words, from its results documents for this period, against the "
            "same period a year earlier. Cite a passage's id when you give a cause it states:",
        ]
        for card in worded:
            lines.append(f'- [{card.evidence_id}] p.{card.source.page}: "{card.passage}"')
    else:
        lines += [
            "",
            "The issuer's own words: none were found for this movement against the same "
            "period a year earlier. Give no cause, and don't comment on the absence: what "
            "you see is a selection of the bank's words, not all of them.",
        ]
    lines += ["", "Figures:"]
    # The headline's cards and the drivers' own cards first, so the ones the narrative
    # should cite are never cut off by the cap.
    wanted = {
        *(i for driver in decomposition.drivers for i in driver.evidence_ids),
        *(c.evidence_id for c in cards if c.component == spec.metric_id),
    }
    figures = [card for card in cards if card.value is not None and not card.passage]
    ordered_cards = [c for c in figures if c.evidence_id in wanted] + [
        c for c in figures if c.evidence_id not in wanted
    ]
    for card in ordered_cards[:40]:
        lines.append(
            f"- [{card.evidence_id}] {card.component}, {card.period}: "
            f"{format_value(card.value, card.unit)} ({card.source.cite()})"
        )
    return "\n".join(lines)


def _driver_line(
    driver: DriverContribution,
    decomposition: Decomposition,
    change_unit: str,
    noun: str,
    with_share: bool,
    places: int = 0,
) -> str:
    """One driver in words: its label, id, what it did to the metric, and its citations."""
    value = driver.contribution
    if value is None or value == 0:
        effect = "no effect"
    elif value > 0:
        effect = f"added {format_magnitude(value, change_unit, places)} to the {noun}"
    else:
        effect = f"took {format_magnitude(value, change_unit, places)} off the {noun}"
    share = ""
    if with_share:
        shown = share_of_change(value, decomposition.headline_change, decomposition.tolerance)
        if shown is not None and 0 < shown <= 100:
            share = f", about {format_share(shown)} of the movement"
    how = "computed from" if driver.method == "computed" else "evidence"
    cited = ", ".join(driver.evidence_ids) or "no citation"
    return f"{driver.label or driver.driver_id} (id {driver.driver_id}): {effect}{share} [{how} {cited}]"


#: An evidence id as the narrative cites it: ev_017, in brackets or parentheses.
_CITATION = re.compile(r"\bev_\d{3,}\b")
#: A sentence boundary: a full stop, question or exclamation mark followed by space. A
#: decimal point never has a space after it, so 1.89% stays whole.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
#: Citations a writer put after the full stop, which belong to the sentence before.
_LEADING_CITATIONS = re.compile(r"^((?:\s*[\[(]\s*ev_\d{3,}(?:\s*,\s*ev_\d{3,})*\s*[\])])+)\s*")
#: How much of each cited page the verifier sees.
_PAGE_CHARS = 3000
_MAX_PAGES = 6


def cited_sentences(narrative: str) -> list[tuple[str, list[str]]]:
    """The narrative split into sentences, each with the evidence ids it cites.

    A citation written after the full stop ("...fell 3bps. [ev_001]") goes with the
    sentence before it, which is the one it supports.
    """
    pieces = [piece for piece in _SENTENCE_END.split(narrative.strip()) if piece.strip()]
    sentences: list[str] = []
    for piece in pieces:
        moved = _LEADING_CITATIONS.match(piece)
        if moved and sentences:
            sentences[-1] = f"{sentences[-1]} {moved.group(1).strip()}"
            piece = piece[moved.end() :]
        if piece.strip():
            sentences.append(piece.strip())
    return [(sentence, list(dict.fromkeys(_CITATION.findall(sentence)))) for sentence in sentences]


def _relevant_components(spec: RequestSpec, decomposition: Decomposition) -> set[str]:
    """The metric, its identity's components, and whatever its drivers were computed from.

    For a headline struck before tax the measure is profit before tax, and the after-tax
    figure, tax and what follows it aren't part of this answer.
    """
    from ...decomposition.identities import identity_for

    identity = identity_for(spec.metric_id, pre_tax=spec.pre_tax)
    relevant: set[str] = {spec.measure}
    if identity is not None:
        relevant.update(identity.components)
        relevant.update(identity.optional_components)
    for driver in decomposition.drivers:
        relevant.update(driver.from_components)
    return relevant


@dataclass
class ClaimVerdict:
    claim: str
    evidence_ids: list[str]
    verdict: str = "unsupported"
    reason: str = ""
    driver_id: str | None = None
    """Set when the claim is a driver's stated cause rather than a sentence of the narrative."""


@dataclass
class NarrativeCheck:
    """What the verifier made of the memo, and the narrative and causes that survived it."""

    narrative: str
    verdicts: list[ClaimVerdict] = field(default_factory=list)
    skipped: str = ""
    causes: dict[str, tuple[str, list[str]]] = field(default_factory=dict)
    """The drivers' stated causes that survived, by driver id."""

    @property
    def unsupported(self) -> list[ClaimVerdict]:
        return [v for v in self.verdicts if v.verdict == "unsupported" and v.driver_id is None]

    @property
    def partial(self) -> list[ClaimVerdict]:
        return [v for v in self.verdicts if v.verdict == "partial" and v.driver_id is None]

    @property
    def causes_removed(self) -> list[ClaimVerdict]:
        return [v for v in self.verdicts if v.driver_id is not None and v.verdict != "supported"]


def computed_lines(
    decomposition: Decomposition, places: int = 0, movement: str | None = None
) -> list[str]:
    """Each driver's contribution, as the checker is told about it.

    The checker used to see only the cards, so a sentence stating a computed contribution
    ("operating expenses took A$759m off cash earnings") was judged against pages that
    can't print it, called "a derived figure appearing nowhere on the cited pages", and
    removed. Five of the eight sentences removed across the Opus 5 runs were exactly that.
    """
    out: list[str] = []
    for driver in decomposition.drivers:
        if driver.contribution is None:
            continue
        how = (
            f"computed exactly from {', '.join(driver.evidence_ids) or 'no cited figure'}"
            if driver.method == "computed"
            else f"from the issuer's own walk ({', '.join(driver.evidence_ids) or 'uncited'})"
        )
        out.append(
            f"{driver.label or driver.driver_id}: "
            f"{format_change(driver.contribution, decomposition.change_unit, places)}, {how}"
        )
    if movement is not None:
        out.append(f"The movement itself: {movement}, computed from the two published figures")
    return out


def check_narrative(
    narrative: str,
    cards: list[EvidenceCard],
    pages: dict[tuple[str, int], str],
    client: LLMClient,
    *,
    budget: Budget | None = None,
    model: str = "claude-fable-5-1",
    effort: str = "medium",
    tracer: Tracer | None = None,
    prompts: PromptLibrary | None = None,
    causes: dict[str, tuple[str, list[str]]] | None = None,
    labels: dict[str, str] | None = None,
    computed: list[str] | None = None,
) -> NarrativeCheck:
    """Each cited sentence, and each driver's stated cause, judged against what it cites.

    One call for every claim, which is what the step's budget allows (one call and
    one recheck). Each claim is still judged on its own: the prompt says so, and the verdicts
    come back one per claim. A sentence with no citation makes no claim about the evidence
    and isn't sent. An unsupported sentence is taken out of the narrative, and an unsupported
    cause is dropped. If the call can't be made the narrative is kept as written and the
    result says it wasn't checked; a check that didn't happen is never reported as a pass.
    """
    causes = dict(causes or {})
    labels = labels or {}
    check = NarrativeCheck(narrative=narrative, causes=dict(causes))
    sentences = cited_sentences(narrative)
    claims = [(sentence, ids) for sentence, ids in sentences if ids]
    by_id = {card.evidence_id: card for card in cards}

    # A claim citing evidence that doesn't exist is unsupported by definition, and needs no
    # model to say so.
    invented = [(sentence, ids) for sentence, ids in claims if any(i not in by_id for i in ids)]
    for sentence, ids in invented:
        check.verdicts.append(
            ClaimVerdict(
                claim=sentence,
                evidence_ids=ids,
                verdict="unsupported",
                reason="it cites evidence that does not exist",
            )
        )
    for driver_id, (why, ids) in list(causes.items()):
        if any(i not in by_id for i in ids):
            check.verdicts.append(
                ClaimVerdict(
                    claim=why,
                    evidence_ids=ids,
                    verdict="unsupported",
                    reason="it cites evidence that does not exist",
                    driver_id=driver_id,
                )
            )
            del causes[driver_id]
    check.causes = dict(causes)
    claims = [claim for claim in claims if claim not in invented]
    check.narrative = _without(narrative, sentences, {v.claim for v in check.unsupported})
    if not claims and not causes:
        if not check.verdicts:
            check.skipped = "no sentence cites any evidence"
        return check
    if not client.available:
        check.skipped = "no model configured"
        return check
    if budget is not None and not budget.may_call("verify"):
        check.skipped = "budget"
        return check

    numbered: list[tuple[str, list[str], str | None]] = [(s, ids, None) for s, ids in claims]
    numbered += [
        (f"{labels.get(driver_id, driver_id)}: {why}", ids, driver_id)
        for driver_id, (why, ids) in causes.items()
    ]
    cited_ids = list(dict.fromkeys(i for _, ids, _ in numbered for i in ids if i in by_id))
    evidence_lines = []
    page_keys: list[tuple[str, int]] = []
    for card in (by_id[i] for i in cited_ids):
        where = f"({card.source.doc_id} p.{card.source.page})"
        if card.passage:
            evidence_lines.append(
                f'[{card.evidence_id}] the issuer writes {where}: "{card.passage}"'
            )
        else:
            value = format_value(card.value, card.unit) if card.value is not None else "no figure"
            how = (
                "read from a chart image and checked against the published endpoints"
                if card.method == "chart"
                else f'printed as "{card.quote}"'
                if card.quote
                else card.statement
            )
            evidence_lines.append(
                f"[{card.evidence_id}] {card.component}, {card.period}: {value}, {how} {where}"
            )
        key = (card.source.doc_id, card.source.page)
        if key not in page_keys and key in pages and not card.passage:
            page_keys.append(key)
    page_blocks = [
        f'<document_content doc="{doc}" page="{page}">\n{pages[(doc, page)][:_PAGE_CHARS]}\n'
        "</document_content>"
        for doc, page in page_keys[:_MAX_PAGES]
    ]
    claim_lines = [f"{n}. {text}" for n, (text, _, _) in enumerate(numbered, start=1)]

    library = prompts or default_library()
    prompt = library.get("verify")
    request = LLMRequest(
        step="verify",
        model=model,
        effort=effort,
        system=prompt.text,
        messages=(
            {
                "role": "user",
                "content": (
                    "Claims:\n"
                    + "\n".join(claim_lines)
                    + "\n\nEvidence cited:\n"
                    + ("\n".join(evidence_lines) or "none of the cited ids exist")
                    + "\n\nComputed by the system (exact arithmetic over the evidence named; "
                    "never printed on a page):\n"
                    + ("\n".join(f"- {line}" for line in computed or ()) or "nothing")
                    + "\n\nPages cited:\n"
                    + ("\n\n".join(page_blocks) or "none needed")
                ),
            },
        ),
        output_schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["verdicts"],
            "properties": {
                "verdicts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["claim", "verdict", "reason"],
                        "properties": {
                            "claim": {"type": "integer"},
                            "verdict": {
                                "type": "string",
                                "enum": ["supported", "partial", "unsupported"],
                            },
                            "reason": {"type": "string"},
                        },
                    },
                }
            },
        },
    )
    try:
        response = client.complete(request)
    except LLMUnavailable as error:
        if isinstance(error, ModelUnreachable):  # the run stops; see llm.client
            raise
        check.skipped = f"the verifier was unavailable: {error}"
        return check
    if budget is not None:
        budget.record("verify", model, response.usage, cached=response.cached)
    if tracer is not None:
        tracer.model_call(request, response, prompt=prompt)

    if response.stop_reason == "max_tokens" or not isinstance(response.structured, dict):
        # No verdicts is not a verdict. Reading a cut-off answer as "every claim
        # unsupported" would delete a memo nobody had judged; it's shown as written, and
        # the run says it wasn't checked.
        check.skipped = (
            "the check was cut off at its output limit"
            if response.stop_reason == "max_tokens"
            else "the check returned no verdicts"
        )
        return check
    answers: dict[int, dict[str, Any]] = {}
    for item in response.structured.get("verdicts") or ():
        if isinstance(item, dict) and isinstance(item.get("claim"), int):
            answers.setdefault(item["claim"], item)
    for n, (text, ids, about) in enumerate(numbered, start=1):
        answer = answers.get(n, {})
        check.verdicts.append(
            ClaimVerdict(
                claim=text if about is None else causes[about][0],
                evidence_ids=ids,
                # Unsupported by default: a verifier that said nothing about a claim hasn't
                # approved it.
                verdict=str(answer.get("verdict") or "unsupported"),
                reason=str(answer.get("reason") or "the verifier returned no verdict"),
                driver_id=about,
            )
        )
    # A partly supported sentence goes too. In practice "partly" is a real figure with a
    # cause attached that the evidence doesn't state, which is the failure this check
    # exists to stop; keeping the sentence with a caveat elsewhere still puts it in front of
    # the reader as the memo's own claim.
    check.narrative = _without(
        narrative, sentences, {v.claim for v in (*check.unsupported, *check.partial)}
    )
    check.causes = {
        verdict.driver_id: causes[verdict.driver_id]
        for verdict in check.verdicts
        if verdict.driver_id is not None
        and verdict.verdict == "supported"
        and verdict.driver_id in causes
    }
    return check


def _without(narrative: str, sentences: list[tuple[str, list[str]]], removed: set[str]) -> str:
    """The narrative with the removed sentences cut out, its paragraphs left as they were."""
    if not removed:
        return narrative
    kept = narrative
    for sentence in removed:
        if sentence in kept:
            kept = kept.replace(sentence, "", 1)
        else:  # whitespace inside it differed; rebuild from the sentences instead
            return " ".join(s for s, _ in sentences if s not in removed)
    paragraphs = [re.sub(r"[ \t]+", " ", block).strip() for block in kept.split("\n\n")]
    return "\n\n".join(block for block in paragraphs if block)
