"""A model's reading of a walk row the synonym index didn't recognise."""

from __future__ import annotations

from dataclasses import dataclass, field

from ...decomposition.taxonomy import driver_label, map_labels, walk_driver_ids
from ...llm.budget import Budget
from ...llm.client import LLMClient, LLMRequest, LLMUnavailable, ModelUnreachable
from ...llm.tracing import PromptLibrary, Tracer, default_library

__all__ = ["MappingOutcome", "map_labels_with_model"]


@dataclass
class MappingOutcome:
    """Which rows were mapped, by what, and which were left alone."""

    mapping: dict[str, str] = field(default_factory=dict)
    unmapped: list[str] = field(default_factory=list)
    from_model: dict[str, str] = field(default_factory=dict)
    refused: dict[str, str] = field(default_factory=dict)
    """Rows the model answered for and whose answer was discarded, with why."""
    notes: list[str] = field(default_factory=list)


def map_labels_with_model(
    metric_id: str,
    labels: list[str],
    client: LLMClient,
    *,
    budget: Budget | None = None,
    # Overridden from config/models.yaml through ``Registry.route("taxonomy_classify")``. A
    # default that differs from the routing is how that config ended up unused for the whole
    # build.
    model: str = "claude-opus-5",
    effort: str = "low",
    tracer: Tracer | None = None,
    prompts: PromptLibrary | None = None,
    context: list[str] | None = None,
) -> MappingOutcome:
    """Map every row label, deterministically first and with one model call for the rest.

    One call for all the leftover rows rather than one each, with every bar on the walk as
    context. A bar's meaning often depends on its neighbours: a bare "RWA" next to separate
    "IRRBB" and "Operational risk" bars is the rest of risk-weighted assets, not all of it.
    """
    mapping, unmapped = map_labels(metric_id, labels)
    outcome = MappingOutcome(mapping=dict(mapping), unmapped=list(unmapped))
    if not unmapped:
        return outcome
    if not client.available:
        outcome.notes.append(
            f"{len(unmapped)} row(s) left unmapped: no model configured. They are reported "
            "as 'other', which reconciles and names no cause."
        )
        return outcome
    if budget is not None and not budget.may_call("taxonomy_classify"):
        outcome.notes.append("taxonomy mapping skipped on its budget")
        return outcome

    allowed = walk_driver_ids(metric_id)
    library = prompts or default_library()
    prompt = library.get("taxonomy_classify")
    menu = "\n".join(
        f"- {driver_id}: {driver_label(metric_id, driver_id)}" for driver_id in allowed
    )
    request = LLMRequest(
        step="taxonomy_classify",
        model=model,
        effort=effort,
        system=prompt.text,
        messages=(
            {
                "role": "user",
                "content": (
                    f"Metric: {metric_id}\n"
                    f"Every bar on the walk, in order: {' | '.join(context or labels)}\n\n"
                    "Rows to map:\n"
                    + "\n".join(f"- {label}" for label in unmapped)
                    + f"\n\nCanonical ids, with what each covers:\n{menu}\n\n"
                    "Return one entry for each row to map: the one id it corresponds to, "
                    "or null."
                ),
            },
        ),
        output_schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["mappings"],
            "properties": {
                "mappings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["label", "driver_id"],
                        "properties": {
                            "label": {"type": "string"},
                            "driver_id": {"type": ["string", "null"]},
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
        outcome.notes.append(f"taxonomy mapping unavailable: {error}")
        return outcome
    if budget is not None:
        budget.record("taxonomy_classify", model, response.usage, cached=response.cached)
    if tracer is not None:
        tracer.model_call(request, response, prompt=prompt)

    answers = (response.structured or {}).get("mappings") or []
    for answer in answers if isinstance(answers, list) else ():
        if not isinstance(answer, dict):
            continue
        label = str(answer.get("label") or "").strip()
        proposed = answer.get("driver_id")
        if label not in outcome.unmapped or not proposed:
            continue  # null is the expected answer for a row outside the taxonomy
        if proposed not in allowed:
            # A plausible id that does not exist would roll up under a parent that was
            # never defined, and the taxonomy test would not catch it because the row
            # came from a document. An aggregate node, ``.other`` or ``.residual`` is
            # refused for the same reason: none of them is a bar.
            outcome.refused[label] = f"{proposed!r} is not a walk driver of {metric_id}"
            continue
        outcome.mapping[label] = proposed
        outcome.from_model[label] = proposed
        outcome.unmapped.remove(label)

    if outcome.from_model:
        outcome.notes.append(
            f"{len(outcome.from_model)} walk row(s) mapped by model rather than by the "
            "synonym index; each is recorded against the row it mapped"
        )
    if outcome.refused:
        outcome.notes.append(
            f"{len(outcome.refused)} proposed mapping(s) refused for naming an id the "
            "metric does not declare"
        )
    return outcome
