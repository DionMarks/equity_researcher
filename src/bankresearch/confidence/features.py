"""What a confidence score is computed from: the evidence behind each figure and driver, and
the named problems that cap it."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Final, Literal

__all__ = [
    "BY_IDENTITY",
    "ConfidenceFeatures",
    "DriverEvidence",
    "FigureEvidence",
    "ReadFrom",
    "VerifierVerdict",
]

VerifierVerdict = Literal["supported", "partial", "unsupported"]
ReadFrom = Literal["table", "text", "chart", "computed"]

#: How most figures are confirmed: an identity rebuilt them from the figures they are made of.
BY_IDENTITY: Final = "an accounting identity rebuilt them from their parts"


@dataclass(frozen=True, slots=True)
class FigureEvidence:
    """What stands behind a set of printed figures, counted off the store: the two headline
    figures, or the figures a driver rests on (ADR-069)."""

    figures: int = 2
    """How many printed figures were counted: one per period for a headline, a driver's
    inputs over both periods, or a walk step as printed."""
    primary: int = 2
    """How many are printed in a primary filing (the results document, the annual report or
    the Pillar 3 report), not only in a presentation or a release."""
    corroborated: int = 0
    """How many another filing prints identically."""
    confirmed: int = 0
    """How many a check against other printed figures confirmed."""
    confirmed_by: str = BY_IDENTITY
    model_matched: bool = False
    """A row was named by a model, for a bank nobody configured, rather than by a label
    checked against the bank's own page when it was configured."""
    restated: bool = False
    restatement_footnoted: bool = True
    rounding_share: Decimal = Decimal(0)
    """Half the last printed digit of each figure, added together, as a share of the
    movement being scored: how much of it the printing alone could account for."""


@dataclass(frozen=True, slots=True)
class DriverEvidence:
    """What stands behind one driver's number: the printed figures it rests on, and how the
    number came from them (ADR-072)."""

    figures: FigureEvidence = field(default_factory=FigureEvidence)
    read_from: ReadFrom = "computed"
    """A printed cell of the bank's walk (``table``), its words read by a model (``text``),
    its chart read by a model (``chart``), or exact arithmetic over figures (``computed``)."""
    checksum_ok: bool | None = None
    """For a chart: whether its bars added up to the published endpoints. ``None`` means no
    check was made, which counts as a failed one."""
    chained: bool = False
    """Joined from two half-year walks at the balance between them."""


@dataclass(frozen=True, slots=True)
class ConfidenceFeatures:
    """The named problems a claim has. Each takes from its score or caps it."""

    path_b_residual: Decimal | None = None
    path_b_tolerance: Decimal | None = None
    paths_disagree: bool = False
    """Paths A and B differ on this driver by more than twice the tolerance."""
    uncited: bool = False
    """No evidence card supports this driver's figure."""
    indicative_split: bool = False
    """A sub-split whose own identity rebuilds its parent only approximately."""
    inputs_aggregated: bool = False
    """A total standing in for a split the bank doesn't give (CET1 capital as one figure),
    so it says what moved but not why."""
    unexplained_conflict: bool = False
    conflict_flips_sign: bool = False
    alternative_basis_used: bool = False
    definition_break: bool = False
    verifier: VerifierVerdict | None = None
    input_failed_validation: bool = False
    notes: tuple[str, ...] = field(default=())

    @property
    def residual_multiples(self) -> Decimal:
        """How many tolerances the Path B residual exceeded, zero if within."""
        if self.path_b_residual is None or self.path_b_tolerance is None:
            return Decimal(0)
        if self.path_b_tolerance == 0:
            return Decimal(0)
        over = abs(self.path_b_residual) / self.path_b_tolerance
        return over if over > 1 else Decimal(0)
