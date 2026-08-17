"""Whether a request asks the model about conditions it was fitted on.

Why the service reports this at all
------------------------------------
Extrapolation is this model's entire subject. Every test run lasts 14 days while
only 10 of 100 training runs reach that horizon, and for eight of the ten held-out
runs the dominant extensive quantity lies *above the whole training range*. A
service that answers such a request with a bare number, identical in appearance to
one it is confident about, withholds the thing the user most needs to know.

It is also the honest continuation of what the model documents about itself. The
kinetic structure was chosen precisely because it extrapolates more gracefully than
a data-driven fit -- the benchmarks lose by 757 RMSE on exactly this shift -- but
"more gracefully" is not "reliably", and the boundary is knowable.

Why it warns rather than refuses
---------------------------------
Refusing would be defensible for a model asked far outside its range, and is the
wrong choice here for a concrete reason: **the task's own test set is mostly out of
range.** A service that rejected runs above the training maximum for ``cell_days``
would refuse most of the runs it exists to predict. The warning carries the
information without making the service useless.

What is compared
----------------
The quantities recorded by :func:`titre_predictor.features.applicability_ranges` at
training time: ``cell_days`` and ``duration_days``, which are structural, and the
observed span of each measured series, which is what the mechanism constants were
estimated against.

Only series the **loaded model actually reads** are checked. Reporting that ammonia
is out of range would be noise when no mechanism consumes it.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from titre_predictor import features
from titre_predictor.domain import ExperimentRun
from titre_predictor.model import LuedekingPiretModel
from titre_predictor.service.translation import (
    required_control_profiles,
    required_observations,
)

# Structural quantities, checked whatever the mechanism set.
STRUCTURAL_QUANTITIES = ("cell_days", "duration_days")


@dataclass(frozen=True)
class Exceedance:
    """One quantity outside the range the model was fitted on.

    Args:
        quantity: what was measured, e.g. ``"cell_days"`` or ``"X:Glc"``.
        value: the value from the request that falls outside.
        training_minimum: lowest value seen in training.
        training_maximum: highest value seen in training.
    """

    quantity: str
    value: float
    training_minimum: float
    training_maximum: float

    @property
    def direction(self) -> str:
        return "above" if self.value > self.training_maximum else "below"

    def describe(self) -> str:
        return (
            f"{self.quantity} is {self.value:.4g}, {self.direction} the training range "
            f"[{self.training_minimum:.4g}, {self.training_maximum:.4g}]"
        )


def _exceedance(
    quantity: str,
    value: float,
    span: Sequence[float],
) -> Exceedance | None:
    low, high = float(span[0]), float(span[1])
    if low <= value <= high:
        return None
    return Exceedance(quantity=quantity, value=value, training_minimum=low, training_maximum=high)


def assess(
    run: ExperimentRun,
    model: LuedekingPiretModel,
    training_ranges: Mapping[str, Sequence[float]] | None,
) -> tuple[Exceedance, ...]:
    """Which quantities in ``run`` lie outside the model's fitted range.

    Args:
        run: the translated request.
        model: the loaded model, which determines which series are read.
        training_ranges: from the artefact. **May be absent**: an artefact produced
            before ranges were recorded is still a perfectly good model, and
            refusing to serve it would be a worse failure than serving it without
            a warning. Absent ranges therefore yield no exceedances, and the
            service says so rather than implying the run was checked and passed.

    Returns:
        The exceedances, in a stable order. Empty when everything is inside range,
        or when no ranges are available to compare against.
    """
    if not training_ranges:
        return ()

    found: list[Exceedance] = []

    quantities = features.run_quantities(run)
    measured: dict[str, float] = {
        "cell_days": quantities.cell_days,
        "duration_days": run.duration_days,
    }
    for name in STRUCTURAL_QUANTITIES:
        span = training_ranges.get(name)
        if span is not None:
            exceedance = _exceedance(name, measured[name], span)
            if exceedance is not None:
                found.append(exceedance)

    # Only the series this model reads: a warning about a variable no mechanism
    # consumes would be noise the caller cannot act on.
    consumed = (*required_observations(model), *required_control_profiles(model))
    for name in consumed:
        span = training_ranges.get(name)
        if span is None:
            continue
        series = quantities.state.get(name)
        if series is None or series.size == 0:
            continue
        for value in (float(series.min()), float(series.max())):
            exceedance = _exceedance(name, value, span)
            if exceedance is not None:
                found.append(exceedance)
                break  # one report per series is enough to act on

    return tuple(found)
