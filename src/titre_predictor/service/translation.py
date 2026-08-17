"""Turning a request payload into an :class:`ExperimentRun`.

This is the boundary. Above it, data is untrusted JSON that arrived over HTTP;
below it, an ``ExperimentRun`` whose invariants the whole model relies on. Keeping
the crossing in one function means there is one place to look for "what does the
service accept", and no path into the model that skips it.

Reject or repair?
-----------------
The rule applied here: **repair only what can be repaired exactly; reject anything
that would require a guess.**

Repairing is legitimate for the ``W:`` control profiles, because they are not
independent data -- they are exact step functions of the ``Z:`` scalars, verified
to reconstruct to machine precision on all 1290 rows of both supplied files. A
caller who sends ``Z:tempStart``, ``Z:tempEnd`` and ``Z:tempShift`` has already
determined ``W:temp`` completely; asking for it again is asking them to repeat
themselves.

Everything else is rejected. Interpolating a missing observation, or padding a
short series, would let the service return a confident number computed partly from
invented data -- and the caller would have no way to tell. A 400 naming the problem
is more useful than a plausible answer.

Which inputs are required is a property of the model, not of this module
-----------------------------------------------------------------------
The viable-cell and lysed-cell series are always needed: they are what the growth
term and the dead pool are built from. Beyond that, the requirement is whatever the
loaded model's mechanisms read, taken from
:attr:`titre_predictor.kinetics.Mechanism.required_series`.

So the shipped model, whose factor is Monod in glutamine and glucose, requires
``X:Gln`` and ``X:Glc`` and does **not** require temperature or pH. If a future
model includes a temperature term, this service starts requiring ``W:temp`` with no
change here. Hardcoding the list would have frozen it against one fitted model.
"""

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from titre_predictor import kinetics
from titre_predictor.data import controls, schema
from titre_predictor.domain import ExperimentRun, InvalidExperimentRunError
from titre_predictor.model import LuedekingPiretModel
from titre_predictor.service.dto import PredictRequest

# Needed to compute the growth term and the dead pool, whatever the mechanism set.
ALWAYS_REQUIRED_OBSERVATIONS = (
    schema.OBSERVATION_VIABLE_CELL_DENSITY,
    schema.OBSERVATION_LYSED_CELLS,
)

DEFAULT_EXPERIMENT_ID = "request"


class PayloadError(ValueError):
    """The payload cannot be turned into a run. Becomes an HTTP 400.

    Distinct from :class:`InvalidExperimentRunError` only in where it is raised:
    this covers what is missing or misplaced, that covers what is internally
    inconsistent. Both are the caller's problem and both are reported as such.
    """


def required_observations(model: LuedekingPiretModel) -> tuple[str, ...]:
    """``X:`` series this model needs, in a stable order."""
    required = set(ALWAYS_REQUIRED_OBSERVATIONS)
    for mechanism in kinetics.resolve(model.mechanisms):
        required.update(
            name for name in mechanism.required_series if name.startswith(schema.OBSERVATION_PREFIX)
        )
    return tuple(sorted(required))


def required_control_profiles(model: LuedekingPiretModel) -> tuple[str, ...]:
    """``W:`` series this model needs, in a stable order.

    Empty for the shipped model, whose mechanisms read only observations.
    """
    required = {
        name
        for mechanism in kinetics.resolve(model.mechanisms)
        for name in mechanism.required_series
        if name.startswith(schema.CONTROL_PROFILE_PREFIX)
    }
    return tuple(sorted(required))


def _split_by_prefix(
    values: Mapping[str, Sequence[float]],
) -> tuple[dict[str, float], dict[str, NDArray[np.float64]], dict[str, NDArray[np.float64]]]:
    """Partition the payload into design scalars, control profiles and observations."""
    design: dict[str, float] = {}
    profiles: dict[str, NDArray[np.float64]] = {}
    observations: dict[str, NDArray[np.float64]] = {}

    for name, series in values.items():
        if name.startswith(schema.DESIGN_SCALAR_PREFIX):
            design[name] = float(series[0])
        elif name.startswith(schema.CONTROL_PROFILE_PREFIX):
            profiles[name] = np.asarray(series, dtype=np.float64)
        else:
            observations[name] = np.asarray(series, dtype=np.float64)

    return design, profiles, observations


def _missing(required: Sequence[str], supplied: Mapping[str, object]) -> list[str]:
    return [name for name in required if name not in supplied]


def to_experiment_run(
    request: PredictRequest,
    model: LuedekingPiretModel,
) -> ExperimentRun:
    """Build the run this model can predict from, or explain why it cannot.

    Args:
        request: the validated payload. Shape has already been checked; this
            checks that the *contents* are sufficient for ``model``.
        model: the loaded model, which determines what is required.

    Returns:
        A run with exactly the series the model reads, plus any others supplied.

    Raises:
        PayloadError: if a required series is absent, or if the control profiles
            can be neither supplied nor reconstructed.
        InvalidExperimentRunError: if the series are internally inconsistent --
            wrong lengths, non-finite values, timestamps not increasing. Raised by
            :class:`ExperimentRun` itself, so the service and the training pipeline
            enforce identical invariants.
    """
    experiment_id = request.experiment_id or DEFAULT_EXPERIMENT_ID
    timestamps = np.asarray(request.timestamps, dtype=np.float64)
    design, profiles, observations = _split_by_prefix(request.values)

    absent = _missing(required_observations(model), observations)
    if absent:
        raise PayloadError(
            f"{experiment_id}: missing observation(s) {absent}, which this model needs. "
            f"It reads {list(required_observations(model))}."
        )

    needed_profiles = required_control_profiles(model)
    if _missing(needed_profiles, profiles):
        # Exact reconstruction from the Z: scalars, not an approximation -- the
        # profiles are step functions of them. Only attempted when the model
        # actually needs a profile that was not supplied.
        try:
            profiles = {**controls.reconstruct_control_profiles(design, timestamps), **profiles}
        except KeyError as exception:
            raise PayloadError(
                f"{experiment_id}: this model needs {list(needed_profiles)}, which were "
                f"not supplied and cannot be reconstructed because the design scalar "
                f"{exception} is also absent."
            ) from exception

        still_absent = _missing(needed_profiles, profiles)
        if still_absent:  # pragma: no cover - reconstruction covers all four profiles
            raise PayloadError(f"{experiment_id}: missing control profile(s) {still_absent}.")

    return ExperimentRun(
        experiment_id=experiment_id,
        timestamps=timestamps,
        design_scalars=design,
        control_profiles=profiles,
        observations=observations,
    )


__all__ = [
    "InvalidExperimentRunError",
    "PayloadError",
    "required_control_profiles",
    "required_observations",
    "to_experiment_run",
]
