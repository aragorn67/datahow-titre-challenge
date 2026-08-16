"""The core domain object: one bioprocess experiment.

``ExperimentRun`` deliberately mirrors the shape of the ``/predict`` request body in
``inference_server_spec.yml`` — a timestamp array plus prefixed variable arrays. The
inference service therefore deserialises straight into this object, and training and
serving share one representation of an experiment rather than two that can drift apart.

Validation lives in ``__post_init__`` so that an invalid run cannot be constructed at
all: the same length and ordering rules protect the training pipeline and the API.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


class InvalidExperimentRunError(ValueError):
    """Raised when the arrays describing an experiment are mutually inconsistent.

    The inference layer maps this to HTTP 400: it always indicates a malformed
    request rather than a server fault.
    """


@dataclass(frozen=True, eq=False)
class ExperimentRun:
    """One bioprocess experiment: its design, its controls and its observations.

    Args:
        experiment_id: identifier for the run, e.g. ``"Exp 1"`` or ``"Test Exp 1"``.
        timestamps: sample times in days, strictly increasing, at least two points.
        design_scalars: ``Z:``-prefixed process design values, constant over the run.
        control_profiles: ``W:``-prefixed control values, one per timestamp.
        observations: ``X:``-prefixed measurements, one per timestamp.
    """

    experiment_id: str
    timestamps: NDArray[np.float64]
    design_scalars: Mapping[str, float]
    control_profiles: Mapping[str, NDArray[np.float64]]
    observations: Mapping[str, NDArray[np.float64]]

    def __post_init__(self) -> None:
        if self.timestamps.ndim != 1:
            raise InvalidExperimentRunError(
                f"{self.experiment_id}: timestamps must be one-dimensional, "
                f"got {self.timestamps.ndim} dimensions"
            )
        if self.timestamps.size < 2:
            raise InvalidExperimentRunError(
                f"{self.experiment_id}: need at least two timestamps to integrate, "
                f"got {self.timestamps.size}"
            )
        if not np.all(np.isfinite(self.timestamps)):
            raise InvalidExperimentRunError(
                f"{self.experiment_id}: timestamps contain missing or infinite values"
            )
        if not np.all(np.diff(self.timestamps) > 0):
            raise InvalidExperimentRunError(
                f"{self.experiment_id}: timestamps must be strictly increasing"
            )

        expected_length = self.timestamps.size
        for group_name, group in (
            ("control profile", self.control_profiles),
            ("observation", self.observations),
        ):
            for variable_name, values in group.items():
                if values.shape != (expected_length,):
                    raise InvalidExperimentRunError(
                        f"{self.experiment_id}: {group_name} {variable_name!r} has "
                        f"length {values.shape} but there are {expected_length} timestamps"
                    )
                # The supplied CSVs contain no gaps in these series, but a request to
                # the inference API can. A single NaN here would propagate silently
                # through the quadratures and return a titre of nan, so it is rejected
                # at construction with the offending positions named.
                non_finite_positions = np.flatnonzero(~np.isfinite(values))
                if non_finite_positions.size:
                    raise InvalidExperimentRunError(
                        f"{self.experiment_id}: {group_name} {variable_name!r} has "
                        f"missing or infinite values at index "
                        f"{non_finite_positions.tolist()}"
                    )

        for variable_name, value in self.design_scalars.items():
            if not np.isfinite(value):
                raise InvalidExperimentRunError(
                    f"{self.experiment_id}: design scalar {variable_name!r} is {value}, "
                    "which is not finite"
                )

    @property
    def duration_days(self) -> float:
        """Harvest time: the last observed timestamp.

        Equals ``Z:ExpDuration`` for every run in both supplied files, but is taken
        from the timestamps because those are what the model actually integrates over.
        """
        return float(self.timestamps[-1])

    def observation(self, variable_name: str) -> NDArray[np.float64]:
        """Return one observation series, with a clear error if it is absent.

        Args:
            variable_name: an ``X:``-prefixed name, e.g. ``schema.OBSERVATION_VIABLE_CELL_DENSITY``.
        """
        try:
            return self.observations[variable_name]
        except KeyError:
            raise InvalidExperimentRunError(
                f"{self.experiment_id}: required observation {variable_name!r} is missing; "
                f"available: {sorted(self.observations)}"
            ) from None
