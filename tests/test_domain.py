"""Tests for the ExperimentRun invariants.

These rules protect the training pipeline and the inference API equally: the same
validation runs whether a run is built from a CSV or deserialised from a request body,
so an invalid experiment cannot be constructed at all.
"""

import numpy as np
import pytest

from titre_predictor.data import schema
from titre_predictor.domain import ExperimentRun, InvalidExperimentRunError


def _run(
    timestamps: np.ndarray,
    observations: dict[str, np.ndarray] | None = None,
    design_scalars: dict[str, float] | None = None,
) -> ExperimentRun:
    return ExperimentRun(
        experiment_id="Exp 1",
        timestamps=timestamps,
        design_scalars={} if design_scalars is None else design_scalars,
        control_profiles={},
        observations={} if observations is None else observations,
    )


def test_a_well_formed_run_is_accepted() -> None:
    run = _run(
        np.arange(5, dtype=np.float64),
        {schema.OBSERVATION_VIABLE_CELL_DENSITY: np.ones(5)},
    )
    assert run.duration_days == 4.0


def test_timestamps_must_be_strictly_increasing() -> None:
    with pytest.raises(InvalidExperimentRunError, match="strictly increasing"):
        _run(np.array([0.0, 1.0, 1.0, 2.0]))


def test_timestamps_must_not_go_backwards() -> None:
    with pytest.raises(InvalidExperimentRunError, match="strictly increasing"):
        _run(np.array([0.0, 2.0, 1.0]))


def test_a_single_timepoint_cannot_be_integrated() -> None:
    with pytest.raises(InvalidExperimentRunError, match="at least two timestamps"):
        _run(np.array([0.0]))


def test_timestamps_must_be_one_dimensional() -> None:
    with pytest.raises(InvalidExperimentRunError, match="one-dimensional"):
        _run(np.zeros((3, 2)))


def test_an_observation_of_the_wrong_length_is_rejected() -> None:
    with pytest.raises(InvalidExperimentRunError, match="but there are 5 timestamps"):
        _run(
            np.arange(5, dtype=np.float64),
            {schema.OBSERVATION_VIABLE_CELL_DENSITY: np.ones(4)},
        )


def test_a_missing_observation_value_is_rejected() -> None:
    """The CSVs have no gaps here, but an API request can. A single NaN would
    propagate through the quadratures and return a titre of nan."""
    values = np.array([1.0, np.nan, 3.0])
    with pytest.raises(InvalidExperimentRunError, match="missing or infinite"):
        _run(np.arange(3, dtype=np.float64), {schema.OBSERVATION_VIABLE_CELL_DENSITY: values})


def test_a_missing_observation_value_is_located_by_index() -> None:
    values = np.array([1.0, 2.0, np.nan, 4.0, np.nan])
    with pytest.raises(InvalidExperimentRunError, match=r"index \[2, 4\]"):
        _run(np.arange(5, dtype=np.float64), {schema.OBSERVATION_GLUCOSE: values})


def test_an_infinite_observation_value_is_rejected() -> None:
    values = np.array([1.0, np.inf, 3.0])
    with pytest.raises(InvalidExperimentRunError, match="missing or infinite"):
        _run(np.arange(3, dtype=np.float64), {schema.OBSERVATION_LACTATE: values})


def test_a_missing_control_value_is_rejected() -> None:
    with pytest.raises(InvalidExperimentRunError, match="control profile"):
        ExperimentRun(
            experiment_id="Exp 1",
            timestamps=np.arange(3, dtype=np.float64),
            design_scalars={},
            control_profiles={schema.CONTROL_TEMPERATURE: np.array([36.0, np.nan, 36.0])},
            observations={},
        )


def test_missing_timestamps_are_reported_as_such() -> None:
    """Without an explicit check this would surface as 'not strictly increasing',
    because every comparison against NaN is false -- a misleading message."""
    with pytest.raises(InvalidExperimentRunError, match="timestamps contain missing"):
        _run(np.array([0.0, np.nan, 2.0]))


def test_a_non_finite_design_scalar_is_rejected() -> None:
    with pytest.raises(InvalidExperimentRunError, match="not finite"):
        _run(
            np.arange(3, dtype=np.float64),
            design_scalars={schema.DESIGN_FEED_START: float("nan")},
        )


def test_requesting_an_absent_observation_names_what_is_available() -> None:
    run = _run(np.arange(3, dtype=np.float64), {schema.OBSERVATION_GLUCOSE: np.ones(3)})

    with pytest.raises(InvalidExperimentRunError) as raised:
        run.observation(schema.OBSERVATION_VIABLE_CELL_DENSITY)

    assert schema.OBSERVATION_GLUCOSE in str(raised.value)


def test_observation_returns_the_stored_series() -> None:
    values = np.array([1.0, 2.0, 3.0])
    run = _run(np.arange(3, dtype=np.float64), {schema.OBSERVATION_LYSED_CELLS: values})

    np.testing.assert_array_equal(run.observation(schema.OBSERVATION_LYSED_CELLS), values)
