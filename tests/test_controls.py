"""Tests for reconstructing the ``W:`` control profiles from the ``Z:`` design scalars.

The headline test asserts the reconstruction reproduces the supplied columns exactly,
across every run of both files. It is a guard on a claim the model relies on: that
``W:`` is a re-parameterisation of ``Z:`` and carries no independent information. If the
data format or the control convention ever changes, this fails loudly.
"""

import numpy as np
import pytest

from titre_predictor.data import schema
from titre_predictor.data.controls import (
    control_profiles_match,
    reconstruct_control_profiles,
)
from titre_predictor.domain import ExperimentRun


def test_reconstruction_reproduces_supplied_profiles_for_every_training_run(
    train_runs: list[ExperimentRun],
) -> None:
    for run in train_runs:
        reconstructed = reconstruct_control_profiles(run.design_scalars, run.timestamps)
        for name, expected in run.control_profiles.items():
            np.testing.assert_allclose(
                reconstructed[name], expected, atol=0.0, rtol=0.0, err_msg=run.experiment_id
            )


def test_reconstruction_reproduces_supplied_profiles_for_every_test_run(
    test_runs: list[ExperimentRun],
) -> None:
    for run in test_runs:
        reconstructed = reconstruct_control_profiles(run.design_scalars, run.timestamps)
        assert control_profiles_match(run.control_profiles, reconstructed), run.experiment_id


def _design_scalars() -> dict[str, float]:
    """The design of ``Test Exp 1``, which is also the example payload in the spec."""
    return {
        schema.DESIGN_FEED_START: 3.0,
        schema.DESIGN_FEED_END: 11.0,
        schema.DESIGN_FEED_RATE_GLUCOSE: 5.0,
        schema.DESIGN_FEED_RATE_GLUTAMINE: 6.0,
        schema.DESIGN_PH_START: 7.4,
        schema.DESIGN_PH_END: 6.3,
        schema.DESIGN_PH_SHIFT: 13.0,
        schema.DESIGN_TEMPERATURE_START: 36.3,
        schema.DESIGN_TEMPERATURE_END: 36.9,
        schema.DESIGN_TEMPERATURE_SHIFT: 10.0,
    }


def test_temperature_steps_at_the_shift_day() -> None:
    timestamps = np.arange(15, dtype=np.float64)
    profiles = reconstruct_control_profiles(_design_scalars(), timestamps)

    temperature = profiles[schema.CONTROL_TEMPERATURE]
    assert np.all(temperature[:10] == 36.3), "before the shift"
    assert np.all(temperature[10:] == 36.9), "from the shift day onwards"


def test_a_shift_beyond_the_run_never_takes_effect() -> None:
    """In many training runs the shift day exceeds the run length, so the profile stays
    at its starting value throughout. The test runs are all long enough to shift."""
    timestamps = np.arange(8, dtype=np.float64)
    profiles = reconstruct_control_profiles(_design_scalars(), timestamps)

    assert np.all(profiles[schema.CONTROL_TEMPERATURE] == 36.3)
    assert np.all(profiles[schema.CONTROL_PH] == 7.4)


def test_feed_window_is_closed_left_and_open_right() -> None:
    """FeedStart = 3, FeedEnd = 11 means fed on days 3..10 and not on day 11."""
    timestamps = np.arange(15, dtype=np.float64)
    profiles = reconstruct_control_profiles(_design_scalars(), timestamps)

    feed = profiles[schema.CONTROL_FEED_GLUCOSE]
    assert np.all(feed[:3] == 0.0), "before FeedStart"
    assert np.all(feed[3:11] == 5.0), "days 3 to 10 inclusive"
    assert np.all(feed[11:] == 0.0), "from FeedEnd onwards"


def test_missing_design_scalar_is_reported_by_name() -> None:
    incomplete = _design_scalars()
    del incomplete[schema.DESIGN_PH_SHIFT]

    with pytest.raises(KeyError, match=schema.DESIGN_PH_SHIFT):
        reconstruct_control_profiles(incomplete, np.arange(5, dtype=np.float64))


def test_profiles_do_not_match_when_a_variable_is_absent() -> None:
    timestamps = np.arange(5, dtype=np.float64)
    reconstructed = reconstruct_control_profiles(_design_scalars(), timestamps)
    incomplete = dict(reconstructed)
    del incomplete[schema.CONTROL_PH]

    assert not control_profiles_match(incomplete, reconstructed)
