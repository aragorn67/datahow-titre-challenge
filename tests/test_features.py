"""Tests for the two model regressors.

The features are the one piece of code shared between training and the inference
service, so a silent error here would be served as a confident wrong titre. They are
therefore checked against hand-computable cases wherever possible, rather than only
against themselves.
"""

import numpy as np
import pytest

from titre_predictor import features
from titre_predictor.data import schema
from titre_predictor.domain import ExperimentRun


def _run(
    timestamps: np.ndarray,
    viable_cell_density: np.ndarray,
    lysed_cells: np.ndarray | None = None,
) -> ExperimentRun:
    return ExperimentRun(
        experiment_id="Exp 1",
        timestamps=timestamps,
        design_scalars={},
        control_profiles={},
        observations={
            schema.OBSERVATION_VIABLE_CELL_DENSITY: viable_cell_density,
            schema.OBSERVATION_LYSED_CELLS: (
                np.zeros_like(timestamps) if lysed_cells is None else lysed_cells
            ),
        },
    )


# --- cell-days ----------------------------------------------------------------------


def test_cell_days_of_a_constant_trajectory_is_height_times_duration() -> None:
    timestamps = np.arange(11, dtype=np.float64)
    viable = np.full(11, 3.0)

    assert features.cell_days(timestamps, viable) == pytest.approx(30.0)


def test_cell_days_of_a_straight_ramp_is_the_triangle_area() -> None:
    """A ramp from 0 to 10 over 10 days encloses 50 cell-days."""
    timestamps = np.arange(11, dtype=np.float64)

    assert features.cell_days(timestamps, timestamps.copy()) == pytest.approx(50.0)


def test_cell_days_matches_a_hand_computed_trapezoid() -> None:
    timestamps = np.array([0.0, 1.0, 3.0])
    viable = np.array([2.0, 4.0, 8.0])
    # (2+4)/2 * 1  +  (4+8)/2 * 2  =  3 + 12 = 15
    assert features.cell_days(timestamps, viable) == pytest.approx(15.0)


def test_cell_days_is_zero_for_a_dead_culture() -> None:
    timestamps = np.arange(5, dtype=np.float64)

    assert features.cell_days(timestamps, np.zeros(5)) == 0.0


# --- lysed-cell slope ---------------------------------------------------------------


def test_slope_of_a_straight_line_is_its_gradient() -> None:
    timestamps = np.arange(10, dtype=np.float64)
    lysed = 0.05 * timestamps

    assert features.lysed_cell_slope_at_end(timestamps, lysed) == pytest.approx(0.05)


def test_slope_uses_only_the_trailing_window() -> None:
    """Early behaviour must not influence the estimate at harvest: the pool accelerates
    late, which is exactly the regime the dead-cell recovery depends on."""
    timestamps = np.arange(10, dtype=np.float64)
    lysed = np.concatenate([np.zeros(6), 0.1 * np.arange(1, 5)])

    assert features.lysed_cell_slope_at_end(timestamps, lysed, window_points=4) == pytest.approx(
        0.1
    )


def test_a_negative_slope_is_clipped_to_zero() -> None:
    """The lysed pool is cumulative, so a falling tail is measurement noise. Left
    unclipped it would imply a negative dead-cell pool."""
    timestamps = np.arange(6, dtype=np.float64)
    lysed = np.array([0.0, 0.1, 0.2, 0.3, 0.2, 0.1])

    assert features.lysed_cell_slope_at_end(timestamps, lysed) == 0.0


def test_a_wider_window_averages_more_noise() -> None:
    timestamps = np.arange(8, dtype=np.float64)
    lysed = 0.05 * timestamps
    lysed[-1] += 0.02  # a single noisy final reading

    narrow = features.lysed_cell_slope_at_end(timestamps, lysed, window_points=3)
    wide = features.lysed_cell_slope_at_end(timestamps, lysed, window_points=5)

    assert abs(wide - 0.05) < abs(narrow - 0.05)


def test_window_must_contain_at_least_two_points() -> None:
    timestamps = np.arange(5, dtype=np.float64)
    with pytest.raises(ValueError, match="at least two points"):
        features.lysed_cell_slope_at_end(timestamps, np.zeros(5), window_points=1)


def test_window_cannot_exceed_the_run_length() -> None:
    timestamps = np.arange(5, dtype=np.float64)
    with pytest.raises(ValueError, match="exceeds the 5 available timepoints"):
        features.lysed_cell_slope_at_end(timestamps, np.zeros(5), window_points=6)


# --- dead cells ---------------------------------------------------------------------


def test_dead_pool_is_the_slope_divided_by_the_lysis_constant() -> None:
    timestamps = np.arange(10, dtype=np.float64)
    lysed = 0.002 * timestamps

    dead = features.dead_cells_at_harvest(timestamps, lysed, lysis_rate_constant=0.001)

    assert dead == pytest.approx(2.0)


def test_a_smaller_lysis_constant_implies_a_larger_dead_pool() -> None:
    """Physically: if cells lyse slowly, the same rate of lysate accumulation requires
    more dead cells behind it."""
    timestamps = np.arange(10, dtype=np.float64)
    lysed = 0.002 * timestamps

    slow = features.dead_cells_at_harvest(timestamps, lysed, lysis_rate_constant=0.0005)
    fast = features.dead_cells_at_harvest(timestamps, lysed, lysis_rate_constant=0.005)

    assert slow > fast


def test_lysis_constant_must_be_positive() -> None:
    timestamps = np.arange(5, dtype=np.float64)
    with pytest.raises(ValueError, match="strictly positive"):
        features.dead_cells_at_harvest(timestamps, np.zeros(5), lysis_rate_constant=0.0)


# --- cells synthesised --------------------------------------------------------------


def test_cells_synthesised_sums_viable_growth_lysed_and_dead() -> None:
    timestamps = np.arange(10, dtype=np.float64)
    viable = np.linspace(1.0, 21.0, 10)  # net gain of 20
    lysed = 0.002 * timestamps  # 0.018 at harvest, slope 0.002

    total = features.cells_synthesised(_run(timestamps, viable, lysed), lysis_rate_constant=0.001)

    # 20 (viable) + 0.018 (lysed) + 0.002/0.001 (dead) = 22.018
    assert total == pytest.approx(22.018)


def test_a_culture_that_never_lyses_synthesised_only_its_viable_gain() -> None:
    timestamps = np.arange(10, dtype=np.float64)
    viable = np.linspace(1.0, 11.0, 10)

    total = features.cells_synthesised(_run(timestamps, viable), lysis_rate_constant=0.001)

    assert total == pytest.approx(10.0)


def test_cells_synthesised_exceeds_the_net_viable_gain_when_cells_die() -> None:
    """The point of the decomposition: a culture that peaks and crashes has made far
    more cells than its endpoint difference suggests."""
    timestamps = np.arange(11, dtype=np.float64)
    viable = np.concatenate([np.linspace(1.0, 30.0, 6), np.linspace(26.0, 10.0, 5)])
    lysed = 0.003 * timestamps

    total = features.cells_synthesised(_run(timestamps, viable, lysed), lysis_rate_constant=0.001)
    net_gain = float(viable[-1] - viable[0])

    assert total > net_gain


# --- assembly -----------------------------------------------------------------------


def test_feature_vector_is_ordered_as_declared() -> None:
    timestamps = np.arange(10, dtype=np.float64)
    viable = np.full(10, 2.0)
    run = _run(timestamps, viable)

    vector = features.feature_vector(run, lysis_rate_constant=0.001)

    assert features.FEATURE_NAMES == ("cells_synthesised", "cell_days")
    assert vector[0] == pytest.approx(features.cells_synthesised(run, 0.001))
    assert vector[1] == pytest.approx(features.cell_days(timestamps, viable))


def test_design_matrix_has_one_row_per_run_and_one_column_per_feature() -> None:
    timestamps = np.arange(10, dtype=np.float64)
    runs = [_run(timestamps, np.full(10, float(height))) for height in (1, 2, 3)]

    matrix = features.design_matrix(runs, lysis_rate_constant=0.001)

    assert matrix.shape == (3, len(features.FEATURE_NAMES))


def test_design_matrix_preserves_run_order() -> None:
    """Rows must line up with the targets they will be regressed against."""
    timestamps = np.arange(10, dtype=np.float64)
    runs = [_run(timestamps, np.full(10, float(height))) for height in (1, 2, 3)]

    matrix = features.design_matrix(runs, lysis_rate_constant=0.001)

    cell_days_column = matrix[:, features.FEATURE_NAMES.index("cell_days")]
    np.testing.assert_allclose(cell_days_column, [9.0, 18.0, 27.0])


def test_design_matrix_rejects_an_empty_sequence() -> None:
    with pytest.raises(ValueError, match="at least one run"):
        features.design_matrix([], lysis_rate_constant=0.001)


# --- against the real data ----------------------------------------------------------


def test_features_are_finite_and_positive_for_every_training_run(
    train_runs: list[ExperimentRun],
) -> None:
    matrix = features.design_matrix(train_runs, lysis_rate_constant=0.0012)

    assert np.isfinite(matrix).all()
    assert (matrix > 0).all(), "every real run both grows and accumulates cell-days"


def test_features_are_finite_for_every_test_run(test_runs: list[ExperimentRun]) -> None:
    matrix = features.design_matrix(test_runs, lysis_rate_constant=0.0012)

    assert np.isfinite(matrix).all()
