"""Tests for the quantities the titre model integrates, and the screening features.

The features are the one piece of code shared between training and the inference service,
so a silent error here would be served as a confident wrong titre. They are checked against
hand-computable cases wherever possible, rather than only against themselves.

Three groups carry weight beyond ordinary coverage:

* the **monotone fit's guarantees** -- a non-negative derivative everywhere and smoothing
  rather than interpolation -- because the entire case for replacing the old 4-point slope
  window rests on them;
* the **telescoping identities** ``sum_j dC_j = Delta(Xv+Xd+Xl)`` and ``sum_j gX_j = gammaX``,
  because they are what make M2 nest M1 and M3 nest both exactly;
* the ``nan`` **policy** on screening features, because the missingness is collinear with
  duration and quietly filling it would inject a duration signal disguised as something else.
"""

import numpy as np
import pytest

from titre_predictor import features, kinetics
from titre_predictor.data import schema
from titre_predictor.domain import ExperimentRun

DESIGN_DEFAULTS = {
    schema.DESIGN_DISSOLVED_OXYGEN: 50.0,
    schema.DESIGN_STIRRING: 200.0,
    schema.DESIGN_FEED_RATE_GLUCOSE: 4.0,
    schema.DESIGN_FEED_RATE_GLUTAMINE: 7.0,
    schema.DESIGN_FEED_START: 3.0,
    schema.DESIGN_FEED_END: 11.0,
    schema.DESIGN_TEMPERATURE_SHIFT: 5.0,
    schema.DESIGN_PH_SHIFT: 6.0,
}


def _run(
    timestamps: np.ndarray,
    viable_cell_density: np.ndarray,
    lysed_cells: np.ndarray | None = None,
    **design: float,
) -> ExperimentRun:
    """A synthetic run carrying everything the features need."""
    point_count = timestamps.size
    return ExperimentRun(
        experiment_id="Exp 1",
        timestamps=timestamps,
        design_scalars={**DESIGN_DEFAULTS, **design},
        control_profiles={
            schema.CONTROL_TEMPERATURE: np.full(point_count, 37.0),
            schema.CONTROL_PH: np.full(point_count, 7.0),
        },
        observations={
            schema.OBSERVATION_VIABLE_CELL_DENSITY: viable_cell_density,
            schema.OBSERVATION_LYSED_CELLS: (
                np.zeros_like(timestamps) if lysed_cells is None else lysed_cells
            ),
            schema.OBSERVATION_GLUCOSE: np.linspace(20.0, 2.0, point_count),
            schema.OBSERVATION_GLUTAMINE: np.linspace(5.0, 0.5, point_count),
            schema.OBSERVATION_LACTATE: np.linspace(0.0, 6.0, point_count),
            schema.OBSERVATION_AMMONIA: np.linspace(0.0, 9.0, point_count),
        },
    )


# --- cell-days ------------------------------------------------------------------------


def test_cell_days_of_a_constant_trajectory_is_height_times_duration() -> None:
    timestamps = np.arange(11, dtype=np.float64)

    assert features.cell_days(timestamps, np.full(11, 3.0)) == pytest.approx(30.0)


def test_cell_days_matches_a_hand_computed_trapezoid() -> None:
    timestamps = np.array([0.0, 1.0, 3.0])
    viable = np.array([2.0, 4.0, 8.0])
    # (2+4)/2 * 1  +  (4+8)/2 * 2  =  3 + 12 = 15
    assert features.cell_days(timestamps, viable) == pytest.approx(15.0)


def test_cell_days_is_zero_for_a_dead_culture() -> None:
    timestamps = np.arange(5, dtype=np.float64)

    assert features.cell_days(timestamps, np.zeros(5)) == 0.0


def test_interval_cell_days_sum_to_the_whole_run() -> None:
    """Required exactly, so that F == 1 reduces the weighted integral to the plain one."""
    timestamps = np.array([0.0, 1.0, 3.0, 6.5])
    viable = np.array([2.0, 4.0, 8.0, 3.0])

    per_interval = features.interval_cell_days(timestamps, viable)

    assert float(np.sum(per_interval)) == pytest.approx(features.cell_days(timestamps, viable))


# --- the monotone lysate fit ----------------------------------------------------------


def test_a_straight_line_is_recovered_with_the_right_slope() -> None:
    timestamps = np.arange(11, dtype=np.float64)
    lysed = 0.05 * timestamps

    curve = features.fit_lysate_curve(timestamps, lysed)

    np.testing.assert_allclose(curve.value(timestamps), lysed, atol=1e-9)
    np.testing.assert_allclose(curve.derivative(timestamps), 0.05, atol=1e-9)


def test_the_derivative_is_never_negative_even_for_a_falling_input() -> None:
    """The physical constraint. X:Lysed is a cumulative pool, so a decreasing stretch is
    measurement noise -- and it is common: 247 of the 1170 intervals in the supplied data
    decrease. Left unconstrained those would imply a negative dead-cell pool."""
    timestamps = np.arange(12, dtype=np.float64)
    lysed = np.array([0.0, 0.02, 0.0, 0.03, 0.01, 0.0, 0.05, 0.04, 0.09, 0.08, 0.15, 0.13])

    curve = features.fit_lysate_curve(timestamps, lysed)
    dense = np.linspace(0.0, 11.0, 500)

    assert np.all(curve.derivative(dense) >= 0.0)


def test_the_fitted_curve_itself_never_decreases() -> None:
    timestamps = np.arange(12, dtype=np.float64)
    lysed = np.array([0.0, 0.02, 0.0, 0.03, 0.01, 0.0, 0.05, 0.04, 0.09, 0.08, 0.15, 0.13])

    curve = features.fit_lysate_curve(timestamps, lysed)
    dense = np.linspace(0.0, 11.0, 500)

    assert np.all(np.diff(curve.value(dense)) >= -1e-12)


def test_the_fit_smooths_rather_than_interpolates() -> None:
    """The reason PCHIP was rejected. An interpolating fit would pass through every point
    and hand the noise straight to the derivative."""
    timestamps = np.arange(12, dtype=np.float64)
    lysed = 0.02 * timestamps
    lysed[5] += 0.05  # one badly wrong reading

    curve = features.fit_lysate_curve(timestamps, lysed)

    assert curve.value(timestamps)[5] < lysed[5]
    assert curve.residual_root_mean_square > 0.0


def test_a_flat_input_gives_a_zero_derivative_rather_than_a_negative_one() -> None:
    timestamps = np.arange(9, dtype=np.float64)

    curve = features.fit_lysate_curve(timestamps, np.zeros(9))

    np.testing.assert_allclose(curve.derivative(timestamps), 0.0, atol=1e-12)


def test_the_fit_rejects_mismatched_arrays() -> None:
    with pytest.raises(ValueError, match="same shape"):
        features.fit_lysate_curve(np.arange(5, dtype=np.float64), np.zeros(4))


def test_the_fit_rejects_a_single_point() -> None:
    with pytest.raises(ValueError, match="at least two points"):
        features.fit_lysate_curve(np.zeros(1), np.zeros(1))


def test_a_short_run_shrinks_the_basis_instead_of_interpolating() -> None:
    """The inference service may be handed a run shorter than anything in training. The
    basis must shrink rather than become underdetermined, which would defeat the smoothing."""
    timestamps = np.arange(4, dtype=np.float64)
    lysed = np.array([0.0, 0.03, 0.01, 0.06])

    curve = features.fit_lysate_curve(timestamps, lysed)

    assert curve.coefficients.size < timestamps.size
    assert np.all(curve.derivative(np.linspace(0.0, 3.0, 100)) >= 0.0)


# --- cells made, and the telescoping identities ---------------------------------------


def test_cells_made_is_viable_plus_lysate_plus_the_recovered_dead_pool() -> None:
    timestamps = np.arange(10, dtype=np.float64)
    viable = np.linspace(1.0, 11.0, 10)
    lysed = 0.002 * timestamps
    curve = features.fit_lysate_curve(timestamps, lysed)

    made = features.cells_made(timestamps, viable, curve, lysis_rate_constant=0.001)
    expected = viable + curve.value(timestamps) + curve.derivative(timestamps) / 0.001

    np.testing.assert_allclose(made, expected)


def test_cells_made_rejects_a_non_positive_lysis_constant() -> None:
    timestamps = np.arange(5, dtype=np.float64)
    curve = features.fit_lysate_curve(timestamps, np.zeros(5))

    with pytest.raises(ValueError, match="strictly positive"):
        features.cells_made(timestamps, np.ones(5), curve, lysis_rate_constant=0.0)


def test_interval_growth_sums_to_the_endpoint_difference() -> None:
    """``sum_j dC_j = C(T) - C(0)``. This is what makes M3 nest M1 exactly rather than
    approximately, so it is asserted rather than assumed."""
    timestamps = np.arange(11, dtype=np.float64)
    viable = np.concatenate([np.linspace(1.0, 30.0, 6), np.linspace(26.0, 10.0, 5)])
    lysed = 0.003 * timestamps
    run = _run(timestamps, viable, lysed)
    quantities = features.run_quantities(run)

    made = features.cells_made(timestamps, viable, quantities.lysate_curve, 0.05)

    assert quantities.cells_synthesised(0.05) == pytest.approx(float(made[-1] - made[0]))


def test_a_smaller_lysis_constant_implies_more_cells_synthesised() -> None:
    """Physically: if cells lyse slowly, the same lysate accumulation requires a larger
    dead pool behind it, so more cells were made."""
    timestamps = np.arange(10, dtype=np.float64)
    viable = np.linspace(1.0, 21.0, 10)
    quantities = features.run_quantities(_run(timestamps, viable, 0.002 * timestamps))

    assert quantities.cells_synthesised(0.0005) > quantities.cells_synthesised(0.005)


def test_interval_growth_rejects_a_non_positive_lysis_constant() -> None:
    timestamps = np.arange(10, dtype=np.float64)
    quantities = features.run_quantities(_run(timestamps, np.linspace(1.0, 5.0, 10)))

    with pytest.raises(ValueError, match="strictly positive"):
        quantities.interval_growth(0.0)


def test_cells_synthesised_exceeds_the_net_viable_gain_when_cells_die() -> None:
    """The point of the decomposition: a culture that peaks and crashes has made far more
    cells than its endpoint difference suggests. Using net growth instead is M0, which
    fitted alpha = -9.9 -- the model claiming secreted antibody disappears as cells die."""
    timestamps = np.arange(11, dtype=np.float64)
    viable = np.concatenate([np.linspace(1.0, 30.0, 6), np.linspace(26.0, 10.0, 5)])
    quantities = features.run_quantities(_run(timestamps, viable, 0.003 * timestamps))

    assert quantities.cells_synthesised(0.05) > float(viable[-1] - viable[0])


def test_run_quantities_reproduces_the_whole_run_cell_days() -> None:
    timestamps = np.arange(10, dtype=np.float64)
    viable = np.linspace(1.0, 21.0, 10)
    quantities = features.run_quantities(_run(timestamps, viable))

    assert quantities.cell_days == pytest.approx(features.cell_days(timestamps, viable))


# --- the environmental factor across an interval --------------------------------------


def test_no_mechanisms_leaves_every_interval_factor_at_one() -> None:
    """What makes M2 reduce exactly to M1."""
    timestamps = np.arange(8, dtype=np.float64)
    quantities = features.run_quantities(_run(timestamps, np.linspace(1.0, 8.0, 8)))

    factor = features.interval_factor((), quantities, ())

    np.testing.assert_array_equal(factor, np.ones(7))


def test_the_interval_factor_is_the_trapezoidal_average_of_its_endpoints() -> None:
    """Trapezoidal in the product, not F evaluated at a midpoint state: F is nonlinear, so
    F(mean z) != mean F(z), and metabolites move sharply within a day once feeding starts."""
    timestamps = np.arange(8, dtype=np.float64)
    quantities = features.run_quantities(_run(timestamps, np.linspace(1.0, 8.0, 8)))
    mechanisms = kinetics.resolve(["glucose_limitation"])

    pointwise = kinetics.environmental_factor(mechanisms, quantities.state, [4.0])
    factor = features.interval_factor(mechanisms, quantities, [4.0])

    np.testing.assert_allclose(factor, 0.5 * (pointwise[:-1] + pointwise[1:]))


# --- screening features ---------------------------------------------------------------


def test_a_constant_series_has_that_value_as_its_cell_weighted_exposure() -> None:
    timestamps = np.arange(8, dtype=np.float64)
    run = _run(timestamps, np.linspace(1.0, 8.0, 8))

    assert features.run_features(run)["exposure_temp"] == pytest.approx(37.0)


def test_exposure_is_weighted_by_cell_days_not_by_time() -> None:
    """Derived, not chosen: qbar_P depends on the cell-weighted mean of z, so a period with
    few cells must count for less than one with many."""
    timestamps = np.array([0.0, 1.0, 2.0])
    viable = np.array([1.0, 1.0, 100.0])  # almost all cell-days sit in the second interval
    run = _run(timestamps, viable)
    run.observations[schema.OBSERVATION_GLUCOSE][:] = np.array([0.0, 0.0, 10.0])

    exposure = features.run_features(run)["exposure_Glc"]
    time_weighted = float(np.trapezoid([0.0, 0.0, 10.0], timestamps)) / 2.0

    assert exposure > time_weighted


def test_the_phase_split_is_at_a_fixed_absolute_day_for_every_run() -> None:
    """A relative midpoint would make 'late' mean days 3.5-7 in a short run and 7-14 in a
    long one -- different regimes, confounded with duration in exactly the wrong direction."""
    short = _run(np.arange(9, dtype=np.float64), np.full(9, 5.0))
    long_run = _run(np.arange(15, dtype=np.float64), np.full(15, 5.0))

    short_features = features.run_features(short, phase_split_day=7.0)
    long_features = features.run_features(long_run, phase_split_day=7.0)

    # Both runs hold glucose at the same linear ramp shape, but over different horizons, so
    # a late window that meant "the last half" would give these the same value.
    assert short_features["exposure_Glc_late"] != pytest.approx(long_features["exposure_Glc_late"])


def test_a_run_ending_at_the_split_day_has_no_late_window() -> None:
    """Reported as nan rather than filled, because the missingness is exactly 'this run
    never reached that regime' -- which is the train/test shift itself."""
    run = _run(np.arange(8, dtype=np.float64), np.full(8, 5.0))

    values = features.run_features(run, phase_split_day=7.0)

    assert np.isnan(values["exposure_Glc_late"])
    assert np.isfinite(values["exposure_Glc_early"])


def test_viable_density_at_a_shift_is_interpolated_from_the_samples() -> None:
    timestamps = np.arange(11, dtype=np.float64)
    viable = 2.0 * timestamps
    run = _run(timestamps, viable, **{schema.DESIGN_TEMPERATURE_SHIFT: 4.5})

    assert features.run_features(run)["vcd_at_tempShift"] == pytest.approx(9.0)


def test_a_shift_beyond_the_end_of_the_run_is_not_reported_as_the_final_density() -> None:
    """Over half the training runs never reach their temperature shift. Clamping would
    report final VCD under a shift feature's name -- a different quantity entirely."""
    timestamps = np.arange(8, dtype=np.float64)
    run = _run(timestamps, np.linspace(1.0, 8.0, 8), **{schema.DESIGN_TEMPERATURE_SHIFT: 12.0})

    assert np.isnan(features.run_features(run)["vcd_at_tempShift"])


def test_the_feed_window_counts_only_feeding_delivered_before_harvest() -> None:
    timestamps = np.arange(8, dtype=np.float64)
    run = _run(
        timestamps,
        np.full(8, 5.0),
        **{schema.DESIGN_FEED_START: 3.0, schema.DESIGN_FEED_END: 11.0},
    )

    # Feeding runs from day 3, but the run ends on day 7.
    assert features.run_features(run)["feed_window_days"] == pytest.approx(4.0)


def test_duration_is_offered_so_residual_duration_dependence_can_be_tested() -> None:
    run = _run(np.arange(15, dtype=np.float64), np.full(15, 5.0))

    assert features.run_features(run)["duration_days"] == pytest.approx(14.0)


def test_growth_rate_is_positive_while_growing_and_negative_while_dying() -> None:
    growing = _run(np.arange(8, dtype=np.float64), np.linspace(1.0, 20.0, 8))
    dying = _run(np.arange(8, dtype=np.float64), np.linspace(20.0, 1.0, 8))

    assert features.run_features(growing)["growth_rate_mean"] > 0.0
    assert features.run_features(dying)["growth_rate_mean"] < 0.0


# --- assembly -------------------------------------------------------------------------


def test_the_feature_frame_has_one_row_per_run_and_stable_columns() -> None:
    timestamps = np.arange(9, dtype=np.float64)
    runs = [_run(timestamps, np.full(9, float(height))) for height in (1, 2, 3)]

    names, matrix = features.feature_frame(runs)

    assert matrix.shape == (3, len(names))
    assert len(set(names)) == len(names)


def test_the_feature_frame_preserves_run_order() -> None:
    """Rows must line up with the targets they will be regressed against."""
    timestamps = np.arange(9, dtype=np.float64)
    runs = [_run(timestamps, np.full(9, float(height))) for height in (1, 2, 3)]

    names, matrix = features.feature_frame(runs)

    np.testing.assert_allclose(matrix[:, names.index("cell_days")], [8.0, 16.0, 24.0])


def test_the_feature_frame_rejects_an_empty_sequence() -> None:
    with pytest.raises(ValueError, match="at least one run"):
        features.feature_frame([])


def test_runs_producing_different_columns_are_rejected_rather_than_stacked() -> None:
    """A run missing a control profile yields fewer features. Stacking those rows would
    silently misalign every column after the gap."""
    timestamps = np.arange(9, dtype=np.float64)
    complete = _run(timestamps, np.full(9, 5.0))
    without_ph = ExperimentRun(
        experiment_id="Exp 2",
        timestamps=timestamps,
        design_scalars=dict(DESIGN_DEFAULTS),
        control_profiles={schema.CONTROL_TEMPERATURE: np.full(9, 37.0)},
        observations=dict(complete.observations),
    )

    with pytest.raises(ValueError, match="would not line up"):
        features.feature_frame([complete, without_ph])


# --- against the real data ------------------------------------------------------------


def test_the_lysate_derivative_is_non_negative_for_every_supplied_run(
    train_runs: list[ExperimentRun],
    test_runs: list[ExperimentRun],
) -> None:
    """The guarantee the whole construction exists for, checked on a dense grid rather
    than only at the sample points."""
    for run in train_runs + test_runs:
        curve = features.fit_lysate_curve(
            run.timestamps, run.observation(schema.OBSERVATION_LYSED_CELLS)
        )
        dense = np.linspace(run.timestamps[0], run.timestamps[-1], 200)

        assert np.all(curve.derivative(dense) >= 0.0), run.experiment_id


def test_the_fit_residuals_sit_within_the_measurement_noise(
    train_runs: list[ExperimentRun],
) -> None:
    """The acceptance test for the smoothing level. Noise on X:Lysed is additive with an
    absolute sd of roughly 0.0087 -- measured from the early window, where the true pool is
    still ~0 -- and is clipped at zero. A residual RMS far below that would mean the curve
    is chasing noise; far above would mean it is missing signal."""
    residuals = [
        features.run_quantities(run).lysate_curve.residual_root_mean_square for run in train_runs
    ]

    assert float(np.median(residuals)) < 0.0087
    assert float(np.median(residuals)) > 0.002


def test_the_telescoping_identity_holds_for_every_supplied_run(
    train_runs: list[ExperimentRun],
    test_runs: list[ExperimentRun],
) -> None:
    for run in train_runs + test_runs:
        quantities = features.run_quantities(run)
        made = features.cells_made(
            run.timestamps,
            run.observation(schema.OBSERVATION_VIABLE_CELL_DENSITY),
            quantities.lysate_curve,
            0.05,
        )

        assert quantities.cells_synthesised(0.05) == pytest.approx(
            float(made[-1] - made[0]), abs=1e-9
        ), run.experiment_id


def test_cell_days_are_positive_and_finite_for_every_supplied_run(
    train_runs: list[ExperimentRun],
    test_runs: list[ExperimentRun],
) -> None:
    for run in train_runs + test_runs:
        quantities = features.run_quantities(run)

        assert np.isfinite(quantities.cell_days)
        assert quantities.cell_days > 0.0, run.experiment_id


def test_every_screening_feature_that_is_defined_is_finite(
    train_runs: list[ExperimentRun],
    test_runs: list[ExperimentRun],
) -> None:
    """nan is a deliberate signal for 'this run never reached that regime'. An infinity
    would be a division that went wrong, and must not appear."""
    _names, matrix = features.feature_frame(train_runs + test_runs)

    assert not np.isinf(matrix).any()


def test_only_the_phase_and_shift_features_are_ever_undefined(
    train_runs: list[ExperimentRun],
    test_runs: list[ExperimentRun],
) -> None:
    """Pins the nan policy: everything else must be defined for every run, so a new nan
    appearing elsewhere is a bug rather than a documented gap."""
    names, matrix = features.feature_frame(train_runs + test_runs)
    undefined = {
        name for name, column in zip(names, matrix.T, strict=True) if np.isnan(column).any()
    }

    assert undefined == {
        "exposure_Glc_late",
        "exposure_Gln_late",
        "exposure_Lac_late",
        "exposure_Amm_late",
        "exposure_Lysed_late",
        "exposure_temp_late",
        "exposure_pH_late",
        "vcd_at_tempShift",
        "vcd_at_phShift",
        "vcd_at_FeedEnd",
    }
