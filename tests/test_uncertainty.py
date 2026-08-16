"""Tests for identifiability analysis and prediction uncertainty.

The two properties worth asserting are the ones that make the output trustworthy:

* a profile must be **able to say "undetermined"**. If it reports a narrow interval for a
  parameter the data cannot constrain, it is worse than no analysis at all -- it manufactures
  confidence. The constant-pH fixture below is unidentifiable *by construction*, and the
  profile must report it as unbounded rather than inventing bounds.
* a prediction interval must be **wider** than a parameter interval, because runs scatter
  around the model for reasons beyond parameter ignorance. Reporting the narrow one where
  the wide one is wanted is the specific failure this module exists to prevent.
"""

import numpy as np
import pytest

from titre_predictor import evaluation, model, uncertainty
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


def _run(identifier: str, peak: float, ph: float, point_count: int = 9) -> ExperimentRun:
    timestamps = np.arange(point_count, dtype=np.float64)
    rise = np.linspace(1.0, peak, point_count // 2 + 1)
    fall = np.linspace(peak * 0.9, peak * 0.4, point_count - rise.size)
    viable = np.concatenate([rise, fall])
    return ExperimentRun(
        experiment_id=identifier,
        timestamps=timestamps,
        design_scalars=dict(DESIGN_DEFAULTS),
        control_profiles={
            schema.CONTROL_TEMPERATURE: np.full(point_count, 37.0),
            schema.CONTROL_PH: np.full(point_count, ph),
        },
        observations={
            schema.OBSERVATION_VIABLE_CELL_DENSITY: viable,
            schema.OBSERVATION_LYSED_CELLS: 0.004 * timestamps,
            schema.OBSERVATION_GLUCOSE: np.linspace(20.0, 2.0, point_count),
            schema.OBSERVATION_GLUTAMINE: np.linspace(5.0, 0.5, point_count),
            schema.OBSERVATION_LACTATE: np.linspace(0.0, 6.0, point_count),
            schema.OBSERVATION_AMMONIA: np.linspace(0.0, 9.0, point_count),
        },
    )


def _dataset(
    varying_ph: bool,
    run_count: int = 24,
    true_sensitivity: float = 1.5,
    noise: float = 0.01,
    seed: int = 0,
) -> tuple[list[ExperimentRun], dict[str, float]]:
    """Runs and targets generated from M2 with a known pH sensitivity.

    With ``varying_ph`` false every run sits at the same pH, so the pH factor is a single
    constant multiplying every run's non-growth term -- which ``beta`` absorbs exactly.
    ``theta_pH`` is then unidentifiable by construction, which is what makes it a fair test
    of whether the profile can say so.
    """
    generator = np.random.default_rng(seed)
    peaks = np.linspace(10.0, 44.0, run_count)
    phs = np.linspace(6.1, 7.4, run_count) if varying_ph else np.full(run_count, 6.75)
    runs = [
        _run(f"Exp {index}", float(peak), float(ph))
        for index, (peak, ph) in enumerate(zip(peaks, phs, strict=True))
    ]
    from titre_predictor import features, kinetics

    quantities = [features.run_quantities(run) for run in runs]
    growth, non_growth = model.design_columns(
        model.VARIANTS["M2"],
        quantities,
        0.05,
        kinetics.resolve(["ph_response"]),
        [true_sensitivity],
    )
    clean = 20.0 * growth + 3.0 * non_growth
    targets = {
        run.experiment_id: float(value * (1.0 + noise * generator.normal()))
        for run, value in zip(runs, clean, strict=True)
    }
    return runs, targets


# --- pinning a shape constant -----------------------------------------------------------


def test_pinning_a_constant_fixes_it_and_re_optimises_the_rest() -> None:
    """What separates a profile from a slice: the other parameters must stay free."""
    runs, targets = _dataset(varying_ph=True)

    free, free_diagnostics = model.fit(runs, targets, "M2", ["ph_response"])
    pinned, pinned_diagnostics = model.fit(
        runs, targets, "M2", ["ph_response"], fixed_shape_constants={"theta_pH": 0.2}
    )

    assert dict(
        zip(
            pinned_diagnostics.shape_constant_names,
            pinned_diagnostics.shape_constant_values,
            strict=True,
        )
    )["theta_pH"] == pytest.approx(0.2)
    # kl was free to move and should have, to compensate for the imposed value.
    assert pinned.lysis_rate_constant != pytest.approx(free.lysis_rate_constant)
    assert pinned_diagnostics.residual_sum_of_squares >= free_diagnostics.residual_sum_of_squares
    del free_diagnostics


def test_pinning_an_unknown_constant_is_rejected() -> None:
    runs, targets = _dataset(varying_ph=True)

    with pytest.raises(KeyError, match="theta_T"):
        model.fit(runs, targets, "M2", ["ph_response"], fixed_shape_constants={"theta_T": 1.0})


# --- profile likelihood -------------------------------------------------------------------


def test_a_determined_parameter_profiles_sharply_and_brackets_the_truth() -> None:
    runs, targets = _dataset(varying_ph=True, true_sensitivity=1.5, noise=0.01)

    profile = uncertainty.profile_likelihood(
        runs, targets, "theta_pH", "M2", ["ph_response"], point_count=13
    )

    assert profile.is_identified, "a strong, varying pH effect must be bounded on both sides"
    assert profile.lower < 1.5 < profile.upper
    assert profile.relative_rise > 0.5, "the residual must rise substantially across the range"


def test_an_unidentifiable_parameter_is_reported_as_unbounded() -> None:
    """The test that matters. With pH constant across runs the pH factor is a single
    constant absorbed by beta, so theta_pH cannot be determined at all. A profile that
    returned a narrow interval here would be manufacturing confidence."""
    runs, targets = _dataset(varying_ph=False)

    profile = uncertainty.profile_likelihood(
        runs, targets, "theta_pH", "M2", ["ph_response"], point_count=9
    )

    assert not profile.is_identified
    assert profile.relative_rise < 1e-6, "the objective must be flat when nothing pins it"


def test_a_profile_never_dips_below_the_free_fit() -> None:
    """Pinning is a constraint, so it cannot fit better than leaving the parameter free."""
    runs, targets = _dataset(varying_ph=True)

    profile = uncertainty.profile_likelihood(
        runs, targets, "theta_pH", "M2", ["ph_response"], point_count=9
    )

    assert np.all(profile.residuals >= profile.minimum_residual * (1.0 - 1e-9))


def test_profiling_an_unknown_parameter_names_what_is_available() -> None:
    runs, targets = _dataset(varying_ph=True)

    with pytest.raises(KeyError, match="theta_pH"):
        uncertainty.profile_likelihood(runs, targets, "K_G", "M2", ["ph_response"])


# --- bootstrap ------------------------------------------------------------------------------


def test_the_bootstrap_brackets_the_truth_when_the_fit_is_precise() -> None:
    runs, targets = _dataset(varying_ph=True, noise=0.001)

    result = uncertainty.bootstrap_parameters(
        runs, targets, "M2", ["ph_response"], resamples=30, random_seed=1
    )
    _median, low, high = result.interval("alpha")

    assert low < 20.0 < high


def test_the_bootstrap_centres_on_the_fit_rather_than_on_the_truth() -> None:
    """A real limitation, asserted so it is not mistaken for a bug later.

    A bootstrap characterises how much an estimate moves when the *data* are resampled. It
    does not correct bias in the estimate itself. In a collinear problem the fit can sit
    away from the generating value along the ridge -- here 2% noise moves alpha from 20 to
    about 22.5 -- and the bootstrap will faithfully report a tight interval around 22.5, not
    around 20. So a narrow bootstrap interval means 'stable under resampling', never
    'close to the truth'.
    """
    runs, targets = _dataset(varying_ph=True, noise=0.02)
    point, _diagnostics = model.fit(runs, targets, "M2", ["ph_response"])

    result = uncertainty.bootstrap_parameters(
        runs, targets, "M2", ["ph_response"], resamples=30, random_seed=1
    )
    median, low, high = result.interval("alpha")

    assert low <= median <= high
    assert median == pytest.approx(point.alpha, rel=0.15)


def test_more_noise_widens_the_bootstrap_interval() -> None:
    quiet, quiet_targets = _dataset(varying_ph=True, noise=0.01, seed=3)
    loud, loud_targets = _dataset(varying_ph=True, noise=0.15, seed=3)

    quiet_result = uncertainty.bootstrap_parameters(
        quiet, quiet_targets, "M2", ["ph_response"], resamples=30, random_seed=1
    )
    loud_result = uncertainty.bootstrap_parameters(
        loud, loud_targets, "M2", ["ph_response"], resamples=30, random_seed=1
    )

    _median, quiet_low, quiet_high = quiet_result.interval("alpha")
    _median, loud_low, loud_high = loud_result.interval("alpha")
    assert (loud_high - loud_low) > (quiet_high - quiet_low)


def test_the_bootstrap_is_reproducible_under_a_fixed_seed() -> None:
    runs, targets = _dataset(varying_ph=True)

    first = uncertainty.bootstrap_parameters(
        runs, targets, "M2", ["ph_response"], resamples=15, random_seed=7
    )
    second = uncertainty.bootstrap_parameters(
        runs, targets, "M2", ["ph_response"], resamples=15, random_seed=7
    )

    np.testing.assert_allclose(first.draws, second.draws)


def test_the_correlation_matrix_covers_every_parameter_pair() -> None:
    runs, targets = _dataset(varying_ph=True)

    result = uncertainty.bootstrap_parameters(
        runs, targets, "M2", ["ph_response"], resamples=25, random_seed=1
    )
    correlation = result.correlation()

    assert correlation.shape == (len(result.parameter_names), len(result.parameter_names))
    np.testing.assert_allclose(np.diag(correlation), 1.0, atol=1e-9)


# --- prediction intervals --------------------------------------------------------------------


def test_a_prediction_interval_is_wider_than_parameter_uncertainty_alone() -> None:
    """The distinction the module exists to enforce. Runs scatter around the model for
    reasons beyond parameter ignorance, so the predictive interval must be the wider one."""
    runs, targets = _dataset(varying_ph=True, noise=0.05)
    result = uncertainty.bootstrap_parameters(
        runs, targets, "M2", ["ph_response"], resamples=40, random_seed=1
    )
    # Relative residuals: fractions of the prediction, not titre units.
    residuals = np.array([-0.20, -0.08, 0.04, 0.09, 0.21])

    parameter_only = np.array(
        [fitted.predict_many(runs) for fitted in result.models], dtype=np.float64
    )
    narrow = np.quantile(parameter_only, 0.95, axis=0) - np.quantile(parameter_only, 0.05, axis=0)
    lower, _median, upper = uncertainty.prediction_intervals(result, runs, residuals)

    assert float(np.mean(upper - lower)) > float(np.mean(narrow))


def test_larger_residual_scatter_widens_the_prediction_interval() -> None:
    runs, targets = _dataset(varying_ph=True)
    result = uncertainty.bootstrap_parameters(
        runs, targets, "M2", ["ph_response"], resamples=25, random_seed=1
    )

    tight = uncertainty.prediction_intervals(result, runs, np.array([-0.01, 0.0, 0.01]))
    loose = uncertainty.prediction_intervals(result, runs, np.array([-0.60, 0.0, 0.60]))

    assert float(np.mean(loose[2] - loose[0])) > float(np.mean(tight[2] - tight[0]))


def test_the_interval_scales_with_the_prediction() -> None:
    """The property the relative residual buys, and the whole reason for the change.

    One pooled residual set must produce a *proportionally* wider interval for a run
    predicted to make a lot of titre than for one predicted to make little. Adding absolute
    residuals gives every run the same width, which is what under-covered the large-titre
    runs by 40 points.
    """
    runs, targets = _dataset(varying_ph=True)
    result = uncertainty.bootstrap_parameters(
        runs, targets, "M2", ["ph_response"], resamples=25, random_seed=1
    )
    residuals = np.array([-0.15, 0.0, 0.15])

    lower, median, upper = uncertainty.prediction_intervals(result, runs, residuals)
    width = upper - lower

    # The property, stated over every run rather than over a hand-picked pair: width per unit
    # of predicted titre is near constant. That is what "relative" means, and it is what an
    # additive pool cannot produce -- there, width would be constant and the *ratio* would
    # vary inversely with the prediction.
    relative_width = width / median
    assert float(np.std(relative_width) / np.mean(relative_width)) < 0.10

    # And width rises with the prediction. Not asserted as strictly monotone: parameter
    # uncertainty contributes a component that is not proportional to the prediction, so
    # neighbouring runs can invert. The trend across the range is the claim.
    order = np.argsort(median)
    third = max(len(order) // 3, 1)
    assert float(np.mean(width[order[-third:]])) > float(np.mean(width[order[:third]]))


def test_prediction_intervals_need_residuals() -> None:
    runs, targets = _dataset(varying_ph=True)
    result = uncertainty.bootstrap_parameters(
        runs, targets, "M2", ["ph_response"], resamples=10, random_seed=1
    )

    with pytest.raises(ValueError, match="at least one residual"):
        uncertainty.prediction_intervals(result, runs, np.zeros(0))


def test_coverage_counts_outcomes_inside_their_interval() -> None:
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    lower = np.array([0.0, 0.0, 10.0, 0.0])
    upper = np.array([2.0, 3.0, 20.0, 5.0])

    assert uncertainty.coverage(actual, lower, upper) == pytest.approx(0.75)


def test_coverage_is_zero_when_every_outcome_misses() -> None:
    actual = np.array([5.0, 6.0])

    assert uncertainty.coverage(actual, np.array([0.0, 0.0]), np.array([1.0, 1.0])) == 0.0


# --- against the real data ---------------------------------------------------------------------


def test_out_of_fold_residuals_are_larger_than_in_sample_ones(
    train_runs: list[ExperimentRun],
    train_targets: dict[str, float],
) -> None:
    """Why the prediction interval must use out-of-fold residuals: in-sample ones are
    shrunk by the fitting itself and would give intervals that are too narrow."""
    mechanisms = ("ph_response",)
    residuals = uncertainty.out_of_fold_residuals(
        train_runs, train_targets, "M2", mechanisms, fold_count=5
    )
    fitted, _diagnostics = model.fit(train_runs, train_targets, "M2", mechanisms)
    actual = np.array([train_targets[run.experiment_id] for run in train_runs])
    predicted = fitted.predict_many(train_runs)
    # Compared on the same scale: both relative, since out_of_fold_residuals returns fractions.
    in_sample = (actual - predicted) / predicted

    assert float(np.std(residuals)) > float(np.std(in_sample))


def test_out_of_fold_residuals_are_relative_and_dimensionless(
    train_runs: list[ExperimentRun],
    train_targets: dict[str, float],
) -> None:
    """They are fractions of the prediction, not titre units. Titres here run into the
    thousands, so an absolute residual pool would be off by three orders of magnitude when
    multiplied into a prediction."""
    residuals = uncertainty.out_of_fold_residuals(
        train_runs,
        train_targets,
        "M2",
        ("glutamine_limitation", "glucose_limitation"),
        fold_count=5,
    )

    assert np.all(np.abs(residuals) < 5.0), "a relative residual of 500% would be extraordinary"
    assert float(np.mean(np.abs(residuals))) < 1.0


def test_relative_residuals_cover_the_long_runs_far_better_than_absolute_ones(
    train_runs: list[ExperimentRun],
    train_targets: dict[str, float],
) -> None:
    """The measurement that motivated using relative residuals, asserted so it cannot regress.

    The titre error on this data is multiplicative: across the duration split the mean
    absolute residual nearly triples while the mean relative residual barely moves. Pooling
    absolute residuals therefore under-covers the large-titre 14-day runs badly -- and those
    are the runs the task is about.

    Compared at the residual level, without the bootstrap, so the assertion is about the error
    model rather than about parameter uncertainty.
    """
    mechanisms = ("glutamine_limitation", "glucose_limitation")

    def fit_fold(runs: list[ExperimentRun], targets: dict[str, float]) -> model.LuedekingPiretModel:
        fitted, _diagnostics = model.fit(runs, targets, "M2", mechanisms)
        return fitted

    predicted = evaluation.cross_validated_predictions(train_runs, train_targets, fit_fold, 10, 0)
    actual = np.array([train_targets[run.experiment_id] for run in train_runs])
    is_long = np.array([run.duration_days > 10.0 for run in train_runs])
    relative = (actual - predicted) / predicted
    absolute = actual - predicted

    # The premise: absolute error grows with run size, relative error much less so.
    assert np.mean(np.abs(absolute[is_long])) > 2.0 * np.mean(np.abs(absolute[~is_long]))
    assert np.mean(np.abs(relative[is_long])) < 1.6 * np.mean(np.abs(relative[~is_long]))

    def long_run_coverage(lower: np.ndarray, upper: np.ndarray) -> float:
        return uncertainty.coverage(actual[is_long], lower[is_long], upper[is_long])

    absolute_coverage = long_run_coverage(
        predicted + np.percentile(absolute, 5.0), predicted + np.percentile(absolute, 95.0)
    )
    relative_coverage = long_run_coverage(
        predicted * (1.0 + np.percentile(relative, 5.0)),
        predicted * (1.0 + np.percentile(relative, 95.0)),
    )

    assert relative_coverage >= absolute_coverage + 0.2, (
        f"relative residuals covered {relative_coverage:.0%} of long runs against "
        f"{absolute_coverage:.0%} for absolute; the gain is the reason for the change"
    )


def test_a_non_positive_out_of_fold_prediction_is_rejected_rather_than_divided_by(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A relative residual is undefined against a non-positive prediction, so the guard must
    name the offending run rather than silently return an infinity that would poison every
    interval built from the pool.

    The prediction is forced rather than found: on the supplied data even M0, which regresses
    on net growth and is the natural candidate, stays positive out of fold (minimum 94.8). A
    guard that cannot be reached by any real input still has to be correct, and stubbing the
    cross-validation is the only way to reach it deterministically.
    """
    runs, targets = _dataset(varying_ph=True, run_count=6)
    forced = np.array([100.0, 200.0, 0.0, 300.0, 400.0, 500.0])
    monkeypatch.setattr(
        uncertainty.evaluation,
        "cross_validated_predictions",
        lambda *args, **kwargs: forced,
    )

    with pytest.raises(ValueError, match="strictly positive"):
        uncertainty.out_of_fold_residuals(runs, targets, "M2", ["ph_response"], fold_count=3)
