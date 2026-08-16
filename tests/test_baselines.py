"""Tests for the data-driven benchmarks.

The tests that carry weight beyond ordinary coverage:

* **the benchmark is not accidentally handicapped.** A comparator that loses because it was
  given fewer features, or a leaky standardisation, or a column list computed at the wrong
  moment, proves nothing about the mechanistic model. Most of these tests are checks that the
  benchmark is being treated fairly;
* **nothing is fitted on data the fold should not see.** The column list, the standardisation
  and the PLS component count are all fitted quantities, and all three are easy to compute
  once on everything by accident;
* **the extrapolation behaviour is what the docs claim.** Trees saturate outside their
  training range and linear models do not. Those two statements are the argument the duration
  split rests on, so they are asserted rather than described.
"""

import numpy as np
import pytest

from titre_predictor import baselines, evaluation, features
from titre_predictor.data import schema
from titre_predictor.domain import ExperimentRun

DESIGN_DEFAULTS = {
    schema.DESIGN_DISSOLVED_OXYGEN: 50.0,
    schema.DESIGN_STIRRING: 200.0,
    schema.DESIGN_FEED_RATE_GLUCOSE: 4.0,
    schema.DESIGN_FEED_RATE_GLUTAMINE: 7.0,
    schema.DESIGN_FEED_START: 3.0,
    schema.DESIGN_FEED_END: 6.0,
    schema.DESIGN_TEMPERATURE_SHIFT: 4.0,
    schema.DESIGN_PH_SHIFT: 5.0,
}


def _run(identifier: str, peak: float, point_count: int = 9) -> ExperimentRun:
    """A synthetic run whose scale is set by ``peak``, so titre can be made to track it."""
    timestamps = np.arange(point_count, dtype=np.float64)
    rise = np.linspace(1.0, peak, point_count // 2 + 1)
    fall = np.linspace(peak * 0.9, peak * 0.4, point_count - rise.size)
    viable = np.concatenate([rise, fall])
    return ExperimentRun(
        experiment_id=identifier,
        timestamps=timestamps,
        design_scalars=dict(DESIGN_DEFAULTS),
        control_profiles={
            schema.CONTROL_TEMPERATURE: np.full(point_count, 37.0 - 0.01 * peak),
            schema.CONTROL_PH: np.full(point_count, 7.0),
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


def _training_set(count: int = 24) -> tuple[list[ExperimentRun], dict[str, float]]:
    """Runs whose titre is a clean linear function of cell-days, so a fit must find it."""
    peaks = np.linspace(8.0, 40.0, count)
    runs = [_run(f"Exp {index}", float(peak)) for index, peak in enumerate(peaks)]
    targets = {
        run.experiment_id: 12.0 * features.run_quantities(run).cell_days + 100.0 for run in runs
    }
    return runs, targets


# --- the feature block ---------------------------------------------------------------------


def test_a_feature_undefined_for_any_training_run_is_dropped_not_imputed() -> None:
    """Event-conditional features go missing in lockstep with duration, so filling them at fit
    time would encode duration into a column named after a process shift.

    The feed ends at day 6, so a run ending at day 4 never reaches it and ``vcd_at_FeedEnd``
    is undefined for that run alone. One such run must remove the column for everyone.
    """
    runs, _targets = _training_set(4)
    all_names, full_matrix = features.feature_frame(runs)
    assert "vcd_at_FeedEnd" in baselines.usable_feature_names(runs), "premise of the test"

    with_a_short_run = [*runs, _run("Exp short", 20.0, point_count=5)]
    names = baselines.usable_feature_names(with_a_short_run)

    assert "vcd_at_FeedEnd" not in names
    assert np.isfinite(baselines._feature_matrix(with_a_short_run, names)).all()
    assert len(names) < len(all_names) <= full_matrix.shape[1]


def test_a_feature_constant_across_the_training_fold_is_dropped() -> None:
    """A constant column carries no information and cannot be standardised; keeping it would
    mean dividing by a standard deviation of zero."""
    runs, _targets = _training_set(4)

    names = baselines.usable_feature_names(runs)

    # Every run shares the design scalars, so those columns are constant by construction.
    assert "DO" not in names
    assert "Stir" not in names


def test_the_column_list_is_fitted_and_carried_rather_than_recomputed() -> None:
    """A benchmark that recomputed its columns at prediction time could apply coefficients to
    a different set of features than it was fitted on."""
    runs, targets = _training_set()

    fitted = baselines.fit_partial_least_squares(runs, targets)

    assert fitted.feature_names == baselines.usable_feature_names(runs)
    assert fitted.centre.shape == (len(fitted.feature_names),)
    assert fitted.scale.shape == (len(fitted.feature_names),)


# --- fairness of the comparison ------------------------------------------------------------


def test_every_baseline_sees_the_same_features() -> None:
    """A difference between two benchmarks must be a difference of estimator, not of input."""
    runs, targets = _training_set()

    fitted = [fitter(runs, targets) for fitter in baselines.BASELINES.values()]

    assert len({item.feature_names for item in fitted}) == 1


def test_the_baselines_are_given_more_features_than_the_kinetic_model_reads() -> None:
    """The comparison is only worth making if the baseline is not starved. The selected
    mechanistic model reads two metabolite series through ``F``; the baselines get every
    always-defined aggregate."""
    runs, _targets = _training_set()

    assert len(baselines.usable_feature_names(runs)) > 2


def test_standardisation_is_fitted_on_the_training_runs_only() -> None:
    """Standardising on all the data before splitting is the classic leak. The centre must be
    the training mean, so it must move when the training set changes."""
    runs, targets = _training_set()

    first = baselines.fit_partial_least_squares(runs[:12], targets)
    second = baselines.fit_partial_least_squares(runs[12:], targets)

    assert not np.allclose(first.centre, second.centre)


def test_the_component_count_is_selected_inside_the_training_set() -> None:
    """Choosing it once on everything would hand the baseline a leakage advantage the
    mechanistic model was never given."""
    runs, targets = _training_set()

    first = baselines.fit_partial_least_squares(runs, targets)
    reduced = baselines.fit_partial_least_squares(runs[:8], targets)

    assert "component" in first.detail
    # Eight runs cannot support the same number of latent directions as twenty-four.
    assert int(reduced.detail.split()[0]) <= int(first.detail.split()[0])


def test_the_component_count_never_exceeds_what_the_fold_supports() -> None:
    """With five runs, an eight-component PLS is not estimable and must not be requested."""
    runs, targets = _training_set(5)

    fitted = baselines.fit_partial_least_squares(runs, targets)

    assert int(fitted.detail.split()[0]) <= 3


# --- prediction ----------------------------------------------------------------------------


def test_pls_recovers_a_linear_relationship_it_is_given() -> None:
    """Titre built as ``12*cell_days + 100`` is inside PLS's hypothesis class, so failing here
    would mean the wiring is wrong rather than that the method is weak."""
    runs, targets = _training_set()

    fitted = baselines.fit_partial_least_squares(runs, targets)
    predicted = fitted.predict_many(runs)
    actual = np.array([targets[run.experiment_id] for run in runs])

    assert evaluation.mean_absolute_percentage_error(actual, predicted) < 5.0


def test_predict_many_preserves_run_order() -> None:
    runs, targets = _training_set()
    fitted = baselines.fit_partial_least_squares(runs, targets)

    forward = fitted.predict_many(runs)
    reversed_order = fitted.predict_many(list(reversed(runs)))

    assert forward == pytest.approx(reversed_order[::-1])


def test_predicting_one_run_agrees_with_predicting_many() -> None:
    runs, targets = _training_set()
    fitted = baselines.fit_partial_least_squares(runs, targets)

    assert fitted.predict(runs[3]) == pytest.approx(float(fitted.predict_many(runs)[3]))


def test_a_missing_value_at_prediction_time_is_counted_rather_than_silently_filled() -> None:
    """The fallback to a column mean is imputation, and a benchmark that imputes silently
    would report an unearned number.

    ``vcd_at_FeedEnd`` is defined across the training runs, so it survives into the column
    list; a run ending at day 4 never reaches the feed end, so predicting it needs the
    fallback. That is exactly the case the docs describe, made to happen rather than waited
    for.
    """
    runs, targets = _training_set()
    fitted = baselines.fit_partial_least_squares(runs, targets)
    assert "vcd_at_FeedEnd" in fitted.feature_names, "premise of the test"
    assert sum(fitted.imputed) == 0

    short = _run("Exp short", 20.0, point_count=5)
    assert not np.isfinite(baselines._feature_matrix([short], fitted.feature_names)).all()

    prediction = fitted.predict_many([short])

    assert sum(fitted.imputed) > 0
    assert np.isfinite(prediction).all(), "the fallback must produce a usable prediction"


# --- the extrapolation claim ---------------------------------------------------------------


def test_gradient_boosting_cannot_predict_beyond_its_training_targets() -> None:
    """The saturation the duration-split result turns on: a tree averages training targets in
    a leaf, so no prediction can exceed the largest one it was fitted on. This is why it
    under-predicts the highest-titre held-out run by a factor of nearly two."""
    runs, targets = _training_set()
    fitted = baselines.fit_gradient_boosting(runs, targets)
    largest = max(targets.values())

    # A run far larger than anything in training.
    beyond = [_run("Exp beyond", 400.0)]

    assert float(fitted.predict_many(beyond)[0]) <= largest


def test_pls_does_extrapolate_beyond_its_training_targets() -> None:
    """The complementary half: a linear model has no such ceiling, which is why it overshoots
    rather than saturating. The two failure modes are opposite, and both are failures."""
    runs, targets = _training_set()
    fitted = baselines.fit_partial_least_squares(runs, targets)
    largest = max(targets.values())

    beyond = [_run("Exp beyond", 400.0)]

    assert float(fitted.predict_many(beyond)[0]) > largest


# --- input validation ----------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(baselines.BASELINES))
def test_a_run_without_a_target_is_named(name: str) -> None:
    runs, targets = _training_set()
    del targets[runs[2].experiment_id]

    with pytest.raises(KeyError, match=runs[2].experiment_id):
        baselines.BASELINES[name](runs, targets)


@pytest.mark.parametrize("name", sorted(baselines.BASELINES))
def test_too_few_runs_to_fit_is_rejected(name: str) -> None:
    runs, targets = _training_set(2)

    with pytest.raises(ValueError, match="at least three runs"):
        baselines.BASELINES[name](runs, targets)


# --- the real data -------------------------------------------------------------------------


def test_both_baselines_beat_the_mean_on_random_folds(
    train_runs: list[ExperimentRun],
    train_targets: dict[str, float],
) -> None:
    """A benchmark that cannot beat the mean is not a benchmark. Both must clear it on the
    split where no extrapolation is required, or their loss on the duration split says
    nothing about extrapolation in particular."""
    from titre_predictor import model

    actual = np.array([train_targets[run.experiment_id] for run in train_runs])
    mean_predictions = evaluation.cross_validated_predictions(
        train_runs, train_targets, model.MeanTitreModel.fit, 5, 0
    )
    mean_error = evaluation.root_mean_squared_error(actual, mean_predictions)

    for name, fitter in baselines.BASELINES.items():
        predictions = evaluation.cross_validated_predictions(
            train_runs, train_targets, fitter, 5, 0
        )
        error = evaluation.root_mean_squared_error(actual, predictions)
        assert error < mean_error, f"{name} did not beat the mean baseline"


def test_the_duration_split_is_where_the_baselines_break(
    train_runs: list[ExperimentRun],
    train_targets: dict[str, float],
) -> None:
    """The measurement the whole benchmark exists to make: both baselines are far worse
    predicting ten 14-day runs from 90 short ones than they are on random folds, because
    eight of the ten sit beyond the training range of the dominant feature."""
    short_runs, long_runs = evaluation.split_by_duration(train_runs)
    long_actual = [train_targets[run.experiment_id] for run in long_runs]
    short_actual = np.array([train_targets[run.experiment_id] for run in short_runs])

    for name, fitter in baselines.BASELINES.items():
        random_folds = evaluation.cross_validated_predictions(
            short_runs, train_targets, fitter, 5, 0
        )
        within = evaluation.root_mean_squared_error(short_actual, random_folds)
        extrapolated = evaluation.root_mean_squared_error(
            long_actual, fitter(short_runs, train_targets).predict_many(long_runs)
        )
        assert extrapolated > within, f"{name} showed no extrapolation penalty"
