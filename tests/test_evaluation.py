"""Tests for metrics and splits.

Metrics are checked against hand-computed values rather than against each other, so a
sign error or a misplaced square cannot pass by being consistently wrong.
"""

import numpy as np
import pytest

from titre_predictor import evaluation as ev
from titre_predictor.domain import ExperimentRun

# --- metrics ------------------------------------------------------------------------


def test_rmse_of_a_perfect_prediction_is_zero() -> None:
    assert ev.root_mean_squared_error([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0


def test_rmse_matches_a_hand_computed_value() -> None:
    # errors 3 and 4 -> mean square (9 + 16)/2 = 12.5 -> sqrt = 3.5355...
    assert ev.root_mean_squared_error([10.0, 20.0], [13.0, 16.0]) == pytest.approx(np.sqrt(12.5))


def test_rmse_punishes_one_large_miss_more_than_mae() -> None:
    """Why both are reported: they rank models differently."""
    actual = [100.0, 100.0, 100.0, 100.0]
    one_big_miss = [100.0, 100.0, 100.0, 140.0]
    spread_out = [110.0, 110.0, 110.0, 110.0]

    assert ev.mean_absolute_error(actual, one_big_miss) == ev.mean_absolute_error(
        actual, spread_out
    )
    assert ev.root_mean_squared_error(actual, one_big_miss) > ev.root_mean_squared_error(
        actual, spread_out
    )


def test_mae_matches_a_hand_computed_value() -> None:
    assert ev.mean_absolute_error([10.0, 20.0], [13.0, 16.0]) == pytest.approx(3.5)


def test_mape_matches_a_hand_computed_value() -> None:
    # |10-11|/10 = 10%, |200-180|/200 = 10% -> mean 10%
    assert ev.mean_absolute_percentage_error([10.0, 200.0], [11.0, 180.0]) == pytest.approx(10.0)


def test_mape_weights_small_and_large_targets_equally() -> None:
    """The reason it disagrees with RMSE on this dataset: the test runs are all in the
    upper tail, so RMSE and MAPE reward different things."""
    small_miss_on_small_target = ev.mean_absolute_percentage_error([10.0], [11.0])
    large_miss_on_large_target = ev.mean_absolute_percentage_error([1000.0], [1100.0])

    assert small_miss_on_small_target == pytest.approx(large_miss_on_large_target)


def test_mape_rejects_a_zero_target() -> None:
    with pytest.raises(ValueError, match="undefined where the actual value is zero"):
        ev.mean_absolute_percentage_error([0.0, 5.0], [1.0, 5.0])


def test_r_squared_of_the_mean_predictor_is_zero() -> None:
    actual = [1.0, 2.0, 3.0, 4.0]
    mean_prediction = [2.5, 2.5, 2.5, 2.5]

    assert ev.coefficient_of_determination(actual, mean_prediction) == pytest.approx(0.0)


def test_r_squared_is_negative_when_worse_than_the_mean() -> None:
    assert ev.coefficient_of_determination([1.0, 2.0, 3.0], [10.0, 10.0, 10.0]) < 0.0


def test_r_squared_of_a_perfect_prediction_is_one() -> None:
    assert ev.coefficient_of_determination([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_r_squared_is_undefined_for_a_constant_target() -> None:
    with pytest.raises(ValueError, match="every actual value is identical"):
        ev.coefficient_of_determination([5.0, 5.0], [4.0, 6.0])


def test_metrics_reject_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same shape"):
        ev.root_mean_squared_error([1.0, 2.0], [1.0])


def test_metrics_reject_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one observation"):
        ev.mean_absolute_error([], [])


# --- duration split -----------------------------------------------------------------


def test_duration_split_reproduces_the_real_shift(train_runs: list[ExperimentRun]) -> None:
    short_runs, long_runs = ev.split_by_duration(train_runs)

    assert len(short_runs) == 90
    assert len(long_runs) == 10
    assert {run.duration_days for run in long_runs} == {14.0}


def test_duration_split_covers_every_run_exactly_once(train_runs: list[ExperimentRun]) -> None:
    short_runs, long_runs = ev.split_by_duration(train_runs)

    identifiers = [run.experiment_id for run in (*short_runs, *long_runs)]
    assert sorted(identifiers) == sorted(run.experiment_id for run in train_runs)


def test_duration_threshold_is_inclusive_on_the_short_side(
    train_runs: list[ExperimentRun],
) -> None:
    short_runs, _ = ev.split_by_duration(train_runs, maximum_training_duration_days=10.0)

    assert max(run.duration_days for run in short_runs) == 10.0


def test_duration_split_rejects_a_threshold_that_empties_one_side(
    train_runs: list[ExperimentRun],
) -> None:
    with pytest.raises(ValueError, match="leaves one side empty"):
        ev.split_by_duration(train_runs, maximum_training_duration_days=100.0)


# --- k-fold -------------------------------------------------------------------------


def test_folds_partition_the_samples() -> None:
    folds = ev.k_fold_indices(sample_count=20, fold_count=5)

    held_out = np.concatenate([test_indices for _, test_indices in folds])
    assert sorted(held_out.tolist()) == list(range(20))


def test_each_sample_is_held_out_exactly_once() -> None:
    folds = ev.k_fold_indices(sample_count=17, fold_count=4)

    held_out = np.concatenate([test_indices for _, test_indices in folds])
    assert len(set(held_out.tolist())) == 17


def test_train_and_test_indices_never_overlap() -> None:
    for train_indices, test_indices in ev.k_fold_indices(sample_count=23, fold_count=5):
        assert not set(train_indices.tolist()) & set(test_indices.tolist())
        assert len(train_indices) + len(test_indices) == 23


def test_folds_are_reproducible_for_a_fixed_seed() -> None:
    first = ev.k_fold_indices(sample_count=20, fold_count=5, random_seed=7)
    second = ev.k_fold_indices(sample_count=20, fold_count=5, random_seed=7)

    for (_, a), (_, b) in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)


def test_a_different_seed_gives_a_different_partition() -> None:
    first = ev.k_fold_indices(sample_count=20, fold_count=5, random_seed=1)
    second = ev.k_fold_indices(sample_count=20, fold_count=5, random_seed=2)

    assert any(not np.array_equal(a, b) for (_, a), (_, b) in zip(first, second, strict=True))


def test_k_fold_rejects_fewer_than_two_folds() -> None:
    with pytest.raises(ValueError, match="at least two folds"):
        ev.k_fold_indices(sample_count=10, fold_count=1)


def test_k_fold_rejects_more_folds_than_samples() -> None:
    with pytest.raises(ValueError, match="cannot make 11 folds from 10 samples"):
        ev.k_fold_indices(sample_count=10, fold_count=11)


# --- specific productivity ------------------------------------------------------------


def _productivity_run(identifier: str, height: float, point_count: int = 9) -> ExperimentRun:
    timestamps = np.arange(point_count, dtype=np.float64)
    return ExperimentRun(
        experiment_id=identifier,
        timestamps=timestamps,
        design_scalars={},
        control_profiles={},
        observations={
            "X:VCD": np.full(point_count, height),
            "X:Lysed": np.zeros(point_count),
        },
    )


def test_specific_productivity_is_titre_divided_by_cell_days() -> None:
    """An exact identity, not an approximation: P = qbar_P * gammaX by definition of the
    cell-weighted average."""
    run = _productivity_run("Exp 1", height=2.0)  # 8 days at VCD 2 = 16 cell-days

    productivity = ev.specific_productivity_targets([run], {"Exp 1": 800.0})

    assert productivity["Exp 1"] == pytest.approx(50.0)


def test_specific_productivity_is_intensive() -> None:
    """Two runs at the same productivity but different scale must give the same value --
    that is what makes it the screening target rather than titre itself."""
    small = _productivity_run("Exp 1", height=2.0)
    large = _productivity_run("Exp 2", height=8.0)

    productivity = ev.specific_productivity_targets(
        [small, large], {"Exp 1": 800.0, "Exp 2": 3200.0}
    )

    assert productivity["Exp 1"] == pytest.approx(productivity["Exp 2"])


def test_specific_productivity_names_a_run_without_a_target() -> None:
    with pytest.raises(KeyError, match="Exp 1"):
        ev.specific_productivity_targets([_productivity_run("Exp 1", 2.0)], {})


def test_specific_productivity_rejects_a_run_with_no_cell_days() -> None:
    """The ratio is undefined there, and a zero denominator would return inf rather than
    failing loudly."""
    run = _productivity_run("Exp 1", height=0.0)

    with pytest.raises(ValueError, match="undefined"):
        ev.specific_productivity_targets([run], {"Exp 1": 800.0})
