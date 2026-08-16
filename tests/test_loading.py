"""Tests for reading the supplied CSV files into experiment runs."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from titre_predictor.data import schema
from titre_predictor.data.loading import (
    forward_fill_design_scalars,
    load_experiment_table,
    load_targets,
    split_into_runs,
)
from titre_predictor.domain import ExperimentRun


def test_train_file_yields_one_hundred_runs(train_runs: list[ExperimentRun]) -> None:
    assert len(train_runs) == 100


def test_test_file_yields_twenty_runs(test_runs: list[ExperimentRun]) -> None:
    assert len(test_runs) == 20


def test_every_test_run_lasts_fourteen_days(test_runs: list[ExperimentRun]) -> None:
    """The shift the model must survive: all test runs reach a horizon that only ten
    of the hundred training runs do."""
    assert {run.duration_days for run in test_runs} == {14.0}


def test_training_durations_are_mostly_shorter_than_the_test_horizon(
    train_runs: list[ExperimentRun],
) -> None:
    durations = [run.duration_days for run in train_runs]
    assert sum(duration == 14.0 for duration in durations) == 10


def test_forward_fill_leaves_no_blank_design_scalars(train_data_path: Path) -> None:
    table = load_experiment_table(train_data_path)
    design_columns = schema.columns_with_prefix(list(table.columns), schema.DESIGN_SCALAR_PREFIX)
    assert table[design_columns].isna().to_numpy().any(), (
        "expected the raw file to carry design scalars only on each run's first row"
    )

    filled = forward_fill_design_scalars(table)
    assert not filled[design_columns].isna().to_numpy().any()


def test_forward_fill_does_not_mutate_its_input(train_data_path: Path) -> None:
    table = load_experiment_table(train_data_path)
    before = table.copy()
    forward_fill_design_scalars(table)
    pd.testing.assert_frame_equal(table, before)


def test_forward_fill_rejects_a_run_whose_first_row_is_blank() -> None:
    table = pd.DataFrame(
        {
            schema.EXPERIMENT_COLUMN: ["Exp 1", "Exp 1"],
            schema.TIME_COLUMN: [0.0, 1.0],
            schema.DESIGN_FEED_START: [np.nan, np.nan],
        }
    )
    with pytest.raises(ValueError, match="still blank after forward fill"):
        forward_fill_design_scalars(table)


def test_observations_and_controls_are_not_blank(train_runs: list[ExperimentRun]) -> None:
    """Only the design scalars use the blank-after-first-row convention."""
    for run in train_runs:
        for series in (*run.observations.values(), *run.control_profiles.values()):
            assert np.isfinite(series).all(), run.experiment_id


def test_runs_are_ordered_by_time_regardless_of_file_order(train_data_path: Path) -> None:
    table = forward_fill_design_scalars(load_experiment_table(train_data_path))
    shuffled = table.iloc[::-1].reset_index(drop=True)

    runs = split_into_runs(shuffled)

    for run in runs:
        assert np.all(np.diff(run.timestamps) > 0), run.experiment_id


def test_timestamps_form_an_exact_one_day_grid(train_runs: list[ExperimentRun]) -> None:
    for run in train_runs:
        assert run.timestamps[0] == 0.0
        np.testing.assert_array_equal(np.diff(run.timestamps), np.ones(run.timestamps.size - 1))


def test_duration_matches_the_declared_design_duration(train_runs: list[ExperimentRun]) -> None:
    """``Z:ExpDuration`` and the last observed timestamp agree for every run, so the
    two possible readings of 'harvest time' cannot disagree."""
    for run in train_runs:
        assert run.duration_days == run.design_scalars[schema.DESIGN_EXPERIMENT_DURATION]


def test_targets_are_one_scalar_per_experiment(train_targets: dict[str, float]) -> None:
    assert len(train_targets) == 100
    assert all(titre > 0 for titre in train_targets.values())


def test_every_training_run_has_a_target(
    train_runs: list[ExperimentRun], train_targets: dict[str, float]
) -> None:
    assert {run.experiment_id for run in train_runs} == set(train_targets)


def test_targets_reject_a_repeated_experiment(tmp_path: Path) -> None:
    csv_path = tmp_path / "targets.csv"
    pd.DataFrame(
        {
            schema.EXPERIMENT_COLUMN: ["Exp 1", "Exp 1"],
            schema.TARGET_COLUMN: [1000.0, 2000.0],
        }
    ).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="repeated experiments"):
        load_targets(csv_path)
