"""Tests that the loaded data is *exactly* the data in the files.

The tests in ``test_loading.py`` check structure: how many runs, what shape, what
order. These check content: that every number survives the journey from CSV cell to
:class:`ExperimentRun` unchanged, that nothing is rounded, and that no value goes
missing.

Two complementary strategies:

* **Exhaustive comparison** — every cell of every file is compared against a direct
  pandas read. This catches a column being misassigned, rows being misordered, or a
  run picking up its neighbour's design scalars, anywhere in the data rather than only
  where someone thought to look.
* **Hardcoded anchors** — a scatter of individual cells transcribed as literals. These
  catch something the exhaustive comparison cannot: the data files themselves being
  altered or replaced, since both sides of the exhaustive comparison would move
  together.

Every comparison uses *exact* equality rather than a tolerance. That is the point: it
demonstrates the loader rounds nothing.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from titre_predictor.data import schema
from titre_predictor.data.loading import load_runs, load_targets
from titre_predictor.domain import ExperimentRun

DATA_FILENAMES = [
    "datahow_interview_train_data.csv",
    "datahow_interview_test_data.csv",
]

# --- Exhaustive comparison against a direct read ------------------------------------


@pytest.mark.parametrize("filename", DATA_FILENAMES)
def test_every_cell_survives_loading_exactly(raw_data_directory: Path, filename: str) -> None:
    """Compare every time-varying cell of the file against the loaded runs.

    Exact equality, not ``allclose``: any rounding at all would fail this.
    """
    csv_path = raw_data_directory / filename
    raw = pd.read_csv(csv_path)
    runs = load_runs(csv_path)

    time_varying_columns = schema.columns_with_prefix(
        list(raw.columns), schema.CONTROL_PROFILE_PREFIX
    ) + schema.columns_with_prefix(list(raw.columns), schema.OBSERVATION_PREFIX)

    compared_cells = 0
    for run in runs:
        raw_rows = raw[raw[schema.EXPERIMENT_COLUMN] == run.experiment_id].sort_values(
            schema.TIME_COLUMN
        )

        np.testing.assert_array_equal(
            run.timestamps,
            raw_rows[schema.TIME_COLUMN].to_numpy(dtype=np.float64),
            err_msg=f"{run.experiment_id}: timestamps",
        )

        for column in time_varying_columns:
            loaded = (
                run.control_profiles[column]
                if column.startswith(schema.CONTROL_PROFILE_PREFIX)
                else run.observations[column]
            )
            np.testing.assert_array_equal(
                loaded,
                raw_rows[column].to_numpy(dtype=np.float64),
                err_msg=f"{run.experiment_id}: {column}",
            )
            compared_cells += loaded.size

    expected_cells = len(raw) * len(time_varying_columns)
    assert compared_cells == expected_cells, "not every cell was actually compared"


@pytest.mark.parametrize("filename", DATA_FILENAMES)
def test_design_scalars_come_from_the_run_s_own_first_row(
    raw_data_directory: Path, filename: str
) -> None:
    """Guards against the forward fill leaking a neighbouring run's design across a
    run boundary, which would silently attach the wrong process conditions."""
    csv_path = raw_data_directory / filename
    raw = pd.read_csv(csv_path)
    runs = load_runs(csv_path)

    design_columns = schema.columns_with_prefix(list(raw.columns), schema.DESIGN_SCALAR_PREFIX)

    for run in runs:
        first_raw_row = (
            raw[raw[schema.EXPERIMENT_COLUMN] == run.experiment_id]
            .sort_values(schema.TIME_COLUMN)
            .iloc[0]
        )
        for column in design_columns:
            assert run.design_scalars[column] == first_raw_row[column], (
                f"{run.experiment_id}: {column}"
            )


# --- Hardcoded anchors --------------------------------------------------------------

# (experiment, day, column, exact value) transcribed from the raw CSV files.
TRAIN_CELL_ANCHORS = [
    ("Exp 5", 3.0, schema.CONTROL_FEED_GLUTAMINE, 6.232323232),
    ("Exp 12", 14.0, schema.OBSERVATION_VIABLE_CELL_DENSITY, 22.61162022),
    ("Exp 37", 5.0, schema.OBSERVATION_LACTATE, 5.832107664),
    ("Exp 63", 0.0, "Z:Stir", 233.3333333),
    ("Exp 88", 2.0, schema.OBSERVATION_GLUTAMINE, 8.674693547),
    ("Exp 100", 7.0, schema.OBSERVATION_LYSED_CELLS, 0.040305943),
]

TEST_CELL_ANCHORS = [
    ("Test Exp 2", 11.0, schema.CONTROL_PH, 7.210526316),
    ("Test Exp 7", 0.0, "Z:DO", 33.94736842),
    ("Test Exp 13", 9.0, schema.OBSERVATION_AMMONIA, 4.280164469),
    ("Test Exp 20", 14.0, schema.OBSERVATION_GLUCOSE, 0.520453614),
]

TRAIN_TARGET_ANCHORS = [
    ("Exp 12", 3953.18847),
    ("Exp 37", 979.8313018),
    ("Exp 88", 1547.253264),
    ("Exp 100", 948.5983025),
]


def _value_at(run: ExperimentRun, day: float, column: str) -> float:
    index = int(np.flatnonzero(run.timestamps == day)[0])
    if column.startswith(schema.DESIGN_SCALAR_PREFIX):
        return run.design_scalars[column]
    if column.startswith(schema.CONTROL_PROFILE_PREFIX):
        return float(run.control_profiles[column][index])
    return float(run.observations[column][index])


@pytest.mark.parametrize(("experiment_id", "day", "column", "expected"), TRAIN_CELL_ANCHORS)
def test_training_cell_anchor(
    train_runs: list[ExperimentRun],
    experiment_id: str,
    day: float,
    column: str,
    expected: float,
) -> None:
    run = next(candidate for candidate in train_runs if candidate.experiment_id == experiment_id)
    assert _value_at(run, day, column) == expected


@pytest.mark.parametrize(("experiment_id", "day", "column", "expected"), TEST_CELL_ANCHORS)
def test_test_cell_anchor(
    test_runs: list[ExperimentRun],
    experiment_id: str,
    day: float,
    column: str,
    expected: float,
) -> None:
    run = next(candidate for candidate in test_runs if candidate.experiment_id == experiment_id)
    assert _value_at(run, day, column) == expected


@pytest.mark.parametrize(("experiment_id", "expected"), TRAIN_TARGET_ANCHORS)
def test_training_target_anchor(
    train_targets: dict[str, float], experiment_id: str, expected: float
) -> None:
    assert train_targets[experiment_id] == expected


@pytest.mark.parametrize("filename", DATA_FILENAMES)
def test_loaded_values_equal_the_raw_text_of_the_file(
    raw_data_directory: Path, filename: str
) -> None:
    """Compare against the file's own characters, not against another pandas read.

    The exhaustive test above compares the loader with ``pd.read_csv``; if pandas
    itself rounded, both sides would move together and the test would still pass. Here
    the CSV is read as *strings* and converted with Python's own float parser, so the
    comparison is against the text on disk. The files carry ten significant figures and
    float64 holds about fifteen, so every digit must survive.
    """
    csv_path = raw_data_directory / filename
    as_text = pd.read_csv(csv_path, dtype=str)
    runs = {run.experiment_id: run for run in load_runs(csv_path)}

    observation_columns = schema.columns_with_prefix(
        list(as_text.columns), schema.OBSERVATION_PREFIX
    )

    for _, text_row in as_text.iterrows():
        run = runs[text_row[schema.EXPERIMENT_COLUMN]]
        index = int(np.flatnonzero(run.timestamps == float(text_row[schema.TIME_COLUMN]))[0])
        for column in observation_columns:
            assert run.observations[column][index] == float(text_row[column]), (
                f"{run.experiment_id} day {text_row[schema.TIME_COLUMN]}: {column}"
            )


# --- Missing values -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("filename", "expected_design_blanks"),
    [
        ("datahow_interview_train_data.csv", 11570),
        ("datahow_interview_test_data.csv", 3640),
    ],
)
def test_blanks_appear_only_in_design_scalars(
    raw_data_directory: Path, filename: str, expected_design_blanks: int
) -> None:
    """In the raw files, blanks are the design-scalar storage convention and nothing
    else. The exact counts are pinned so that a changed file is noticed."""
    raw = pd.read_csv(raw_data_directory / filename)

    design_columns = schema.columns_with_prefix(list(raw.columns), schema.DESIGN_SCALAR_PREFIX)
    control_columns = schema.columns_with_prefix(list(raw.columns), schema.CONTROL_PROFILE_PREFIX)
    observation_columns = schema.columns_with_prefix(list(raw.columns), schema.OBSERVATION_PREFIX)

    assert int(raw[design_columns].isna().sum().sum()) == expected_design_blanks
    assert int(raw[control_columns].isna().sum().sum()) == 0
    assert int(raw[observation_columns].isna().sum().sum()) == 0
    assert int(raw[schema.TIME_COLUMN].isna().sum()) == 0
    assert int(raw[schema.EXPERIMENT_COLUMN].isna().sum()) == 0


@pytest.mark.parametrize("filename", DATA_FILENAMES)
def test_no_missing_values_survive_into_the_loaded_runs(
    raw_data_directory: Path, filename: str
) -> None:
    for run in load_runs(raw_data_directory / filename):
        assert np.isfinite(run.timestamps).all(), run.experiment_id
        for series in (*run.control_profiles.values(), *run.observations.values()):
            assert np.isfinite(series).all(), run.experiment_id
        for name, value in run.design_scalars.items():
            assert np.isfinite(value), f"{run.experiment_id}: {name}"


def test_training_targets_have_no_missing_values(train_targets: dict[str, float]) -> None:
    assert all(np.isfinite(titre) for titre in train_targets.values())


# --- The placeholder targets file ---------------------------------------------------


def test_test_targets_are_still_the_placeholder(test_targets_template_path: Path) -> None:
    """The supplied test targets are all 2000, a placeholder. Real values arrive at
    interview time; when they are dropped in, this test fails and is the reminder to
    wire them into the evaluation."""
    targets = load_targets(test_targets_template_path)

    assert len(targets) == 20
    assert set(targets.values()) == {2000.0}


def test_test_target_experiments_match_the_test_runs(
    test_runs: list[ExperimentRun], test_targets_template_path: Path
) -> None:
    """The real targets will arrive in this same format, so the join must already work."""
    targets = load_targets(test_targets_template_path)
    assert {run.experiment_id for run in test_runs} == set(targets)
