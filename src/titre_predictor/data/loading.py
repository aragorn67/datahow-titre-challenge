"""Loading the supplied CSV files into :class:`ExperimentRun` objects.

The files are in long format: one row per (experiment, day). The ``Z:`` design scalars
are written on the first row of each experiment and left blank on the rest — that is a
storage convention, not missing data, and a forward fill within each experiment
resolves it completely. There are no other blanks in either file.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from titre_predictor.data import schema
from titre_predictor.domain import ExperimentRun


def load_experiment_table(csv_path: Path) -> pd.DataFrame:
    """Read one raw input CSV without altering it.

    Args:
        csv_path: path to ``datahow_interview_{train,test}_data.csv``.
    """
    return pd.read_csv(csv_path)


def forward_fill_design_scalars(
    table: pd.DataFrame,
    experiment_column: str = schema.EXPERIMENT_COLUMN,  # "Exp"
    design_scalar_prefix: str = schema.DESIGN_SCALAR_PREFIX,  # "Z:"
) -> pd.DataFrame:
    """Propagate each run's design scalars from its first row to all its rows.

    Returns a copy; the input is not modified.

    Args:
        table: a table as returned by :func:`load_experiment_table`.
        experiment_column: column identifying the run.
        design_scalar_prefix: prefix marking the constant-per-run columns.

    Raises:
        ValueError: if any design scalar is still blank afterwards, which would mean
            the first row of some run was itself blank.
    """
    design_columns = schema.columns_with_prefix(list(table.columns), design_scalar_prefix)
    filled = table.copy()
    filled[design_columns] = filled.groupby(experiment_column)[design_columns].ffill()

    remaining_blanks = int(filled[design_columns].isna().sum().sum())
    if remaining_blanks:
        raise ValueError(
            f"{remaining_blanks} design scalar cells are still blank after forward fill; "
            "this means some experiment has no value on its first row"
        )
    return filled


def split_into_runs(
    table: pd.DataFrame,
    experiment_column: str = schema.EXPERIMENT_COLUMN,  # "Exp"
    time_column: str = schema.TIME_COLUMN,  # "Time[day]"
    design_scalar_prefix: str = schema.DESIGN_SCALAR_PREFIX,  # "Z:"
    control_profile_prefix: str = schema.CONTROL_PROFILE_PREFIX,  # "W:"
    observation_prefix: str = schema.OBSERVATION_PREFIX,  # "X:"
) -> list[ExperimentRun]:
    """Convert a forward-filled long table into one :class:`ExperimentRun` per experiment.

    Rows are sorted by time within each run, so the caller need not rely on file order.

    Args:
        table: output of :func:`forward_fill_design_scalars`.
        experiment_column: column identifying the run.
        time_column: column holding the sample time in days.
        design_scalar_prefix: prefix for constant-per-run columns.
        control_profile_prefix: prefix for time-varying control columns.
        observation_prefix: prefix for measured columns.

    Returns:
        Runs in order of first appearance in the table.
    """
    columns = list(table.columns)
    design_columns = schema.columns_with_prefix(columns, design_scalar_prefix)
    control_columns = schema.columns_with_prefix(columns, control_profile_prefix)
    observation_columns = schema.columns_with_prefix(columns, observation_prefix)

    runs: list[ExperimentRun] = []
    for experiment_id, group in table.groupby(experiment_column, sort=False):
        ordered = group.sort_values(time_column)
        runs.append(
            ExperimentRun(
                experiment_id=str(experiment_id),
                timestamps=ordered[time_column].to_numpy(dtype=np.float64),
                design_scalars={name: float(ordered[name].iloc[0]) for name in design_columns},
                control_profiles={
                    name: ordered[name].to_numpy(dtype=np.float64) for name in control_columns
                },
                observations={
                    name: ordered[name].to_numpy(dtype=np.float64) for name in observation_columns
                },
            )
        )
    return runs


def load_runs(
    csv_path: Path,
    experiment_column: str = schema.EXPERIMENT_COLUMN,  # "Exp"
    time_column: str = schema.TIME_COLUMN,  # "Time[day]"
) -> list[ExperimentRun]:
    """Read an input CSV and return its experiments, forward fill included.

    Args:
        csv_path: path to ``datahow_interview_{train,test}_data.csv``.
        experiment_column: column identifying the run.
        time_column: column holding the sample time in days.
    """
    table = load_experiment_table(csv_path)
    filled = forward_fill_design_scalars(table, experiment_column=experiment_column)
    return split_into_runs(filled, experiment_column=experiment_column, time_column=time_column)


def load_targets(
    csv_path: Path,
    experiment_column: str = schema.EXPERIMENT_COLUMN,  # "Exp"
    target_column: str = schema.TARGET_COLUMN,  # "Y:Titer"
) -> dict[str, float]:
    """Read a targets CSV into a mapping from experiment identifier to final titre.

    The targets files hold exactly one row per experiment, recorded at harvest.

    Args:
        csv_path: path to ``datahow_interview_{train,test}_targets*.csv``.
        experiment_column: column identifying the run.
        target_column: column holding the titre.

    Raises:
        ValueError: if any experiment appears more than once, which would mean the file
            is not the one-scalar-per-experiment format the task describes.
    """
    table = pd.read_csv(csv_path)
    duplicated = table[experiment_column].duplicated()
    if bool(duplicated.any()):
        repeated = sorted(table.loc[duplicated, experiment_column].unique())
        raise ValueError(f"targets file has repeated experiments: {repeated}")
    return {
        str(experiment_id): float(titre)
        for experiment_id, titre in zip(table[experiment_column], table[target_column], strict=True)
    }
