"""Metrics and data splits for judging the titre model.

Choice of metric
----------------
Titres in the training set span roughly 283 to 4823 -- a factor of seventeen -- and the
test set sits in the upper tail, since every test run lasts 14 days. RMSE and a relative
metric therefore disagree, and each answers a different question:

* ``root_mean_squared_error`` is in the units of the target and is dominated by the
  high-titre runs;
* ``mean_absolute_percentage_error`` weights every run equally regardless of size, so it
  is dominated by the low-titre runs.

Both are reported. A model that improves one while worsening the other has not improved;
it has changed which runs it is good at, and that is worth knowing rather than hiding
behind a single headline number.

Choice of split
---------------
The supplied test set is entirely 14-day runs, while only ten of the hundred training
runs reach that horizon. A random split would test the model on the same durations it
was trained on and report a number that says nothing about the real task.
``split_by_duration`` reproduces the shift instead.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from titre_predictor import features
from titre_predictor.data import schema
from titre_predictor.domain import ExperimentRun

# Runs longer than this are held out by split_by_duration.
DEFAULT_MAXIMUM_TRAINING_DURATION_DAYS = 10.0  # 7 | 8 | 9 | 10


def _as_paired_arrays(
    actual: Sequence[float] | NDArray[np.float64],
    predicted: Sequence[float] | NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    actual_array = np.asarray(actual, dtype=np.float64)
    predicted_array = np.asarray(predicted, dtype=np.float64)
    if actual_array.shape != predicted_array.shape:
        raise ValueError(
            f"actual and predicted must have the same shape, got "
            f"{actual_array.shape} and {predicted_array.shape}"
        )
    if actual_array.size == 0:
        raise ValueError("need at least one observation to compute a metric")
    return actual_array, predicted_array


def root_mean_squared_error(
    actual: Sequence[float] | NDArray[np.float64],
    predicted: Sequence[float] | NDArray[np.float64],
) -> float:
    """Error in the units of the target. Dominated by the high-titre runs."""
    actual_array, predicted_array = _as_paired_arrays(actual, predicted)
    return float(np.sqrt(np.mean((actual_array - predicted_array) ** 2)))


def mean_absolute_error(
    actual: Sequence[float] | NDArray[np.float64],
    predicted: Sequence[float] | NDArray[np.float64],
) -> float:
    """Error in the units of the target, less sensitive to single large misses."""
    actual_array, predicted_array = _as_paired_arrays(actual, predicted)
    return float(np.mean(np.abs(actual_array - predicted_array)))


def mean_absolute_percentage_error(
    actual: Sequence[float] | NDArray[np.float64],
    predicted: Sequence[float] | NDArray[np.float64],
) -> float:
    """Relative error as a percentage. Weights every run equally regardless of size.

    Raises:
        ValueError: if any actual value is zero, where relative error is undefined.
            Titres are strictly positive, so this would indicate a data problem.
    """
    actual_array, predicted_array = _as_paired_arrays(actual, predicted)
    if np.any(actual_array == 0.0):
        raise ValueError("relative error is undefined where the actual value is zero")
    return float(np.mean(np.abs((actual_array - predicted_array) / actual_array)) * 100.0)


def coefficient_of_determination(
    actual: Sequence[float] | NDArray[np.float64],
    predicted: Sequence[float] | NDArray[np.float64],
) -> float:
    """Fraction of variance explained, relative to predicting the mean of ``actual``.

    Negative means worse than predicting that mean. On a held-out set of ten runs
    concentrated in the upper tail this is unstable, and should be read alongside the
    absolute errors rather than instead of them.
    """
    actual_array, predicted_array = _as_paired_arrays(actual, predicted)
    total_sum_of_squares = float(np.sum((actual_array - actual_array.mean()) ** 2))
    if total_sum_of_squares == 0.0:
        raise ValueError("cannot compute R^2 when every actual value is identical")
    residual_sum_of_squares = float(np.sum((actual_array - predicted_array) ** 2))
    return 1.0 - residual_sum_of_squares / total_sum_of_squares


def specific_productivity_targets(
    runs: Sequence[ExperimentRun],
    targets: dict[str, float],
) -> dict[str, float]:
    """``qbar_P = Y_titer / gammaX`` per run: the screening target, not a model target.

    This is the cell-weighted time-average of specific productivity, and it is an exact
    identity rather than an approximation: ``P = qbar_P * gammaX`` by definition of the
    average.

    **It is used for stage-1 screening only.** Dividing by ``gammaX`` creates a ratio whose
    numerator and denominator share measurement error, so any feature tracking ``gammaX``
    is mechanically anti-correlated with it -- the artefact that made an earlier
    ``corr(qP, mu_max) = -0.590`` meaningless. The models fit directly against measured
    titre, so that artefact never reaches them. The division is a screening device for
    deciding *which* variables matter, and the M2/M3 comparison then answers where they act.

    Args:
        runs: the experiments, each carrying ``X:VCD``.
        targets: measured final titre per experiment identifier.

    Returns:
        Specific productivity per experiment identifier.

    Raises:
        KeyError: if a run has no target.
        ValueError: if a run accumulated no cell-days, leaving the ratio undefined.
    """
    missing = [run.experiment_id for run in runs if run.experiment_id not in targets]
    if missing:
        raise KeyError(f"no target for {missing}")

    productivity: dict[str, float] = {}
    for run in runs:
        biomaterial = features.cell_days(
            run.timestamps, run.observation(schema.OBSERVATION_VIABLE_CELL_DENSITY)
        )
        if biomaterial <= 0.0:
            raise ValueError(
                f"{run.experiment_id} accumulated {biomaterial} cell-days, so specific "
                "productivity is undefined"
            )
        productivity[run.experiment_id] = targets[run.experiment_id] / biomaterial
    return productivity


def split_by_duration(
    runs: Sequence[ExperimentRun],
    maximum_training_duration_days: float = DEFAULT_MAXIMUM_TRAINING_DURATION_DAYS,  # 7|8|9|10
) -> tuple[list[ExperimentRun], list[ExperimentRun]]:
    """Split runs into short (training) and long (held out), reproducing the real shift.

    Holding out the long runs measures the extrapolation the task actually demands. The
    held-out group is small -- ten runs -- and that limitation is real. It is reported
    rather than avoided, because the alternative measures the wrong thing.

    Args:
        runs: all available experiments.
        maximum_training_duration_days: runs longer than this are held out.

    Returns:
        ``(short_runs, long_runs)``, input order preserved within each group.

    Raises:
        ValueError: if the threshold leaves either group empty.
    """
    short_runs = [run for run in runs if run.duration_days <= maximum_training_duration_days]
    long_runs = [run for run in runs if run.duration_days > maximum_training_duration_days]
    if not short_runs or not long_runs:
        raise ValueError(
            f"a threshold of {maximum_training_duration_days} days leaves one side empty: "
            f"{len(short_runs)} short and {len(long_runs)} long"
        )
    return short_runs, long_runs


def k_fold_indices(
    sample_count: int,
    fold_count: int = 5,  # 5 | 10 | sample_count (leave-one-out)
    random_seed: int = 0,  # any int; fixed so folds are reproducible
) -> list[tuple[NDArray[np.int_], NDArray[np.int_]]]:
    """Indices for k-fold cross-validation: the in-distribution error estimate.

    Each fold re-estimates every parameter -- including ``kl`` -- from its own training
    runs. Reusing a ``kl`` fitted on all the data would leak the held-out runs into every
    fold and flatter the result.

    No grouping parameter is needed. Each experiment contributes exactly one sample,
    because the features reduce a whole trajectory to two numbers, so the usual risk of
    several rows from one run landing on both sides of a split was removed by that
    reduction rather than by a grouped splitter.

    Args:
        sample_count: number of runs.
        fold_count: number of folds.
        random_seed: fixed so folds are reproducible across invocations.

    Returns:
        One ``(train_indices, test_indices)`` pair per fold.
    """
    if fold_count < 2:
        raise ValueError(f"need at least two folds, got {fold_count}")
    if fold_count > sample_count:
        raise ValueError(f"cannot make {fold_count} folds from {sample_count} samples")

    shuffled = np.random.default_rng(random_seed).permutation(sample_count)
    folds = np.array_split(shuffled, fold_count)
    return [
        (
            np.concatenate([fold for index, fold in enumerate(folds) if index != held_out]),
            folds[held_out],
        )
        for held_out in range(fold_count)
    ]
