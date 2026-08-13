"""The two quantities the titre model regresses on.

This module is imported by **both** the training pipeline and the inference service.
Neither computes features its own way, so the numbers served can never drift from the
numbers fitted.

Where the two features come from
--------------------------------
Product formation follows a Luedeking-Piret rate law, ``dP/dt = (alpha*mu_eff + beta)*Xv``.
Integrating over the run and substituting ``mu_eff*Xv = dXv/dt + mu_d*Xv`` gives

    titre = alpha * (cells synthesised) + beta * (cell-days)

Under the sequential lysis form, in which lysed cells derive from the dead pool
(``dXd/dt = mu_d*Xv - kl*Xd``, ``dXl/dt = kl*Xd``), the integral of ``mu_d*Xv`` is the
dead pool at harvest, and the dead pool itself is readable from the lysed-cell
trajectory:

    Xd(t) = (1/kl) * dXl/dt

So both features are quadratures over **measured** trajectories, with a single unknown
constant ``kl``. No differential equation is integrated, at training or at inference.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from titre_predictor.data import schema
from titre_predictor.domain import ExperimentRun

# Feature order is declared once and used by both fitting and prediction, so a
# coefficient can never be applied to the wrong column.
FEATURE_NAMES = ("cells_synthesised", "cell_days")

DEFAULT_SLOPE_WINDOW_POINTS = 4  # 3 | 4 | 5 -- points used for the end-of-run lysis slope


def cell_days(
    timestamps: NDArray[np.float64],
    viable_cell_density: NDArray[np.float64],
) -> float:
    """Integral of viable cell density over the run, by the trapezoidal rule.

    This is the biomaterial variable of Richelle et al., since ``dgammaX/dt = Xv``, and
    it is the non-growth-associated regressor.

    Args:
        timestamps: sample times in days.
        viable_cell_density: ``X:VCD`` at each timestamp.
    """
    return float(np.trapezoid(viable_cell_density, timestamps))


def lysed_cell_slope_at_end(
    timestamps: NDArray[np.float64],
    lysed_cells: NDArray[np.float64],
    window_points: int = DEFAULT_SLOPE_WINDOW_POINTS,  # 3 | 4 | 5
) -> float:
    """Rate of lysed-cell accumulation at harvest, by a straight-line fit to the tail.

    ``X:Lysed`` is the noisiest series in the dataset (roughly 7% of its range), and a
    two-point difference at the endpoint would be dominated by that noise. Fitting a
    line through the last few points averages it down.

    The result is clipped at zero: the underlying pool is cumulative, so a negative
    slope is measurement noise rather than signal, and it would otherwise produce a
    negative dead-cell pool.

    The window is fixed a priori rather than tuned, because tuning it against held-out
    runs would be selecting on the very data used to judge the model.

    Args:
        timestamps: sample times in days.
        lysed_cells: ``X:Lysed`` at each timestamp.
        window_points: how many trailing points to fit. Larger is smoother but reaches
            further back from harvest.

    Raises:
        ValueError: if ``window_points`` is below two or exceeds the run length.
    """
    if window_points < 2:
        raise ValueError(f"need at least two points to fit a slope, got {window_points}")
    if window_points > timestamps.size:
        raise ValueError(
            f"window_points={window_points} exceeds the {timestamps.size} available timepoints"
        )

    slope, _intercept = np.polyfit(timestamps[-window_points:], lysed_cells[-window_points:], 1)
    return float(max(slope, 0.0))


def dead_cells_at_harvest(
    timestamps: NDArray[np.float64],
    lysed_cells: NDArray[np.float64],
    lysis_rate_constant: float,
    window_points: int = DEFAULT_SLOPE_WINDOW_POINTS,  # 3 | 4 | 5
) -> float:
    """The dead-cell pool at harvest, recovered from the lysed-cell trajectory.

    The dead pool is never measured. Under the sequential lysis form it is nonetheless
    determined by measured data up to the single constant ``kl``, because lysed cells
    derive from it: ``dXl/dt = kl * Xd``.

    Args:
        timestamps: sample times in days.
        lysed_cells: ``X:Lysed`` at each timestamp.
        lysis_rate_constant: ``kl``, in reciprocal days. Fitted, not assumed.
        window_points: passed to :func:`lysed_cell_slope_at_end`.

    Raises:
        ValueError: if ``lysis_rate_constant`` is not strictly positive.
    """
    if lysis_rate_constant <= 0.0:
        raise ValueError(
            f"lysis_rate_constant must be strictly positive, got {lysis_rate_constant}"
        )
    return lysed_cell_slope_at_end(timestamps, lysed_cells, window_points) / lysis_rate_constant


def cells_synthesised(
    run: ExperimentRun,
    lysis_rate_constant: float,
    window_points: int = DEFAULT_SLOPE_WINDOW_POINTS,  # 3 | 4 | 5
) -> float:
    """Total viable cells produced over the run: the growth-associated regressor.

    Every cell ever made is either still viable, dead, or lysed, so

        cells synthesised = change in viable + lysed at harvest + dead at harvest

    The first two terms are read directly off the measurements; only the third involves
    a fitted constant.

    Args:
        run: the experiment.
        lysis_rate_constant: ``kl``, in reciprocal days.
        window_points: passed to :func:`lysed_cell_slope_at_end`.
    """
    viable = run.observation(schema.OBSERVATION_VIABLE_CELL_DENSITY)
    lysed = run.observation(schema.OBSERVATION_LYSED_CELLS)

    change_in_viable = float(viable[-1] - viable[0])
    lysed_at_harvest = float(lysed[-1] - lysed[0])
    dead_at_harvest = dead_cells_at_harvest(
        run.timestamps, lysed, lysis_rate_constant, window_points
    )
    return change_in_viable + lysed_at_harvest + dead_at_harvest


def feature_vector(
    run: ExperimentRun,
    lysis_rate_constant: float,
    window_points: int = DEFAULT_SLOPE_WINDOW_POINTS,  # 3 | 4 | 5
) -> NDArray[np.float64]:
    """The two regressors for one experiment, ordered as :data:`FEATURE_NAMES`.

    Args:
        run: the experiment.
        lysis_rate_constant: ``kl``, in reciprocal days.
        window_points: passed to :func:`lysed_cell_slope_at_end`.
    """
    return np.array(
        [
            cells_synthesised(run, lysis_rate_constant, window_points),
            cell_days(run.timestamps, run.observation(schema.OBSERVATION_VIABLE_CELL_DENSITY)),
        ],
        dtype=np.float64,
    )


def design_matrix(
    runs: Sequence[ExperimentRun],
    lysis_rate_constant: float,
    window_points: int = DEFAULT_SLOPE_WINDOW_POINTS,  # 3 | 4 | 5
) -> NDArray[np.float64]:
    """Stack :func:`feature_vector` over many experiments, one run per row.

    Args:
        runs: the experiments, in the order their targets will be supplied.
        lysis_rate_constant: ``kl``, in reciprocal days.
        window_points: passed to :func:`lysed_cell_slope_at_end`.
    """
    if not runs:
        raise ValueError("need at least one run to build a design matrix")
    return np.vstack([feature_vector(run, lysis_rate_constant, window_points) for run in runs])
