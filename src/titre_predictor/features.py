"""Per-interval quantities for the titre model, and per-run features for screening.

This module is imported by **both** the training pipeline and the inference service, so the
numbers served can never drift from the numbers fitted.

What the model integrates
-------------------------
Product formation follows a Luedeking-Piret rate law with specific productivity modulated
by the culture environment, ``qP = (alpha*mu_eff + beta) * F(z)``, and titre is its integral
against the measured viable-cell curve. From Richelle 2026 Eq. 14, ``mu_eff = mu + mu_d +
mu_l``, which worked through either paper's population structure gives the same identity:

    INT mu_eff*Xv dt  =  Delta(Xv + Xd + Xl)  =  total cells ever synthesised

That is an **endpoint** quantity, so no numerical differentiation of the viable-cell curve
happens anywhere. It matters: differencing a 2%-noise signal over daily samples produces
roughly 20% error on the derivative.

The dead pool ``Xd`` is never measured. Under the sequential lysis form it is nonetheless
determined by measured data up to the single constant ``kl``, since ``dXl/dt = kl*Xd``:

    Xd(t) = Xl'(t) / kl
    C(t)  = Xv(t) + Xl(t) + Xl'(t)/kl        cells made up to time t

Per interval ``j`` between consecutive samples:

    dC_j = C(t_{j+1}) - C(t_j)                       growth contribution
    gX_j = (Xv(t_j) + Xv(t_{j+1}))/2 * dt_j          trapezoidal cell-days

so that ``sum_j dC_j = Delta(Xv+Xd+Xl)`` and ``sum_j gX_j = gammaX`` exactly. Those two
identities are what make M2 nest M1 and M3 nest both, and they are asserted in the tests
rather than assumed.

Why ``Xl(t)`` is fitted rather than differenced
------------------------------------------------
M1 and M2 need ``Xd`` only at harvest; M3 needs ``Xl'(t)`` at every timepoint. Rather than
finite-difference the noisiest series in the dataset repeatedly, a smooth **monotone** curve
is fitted to ``Xl(t)`` per run and the derivative taken analytically. Monotonicity is a
physical constraint on a cumulative pool, not a smoothing convenience.

The form is non-negative least squares on an integrated B-spline basis:

    Xl'(t) = sum_k c_k B_k(t),  c_k >= 0    =>    Xl(t) = c_0 + sum_k c_k INT_0^t B_k

B-splines are non-negative everywhere, so non-negative coefficients make the derivative
non-negative **everywhere**, not merely at the knots. That is the whole reason for this
construction: PCHIP is monotone but interpolates, so 7% noise would pass straight into
``Xl'``; isotonic regression is monotone but piecewise constant, so its derivative is a
train of spikes; an unconstrained smoothing spline smooths but can turn downwards. NNLS is
convex with a finite active-set solution, so there is no starting guess and no local
minimum -- the same estimation posture as the rest of the model.

This also replaces an arbitrary 4-point slope window at harvest whose choice swung held-out
RMSE between 1430 and 1648. The fit uses every point and has no window to select, so a known
sensitivity disappears rather than being tuned away.

**Settings chosen by measurement.** Leave-one-point-out cross-validation pooled over all 120
runs, sweeping degree 1-3 against both a fixed knot count and a fixed knot spacing, put a
degree-2 derivative basis with one knot every 5 days lowest (LOO RMS 0.01257). The top three
candidates sit within 2% of each other, so the tie is broken on smoothness rather than on
the third decimal: a quadratic basis is C1, so ``Xd(t)`` has no kinks, which matters because
M3 evaluates it at every interior timepoint. Residual RMS is 0.0068 against a measured noise
sd of about 0.0087 -- the fit is not chasing noise.

Knot *spacing* rather than knot *count* is deliberate as well as empirically better: it
gives a 7-day and a 14-day run the same flexibility per unit time, where a fixed count would
give short runs more.

**Two properties to know about.** The fitted harvest slope runs about 45% above the 4-point
window it replaces, because a straight line through the last four points of a convex
accelerating curve has that window's *average* slope, understating the slope *at* harvest.
Since the model only ever sees ``Xl'(T)/kl`` and ``kl`` is refitted, this is absorbed into
``kl`` rather than changing the fit. And a handful of runs get a fitted harvest slope of
exactly zero, hence ``Xd(T) = 0`` -- implausible physically, but the honest reading of a
lysate curve that is flat at the end relative to the noise floor.

The intercept ``c_0`` is left free and non-negative. The noise on ``X:Lysed`` is clipped at
zero -- 48% of early values are exactly 0.0 -- which biases the observations *upward* early
and not late; forcing the curve through ``Xl(0) = 0`` would make it tilt to absorb that bias,
corrupting the derivative, which is the quantity actually wanted. The intercept is harmless
because every use of ``Xl`` here is via a difference, in which a constant offset cancels.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline
from scipy.optimize import nnls

from titre_predictor import kinetics
from titre_predictor.data import schema
from titre_predictor.domain import ExperimentRun

# --- monotone lysate fit -------------------------------------------------------------

# Degree of the B-spline basis used for the *derivative*; the curve itself is one degree
# higher. 2 gives a C1 derivative, so Xd(t) is smooth at the knots.
DEFAULT_DERIVATIVE_SPLINE_DEGREE = 2  # 1 | 2 | 3

# One interior knot roughly every this many days, so smoothing per unit time is the same
# for a 7-day and a 14-day run.
DEFAULT_KNOT_SPACING_DAYS = 5.0  # 2.5 | 3.5 | 5.0 | 7.0

# --- screening features --------------------------------------------------------------

# Exposure is split early/late at a fixed absolute day, never at each run's own midpoint. A
# relative midpoint would make "late" mean days 3.5-7 in a short run and 7-14 in a long one
# -- different physiological regimes, confounded with duration in exactly the direction that
# matters. With a fixed cut, short runs have little or no late window, which is honest: they
# genuinely never reached that regime, and that *is* the train/test shift.
DEFAULT_PHASE_SPLIT_DAY = 7.0  # 5 | 6 | 7 | 8

# Design scalars offered to screening in their own right. The rest of the Z: block either
# reappears through a derived feature (the shift times, via VCD at the shift; the feed rates'
# window, via feed_window_days) or is the target horizon itself (Z:ExpDuration).
SCREENING_DESIGN_SCALARS = (
    schema.DESIGN_DISSOLVED_OXYGEN,
    schema.DESIGN_STIRRING,
    schema.DESIGN_FEED_RATE_GLUCOSE,
    schema.DESIGN_FEED_RATE_GLUTAMINE,
)

# Series offered as cell-day-weighted exposures, with early/late variants.
SCREENING_EXPOSURE_SERIES = (
    schema.OBSERVATION_GLUCOSE,
    schema.OBSERVATION_GLUTAMINE,
    schema.OBSERVATION_LACTATE,
    schema.OBSERVATION_AMMONIA,
    schema.OBSERVATION_LYSED_CELLS,
    schema.CONTROL_TEMPERATURE,
    schema.CONTROL_PH,
)

# Process events at which viable cell density is reported.
SCREENING_SHIFT_TIMES = (
    schema.DESIGN_TEMPERATURE_SHIFT,
    schema.DESIGN_PH_SHIFT,
    schema.DESIGN_FEED_START,
    schema.DESIGN_FEED_END,
)


def cell_days(
    timestamps: NDArray[np.float64],
    viable_cell_density: NDArray[np.float64],
) -> float:
    """Integral of viable cell density over the run, by the trapezoidal rule.

    This is the biomaterial variable of Richelle et al., since ``dgammaX/dt = Xv``.

    Args:
        timestamps: sample times in days.
        viable_cell_density: ``X:VCD`` at each timestamp.
    """
    return float(np.trapezoid(viable_cell_density, timestamps))


def interval_cell_days(
    timestamps: NDArray[np.float64],
    viable_cell_density: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Cell-days contributed by each interval between consecutive samples.

    Summing these reproduces :func:`cell_days` exactly. They are kept apart so that each
    interval can carry its own environmental factor.

    Args:
        timestamps: sample times in days.
        viable_cell_density: ``X:VCD`` at each timestamp.
    """
    midpoint_density = 0.5 * (viable_cell_density[:-1] + viable_cell_density[1:])
    return np.asarray(midpoint_density * np.diff(timestamps), dtype=np.float64)


def _basis_shape(
    point_count: int,
    duration_days: float,
    requested_degree: int,
    knot_spacing_days: float,
) -> tuple[int, int]:
    """Degree and interior-knot count that fit within the available observations.

    The requested settings apply whenever the run is long enough, which it is for every run
    in the supplied data. The reduction below exists for the inference service, which may be
    handed a shorter run than anything in the training files: rather than let the basis
    become underdetermined -- at which point NNLS would interpolate the noise, defeating the
    entire purpose of smoothing -- the basis shrinks until at least one residual degree of
    freedom remains.

    Args:
        point_count: number of observations in the run.
        duration_days: last timestamp.
        requested_degree: preferred degree of the derivative basis.
        knot_spacing_days: preferred spacing between interior knots.

    Returns:
        ``(degree, interior_knot_count)``.
    """
    requested_interior = max(round(duration_days / knot_spacing_days) - 1, 0)
    for degree in range(requested_degree, 0, -1):
        for interior in range(requested_interior, -1, -1):
            # Columns: one intercept plus (degree + interior + 1) basis functions.
            if degree + interior + 2 <= point_count - 1:
                return degree, interior
    return 1, 0


def _spline_design(
    evaluation_times: NDArray[np.float64],
    knots: NDArray[np.float64],
    degree: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Design matrices for the curve and its derivative.

    Column 0 is the intercept: constant 1 in the value matrix, 0 in the derivative matrix.
    Column ``k+1`` holds the integrated basis function ``INT_0^t B_k`` and, in the derivative
    matrix, ``B_k(t)`` itself.

    Args:
        evaluation_times: where to evaluate, in days.
        knots: boundary and interior knots, ascending, spanning the run.
        degree: degree of the derivative basis.

    Returns:
        ``(value_design, derivative_design)``, each ``(len(evaluation_times), columns)``.
    """
    full_knots = np.concatenate([np.repeat(knots[0], degree), knots, np.repeat(knots[-1], degree)])
    basis_count = full_knots.size - degree - 1

    value_columns = [np.ones_like(evaluation_times)]
    derivative_columns = [np.zeros_like(evaluation_times)]
    for index in range(basis_count):
        coefficients = np.zeros(basis_count)
        coefficients[index] = 1.0
        # extrapolate=True is safe and NaN-free here because every evaluation point lies
        # inside [knots[0], knots[-1]]; it exists only to make the closed right endpoint
        # evaluate to the polynomial's limit rather than to NaN.
        basis_function = BSpline(full_knots, coefficients, degree, extrapolate=True)
        antiderivative = basis_function.antiderivative()
        value_columns.append(antiderivative(evaluation_times) - antiderivative(knots[0]))
        derivative_columns.append(basis_function(evaluation_times))

    return (
        np.column_stack(value_columns).astype(np.float64),
        np.column_stack(derivative_columns).astype(np.float64),
    )


@dataclass(frozen=True, eq=False)
class LysateCurve:
    """A fitted monotone curve through ``X:Lysed``, with an analytic derivative.

    The derivative is non-negative everywhere by construction, not by checking: the basis
    functions are non-negative and :attr:`coefficients` are constrained non-negative.

    Args:
        coefficients: intercept followed by one non-negative weight per basis function.
        knots: boundary and interior knots, in days.
        degree: degree of the derivative basis.
        residual_root_mean_square: fit residual against the observations it was built from,
            for comparison against the measurement noise scale.
    """

    coefficients: NDArray[np.float64]
    knots: NDArray[np.float64]
    degree: int
    residual_root_mean_square: float

    def value(self, evaluation_times: NDArray[np.float64]) -> NDArray[np.float64]:
        """``Xl(t)``: the smoothed cumulative lysed-cell pool."""
        value_design, _ = _spline_design(evaluation_times, self.knots, self.degree)
        return np.asarray(value_design @ self.coefficients, dtype=np.float64)

    def derivative(self, evaluation_times: NDArray[np.float64]) -> NDArray[np.float64]:
        """``Xl'(t)``, guaranteed non-negative. Multiply by ``1/kl`` to get ``Xd(t)``."""
        _, derivative_design = _spline_design(evaluation_times, self.knots, self.degree)
        return np.asarray(derivative_design @ self.coefficients, dtype=np.float64)


def fit_lysate_curve(
    timestamps: NDArray[np.float64],
    lysed_cells: NDArray[np.float64],
    derivative_degree: int = DEFAULT_DERIVATIVE_SPLINE_DEGREE,  # 1 | 2 | 3
    knot_spacing_days: float = DEFAULT_KNOT_SPACING_DAYS,  # 2.5 | 3.5 | 5.0 | 7.0
) -> LysateCurve:
    """Fit a smooth monotone curve to the lysed-cell trajectory.

    Args:
        timestamps: sample times in days, strictly increasing.
        lysed_cells: ``X:Lysed`` at each timestamp.
        derivative_degree: degree of the B-spline basis for the derivative.
        knot_spacing_days: target spacing between interior knots.

    Returns:
        The fitted curve, whose derivative is non-negative everywhere.

    Raises:
        ValueError: if the two arrays differ in length or there are fewer than two points.
    """
    if timestamps.shape != lysed_cells.shape:
        raise ValueError(
            f"timestamps and lysed_cells must have the same shape, got "
            f"{timestamps.shape} and {lysed_cells.shape}"
        )
    if timestamps.size < 2:
        raise ValueError(f"need at least two points to fit a curve, got {timestamps.size}")

    degree, interior_count = _basis_shape(
        timestamps.size, float(timestamps[-1]), derivative_degree, knot_spacing_days
    )
    knots = np.linspace(float(timestamps[0]), float(timestamps[-1]), interior_count + 2)

    value_design, _ = _spline_design(timestamps, knots, degree)
    coefficients, _residual_norm = nnls(value_design, lysed_cells)

    residuals = lysed_cells - value_design @ coefficients
    return LysateCurve(
        coefficients=np.asarray(coefficients, dtype=np.float64),
        knots=knots,
        degree=degree,
        residual_root_mean_square=float(np.sqrt(np.mean(residuals**2))),
    )


# --- interval quantities -------------------------------------------------------------


def cells_made(
    timestamps: NDArray[np.float64],
    viable_cell_density: NDArray[np.float64],
    curve: LysateCurve,
    lysis_rate_constant: float,
) -> NDArray[np.float64]:
    """``C(t) = Xv(t) + Xl(t) + Xl'(t)/kl``: cells made up to each timestamp.

    Both the lysate level and its derivative come from the fitted curve. Using the raw
    measurements for the level instead would make ``dC_j`` negative wherever the noisy
    series happens to dip -- which it does across 247 of the 1170 intervals in the supplied
    data -- and would leave ``C`` with two inconsistent definitions between M1 and M3.

    Args:
        timestamps: sample times in days.
        viable_cell_density: ``X:VCD`` at each timestamp.
        curve: the fitted monotone lysate curve.
        lysis_rate_constant: ``kl``, in reciprocal days. Fitted, not assumed.

    Raises:
        ValueError: if ``lysis_rate_constant`` is not strictly positive.
    """
    if lysis_rate_constant <= 0.0:
        raise ValueError(
            f"lysis_rate_constant must be strictly positive, got {lysis_rate_constant}"
        )
    return np.asarray(
        viable_cell_density
        + curve.value(timestamps)
        + curve.derivative(timestamps) / lysis_rate_constant,
        dtype=np.float64,
    )


# --- per-run reduction ---------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class RunQuantities:
    """One run reduced to the parts that do not depend on a fitted parameter.

    Fitting scans a grid over ``kl`` and the mechanism shape constants. Only ``1/kl`` and
    the environmental factor change between candidates; the spline fit and the quadratures
    do not. Recomputing them per candidate would repeat identical arithmetic many thousands
    of times.

    The growth contribution of an interval is **linear in** ``1/kl``:

        dC_j = growth_base_j + lysate_slope_change_j / kl

    which is what lets the grid search evaluate every ``kl`` candidate as one array
    operation rather than one loop iteration.

    The ``interval_*`` arrays are aligned and one element shorter than :attr:`timestamps`.
    """

    experiment_id: str
    timestamps: NDArray[np.float64]
    interval_cell_days: NDArray[np.float64]
    interval_viable_change: NDArray[np.float64]  # dXv_j -- net growth, which is M0's regressor
    interval_lysate_change: NDArray[np.float64]  # dXl_j, from the fitted lysate curve
    interval_lysate_slope_change: NDArray[np.float64]  # d(Xl')_j; divide by kl for d(Xd)_j
    state: Mapping[str, NDArray[np.float64]]  # measured series, for the environmental factor
    lysate_curve: LysateCurve

    @property
    def interval_growth_base(self) -> NDArray[np.float64]:
        """``d(Xv + Xl)_j``: the part of ``dC_j`` that does not depend on ``kl``.

        Kept separate from :attr:`interval_viable_change` because M0 regresses on net
        growth alone -- the documented failure the nesting exists to demonstrate against.
        """
        return np.asarray(self.interval_viable_change + self.interval_lysate_change)

    @property
    def cell_days(self) -> float:
        """``gammaX``: what the weighted integral reduces to when ``F == 1``."""
        return float(np.sum(self.interval_cell_days))

    def interval_growth(self, lysis_rate_constant: float) -> NDArray[np.float64]:
        """``dC_j`` for every interval, at a given ``kl``.

        Raises:
            ValueError: if ``lysis_rate_constant`` is not strictly positive.
        """
        if lysis_rate_constant <= 0.0:
            raise ValueError(
                f"lysis_rate_constant must be strictly positive, got {lysis_rate_constant}"
            )
        return np.asarray(
            self.interval_growth_base + self.interval_lysate_slope_change / lysis_rate_constant,
            dtype=np.float64,
        )

    def cells_synthesised(self, lysis_rate_constant: float) -> float:
        """``Delta(Xv + Xd + Xl)`` over the whole run: the M1 growth regressor.

        Equal to ``sum_j dC_j`` by construction, since the intervals telescope.
        """
        return float(np.sum(self.interval_growth(lysis_rate_constant)))


def run_quantities(
    run: ExperimentRun,
    derivative_degree: int = DEFAULT_DERIVATIVE_SPLINE_DEGREE,  # 1 | 2 | 3
    knot_spacing_days: float = DEFAULT_KNOT_SPACING_DAYS,  # 2.5 | 3.5 | 5.0 | 7.0
) -> RunQuantities:
    """Reduce a run to the observation-only quantities the model is built from.

    Args:
        run: the experiment. Must carry the ``W:`` control profiles if a temperature or pH
            mechanism is in force; a caller holding only the ``Z:`` scalars should
            reconstruct them first with
            :func:`titre_predictor.data.controls.reconstruct_control_profiles`.
        derivative_degree: passed to :func:`fit_lysate_curve`.
        knot_spacing_days: passed to :func:`fit_lysate_curve`.

    Raises:
        InvalidExperimentRunError: if a required observation is absent.
    """
    viable = run.observation(schema.OBSERVATION_VIABLE_CELL_DENSITY)
    lysed = run.observation(schema.OBSERVATION_LYSED_CELLS)

    curve = fit_lysate_curve(run.timestamps, lysed, derivative_degree, knot_spacing_days)
    fitted_level = curve.value(run.timestamps)
    fitted_slope = curve.derivative(run.timestamps)

    return RunQuantities(
        experiment_id=run.experiment_id,
        timestamps=run.timestamps,
        interval_cell_days=interval_cell_days(run.timestamps, viable),
        interval_viable_change=np.diff(viable),
        interval_lysate_change=np.diff(fitted_level),
        interval_lysate_slope_change=np.diff(fitted_slope),
        state=kinetics.state_from_run(run),
        lysate_curve=curve,
    )


def interval_factor(
    mechanisms: Sequence[kinetics.Mechanism],
    quantities: RunQuantities,
    parameters: Sequence[float],
) -> NDArray[np.float64]:
    """``F_j``: the environmental factor averaged across each interval.

    The contribution to ``INT F*Xv dt`` is trapezoidal in the **product**, so the factor is
    averaged between the interval's endpoints rather than evaluated at a midpoint state.
    This matters because ``F`` is nonlinear -- ``F(mean z) != mean F(z)`` -- and metabolites
    move sharply within a day once feeding starts.

    With no mechanisms in force this returns all ones, so the weighted sums reduce exactly to
    the plain trapezoid. That is what makes the M1/M2/M3 nesting exact rather than
    approximate.

    Args:
        mechanisms: the factors in force, from :func:`titre_predictor.kinetics.resolve`.
        quantities: the reduction of one run.
        parameters: shape constants, concatenated in mechanism order.
    """
    pointwise = kinetics.environmental_factor(mechanisms, quantities.state, parameters)
    return np.asarray(0.5 * (pointwise[:-1] + pointwise[1:]), dtype=np.float64)


# --- screening features --------------------------------------------------------------


def _interval_overlap(
    timestamps: NDArray[np.float64],
    window_start: float,
    window_end: float,
) -> NDArray[np.float64]:
    """Fraction of each interval lying inside ``[window_start, window_end]``.

    Exact for intervals wholly inside or wholly outside the window. The supplied data are
    sampled on whole days and the phase split falls on a sample point, so no interval
    straddles the boundary and the fractional case never arises here; it is handled anyway
    so a different sampling grid does not silently misattribute a whole interval.
    """
    left = timestamps[:-1]
    right = timestamps[1:]
    overlap = np.minimum(right, window_end) - np.maximum(left, window_start)
    return np.asarray(np.clip(overlap, 0.0, None) / (right - left), dtype=np.float64)


def _cell_weighted_exposure(
    quantities: RunQuantities,
    values: NDArray[np.float64],
    window_start: float,
    window_end: float,
) -> float:
    """Cell-day-weighted mean of a series over a time window.

    The weighting is **derived, not chosen**. From ``qbar_P = INT qP*Xv dt / INT Xv dt``, if
    ``qP`` depends on ``z(t)`` then to first order ``qbar_P`` depends on the cell-weighted
    mean of ``z``. An arithmetic mean or an endpoint value would be the wrong summary under
    the mechanism.

    Returns ``nan`` when the window holds no cell-days -- a short run has no late phase, and
    the mean of an empty window does not exist. See :func:`run_features` on why that is
    reported rather than filled in.
    """
    interval_mean = 0.5 * (values[:-1] + values[1:])
    weights = quantities.interval_cell_days * _interval_overlap(
        quantities.timestamps, window_start, window_end
    )
    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        return float("nan")
    return float(np.sum(interval_mean * weights) / total_weight)


def _specific_growth_rates(
    quantities: RunQuantities,
    viable_cell_density: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Net specific growth rate per interval: ``dXv_j / (mean Xv_j * dt_j)``.

    The denominator is the interval-mean viable density -- the same quantity the cell-day
    trapezoid uses -- rather than the left endpoint, which would be badly conditioned in the
    first interval where the inoculum is small.

    This is *net* growth and is a screening feature only. The model itself uses effective
    growth, which is why it never differentiates ``Xv``.
    """
    interval_mean = 0.5 * (viable_cell_density[:-1] + viable_cell_density[1:])
    interval_length = np.diff(quantities.timestamps)
    safe_mean = np.where(interval_mean > 0.0, interval_mean, np.nan)
    rates = np.diff(viable_cell_density) / (safe_mean * interval_length)
    return np.asarray(rates, dtype=np.float64)


def _viable_density_at(
    run: ExperimentRun,
    event_time: float,
) -> float:
    """Viable cell density at a process event, linearly interpolated between samples.

    Returns ``nan`` when the event falls beyond the end of the run. Over half the training
    runs never reach their temperature shift, and for those the question "how large was the
    culture when the shift happened?" has no answer. Clamping to the final value would
    silently report a different quantity -- final VCD -- under a shift feature's name.
    """
    if event_time > run.duration_days or event_time < run.timestamps[0]:
        return float("nan")
    viable = run.observation(schema.OBSERVATION_VIABLE_CELL_DENSITY)
    return float(np.interp(event_time, run.timestamps, viable))


def run_features(
    run: ExperimentRun,
    quantities: RunQuantities | None = None,
    phase_split_day: float = DEFAULT_PHASE_SPLIT_DAY,  # 5 | 6 | 7 | 8
) -> dict[str, float]:
    """Per-run features offered to stage-1 screening.

    Every provided input is offered; nothing is pre-filtered by judgement. Screening decides
    which of these matter, fold-wise and by stability, and the surviving variables determine
    which mechanisms enter ``F(z)``.

    These features are used **only** for screening. The models fit directly against measured
    titre, so the ``qbar_P = Y_titer / gammaX`` ratio that screening regresses on never
    touches them.

    **Some features are ``nan`` by design.** A late-phase exposure is undefined for a run
    that ended at the split day, and VCD at a shift is undefined for a run that never reached
    it -- which is true of over half the training runs for the temperature shift. Both are
    filled with ``nan`` rather than a substitute, because in both cases the missingness is
    perfectly collinear with duration: imputing a value would inject a duration signal
    disguised as a metabolite or shift effect, which is the exact confounding the fixed
    absolute phase split exists to avoid. Screening must therefore adopt an explicit policy
    for them rather than inherit a hidden one from here.

    Args:
        run: the experiment, carrying its ``W:`` control profiles.
        quantities: the run's reduction, if already computed. Recomputed if omitted.
        phase_split_day: absolute day at which exposure is split early/late.

    Returns:
        Feature name to value. Keys are stable across runs, so the dicts stack into a frame.
    """
    if quantities is None:
        quantities = run_quantities(run)

    viable = run.observation(schema.OBSERVATION_VIABLE_CELL_DENSITY)
    state = quantities.state
    duration = run.duration_days
    features: dict[str, float] = {}

    # Cell-day-weighted exposures: whole run, then early and late at a fixed absolute day.
    for name in SCREENING_EXPOSURE_SERIES:
        if name not in state:
            continue
        values = state[name]
        label = name.split(":", 1)[1]
        features[f"exposure_{label}"] = _cell_weighted_exposure(quantities, values, 0.0, np.inf)
        features[f"exposure_{label}_early"] = _cell_weighted_exposure(
            quantities, values, 0.0, phase_split_day
        )
        features[f"exposure_{label}_late"] = _cell_weighted_exposure(
            quantities, values, phase_split_day, np.inf
        )

    # Growth summaries. Cell-weighted so they answer "what rate did the average cell-day
    # experience?", which is the aggregation the rate law implies.
    growth_rates = _specific_growth_rates(quantities, viable)
    finite = np.isfinite(growth_rates)
    weights = quantities.interval_cell_days
    weight_total = float(np.sum(weights[finite]))
    features["growth_rate_mean"] = (
        float(np.sum(growth_rates[finite] * weights[finite]) / weight_total)
        if weight_total > 0.0
        else float("nan")
    )
    features["growth_rate_peak"] = (
        float(np.max(growth_rates[finite])) if finite.any() else float("nan")
    )

    # Biomaterial and the culture's own scale.
    features["cell_days"] = quantities.cell_days
    features["peak_viable_density"] = float(np.max(viable))

    # Viable density at each process event.
    for shift_name in SCREENING_SHIFT_TIMES:
        label = shift_name.split(":", 1)[1]
        features[f"vcd_at_{label}"] = _viable_density_at(run, run.design_scalars[shift_name])

    # Design scalars that reach titre only through specific productivity.
    for design_name in SCREENING_DESIGN_SCALARS:
        features[design_name.split(":", 1)[1]] = float(run.design_scalars[design_name])

    # Feeding actually delivered within the run, not the window as designed: a feed window
    # ending after harvest was never fully applied.
    feed_start = min(float(run.design_scalars[schema.DESIGN_FEED_START]), duration)
    feed_end = min(float(run.design_scalars[schema.DESIGN_FEED_END]), duration)
    features["feed_window_days"] = max(feed_end - feed_start, 0.0)

    # Deliberately not intensive. Included only to test whether residual duration dependence
    # survives the factorisation -- prediction 2 of the four made before any of this was fit.
    features["duration_days"] = duration

    return features


def feature_frame(
    runs: Sequence[ExperimentRun],
    phase_split_day: float = DEFAULT_PHASE_SPLIT_DAY,  # 5 | 6 | 7 | 8
) -> tuple[tuple[str, ...], NDArray[np.float64]]:
    """Stack :func:`run_features` over many runs.

    Args:
        runs: the experiments, in the order their targets will be supplied.
        phase_split_day: absolute day at which exposure is split early/late.

    Returns:
        ``(feature_names, matrix)`` with one row per run and one column per feature, the
        columns ordered as ``feature_names``. May contain ``nan``; see :func:`run_features`.

    Raises:
        ValueError: if no runs are given, or if they do not all produce the same feature
            names -- which would mean the columns did not line up between rows.
    """
    if not runs:
        raise ValueError("need at least one run to build a feature frame")

    rows = [run_features(run, phase_split_day=phase_split_day) for run in runs]
    names = tuple(rows[0])
    for run, row in zip(runs, rows, strict=True):
        if tuple(row) != names:
            raise ValueError(
                f"{run.experiment_id} produced feature names differing from "
                f"{runs[0].experiment_id}; columns would not line up"
            )
    matrix = np.array([[row[name] for name in names] for row in rows], dtype=np.float64)
    return names, matrix
