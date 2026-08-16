"""How well the data determine the model, and how wrong a prediction may be.

These are two different questions and the module keeps them apart deliberately.

*How well are the parameters determined?* -- profile likelihood and a bootstrap over runs.
A parameter can be reported by the optimiser and still be undetermined: the objective is
nearly flat along some directions, so a number comes out that looks like an estimate and
is not one. ``alpha`` here can move from 15 to 23 for under 3% change in fit quality.

*How wrong might a predicted titre be?* -- :func:`prediction_intervals`, which is wider and
answers a different question. Parameter uncertainty is only one of its two components, and
on this data it is the smaller one.

Why the distinction matters
---------------------------
A confidence interval on a parameter shrinks towards a point as data accumulate. A
prediction interval for a new run does not, because individual runs scatter around the model
however well its parameters are known. Reporting the first where the second is wanted
produces an interval that is far too narrow and looks rigorous while being wrong.

Concretely: the out-of-fold error of the selected model is of order 440 titre units, while
parameter uncertainty alone contributes a fraction of that. An interval built from the
parameter bootstrap alone would understate the real uncertainty several-fold, so
:func:`prediction_intervals` combines **both** sources, and :func:`coverage` checks the
result against out-of-fold outcomes rather than trusting it.

Why resample runs rather than residuals
----------------------------------------
A residual bootstrap holds the fitted structure fixed and reshuffles the noise around it,
so it assumes the model is correct and only measurement error varies. That assumption is
precisely what is in question, and it is the assumption most likely to be wrong here given
the model's weakness at the 14-day horizon. Resampling whole runs makes no such assumption:
it asks what would have been concluded from a different draw of experiments.

Cost
----
Every routine here refits the model many times. That is affordable only because the grid
search is fast; a bootstrap of 200 refits is a couple of minutes for the selected model.
Prediction remains one rate-law evaluation per interval whatever was done here.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from titre_predictor import evaluation, kinetics, model, screening
from titre_predictor.domain import ExperimentRun

DEFAULT_PROFILE_POINTS = 15  # 9 | 15 | 25
DEFAULT_BOOTSTRAP_RESAMPLES = 200  # 100 | 200 | 500
DEFAULT_STABILITY_RESAMPLES = 100  # re-screens inside each, so deliberately fewer
DEFAULT_CONFIDENCE = 0.90


# --- profile likelihood ----------------------------------------------------------------


@dataclass(frozen=True)
class ProfileResult:
    """How sharply the data determine one shape constant.

    Args:
        parameter_name: the constant profiled.
        unit: its physical unit, so the interval can be read as a quantity.
        values: the grid it was pinned at.
        residuals: best achievable residual sum of squares at each pinned value, with every
            other parameter re-optimised. That re-optimisation is what makes this a profile
            rather than a slice: holding the others fixed would forbid them from
            compensating and make any parameter look sharply determined.
        best_value: the freely fitted value.
        minimum_residual: the freely fitted residual.
        threshold: residual level bounding the confidence interval.
        lower: interval lower bound, ``-inf`` if the profile never rises above the
            threshold on that side -- the honest report for an unbounded direction.
        upper: interval upper bound, ``+inf`` likewise.
    """

    parameter_name: str
    unit: str
    values: NDArray[np.float64]
    residuals: NDArray[np.float64]
    best_value: float
    minimum_residual: float
    threshold: float
    lower: float
    upper: float

    @property
    def is_identified(self) -> bool:
        """Whether the data bound the parameter on both sides within its search range."""
        return bool(np.isfinite(self.lower)) and bool(np.isfinite(self.upper))

    @property
    def relative_rise(self) -> float:
        """Largest proportional increase in residual across the profiled range.

        A small number means the objective is nearly flat: the optimiser returned a value,
        but almost any other value in the range fits about as well.
        """
        return float(np.max(self.residuals) / self.minimum_residual - 1.0)


def profile_likelihood(
    runs: Sequence[ExperimentRun],
    targets: dict[str, float],
    parameter_name: str,
    variant_name: str = "M2",
    mechanism_names: Sequence[str] = (),
    point_count: int = DEFAULT_PROFILE_POINTS,
    confidence: float = DEFAULT_CONFIDENCE,
    value_range: tuple[float, float] | None = None,
    **fit_options: Any,
) -> ProfileResult:
    """Pin one shape constant across its range, re-optimising everything else at each point.

    The interval uses the standard threshold for nonlinear least squares,

        RSS(theta) <= RSS_min * (1 + F(1, n-p, confidence) / (n-p))

    which is the region not rejected by an F-test against the freely fitted model. It is an
    approximation -- it treats the model as locally linear in the parameters -- and on 100
    observations with a strongly curved objective it should be read as indicative rather
    than exact. It is still far more informative than a standard error computed at the
    optimum, which assumes the other parameters are known exactly.

    Args:
        runs: training experiments.
        targets: measured final titre per experiment identifier.
        parameter_name: shape constant to profile, e.g. ``"kl"`` or ``"theta_pH"``.
        variant_name: model variant.
        mechanism_names: mechanisms composing ``F``.
        point_count: grid points across the profiled range.
        confidence: for the interval threshold.
        value_range: range to profile over. Defaults to the parameter's own search range,
            which is **not always wide enough**: a search range only has to contain the
            optimum, while a profile must contain the whole interval or it will report an
            unbounded side that is really just the edge of the grid. Widening the search
            range instead would coarsen the fit's first sweep, so the two are kept separate.
        **fit_options: forwarded to :func:`titre_predictor.model.fit`.

    Raises:
        KeyError: if the parameter is not a shape constant of this model.
    """
    variant = model.resolve_variant(variant_name)
    mechanisms = kinetics.resolve(mechanism_names) if variant.needs_factor else ()
    specs = list(kinetics.parameter_specs(mechanisms))
    if variant.needs_lysis_rate:
        specs.insert(0, model.LYSIS_RATE_SPEC)

    matching = [spec for spec in specs if spec.name == parameter_name]
    if not matching:
        raise KeyError(
            f"{parameter_name!r} is not a shape constant of {variant_name} with mechanisms "
            f"{list(mechanism_names)}; available: {[spec.name for spec in specs]}"
        )
    spec = matching[0]

    free_model, free_diagnostics = model.fit(
        runs, targets, variant_name, mechanism_names, **fit_options
    )
    best_value = dict(
        zip(
            free_diagnostics.shape_constant_names,
            free_diagnostics.shape_constant_values,
            strict=True,
        )
    )[parameter_name]
    del free_model

    if value_range is None:
        grid = spec.grid(point_count)
    else:
        low, high = value_range
        grid = kinetics.ParameterSpec(
            spec.name, spec.unit, low, high, logarithmic=spec.logarithmic
        ).grid(point_count)
    residuals = np.empty(grid.size, dtype=np.float64)
    for index, value in enumerate(grid):
        _pinned, diagnostics = model.fit(
            runs,
            targets,
            variant_name,
            mechanism_names,
            fixed_shape_constants={parameter_name: float(value)},
            **fit_options,
        )
        residuals[index] = diagnostics.residual_sum_of_squares

    minimum_residual = min(float(free_diagnostics.residual_sum_of_squares), float(residuals.min()))
    parameter_count = len(specs) + 2  # shape constants plus alpha and beta
    degrees_of_freedom = max(len(runs) - parameter_count, 1)
    threshold = minimum_residual * (
        1.0 + float(stats.f.ppf(confidence, 1, degrees_of_freedom)) / degrees_of_freedom
    )
    lower, upper = _crossing_points(grid, residuals, threshold, best_value, spec.logarithmic)

    return ProfileResult(
        parameter_name=parameter_name,
        unit=spec.unit,
        values=grid,
        residuals=residuals,
        best_value=float(best_value),
        minimum_residual=minimum_residual,
        threshold=threshold,
        lower=lower,
        upper=upper,
    )


def _crossing_points(
    values: NDArray[np.float64],
    residuals: NDArray[np.float64],
    threshold: float,
    best_value: float,
    logarithmic: bool,
) -> tuple[float, float]:
    """Where the profile crosses the threshold either side of the optimum.

    Interpolates between grid points -- linearly in the parameter's own scale, so a constant
    searched over decades is interpolated in log space. Returns an infinite bound when the
    profile stays below the threshold all the way to the edge of the search range, which is
    the honest statement that the data do not bound the parameter on that side.
    """
    transform = np.log10 if logarithmic else (lambda array: array)
    scaled = transform(np.asarray(values, dtype=np.float64))
    scaled_best = float(transform(np.asarray([best_value], dtype=np.float64))[0])

    def crossing(indices: NDArray[np.int_], fallback: float) -> float:
        previous_index = None
        for index in indices:
            if residuals[index] > threshold:
                if previous_index is None:
                    return float(values[index])
                span = residuals[index] - residuals[previous_index]
                if span <= 0.0:
                    return float(values[index])
                fraction = (threshold - residuals[previous_index]) / span
                crossed = scaled[previous_index] + fraction * (
                    scaled[index] - scaled[previous_index]
                )
                return float(10.0**crossed if logarithmic else crossed)
            previous_index = index
        return fallback

    below = np.flatnonzero(scaled <= scaled_best)[::-1]
    above = np.flatnonzero(scaled >= scaled_best)
    return crossing(below, -np.inf), crossing(above, np.inf)


# --- bootstrap over runs ------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapResult:
    """Fitted models from resampled datasets, and what their spread implies.

    Args:
        parameter_names: every fitted parameter, coefficients first then shape constants.
        draws: ``(resamples, parameters)`` fitted values.
        point_estimate: the fit on the original runs, for reference.
        models: the fitted models, kept so predictions can be bootstrapped without refitting.
        failed_resamples: draws discarded because the fit was singular. A resample can omit
            enough distinct runs to leave the two regressors parallel; discarding is correct
            but the count is reported, since a large one would mean the model is fragile
            rather than that the bootstrap is fine.
    """

    parameter_names: tuple[str, ...]
    draws: NDArray[np.float64]
    point_estimate: tuple[float, ...]
    models: tuple[model.LuedekingPiretModel, ...]
    failed_resamples: int

    def interval(
        self, parameter_name: str, confidence: float = DEFAULT_CONFIDENCE
    ) -> tuple[float, float, float]:
        """``(median, lower, upper)`` percentile interval for one parameter."""
        column = self.draws[:, self.parameter_names.index(parameter_name)]
        tail = (1.0 - confidence) / 2.0
        return (
            float(np.median(column)),
            float(np.quantile(column, tail)),
            float(np.quantile(column, 1.0 - tail)),
        )

    def correlation(self) -> NDArray[np.float64]:
        """Correlation between parameters across resamples.

        Unlike the closed-form covariance at the optimum, this **includes** uncertainty in
        the shape constants rather than conditioning on them. That matters for the
        ``alpha``/``kl`` pair in particular: the dead-cell part of the growth term scales as
        ``alpha/kl``, so the two are separated only by the parts of the growth regressor
        that do not involve ``kl``.
        """
        return np.asarray(np.corrcoef(self.draws, rowvar=False), dtype=np.float64)


def bootstrap_parameters(
    runs: Sequence[ExperimentRun],
    targets: dict[str, float],
    variant_name: str = "M2",
    mechanism_names: Sequence[str] = (),
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    random_seed: int = 0,
    **fit_options: Any,
) -> BootstrapResult:
    """Refit the model on datasets resampled from the runs, with replacement.

    The mechanism set is held fixed across resamples. Re-selecting it inside each draw is a
    different and stricter question, answered by
    :func:`bootstrap_mechanism_stability`; holding it fixed here isolates parameter
    uncertainty from selection uncertainty rather than conflating the two.

    Args:
        runs: training experiments.
        targets: measured final titre per experiment identifier.
        variant_name: model variant.
        mechanism_names: mechanisms composing ``F``.
        resamples: bootstrap draws.
        random_seed: fixed so results are reproducible.
        **fit_options: forwarded to :func:`titre_predictor.model.fit`.
    """
    point_model, point_diagnostics = model.fit(
        runs, targets, variant_name, mechanism_names, **fit_options
    )
    names = ("alpha", "beta", *point_diagnostics.shape_constant_names)
    point_estimate = (
        point_model.alpha,
        point_model.beta,
        *point_diagnostics.shape_constant_values,
    )

    generator = np.random.default_rng(random_seed)
    rows: list[tuple[float, ...]] = []
    fitted_models: list[model.LuedekingPiretModel] = []
    failures = 0
    for _draw in range(resamples):
        indices = generator.integers(0, len(runs), len(runs))
        resampled = [runs[index] for index in indices]
        try:
            drawn_model, drawn_diagnostics = model.fit(
                resampled, targets, variant_name, mechanism_names, **fit_options
            )
        except (ValueError, np.linalg.LinAlgError):
            failures += 1
            continue
        if not np.isfinite(drawn_model.alpha) or not np.isfinite(drawn_model.beta):
            failures += 1
            continue
        rows.append((drawn_model.alpha, drawn_model.beta, *drawn_diagnostics.shape_constant_values))
        fitted_models.append(drawn_model)

    if not rows:
        raise RuntimeError(f"every one of {resamples} bootstrap resamples failed to fit")

    return BootstrapResult(
        parameter_names=names,
        draws=np.array(rows, dtype=np.float64),
        point_estimate=point_estimate,
        models=tuple(fitted_models),
        failed_resamples=failures,
    )


# --- prediction intervals -------------------------------------------------------------------


def prediction_intervals(
    bootstrap: BootstrapResult,
    runs: Sequence[ExperimentRun],
    residuals: NDArray[np.float64],
    confidence: float = DEFAULT_CONFIDENCE,
    random_seed: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Intervals for a **new** run's titre, combining both sources of uncertainty.

    Two things make a prediction wrong, and both must be included:

    1. **Parameter uncertainty** -- the bootstrap models disagree about the coefficients.
    2. **Residual scatter** -- runs differ from the model's expectation for reasons the
       model does not capture at all. This includes structural inadequacy, not only
       measurement noise.

    The predictive distribution is formed by pairing each bootstrap model's prediction with
    a residual drawn from the supplied set, so the two sources compose rather than one
    standing in for the other. Supplying **out-of-fold** residuals is what makes the result
    honest: in-sample residuals are shrunk by the fitting and would give intervals that are
    too narrow, which is the failure this function exists to avoid.

    On this data the second component dominates. That is itself the finding: most of the
    error is model inadequacy rather than ignorance of the parameters, so effort belongs in
    the structure rather than in pinning constants down.

    **The residuals are relative, and that is a correction rather than a preference.** An
    earlier version pooled *absolute* residuals and added them, which assumes the error is
    the same size for a run making 300 units of titre and one making 4800. It is not.
    Measured out of fold across the duration split:

        short runs (90)   mean |absolute residual|  158.9   mean |relative|  13.7%
        long runs (10)    mean |absolute residual|  459.3   mean |relative|  17.7%

    The absolute error nearly triples while the relative error barely moves, so the error is
    multiplicative and an additive pool is the wrong model. The consequence was not cosmetic:
    at 90% nominal, additive intervals covered 94% of short runs and only **50%** of the long
    ones -- and the long runs are what the task is about. Multiplying instead gives 91% and
    **80%**.

    That is a genuine calibration gain and not merely a wider interval: widened to the *same*
    mean width, additive intervals still cover only 50% of long runs, and reaching 80% that
    way needs 2.2x the width and then over-covers short runs at 99%. The residual 80% against
    90% nominal sits inside sampling noise -- with true coverage 90% on ten runs, seeing eight
    or fewer covered has probability 0.26.

    Args:
        bootstrap: fitted models from :func:`bootstrap_parameters`.
        runs: experiments to predict.
        residuals: out-of-fold **relative** residuals ``(actual - predicted) / predicted``,
            from :func:`out_of_fold_residuals`.
        confidence: interval width.
        random_seed: fixed so results are reproducible.

    Returns:
        ``(lower, median, upper)``, one entry per run.
    """
    if residuals.size == 0:
        raise ValueError("need at least one residual to build a predictive distribution")

    predictions = np.array(
        [fitted.predict_many(runs) for fitted in bootstrap.models], dtype=np.float64
    )
    generator = np.random.default_rng(random_seed)
    drawn = generator.choice(residuals, size=predictions.shape, replace=True)
    # Residuals are relative, so they scale with the prediction rather than being added to
    # it. See the module note on why: the error is multiplicative on this data, and adding a
    # pooled absolute residual under-covers exactly the large-titre runs the task cares about.
    predictive = predictions * (1.0 + drawn)

    tail = (1.0 - confidence) / 2.0
    return (
        np.quantile(predictive, tail, axis=0),
        np.quantile(predictive, 0.5, axis=0),
        np.quantile(predictive, 1.0 - tail, axis=0),
    )


def coverage(
    actual: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
) -> float:
    """Fraction of outcomes falling inside their interval.

    The check that separates an interval one can defend from one that merely looks
    quantitative. A nominal 90% interval covering 60% of outcomes is not conservative or
    approximate, it is wrong, and saying so is more useful than shipping it.
    """
    return float(np.mean((actual >= lower) & (actual <= upper)))


# --- mechanism stability ---------------------------------------------------------------------


def bootstrap_mechanism_stability(
    runs: Sequence[ExperimentRun],
    targets: dict[str, float],
    resamples: int = DEFAULT_STABILITY_RESAMPLES,
    fold_count: int = screening.DEFAULT_FOLD_COUNT,
    random_seed: int = 0,
) -> dict[str, float]:
    """How often each mechanism is selected across resampled datasets.

    This is the honest answer to the post-selection caveat. Coefficients and errors computed
    after selecting mechanisms on the same data are optimistically biased, and the usual
    remedy -- reporting the caveat -- says nothing about how large the problem is. Re-running
    the **whole selection** inside each resample does: a mechanism chosen in 90% of draws was
    chosen for a reason that survives resampling, while one chosen in 55% was close to a coin
    toss and its constants should not be quoted as findings.

    Note this re-runs :func:`titre_predictor.screening.select_mechanism_set`, the
    cross-validated forward search that actually picks the model -- **not** the variable
    screen, which is a diagnostic and does not choose anything. Measuring the stability of a
    step that no longer selects would answer the wrong question.

    That makes this the slowest routine in the module: each resample is a full forward
    search, itself many cross-validations. The default resample count is correspondingly
    modest, and it is the reason ``--uncertainty`` takes minutes rather than seconds.

    Args:
        runs: training experiments.
        targets: measured final titre per experiment identifier.
        resamples: bootstrap draws.
        fold_count: folds used inside each selection.
        random_seed: fixed so results are reproducible.

    Returns:
        Mechanism name to the fraction of resamples in which it was selected.
    """
    generator = np.random.default_rng(random_seed)
    counts: dict[str, int] = {name: 0 for name in kinetics.MECHANISMS}
    completed = 0

    for _draw in range(resamples):
        indices = generator.integers(0, len(runs), len(runs))
        resampled = [runs[index] for index in indices]
        try:
            selection = screening.select_mechanism_set(
                resampled, targets, fold_count=fold_count, random_seed=random_seed
            )
        except (ValueError, KeyError, np.linalg.LinAlgError):
            continue
        completed += 1
        for mechanism in selection.chosen:
            counts[mechanism] += 1

    if completed == 0:
        raise RuntimeError(f"every one of {resamples} screening resamples failed")
    return {name: count / completed for name, count in counts.items()}


def out_of_fold_residuals(
    runs: Sequence[ExperimentRun],
    targets: dict[str, float],
    variant_name: str = "M2",
    mechanism_names: Sequence[str] = (),
    fold_count: int = 10,
    random_seed: int = 0,
    **fit_options: Any,
) -> NDArray[np.float64]:
    """``(actual - predicted) / predicted`` from models that never saw the run they predict.

    These are the residuals :func:`prediction_intervals` needs. Two properties matter and
    both are corrections of an earlier version:

    * **Out of fold.** In-sample residuals are shrunk by the fitting itself and would produce
      intervals that are too narrow.
    * **Relative, not absolute.** The error on this data is multiplicative -- across the
      duration split the mean absolute residual nearly triples while the mean relative
      residual moves from 13.7% to 17.7%. Pooling absolute residuals therefore under-covers
      the large-titre runs badly; see :func:`prediction_intervals` for the measured effect.

    Raises:
        ValueError: if any out-of-fold prediction is not strictly positive, since a relative
            residual is undefined there. A non-positive predicted titre is itself a modelling
            failure and should surface rather than be divided by.
    """

    def fit_fold(
        train_runs: Sequence[ExperimentRun], train_targets: dict[str, float]
    ) -> model.LuedekingPiretModel:
        fitted, _diagnostics = model.fit(
            train_runs, train_targets, variant_name, mechanism_names, **fit_options
        )
        return fitted

    predicted = evaluation.cross_validated_predictions(
        runs, targets, fit_fold, fold_count, random_seed
    )
    if np.any(predicted <= 0.0):
        offending = [
            run.experiment_id for run, value in zip(runs, predicted, strict=True) if value <= 0.0
        ]
        raise ValueError(
            f"relative residuals need strictly positive predictions; {offending} predicted "
            f"non-positive titre out of fold"
        )
    actual = np.array([targets[run.experiment_id] for run in runs], dtype=np.float64)
    return (actual - predicted) / predicted
