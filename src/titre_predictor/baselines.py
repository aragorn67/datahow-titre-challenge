"""Data-driven benchmarks: what a standard ML model achieves on the same problem.

Why these exist
---------------
The mean baseline establishes a floor, but it cannot answer the question the mechanistic
model's whole design rests on: *does the structure buy anything a general-purpose learner
does not get for free?* Without a data-driven comparator, "a kinetic model suits an
extrapolation to unseen run durations" is an assertion. With one it is a measurement.

Partial least squares is the specific comparator to beat. It is the established method for
this exact problem shape -- many correlated process variables, few batches -- and it is not a
neutral choice of straw man: Johan Trygg co-authored both Richelle papers this model's
structure follows *and* co-invented OPLS. If a mechanistic model cannot beat PLS here, the
honest conclusion is to ship PLS.

Gradient boosting is included for a different reason. It is the strongest general-purpose
learner for tabular data of this size, so it bounds what flexibility alone can achieve, and it
extrapolates in the opposite manner to a linear model -- a tree predicts a constant beyond its
training range. The two benchmarks therefore answer different questions and neither
substitutes for the other.

That difference turned out to be the informative part, and not in the direction expected. On
the duration split, ``cell_days`` for eight of the ten held-out runs lies above the training
maximum (up to 549 against 243), and the two baselines break in opposite characteristic ways:

* **PLS overshoots.** Extrapolating a linear fit into that region predicts 3889 where the run
  measured 1790, and 4122 where it measured 1727. Its predictions span 537-4218 against
  actuals of 610-4823, so the range is roughly right and the individual assignments are not.
* **Gradient boosting saturates.** Its predictions compress into 730-2912 against the same
  actuals: no tree can return a value above the training range, so the highest-titre run
  (4823) is capped at 2805. Bounded error, but systematically biased low.

The tree scores better on RMSE precisely *because* it cannot extrapolate -- flatness is a
cheap form of safety when the alternative is running away. Neither is a model of the process;
they are two ways of failing to leave the training distribution, which is the failure the
kinetic structure exists to avoid. This is worth more to the argument than either number
alone, and it is why both are kept.

**The baseline settings were fixed before these numbers were seen** -- modest capacity chosen
for 90 runs and 25 features, component count delegated to an inner CV -- and have not been
revisited since. Tuning a comparator after seeing that it loses is how a benchmark becomes
decoration.

Making the comparison fair
--------------------------
A benchmark that is easy to beat proves nothing, so every advantage that can be given to the
baselines is given to them:

* **The same features, from the same code.** Baselines are fitted on the run-level aggregates
  of :func:`titre_predictor.features.run_features` -- the same function stage-1 screening
  uses. They are not handed a worse featurisation than the mechanistic model's inputs were
  derived from.
* **More information, not less.** They receive every always-defined feature -- 25 of them,
  including the design scalars, ``cell_days``, and ``duration_days`` itself, so the baselines
  are told how long the run lasted. The selected mechanistic model reads two metabolites
  through ``F``. If it wins, it wins on strictly less input.
* **Hyperparameters chosen inside the fold.** The PLS component count is selected by an inner
  cross-validation over the training fold only. Choosing it once on all the data would give
  the baseline a leakage advantage the mechanistic model was never given, which would flatter
  the conclusion in the direction this module exists to test.
* **The identical splits and metrics.** Both benchmarks go through
  :func:`titre_predictor.evaluation.cross_validated_predictions` and
  :func:`titre_predictor.evaluation.split_by_duration`, so the numbers sit in the same table
  as the model's rather than beside it.

The one asymmetry that cannot be removed
----------------------------------------
Event-conditional features -- late-phase exposures, VCD at a process shift -- are undefined
for runs that never reached the event, and which runs those are is near-perfectly determined
by duration. A matrix-shaped learner needs a rectangular input, so those columns are dropped
when they are not defined across the whole training fold, and a value missing at prediction
time falls back to the training mean.

That fallback is imputation, and it is reported rather than buried: :attr:`Fitted.imputed`
counts it. It is also, in itself, part of the answer. The mechanistic model never faces this
choice, because it integrates along the trajectory it is given instead of reducing the run to
a fixed-width row -- so a 14-day run is simply a longer integral, not a row with holes in it.
That is a structural difference between the two approaches, not a tuning detail.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from titre_predictor import features
from titre_predictor.domain import ExperimentRun

# Candidate latent-variable counts for PLS, chosen inside each training fold. One component is
# a single latent direction -- close to a scaled univariate regression on the dominant feature
# -- and beyond about eight the model is fitting noise on 90 runs.
DEFAULT_PLS_COMPONENTS = (1, 2, 3, 4, 5, 6, 8)

# Folds for the inner selection of the PLS component count. Fewer than the outer ten, because
# an inner fold is carved from an already-reduced training set and the count only has to be
# resolved to within one.
DEFAULT_INNER_FOLD_COUNT = 5

# Gradient boosting settings. Deliberately modest: 90 training runs and 25 features do not
# support a deep ensemble, and the point is to bound what flexibility achieves rather than to
# win a tuning contest. The learning rate and depth are sklearn's defaults for this estimator
# apart from the leaf minimum, which is raised because the default of 20 exceeds a fifth of
# the training set.
DEFAULT_BOOSTING_MAXIMUM_ITERATIONS = 200
DEFAULT_BOOSTING_MINIMUM_LEAF = 5

# Below this a feature is treated as constant across the training fold and dropped: dividing
# by its standard deviation would amplify floating-point noise into a large standardised value.
MINIMUM_FEATURE_STANDARD_DEVIATION = 1e-12


@dataclass(frozen=True, eq=False)
class Fitted:
    """A fitted benchmark, satisfying the same prediction protocol as the kinetic model.

    Everything needed to turn a run into a prediction travels together: which feature columns
    were used, the standardisation those columns were fitted under, and the estimator. A
    baseline that recomputed its own column list at prediction time could silently apply
    coefficients to a different set of features than it was fitted on.

    Args:
        name: how the benchmark is labelled in the report.
        feature_names: the columns used, in order.
        centre: per-column training mean, also the fallback for a missing value.
        scale: per-column training standard deviation.
        predict_standardised: the fitted estimator, taking a standardised matrix.
        detail: what was selected inside the fold, for the report -- e.g. the component count.
        imputed: how many individual values were replaced by a column mean at prediction time.
            A list so it can be appended to after construction; the object is otherwise frozen.
    """

    name: str
    feature_names: tuple[str, ...]
    centre: NDArray[np.float64]
    scale: NDArray[np.float64]
    predict_standardised: Callable[[NDArray[np.float64]], NDArray[np.float64]]
    detail: str = ""
    imputed: list[int] = field(default_factory=list)

    def _standardised(self, runs: Sequence[ExperimentRun]) -> NDArray[np.float64]:
        matrix = _feature_matrix(runs, self.feature_names)
        missing = ~np.isfinite(matrix)
        if missing.any():
            self.imputed.append(int(missing.sum()))
            matrix = np.where(missing, self.centre[None, :], matrix)
        return (matrix - self.centre[None, :]) / self.scale[None, :]

    def predict_many(self, runs: Sequence[ExperimentRun]) -> NDArray[np.float64]:
        """Predicted final titres, in the order the runs were given."""
        return np.asarray(
            self.predict_standardised(self._standardised(runs)).ravel(), dtype=np.float64
        )

    def predict(self, run: ExperimentRun) -> float:
        """Predicted final titre for one experiment."""
        return float(self.predict_many([run])[0])


def _feature_matrix(
    runs: Sequence[ExperimentRun],
    feature_names: Sequence[str],
) -> NDArray[np.float64]:
    """The named features for each run, in the order given. May contain ``nan``."""
    rows = [features.run_features(run) for run in runs]
    return np.array([[row[name] for name in feature_names] for row in rows], dtype=np.float64)


def usable_feature_names(runs: Sequence[ExperimentRun]) -> tuple[str, ...]:
    """Features defined and varying across every one of ``runs``.

    Called with a training fold, never with all the data, so the column list is itself fitted
    rather than chosen with knowledge of the runs being predicted.

    A feature is kept only if it is finite for every training run -- an event-conditional
    feature is dropped rather than imputed at fit time -- and only if it varies, since a
    column constant across the fold carries no information and cannot be standardised.
    """
    names, matrix = features.feature_frame(runs)
    finite = np.isfinite(matrix).all(axis=0)
    varies = np.nanstd(matrix, axis=0) > MINIMUM_FEATURE_STANDARD_DEVIATION
    keep = finite & varies
    return tuple(name for name, flag in zip(names, keep, strict=True) if flag)


def _standardisation(
    matrix: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Column means and standard deviations, with zero-variance columns left unscaled."""
    centre = np.asarray(matrix.mean(axis=0), dtype=np.float64)
    scale = np.asarray(matrix.std(axis=0), dtype=np.float64)
    scale = np.where(scale > MINIMUM_FEATURE_STANDARD_DEVIATION, scale, 1.0)
    return centre, scale


@dataclass(frozen=True, eq=False)
class _Prepared:
    """A training fold reduced to the matrix every benchmark reads."""

    feature_names: tuple[str, ...]
    design: NDArray[np.float64]  # standardised
    centre: NDArray[np.float64]
    scale: NDArray[np.float64]
    targets: NDArray[np.float64]


def _prepared(runs: Sequence[ExperimentRun], targets: dict[str, float]) -> _Prepared:
    """Fit the column list and standardisation to a training fold, and apply them.

    Raises:
        KeyError: if a run has no target.
        ValueError: if fewer than three runs are supplied, or no feature survives.
    """
    if len(runs) < 3:
        raise ValueError(f"need at least three runs to fit a benchmark, got {len(runs)}")
    missing = [run.experiment_id for run in runs if run.experiment_id not in targets]
    if missing:
        raise KeyError(f"no target for {missing}")

    feature_names = usable_feature_names(runs)
    if not feature_names:
        raise ValueError("no feature is defined and varying across every training run")

    matrix = _feature_matrix(runs, feature_names)
    centre, scale = _standardisation(matrix)
    return _Prepared(
        feature_names=feature_names,
        design=(matrix - centre[None, :]) / scale[None, :],
        centre=centre,
        scale=scale,
        targets=np.array([targets[run.experiment_id] for run in runs], dtype=np.float64),
    )


def _select_component_count(
    design: NDArray[np.float64],
    targets: NDArray[np.float64],
    candidates: Sequence[int],
    fold_count: int,
    random_seed: int,
) -> int:
    """The PLS component count with the lowest inner cross-validated error.

    Inside the training fold only. A component count chosen once on all the data would leak
    the outer test fold into every model, which is precisely the advantage this comparison
    must not give the baseline.
    """
    usable = [count for count in candidates if count <= min(design.shape[1], design.shape[0] - 2)]
    if not usable:
        return 1
    if len(usable) == 1:
        return usable[0]

    inner_folds = min(fold_count, design.shape[0])
    splitter = KFold(n_splits=inner_folds, shuffle=True, random_state=random_seed)
    scores: dict[int, float] = {}
    for count in usable:
        errors: list[float] = []
        for train_index, test_index in splitter.split(design):
            if train_index.size <= count + 1:
                continue
            estimator = PLSRegression(n_components=count, scale=False)
            estimator.fit(design[train_index], targets[train_index])
            predicted = estimator.predict(design[test_index]).ravel()
            errors.append(float(np.mean((targets[test_index] - predicted) ** 2)))
        if errors:
            scores[count] = float(np.mean(errors))
    if not scores:
        return usable[0]
    # Fewest components on a tie: the same "prefer the simpler option when the measurement
    # cannot separate them" rule the model selection steps apply.
    return min(sorted(scores), key=lambda count: scores[count])


def fit_partial_least_squares(
    runs: Sequence[ExperimentRun],
    targets: dict[str, float],
    component_candidates: Sequence[int] = DEFAULT_PLS_COMPONENTS,
    inner_fold_count: int = DEFAULT_INNER_FOLD_COUNT,
    random_seed: int = 0,
) -> Fitted:
    """PLS on the run-level aggregates, component count selected inside the training set.

    Args:
        runs: training experiments, each carrying its ``W:`` control profiles.
        targets: measured final titre per experiment identifier.
        component_candidates: latent-variable counts to try.
        inner_fold_count: folds for the inner selection.
        random_seed: seed for the inner fold assignment.
    """
    prepared = _prepared(runs, targets)
    count = _select_component_count(
        prepared.design, prepared.targets, component_candidates, inner_fold_count, random_seed
    )
    estimator = PLSRegression(n_components=count, scale=False)
    estimator.fit(prepared.design, prepared.targets)
    return Fitted(
        name="PLS",
        feature_names=prepared.feature_names,
        centre=prepared.centre,
        scale=prepared.scale,
        predict_standardised=estimator.predict,
        detail=f"{count} component{'s' if count != 1 else ''}",
    )


def fit_gradient_boosting(
    runs: Sequence[ExperimentRun],
    targets: dict[str, float],
    maximum_iterations: int = DEFAULT_BOOSTING_MAXIMUM_ITERATIONS,
    minimum_leaf: int = DEFAULT_BOOSTING_MINIMUM_LEAF,
    random_seed: int = 0,
) -> Fitted:
    """Gradient-boosted trees on the same aggregates.

    Standardisation is irrelevant to a tree and applied anyway, so that every benchmark reads
    the identical matrix and a difference between them is a difference of estimator.

    Args:
        runs: training experiments, each carrying its ``W:`` control profiles.
        targets: measured final titre per experiment identifier.
        maximum_iterations: boosting iterations.
        minimum_leaf: minimum samples per leaf.
        random_seed: seed for the estimator.
    """
    prepared = _prepared(runs, targets)
    estimator = HistGradientBoostingRegressor(
        max_iter=maximum_iterations,
        min_samples_leaf=minimum_leaf,
        random_state=random_seed,
    )
    estimator.fit(prepared.design, prepared.targets)
    return Fitted(
        name="gradient boosting",
        feature_names=prepared.feature_names,
        centre=prepared.centre,
        scale=prepared.scale,
        predict_standardised=estimator.predict,
        detail=f"{maximum_iterations} iterations, min leaf {minimum_leaf}",
    )


# The registry the pipeline loops over, so adding a benchmark is an entry here rather than
# another branch in the reporting code.
BASELINES: dict[str, Callable[[Sequence[ExperimentRun], dict[str, float]], Fitted]] = {
    "PLS": fit_partial_least_squares,
    "gradient boosting": fit_gradient_boosting,
}
