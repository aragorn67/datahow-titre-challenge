"""Stage 1: deciding which physiological variables belong in ``F(z)``.

The two-stage split
-------------------
Screening is data-driven and decides *which* variables matter. The rate law in ``model.py``
is mechanistic and specifies *how* they act. Keeping them apart is what stops the mechanism
set from being an assertion: a factor enters ``F`` because a variable earned its place here,
not because it seemed plausible.

The target is specific productivity, ``qbar_P = Y_titer / gammaX``. Dividing by biomaterial
removes the extensive part of the problem, leaving an intensive quantity whose variation is
what a productivity mechanism has to explain. **The ratio is a screening device only** --
the models fit directly against measured titre, so the artefact that makes anything tracking
``gammaX`` mechanically anti-correlated with ``qbar_P`` never reaches them.

Why stability rather than a ranking
-----------------------------------
With 100 runs, 25-35 candidate features and known collinearity, any single ranking on a
single fit is close to arbitrary: swap a few runs and a different variable tops the list.
So every method is run **inside each fold**, on that fold's training rows only, and a
variable earns its place by being chosen **repeatedly and by more than one method**.

Four methods are used because they fail differently. Correlation sees only marginal
association and is fooled by collinearity. ElasticNetCV picks one of a correlated group
almost at random. PLS projects onto latent variables and so spreads weight across a
correlated group instead of choosing within it. Permutation importance is measured on data
the fit did not see, so it is the only one that answers "does this help prediction?" rather
than "does this fit?". Agreement across methods that fail in different directions is
evidence; agreement among four rankings that share a blind spot would not be.

The per-variable per-method selection frequency is printed so the choice is auditable rather
than asserted.

Missing values, and why there are two tables
--------------------------------------------
The supplied CSVs have no missing observations -- every ``X:`` and ``W:`` cell is present.
The gaps are in *derived* features, and they mean the event never happened rather than that
a measurement was lost: a run ending on day 7 has no day-7-onwards window, and a run whose
temperature shift falls beyond its harvest has no viable density at that shift.

Whether such a feature exists is almost perfectly determined by run duration. Imputing would
therefore encode "this was a short run" into a feature labelled as a metabolite or a shift
effect, and screening would then select it for a reason that has nothing to do with the
mechanism it names. Dropping the affected runs instead would discard most of the dataset,
since short runs are the majority.

So the screen runs twice:

* the **primary** table over features defined for every run;
* a **conditional** table over the event-dependent features, restricted to the runs where
  the event actually occurred.

Neither imputes. The conditional table carries its own caveat -- the surviving subset skews
long, so it is a smaller and duration-biased sample, and a variable that only appears there
is weaker evidence than one that survives the primary table.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from sklearn.cross_decomposition import PLSRegression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNetCV

from titre_predictor import evaluation, features, kinetics
from titre_predictor.domain import ExperimentRun

if TYPE_CHECKING:  # model.py imports nothing from here; this keeps that direction intact
    from titre_predictor.model import FitDiagnostics

# Features per method per fold that count as "selected". Small relative to the candidate
# count, so that agreeing is informative: with 25 candidates, four methods each picking five
# would agree by chance far less often than if each picked fifteen.
DEFAULT_SELECTION_SIZE = 5  # 3 | 5 | 8

DEFAULT_FOLD_COUNT = 10  # matches the selection cross-validation used for the models

# A variable survives if at least this fraction of folds chose it, under at least this many
# methods. Both are deliberately strict: the point is to end with two or three mechanisms,
# and the recorded expectation is that effective capacity is two to three.
DEFAULT_MINIMUM_FREQUENCY = 0.6  # 0.5 | 0.6 | 0.8
DEFAULT_MINIMUM_METHODS = 2  # 1 | 2 | 3

# Mechanism selection. A mechanism costs one or two parameters against 100 observations, so
# it must reduce cross-validated error by a clear margin rather than a hair.
DEFAULT_MAXIMUM_MECHANISMS = 4  # 2 | 3 | 4

# The margin every selection step demands, not just this one. A cross-validated RMSE over 90
# runs is itself an estimate: resampling the folds moves it by more than a fraction of a
# percent, so a difference smaller than this is not evidence that one option is better. The
# same instrument must therefore be read to the same precision wherever it is used --
# mechanisms, model variant, and ridge strength alike. Applying 1% to mechanisms and then
# accepting a 0.5% gain elsewhere would be two bars for one measurement.
#
# It is used by :func:`select_mechanism_set` and by :func:`choose_by_improvement`, which is
# what the pipeline's variant and ridge steps go through.
DEFAULT_MINIMUM_RELATIVE_IMPROVEMENT = 0.01  # 1% of the current error

# Identifiability, the second admission test. Predicting well is not sufficient: a mechanism
# ships as an *equation with constants in stated units*, so a constant the data do not
# determine cannot be quoted, and a mechanism whose constants are all like that is not a
# mechanism -- it is a spare degree of freedom wearing a physical name.
#
# The test is run on the selection folds only, using the fits already performed for scoring,
# so it costs nothing and reads no held-out data.
#
# Of the two tests below, **the bound test is the decisive one and the principled one.** A
# constant resting on the edge of its search range is not an estimate at all: the data wanted
# to go further and the grid stopped it. On the supplied runs it is what rejects both
# mechanisms that are rejected -- metabolic_burden's K_L and lysate_inhibition's K_X each rest
# on a bound in 3 of the 10 folds.
#
# The spread test is a softer secondary check and its threshold is genuinely a judgement, so
# the sensitivity is recorded rather than glossed. Measured across the ten selection folds:
#
#     K_Q  1.88x   determined
#     K_G  8.28x   accepted, but only just inside the threshold -- see below
#     K_L  8.98x   rejected, on a bound in 30% of folds
#     K_X  52.3x   rejected, on a bound in 30% of folds
#
# So a threshold of 10 admits the shipped model and a threshold of 3 would additionally reject
# glucose_limitation, leaving glutamine alone. **That is a real sensitivity and the honest
# reading is that K_G is determined only to within about an order of magnitude** -- it must be
# quoted with that caveat rather than to four figures. The mechanism still earns its place on
# prediction (174.1 against 228.7 for glutamine alone, a 24% reduction), and the shape of the
# Monod term means predictions are far less sensitive to K_G than the constant itself is: every
# fold value from 0.18 to 1.47 mM is small against a glucose range of 0-44 mM, so the factor is
# near one except where glucose is nearly exhausted. Sensitive constant, insensitive
# prediction, and those are different claims.
DEFAULT_MAXIMUM_CONSTANT_SPREAD = 10.0  # 3 | 10 | 30

# A linear-scaled constant has no meaningful ratio -- it may be negative or zero -- so its
# movement is measured against the width of its own search range instead.
DEFAULT_MAXIMUM_RANGE_FRACTION = 0.25  # 0.1 | 0.25 | 0.5

# Folds allowed to place a constant on a search bound before it stops being an estimate. Two
# in ten is tolerance for an awkward fold; more than that and the fit is routinely being
# stopped by the grid rather than by the data.
DEFAULT_MAXIMUM_PINNED_FRACTION = 0.2  # 0.0 | 0.2 | 0.4

METHOD_NAMES = ("correlation", "elastic_net", "pls", "permutation")

# Which mechanism a surviving exposure variable licenses. This is the only place the two
# stages meet, and it is a lookup rather than a judgement made per run of the script.
MECHANISM_FOR_FEATURE: Mapping[str, str] = {
    "exposure_Glc": "glucose_limitation",
    "exposure_Gln": "glutamine_limitation",
    "exposure_Lac": "metabolic_burden",
    "exposure_Amm": "metabolic_burden",
    "exposure_Lysed": "lysate_inhibition",
    "exposure_temp": "temperature_response",
    "exposure_pH": "ph_response",
}


@dataclass(frozen=True)
class StabilityTable:
    """How often each feature was chosen, by each method, across folds.

    Args:
        feature_names: candidates screened, in column order.
        frequencies: ``(features, methods)`` selection frequency in ``[0, 1]``.
        method_names: the methods, in column order of ``frequencies``.
        variance_inflation: per feature, ``1/(1-R^2)`` against the others. Above ten is the
            usual flag for a variable that is nearly a combination of the rest.
        maximum_absolute_correlation: per feature, its strongest pairwise correlation.
        run_count: runs the screen was computed over.
    """

    feature_names: tuple[str, ...]
    frequencies: NDArray[np.float64]
    method_names: tuple[str, ...]
    variance_inflation: NDArray[np.float64]
    maximum_absolute_correlation: NDArray[np.float64]
    run_count: int

    def method_count(self, minimum_frequency: float) -> NDArray[np.int_]:
        """How many methods chose each feature at or above ``minimum_frequency``."""
        return np.asarray(np.sum(self.frequencies >= minimum_frequency, axis=1), dtype=np.int_)

    def survivors(
        self,
        minimum_frequency: float = DEFAULT_MINIMUM_FREQUENCY,
        minimum_methods: int = DEFAULT_MINIMUM_METHODS,
    ) -> tuple[str, ...]:
        """Features chosen often enough, by enough different methods.

        Ordered by mean selection frequency, strongest first, so the report reads as a
        ranking even though the rule is a threshold.
        """
        qualifying = self.method_count(minimum_frequency) >= minimum_methods
        order = np.argsort(-self.frequencies.mean(axis=1))
        return tuple(self.feature_names[index] for index in order if bool(qualifying[index]))

    def mechanisms(
        self,
        minimum_frequency: float = DEFAULT_MINIMUM_FREQUENCY,
        minimum_methods: int = DEFAULT_MINIMUM_METHODS,
    ) -> tuple[str, ...]:
        """Mechanisms licensed by the surviving variables, de-duplicated, order preserved.

        Lactate and ammonia both map to ``metabolic_burden``, so either one surviving
        licenses that single two-parameter factor rather than two separate ones.
        """
        chosen: list[str] = []
        for name in self.survivors(minimum_frequency, minimum_methods):
            mechanism = MECHANISM_FOR_FEATURE.get(_base_feature_name(name))
            if mechanism is not None and mechanism not in chosen:
                chosen.append(mechanism)
        return tuple(chosen)


def _base_feature_name(name: str) -> str:
    """Strip an ``_early`` / ``_late`` phase suffix, which does not change the mechanism."""
    for suffix in ("_early", "_late"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _standardise(
    train: NDArray[np.float64],
    apply_to: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Centre and scale by the **training** fold's statistics only.

    Using the whole dataset's mean and scale would leak the held-out rows into every fold,
    which is the quiet version of the mistake this whole module exists to avoid.
    """
    centre = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)  # a constant column carries no information
    return (train - centre) / scale, (apply_to - centre) / scale


def _top_indices(scores: NDArray[np.float64], count: int) -> NDArray[np.int_]:
    """Indices of the ``count`` largest finite scores, strictly positive ones only."""
    usable = np.where(np.isfinite(scores), scores, -np.inf)
    order = np.argsort(-usable)[:count]
    return np.asarray([index for index in order if usable[index] > 0.0], dtype=np.int_)


def screen(
    feature_names: Sequence[str],
    matrix: NDArray[np.float64],
    productivity: NDArray[np.float64],
    fold_count: int = DEFAULT_FOLD_COUNT,
    selection_size: int = DEFAULT_SELECTION_SIZE,
    random_seed: int = 0,
) -> StabilityTable:
    """Run every method inside every fold and count how often each feature is chosen.

    Args:
        feature_names: candidate names, matching ``matrix`` columns.
        matrix: ``(runs, features)``. Must contain no ``nan``; see the module docstring on
            why the caller splits its features into a complete set and a conditional one
            rather than imputing.
        productivity: ``qbar_P`` per run, aligned to ``matrix`` rows.
        fold_count: folds to run the methods inside.
        selection_size: features each method may select per fold.
        random_seed: fixed so the folds and the estimators are reproducible.

    Raises:
        ValueError: if the shapes disagree or the matrix contains missing values.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    productivity = np.asarray(productivity, dtype=np.float64)
    if matrix.shape[0] != productivity.size:
        raise ValueError(
            f"matrix has {matrix.shape[0]} rows but {productivity.size} productivity values"
        )
    if matrix.shape[1] != len(feature_names):
        raise ValueError(
            f"matrix has {matrix.shape[1]} columns but {len(feature_names)} feature names"
        )
    if not np.isfinite(matrix).all():
        raise ValueError(
            "screening needs a fully observed matrix; split conditional features out and "
            "screen them on the runs where they are defined rather than imputing them"
        )

    counts = np.zeros((len(feature_names), len(METHOD_NAMES)), dtype=np.float64)
    folds = evaluation.k_fold_indices(matrix.shape[0], fold_count, random_seed)

    for train_indices, test_indices in folds:
        train_features, test_features = _standardise(matrix[train_indices], matrix[test_indices])
        train_target = productivity[train_indices]
        test_target = productivity[test_indices]

        # 1. Marginal association. Blind to collinearity, which is exactly why it is here:
        #    it fails differently from the multivariate methods.
        centred = train_target - train_target.mean()
        denominator = np.sqrt(np.sum(train_features**2, axis=0) * np.sum(centred**2))
        correlation = np.divide(
            np.abs(train_features.T @ centred),
            denominator,
            out=np.zeros(len(feature_names)),
            where=denominator > 0.0,
        )
        counts[_top_indices(correlation, selection_size), 0] += 1.0

        # 2. Sparse linear selection. Picks one of a correlated group somewhat arbitrarily.
        elastic_net = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
            cv=5,
            random_state=random_seed,
            max_iter=20000,
        ).fit(train_features, train_target)
        counts[_top_indices(np.abs(elastic_net.coef_), selection_size), 1] += 1.0

        # 3. Latent projection. Spreads weight across a correlated group rather than
        #    choosing within it -- the house method at Sartorius/DataHow, and the natural
        #    complement to the sparse selector above.
        components = min(2, train_features.shape[1], train_features.shape[0] - 1)
        pls = PLSRegression(n_components=max(components, 1)).fit(train_features, train_target)
        counts[_top_indices(np.abs(np.ravel(pls.coef_)), selection_size), 2] += 1.0

        # 4. Permutation importance, measured on the held-out rows of this fold. The only
        #    method here that asks "does this help prediction?" rather than "does this fit?".
        importance = permutation_importance(
            elastic_net,
            test_features,
            test_target,
            n_repeats=10,
            random_state=random_seed,
        )
        counts[_top_indices(importance.importances_mean, selection_size), 3] += 1.0

    standardised, _ = _standardise(matrix, matrix)
    correlation_matrix = np.atleast_2d(
        np.nan_to_num(np.corrcoef(standardised, rowvar=False), nan=0.0)
    ).astype(np.float64)
    np.fill_diagonal(correlation_matrix, 1.0)
    inverse = np.linalg.pinv(correlation_matrix)
    off_diagonal = np.abs(correlation_matrix) - np.eye(len(feature_names))

    return StabilityTable(
        feature_names=tuple(feature_names),
        frequencies=counts / len(folds),
        method_names=METHOD_NAMES,
        variance_inflation=np.asarray(np.diag(inverse), dtype=np.float64),
        maximum_absolute_correlation=np.max(off_diagonal, axis=1),
        run_count=matrix.shape[0],
    )


def split_complete_and_conditional(
    feature_names: Sequence[str],
    matrix: NDArray[np.float64],
) -> tuple[tuple[str, ...], NDArray[np.float64], tuple[str, ...], NDArray[np.float64]]:
    """Separate always-defined features from event-conditional ones.

    Returns:
        ``(complete_names, complete_matrix, conditional_names, conditional_matrix)``. The
        conditional matrix still contains ``nan``; :func:`conditional_subset` selects the
        runs on which a given conditional feature is defined.
    """
    defined = np.isfinite(matrix).all(axis=0)
    complete = [name for name, flag in zip(feature_names, defined, strict=True) if flag]
    conditional = [name for name, flag in zip(feature_names, defined, strict=True) if not flag]
    return (
        tuple(complete),
        matrix[:, defined],
        tuple(conditional),
        matrix[:, ~defined],
    )


def conditional_subset(
    conditional_names: Sequence[str],
    conditional_matrix: NDArray[np.float64],
    productivity: NDArray[np.float64],
) -> tuple[tuple[str, ...], NDArray[np.float64], NDArray[np.float64]]:
    """The largest block of conditional features and runs with no missing values.

    Conditional features fall into groups that go missing together -- every late-phase
    exposure is undefined for exactly the runs that ended at the split day. Taking the runs
    on which the largest group is defined keeps that group intact rather than intersecting
    across incompatible groups and emptying the table.

    Returns:
        ``(names, matrix, productivity)`` restricted to the retained runs, free of ``nan``.
    """
    if conditional_matrix.shape[1] == 0:
        return (), np.zeros((0, 0)), np.zeros(0)

    # Group features by which runs they are defined on; keep the group covering most runs.
    patterns: dict[bytes, list[int]] = {}
    for column in range(conditional_matrix.shape[1]):
        key = np.isfinite(conditional_matrix[:, column]).tobytes()
        patterns.setdefault(key, []).append(column)

    best_columns = max(
        patterns.values(),
        key=lambda columns: (
            int(np.isfinite(conditional_matrix[:, columns[0]]).sum()) * len(columns)
        ),
    )
    rows = np.isfinite(conditional_matrix[:, best_columns[0]])
    return (
        tuple(conditional_names[column] for column in best_columns),
        conditional_matrix[np.ix_(rows, best_columns)],
        productivity[rows],
    )


def screening_inputs(
    runs: Sequence[ExperimentRun],
    targets: dict[str, float],
) -> tuple[tuple[str, ...], NDArray[np.float64], NDArray[np.float64]]:
    """Assemble the feature frame and the screening target for a set of runs.

    Returns:
        ``(feature_names, matrix, productivity)``.
    """
    feature_names, matrix = features.feature_frame(runs)
    productivity_by_id = evaluation.specific_productivity_targets(runs, targets)
    productivity = np.array(
        [productivity_by_id[run.experiment_id] for run in runs], dtype=np.float64
    )
    return feature_names, matrix, productivity


def format_table(
    table: StabilityTable,
    minimum_frequency: float = DEFAULT_MINIMUM_FREQUENCY,
    minimum_methods: int = DEFAULT_MINIMUM_METHODS,
) -> str:
    """The stability table as text, ordered by mean selection frequency."""
    header = (
        f"{'feature':<26}"
        + "".join(f"{name:>13}" for name in table.method_names)
        + f"{'mean':>8}{'VIF':>8}{'max|r|':>8}  survives"
    )
    lines = [header, "-" * len(header)]
    order = np.argsort(-table.frequencies.mean(axis=1))
    survivors = set(table.survivors(minimum_frequency, minimum_methods))
    for index in order:
        name = table.feature_names[index]
        row = "".join(f"{value:>13.1f}" for value in table.frequencies[index])
        lines.append(
            f"{name:<26}{row}"
            f"{table.frequencies[index].mean():>8.2f}"
            f"{table.variance_inflation[index]:>8.1f}"
            f"{table.maximum_absolute_correlation[index]:>8.2f}"
            f"  {'yes' if name in survivors else ''}"
        )
    return "\n".join(lines)


@dataclass(frozen=True)
class Identifiability:
    """Whether a candidate set's constants are determined, measured across the folds.

    A mechanism set is scored by how well it predicts, but it is *reported* as an equation
    with constants in named units. Those are different requirements, and the second is the one
    that fails quietly: a set can improve cross-validated error while its constants wander
    across folds, because what improved prediction was the extra degree of freedom rather than
    the mechanism it is attributed to.

    What that looks like on the supplied data is worth stating, because it is not subtle once
    the fold fits are laid out. ``metabolic_burden`` is
    ``1/(1 + Lac/K_L + NH4/K_A)``, and across the ten folds it fits

        K_L  1113 to 10000 mM   against a measured lactate range of  0 to 8 mM
        K_A   175 to  562 mM    against a measured ammonia range of  0 to 14 mM

    Both constants land one to three orders of magnitude above the concentrations that exist in
    the data, and ``K_L`` rests on its upper bound in three folds. A half-saturation constant
    far above the observed range means the factor never leaves the neighbourhood of one -- the
    fit is switching the mechanism **off** in every fold, lactate entirely and ammonia nearly
    so. Whatever earned the 3% improvement in cross-validated error, it was not inhibition by
    these metabolites, because no fold ever applied any.

    That is a result about the data and belongs in the report rather than being tuned away. It
    also agrees with what the variable screen found for a different reason: lactate correlates
    **+0.767** with specific productivity, and an inhibition term can only bend downwards, so
    there is no value of ``K_L`` that expresses the association. The mechanism cannot represent
    the sign, so the fit's only remaining move is to disable it.

    Args:
        constant_names: the mechanism constants examined, excluding the variant's own ``kl``.
        spread: per constant, the factor between its largest and smallest fold estimate for a
            log-scaled constant, or its movement as a fraction of its search range otherwise.
        pinned_fraction: per constant, the fraction of folds placing it on a search bound.
        unidentified: constants failing either test, which is what blocks admission.
    """

    constant_names: tuple[str, ...]
    spread: Mapping[str, float]
    pinned_fraction: Mapping[str, float]
    unidentified: tuple[str, ...]

    @property
    def is_identified(self) -> bool:
        return not self.unidentified

    def reason(self) -> str:
        """Why the set was rejected, in a form fit to print."""
        if self.is_identified:
            return ""
        parts: list[str] = []
        for name in self.unidentified:
            part = f"{name} moves {self.spread[name]:.3g}x across folds"
            if self.pinned_fraction[name] > 0.0:
                part += f", on a bound in {100 * self.pinned_fraction[name]:.0f}% of them"
            parts.append(part)
        return "; ".join(parts)


@dataclass(frozen=True)
class MechanismSelection:
    """Which mechanisms earned a place in ``F(z)``, and what every candidate scored.

    Args:
        chosen: the selected mechanism set, in the order they were added.
        baseline_score: cross-validated error with no environmental factor at all.
        trials: every set evaluated, as ``(mechanism_names, score)``, in evaluation order.
            Printed so the choice is auditable rather than asserted.
        rejected: sets that scored well enough to be admitted but whose constants were not
            determined, with the evidence. Kept because a mechanism rejected on
            identifiability is a finding about the data, and silently dropping it would leave
            the trials table looking as though it had simply lost on error.
    """

    chosen: tuple[str, ...]
    baseline_score: float
    trials: tuple[tuple[tuple[str, ...], float], ...]
    rejected: tuple[tuple[tuple[str, ...], float, Identifiability], ...] = ()

    @property
    def improvement(self) -> float:
        """Reduction in cross-validated error against having no factor."""
        chosen_score = dict(self.trials).get(self.chosen, self.baseline_score)
        return self.baseline_score - chosen_score

    def single_mechanism_scores(self) -> dict[str, float]:
        """Score of each mechanism tested on its own, for the report."""
        return {names[0]: score for names, score in self.trials if len(names) == 1}


def select_mechanism_set(
    runs: Sequence[ExperimentRun],
    targets: dict[str, float],
    variant_name: str = "M2",
    candidates: Sequence[str] | None = None,
    fold_count: int = DEFAULT_FOLD_COUNT,
    random_seed: int = 0,
    maximum_mechanisms: int = DEFAULT_MAXIMUM_MECHANISMS,
    minimum_relative_improvement: float = DEFAULT_MINIMUM_RELATIVE_IMPROVEMENT,
    maximum_constant_spread: float = DEFAULT_MAXIMUM_CONSTANT_SPREAD,
    maximum_range_fraction: float = DEFAULT_MAXIMUM_RANGE_FRACTION,
    maximum_pinned_fraction: float = DEFAULT_MAXIMUM_PINNED_FRACTION,
) -> MechanismSelection:
    """Choose the mechanism set by cross-validated prediction error and identifiability.

    **This replaces selecting variables and then mapping them to mechanisms**, which was
    wrong for a reason worth stating: whether a variable helps depends on whether the
    mechanism assigned to it can represent the *sign* of its association, and a variable
    screen cannot see that.

    Concretely, lactate exposure correlates **+0.767** with specific productivity, so it
    topped the variable screen. Its mechanism, ``metabolic_burden``, is an inhibition term
    that can only bend downwards, so no parameter value can express a positive association
    and the fit switched it off. Meanwhile glucose -- correlating +0.719, nearly as strongly,
    and mapping to a Monod term that bends *upwards* -- was dropped by the screen because it
    is collinear with lactate and a sparse selector keeps one of a correlated pair
    arbitrarily. The screen discarded the mechanism that worked and kept the one that could
    not.

    Selecting on cross-validated error removes both failures at once. A mechanism whose shape
    cannot express the data simply fails to improve prediction and is dropped, and between
    two collinear variables the one whose mechanism *expresses* the information wins.

    The search is forward stepwise: score every candidate alone, take the best, then try
    adding each remaining one, and stop when nothing earns its place. That is at most
    ``n(n+1)/2`` cross-validations rather than the ``2^n - 1`` of an exhaustive subset
    search, which matters both for cost and for how much the cross-validation is being
    looked at.

    Two admission tests, not one
    ---------------------------
    Cross-validated error alone is not sufficient, and on this data it demonstrably is not.
    ``metabolic_burden`` reduces the selection error by 3.0% (174.1 to 168.9) -- comfortably
    past the 1% margin -- and then **fails to transfer**: over all 100 runs it is marginally
    worse (287.2 against 286.6), and on the ten held-out 14-day runs it is worse in 98% of
    paired bootstrap resamples. The gain lives entirely in the instrument it was selected on,
    which is what a forward search over 21 candidate sets on one cross-validation should be
    expected to produce.

    The fold fits say why, and they say it *without* consulting held-out data: ``K_L`` rests on
    its upper search bound in three of the ten folds, and in the other seven sits between 1113
    and 3000 mM against a lactate range of 0-8 mM. Every fold switches the mechanism off, so
    the improvement cannot be attributed to the inhibition it names. So
    :class:`Identifiability` is applied as a second test -- a mechanism must both earn its
    margin **and** have constants the folds determine. Both tests read only the runs passed in.

    This is deliberately a rule rather than a judgement made once about one mechanism. A
    selection procedure that needs a human to notice the bad case has not been specified.

    Args:
        runs: training experiments.
        targets: measured final titre per experiment identifier.
        variant_name: model variant the mechanisms are selected for.
        candidates: mechanisms to consider. Defaults to the whole registry.
        fold_count: folds per evaluation.
        random_seed: fixed so folds, and therefore the choice, are reproducible.
        maximum_mechanisms: hard cap on set size, independent of the improvement rule.
        minimum_relative_improvement: a mechanism must reduce cross-validated error by at
            least this fraction to be admitted. Each one costs one or two parameters against
            100 observations, so a gain indistinguishable from noise must not buy a place.
        maximum_constant_spread: largest factor a log-scaled constant may move across folds.
        maximum_range_fraction: largest movement a linear-scaled constant may show, as a
            fraction of its search range.
        maximum_pinned_fraction: largest fraction of folds that may place a constant on a
            search bound.

    Returns:
        The selection, including every set evaluated and every set rejected for
        identifiability with the evidence against it.
    """
    from titre_predictor import model  # imported here: model.py has no need of screening

    variant = model.resolve_variant(variant_name)
    if not variant.needs_factor:
        raise ValueError(f"{variant_name} applies no environmental factor to select for")
    pool = list(candidates) if candidates is not None else sorted(kinetics.MECHANISMS)
    unknown = [name for name in pool if name not in kinetics.MECHANISMS]
    if unknown:
        raise KeyError(f"unknown mechanisms {unknown}; available: {sorted(kinetics.MECHANISMS)}")

    actual = np.array([targets[run.experiment_id] for run in runs], dtype=np.float64)

    def score(mechanism_names: tuple[str, ...]) -> tuple[float, Identifiability]:
        """Cross-validated error and constant stability, from one pass over the folds.

        The identifiability check reuses the fits performed for scoring rather than refitting,
        so the second admission test is free and, more importantly, is measured on exactly the
        folds the score came from.
        """
        collected: list[model.FitDiagnostics] = []

        def fit_fold(
            train_runs: Sequence[ExperimentRun], train_targets: dict[str, float]
        ) -> model.LuedekingPiretModel:
            fitted, diagnostics = model.fit(
                train_runs, train_targets, variant_name, mechanism_names
            )
            collected.append(diagnostics)
            return fitted

        predicted = evaluation.cross_validated_predictions(
            runs, targets, fit_fold, fold_count, random_seed
        )
        return (
            evaluation.root_mean_squared_error(actual, predicted),
            _identifiability(
                mechanism_names,
                collected,
                maximum_constant_spread,
                maximum_range_fraction,
                maximum_pinned_fraction,
            ),
        )

    baseline, _baseline_identifiability = score(())
    trials: list[tuple[tuple[str, ...], float]] = [((), baseline)]
    rejected: list[tuple[tuple[str, ...], float, Identifiability]] = []
    chosen: tuple[str, ...] = ()
    best_score = baseline

    while len(chosen) < maximum_mechanisms:
        remaining = [name for name in pool if name not in chosen]
        if not remaining:
            break
        attempts = [((*chosen, name), *score((*chosen, name))) for name in remaining]
        trials.extend((names, value) for names, value, _ in attempts)

        # Both tests, in the order that keeps the report honest. Filtering to identified sets
        # *before* taking the best means an unidentifiable mechanism cannot block a usable one
        # merely by scoring lower than it -- and the ones it rules out are recorded rather than
        # vanishing into the trials table looking like ordinary losers.
        rejected.extend(
            (names, value, identifiable)
            for names, value, identifiable in attempts
            if not identifiable.is_identified
            and best_score - value >= minimum_relative_improvement * best_score
        )
        admissible = [item for item in attempts if item[2].is_identified]
        if not admissible:
            break
        candidate_set, candidate_score, _identifiable = min(admissible, key=lambda item: item[1])
        if best_score - candidate_score < minimum_relative_improvement * best_score:
            break
        chosen, best_score = candidate_set, candidate_score

    return MechanismSelection(
        chosen=chosen,
        baseline_score=baseline,
        trials=tuple(trials),
        rejected=tuple(rejected),
    )


def _identifiability(
    mechanism_names: Sequence[str],
    fold_diagnostics: Sequence["FitDiagnostics"],
    maximum_constant_spread: float,
    maximum_range_fraction: float,
    maximum_pinned_fraction: float,
) -> Identifiability:
    """Measure how far each mechanism constant moves across the folds.

    Only the mechanisms' own constants are examined. The variant's ``kl`` is excluded because
    it is not what is being admitted -- it is present in every candidate including the empty
    set, so it cannot discriminate between them, and its own stability is reported separately.
    """
    specs = kinetics.parameter_specs(kinetics.resolve(mechanism_names))
    if not specs or not fold_diagnostics:
        return Identifiability((), {}, {}, ())

    names = tuple(spec.name for spec in specs)
    by_name: dict[str, list[float]] = {name: [] for name in names}
    pinned_counts: dict[str, int] = dict.fromkeys(names, 0)
    for diagnostics in fold_diagnostics:
        recorded = dict(
            zip(
                diagnostics.shape_constant_names,
                diagnostics.shape_constant_values,
                strict=True,
            )
        )
        for name in names:
            if name in recorded:
                by_name[name].append(float(recorded[name]))
        for name in diagnostics.pinned_parameters:
            if name in pinned_counts:
                pinned_counts[name] += 1

    fold_total = len(fold_diagnostics)
    spread: dict[str, float] = {}
    pinned_fraction: dict[str, float] = {}
    unidentified: list[str] = []
    for spec in specs:
        values = np.array(by_name[spec.name], dtype=np.float64)
        pinned_fraction[spec.name] = pinned_counts[spec.name] / fold_total
        if values.size == 0:
            spread[spec.name] = float("inf")
            unidentified.append(spec.name)
            continue
        low, high = float(values.min()), float(values.max())
        if spec.logarithmic and low > 0.0:
            spread[spec.name] = high / low
            too_wide = spread[spec.name] > maximum_constant_spread
        else:
            width = spec.maximum - spec.minimum
            spread[spec.name] = (high - low) / width if width > 0.0 else 0.0
            too_wide = spread[spec.name] > maximum_range_fraction
        if too_wide or pinned_fraction[spec.name] > maximum_pinned_fraction:
            unidentified.append(spec.name)

    return Identifiability(
        constant_names=names,
        spread=spread,
        pinned_fraction=pinned_fraction,
        unidentified=tuple(unidentified),
    )


def choose_by_improvement(
    scores: Mapping[str, float],
    ordering: Sequence[str],
    minimum_relative_improvement: float = DEFAULT_MINIMUM_RELATIVE_IMPROVEMENT,
) -> str:
    """Pick from scored candidates, defaulting to the simplest unless one clearly beats it.

    The plain ``min`` this replaces reads a cross-validated RMSE as though it were exact. It
    is not: a difference of a few tenths of a percent between two candidates is inside the
    noise of the fold assignment, and taking the lower one is then a coin toss dressed as a
    decision. Worse, it is a coin toss that systematically favours the more elaborate option,
    because more flexible models have more ways to score marginally better by luck.

    So the first entry of ``ordering`` is the incumbent -- the option to prefer when the data
    cannot tell the difference -- and each later candidate must beat the current incumbent by
    ``minimum_relative_improvement`` to displace it. Candidates equal in complexity are
    ordered arbitrarily, and the rule then keeps whichever came first, which is the honest
    outcome when the measurement cannot separate them.

    This is the same rule and the same margin :func:`select_mechanism_set` applies, exposed
    separately because the model variant and the ridge strength are chosen by the pipeline
    rather than here, and must not be held to a different standard.

    Args:
        scores: candidate name to cross-validated error, lower being better.
        ordering: candidates from simplest to most elaborate. The first is the incumbent.
        minimum_relative_improvement: fraction of the incumbent's error a challenger must
            remove to displace it.

    Returns:
        The selected candidate's name.

    Raises:
        ValueError: if ``ordering`` is empty, or names a candidate absent from ``scores``.
    """
    if not ordering:
        raise ValueError("need at least one candidate to choose from")
    missing = [name for name in ordering if name not in scores]
    if missing:
        raise ValueError(f"no score for {missing}")

    incumbent = ordering[0]
    for challenger in ordering[1:]:
        if (
            scores[incumbent] - scores[challenger]
            >= minimum_relative_improvement * scores[incumbent]
        ):
            incumbent = challenger
    return incumbent


def format_selection(selection: MechanismSelection) -> str:
    """The mechanism search as text, every candidate shown with what it scored."""
    lines = [
        f"{'mechanism set':<52}{'CV RMSE':>10}{'vs none':>10}",
        "-" * 72,
        f"{'(none -- no environmental factor)':<52}{selection.baseline_score:>10.1f}{'':>10}",
    ]
    rejected_sets = {names for names, _value, _identifiable in selection.rejected}
    for names, value in selection.trials:
        if not names:
            continue
        if names == selection.chosen:
            marker = "  <- selected"
        elif names in rejected_sets:
            marker = "  <- NOT IDENTIFIED"
        else:
            marker = ""
        lines.append(
            f"{' + '.join(names):<52}{value:>10.1f}"
            f"{value - selection.baseline_score:>+10.1f}{marker}"
        )

    if selection.rejected:
        lines.append("")
        lines.append(
            "REJECTED ON IDENTIFIABILITY -- these scored well enough to be admitted, but their\n"
            "constants are not determined by the data, so the equation could not be reported\n"
            "with them in it. A constant that moves by orders of magnitude when a tenth of the\n"
            "runs is left out is not an estimate, whatever it does for cross-validated error."
        )
        for names, value, identifiable in selection.rejected:
            lines.append(f"  {' + '.join(names)}  (CV RMSE {value:.1f})")
            lines.append(f"    {identifiable.reason()}")
    return "\n".join(lines)


def select_mechanisms(
    runs: Sequence[ExperimentRun],
    targets: dict[str, float],
    fold_count: int = DEFAULT_FOLD_COUNT,
    random_seed: int = 0,
) -> tuple[tuple[str, ...], StabilityTable]:
    """Run the primary screen and return the mechanisms it licenses.

    The single definition of "how the mechanism set is chosen". Both the training pipeline
    and the uncertainty analysis call this rather than repeating the steps, so the two can
    never disagree about which mechanisms the model contains.

    Only the primary table -- features defined for every run -- licenses mechanisms. The
    conditional table is reported separately and deliberately does not feed selection, since
    its runs skew long and a variable surviving only there is weaker evidence.

    Args:
        runs: training experiments.
        targets: measured final titre per experiment identifier.
        fold_count: folds the selection methods run inside.
        random_seed: fixed so the choice is reproducible.

    Returns:
        ``(mechanism_names, primary_table)``.
    """
    feature_names, matrix, productivity = screening_inputs(runs, targets)
    complete_names, complete_matrix, _conditional_names, _conditional = (
        split_complete_and_conditional(feature_names, matrix)
    )
    table = screen(
        complete_names, complete_matrix, productivity, fold_count, random_seed=random_seed
    )
    return table.mechanisms(), table
