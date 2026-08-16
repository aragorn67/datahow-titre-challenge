"""Train the titre model: screen, select, compare, fit, and write the artefacts.

Run:

    python scripts/screen_and_fit.py                  # the pipeline, ~4 minutes
    python scripts/screen_and_fit.py --uncertainty    # ...plus identifiability, ~25 minutes

**The deliverable is the artefacts, not this script's output.** Terminal text cannot be
diffed, versioned, or loaded by anything, so the pipeline writes:

    artefacts/titre_model.json       the fitted rate law -- what the Part 2 service loads
    artefacts/training_report.json   every number below, machine-readable

Printing the constants and expecting them to be copied into the service by hand is exactly
how a served model drifts from the model that was validated. The same argument the data
layer makes for sharing one feature implementation applies to the fitted parameters.

Both artefacts carry provenance -- a hash of the training data, the seed, package versions,
a timestamp -- so a served prediction can be traced to the run that produced it, and a
re-run can be checked for reproducing it.

Order of operations, which is the methodology:

1a. **Screen** the candidate variables, fold-wise, by stability. This is a **diagnostic**:
    it reports which measurements carry information about specific productivity. It does
    **not** choose the model -- an earlier version did, and that was a real error.
1b. **Select mechanisms** by cross-validated prediction error, forward stepwise.
2.  **Select the variant** M0/M1/M2/M3, and **3.** the ridge strength, the same way.
4.  **Report** performance by ten-fold cross-validation over all 100 runs, with paired
    bootstrap intervals. This is reporting, not selection.
5.  **Fit** the chosen configuration on all 100 runs and write it out.
6.  **Test on the held-out ten**, which informed none of the choices above.
7.  Optionally **interrogate identifiability** -- profiles, bootstrap, prediction intervals,
    mechanism stability.

**Every selection step -- mechanisms, variant, ridge -- runs on the 90 short runs only.**
The ten 14-day runs are read once, at the end. That costs accuracy on the reported
numbers and buys a held-out result that needs no caveat. Selecting on all 100 and then
describing the ten as untouched would be a claim the code does not support.
"""

import argparse
import hashlib
import json
import platform
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from titre_predictor import (  # noqa: E402
    baselines,
    evaluation,
    kinetics,
    model,
    screening,
    uncertainty,
)
from titre_predictor.data.loading import load_runs, load_targets  # noqa: E402
from titre_predictor.domain import ExperimentRun  # noqa: E402

DATA_DIRECTORY = REPOSITORY_ROOT / "data" / "raw"
TRAIN_DATA = DATA_DIRECTORY / "datahow_interview_train_data.csv"
TRAIN_TARGETS = DATA_DIRECTORY / "datahow_interview_train_targets.csv"

FOLD_COUNT = 10
RANDOM_SEED = 0
RIDGE_CANDIDATES = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)

# Ordered simplest first, which is also the order the report prints: M0 fits two
# coefficients, M1 adds kl, M2 and M3 add the mechanism constants on top. The ordering is
# what MINIMUM_IMPROVEMENT is applied against, so it is a decision rather than a formatting
# choice.
VARIANTS_IN_REPORT = ("M0", "M1", "M2", "M3")

# One margin for every selection step -- mechanisms, variant, ridge. Read from screening so
# there is a single number to change and no way for the three to drift apart.
MINIMUM_IMPROVEMENT = screening.DEFAULT_MINIMUM_RELATIVE_IMPROVEMENT
BOOTSTRAP_RESAMPLES = 200
STABILITY_RESAMPLES = 25

SELECTION_NOTE = """
Selecting variables and then mapping them to mechanisms is NOT the same as selecting
mechanisms. Whether a variable helps depends on whether its mechanism can represent the
SIGN of its association, and a variable screen cannot see that.

Lactate correlates +0.767 with specific productivity and tops the screen, but its mechanism
is an inhibition term that can only bend downwards, so no parameter value expresses a
positive association -- the fit switches it off. Glucose correlates +0.719, nearly as
strongly, but is dropped by the screen because it is collinear with lactate and a sparse
selector keeps one of a correlated pair arbitrarily. Glucose maps to a Monod term that bends
upwards, and does the work.

The screen discarded the mechanism that worked and kept the one that could not.
"""


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def file_digest(path: Path) -> str:
    """SHA-256 of a file, so an artefact records which data produced it."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display_path(path: Path) -> Path:
    """Repository-relative when possible, absolute otherwise.

    ``--artefact-directory`` may point anywhere, including outside the repository, so this
    cannot assume the output lands underneath it.
    """
    try:
        return path.relative_to(REPOSITORY_ROOT)
    except ValueError:
        return path


def build_provenance(run_count: int) -> dict[str, Any]:
    """Everything needed to trace a prediction back to the run that produced it."""
    import scipy
    import sklearn

    return {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "training_data_sha256": file_digest(TRAIN_DATA),
        "training_targets_sha256": file_digest(TRAIN_TARGETS),
        "training_run_count": run_count,
        "random_seed": RANDOM_SEED,
        "fold_count": FOLD_COUNT,
        "python": platform.python_version(),
        "package_versions": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
        },
    }


def shape_specs(
    variant_name: str, mechanism_names: tuple[str, ...]
) -> list[kinetics.ParameterSpec]:
    """The shape constants of a model, ordered as the diagnostics report them."""
    specs = list(kinetics.parameter_specs(kinetics.resolve(mechanism_names)))
    if model.resolve_variant(variant_name).needs_lysis_rate:
        specs.insert(0, model.LYSIS_RATE_SPEC)
    return specs


def cross_validated(
    runs: list[ExperimentRun],
    targets: dict[str, float],
    variant: str,
    mechanisms: tuple[str, ...],
    ridge_penalty: float,
) -> np.ndarray:
    """Out-of-fold predictions, refitting every parameter inside each fold."""
    in_force = mechanisms if model.resolve_variant(variant).needs_factor else ()

    def fit_fold(
        train_runs: Sequence[ExperimentRun], train_targets: dict[str, float]
    ) -> model.LuedekingPiretModel:
        fitted, _diagnostics = model.fit(
            train_runs, train_targets, variant, in_force, ridge_penalty=ridge_penalty
        )
        return fitted

    return evaluation.cross_validated_predictions(runs, targets, fit_fold, FOLD_COUNT, RANDOM_SEED)


def run_screening(
    runs: list[ExperimentRun], targets: dict[str, float], report: dict[str, Any]
) -> tuple[str, ...]:
    """Report which variables carry information, then select mechanisms by prediction.

    Deliberately two separate steps. Conflating them was a real error in an earlier version
    of this pipeline -- see :data:`SELECTION_NOTE`.
    """
    rule("STAGE 1a -- VARIABLE SCREEN (diagnostic): which measurements carry information?")
    print("This does NOT choose the model. It reports which measurements are associated with")
    print("specific productivity, and how stably. Mechanisms are chosen in stage 1b, by")
    print("cross-validated prediction error.\n")

    names, matrix, productivity = screening.screening_inputs(runs, targets)
    complete_names, complete_matrix, conditional_names, conditional_matrix = (
        screening.split_complete_and_conditional(names, matrix)
    )
    print(
        f"{len(names)} candidate features over {len(runs)} runs: "
        f"{len(complete_names)} defined for every run, "
        f"{len(conditional_names)} event-conditional."
    )
    print(
        "Event-conditional features are NOT imputed. Whether they exist is near-perfectly\n"
        "determined by run duration, so filling them would encode duration into a feature\n"
        "named after a metabolite or a process shift."
    )

    primary = screening.screen(
        complete_names, complete_matrix, productivity, FOLD_COUNT, random_seed=RANDOM_SEED
    )
    print(f"\n--- primary table: {len(complete_names)} always-defined features ---")
    print(screening.format_table(primary))
    print(f"\nsurvive: {list(primary.survivors()) or 'none'}")
    report["screening"] = {
        "primary_survivors": list(primary.survivors()),
        "primary_frequencies": {
            name: dict(zip(primary.method_names, row.tolist(), strict=True))
            for name, row in zip(primary.feature_names, primary.frequencies, strict=True)
        },
        "conditional_features": list(conditional_names),
        "would_have_licensed": list(primary.mechanisms()),
    }

    subset_names, subset_matrix, subset_productivity = screening.conditional_subset(
        conditional_names, conditional_matrix, productivity
    )
    if subset_names:
        print(
            f"\n--- conditional table: {len(subset_names)} features on the "
            f"{subset_matrix.shape[0]} runs where the event occurred ---"
        )
        print("CAVEAT: this subset skews long, so it is smaller and duration-biased.")
        conditional = screening.screen(subset_names, subset_matrix, subset_productivity, FOLD_COUNT)
        print(screening.format_table(conditional))
        print(f"\nsurvive: {list(conditional.survivors()) or 'none'}")
        report["screening"]["conditional_survivors"] = list(conditional.survivors())

    rule("STAGE 1b -- MECHANISM SELECTION: which factors earn a place in F(z)?")
    print(f"Forward stepwise search on cross-validated error over the {len(runs)} SHORT runs.")
    print("The ten 14-day runs are excluded from selection entirely, so the held-out")
    print("diagnostic at the end is a genuinely clean number. Every parameter is refitted")
    print("inside every fold.\n")
    selection = screening.select_mechanism_set(
        runs, targets, "M2", fold_count=FOLD_COUNT, random_seed=RANDOM_SEED
    )
    print(screening.format_selection(selection))
    print(f"\nselected: {list(selection.chosen) or 'none'}")

    licensed = primary.mechanisms()
    if set(licensed) != set(selection.chosen):
        print(f"\nThe variable screen would have licensed: {list(licensed)}")
        print(SELECTION_NOTE)

    report["mechanism_selection"] = {
        "chosen": list(selection.chosen),
        "baseline_score": selection.baseline_score,
        "improvement": selection.improvement,
        "trials": [[list(trial), value] for trial, value in selection.trials],
        "single_mechanism_scores": selection.single_mechanism_scores(),
        # A mechanism dropped for identifiability is a finding about the data, not an
        # absence. Recorded with its evidence so the report can be read without the
        # terminal output, and so the claim is checkable rather than asserted in prose.
        "rejected_for_identifiability": [
            {
                "mechanisms": list(names),
                "cross_validated_rmse": value,
                "would_have_improved_on_chosen": True,
                "unidentified_constants": list(identifiable.unidentified),
                "constant_spread": dict(identifiable.spread),
                "fraction_of_folds_on_a_bound": dict(identifiable.pinned_fraction),
                "reason": identifiable.reason(),
            }
            for names, value, identifiable in selection.rejected
        ],
    }
    return selection.chosen


def compare_variants(
    runs: list[ExperimentRun],
    targets: dict[str, float],
    actual: np.ndarray,
    mechanisms: tuple[str, ...],
    best_variant: str,
    report: dict[str, Any],
) -> dict[str, np.ndarray]:
    rule(f"PERFORMANCE REPORT -- {FOLD_COUNT}-fold CV over all {len(runs)} runs")
    print("REPORTING, not selection -- the variant and mechanisms were already chosen on the")
    print("short runs alone. This measures how the chosen configuration performs under the")
    print("arrangement that matches deployment: fitted on all 100, predicting new runs.\n")
    print("The data-driven benchmarks are here for a reason the mean baseline cannot serve:")
    print("without them, 'a kinetic model suits this problem' is an assertion. They read the")
    print("SAME aggregate features from the SAME code, on the SAME folds, and are given more")
    print("inputs than the kinetic model uses -- see baselines.py on how the comparison is")
    print("kept fair.\n")

    results: dict[str, np.ndarray] = {
        "mean baseline": evaluation.cross_validated_predictions(
            runs, targets, model.MeanTitreModel.fit, FOLD_COUNT, RANDOM_SEED
        )
    }
    for variant in VARIANTS_IN_REPORT:
        results[variant] = cross_validated(runs, targets, variant, mechanisms, 0.0)
    for name, fitter in baselines.BASELINES.items():
        results[name] = evaluation.cross_validated_predictions(
            runs, targets, fitter, FOLD_COUNT, RANDOM_SEED
        )

    print(f"{'model':<20}{'RMSE':>9}{'90% interval':>20}{'MAPE':>9}{'vs baseline':>13}")
    print("-" * 72)
    report["cross_validation"] = {}
    for name, predictions in results.items():
        point, low, high = evaluation.bootstrap_metric(actual, predictions)
        mape = evaluation.mean_absolute_percentage_error(actual, predictions)
        share = ""
        fraction = None
        if name != "mean baseline":
            *_ignored, fraction = evaluation.paired_bootstrap(
                actual, predictions, results["mean baseline"]
            )
            share = f"{100 * fraction:>11.1f}%"
        print(f"{name:<20}{point:>9.1f}{f'[{low:.0f}, {high:.0f}]':>20}{mape:>8.1f}%{share:>13}")
        report["cross_validation"][name] = {
            "rmse": point,
            "rmse_interval": [low, high],
            "mape": mape,
            "beats_baseline_fraction": fraction,
        }

    rule("ARE THESE DIFFERENCES REAL?  (paired bootstrap, same runs)")
    report["paired_comparisons"] = {}
    comparisons = [("M1", "M2"), ("M2", "M3"), ("M1", "M3"), ("M0", "M1")]
    # The kinetic model against each benchmark, on the runs both predicted. Paired, because an
    # unpaired difference of two RMSEs over 100 runs cannot separate a real gap from fold luck.
    comparisons.extend((name, best_variant) for name in baselines.BASELINES)
    for first, second in comparisons:
        difference, low, high, fraction = evaluation.paired_bootstrap(
            actual, results[second], results[first]
        )
        verdict = "distinguishable" if low > 0.0 or high < 0.0 else "NOT distinguishable"
        print(
            f"  {second} vs {first}: RMSE difference {difference:>+8.1f} "
            f"[{low:>+7.1f}, {high:>+7.1f}]  {second} better in {100 * fraction:5.1f}% "
            f"-- {verdict}"
        )
        report["paired_comparisons"][f"{second}_vs_{first}"] = {
            "difference": difference,
            "interval": [low, high],
            "fraction_better": fraction,
            "distinguishable": bool(low > 0.0 or high < 0.0),
        }
    return results


def select_variant(
    runs: list[ExperimentRun],
    targets: dict[str, float],
    mechanisms: tuple[str, ...],
    report: dict[str, Any],
) -> str:
    """Choose M0/M1/M2/M3 on the selection runs only, never on the held-out ten."""
    rule(f"VARIANT SELECTION -- cross-validated over the {len(runs)} selection runs")
    print("Where does F act? Selected on the same runs as the mechanisms, for the same")
    print("reason: any step that reads the ten 14-day runs disqualifies them as a clean")
    print("diagnostic.\n")
    print(f"A variant must beat the simpler one by {100 * MINIMUM_IMPROVEMENT:.0f}% of its CV")
    print("RMSE to be adopted -- the same margin mechanism selection uses, because it is the")
    print("same measurement and a plain argmin would read it as though it were exact.\n")
    actual = np.array([targets[run.experiment_id] for run in runs], dtype=np.float64)
    scores: dict[str, float] = {}
    for variant in VARIANTS_IN_REPORT:
        predictions = cross_validated(runs, targets, variant, mechanisms, 0.0)
        scores[variant] = evaluation.root_mean_squared_error(actual, predictions)
        print(f"  {variant}  CV RMSE {scores[variant]:>9.1f}")

    # VARIANTS_IN_REPORT is ordered by parameter count: M0 fits two coefficients, M1 adds kl,
    # M2 and M3 add the mechanism constants. M2 before M3 among the two equal-cost placements
    # means a tie is kept at M2, which is the more conservative claim -- it says the
    # environment scales non-growth production without also rescaling the yield per cell.
    best = screening.choose_by_improvement(scores, VARIANTS_IN_REPORT, MINIMUM_IMPROVEMENT)
    argmin = min(scores, key=lambda key: scores[key])
    print(f"\nselected variant: {best}")
    if argmin != best:
        print(
            f"({argmin} scored lowest at {scores[argmin]:.1f} but improves on {best}'s "
            f"{scores[best]:.1f} by only "
            f"{100 * (scores[best] - scores[argmin]) / scores[best]:.2f}%, inside the margin)"
        )
    report["variant_selection"] = {
        "scores": scores,
        "selected": best,
        "lowest_scoring": argmin,
        "minimum_relative_improvement": MINIMUM_IMPROVEMENT,
    }
    return best


def select_ridge(
    runs: list[ExperimentRun],
    targets: dict[str, float],
    actual: np.ndarray,
    best_variant: str,
    mechanisms: tuple[str, ...],
    report: dict[str, Any],
) -> float:
    rule(f"RIDGE STRENGTH -- cross-validated over the {len(runs)} selection runs")
    print("Zero is in the grid, so the fit can decline shrinkage rather than be obliged to")
    print("accept it. Columns are standardised before the penalty applies.\n")
    print(f"A penalty must beat plain least squares by {100 * MINIMUM_IMPROVEMENT:.0f}% of its")
    print("CV RMSE to be adopted, the same margin the mechanism and variant steps use.\n")
    scores: dict[float, float] = {}
    for penalty in RIDGE_CANDIDATES:
        predictions = cross_validated(runs, targets, best_variant, mechanisms, penalty)
        scores[penalty] = evaluation.root_mean_squared_error(actual, predictions)
        print(f"  penalty {penalty:<8g} CV RMSE {scores[penalty]:>9.1f}")

    # Keyed by name so the shared rule can be used; RIDGE_CANDIDATES starts at zero, which is
    # the incumbent here for a different reason from the variants. A penalty is not an extra
    # parameter, so this is not a complexity argument: it is that shrinkage is a deliberate
    # departure from the estimator the coefficients are otherwise reported under, and a
    # departure needs evidence the CV can actually resolve.
    named = {f"{penalty:g}": value for penalty, value in scores.items()}
    ordering = [f"{penalty:g}" for penalty in RIDGE_CANDIDATES]
    chosen = screening.choose_by_improvement(named, ordering, MINIMUM_IMPROVEMENT)
    best = float(chosen)
    argmin = min(scores, key=lambda key: scores[key])
    print(f"\nselected penalty for {best_variant}: {best:g}")
    if argmin != best:
        print(
            f"(penalty {argmin:g} scored lowest at {scores[argmin]:.1f} but improves on "
            f"{scores[best]:.1f} by only "
            f"{100 * (scores[best] - scores[argmin]) / scores[best]:.2f}%, inside the margin)"
        )
    report["ridge"] = {
        "scores": {str(key): value for key, value in scores.items()},
        "selected": best,
        "lowest_scoring": argmin,
        "minimum_relative_improvement": MINIMUM_IMPROVEMENT,
    }
    return best


def held_out_diagnostic(
    runs: list[ExperimentRun],
    targets: dict[str, float],
    mechanisms: tuple[str, ...],
    best_variant: str,
    best_penalty: float,
    report: dict[str, Any],
) -> None:
    rule("HELD-OUT TEST -- leave-duration-out, the ten 14-day runs")
    print("These ten informed NOTHING: not the mechanism set, not the variant, not the ridge")
    print("strength. Every model below is fitted on the 90 short runs only, so this is a")
    print("genuine extrapolation test rather than a diagnostic carrying a caveat.")
    print("Still paired and with intervals: an RMSE from ten runs is noisy enough that only")
    print("paired differences are informative.\n")
    short_runs, long_runs = evaluation.split_by_duration(runs)
    actual = [targets[run.experiment_id] for run in long_runs]

    predictions: dict[str, np.ndarray] = {
        "mean baseline": model.MeanTitreModel.fit(short_runs, targets).predict_many(long_runs)
    }
    for variant in VARIANTS_IN_REPORT:
        in_force = mechanisms if model.resolve_variant(variant).needs_factor else ()
        fitted, _diagnostics = model.fit(
            short_runs,
            targets,
            variant,
            in_force,
            ridge_penalty=best_penalty if variant == best_variant else 0.0,
        )
        predictions[variant] = fitted.predict_many(long_runs)

    # This split is what the benchmarks are really for. Random folds ask the baselines to
    # interpolate, which they do well; this asks them to leave the training distribution.
    benchmark_detail: dict[str, str] = {}
    for name, fitter in baselines.BASELINES.items():
        benchmark = fitter(short_runs, targets)
        predictions[name] = benchmark.predict_many(long_runs)
        benchmark_detail[name] = benchmark.detail

    print(f"{'model':<20}{'RMSE':>9}{'90% interval':>20}{'MAPE':>9}{'beats baseline':>16}")
    print("-" * 75)
    report["leave_duration_out"] = {}
    for name, values in predictions.items():
        point, low, high = evaluation.bootstrap_metric(actual, values)
        mape = evaluation.mean_absolute_percentage_error(actual, values)
        share = ""
        fraction = None
        if name != "mean baseline":
            *_ignored, fraction = evaluation.paired_bootstrap(
                actual, values, predictions["mean baseline"]
            )
            share = f"{100 * fraction:>14.1f}%"
        print(f"{name:<20}{point:>9.1f}{f'[{low:.0f}, {high:.0f}]':>20}{mape:>8.1f}%{share:>16}")
        report["leave_duration_out"][name] = {
            "rmse": point,
            "rmse_interval": [low, high],
            "mape": mape,
            "beats_baseline_fraction": fraction,
            "predicted_range": [float(np.min(values)), float(np.max(values))],
        }
    for name, detail in benchmark_detail.items():
        report["leave_duration_out"][name]["selected_inside_fold"] = detail

    # How each benchmark fails is more informative than that it fails, and the ranges say it
    # without needing a plot: a linear model runs past the observed span, a tree cannot reach
    # it. Printed rather than asserted so the claim in the docs is checkable from the output.
    print(f"\n  measured titre over these ten runs: {min(actual):.0f} to {max(actual):.0f}")
    for name in ("mean baseline", best_variant, *baselines.BASELINES):
        values = predictions[name]
        print(f"  {name:<20} predicts {np.min(values):>7.0f} to {np.max(values):>7.0f}")
    print(
        "  A range that is too narrow is a model that cannot leave its training distribution;\n"
        "  one that is too wide is a model extrapolating without constraint."
    )

    print(
        "\nNote: the cross-validation numbers above and these are NOT comparable. Different\n"
        "tasks -- interpolation across mixed durations, versus extrapolation to an unseen\n"
        "horizon from training data that excludes it."
    )


def identifiability(
    runs: list[ExperimentRun],
    targets: dict[str, float],
    actual: np.ndarray,
    best_variant: str,
    mechanisms: tuple[str, ...],
    diagnostics: model.FitDiagnostics,
    report: dict[str, Any],
) -> None:
    rule("PROFILE LIKELIHOOD -- is each constant actually determined?")
    print("Each constant is pinned across a range and EVERY OTHER parameter re-optimised.")
    print("A slice holding the others fixed would look sharp for any parameter.\n")
    print(f"{'constant':<10}{'unit':<22}{'fitted':>11}{'90% interval':>26}{'max RSS rise':>14}")
    print("-" * 84)
    report["profiles"] = {}
    for name, spec in zip(
        diagnostics.shape_constant_names, shape_specs(best_variant, mechanisms), strict=True
    ):
        # A search range only has to contain the optimum; a profile has to contain the whole
        # interval, which for the exponential sensitivities lies outside the search range.
        value_range = None if spec.logarithmic else (2.0 * spec.minimum, 2.0 * spec.maximum)
        profile = uncertainty.profile_likelihood(
            runs, targets, name, best_variant, mechanisms, point_count=25, value_range=value_range
        )
        flag = "" if profile.is_identified else "  UNBOUNDED"
        print(
            f"{name:<10}{profile.unit:<22}{profile.best_value:>11.4g}"
            f"{f'[{profile.lower:.4g}, {profile.upper:.4g}]{flag}':>26}"
            f"{100 * profile.relative_rise:>13.1f}%"
        )
        report["profiles"][name] = {
            "unit": profile.unit,
            "fitted": profile.best_value,
            "interval": [profile.lower, profile.upper],
            "identified": profile.is_identified,
            "relative_rise": profile.relative_rise,
        }

    rule(f"BOOTSTRAP OVER RUNS -- {BOOTSTRAP_RESAMPLES} resampled datasets")
    print("Runs are resampled, not residuals: a residual bootstrap would assume the model")
    print("structure is correct, which is the assumption in question.\n")
    bootstrap = uncertainty.bootstrap_parameters(
        runs, targets, best_variant, mechanisms, resamples=BOOTSTRAP_RESAMPLES
    )
    print(f"{'parameter':<10}{'point':>12}{'median':>12}{'90% interval':>30}")
    print("-" * 66)
    report["bootstrap"] = {}
    for index, name in enumerate(bootstrap.parameter_names):
        median, low, high = bootstrap.interval(name)
        point = bootstrap.point_estimate[index]
        print(f"{name:<10}{point:>12.4g}{median:>12.4g}{f'[{low:.4g}, {high:.4g}]':>30}")
        report["bootstrap"][name] = {"point": point, "median": median, "interval": [low, high]}
    if bootstrap.failed_resamples:
        print(f"\n{bootstrap.failed_resamples} resamples discarded as singular.")

    rule("PARAMETER CORRELATION across bootstrap draws")
    print("Includes shape-constant uncertainty. Watch alpha/kl: the dead-cell term scales")
    print("as alpha/kl.\n")
    correlation = bootstrap.correlation()
    print(" " * 10 + "".join(f"{name:>10}" for name in bootstrap.parameter_names))
    for row, name in enumerate(bootstrap.parameter_names):
        cells = "".join(
            f"{correlation[row, column]:>+10.3f}"
            for column in range(len(bootstrap.parameter_names))
        )
        print(f"{name:<10}{cells}")
    report["parameter_correlation"] = {
        "names": list(bootstrap.parameter_names),
        "matrix": correlation.tolist(),
    }

    rule("PREDICTION INTERVALS -- how wrong might a predicted titre be?")
    print("Parameter uncertainty ALONE is not a prediction interval. Out-of-fold residual")
    print("scatter is added, then coverage is checked rather than assumed.\n")
    residuals = uncertainty.out_of_fold_residuals(
        runs, targets, best_variant, mechanisms, FOLD_COUNT, RANDOM_SEED
    )
    lower, _median, upper = uncertainty.prediction_intervals(bootstrap, runs, residuals)
    parameter_only = np.array(
        [fitted.predict_many(runs) for fitted in bootstrap.models], dtype=np.float64
    )
    parameter_width = float(
        np.mean(
            np.quantile(parameter_only, 0.95, axis=0) - np.quantile(parameter_only, 0.05, axis=0)
        )
    )
    full_width = float(np.mean(upper - lower))
    achieved = uncertainty.coverage(actual, lower, upper)
    residual_share = 100 * (1 - parameter_width / full_width)

    print(f"  mean 90% width, parameter uncertainty only : {parameter_width:8.1f}")
    print(f"  mean 90% width, including residual scatter : {full_width:8.1f}")
    print(f"  residual scatter accounts for              : {residual_share:7.1f}% of the width")
    print(f"\n  nominal coverage 90%, achieved             : {100 * achieved:7.1f}%")
    print("  (same runs the residuals came from -- a consistency check, not validation)")

    _short, long_runs = evaluation.split_by_duration(runs)
    long_actual = np.array([targets[run.experiment_id] for run in long_runs], dtype=np.float64)
    long_lower, _mid, long_upper = uncertainty.prediction_intervals(bootstrap, long_runs, residuals)
    long_coverage = uncertainty.coverage(long_actual, long_lower, long_upper)
    print(f"\n  coverage on the ten 14-day runs            : {100 * long_coverage:7.1f}%")
    print("  (the extrapolation regime; intervals built from mixed-duration residuals)")
    report["prediction_intervals"] = {
        "parameter_only_width": parameter_width,
        "full_width": full_width,
        "residual_share_percent": residual_share,
        "coverage_all_runs": achieved,
        "coverage_long_runs": long_coverage,
    }

    rule(f"MECHANISM STABILITY -- {STABILITY_RESAMPLES} resamples, selection re-run in each")
    print("The measurable version of the post-selection caveat: would a different draw of 100")
    print("experiments have chosen the same mechanisms? The whole forward selection is re-run")
    print("inside each resample, which is why the count is modest.\n")
    stability = uncertainty.bootstrap_mechanism_stability(
        runs, targets, resamples=STABILITY_RESAMPLES, fold_count=FOLD_COUNT
    )
    for name, fraction in sorted(stability.items(), key=lambda item: -item[1]):
        chosen = " <- in the shipped model" if name in mechanisms else ""
        print(f"  {name:<24}{100 * fraction:>6.1f}%{chosen}")
    report["mechanism_stability"] = stability


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--uncertainty",
        action="store_true",
        help="also run profiles, bootstrap and mechanism stability (adds ~20 minutes)",
    )
    parser.add_argument(
        "--artefact-directory",
        type=Path,
        default=REPOSITORY_ROOT / "artefacts",
        help="where to write titre_model.json and training_report.json",
    )
    arguments = parser.parse_args(argv)

    runs = load_runs(TRAIN_DATA)
    targets = load_targets(TRAIN_TARGETS)
    actual = np.array([targets[run.experiment_id] for run in runs], dtype=np.float64)
    provenance = build_provenance(len(runs))
    report: dict[str, Any] = {"provenance": provenance}

    # EVERY selection step runs on the short runs only. The ten 14-day runs are read once,
    # at the end, as a diagnostic -- and that claim is only true if nothing upstream of it
    # touched them. Selecting on all 100 and then calling the ten "never selected on" is a
    # contradiction the code would not support.
    selection_runs, _held_out = evaluation.split_by_duration(runs)
    selection_actual = np.array(
        [targets[run.experiment_id] for run in selection_runs], dtype=np.float64
    )
    report["selection_protocol"] = {
        "selection_run_count": len(selection_runs),
        "held_out_run_count": len(runs) - len(selection_runs),
        "note": "mechanisms, variant and ridge strength are all selected on the short runs",
    }

    mechanisms = run_screening(selection_runs, targets, report)
    best_variant = select_variant(selection_runs, targets, mechanisms, report)
    best_penalty = select_ridge(
        selection_runs, targets, selection_actual, best_variant, mechanisms, report
    )

    compare_variants(runs, targets, actual, mechanisms, best_variant, report)

    rule("SELECTED MODEL, FITTED ON ALL 100 TRAINING RUNS")
    in_force = mechanisms if model.resolve_variant(best_variant).needs_factor else ()
    shipped, diagnostics = model.fit(
        runs, targets, best_variant, in_force, ridge_penalty=best_penalty
    )
    print(shipped.rate_law())
    print(
        f"\ncoefficient correlation {diagnostics.coefficient_correlation:+.3f} "
        f"(post-selection; the standard errors are not valid as stated)"
    )
    if diagnostics.pinned_parameters:
        print(
            f"\nWARNING -- these constants rest on a search bound and are NOT estimates: "
            f"{list(diagnostics.pinned_parameters)}\n"
            "  A constant at the top of an inhibition range means the fit is switching that\n"
            "  factor OFF: 1/(1 + c/K) -> 1 as K grows. That is a legitimate answer, and it\n"
            "  says the mechanism as formulated does not act."
        )
    report["selected_model"] = {
        "variant": best_variant,
        "ridge_penalty": best_penalty,
        "rate_law": shipped.rate_law(),
        "coefficient_correlation": diagnostics.coefficient_correlation,
        "pinned_parameters": list(diagnostics.pinned_parameters),
        "shape_constants": dict(
            zip(
                diagnostics.shape_constant_names,
                diagnostics.shape_constant_values,
                strict=True,
            )
        ),
    }

    held_out_diagnostic(runs, targets, mechanisms, best_variant, best_penalty, report)

    if arguments.uncertainty:
        identifiability(runs, targets, actual, best_variant, mechanisms, diagnostics, report)

    rule("ARTEFACTS")
    model_path = arguments.artefact_directory / "titre_model.json"
    report_path = arguments.artefact_directory / "training_report.json"
    shipped.save(model_path, provenance=provenance)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")

    reloaded = model.LuedekingPiretModel.load(model_path)
    matches = np.allclose(reloaded.predict_many(runs), shipped.predict_many(runs))
    print(f"  {display_path(model_path)}   the fitted rate law, for /predict")
    print(f"  {display_path(report_path)}  every number above, machine-readable")
    print(f"\n  reload check: predictions identical after save/load -- {matches}")
    if not matches:
        raise SystemExit("saved artefact does not reproduce the fitted model's predictions")


if __name__ == "__main__":
    main()
