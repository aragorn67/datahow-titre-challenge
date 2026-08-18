"""Score the shipped model against the test targets, once they are supplied.

The brief asks for code prepared so the final results can be incorporated easily and
discussed in the interview. This is that code, and it takes one argument:

    python scripts/score_test_set.py                       # uses the TEMPLATE
    python scripts/score_test_set.py --targets real.csv    # the real targets

**Run it now, against the template.** The template's placeholder titres are all 2000,
so the error figures it prints are meaningless -- but everything else is exercised:
the file parses, all twenty experiments match up, the model predicts, the benchmarks
fit, the report writes. Discovering a column-name mismatch on the day, with the
interviewer watching, is the failure this exists to prevent.

What it reports
---------------
The model against the same benchmarks used throughout Part 1 -- the mean baseline,
PLS and gradient boosting -- all fitted on the 100 training runs and asked to predict
the 20 test runs. That is the comparison that matters: not whether the number is good
in isolation, but whether the mechanistic structure earned its place on data nobody
tuned against.

Per-experiment predictions are printed alongside an extrapolation flag, checked in
**both** directions. Against the shipped model -- fitted on all 100 runs, so its
gammaX range reaches 548.7 -- the test set sits almost entirely inside: none exceed
the maximum and exactly one, Test Exp 9 at gammaX 9.3, falls below the minimum of
20.1. That run was flagged before any of this was fitted as the highest-leverage risk
in the test set, and a small denominator is why.

The frequently quoted "most of the test set is out of range" is true only of the
*leave-duration-out validation*, where the model is deliberately fitted on the 90
short runs alone and the range stops at 242.7. Confusing the two overstates what the
shipped model is being asked to do.

Nothing here is used to fit or select anything. It runs after the fact, on a model
already written to disk.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from titre_predictor import baselines, evaluation, features, model  # noqa: E402
from titre_predictor.data.loading import load_runs, load_targets  # noqa: E402
from titre_predictor.domain import ExperimentRun  # noqa: E402

DATA_DIRECTORY = REPOSITORY_ROOT / "data" / "raw"
TRAIN_DATA = DATA_DIRECTORY / "datahow_interview_train_data.csv"
TRAIN_TARGETS = DATA_DIRECTORY / "datahow_interview_train_targets.csv"
TEST_DATA = DATA_DIRECTORY / "datahow_interview_test_data.csv"
TEMPLATE_TARGETS = DATA_DIRECTORY / "datahow_interview_test_targets-TEMPLATE.csv"

# The template's placeholder value. If every target equals this, the file is the
# placeholder rather than the real thing, and the error figures mean nothing.
TEMPLATE_PLACEHOLDER_TITRE = 2000.0


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def looks_like_the_template(targets: dict[str, float]) -> bool:
    """Whether every target is the template's placeholder value."""
    return bool(targets) and all(value == TEMPLATE_PLACEHOLDER_TITRE for value in targets.values())


def check_experiments_line_up(
    runs: list[ExperimentRun], targets: dict[str, float]
) -> list[ExperimentRun]:
    """Match runs to targets, failing loudly on any mismatch.

    A silent mismatch is the expensive failure here: scoring 18 of 20 runs and
    reporting the result as though it covered all of them would be wrong in a way
    nothing downstream would reveal.
    """
    missing = [run.experiment_id for run in runs if run.experiment_id not in targets]
    if missing:
        raise SystemExit(
            f"no target supplied for {missing}. The targets file must name every "
            f"experiment in {TEST_DATA.name}, using the same identifiers."
        )
    extra = sorted(set(targets) - {run.experiment_id for run in runs})
    if extra:
        print(f"NOTE: {len(extra)} target(s) with no matching run, ignored: {extra}")
    return runs


def fit_benchmarks(
    train_runs: list[ExperimentRun], train_targets: dict[str, float]
) -> dict[str, Any]:
    """The same comparators used in Part 1, fitted on all 100 training runs."""
    fitted: dict[str, Any] = {"mean baseline": model.MeanTitreModel.fit(train_runs, train_targets)}
    for name, fitter in baselines.BASELINES.items():
        fitted[name] = fitter(train_runs, train_targets)
    return fitted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targets",
        type=Path,
        default=TEMPLATE_TARGETS,
        help="test targets CSV. Defaults to the TEMPLATE, whose values are placeholders.",
    )
    parser.add_argument(
        "--artefact",
        type=Path,
        default=REPOSITORY_ROOT / "artefacts" / "titre_model.json",
        help="the fitted model to score.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "artefacts" / "test_set_results.json",
        help="where to write the machine-readable results.",
    )
    arguments = parser.parse_args(argv)

    if not arguments.artefact.is_file():
        raise SystemExit(
            f"no model at {arguments.artefact}. Run `python scripts/screen_and_fit.py` first."
        )

    shipped = model.LuedekingPiretModel.load(arguments.artefact)
    test_runs = load_runs(TEST_DATA)
    test_targets = load_targets(arguments.targets)
    check_experiments_line_up(test_runs, test_targets)

    is_template = looks_like_the_template(test_targets)

    rule(f"TEST SET -- {len(test_runs)} experiments from {arguments.targets.name}")
    if is_template:
        print("!! THIS IS THE TEMPLATE. Every target is the placeholder value 2000, so")
        print("!! every error figure below is meaningless. What this run proves is that")
        print("!! the plumbing works: the file parses, the experiments match, the model")
        print("!! predicts and the report writes. Re-run with --targets pointing at the")
        print("!! real file when it arrives.\n")

    print(shipped.rate_law().splitlines()[0])
    print(f"trained on data hashed ...{_training_hash(arguments.artefact)[-12:]}\n")

    actual = np.array([test_targets[run.experiment_id] for run in test_runs], dtype=np.float64)
    predicted = shipped.predict_many(test_runs)

    train_runs = load_runs(TRAIN_DATA)
    train_targets = load_targets(TRAIN_TARGETS)
    benchmarks = fit_benchmarks(train_runs, train_targets)

    rule("HOW THE MODEL DID, AGAINST THE SAME BENCHMARKS USED THROUGHOUT PART 1")
    print("All fitted on the 100 training runs, all predicting the 20 test runs.\n")
    print(f"{'model':<20}{'RMSE':>10}{'MAE':>10}{'MAPE':>9}{'R2':>8}")
    print("-" * 57)

    # R^2 measures variance explained, so it is undefined when the targets have no
    # variance -- which is exactly the template, where every value is 2000. Reported
    # as unavailable rather than crashing, since a dry run against the template is
    # the whole point of being able to run this before the interview.
    targets_vary = bool(np.ptp(actual) > 0.0)

    results: dict[str, Any] = {}
    rows = {"selected model": predicted}
    rows.update({name: fitted.predict_many(test_runs) for name, fitted in benchmarks.items()})
    for name, values in rows.items():
        entry: dict[str, float | None] = {
            "rmse": evaluation.root_mean_squared_error(actual, values),
            "mae": evaluation.mean_absolute_error(actual, values),
            "mape": evaluation.mean_absolute_percentage_error(actual, values),
            "r2": evaluation.coefficient_of_determination(actual, values) if targets_vary else None,
        }
        results[name] = entry
        r_squared = f"{entry['r2']:>8.3f}" if entry["r2"] is not None else f"{'n/a':>8}"
        print(
            f"{name:<20}{entry['rmse']:>10.1f}{entry['mae']:>10.1f}"
            f"{entry['mape']:>8.1f}%{r_squared}"
        )
    if not targets_vary:
        print("\nR2 is undefined here: every target is identical, so there is no variance")
        print("to explain. It will be reported once the real targets are supplied.")

    rule("PER EXPERIMENT")
    print("The test set IS the extrapolation regime: every run is 14 days, and most")
    print("exceed the training range on biomaterial exposure. Where the model is wrong,")
    print("gammaX is the first thing to look at.\n")
    training_ranges = _training_ranges(arguments.artefact)
    low_cell_days, high_cell_days = training_ranges.get("cell_days") or [
        -float("inf"),
        float("inf"),
    ]

    print(f"{'experiment':<16}{'actual':>10}{'predicted':>11}{'error':>10}{'gammaX':>9}  outside?")
    print("-" * 70)
    per_run = []
    for run, actual_value, predicted_value in zip(test_runs, actual, predicted, strict=True):
        cell_days = features.run_quantities(run).cell_days
        # Both directions. Checking only the upper bound was an earlier mistake here,
        # and it hid the one test run that is genuinely outside: gammaX 9.3 against a
        # training minimum of 20.1. Below the range is as much an extrapolation as
        # above it, and a small denominator makes that run the most fragile of the
        # twenty.
        if cell_days > high_cell_days:
            flag = "above"
        elif cell_days < low_cell_days:
            flag = "BELOW"
        else:
            flag = ""
        print(
            f"{run.experiment_id:<16}{actual_value:>10.1f}{predicted_value:>11.1f}"
            f"{predicted_value - actual_value:>+10.1f}{cell_days:>9.1f}  {flag}"
        )
        per_run.append(
            {
                "experiment_id": run.experiment_id,
                "actual": float(actual_value),
                "predicted": float(predicted_value),
                "error": float(predicted_value - actual_value),
                "cell_days": float(cell_days),
                "beyond_training_range": bool(flag),
                "direction": flag or None,
            }
        )

    outside = [row for row in per_run if row["beyond_training_range"]]
    print(
        f"\n{len(outside)} of {len(per_run)} test runs fall outside the training range for "
        f"gammaX [{low_cell_days:.1f}, {high_cell_days:.1f}]."
    )
    if outside:
        print("  " + ", ".join(f"{row['experiment_id']} ({row['direction']})" for row in outside))
    print(
        "\nNote the shipped model is fitted on all 100 training runs, 14-day runs\n"
        "included, so its gammaX range reaches 548.7 and the test set mostly sits\n"
        "inside it. The leave-duration-out validation is a deliberately harder\n"
        "setting -- fitted on the 90 short runs, whose range stops at 242.7, where\n"
        "14 of these 20 runs would be extrapolations. The two must not be confused."
    )

    payload = {
        "targets_file": str(arguments.targets),
        "is_template": is_template,
        "artefact": str(arguments.artefact),
        "metrics": results,
        "per_experiment": per_run,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rule("ARTEFACT")
    print(f"  {arguments.output.relative_to(REPOSITORY_ROOT)}")
    if is_template:
        print("\nReminder: these numbers came from the TEMPLATE and mean nothing.")
    return 0


def _artefact_block(artefact_path: Path, key: str) -> dict[str, Any]:
    payload = json.loads(artefact_path.read_text(encoding="utf-8"))
    block = payload.get(key)
    return dict(block) if isinstance(block, dict) else {}


def _training_hash(artefact_path: Path) -> str:
    return str(_artefact_block(artefact_path, "provenance").get("training_data_sha256", "unknown"))


def _training_ranges(artefact_path: Path) -> dict[str, list[float]]:
    return {
        name: [float(span[0]), float(span[1])]
        for name, span in _artefact_block(artefact_path, "training_ranges").items()
        if isinstance(span, (list, tuple)) and len(span) == 2
    }


if __name__ == "__main__":
    raise SystemExit(main())
