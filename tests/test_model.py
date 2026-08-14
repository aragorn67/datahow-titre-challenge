"""Tests for the four nested Luedeking-Piret variants and the fitting procedure.

The tests that carry weight beyond ordinary coverage:

* **the nesting is exact.** M2 and M3 with no mechanisms must reproduce M1's coefficients
  bit for bit, because ``F == 1`` when nothing is in force. If that ever drifts, the
  M1/M2/M3 comparison stops being a comparison of nested hypotheses and becomes a
  comparison of three unrelated models;
* **the recorded results reproduce.** M1 must land in the band already observed, and M0 must
  reproduce its documented failure with a negative ``alpha``. M0 is kept precisely so the
  difference between net and effective growth is demonstrated rather than asserted;
* **the artefact cannot be misapplied.** Coefficients fitted against one mechanism set must
  not silently load against another.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from titre_predictor import evaluation, features, kinetics, model
from titre_predictor.data import schema
from titre_predictor.domain import ExperimentRun

DESIGN_DEFAULTS = {
    schema.DESIGN_DISSOLVED_OXYGEN: 50.0,
    schema.DESIGN_STIRRING: 200.0,
    schema.DESIGN_FEED_RATE_GLUCOSE: 4.0,
    schema.DESIGN_FEED_RATE_GLUTAMINE: 7.0,
    schema.DESIGN_FEED_START: 3.0,
    schema.DESIGN_FEED_END: 11.0,
    schema.DESIGN_TEMPERATURE_SHIFT: 5.0,
    schema.DESIGN_PH_SHIFT: 6.0,
}


def _run(identifier: str, peak: float, point_count: int = 9) -> ExperimentRun:
    """A synthetic run that grows, peaks and declines, so the dead pool is non-trivial."""
    timestamps = np.arange(point_count, dtype=np.float64)
    rise = np.linspace(1.0, peak, point_count // 2 + 1)
    fall = np.linspace(peak * 0.9, peak * 0.4, point_count - rise.size)
    viable = np.concatenate([rise, fall])
    return ExperimentRun(
        experiment_id=identifier,
        timestamps=timestamps,
        design_scalars=dict(DESIGN_DEFAULTS),
        control_profiles={
            schema.CONTROL_TEMPERATURE: np.full(point_count, 37.0),
            schema.CONTROL_PH: np.full(point_count, 7.0),
        },
        observations={
            schema.OBSERVATION_VIABLE_CELL_DENSITY: viable,
            schema.OBSERVATION_LYSED_CELLS: 0.004 * timestamps,
            schema.OBSERVATION_GLUCOSE: np.linspace(20.0, 2.0, point_count),
            schema.OBSERVATION_GLUTAMINE: np.linspace(5.0, 0.5, point_count),
            schema.OBSERVATION_LACTATE: np.linspace(0.0, 6.0, point_count),
            schema.OBSERVATION_AMMONIA: np.linspace(0.0, 9.0, point_count),
        },
    )


def _training_set() -> tuple[list[ExperimentRun], dict[str, float]]:
    runs = [_run(f"Exp {index}", peak) for index, peak in enumerate([10.0, 18.0, 26.0, 34.0, 42.0])]
    quantities = [features.run_quantities(run) for run in runs]
    growth, non_growth = model.design_columns(model.VARIANTS["M1"], quantities, 0.05, (), ())
    targets = {
        run.experiment_id: float(20.0 * g + 3.0 * n)
        for run, g, n in zip(runs, growth, non_growth, strict=True)
    }
    return runs, targets


# --- the variant registry --------------------------------------------------------------


def test_all_four_variants_are_registered() -> None:
    assert sorted(model.VARIANTS) == ["M0", "M1", "M2", "M3"]


def test_only_m0_uses_net_growth() -> None:
    """The single structural difference that M0 exists to demonstrate."""
    assert not model.VARIANTS["M0"].uses_effective_growth
    assert all(model.VARIANTS[name].uses_effective_growth for name in ("M1", "M2", "M3"))


def test_the_factor_is_placed_differently_in_each_variant() -> None:
    assert not model.VARIANTS["M1"].needs_factor
    assert model.VARIANTS["M2"].factor_on_non_growth
    assert not model.VARIANTS["M2"].factor_on_growth
    assert model.VARIANTS["M3"].factor_on_growth
    assert model.VARIANTS["M3"].factor_on_non_growth


def test_m0_needs_no_lysis_rate() -> None:
    """Net growth is read straight off X:VCD, so no dead pool is ever formed."""
    assert not model.VARIANTS["M0"].needs_lysis_rate


def test_an_unknown_variant_is_rejected_with_the_available_names() -> None:
    with pytest.raises(KeyError, match="M3"):
        model.resolve_variant("M9")


# --- design columns ---------------------------------------------------------------------


def test_m0_regresses_on_net_growth_and_m1_on_effective_growth() -> None:
    """The whole point of the comparison: a run that peaks and crashes has synthesised far
    more cells than its endpoint difference shows."""
    quantities = [features.run_quantities(_run("Exp 1", 30.0))]

    net, _ = model.design_columns(model.VARIANTS["M0"], quantities, 0.05, (), ())
    effective, _ = model.design_columns(model.VARIANTS["M1"], quantities, 0.05, (), ())

    assert effective[0] > net[0]


def test_m2_leaves_the_growth_column_unweighted_but_m3_weights_it() -> None:
    quantities = [features.run_quantities(_run("Exp 1", 30.0))]
    mechanisms = kinetics.resolve(["temperature_response"])

    m2_growth, m2_non_growth = model.design_columns(
        model.VARIANTS["M2"], quantities, 0.05, mechanisms, [0.5]
    )
    m3_growth, m3_non_growth = model.design_columns(
        model.VARIANTS["M3"], quantities, 0.05, mechanisms, [0.5]
    )

    assert m2_non_growth[0] == pytest.approx(m3_non_growth[0])
    assert m2_growth[0] != pytest.approx(m3_growth[0])


def test_with_no_mechanisms_every_variant_but_m0_gives_identical_columns() -> None:
    """``F == 1``, so M2 and M3 collapse onto M1 exactly rather than approximately."""
    quantities = [features.run_quantities(_run("Exp 1", 30.0))]

    columns = [
        model.design_columns(model.VARIANTS[name], quantities, 0.05, (), ())
        for name in ("M1", "M2", "M3")
    ]

    for growth, non_growth in columns[1:]:
        assert growth[0] == pytest.approx(columns[0][0][0])
        assert non_growth[0] == pytest.approx(columns[0][1][0])


# --- fitting -----------------------------------------------------------------------------


def test_the_fit_recovers_coefficients_from_data_built_with_them() -> None:
    """Targets constructed as 20*growth + 3*cell_days at kl = 0.05 must fit back to that."""
    runs, targets = _training_set()

    fitted, _diagnostics = model.fit(runs, targets, variant_name="M1")

    assert fitted.alpha == pytest.approx(20.0, rel=0.05)
    assert fitted.beta == pytest.approx(3.0, rel=0.05)


def test_m3_with_no_mechanisms_reproduces_m1_exactly() -> None:
    """The nesting, at the level of the fitted parameters rather than the columns."""
    runs, targets = _training_set()

    m1, _ = model.fit(runs, targets, variant_name="M1")
    m3, _ = model.fit(runs, targets, variant_name="M3", mechanism_names=())

    assert m3.alpha == pytest.approx(m1.alpha)
    assert m3.beta == pytest.approx(m1.beta)
    assert m3.lysis_rate_constant == pytest.approx(m1.lysis_rate_constant)


def test_the_relative_loss_gives_a_different_answer_from_the_absolute_one() -> None:
    """Weights of 1/y^2 are squared relative error, which is a different question -- and
    the one aligned with the recorded weakness on MAPE."""
    runs, targets = _training_set()
    targets = {
        key: value * (1.0 + 0.3 * index) for index, (key, value) in enumerate(targets.items())
    }

    absolute, _ = model.fit(runs, targets, variant_name="M1", loss="absolute")
    relative, _ = model.fit(runs, targets, variant_name="M1", loss="relative")

    assert absolute.alpha != pytest.approx(relative.alpha)


def test_fitting_needs_enough_runs_for_a_residual() -> None:
    runs, targets = _training_set()

    with pytest.raises(ValueError, match="at least three runs"):
        model.fit(runs[:2], targets, variant_name="M1")


def test_a_run_without_a_target_is_named() -> None:
    runs, targets = _training_set()
    del targets["Exp 0"]

    with pytest.raises(KeyError, match="Exp 0"):
        model.fit(runs, targets, variant_name="M1")


def test_mechanisms_given_to_a_variant_without_a_factor_are_rejected() -> None:
    """Silently ignoring them would let a comparison report M1 under M2's name."""
    runs, targets = _training_set()

    with pytest.raises(ValueError, match="applies no environmental factor"):
        model.fit(runs, targets, variant_name="M1", mechanism_names=["temperature_response"])


def test_the_relative_loss_rejects_a_non_positive_target() -> None:
    runs, targets = _training_set()
    targets["Exp 0"] = 0.0

    with pytest.raises(ValueError, match="strictly positive targets"):
        model.fit(runs, targets, variant_name="M1", loss="relative")


def test_diagnostics_report_the_shape_constants_by_name() -> None:
    runs, targets = _training_set()
    # Perturb the targets so there is residual scatter; the unperturbed set is generated
    # exactly from the model and would fit perfectly, leaving the correlation undefined.
    targets = {
        key: value * factor
        for (key, value), factor in zip(
            targets.items(), [1.05, 0.94, 1.02, 0.97, 1.06], strict=True
        )
    }

    _fitted, diagnostics = model.fit(
        runs, targets, variant_name="M3", mechanism_names=["temperature_response"]
    )

    assert diagnostics.shape_constant_names == ("kl", "theta_T")
    assert len(diagnostics.shape_constant_values) == 2
    assert -1.0 <= diagnostics.coefficient_correlation <= 1.0


def test_an_exact_fit_reports_an_undefined_correlation_rather_than_zero() -> None:
    """Zero residual variance makes the coefficient covariance zero, so the correlation is
    0/0. Reporting nan says 'not determined'; reporting 0.0 would claim the coefficients are
    independent, which is the opposite of what a degenerate fit means."""
    runs, targets = _training_set()

    _fitted, diagnostics = model.fit(runs, targets, variant_name="M1")

    assert diagnostics.residual_sum_of_squares == pytest.approx(0.0, abs=1e-6)
    assert np.isnan(diagnostics.coefficient_correlation)


# --- prediction and the artefact ----------------------------------------------------------


def test_prediction_matches_the_columns_the_fit_used() -> None:
    runs, targets = _training_set()
    fitted, _ = model.fit(runs, targets, variant_name="M1")

    quantities = [features.run_quantities(runs[0])]
    growth, non_growth = model.design_columns(
        model.VARIANTS["M1"], quantities, fitted.lysis_rate_constant, (), ()
    )

    assert fitted.predict(runs[0]) == pytest.approx(
        fitted.alpha * growth[0] + fitted.beta * non_growth[0]
    )


def test_predict_many_preserves_run_order() -> None:
    runs, targets = _training_set()
    fitted, _ = model.fit(runs, targets, variant_name="M1")

    predictions = fitted.predict_many(runs)

    assert predictions.shape == (len(runs),)
    np.testing.assert_allclose(predictions[0], fitted.predict(runs[0]))


def test_the_artefact_round_trips(tmp_path: Path) -> None:
    runs, targets = _training_set()
    fitted, _ = model.fit(
        runs, targets, variant_name="M3", mechanism_names=["temperature_response"]
    )

    artefact = tmp_path / "nested" / "model.json"
    fitted.save(artefact)
    restored = model.LuedekingPiretModel.load(artefact)

    assert restored == fitted
    assert restored.predict(runs[0]) == pytest.approx(fitted.predict(runs[0]))


def test_an_artefact_whose_parameters_do_not_match_its_mechanisms_is_rejected() -> None:
    """Guards the case that matters: coefficients fitted against one mechanism set being
    applied to another, which would produce confident wrong titres rather than an error."""
    payload = {
        "variant": "M3",
        "alpha": 1.0,
        "beta": 1.0,
        "lysis_rate_constant": 0.05,
        "mechanisms": ["metabolic_burden"],  # takes two constants
        "mechanism_parameters": [40.0],  # but only one is stored
        "loss": "absolute",
    }

    with pytest.raises(ValueError, match="mechanism parameters"):
        model.LuedekingPiretModel.from_dict(payload)


def test_an_artefact_with_an_unknown_variant_is_rejected(tmp_path: Path) -> None:
    artefact = tmp_path / "model.json"
    artefact.write_text(
        json.dumps(
            {
                "variant": "M9",
                "alpha": 1.0,
                "beta": 1.0,
                "lysis_rate_constant": 0.05,
                "mechanisms": [],
                "mechanism_parameters": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(KeyError):
        model.LuedekingPiretModel.load(artefact)


def test_the_rate_law_is_printed_with_constants_and_units() -> None:
    """The deliverable is a mechanistic equation that can be checked against literature,
    not a coefficient table."""
    runs, targets = _training_set()
    fitted, _ = model.fit(
        runs, targets, variant_name="M3", mechanism_names=["temperature_response"]
    )

    text = fitted.rate_law()

    assert "qP(t)" in text
    assert "mu_eff(t)" in text
    assert "F(z(t))" in text
    assert "theta_T" in text
    assert "per degC" in text
    assert "1/day" in text


def test_m0_prints_net_growth_rather_than_effective() -> None:
    runs, targets = _training_set()
    fitted, _ = model.fit(runs, targets, variant_name="M0")

    assert "mu_net(t)" in fitted.rate_law()


# --- the baseline -------------------------------------------------------------------------


def test_the_mean_baseline_ignores_its_input() -> None:
    runs, targets = _training_set()

    baseline = model.MeanTitreModel.fit(runs, targets)

    assert baseline.predict(runs[0]) == pytest.approx(baseline.predict(runs[-1]))
    assert baseline.mean_titre == pytest.approx(float(np.mean(list(targets.values()))))


def test_the_mean_baseline_needs_at_least_one_run() -> None:
    with pytest.raises(ValueError, match="at least one run"):
        model.MeanTitreModel.fit([], {})


# --- against the real data ------------------------------------------------------------------


def test_m1_reproduces_the_recorded_held_out_result(
    train_runs: list[ExperimentRun],
    train_targets: dict[str, float],
) -> None:
    """M1 is the same model as the one already measured, so a material deviation would mean
    a porting bug rather than a discovery. The monotone lysate fit replaces a 4-point slope
    window, which shifts it slightly; the band below is the range already observed across
    window choices."""
    short, long_runs = evaluation.split_by_duration(train_runs)
    fitted, _ = model.fit(short, train_targets, variant_name="M1")

    actual = [train_targets[run.experiment_id] for run in long_runs]
    error = evaluation.root_mean_squared_error(actual, fitted.predict_many(long_runs))

    assert 1430.0 < error < 1648.0
    assert fitted.alpha > 0.0, "effective growth must contribute positively to titre"


def test_m0_reproduces_its_documented_failure(
    train_runs: list[ExperimentRun],
    train_targets: dict[str, float],
) -> None:
    """The reason M0 is kept. A negative alpha says the model believes secreted antibody
    disappears as cells die, which is how net growth was identified as the wrong quantity."""
    short, long_runs = evaluation.split_by_duration(train_runs)
    fitted, _ = model.fit(short, train_targets, variant_name="M0")

    actual = [train_targets[run.experiment_id] for run in long_runs]
    error = evaluation.root_mean_squared_error(actual, fitted.predict_many(long_runs))

    assert fitted.alpha < 0.0
    assert error > 2500.0


def test_m0_is_beaten_by_the_mean_baseline_and_m1_beats_it(
    train_runs: list[ExperimentRun],
    train_targets: dict[str, float],
) -> None:
    """The ordering that justifies the effective-growth identity."""
    short, long_runs = evaluation.split_by_duration(train_runs)
    actual = [train_targets[run.experiment_id] for run in long_runs]

    def held_out_error(predictions: np.ndarray) -> float:
        return evaluation.root_mean_squared_error(actual, predictions)

    baseline = held_out_error(
        model.MeanTitreModel.fit(short, train_targets).predict_many(long_runs)
    )
    m0 = held_out_error(
        model.fit(short, train_targets, variant_name="M0")[0].predict_many(long_runs)
    )
    m1 = held_out_error(
        model.fit(short, train_targets, variant_name="M1")[0].predict_many(long_runs)
    )

    assert m1 < baseline < m0


def test_the_coefficients_are_strongly_correlated_on_the_real_data(
    train_runs: list[ExperimentRun],
    train_targets: dict[str, float],
) -> None:
    """A known weakness, pinned so it is not mistaken for a healthy fit. The two regressors
    measure closely related quantities, so alpha and beta are not separately identified."""
    short, _ = evaluation.split_by_duration(train_runs)

    _fitted, diagnostics = model.fit(short, train_targets, variant_name="M1")

    assert diagnostics.coefficient_correlation < -0.9
