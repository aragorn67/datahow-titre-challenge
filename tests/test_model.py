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


@pytest.mark.parametrize(
    ("mechanism_names", "parameters"),
    [
        ((), ()),
        (("glucose_limitation",), (5.0,)),
        (("glutamine_limitation", "glucose_limitation"), (0.5, 5.0)),
        (("metabolic_burden", "ph_response", "lysate_inhibition"), (12.0, 9.0, 0.4, 0.05)),
    ],
)
@pytest.mark.parametrize("variant_name", ["M1", "M2", "M3"])
def test_the_search_and_the_prediction_path_build_the_same_columns(
    variant_name: str,
    mechanism_names: tuple[str, ...],
    parameters: tuple[float, ...],
) -> None:
    """The vectorised search must minimise the quantity the model predicts with.

    Two implementations of the same quadrature exist because the search is vectorised over
    thousands of candidates and :func:`model.design_columns` is not. They are only allowed to
    exist if they agree, and they must agree for **more than one mechanism**, where the
    difference between a product of interval means and the interval mean of the product first
    appears -- an earlier version disagreed by up to 14% per run there, so ``alpha`` and
    ``beta`` were fitted against columns no prediction ever used.
    """
    variant = model.resolve_variant(variant_name)
    if mechanism_names and not variant.needs_factor:
        pytest.skip(f"{variant_name} applies no factor")
    lysis_rate = 0.05
    runs = [_run(f"Exp {index}", peak) for index, peak in enumerate([12.0, 24.0, 36.0, 48.0])]
    quantities = [features.run_quantities(run) for run in runs]
    mechanisms = kinetics.resolve(mechanism_names)
    targets = np.array([900.0, 1800.0, 2700.0, 3600.0])
    weights = np.ones(len(runs))

    # A single-point grid per constant, so the search evaluates exactly this candidate and
    # its coefficients are comparable term by term against the prediction path's.
    grids = [np.array([lysis_rate])] + [np.array([value]) for value in parameters]
    search_residual, search_alpha, search_beta, _shape = model._search(
        variant,
        model._pad(quantities),
        mechanisms,
        [],
        grids,
        targets,
        weights,
        model.DEFAULT_CHUNK_SIZE,
        0.0,
    )

    growth, non_growth = model.design_columns(
        variant, quantities, lysis_rate, mechanisms, parameters
    )
    alpha, beta, residual = model._solve(growth[None, :], non_growth[None, :], targets, weights)

    assert search_alpha == pytest.approx(float(alpha[0]), rel=1e-10)
    assert search_beta == pytest.approx(float(beta[0]), rel=1e-10)
    assert search_residual == pytest.approx(float(residual[0]), rel=1e-10)


def test_the_non_growth_column_is_trapezoidal_in_the_product_not_in_each_average() -> None:
    """``INT F*Xv dt`` is a mean of products, and the two differ when ``F`` and ``Xv`` move.

    Hand-computed on two intervals so the assertion does not restate the implementation. The
    product-of-means form that this replaces would give a different number here, and would
    give it in a knowable direction: ``F`` falls as glucose depletes while ``Xv`` rises.
    """
    run = _run("Exp 1", 30.0)
    quantities = features.run_quantities(run)
    mechanisms = kinetics.resolve(["glucose_limitation"])
    half_saturation = 5.0

    weighted = features.interval_weighted_cell_days(mechanisms, quantities, [half_saturation])

    glucose = run.observations[schema.OBSERVATION_GLUCOSE]
    viable = run.observations[schema.OBSERVATION_VIABLE_CELL_DENSITY]
    factor = glucose / (half_saturation + glucose)
    product = factor * viable
    for index in range(2):
        step = run.timestamps[index + 1] - run.timestamps[index]
        expected = 0.5 * (product[index] + product[index + 1]) * step
        assert weighted[index] == pytest.approx(expected)
        # The form it replaces, stated explicitly so the difference is on the record.
        product_of_means = (
            0.5
            * (factor[index] + factor[index + 1])
            * 0.5
            * (viable[index] + viable[index + 1])
            * step
        )
        assert weighted[index] != pytest.approx(product_of_means)


def test_the_weighted_cell_days_reduce_to_plain_cell_days_with_no_mechanisms() -> None:
    """What makes M2 nest M1 *exactly*: with ``F == 1`` the weighted quadrature is the plain
    one, so the nesting is a property of the arithmetic rather than a tolerance."""
    quantities = features.run_quantities(_run("Exp 1", 30.0))

    weighted = features.interval_weighted_cell_days((), quantities, ())

    assert weighted == pytest.approx(quantities.interval_cell_days, rel=1e-15)


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


def test_an_exact_fit_never_reports_the_coefficients_as_independent() -> None:
    """On data generated exactly from the model the residual collapses to rounding error,
    and the coefficient correlation is then either undefined (0/0, reported as nan) or
    numerically near -1. Both say 'not determined'. What must never appear is a value near
    zero, which would claim alpha and beta are independently identified -- the opposite of
    what a degenerate fit means."""
    runs, targets = _training_set()

    _fitted, diagnostics = model.fit(runs, targets, variant_name="M1")
    correlation = diagnostics.coefficient_correlation

    assert diagnostics.residual_sum_of_squares == pytest.approx(0.0, abs=1e-3)
    assert np.isnan(correlation) or abs(correlation) > 0.9


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


# --- the ridge penalty ---------------------------------------------------------------


def test_a_zero_penalty_recovers_ordinary_least_squares_exactly() -> None:
    """Zero must remain in the search grid, so the fit can decline shrinkage rather than
    be obliged to accept it."""
    runs, targets = _training_set()
    targets = {
        key: value * factor
        for (key, value), factor in zip(
            targets.items(), [1.05, 0.94, 1.02, 0.97, 1.06], strict=True
        )
    }

    plain, _ = model.fit(runs, targets, variant_name="M1")
    zero_penalty, _ = model.fit(runs, targets, variant_name="M1", ridge_penalty=0.0)

    assert zero_penalty.alpha == pytest.approx(plain.alpha)
    assert zero_penalty.beta == pytest.approx(plain.beta)


def test_a_larger_penalty_shrinks_the_coefficients() -> None:
    runs, targets = _training_set()
    targets = {
        key: value * factor
        for (key, value), factor in zip(
            targets.items(), [1.05, 0.94, 1.02, 0.97, 1.06], strict=True
        )
    }

    light, _ = model.fit(runs, targets, variant_name="M1", ridge_penalty=1e-3)
    heavy, _ = model.fit(runs, targets, variant_name="M1", ridge_penalty=1.0)

    assert abs(heavy.alpha) < abs(light.alpha)


def test_the_penalty_acts_on_standardised_columns(
    train_runs: list[ExperimentRun],
    train_targets: dict[str, float],
) -> None:
    """alpha multiplies cells synthesised and beta multiplies cell-days -- quantities an
    order of magnitude apart. An unstandardised penalty would shrink them by an accident of
    scale, so equal shrinkage must appear in the standardised coefficients, not the raw
    ones."""
    short, _long = evaluation.split_by_duration(train_runs)
    unpenalised, _ = model.fit(short, train_targets, variant_name="M1")
    penalised, _ = model.fit(short, train_targets, variant_name="M1", ridge_penalty=0.1)

    quantities = [features.run_quantities(run) for run in short]
    growth, non_growth = model.design_columns(
        model.VARIANTS["M1"], quantities, unpenalised.lysis_rate_constant, (), ()
    )
    growth_scale = float(np.sqrt(np.mean(growth**2)))
    non_growth_scale = float(np.sqrt(np.mean(non_growth**2)))

    # In standardised units both coefficients must shrink; in raw units they need not.
    assert abs(penalised.alpha * growth_scale) < abs(unpenalised.alpha * growth_scale)
    assert abs(penalised.alpha * growth_scale) + abs(penalised.beta * non_growth_scale) < abs(
        unpenalised.alpha * growth_scale
    ) + abs(unpenalised.beta * non_growth_scale)


def test_the_penalty_survives_the_artefact_round_trip(tmp_path: Path) -> None:
    runs, targets = _training_set()
    fitted, _ = model.fit(runs, targets, variant_name="M1", ridge_penalty=0.01)

    artefact = tmp_path / "model.json"
    fitted.save(artefact)

    assert model.LuedekingPiretModel.load(artefact).ridge_penalty == pytest.approx(0.01)


def test_a_constant_resting_on_its_search_bound_is_reported(
    train_runs: list[ExperimentRun],
    train_targets: dict[str, float],
) -> None:
    """A parameter at its bound is not an estimate. Here the lactate and ammonia constants
    run to the top of their range, which is the fit switching that factor off -- a real
    answer, but one that must be visible rather than quoted as a fitted value."""
    short, _long = evaluation.split_by_duration(train_runs)

    _fitted, diagnostics = model.fit(
        short, train_targets, variant_name="M2", mechanism_names=["metabolic_burden"]
    )

    assert "K_L" in diagnostics.pinned_parameters


def test_a_saved_artefact_reproduces_its_predictions_exactly(tmp_path: Path) -> None:
    """The training/serving guard. The inference service loads this file, so if a reloaded
    model predicted differently the service would be silently serving a different model from
    the one that was validated -- the same failure the shared feature layer exists to
    prevent, one level up."""
    runs, targets = _training_set()
    fitted, _ = model.fit(runs, targets, variant_name="M2", mechanism_names=["ph_response"])

    artefact = tmp_path / "model.json"
    fitted.save(artefact)

    np.testing.assert_allclose(
        model.LuedekingPiretModel.load(artefact).predict_many(runs), fitted.predict_many(runs)
    )


def test_provenance_is_stored_and_ignored_on_load(tmp_path: Path) -> None:
    """Provenance travels with the parameters so a served prediction can be traced to the
    run that produced it, but it must not be needed to rebuild the model."""
    runs, targets = _training_set()
    fitted, _ = model.fit(runs, targets, variant_name="M1")
    artefact = tmp_path / "model.json"

    fitted.save(artefact, provenance={"training_data_sha256": "abc123", "random_seed": 0})

    stored = json.loads(artefact.read_text(encoding="utf-8"))
    assert stored["provenance"]["training_data_sha256"] == "abc123"
    assert model.LuedekingPiretModel.load(artefact) == fitted
