"""Tests for stage-1 stability screening.

The properties that matter here are about **discipline**, not accuracy:

* every method runs on a fold's training rows only, standardised by that fold's own
  statistics, so nothing about the held-out rows reaches the selection;
* a variable chosen once, or by one method, does not survive -- that is what separates
  stability selection from taking the top of a single ranking;
* the matrix must be fully observed, so the caller is forced to confront event-conditional
  features explicitly rather than imputing them into a duration signal.
"""

import numpy as np
import pytest

from titre_predictor import screening


def _informative_matrix(
    run_count: int = 60, seed: int = 0
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """One genuinely predictive feature, one pure noise feature, and a copy of the first."""
    generator = np.random.default_rng(seed)
    signal = generator.normal(size=run_count)
    noise = generator.normal(size=run_count)
    duplicate = signal + 0.01 * generator.normal(size=run_count)
    matrix = np.column_stack([signal, noise, duplicate])
    productivity = 5.0 * signal + 0.1 * generator.normal(size=run_count)
    return ("signal", "noise", "duplicate"), matrix, productivity


# --- the stability rule ----------------------------------------------------------------


def test_a_predictive_feature_is_chosen_far_more_often_than_noise() -> None:
    names, matrix, productivity = _informative_matrix()

    table = screening.screen(names, matrix, productivity, fold_count=5)

    signal_index = table.feature_names.index("signal")
    noise_index = table.feature_names.index("noise")
    assert table.frequencies[signal_index].mean() > table.frequencies[noise_index].mean()


def test_a_feature_needs_more_than_one_method_to_survive() -> None:
    """The rule that separates stability selection from topping a single ranking."""
    table = screening.StabilityTable(
        feature_names=("chosen_by_one", "chosen_by_three"),
        frequencies=np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0]]),
        method_names=screening.METHOD_NAMES,
        variance_inflation=np.ones(2),
        maximum_absolute_correlation=np.zeros(2),
        run_count=50,
    )

    assert table.survivors(minimum_frequency=0.6, minimum_methods=2) == ("chosen_by_three",)


def test_a_feature_chosen_rarely_does_not_survive() -> None:
    table = screening.StabilityTable(
        feature_names=("occasional",),
        frequencies=np.array([[0.3, 0.3, 0.3, 0.3]]),
        method_names=screening.METHOD_NAMES,
        variance_inflation=np.ones(1),
        maximum_absolute_correlation=np.zeros(1),
        run_count=50,
    )

    assert table.survivors(minimum_frequency=0.6, minimum_methods=2) == ()


def test_survivors_are_ordered_by_mean_selection_frequency() -> None:
    table = screening.StabilityTable(
        feature_names=("weaker", "stronger"),
        frequencies=np.array([[0.7, 0.7, 0.6, 0.6], [1.0, 1.0, 1.0, 1.0]]),
        method_names=screening.METHOD_NAMES,
        variance_inflation=np.ones(2),
        maximum_absolute_correlation=np.zeros(2),
        run_count=50,
    )

    assert table.survivors() == ("stronger", "weaker")


# --- the mechanism mapping --------------------------------------------------------------


def test_lactate_and_ammonia_license_a_single_combined_factor() -> None:
    """They act through one shared denominator, so either surviving licenses one
    two-parameter factor rather than two separate ones."""
    table = screening.StabilityTable(
        feature_names=("exposure_Lac", "exposure_Amm"),
        frequencies=np.ones((2, 4)),
        method_names=screening.METHOD_NAMES,
        variance_inflation=np.ones(2),
        maximum_absolute_correlation=np.zeros(2),
        run_count=50,
    )

    assert table.mechanisms() == ("metabolic_burden",)


def test_a_phase_suffix_does_not_change_the_mechanism() -> None:
    """Late-phase glucose exposure is still glucose."""
    table = screening.StabilityTable(
        feature_names=("exposure_Glc_late",),
        frequencies=np.ones((1, 4)),
        method_names=screening.METHOD_NAMES,
        variance_inflation=np.ones(1),
        maximum_absolute_correlation=np.zeros(1),
        run_count=50,
    )

    assert table.mechanisms() == ("glucose_limitation",)


def test_a_surviving_design_variable_licenses_no_mechanism() -> None:
    """Stirring has no rate-law factor, so surviving screening must not invent one."""
    table = screening.StabilityTable(
        feature_names=("Stir", "duration_days"),
        frequencies=np.ones((2, 4)),
        method_names=screening.METHOD_NAMES,
        variance_inflation=np.ones(2),
        maximum_absolute_correlation=np.zeros(2),
        run_count=50,
    )

    assert table.mechanisms() == ()


# --- the missing-value discipline --------------------------------------------------------


def test_screening_refuses_a_matrix_with_missing_values() -> None:
    """The caller must split conditional features out rather than impute. Imputing would
    encode run duration into a feature named after a metabolite."""
    names, matrix, productivity = _informative_matrix()
    matrix[0, 1] = np.nan

    with pytest.raises(ValueError, match="fully observed"):
        screening.screen(names, matrix, productivity, fold_count=5)


def test_complete_and_conditional_features_are_separated() -> None:
    matrix = np.array([[1.0, np.nan], [2.0, 3.0], [3.0, 4.0]])

    complete_names, complete, conditional_names, conditional = (
        screening.split_complete_and_conditional(("always", "sometimes"), matrix)
    )

    assert complete_names == ("always",)
    assert conditional_names == ("sometimes",)
    assert complete.shape == (3, 1)
    assert conditional.shape == (3, 1)


def test_the_conditional_subset_keeps_features_that_go_missing_together() -> None:
    """Late-phase exposures are all undefined for exactly the runs that ended early, so
    they form one block. Intersecting across incompatible blocks would empty the table."""
    conditional = np.array(
        [
            [np.nan, np.nan, 1.0],
            [1.0, 2.0, np.nan],
            [3.0, 4.0, np.nan],
            [5.0, 6.0, np.nan],
        ]
    )
    productivity = np.array([1.0, 2.0, 3.0, 4.0])

    names, matrix, subset_productivity = screening.conditional_subset(
        ("late_a", "late_b", "other"), conditional, productivity
    )

    assert names == ("late_a", "late_b")
    assert matrix.shape == (3, 2)
    assert np.isfinite(matrix).all()
    np.testing.assert_allclose(subset_productivity, [2.0, 3.0, 4.0])


def test_an_empty_conditional_block_is_handled() -> None:
    names, matrix, productivity = screening.conditional_subset((), np.zeros((5, 0)), np.zeros(5))

    assert names == ()
    assert matrix.size == 0
    assert productivity.size == 0


# --- shape and reporting ------------------------------------------------------------------


def test_screening_reports_one_frequency_per_feature_and_method() -> None:
    names, matrix, productivity = _informative_matrix()

    table = screening.screen(names, matrix, productivity, fold_count=5)

    assert table.frequencies.shape == (len(names), len(screening.METHOD_NAMES))
    assert np.all(table.frequencies >= 0.0)
    assert np.all(table.frequencies <= 1.0)


def test_collinearity_is_reported_so_the_choice_can_be_audited() -> None:
    """A near-duplicate column must show up, since that is what makes any single ranking
    unreliable and stability selection necessary."""
    names, matrix, productivity = _informative_matrix()

    table = screening.screen(names, matrix, productivity, fold_count=5)

    duplicate_index = table.feature_names.index("duplicate")
    assert table.maximum_absolute_correlation[duplicate_index] > 0.9
    assert table.variance_inflation[duplicate_index] > 10.0


def test_mismatched_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="productivity values"):
        screening.screen(("a", "b"), np.zeros((5, 2)), np.zeros(4))

    with pytest.raises(ValueError, match="feature names"):
        screening.screen(("a",), np.zeros((5, 2)), np.zeros(5))


def test_the_formatted_table_names_every_feature_and_method() -> None:
    names, matrix, productivity = _informative_matrix()
    table = screening.screen(names, matrix, productivity, fold_count=5)

    text = screening.format_table(table)

    for name in names:
        assert name in text
    for method in screening.METHOD_NAMES:
        assert method in text


# --- against the real data ------------------------------------------------------------------


def test_the_real_feature_frame_splits_into_complete_and_conditional(
    train_runs: list,
    train_targets: dict[str, float],
) -> None:
    """Pins the split: exactly the late-phase and unreached-shift features are conditional."""
    names, matrix, _productivity = screening.screening_inputs(train_runs, train_targets)

    _complete_names, _complete, conditional_names, _conditional = (
        screening.split_complete_and_conditional(names, matrix)
    )

    assert set(conditional_names) == {
        "exposure_Glc_late",
        "exposure_Gln_late",
        "exposure_Lac_late",
        "exposure_Amm_late",
        "exposure_Lysed_late",
        "exposure_temp_late",
        "exposure_pH_late",
        "vcd_at_tempShift",
        "vcd_at_phShift",
        "vcd_at_FeedEnd",
    }


def test_mechanisms_are_selected_by_prediction_not_by_variable_screening(
    train_runs: list,
    train_targets: dict[str, float],
) -> None:
    """The correction that matters. A variable screen ranks variables and then maps them to
    mechanisms, which cannot see whether the mechanism's shape can represent the SIGN of the
    association. Lactate tops the screen but its inhibition term can only bend downwards, so
    it is switched off; glucose is dropped by the screen but its Monod term does the work.

    Selecting on cross-validated error avoids both failures, so the chosen set must be able
    to differ from what the screen would have licensed."""
    selection = screening.select_mechanism_set(
        train_runs, train_targets, fold_count=3, maximum_mechanisms=2
    )

    assert selection.chosen, "at least one mechanism should earn its place"
    assert selection.improvement > 0.0
    # every chosen set must have been evaluated, and the winner must be the best of them
    scores = dict(selection.trials)
    assert scores[selection.chosen] == min(
        value for names, value in selection.trials if len(names) == len(selection.chosen)
    )


def test_the_selection_reports_every_set_it_evaluated() -> None:
    """The choice has to be auditable rather than asserted."""
    selection = screening.MechanismSelection(
        chosen=("glucose_limitation",),
        baseline_score=577.6,
        trials=(((), 577.6), (("glucose_limitation",), 425.2), (("ph_response",), 439.8)),
    )

    assert selection.improvement == pytest.approx(577.6 - 425.2)
    assert selection.single_mechanism_scores() == {
        "glucose_limitation": 425.2,
        "ph_response": 439.8,
    }


# --- the shared selection margin ----------------------------------------------------------


def test_a_challenger_inside_the_margin_does_not_displace_the_incumbent() -> None:
    """The ridge case that motivated this: 173.0 against 173.8 is a 0.46% gain, and the
    cross-validated RMSE it is read off cannot resolve that. The plain argmin it replaces
    would have taken it."""
    scores = {"0": 173.82, "0.01": 173.05, "0.1": 181.84}

    assert screening.choose_by_improvement(scores, ["0", "0.01", "0.1"], 0.01) == "0"


def test_a_challenger_beyond_the_margin_does_displace_the_incumbent() -> None:
    """The rule must not be a way of never changing anything: the mechanism gain that
    motivated M2 is 41%, far outside the margin, and has to survive it."""
    scores = {"M1": 293.9, "M2": 173.8}

    assert screening.choose_by_improvement(scores, ["M1", "M2"], 0.01) == "M2"


def test_improvements_accumulate_against_the_current_incumbent_not_the_first() -> None:
    """Each step is judged against what it would replace, matching the forward mechanism
    search. Two 2% steps are both adopted even though neither beats the original by 4%."""
    scores = {"a": 100.0, "b": 97.5, "c": 95.0}

    assert screening.choose_by_improvement(scores, ["a", "b", "c"], 0.02) == "c"


def test_the_incumbent_is_kept_when_candidates_cannot_be_separated() -> None:
    """Equal scores mean the measurement has nothing to say, so the ordering decides -- and
    the ordering puts the simpler claim first."""
    scores = {"M2": 200.0, "M3": 200.0}

    assert screening.choose_by_improvement(scores, ["M2", "M3"], 0.01) == "M2"


def test_choosing_from_no_candidates_is_an_error() -> None:
    with pytest.raises(ValueError, match="at least one candidate"):
        screening.choose_by_improvement({"a": 1.0}, [])


def test_an_unscored_candidate_is_named_rather_than_skipped() -> None:
    """A typo in the ordering must not silently drop a candidate from the comparison."""
    with pytest.raises(ValueError, match="M4"):
        screening.choose_by_improvement({"M2": 1.0}, ["M2", "M4"])


# --- the identifiability guard -------------------------------------------------------------


def _diagnostics(values_per_fold, names, pinned_per_fold=None):
    """Fold diagnostics carrying given constant values, for testing the guard alone."""
    from titre_predictor import model

    pinned_per_fold = pinned_per_fold or [() for _ in values_per_fold]
    return [
        model.FitDiagnostics(
            alpha_standard_error=1.0,
            beta_standard_error=1.0,
            coefficient_correlation=0.0,
            residual_standard_deviation=1.0,
            residual_sum_of_squares=1.0,
            training_run_count=80,
            shape_constant_names=("kl", *names),
            shape_constant_values=(0.005, *values),
            pinned_parameters=pinned,
        )
        for values, pinned in zip(values_per_fold, pinned_per_fold, strict=True)
    ]


def test_a_constant_stable_across_folds_is_identified() -> None:
    """K_Q moves by a factor of 1.9 on the real data and must pass."""
    folds = _diagnostics([(0.040,), (0.055,), (0.076,)], ("K_Q",))

    result = screening._identifiability(("glutamine_limitation",), folds, 10.0, 0.25, 0.2)

    assert result.is_identified
    assert result.spread["K_Q"] == pytest.approx(0.076 / 0.040)


def test_a_constant_moving_by_orders_of_magnitude_is_not_identified() -> None:
    folds = _diagnostics([(1.0,), (50.0,), (900.0,)], ("K_Q",))

    result = screening._identifiability(("glutamine_limitation",), folds, 10.0, 0.25, 0.2)

    assert not result.is_identified
    assert result.unidentified == ("K_Q",)
    assert "K_Q" in result.reason()


def test_a_constant_resting_on_a_bound_in_too_many_folds_is_not_identified() -> None:
    """The decisive test on the real data: a constant the grid stopped is not an estimate,
    however narrow its spread happens to look."""
    folds = _diagnostics(
        [(1.0,), (1.05,), (1.1,), (1.0,)],
        ("K_Q",),
        pinned_per_fold=[("K_Q",), ("K_Q",), (), ()],
    )

    result = screening._identifiability(("glutamine_limitation",), folds, 10.0, 0.25, 0.2)

    assert result.spread["K_Q"] < 1.2, "the spread alone would have passed"
    assert not result.is_identified
    assert "bound" in result.reason()


def test_one_awkward_fold_on_a_bound_is_tolerated() -> None:
    """The bound test is a rule about routine behaviour, not a zero-tolerance trip wire."""
    folds = _diagnostics(
        [(1.0,), (1.05,), (1.1,), (1.0,), (1.02,)],
        ("K_Q",),
        pinned_per_fold=[("K_Q",), (), (), (), ()],
    )

    result = screening._identifiability(("glutamine_limitation",), folds, 10.0, 0.25, 0.2)

    assert result.pinned_fraction["K_Q"] == pytest.approx(0.2)
    assert result.is_identified


def test_a_linear_scaled_constant_is_judged_against_its_search_range() -> None:
    """A ratio is meaningless for a sensitivity that may be negative or zero, so theta_T is
    measured as movement relative to its own range of -6 to +6."""
    steady = _diagnostics([(0.07,), (0.10,), (0.12,)], ("theta_T",))
    wandering = _diagnostics([(-5.0,), (0.0,), (5.0,)], ("theta_T",))

    assert screening._identifiability(
        ("temperature_response",), steady, 10.0, 0.25, 0.2
    ).is_identified
    assert not screening._identifiability(
        ("temperature_response",), wandering, 10.0, 0.25, 0.2
    ).is_identified


def test_an_empty_mechanism_set_is_trivially_identified() -> None:
    """The baseline candidate has no constants to determine, and must not be rejected for it."""
    result = screening._identifiability((), _diagnostics([()], ()), 10.0, 0.25, 0.2)

    assert result.is_identified
    assert result.reason() == ""


def test_the_lysis_rate_is_not_judged_as_a_mechanism_constant() -> None:
    """``kl`` belongs to the variant, is present in every candidate including the empty set, and
    so cannot discriminate between them."""
    folds = _diagnostics([(0.040,), (0.055,)], ("K_Q",))

    result = screening._identifiability(("glutamine_limitation",), folds, 10.0, 0.25, 0.2)

    assert "kl" not in result.constant_names
    assert result.constant_names == ("K_Q",)


def test_metabolic_burden_is_rejected_on_the_real_data(
    train_runs: list,
    train_targets: dict[str, float],
) -> None:
    """The finding this guard exists for, asserted end to end.

    ``metabolic_burden`` reduces the selection cross-validation by 3.0%, clearing the 1%
    margin, and every fold nonetheless fits its lactate constant above 1000 mM against a
    measured range of 0-8 mM -- switching off the mechanism it names. It must be reported as
    rejected rather than shipped, and the reason must be recorded.
    """
    from titre_predictor import evaluation

    short_runs, _long_runs = evaluation.split_by_duration(train_runs)

    selection = screening.select_mechanism_set(
        short_runs, train_targets, "M2", fold_count=10, random_seed=0
    )

    assert "metabolic_burden" not in selection.chosen
    assert selection.chosen == ("glutamine_limitation", "glucose_limitation")
    rejected_names = {names for names, _value, _identifiable in selection.rejected}
    assert any("metabolic_burden" in names for names in rejected_names)
    reasons = " ".join(item[2].reason() for item in selection.rejected)
    assert "K_L" in reasons
    # It really did score better on the instrument it was selected on.
    scores = dict(selection.trials)
    assert (
        scores[("glutamine_limitation", "glucose_limitation", "metabolic_burden")]
        < scores[("glutamine_limitation", "glucose_limitation")]
    )


def test_the_rejection_is_visible_in_the_formatted_table() -> None:
    """A set dropped for identifiability must not look like an ordinary loser on error."""
    identifiable = screening.Identifiability(
        constant_names=("K_L",),
        spread={"K_L": 8.98},
        pinned_fraction={"K_L": 0.3},
        unidentified=("K_L",),
    )
    selection = screening.MechanismSelection(
        chosen=("glucose_limitation",),
        baseline_score=293.9,
        trials=(((), 293.9), (("glucose_limitation",), 234.5), (("metabolic_burden",), 168.9)),
        rejected=((("metabolic_burden",), 168.9, identifiable),),
    )

    text = screening.format_selection(selection)

    assert "NOT IDENTIFIED" in text
    assert "K_L" in text
    assert "8.98" in text
