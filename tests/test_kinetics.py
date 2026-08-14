"""Tests for the environmental factor and the rate-law forms it is built from.

Two properties carry real argumentative weight elsewhere in the project and are therefore
asserted here rather than assumed:

* the saturating forms are **bounded and monotone**, which is the whole justification for
  preferring them to linear or polynomial terms given that the test runs reach lysate 1.02
  against a training maximum of 0.53, and glucose 56.8 against 44.0;
* an empty mechanism set gives ``F == 1`` **exactly**, which is what makes M1 a special case
  of M2 and M3 rather than an approximation of them.
"""

import numpy as np
import pytest

from titre_predictor import kinetics
from titre_predictor.data import schema
from titre_predictor.domain import ExperimentRun

# --- Monod limitation ----------------------------------------------------------------


def test_monod_is_one_half_at_the_half_saturation_constant() -> None:
    """This is what makes K readable as a half-saturation constant in mM."""
    assert kinetics.monod(np.array([4.0]), half_saturation=4.0) == pytest.approx(0.5)


def test_monod_vanishes_at_zero_substrate_and_saturates_at_one() -> None:
    values = kinetics.monod(np.array([0.0, 1e9]), half_saturation=4.0)

    assert values[0] == 0.0
    assert values[1] == pytest.approx(1.0)


def test_monod_increases_with_substrate() -> None:
    values = kinetics.monod(np.linspace(0.0, 60.0, 50), half_saturation=4.0)

    assert np.all(np.diff(values) > 0)


def test_monod_stays_bounded_far_beyond_the_training_range() -> None:
    """The extrapolation-safety claim: test glucose reaches 56.8 against a training
    maximum of 44.0, and the factor must flatten rather than diverge."""
    values = kinetics.monod(np.array([44.0, 56.8, 1e6]), half_saturation=4.0)

    assert np.all(values <= 1.0)
    assert np.all(values > 0.0)


def test_monod_rejects_a_non_positive_constant() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        kinetics.monod(np.array([1.0]), half_saturation=0.0)


# --- inhibition ----------------------------------------------------------------------


def test_inhibition_is_one_half_at_the_inhibition_constant() -> None:
    assert kinetics.inhibition(np.array([0.3]), inhibition_constant=0.3) == pytest.approx(0.5)


def test_inhibition_is_one_when_the_inhibitor_is_absent() -> None:
    assert kinetics.inhibition(np.array([0.0]), inhibition_constant=0.3) == pytest.approx(1.0)


def test_inhibition_decreases_with_concentration() -> None:
    values = kinetics.inhibition(np.linspace(0.0, 2.0, 50), inhibition_constant=0.3)

    assert np.all(np.diff(values) < 0)


def test_a_large_constant_switches_the_mechanism_off() -> None:
    """The grid must be able to represent 'this inhibitor does nothing', or the fit is
    obliged to find an effect rather than free to reject one."""
    values = kinetics.inhibition(np.array([0.0, 0.5, 1.02]), inhibition_constant=1e4)

    np.testing.assert_allclose(values, 1.0, atol=1e-3)


def test_inhibition_stays_bounded_beyond_the_training_range() -> None:
    """Test lysate reaches 1.02 against a training maximum of 0.53."""
    values = kinetics.inhibition(np.array([0.53, 1.02, 1e6]), inhibition_constant=0.3)

    assert np.all(values <= 1.0)
    assert np.all(values > 0.0)


def test_inhibition_rejects_a_non_positive_constant() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        kinetics.inhibition(np.array([1.0]), inhibition_constant=-1.0)


# --- combined inhibition -------------------------------------------------------------


def test_combined_inhibition_reduces_to_one_inhibitor_when_the_other_is_absent() -> None:
    lactate = np.array([2.0, 5.0])
    absent = np.zeros(2)

    combined = kinetics.combined_inhibition(lactate, absent, 40.0, 10.0)
    single = kinetics.inhibition(lactate, 40.0)

    np.testing.assert_allclose(combined, single)


def test_combined_inhibition_is_not_the_product_of_two_separate_factors() -> None:
    """The structural claim: lactate and ammonia share a denominator because they act
    through the same route. Two independent factors would multiply, which is a different
    and weaker statement -- and gives a systematically smaller factor."""
    lactate = np.array([5.0])
    ammonia = np.array([8.0])

    shared = kinetics.combined_inhibition(lactate, ammonia, 40.0, 10.0)
    product = kinetics.inhibition(lactate, 40.0) * kinetics.inhibition(ammonia, 10.0)

    assert shared[0] > product[0]
    assert shared[0] != pytest.approx(product[0])


def test_combined_inhibition_is_one_when_both_inhibitors_are_absent() -> None:
    values = kinetics.combined_inhibition(np.zeros(3), np.zeros(3), 40.0, 10.0)

    np.testing.assert_allclose(values, 1.0)


def test_combined_inhibition_rejects_a_non_positive_constant() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        kinetics.combined_inhibition(np.array([1.0]), np.array([1.0]), 40.0, 0.0)


# --- exponential response ------------------------------------------------------------


def test_exponential_response_is_one_at_the_reference() -> None:
    value = kinetics.exponential_response(np.array([36.5]), reference=36.5, sensitivity=1.3)

    assert value == pytest.approx(1.0)


def test_a_positive_sensitivity_favours_values_below_the_reference() -> None:
    values = kinetics.exponential_response(np.array([35.5, 37.5]), reference=36.5, sensitivity=0.4)

    assert values[0] > 1.0 > values[1]


def test_exponential_response_stays_strictly_positive() -> None:
    """A negative productivity factor is not a physical state the optimiser should be able
    to reach, which is why this is an exponential rather than a polynomial."""
    values = kinetics.exponential_response(
        np.linspace(30.0, 45.0, 100), reference=36.5, sensitivity=-1.9
    )

    assert np.all(values > 0.0)


def test_zero_sensitivity_disables_the_response() -> None:
    values = kinetics.exponential_response(
        np.array([35.0, 36.5, 38.0]), reference=36.5, sensitivity=0.0
    )

    np.testing.assert_allclose(values, 1.0)


# --- parameter specifications --------------------------------------------------------


def test_a_logarithmic_grid_spans_its_range_and_is_ordered() -> None:
    spec = kinetics.ParameterSpec("K_G", "mM", 1e-2, 1e3)

    grid = spec.grid(7)

    assert grid[0] == pytest.approx(1e-2)
    assert grid[-1] == pytest.approx(1e3)
    assert np.all(np.diff(grid) > 0)


def test_a_linear_grid_is_evenly_spaced() -> None:
    spec = kinetics.ParameterSpec("theta_T", "per degC", -2.0, 2.0, logarithmic=False)

    grid = spec.grid(5)

    np.testing.assert_allclose(grid, [-2.0, -1.0, 0.0, 1.0, 2.0])


def test_a_grid_needs_at_least_two_points() -> None:
    spec = kinetics.ParameterSpec("K_G", "mM", 1e-2, 1e3)

    with pytest.raises(ValueError, match="at least two grid points"):
        spec.grid(1)


# --- the registry --------------------------------------------------------------------


def test_every_registered_mechanism_declares_units_for_all_its_parameters() -> None:
    """The deliverable is a rate law with constants and units, so a parameter without a
    unit is an incomplete mechanism."""
    for mechanism in kinetics.MECHANISMS.values():
        assert mechanism.parameters, f"{mechanism.name} fits nothing"
        for spec in mechanism.parameters:
            assert spec.unit, f"{mechanism.name}.{spec.name} has no unit"
            assert spec.minimum < spec.maximum


def test_resolve_preserves_the_requested_order() -> None:
    mechanisms = kinetics.resolve(["temperature_response", "glucose_limitation"])

    assert [mechanism.name for mechanism in mechanisms] == [
        "temperature_response",
        "glucose_limitation",
    ]


def test_resolve_rejects_an_unknown_mechanism_and_says_what_is_available() -> None:
    with pytest.raises(KeyError, match="glucose_limitation"):
        kinetics.resolve(["oxygen_limitation"])


def test_resolve_rejects_a_repeated_mechanism() -> None:
    """A factor applied twice would silently square that mechanism's contribution."""
    with pytest.raises(ValueError, match="repeated"):
        kinetics.resolve(["glucose_limitation", "glucose_limitation"])


def test_parameter_specs_concatenate_in_mechanism_order() -> None:
    """This ordering is what the fitting code slices a flat parameter vector by."""
    mechanisms = kinetics.resolve(["metabolic_burden", "temperature_response"])

    assert [spec.name for spec in kinetics.parameter_specs(mechanisms)] == [
        "K_L",
        "K_A",
        "theta_T",
    ]


# --- the environmental factor --------------------------------------------------------


def _state(point_count: int = 4) -> dict[str, np.ndarray]:
    return {
        schema.OBSERVATION_GLUCOSE: np.linspace(20.0, 2.0, point_count),
        schema.OBSERVATION_GLUTAMINE: np.linspace(5.0, 0.5, point_count),
        schema.OBSERVATION_LACTATE: np.linspace(0.0, 6.0, point_count),
        schema.OBSERVATION_AMMONIA: np.linspace(0.0, 9.0, point_count),
        schema.OBSERVATION_LYSED_CELLS: np.linspace(0.0, 0.4, point_count),
        schema.CONTROL_TEMPERATURE: np.full(point_count, 37.0),
        schema.CONTROL_PH: np.full(point_count, 7.0),
    }


def test_no_mechanisms_gives_a_factor_of_exactly_one() -> None:
    """The nesting is a property of this function, so M1 is an exact special case of M2
    and M3 rather than something the model variants have to arrange separately."""
    factor = kinetics.environmental_factor((), _state(), ())

    np.testing.assert_array_equal(factor, np.ones(4))


def test_the_factor_is_the_product_of_its_mechanisms() -> None:
    state = _state()
    mechanisms = kinetics.resolve(["glucose_limitation", "lysate_inhibition"])

    combined = kinetics.environmental_factor(mechanisms, state, [4.0, 0.3])
    separately = kinetics.monod(state[schema.OBSERVATION_GLUCOSE], 4.0) * kinetics.inhibition(
        state[schema.OBSERVATION_LYSED_CELLS], 0.3
    )

    np.testing.assert_allclose(combined, separately)


def test_parameters_are_dealt_out_in_mechanism_order() -> None:
    """Two single-parameter mechanisms with distinct constants: swapping the parameters
    must change the answer, or the ordering contract is not being honoured."""
    state = _state()
    mechanisms = kinetics.resolve(["glucose_limitation", "glutamine_limitation"])

    forward = kinetics.environmental_factor(mechanisms, state, [4.0, 0.2])
    swapped = kinetics.environmental_factor(mechanisms, state, [0.2, 4.0])

    assert not np.allclose(forward, swapped)


def test_the_factor_stays_within_zero_and_one_for_the_saturating_mechanisms() -> None:
    mechanisms = kinetics.resolve(
        ["glucose_limitation", "glutamine_limitation", "metabolic_burden", "lysate_inhibition"]
    )

    factor = kinetics.environmental_factor(mechanisms, _state(), [4.0, 0.2, 40.0, 10.0, 0.3])

    assert np.all(factor > 0.0)
    assert np.all(factor <= 1.0)


def test_a_wrong_parameter_count_is_rejected() -> None:
    mechanisms = kinetics.resolve(["metabolic_burden"])

    with pytest.raises(ValueError, match="takes 2 parameters, got 1"):
        kinetics.environmental_factor(mechanisms, _state(), [40.0])


def test_a_missing_series_names_the_mechanism_that_needed_it() -> None:
    state = _state()
    del state[schema.CONTROL_PH]
    mechanisms = kinetics.resolve(["ph_response"])

    with pytest.raises(KeyError, match="ph_response"):
        kinetics.environmental_factor(mechanisms, state, [1.0])


def test_an_empty_state_is_rejected_rather_than_returning_an_empty_factor() -> None:
    with pytest.raises(ValueError, match="empty state"):
        kinetics.environmental_factor((), {}, ())


# --- state assembly ------------------------------------------------------------------


def test_state_from_run_merges_observations_and_control_profiles() -> None:
    """Mechanisms read both X: and W: series and should not have to know which is which."""
    timestamps = np.arange(3, dtype=np.float64)
    run = ExperimentRun(
        experiment_id="Exp 1",
        timestamps=timestamps,
        design_scalars={},
        control_profiles={schema.CONTROL_TEMPERATURE: np.full(3, 37.0)},
        observations={schema.OBSERVATION_GLUCOSE: np.full(3, 20.0)},
    )

    state = kinetics.state_from_run(run)

    assert set(state) == {schema.CONTROL_TEMPERATURE, schema.OBSERVATION_GLUCOSE}
