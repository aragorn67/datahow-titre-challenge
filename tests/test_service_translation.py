"""Tests for the wire format and the payload-to-run translation.

This is the service's trust boundary, so the tests are mostly about **rejection**.
The question each one answers is "what does a caller get told when they send this?"
-- and the standard being held to is that the answer names the problem, because a
400 saying "invalid request" costs the caller as much time as no error at all.

The rule under test throughout: repair only what can be repaired *exactly*, reject
anything that would require a guess.
"""

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from titre_predictor.data import schema
from titre_predictor.domain import InvalidExperimentRunError
from titre_predictor.model import LuedekingPiretModel
from titre_predictor.service.dto import PredictRequest
from titre_predictor.service.translation import (
    PayloadError,
    required_control_profiles,
    required_observations,
    to_experiment_run,
)

ARTEFACT = Path(__file__).resolve().parents[1] / "artefacts" / "titre_model.json"


@pytest.fixture(scope="module")
def shipped_model() -> LuedekingPiretModel:
    if not ARTEFACT.is_file():
        pytest.skip(f"no artefact at {ARTEFACT}; run scripts/screen_and_fit.py")
    return LuedekingPiretModel.load(ARTEFACT)


def _payload(point_count: int = 8, **overrides: object) -> dict[str, object]:
    """A valid request body, before any override is applied."""
    days = list(range(point_count))
    values: dict[str, list[float]] = {
        schema.OBSERVATION_VIABLE_CELL_DENSITY: [1.0 + index for index in days],
        schema.OBSERVATION_LYSED_CELLS: [0.004 * index for index in days],
        schema.OBSERVATION_GLUCOSE: [20.0 - index for index in days],
        schema.OBSERVATION_GLUTAMINE: [5.0 - 0.5 * index for index in days],
        schema.OBSERVATION_LACTATE: [0.5 * index for index in days],
        schema.OBSERVATION_AMMONIA: [0.3 * index for index in days],
        schema.DESIGN_TEMPERATURE_START: [36.5],
        schema.DESIGN_TEMPERATURE_END: [36.0],
        schema.DESIGN_TEMPERATURE_SHIFT: [4.0],
        schema.DESIGN_PH_START: [7.0],
        schema.DESIGN_PH_END: [6.8],
        schema.DESIGN_PH_SHIFT: [5.0],
        schema.DESIGN_FEED_START: [3.0],
        schema.DESIGN_FEED_END: [6.0],
        schema.DESIGN_FEED_RATE_GLUCOSE: [5.0],
        schema.DESIGN_FEED_RATE_GLUTAMINE: [6.0],
    }
    body: dict[str, object] = {"timestamps": [float(d) for d in days], "values": values}
    body.update(overrides)
    return body


# --- the wire format -----------------------------------------------------------------------


def test_a_valid_payload_parses() -> None:
    request = PredictRequest(**_payload())  # type: ignore[arg-type]

    assert len(request.timestamps) == 8
    assert schema.OBSERVATION_VIABLE_CELL_DENSITY in request.values


def test_a_key_without_a_recognised_prefix_is_rejected_by_name() -> None:
    """An unprefixed key is not a variable the service can place. Accepting it would
    mean silently ignoring input the caller believed was being used."""
    body = _payload()
    body["values"]["VCD"] = [1.0] * 8  # type: ignore[index]

    with pytest.raises(ValidationError, match="VCD"):
        PredictRequest(**body)  # type: ignore[arg-type]


def test_a_design_scalar_with_several_values_is_rejected() -> None:
    """``Z:`` variables are constant within a run by definition. A list of eight
    means the caller has misunderstood the convention, and guessing which value
    they meant would be exactly the wrong repair."""
    body = _payload()
    body["values"][schema.DESIGN_STIRRING] = [200.0] * 8  # type: ignore[index]

    with pytest.raises(ValidationError, match="Z:Stir"):
        PredictRequest(**body)  # type: ignore[arg-type]


def test_an_unknown_top_level_field_is_rejected_rather_than_ignored() -> None:
    """The near-miss key. Silently dropping ``timestamp`` would surface later as a
    baffling "field required" for ``timestamps``."""
    body = _payload()
    body["timestamp"] = [0, 1, 2]

    with pytest.raises(ValidationError):
        PredictRequest(**body)  # type: ignore[arg-type]


def test_the_optional_experiment_id_is_carried_through() -> None:
    """Not used to predict; it exists so a caller batching requests can tell which
    one failed."""
    request = PredictRequest(**_payload(experiment_id="Test Exp 7"))  # type: ignore[arg-type]

    assert request.experiment_id == "Test Exp 7"


# --- what the model requires ---------------------------------------------------------------


def test_the_required_series_are_derived_from_the_model(
    shipped_model: LuedekingPiretModel,
) -> None:
    """Not hardcoded. VCD and Lysed are structural; glucose and glutamine are
    required only because *this* model's mechanisms read them."""
    observations = required_observations(shipped_model)

    assert schema.OBSERVATION_VIABLE_CELL_DENSITY in observations
    assert schema.OBSERVATION_LYSED_CELLS in observations
    assert schema.OBSERVATION_GLUCOSE in observations
    assert schema.OBSERVATION_GLUTAMINE in observations


def test_the_shipped_model_needs_no_control_profiles(
    shipped_model: LuedekingPiretModel,
) -> None:
    """Its factor is Monod in two metabolites, so temperature and pH are not read.
    A model including a temperature term would require ``W:temp`` with no change to
    the service."""
    assert required_control_profiles(shipped_model) == ()


def test_a_model_with_a_temperature_mechanism_would_require_the_temperature_profile() -> None:
    """The other half of the previous test: the requirement genuinely tracks the
    mechanism set rather than always being empty."""
    temperature_model = LuedekingPiretModel(
        variant="M2",
        alpha=16.0,
        beta=14.0,
        lysis_rate_constant=0.005,
        mechanisms=("temperature_response",),
        mechanism_parameters=(0.1,),
    )

    assert required_control_profiles(temperature_model) == (schema.CONTROL_TEMPERATURE,)


def test_a_missing_required_observation_is_named_along_with_what_the_model_reads(
    shipped_model: LuedekingPiretModel,
) -> None:
    body = _payload()
    del body["values"][schema.OBSERVATION_GLUCOSE]  # type: ignore[attr-defined]
    request = PredictRequest(**body)  # type: ignore[arg-type]

    with pytest.raises(PayloadError, match=schema.OBSERVATION_GLUCOSE):
        to_experiment_run(request, shipped_model)


def test_an_unneeded_observation_may_be_absent(shipped_model: LuedekingPiretModel) -> None:
    """The service asks for what the model reads, not for everything the training
    CSVs happened to contain. Ammonia and lactate are not in this model's factor."""
    body = _payload()
    del body["values"][schema.OBSERVATION_AMMONIA]  # type: ignore[attr-defined]
    del body["values"][schema.OBSERVATION_LACTATE]  # type: ignore[attr-defined]

    run = to_experiment_run(PredictRequest(**body), shipped_model)  # type: ignore[arg-type]

    assert run.timestamps.size == 8


# --- repair, only where it is exact --------------------------------------------------------


def test_control_profiles_are_reconstructed_from_design_scalars_when_needed() -> None:
    """The one legitimate repair: ``W:`` profiles are exact step functions of the
    ``Z:`` scalars, so a caller who sent the scalars has already determined them."""
    temperature_model = LuedekingPiretModel(
        variant="M2",
        alpha=16.0,
        beta=14.0,
        lysis_rate_constant=0.005,
        mechanisms=("temperature_response",),
        mechanism_parameters=(0.1,),
    )
    body = _payload()  # carries no W: keys at all

    run = to_experiment_run(PredictRequest(**body), temperature_model)  # type: ignore[arg-type]

    temperature = run.control_profiles[schema.CONTROL_TEMPERATURE]
    # tempStart 36.5 before the shift at day 4, tempEnd 36.0 from day 4 onwards.
    assert temperature[0] == pytest.approx(36.5)
    assert temperature[3] == pytest.approx(36.5)
    assert temperature[4] == pytest.approx(36.0)
    assert temperature[-1] == pytest.approx(36.0)


def test_a_supplied_profile_is_preferred_over_a_reconstructed_one() -> None:
    """Reconstruction fills gaps; it does not overwrite what the caller sent."""
    temperature_model = LuedekingPiretModel(
        variant="M2",
        alpha=16.0,
        beta=14.0,
        lysis_rate_constant=0.005,
        mechanisms=("temperature_response",),
        mechanism_parameters=(0.1,),
    )
    body = _payload()
    body["values"][schema.CONTROL_TEMPERATURE] = [30.0] * 8  # type: ignore[index]

    run = to_experiment_run(PredictRequest(**body), temperature_model)  # type: ignore[arg-type]

    assert run.control_profiles[schema.CONTROL_TEMPERATURE] == pytest.approx(np.full(8, 30.0))


def test_reconstruction_that_cannot_be_done_exactly_is_refused_with_the_reason() -> None:
    """If the design scalars needed to rebuild a profile are themselves absent,
    there is nothing to reconstruct *from*. Inventing a temperature would let the
    service return a confident number computed from made-up data."""
    temperature_model = LuedekingPiretModel(
        variant="M2",
        alpha=16.0,
        beta=14.0,
        lysis_rate_constant=0.005,
        mechanisms=("temperature_response",),
        mechanism_parameters=(0.1,),
    )
    body = _payload()
    del body["values"][schema.DESIGN_TEMPERATURE_SHIFT]  # type: ignore[attr-defined]

    with pytest.raises(PayloadError, match="tempShift"):
        to_experiment_run(PredictRequest(**body), temperature_model)  # type: ignore[arg-type]


# --- invariants enforced by the domain, not duplicated here --------------------------------


def test_a_series_of_the_wrong_length_is_rejected_by_the_domain(
    shipped_model: LuedekingPiretModel,
) -> None:
    """Deliberately *not* re-checked in the DTO. ``ExperimentRun`` already enforces
    it, and the training pipeline goes through the same check -- one source of truth
    rather than two that can drift."""
    body = _payload()
    body["values"][schema.OBSERVATION_GLUCOSE] = [20.0] * 5  # type: ignore[index]

    with pytest.raises(InvalidExperimentRunError, match="X:Glc"):
        to_experiment_run(PredictRequest(**body), shipped_model)  # type: ignore[arg-type]


def test_a_nan_in_a_series_is_rejected_with_its_position(
    shipped_model: LuedekingPiretModel,
) -> None:
    """A single NaN would propagate through the quadratures and return a titre of
    nan -- a response that looks like an answer."""
    body = _payload()
    body["values"][schema.OBSERVATION_VIABLE_CELL_DENSITY][3] = float("nan")  # type: ignore[index]

    with pytest.raises(InvalidExperimentRunError, match="X:VCD"):
        to_experiment_run(PredictRequest(**body), shipped_model)  # type: ignore[arg-type]


def test_timestamps_that_do_not_increase_are_rejected(
    shipped_model: LuedekingPiretModel,
) -> None:
    body = _payload()
    body["timestamps"] = [0.0, 1.0, 1.0, 3.0, 4.0, 5.0, 6.0, 7.0]

    with pytest.raises(InvalidExperimentRunError, match="strictly increasing"):
        to_experiment_run(PredictRequest(**body), shipped_model)  # type: ignore[arg-type]


def test_a_single_timepoint_is_rejected(shipped_model: LuedekingPiretModel) -> None:
    """There is nothing to integrate over."""
    body = _payload(point_count=1)

    with pytest.raises(InvalidExperimentRunError, match="at least two"):
        to_experiment_run(PredictRequest(**body), shipped_model)  # type: ignore[arg-type]


# --- the run is usable ---------------------------------------------------------------------


def test_the_translated_run_can_actually_be_predicted(
    shipped_model: LuedekingPiretModel,
) -> None:
    """The point of the whole layer: what comes out the bottom is something the
    model can consume, not merely something that validated."""
    run = to_experiment_run(PredictRequest(**_payload()), shipped_model)  # type: ignore[arg-type]

    prediction = shipped_model.predict(run)

    assert np.isfinite(prediction)
    assert prediction > 0.0


# --- against the specification's own example -----------------------------------------------


def _spec_example() -> dict[str, object]:
    """The example payload from ``inference_server_spec.yml``, read from the file.

    Transcribing it into the test would let the two drift apart silently, which
    defeats the purpose: the value of this test is that it exercises what the
    *specification* says a request looks like, not what I remember it saying.
    """
    import yaml

    spec_path = Path(__file__).resolve().parents[1] / "inference_server_spec.yml"
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    properties = spec["components"]["schemas"]["PredictRequest"]["properties"]
    return {
        "timestamps": properties["timestamps"]["example"],
        "values": properties["values"]["example"],
    }


def test_the_example_from_the_specification_is_accepted(
    shipped_model: LuedekingPiretModel,
) -> None:
    """The contract test. If this fails, the service does not implement the API
    that was asked for, whatever else works."""
    request = PredictRequest(**_spec_example())  # type: ignore[arg-type]

    run = to_experiment_run(request, shipped_model)

    assert run.timestamps.size == 15
    assert run.duration_days == pytest.approx(14.0)


def test_the_specification_example_predicts_a_plausible_titre(
    shipped_model: LuedekingPiretModel,
) -> None:
    """End to end on the real contract. The example is a 14-day run -- the
    extrapolation regime the whole model was designed around -- so a finite,
    positive, non-absurd number here exercises the case that matters.
    """
    run = to_experiment_run(PredictRequest(**_spec_example()), shipped_model)  # type: ignore[arg-type]

    prediction = shipped_model.predict(run)

    assert np.isfinite(prediction)
    # Training titres span 283-4823; a 14-day run should land in the upper part of
    # that range rather than anywhere at all.
    assert 200.0 < prediction < 10000.0


def test_glutamine_reaching_exactly_zero_does_not_break_the_prediction(
    shipped_model: LuedekingPiretModel,
) -> None:
    """Worth its own test because the shipped model is most sensitive precisely
    there. ``K_Q`` is 0.0219 mM, so the glutamine Monod factor goes to zero when
    glutamine does -- and the specification's own example hits exactly 0.0 three
    times. A division or a log in that factor would surface here."""
    example = _spec_example()
    glutamine = example["values"][schema.OBSERVATION_GLUTAMINE]  # type: ignore[index]
    assert 0.0 in glutamine, "premise: the spec example depletes glutamine to exactly zero"

    run = to_experiment_run(PredictRequest(**example), shipped_model)  # type: ignore[arg-type]

    assert np.isfinite(shipped_model.predict(run))
