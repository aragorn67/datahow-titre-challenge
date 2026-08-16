"""Tests for ``POST /predict``.

The test that matters most is :func:`test_the_api_returns_exactly_what_the_pipeline_computes`.
Everything else here is ordinary endpoint behaviour; that one asserts the property
the whole architecture was arranged to give:

    **a number served over HTTP is the same number the training pipeline computes.**

It holds because ``features.py`` is imported by both, rather than the service
reimplementing the quadratures against a copied set of constants. That is the
failure mode this design exists to prevent -- a served model quietly drifting from
the model that was validated -- and it is worth an explicit assertion because
nothing else would notice.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

from titre_predictor.data import schema
from titre_predictor.data.loading import load_runs
from titre_predictor.model import LuedekingPiretModel
from titre_predictor.service.app import create_app

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARTEFACT = REPOSITORY_ROOT / "artefacts" / "titre_model.json"
SPEC = REPOSITORY_ROOT / "inference_server_spec.yml"
TEST_DATA = REPOSITORY_ROOT / "data" / "raw" / "datahow_interview_test_data.csv"
TRAIN_DATA = REPOSITORY_ROOT / "data" / "raw" / "datahow_interview_train_data.csv"


@pytest.fixture(scope="module")
def artefact_path() -> Path:
    if not ARTEFACT.is_file():
        pytest.skip(f"no artefact at {ARTEFACT}; run scripts/screen_and_fit.py")
    return ARTEFACT


@pytest.fixture
def client(artefact_path: Path) -> Any:
    with TestClient(create_app(artefact_path)) as test_client:
        yield test_client


def _spec_example() -> dict[str, Any]:
    """The example request from the specification, read rather than transcribed."""
    spec = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    properties = spec["components"]["schemas"]["PredictRequest"]["properties"]
    return {
        "timestamps": properties["timestamps"]["example"],
        "values": properties["values"]["example"],
    }


def _payload_from_run(run: Any) -> dict[str, Any]:
    """Turn a loaded ``ExperimentRun`` back into a request body."""
    values: dict[str, list[float]] = {}
    for name, value in run.design_scalars.items():
        values[name] = [float(value)]
    for name, series in run.control_profiles.items():
        values[name] = [float(x) for x in series]
    for name, series in run.observations.items():
        values[name] = [float(x) for x in series]
    return {
        "timestamps": [float(t) for t in run.timestamps],
        "values": values,
        "experiment_id": run.experiment_id,
    }


# --- the property the architecture exists to guarantee --------------------------------------


def test_the_api_returns_exactly_what_the_pipeline_computes(
    artefact_path: Path, client: Any
) -> None:
    """The JSON round-trip does not change the number.

    Precisely what this asserts: a run loaded from CSV, serialised into a request
    body, sent over HTTP, and reconstructed by the translation layer produces the
    *same* prediction as calling the model on the original run directly. That is
    the link this test owns -- the wire format and the translation are lossless.

    The other link in the chain, that the artefact's coefficients are the ones the
    pipeline fitted, is asserted by the pipeline itself: it reloads what it wrote
    and checks the predictions match before declaring success. Together the two
    close the path from fitted model to served number, and neither alone does.

    Run over every supplied test experiment rather than one, since a discrepancy
    could easily appear only for a particular trajectory shape -- a series that
    happens to hit zero, or a duration no other run has.
    """
    model = LuedekingPiretModel.load(artefact_path)
    runs = load_runs(TEST_DATA)

    for run in runs:
        directly = model.predict(run)

        response = client.post("/predict", json=_payload_from_run(run))

        assert response.status_code == 200, response.text
        over_http = response.json()["predicted_titer"]
        assert over_http == pytest.approx(directly, rel=1e-12, abs=1e-9), (
            f"{run.experiment_id}: API returned {over_http} but the pipeline computes "
            f"{directly}. The service has drifted from the fitted model."
        )


# --- the contract --------------------------------------------------------------------------


def test_the_specification_example_is_predicted(client: Any) -> None:
    """The contract test: the API accepts what the spec says a request looks like."""
    response = client.post("/predict", json=_spec_example())

    assert response.status_code == 200, response.text
    body = response.json()
    assert np.isfinite(body["predicted_titer"])
    assert body["predicted_titer"] > 0.0


def test_the_response_names_the_model_that_produced_the_number(client: Any) -> None:
    """A prediction is not interpretable without knowing what produced it, and the
    training-data hash is what ties it to a specific pipeline run."""
    body = client.post("/predict", json=_spec_example()).json()

    assert body["model"]["variant"] == "M2"
    assert body["model"]["mechanisms"] == ["glutamine_limitation", "glucose_limitation"]
    assert len(body["model"]["training_data_sha256"]) == 64


def test_the_experiment_id_is_echoed_back(client: Any) -> None:
    """So a caller batching requests can match responses to requests."""
    payload = {**_spec_example(), "experiment_id": "Test Exp 3"}

    body = client.post("/predict", json=payload).json()

    assert body["experiment_id"] == "Test Exp 3"


# --- extrapolation reporting ----------------------------------------------------------------


def test_a_run_beyond_the_training_range_is_flagged(client: Any, artefact_path: Path) -> None:
    """The report that makes this service honest about its own limits.

    Built by scaling a real run's cell density far past anything in training, so
    ``cell_days`` must land outside the recorded range.
    """
    if not _artefact_has_ranges(artefact_path):
        pytest.skip("artefact predates applicability ranges; re-run the pipeline")

    payload = _spec_example()
    payload["values"][schema.OBSERVATION_VIABLE_CELL_DENSITY] = [
        value * 20.0 for value in payload["values"][schema.OBSERVATION_VIABLE_CELL_DENSITY]
    ]

    body = client.post("/predict", json=payload).json()

    assert body["extrapolation"]["checked"] is True
    assert "cell_days" in body["extrapolation"]["beyond_training_range"]
    assert any("cell_days" in line for line in body["extrapolation"]["detail"])


def test_a_run_inside_the_training_range_is_not_flagged(client: Any, artefact_path: Path) -> None:
    """The complementary half: the warning must be capable of staying silent, or it
    carries no information when it fires."""
    if not _artefact_has_ranges(artefact_path):
        pytest.skip("artefact predates applicability ranges; re-run the pipeline")
    training_run = load_runs(TRAIN_DATA)[0]

    body = client.post("/predict", json=_payload_from_run(training_run)).json()

    assert body["extrapolation"]["checked"] is True
    assert body["extrapolation"]["beyond_training_range"] == []


def test_extrapolation_is_reported_as_unchecked_when_the_artefact_has_no_ranges(
    tmp_path: Path,
) -> None:
    """An older artefact is still a good model, so it is served -- but the response
    must say the check could not be made rather than implying a clean result.
    An empty list with ``checked: false`` means "not checked", not "all clear"."""
    import json

    payload = json.loads(ARTEFACT.read_text(encoding="utf-8"))
    payload.pop("training_ranges", None)
    stripped = tmp_path / "no_ranges.json"
    stripped.write_text(json.dumps(payload), encoding="utf-8")

    with TestClient(create_app(stripped)) as client:
        body = client.post("/predict", json=_spec_example()).json()

    assert body["extrapolation"]["checked"] is False
    assert body["extrapolation"]["beyond_training_range"] == []
    assert np.isfinite(body["predicted_titer"]), "it still serves the prediction"


def _artefact_has_ranges(path: Path) -> bool:
    import json

    return "training_ranges" in json.loads(path.read_text(encoding="utf-8"))


# --- rejection, with the reason -------------------------------------------------------------


def test_a_missing_required_observation_is_a_400_naming_it(client: Any) -> None:
    payload = _spec_example()
    del payload["values"][schema.OBSERVATION_GLUCOSE]

    response = client.post("/predict", json=payload)

    assert response.status_code == 400
    assert schema.OBSERVATION_GLUCOSE in response.json()["detail"]


def test_a_series_of_the_wrong_length_is_a_400_naming_it(client: Any) -> None:
    payload = _spec_example()
    payload["values"][schema.OBSERVATION_GLUCOSE] = [20.0, 19.0, 18.0]

    response = client.post("/predict", json=payload)

    assert response.status_code == 400
    assert schema.OBSERVATION_GLUCOSE in response.json()["detail"]


def test_a_nan_is_a_400_rather_than_a_titre_of_nan(client: Any) -> None:
    """The failure worth being strict about. A NaN would propagate through the
    quadratures and return successfully with ``"predicted_titer": NaN`` -- a
    response that looks like an answer."""
    payload = _spec_example()
    series = list(payload["values"][schema.OBSERVATION_VIABLE_CELL_DENSITY])
    series[4] = None  # JSON null -> not a float
    payload["values"][schema.OBSERVATION_VIABLE_CELL_DENSITY] = series

    response = client.post("/predict", json=payload)

    assert response.status_code in (400, 422)


def test_a_malformed_body_is_rejected_by_the_schema(client: Any) -> None:
    """422 from FastAPI's own validation: the body does not match the schema at all,
    which is a different failure from a well-formed but unusable run."""
    response = client.post("/predict", json={"timestamps": "not a list"})

    assert response.status_code == 422


def test_an_unprefixed_variable_is_rejected(client: Any) -> None:
    payload = _spec_example()
    payload["values"]["VCD"] = payload["values"][schema.OBSERVATION_VIABLE_CELL_DENSITY]

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert "VCD" in response.text


# --- unavailable ----------------------------------------------------------------------------


def test_predicting_without_a_loaded_model_is_503_not_500(tmp_path: Path) -> None:
    """The request is fine; the service is not. A 500 would tell the caller to change
    their payload, which would not help."""
    with TestClient(create_app(tmp_path / "absent.json")) as client:
        response = client.post("/predict", json=_spec_example())

    assert response.status_code == 503
    assert "absent.json" in response.json()["detail"]
