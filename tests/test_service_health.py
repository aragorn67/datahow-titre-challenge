"""Tests for the health endpoint and model loading.

The property worth asserting is not "``/health`` returns 200". It is that
**``/health`` tells the truth**: 200 exactly when a prediction can actually be
served, and 503 with a usable reason when it cannot.

A health check that always returns 200 is worse than none, because an
orchestrator believes it and routes traffic to a container that cannot serve.
Most of the tests below are therefore about the *unhealthy* paths, which are the
ones a naive implementation gets wrong.
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from titre_predictor.service import state as state_module
from titre_predictor.service.app import STATUS_OK, STATUS_UNAVAILABLE, create_app

ARTEFACT = Path("artefacts/titre_model.json")


@pytest.fixture
def artefact_path() -> Path:
    """The real fitted artefact, skipping if the pipeline has not been run."""
    path = Path(__file__).resolve().parents[1] / ARTEFACT
    if not path.is_file():
        pytest.skip(f"no artefact at {path}; run scripts/screen_and_fit.py")
    return path


# --- the healthy path ----------------------------------------------------------------------


def test_health_reports_ok_when_a_model_is_loaded(artefact_path: Path) -> None:
    with TestClient(create_app(artefact_path)) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == STATUS_OK


def test_health_reports_which_model_is_serving(artefact_path: Path) -> None:
    """Reporting *which* model is serving, not merely that one is.

    "The service is up" says nothing about whether the model behind it is the one
    that was validated. A prediction has to be traceable to the run that produced
    it, which is what the provenance block is for.
    """
    with TestClient(create_app(artefact_path)) as client:
        payload = client.get("/health").json()

    assert payload["model"]["variant"] == "M2"
    assert payload["model"]["mechanisms"] == ["glutamine_limitation", "glucose_limitation"]
    provenance = payload["provenance"]
    assert provenance["training_data_sha256"], "the data hash is what makes it traceable"
    assert provenance["training_run_count"] == 100
    assert "numpy" in provenance["package_versions"]


# --- the unhealthy paths, which are the ones that matter -----------------------------------


def test_health_reports_503_when_the_artefact_is_missing(tmp_path: Path) -> None:
    """The failure a fresh deployment actually hits: image built, model not mounted."""
    with TestClient(create_app(tmp_path / "absent.json")) as client:
        response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["status"] == STATUS_UNAVAILABLE


def test_the_missing_artefact_message_says_how_to_fix_it(tmp_path: Path) -> None:
    """A reason an operator can act on, not just a failure flag. It must name the
    path tried and the environment variable that overrides it."""
    missing = tmp_path / "absent.json"

    with TestClient(create_app(missing)) as client:
        payload = client.get("/health").json()

    assert str(missing) in payload["detail"]
    assert state_module.ARTEFACT_PATH_VARIABLE in payload["detail"]
    assert payload["artefact_path"] == str(missing)


def test_health_reports_503_when_the_artefact_is_corrupt(tmp_path: Path) -> None:
    """A file that exists but is not JSON. Distinct from missing, and it must not
    take the process down."""
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json at all", encoding="utf-8")

    with TestClient(create_app(corrupt)) as client:
        response = client.get("/health")

    assert response.status_code == 503
    assert "unusable" in response.json()["detail"]


def test_health_reports_503_when_the_artefact_is_from_another_version(tmp_path: Path) -> None:
    """The subtle deployment failure: valid JSON, plausible shape, naming a
    mechanism this build does not have. Serving predictions from it would apply
    coefficients to quantities they were not fitted against."""
    incompatible = tmp_path / "future.json"
    incompatible.write_text(
        json.dumps(
            {
                "variant": "M2",
                "alpha": 16.0,
                "beta": 14.0,
                "lysis_rate_constant": 0.005,
                "mechanisms": ["a_mechanism_from_the_future"],
                "mechanism_parameters": [1.0],
            }
        ),
        encoding="utf-8",
    )

    with TestClient(create_app(incompatible)) as client:
        payload = client.get("/health").json()

    assert payload["status"] == STATUS_UNAVAILABLE
    assert "a_mechanism_from_the_future" in payload["detail"]


def test_a_broken_artefact_does_not_prevent_the_service_starting(tmp_path: Path) -> None:
    """The design decision in state.py, asserted. The process must come up so the
    reason is reachable over HTTP rather than only in container logs."""
    with TestClient(create_app(tmp_path / "absent.json")) as client:
        assert client.get("/health").status_code == 503
        # And the app is genuinely running, not merely returning one canned error.
        assert client.get("/no-such-route").status_code == 404


# --- loading, without the HTTP layer -------------------------------------------------------


def test_the_artefact_path_comes_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Configuration through the environment, so pointing at a different model
    does not require rebuilding the image."""
    monkeypatch.setenv(state_module.ARTEFACT_PATH_VARIABLE, str(tmp_path / "elsewhere.json"))

    assert state_module.configured_artefact_path() == tmp_path / "elsewhere.json"


def test_loading_never_raises(tmp_path: Path) -> None:
    """Startup calls this, and an exception there would take the process down --
    which is the behaviour state.py exists to avoid."""
    result = state_module.load_model_state(tmp_path / "absent.json")

    assert not result.is_ready
    assert result.error


def test_using_an_unloaded_model_fails_with_a_diagnosis_not_an_attribute_error(
    tmp_path: Path,
) -> None:
    """``require_model`` exists so no code path can reach ``None.predict`` and
    report the symptom instead of the cause."""
    result = state_module.load_model_state(tmp_path / "absent.json")

    with pytest.raises(RuntimeError, match=r"absent\.json"):
        result.require_model()
