"""The FastAPI application: ``GET /health``, and later ``POST /predict``.

Structure
---------
The application is built by :func:`create_app` rather than existing as a module
level global. That is what makes it testable: a test can build an app around a
deliberately broken artefact path and assert the service reports itself unhealthy,
which is impossible if the app loads its model at import time.

``uvicorn`` still needs something to point at, so :data:`app` is created at the
bottom for deployment. Tests do not use it.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Response, status

from titre_predictor.domain import InvalidExperimentRunError
from titre_predictor.service import extrapolation
from titre_predictor.service.dto import (
    ExtrapolationReport,
    ModelDescription,
    PredictRequest,
    PredictResponse,
)
from titre_predictor.service.state import ModelState, load_model_state
from titre_predictor.service.translation import PayloadError, to_experiment_run

SERVICE_TITLE = "Titre Prediction API"
SERVICE_VERSION = "1.0.0"

# Returned by /health. "ok" matches the example in inference_server_spec.yml;
# the unhealthy value is ours, since the specification does not define one.
STATUS_OK = "ok"
STATUS_UNAVAILABLE = "unavailable"


def create_app(artefact_path: Path | None = None) -> FastAPI:
    """Build the application.

    Args:
        artefact_path: where to load the fitted model from. Defaults to the
            configured path; tests pass an explicit one, including paths that do
            not exist, to exercise the unhealthy branch.
    """

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        """Load the model once, before the first request is accepted.

        Failures are recorded rather than raised -- see ``state.py`` on why the
        process comes up and reports the reason instead of crash-looping.
        """
        application.state.model_state = load_model_state(artefact_path)
        yield

    application = FastAPI(
        title=SERVICE_TITLE,
        version=SERVICE_VERSION,
        description=(
            "Predicts the final monoclonal antibody titre of a fed-batch bioprocess "
            "from its observed trajectories."
        ),
        lifespan=lifespan,
    )

    @application.get(
        "/health",
        summary="Health check",
        description=(
            "Readiness, not liveness: 200 only when a model is loaded and the service "
            "can actually predict. 503 with the reason otherwise."
        ),
    )
    def get_health(response: Response) -> dict[str, Any]:
        """Whether the service can serve a prediction, and what it is serving.

        Reporting *which* model is loaded matters as much as whether one is. A
        service that answers "ok" tells an operator nothing about whether the
        model behind it is the one that was validated; the provenance block --
        a hash of the training data, the seed, package versions -- lets a served
        prediction be traced to the run that produced it.
        """
        state: ModelState = application.state.model_state

        if not state.is_ready:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return {
                "status": STATUS_UNAVAILABLE,
                "detail": state.error,
                "artefact_path": str(state.artefact_path) if state.artefact_path else None,
            }

        model = state.require_model()
        return {
            "status": STATUS_OK,
            "model": {
                "variant": model.variant,
                "mechanisms": list(model.mechanisms),
            },
            "provenance": state.provenance,
        }

    @application.post(
        "/predict",
        response_model=PredictResponse,
        summary="Run inference",
        description=(
            "Predicts the final titre from an experiment's observed trajectories. "
            "Returns 400 with the offending variable named if the payload cannot be "
            "used, and 503 if no model is loaded."
        ),
        responses={
            400: {"description": "The payload cannot be turned into a run"},
            503: {"description": "No model is loaded"},
        },
    )
    def post_predict(request: PredictRequest) -> PredictResponse:
        """Predict one experiment's final titre.

        The whole body of this function is translate, predict, describe. That it is
        short is the point: everything that could go wrong is handled by a layer
        that is tested on its own, and there is no arithmetic here that could
        disagree with the training pipeline's.
        """
        state: ModelState = application.state.model_state
        if not state.is_ready:
            # 503 rather than 500: the request is fine, the service is not, and the
            # caller should retry elsewhere or later rather than change the payload.
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=state.error or "no model is loaded",
            )
        model = state.require_model()

        try:
            run = to_experiment_run(request, model)
        except (PayloadError, InvalidExperimentRunError) as exception:
            # 400, not 422: the payload parsed as JSON and matched the schema, so
            # this is a semantic problem with the run rather than a malformed body.
            # The message names the variable at fault, which is what the caller
            # needs in order to fix it.
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exception)
            ) from exception

        exceedances = extrapolation.assess(run, model, state.training_ranges)
        provenance = state.provenance or {}

        return PredictResponse(
            predicted_titer=model.predict(run),
            experiment_id=request.experiment_id,
            model=ModelDescription(
                variant=model.variant,
                mechanisms=list(model.mechanisms),
                training_data_sha256=provenance.get("training_data_sha256"),
            ),
            extrapolation=ExtrapolationReport(
                checked=state.training_ranges is not None,
                beyond_training_range=[item.quantity for item in exceedances],
                detail=[item.describe() for item in exceedances],
            ),
        )

    return application


app = create_app()
"""The deployed application, for ``uvicorn titre_predictor.service.app:app``."""
