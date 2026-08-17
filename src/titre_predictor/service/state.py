"""Loading the fitted model once, and reporting whether the service can serve.

Why the model is loaded at startup, not per request
---------------------------------------------------
Reading and parsing the artefact is pure overhead repeated on every call, and it
introduces a failure mode that only appears under traffic: a malformed or missing
artefact would surface as a 500 on a user's request rather than as a service that
never became ready. Loading once at startup turns a runtime error into a
deployment error, which is the cheaper place to have it.

Why a failed load does not crash the process
--------------------------------------------
The tempting alternative is to let the exception propagate and have the container
exit. A crash loop is certainly loud. But it puts the *reason* only in the logs,
and the reason is what an operator needs: a missing file, an artefact from an
incompatible version, and a corrupt JSON are three different problems with three
different fixes.

So the failure is caught and recorded, the service starts, and ``/health``
reports 503 with the reason. An orchestrator's readiness probe keeps the
container out of the load balancer either way -- so nothing degraded is served --
while ``curl /health`` answers "why" without digging through logs.

The distinction being drawn is the standard one: **liveness** asks "is the process
alive?", **readiness** asks "can it serve traffic?". The specification defines a
single ``/health`` endpoint, so it is implemented as a readiness check, which is
the stricter and more useful of the two.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from titre_predictor.model import LuedekingPiretModel

# Where the artefact lives, overridable so the container can mount it elsewhere.
# Configuration through the environment rather than a config file: there is one
# setting, and an image should not need rebuilding to point at a different model.
ARTEFACT_PATH_VARIABLE = "TITRE_MODEL_PATH"
DEFAULT_ARTEFACT_PATH = Path("artefacts/titre_model.json")


def configured_artefact_path() -> Path:
    """The artefact path, from the environment or the default."""
    return Path(os.environ.get(ARTEFACT_PATH_VARIABLE, str(DEFAULT_ARTEFACT_PATH)))


@dataclass
class ModelState:
    """The loaded model, or the reason there isn't one.

    Exactly one of :attr:`model` and :attr:`error` is set. Callers check
    :attr:`is_ready` rather than testing either directly, so the invariant lives
    in one place.

    Args:
        model: the fitted rate law, once loaded.
        error: why loading failed, phrased for whoever has to fix it.
        artefact_path: where loading was attempted, so a wrong path is visible in
            the health response rather than guessed at.
        provenance: the artefact's record of how it was produced. Surfaced by
            ``/health`` so it is possible to tell *which* model is serving --
            without it, "the service is up" says nothing about what it is serving.
        training_ranges: the span of each quantity over the training runs, used to
            tell a caller whether their request asks about conditions the model was
            fitted on. ``None`` for an artefact produced before these were recorded,
            which is still a usable model -- so the service serves it and reports
            that the check could not be made, rather than refusing or implying a
            clean result.
    """

    model: LuedekingPiretModel | None = None
    error: str | None = None
    artefact_path: Path | None = None
    provenance: dict[str, Any] | None = None
    training_ranges: dict[str, list[float]] | None = None

    @property
    def is_ready(self) -> bool:
        """Whether a prediction can be served."""
        return self.model is not None

    def require_model(self) -> LuedekingPiretModel:
        """The loaded model, or a clear failure.

        Raises:
            RuntimeError: if no model is loaded. Endpoints translate this into a
                503; it exists so no code path can use ``model`` while it is
                ``None`` and get an ``AttributeError`` instead of a diagnosis.
        """
        if self.model is None:
            raise RuntimeError(self.error or "no model has been loaded")
        return self.model


def load_model_state(artefact_path: Path | None = None) -> ModelState:
    """Read the artefact and build the service's state.

    Never raises. Every failure becomes a populated :attr:`ModelState.error`,
    because the caller is application startup and the useful behaviour there is to
    come up and report the problem rather than to disappear.

    Args:
        artefact_path: where to read from. Defaults to
            :func:`configured_artefact_path`.
    """
    path = artefact_path if artefact_path is not None else configured_artefact_path()

    if not path.is_file():
        return ModelState(
            error=(
                f"no model artefact at {path}. Set {ARTEFACT_PATH_VARIABLE}, or run "
                f"`python scripts/screen_and_fit.py` to produce one."
            ),
            artefact_path=path,
        )

    try:
        model = LuedekingPiretModel.load(path)
    except (KeyError, ValueError) as exception:
        # KeyError: the artefact names a variant or mechanism this build does not
        # have. ValueError: its parameter count disagrees with its mechanisms, or
        # the file is not valid JSON. All mean the artefact and the code disagree,
        # which is a deployment mistake and is worth saying so plainly.
        return ModelState(
            error=(
                f"artefact at {path} is unusable by this build of titre_predictor "
                f"({type(exception).__name__}: {exception})"
            ),
            artefact_path=path,
        )
    except (OSError, UnicodeDecodeError) as exception:
        # Readable path, unreadable file: permissions, a directory, bad encoding.
        return ModelState(
            error=f"could not read artefact at {path}: {exception}",
            artefact_path=path,
        )

    side_car = _read_side_car_blocks(path)
    return ModelState(
        model=model,
        artefact_path=path,
        provenance=side_car.get("provenance"),
        training_ranges=side_car.get("training_ranges"),
    )


def _read_side_car_blocks(path: Path) -> dict[str, Any]:
    """The artefact's ``provenance`` and ``training_ranges`` blocks, if present.

    Read separately rather than through :meth:`LuedekingPiretModel.from_dict`, which
    deliberately ignores both: neither is needed to *predict*, so the model object
    has no business holding them. Both are needed to *interpret* a prediction --
    which model produced it, and whether it was asked about familiar conditions --
    and that is this module's concern.

    Missing blocks are not an error. An artefact from before either was recorded is
    a perfectly good model, and the service degrades to serving without that
    context rather than refusing.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # pragma: no cover - the model already loaded
        return {}

    blocks: dict[str, Any] = {}
    provenance = payload.get("provenance")
    if isinstance(provenance, dict):
        blocks["provenance"] = dict(provenance)

    ranges = payload.get("training_ranges")
    if isinstance(ranges, dict):
        # Kept only where the span is a usable pair; a malformed entry is dropped
        # rather than allowed to raise later, inside a request.
        blocks["training_ranges"] = {
            name: [float(span[0]), float(span[1])]
            for name, span in ranges.items()
            if isinstance(span, (list, tuple)) and len(span) == 2
        }
    return blocks
