"""Wire formats for the API: what a request looks like, and what comes back.

Why these are separate types from ``ExperimentRun``
---------------------------------------------------
``ExperimentRun`` is the domain object: numpy arrays, validated invariants, no
knowledge of HTTP or of the ``Z:``/``W:``/``X:`` naming convention beyond the
prefix constants. These are *transport* objects: JSON-shaped, string-keyed, and
tolerant of being wrong because they are what arrives from outside.

Keeping them apart is what stops the naming convention leaking into the model, and
what lets the request schema change without touching anything that computes. The
translation between them is one function, in ``translation.py``, so there is a
single place where an untrusted payload becomes a trusted object.

What is validated here, and what is not
----------------------------------------
Only what is specific to the **wire format**:

* the top-level shape,
* the prefix convention -- every key must be ``Z:``, ``W:`` or ``X:``,
* ``Z:`` entries carrying exactly one value, since a design scalar is constant.

Everything about the *run itself* -- at least two timestamps, strictly increasing,
series matching the timestamp count, no NaNs -- is left to ``ExperimentRun``,
which already enforces all of it and is used identically by the training pipeline.
Re-checking here would create a second source of truth that could drift from the
first, and the drift would be silent.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from titre_predictor.data import schema

VALID_PREFIXES = (
    schema.DESIGN_SCALAR_PREFIX,
    schema.CONTROL_PROFILE_PREFIX,
    schema.OBSERVATION_PREFIX,
)


class PredictRequest(BaseModel):
    """One experiment's trajectories, as ``inference_server_spec.yml`` defines them.

    Args:
        timestamps: sample times in days.
        values: variable name to values. ``Z:`` keys carry a single-element list;
            ``W:`` and ``X:`` keys carry one value per timestamp.
        experiment_id: optional caller-supplied identifier. Not used to predict --
            it is echoed back and included in error messages, so a caller batching
            requests can tell which one failed.
    """

    # Unknown top-level fields are rejected rather than ignored. The common failure
    # this catches is a near-miss key -- "timestamp" for "timestamps" -- which would
    # otherwise be silently dropped and surface as a confusing "field required".
    model_config = ConfigDict(extra="forbid")

    timestamps: list[float]
    values: dict[str, list[float]]
    experiment_id: str | None = Field(
        default=None,
        description="Optional caller-supplied identifier, echoed back in the response.",
    )

    @field_validator("values")
    @classmethod
    def _keys_use_the_prefix_convention(
        cls, values: dict[str, list[float]]
    ) -> dict[str, list[float]]:
        """Every key must be prefixed, and design scalars must be scalar.

        A key with no prefix is not a variable this service knows how to place --
        it is neither a design value, a control nor an observation -- so accepting
        it would mean silently ignoring input the caller believed was used.
        """
        unprefixed = sorted(name for name in values if not name.startswith(VALID_PREFIXES))
        if unprefixed:
            raise ValueError(
                f"keys must start with one of {list(VALID_PREFIXES)}; got {unprefixed}"
            )

        wrong_length = sorted(
            name
            for name, series in values.items()
            if name.startswith(schema.DESIGN_SCALAR_PREFIX) and len(series) != 1
        )
        if wrong_length:
            raise ValueError(
                f"design scalars are constant within a run, so each must carry exactly "
                f"one value; got several for {wrong_length}"
            )
        return values


class ModelDescription(BaseModel):
    """Which fitted model produced a prediction.

    Returned with every prediction for the same reason ``/health`` reports it: a
    number is not interpretable without knowing what produced it, and the
    training-data hash is what ties it to a specific pipeline run.
    """

    variant: str
    mechanisms: list[str]
    training_data_sha256: str | None = None


class ExtrapolationReport(BaseModel):
    """Which inputs lie outside the range the model was fitted on.

    Present on every successful prediction, and **not** omitted when everything is
    in range: a caller should be able to read one field to know the answer rather
    than infer it from a key's absence.

    Args:
        beyond_training_range: quantities outside the fitted range, by name.
        detail: one human-readable line per exceedance, with the value and the
            range it left.
        checked: whether a comparison was possible at all. False when the artefact
            carries no recorded ranges, in which case an empty
            ``beyond_training_range`` means "not checked" rather than "all clear" --
            a distinction worth stating rather than leaving to be assumed.
    """

    checked: bool
    beyond_training_range: list[str] = Field(default_factory=list)
    detail: list[str] = Field(default_factory=list)


class PredictResponse(BaseModel):
    """A predicted final titre, and what produced it.

    The specification leaves the response schema to the implementer and describes
    only "a predicted final titer". This returns an **object** rather than a bare
    number so the contract can grow -- a prediction interval is the obvious
    addition -- without breaking callers.

    Args:
        predicted_titer: final titre, in the units of the training targets.
        experiment_id: echoed from the request when supplied.
        model: what produced the number.
        extrapolation: whether the request asked about conditions the model was
            fitted on. Reported because this model exists to extrapolate and its
            accuracy differs materially inside and outside the training range.
    """

    predicted_titer: float
    experiment_id: str | None = None
    model: ModelDescription
    extrapolation: ExtrapolationReport


class ErrorResponse(BaseModel):
    """A rejected request, with enough detail to fix it.

    Deliberately not FastAPI's default ``{"detail": ...}`` shape alone: a caller
    sending a malformed trajectory needs to know *which* variable was wrong, not
    only that something was.
    """

    detail: str
    experiment_id: str | None = None

    @classmethod
    def from_error(cls, error: Exception, experiment_id: str | None = None) -> dict[str, Any]:
        """Build the JSON body for an error response."""
        return cls(detail=str(error), experiment_id=experiment_id).model_dump()
