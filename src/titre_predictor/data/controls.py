"""Reconstruction of the ``W:`` control profiles from the ``Z:`` design scalars.

The control profiles are exact step functions of the design scalars. This was verified
against both supplied files: the reconstruction below reproduces all four ``W:`` columns
to machine precision across all 1290 rows. ``W:`` therefore carries no information
beyond ``Z:`` — it is a re-parameterisation, not an extra input.

Two uses:

* training — assert the convention still holds, so a change to the data format fails
  loudly rather than silently altering the model's inputs;
* inference — derive the controls when a request omits them, or validate them when a
  request supplies them.
"""

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from titre_predictor.data import schema


def reconstruct_control_profiles(
    design_scalars: Mapping[str, float],
    timestamps: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:
    """Rebuild the four ``W:`` profiles from the ``Z:`` scalars.

    Conventions, verified against both supplied CSV files:

        W:temp    = tempStart   while t <  tempShift, else tempEnd
        W:pH      = phStart     while t <  phShift,   else phEnd
        W:FeedGlc = FeedRateGlc while FeedStart <= t < FeedEnd, else 0
        W:FeedGln = FeedRateGln while FeedStart <= t < FeedEnd, else 0

    Note the feed window is closed on the left and open on the right: a run with
    ``FeedEnd = 11`` is fed on days 3..10 inclusive and not on day 11.

    Args:
        design_scalars: ``Z:``-prefixed values for one run.
        timestamps: sample times in days.

    Returns:
        The four ``W:`` profiles, each the same length as ``timestamps``.

    Raises:
        KeyError: if a design scalar required by the conventions is absent.
    """
    temperature = np.where(
        timestamps < design_scalars[schema.DESIGN_TEMPERATURE_SHIFT],
        design_scalars[schema.DESIGN_TEMPERATURE_START],
        design_scalars[schema.DESIGN_TEMPERATURE_END],
    )
    ph = np.where(
        timestamps < design_scalars[schema.DESIGN_PH_SHIFT],
        design_scalars[schema.DESIGN_PH_START],
        design_scalars[schema.DESIGN_PH_END],
    )

    feeding = (timestamps >= design_scalars[schema.DESIGN_FEED_START]) & (
        timestamps < design_scalars[schema.DESIGN_FEED_END]
    )
    feed_glucose = np.where(feeding, design_scalars[schema.DESIGN_FEED_RATE_GLUCOSE], 0.0)
    feed_glutamine = np.where(feeding, design_scalars[schema.DESIGN_FEED_RATE_GLUTAMINE], 0.0)

    return {
        schema.CONTROL_TEMPERATURE: temperature.astype(np.float64),
        schema.CONTROL_PH: ph.astype(np.float64),
        schema.CONTROL_FEED_GLUCOSE: feed_glucose.astype(np.float64),
        schema.CONTROL_FEED_GLUTAMINE: feed_glutamine.astype(np.float64),
    }


def control_profiles_match(
    supplied: Mapping[str, NDArray[np.float64]],
    reconstructed: Mapping[str, NDArray[np.float64]],
    absolute_tolerance: float = 1e-9,  # 1e-9 (exact-match check) | 1e-6 (lenient)
) -> bool:
    """Whether supplied control profiles agree with those implied by the design scalars.

    Args:
        supplied: ``W:`` profiles as given, e.g. read from a CSV or a request body.
        reconstructed: output of :func:`reconstruct_control_profiles`.
        absolute_tolerance: the data reproduce exactly, so the default is tight; loosen
            only if a caller legitimately supplies rounded values.
    """
    if set(supplied) != set(reconstructed):
        return False
    return all(
        np.allclose(supplied[name], reconstructed[name], atol=absolute_tolerance, rtol=0.0)
        for name in reconstructed
    )
