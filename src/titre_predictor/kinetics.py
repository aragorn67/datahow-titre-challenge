"""The environmental factor ``F(z)`` and the elementary rate-law forms it is built from.

Specific productivity is modulated by the culture environment:

    F(z) = f_T(T) * f_G(Glc) * f_Q(Gln) * f_metab(Lac, NH4) * f_lys(Lysed) * f_pH(pH)

Each ``f_i`` is a named callable taking the measured state and returning a factor. The set
of mechanisms in force is therefore **data** -- a tuple of names looked up in
:data:`MECHANISMS` -- rather than branching logic inside the model. Adding or removing a
mechanism is adding or removing a registry entry, and the fitting code in ``model.py``
reads the parameter count, units and search grids off the registry instead of hardcoding
them per variant. That is what makes the M1/M2/M3 comparison mechanical rather than three
parallel implementations.

Why saturating forms rather than linear terms
---------------------------------------------
Two reasons, both about what happens outside the fitted range.

*Interpretability.* A half-saturation constant is a physical quantity that can be checked
against literature. ``K_L = 40 mM`` is a claim; a regression slope is not.

*Extrapolation safety.* The test runs reach ``X:Lysed`` of 1.02 against a training maximum
of 0.53, and glucose 56.8 against 44.0. The Monod and inhibition forms are monotone and
bounded in ``(0, 1]``, so beyond the fitted range they flatten rather than diverge. A
polynomial or a bare linear coefficient would run away exactly where the test set lives.

The exponential form is the exception, and deliberately so
-----------------------------------------------------------
:func:`exponential_response` is monotone but **not** bounded -- it grows without limit as
its argument moves away from the reference. That is tolerable here only because the two
variables using it are controlled and the test set stays inside the training range:
temperature 35.05-37.95 against 35.09-37.99, pH 6.03-7.47 against 6.03-7.50. There is no
extrapolation on these axes to run away on. The alternative, a Gaussian optimum, needs two
parameters and would place ``T_opt`` outside a 3 degC observation window -- reporting a
``T_opt`` of 41 degC fitted from 35-38 degC data would be indefensible. Escalate to an
optimum only if the residuals show curvature.

Reference values
----------------
The reference in an exponential form sets only the scale of ``beta``, which absorbs it:
shifting the reference multiplies every factor by a constant. It exists so that ``beta``
reads as productivity at a nominal set point rather than at 0 degC or pH 0.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from titre_predictor.data import schema
from titre_predictor.domain import ExperimentRun

# Reference points for the exponential forms: the mid-range of the observed data, so the
# factor equals one near the middle of the design space rather than at an unreachable value.
REFERENCE_TEMPERATURE_CELSIUS = 36.5  # observed 35.0-38.0
REFERENCE_PH = 6.75  # observed 6.03-7.50

# Below this a concentration-scale constant is treated as invalid rather than as a hard
# switch. A half-saturation constant of zero would make the Monod term a step function.
MINIMUM_SCALE_CONSTANT = 1e-12


# --- elementary forms ----------------------------------------------------------------
#
# Each takes plain arrays and returns a factor array. They are kept free of any knowledge
# of runs or column names so they can be tested against hand-computable values.


def monod(
    concentration: NDArray[np.float64],
    half_saturation: float,
) -> NDArray[np.float64]:
    """Substrate limitation: ``c / (K + c)``, rising from 0 to 1 as substrate accumulates.

    Equals exactly ``0.5`` at ``c == K``, which is what makes ``K`` readable as a
    half-saturation constant in the units of the substrate.

    Args:
        concentration: substrate concentration, in mM.
        half_saturation: ``K``, in mM. The concentration at which the factor is one half.

    Raises:
        ValueError: if ``half_saturation`` is not strictly positive.
    """
    if half_saturation <= MINIMUM_SCALE_CONSTANT:
        raise ValueError(f"half_saturation must be strictly positive, got {half_saturation}")
    return np.asarray(concentration / (half_saturation + concentration), dtype=np.float64)


def inhibition(
    concentration: NDArray[np.float64],
    inhibition_constant: float,
) -> NDArray[np.float64]:
    """Non-competitive inhibition: ``1 / (1 + c/K)``, falling from 1 towards 0.

    Equals exactly ``0.5`` at ``c == K``. A large ``K`` means the inhibitor is not acting,
    so the search grid must reach high enough for "no effect" to be representable -- that
    is how the fit is allowed to reject a mechanism rather than being forced to use it.

    Args:
        concentration: inhibitor concentration, in the inhibitor's own units.
        inhibition_constant: ``K``, same units. The concentration halving productivity.

    Raises:
        ValueError: if ``inhibition_constant`` is not strictly positive.
    """
    if inhibition_constant <= MINIMUM_SCALE_CONSTANT:
        raise ValueError(
            f"inhibition_constant must be strictly positive, got {inhibition_constant}"
        )
    return np.asarray(1.0 / (1.0 + concentration / inhibition_constant), dtype=np.float64)


def combined_inhibition(
    first_concentration: NDArray[np.float64],
    second_concentration: NDArray[np.float64],
    first_constant: float,
    second_constant: float,
) -> NDArray[np.float64]:
    """Two inhibitors sharing one denominator: ``1 / (1 + a/K_a + b/K_b)``.

    This is *not* the product of two separate inhibition terms. Sharing the denominator
    says the two act through the same route -- here lactate and ammonia acting together
    through cellular energetics -- so their burdens add before taking effect. Two
    independent factors would say they act through separate routes and multiply.

    Args:
        first_concentration: first inhibitor, in mM.
        second_concentration: second inhibitor, in mM.
        first_constant: ``K_a``, in mM.
        second_constant: ``K_b``, in mM.

    Raises:
        ValueError: if either constant is not strictly positive.
    """
    if first_constant <= MINIMUM_SCALE_CONSTANT or second_constant <= MINIMUM_SCALE_CONSTANT:
        raise ValueError(
            f"inhibition constants must be strictly positive, got "
            f"{first_constant} and {second_constant}"
        )
    burden = first_concentration / first_constant + second_concentration / second_constant
    return np.asarray(1.0 / (1.0 + burden), dtype=np.float64)


def exponential_response(
    value: NDArray[np.float64],
    reference: float,
    sensitivity: float,
) -> NDArray[np.float64]:
    """Monotone response to a controlled variable: ``exp(theta * (reference - value))``.

    Equals exactly ``1.0`` at ``value == reference``. A positive ``sensitivity`` means the
    culture is more productive *below* the reference.

    An exponential is used rather than a polynomial so the factor stays strictly positive
    for any parameter value -- a negative productivity factor is not a physical state the
    optimiser should be able to reach. See the module docstring on why its unboundedness is
    acceptable for these two variables and would not be for the metabolites.

    Args:
        value: the controlled variable, in its own units (degC or pH units).
        reference: the value at which the factor equals one.
        sensitivity: ``theta``, per unit below the reference.
    """
    return np.asarray(np.exp(sensitivity * (reference - value)), dtype=np.float64)


# --- the mechanism registry ----------------------------------------------------------


@dataclass(frozen=True)
class ParameterSpec:
    """One fitted shape constant: what it is called, its units, and where to search.

    The search range travels with the parameter rather than living in the fitting code, so
    ``model.py`` builds its grid by reading the registry. A range is part of the mechanism's
    definition: it encodes what values are physically sensible for *that* constant, which
    the fitting code has no way to know.
    """

    name: str
    unit: str
    minimum: float
    maximum: float
    logarithmic: bool = True  # concentration scales span decades; sensitivities do not

    def grid(self, point_count: int) -> NDArray[np.float64]:
        """Candidate values spanning the search range.

        Args:
            point_count: how many candidates to return.
        """
        if point_count < 2:
            raise ValueError(f"need at least two grid points, got {point_count}")
        if self.logarithmic:
            return np.logspace(np.log10(self.minimum), np.log10(self.maximum), point_count)
        return np.linspace(self.minimum, self.maximum, point_count)


# The signature every mechanism honours: measured state in, one factor per timepoint out.
MechanismFunction = Callable[
    [Mapping[str, NDArray[np.float64]], Sequence[float]], NDArray[np.float64]
]


@dataclass(frozen=True)
class Mechanism:
    """One factor of ``F(z)``: the variables it reads, the constants it fits, and its form.

    Args:
        name: registry key, used in artefacts and in the printed rate law.
        description: what the factor claims physically, for the reported equation.
        required_series: the ``X:`` or ``W:`` names :attr:`evaluate` reads. Checked before
            fitting so a missing series is an error at setup rather than a ``KeyError``
            thousands of grid points into a search.
        parameters: the fitted shape constants, in the order :attr:`evaluate` expects.
        evaluate: state and parameter values in, one factor per timepoint out.
    """

    name: str
    description: str
    required_series: tuple[str, ...]
    parameters: tuple[ParameterSpec, ...]
    evaluate: MechanismFunction

    @property
    def parameter_count(self) -> int:
        return len(self.parameters)


def _glucose_limitation(
    state: Mapping[str, NDArray[np.float64]],
    parameters: Sequence[float],
) -> NDArray[np.float64]:
    return monod(state[schema.OBSERVATION_GLUCOSE], parameters[0])


def _glutamine_limitation(
    state: Mapping[str, NDArray[np.float64]],
    parameters: Sequence[float],
) -> NDArray[np.float64]:
    return monod(state[schema.OBSERVATION_GLUTAMINE], parameters[0])


def _metabolic_burden(
    state: Mapping[str, NDArray[np.float64]],
    parameters: Sequence[float],
) -> NDArray[np.float64]:
    return combined_inhibition(
        state[schema.OBSERVATION_LACTATE],
        state[schema.OBSERVATION_AMMONIA],
        parameters[0],
        parameters[1],
    )


def _lysate_inhibition(
    state: Mapping[str, NDArray[np.float64]],
    parameters: Sequence[float],
) -> NDArray[np.float64]:
    return inhibition(state[schema.OBSERVATION_LYSED_CELLS], parameters[0])


def _temperature_response(
    state: Mapping[str, NDArray[np.float64]],
    parameters: Sequence[float],
) -> NDArray[np.float64]:
    return exponential_response(
        state[schema.CONTROL_TEMPERATURE], REFERENCE_TEMPERATURE_CELSIUS, parameters[0]
    )


def _ph_response(
    state: Mapping[str, NDArray[np.float64]],
    parameters: Sequence[float],
) -> NDArray[np.float64]:
    return exponential_response(state[schema.CONTROL_PH], REFERENCE_PH, parameters[0])


# Search ranges. The concentration scales are log-spaced and deliberately reach an order of
# magnitude beyond the observed range at the top, so that "this inhibitor does nothing" is
# inside the grid: the fit must be able to switch a mechanism off rather than be obliged to
# find an effect. The lower ends sit below the smallest non-zero concentration observed.
#
# The exponential sensitivities span +/-6, widened from +/-3 on evidence: theta_pH fitted
# hard against the narrower bound and only settled, at about 4.28, once it was widened.
#
# A SEARCH range and a PROFILE range are different requirements and are kept apart. This one
# must contain the optimum and be resolvable by the grid; widening it further coarsens the
# first sweep and measurably degrades the fit (cross-validated RMSE 440 -> 462 at +/-12).
# A profile range must instead contain the whole confidence interval, whose upper bound here
# is 6.19, so profile_likelihood() takes its own range rather than inheriting this one.
#
# A parameter resting on its bound is not an estimate. FitDiagnostics reports any that do
# rather than leaving it to be noticed by eye.
MECHANISMS: dict[str, Mechanism] = {
    "glucose_limitation": Mechanism(
        name="glucose_limitation",
        description="Monod limitation by glucose",
        required_series=(schema.OBSERVATION_GLUCOSE,),
        parameters=(ParameterSpec("K_G", "mM", 1e-2, 1e3),),
        evaluate=_glucose_limitation,
    ),
    "glutamine_limitation": Mechanism(
        name="glutamine_limitation",
        description="Monod limitation by glutamine",
        required_series=(schema.OBSERVATION_GLUTAMINE,),
        parameters=(ParameterSpec("K_Q", "mM", 1e-2, 1e3),),
        evaluate=_glutamine_limitation,
    ),
    "metabolic_burden": Mechanism(
        name="metabolic_burden",
        description="Combined lactate and ammonia inhibition through shared energetics",
        required_series=(schema.OBSERVATION_LACTATE, schema.OBSERVATION_AMMONIA),
        parameters=(
            ParameterSpec("K_L", "mM", 1e-1, 1e4),
            ParameterSpec("K_A", "mM", 1e-1, 1e4),
        ),
        evaluate=_metabolic_burden,
    ),
    "lysate_inhibition": Mechanism(
        name="lysate_inhibition",
        description="Inhibition by lysed-cell material; synthesis and degradation are "
        "not separable at a single endpoint",
        required_series=(schema.OBSERVATION_LYSED_CELLS,),
        parameters=(ParameterSpec("K_X", "same units as VCD", 1e-3, 1e2),),
        evaluate=_lysate_inhibition,
    ),
    "temperature_response": Mechanism(
        name="temperature_response",
        description="Monotone temperature response about the reference",
        required_series=(schema.CONTROL_TEMPERATURE,),
        parameters=(ParameterSpec("theta_T", "per degC", -6.0, 6.0, logarithmic=False),),
        evaluate=_temperature_response,
    ),
    "ph_response": Mechanism(
        name="ph_response",
        description="Monotone pH response about the reference",
        required_series=(schema.CONTROL_PH,),
        parameters=(ParameterSpec("theta_pH", "per pH unit", -6.0, 6.0, logarithmic=False),),
        evaluate=_ph_response,
    ),
}


def state_from_run(run: ExperimentRun) -> dict[str, NDArray[np.float64]]:
    """The measured state a mechanism reads: observations and control profiles together.

    Mechanisms consume both ``X:`` and ``W:`` series and should not have to know which is
    which. The two namespaces are disjoint by the prefix convention, so merging them cannot
    collide.

    Args:
        run: the experiment. Must already carry its ``W:`` profiles; a caller holding only
            the ``Z:`` scalars should reconstruct them first with
            :func:`titre_predictor.data.controls.reconstruct_control_profiles`.
    """
    return {**run.observations, **run.control_profiles}


def resolve(mechanism_names: Sequence[str]) -> tuple[Mechanism, ...]:
    """Look up mechanisms by name, with a readable error for an unknown one.

    Args:
        mechanism_names: registry keys, in the order their parameters will be supplied.

    Raises:
        KeyError: if a name is not in :data:`MECHANISMS`, listing what is available.
        ValueError: if a name is repeated -- a factor applied twice is almost certainly a
            mistake, and would silently square that mechanism's contribution.
    """
    duplicates = sorted({name for name in mechanism_names if mechanism_names.count(name) > 1})
    if duplicates:
        raise ValueError(f"mechanisms repeated: {duplicates}")
    unknown = [name for name in mechanism_names if name not in MECHANISMS]
    if unknown:
        raise KeyError(f"unknown mechanisms {unknown}; available: {sorted(MECHANISMS)}")
    return tuple(MECHANISMS[name] for name in mechanism_names)


def parameter_specs(mechanisms: Sequence[Mechanism]) -> tuple[ParameterSpec, ...]:
    """Every fitted shape constant across a mechanism set, concatenated in order.

    This is the ordering the fitting code slices a flat parameter vector by, so it is
    defined once here rather than reimplemented at each call site.
    """
    return tuple(spec for mechanism in mechanisms for spec in mechanism.parameters)


def environmental_factor(
    mechanisms: Sequence[Mechanism],
    state: Mapping[str, NDArray[np.float64]],
    parameters: Sequence[float],
) -> NDArray[np.float64]:
    """``F(z)`` at every timepoint: the product of the mechanisms in force.

    An empty mechanism set gives ``F == 1`` at every timepoint, which is what makes M1 the
    exact special case of M2 and M3 rather than an approximation of them. The nesting is
    therefore a property of this function, not something the model variants arrange
    separately.

    Args:
        mechanisms: the factors in force, from :func:`resolve`.
        state: measured series, from :func:`state_from_run`.
        parameters: shape constants for every mechanism, concatenated in the order given by
            :func:`parameter_specs`.

    Returns:
        One factor per timepoint.

    Raises:
        KeyError: if a mechanism's required series is absent from ``state``.
        ValueError: if the number of parameters does not match the mechanism set.
    """
    expected = sum(mechanism.parameter_count for mechanism in mechanisms)
    if len(parameters) != expected:
        raise ValueError(
            f"mechanism set {[mechanism.name for mechanism in mechanisms]} takes "
            f"{expected} parameters, got {len(parameters)}"
        )

    length = _state_length(state)
    factor = np.ones(length, dtype=np.float64)
    offset = 0
    for mechanism in mechanisms:
        missing = [name for name in mechanism.required_series if name not in state]
        if missing:
            raise KeyError(
                f"mechanism {mechanism.name!r} needs {missing}, which the run does not "
                f"carry; available: {sorted(state)}"
            )
        count = mechanism.parameter_count
        factor = factor * mechanism.evaluate(state, parameters[offset : offset + count])
        offset += count
    return factor


def _state_length(state: Mapping[str, NDArray[np.float64]]) -> int:
    """Number of timepoints, taken from any series; they are equal length by construction.

    ``ExperimentRun`` validates that on construction, so this only has to pick one. It
    raises rather than guessing when handed an empty state, because returning a
    zero-length factor would let an empty run pass silently into the quadrature.
    """
    for values in state.values():
        return int(values.size)
    raise ValueError("cannot evaluate an environmental factor against an empty state")
