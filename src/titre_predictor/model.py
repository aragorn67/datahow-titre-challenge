"""Fitting and applying the nested Luedeking-Piret titre models.

The rate law
------------
    qP(t) = ( alpha*mu_eff(t) + beta ) * F(z(t))
    P     = INT qP(t)*Xv(t) dt

Four variants, differing only in **where** the environmental factor is applied and in
whether growth is counted as net or effective. That is a testable claim, not an assertion,
so all four are fitted and compared rather than one being asserted:

    M0   qP = alpha*mu_net + beta          P = alpha*dXv  + beta*gammaX
    M1   qP = alpha*mu_eff + beta          P = alpha*dC   + beta*gammaX
    M2   qP = alpha*mu_eff + beta*F        P = alpha*dC   + beta*INT F*Xv dt
    M3   qP = (alpha*mu_eff + beta)*F      P = alpha*sum_j Fbar_j dC_j + beta*INT F*Xv dt

where ``INT F*Xv dt`` is accumulated as a trapezoid of the product ``F*Xv`` and ``Fbar_j``
is the interval average of ``F``, which is the only weight an endpoint difference like
``dC_j`` admits. Both quadratures are defined once in ``features.py``; the vectorised search
below re-derives them from the same pointwise factor rather than approximating them, so the
objective being minimised is exactly the quantity :func:`design_columns` predicts with.

M0 is the documented failure, kept as a benchmark so the difference between net and
effective growth is *demonstrated* rather than claimed: it previously fitted alpha = -9.9,
the model asserting that secreted antibody disappears as cells die.

M3 minus M2 has a clean mechanistic reading. M2 says the environment changes the rate of
non-growth-associated production. M3 says it additionally changes the yield of product per
cell synthesised. Fewer cells made versus less product per cell made -- distinct claims,
and the comparison tests exactly that.

Because ``F == 1`` when no mechanisms are in force, M2 and M3 collapse **exactly** onto M1,
and the nesting is a property of the arithmetic rather than something arranged here.

How it is fitted
----------------
Every variant is **linear in (alpha, beta)** once the shape constants -- ``kl`` and the
mechanism parameters -- are fixed. That linearity is a property of Luedeking-Piret, not an
empirical shortcut, and it is what lets one procedure serve all four:

* **inner:** the 2x2 normal equations, solved in closed form;
* **outer:** a grid over the shape constants, with successive refinement.

So the least squares sits *inside* the optimiser rather than in the model. There is no
starting guess, no convergence criterion and no local minimum to miss -- a materially
stronger position than the multi-start Nelder-Mead the reference papers need, because their
systems are fully nonlinear in 5-20 parameters. A grid also suits an objective made
piecewise by the non-negativity in the lysate fit.

The environmental factor is evaluated **once per mechanism sub-grid point**, not once per
combination: ``F`` is a product over mechanisms, so each mechanism's contribution is
precomputed against its own parameter grid and the combinations are formed by multiplying
lookups. Without that the cost would be one ``exp`` or Monod evaluation per combination per
run per timepoint.

The coefficients are fitted **unconstrained**. Their sign is diagnostic: a negative
``alpha`` says antibody is destroyed per cell produced, which means the structure is wrong
rather than that the fit needs a constraint. That is exactly how M0 was found to be broken.

Choice of loss
--------------
Both an absolute least squares and a relative one (weights ``1/y^2``) are supported and
compared. They answer different questions -- see ``evaluation.py`` on why RMSE and MAPE
disagree here -- and the relative loss is the one aligned with the recorded weakness on
MAPE. Which to ship is a decision made on cross-validated error, not here.
"""

import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from titre_predictor import features, kinetics
from titre_predictor.data import schema
from titre_predictor.domain import ExperimentRun

# Search range for the lysis rate constant, in reciprocal days. Spans four decades around
# the value the data imply, so the optimum is interior rather than pinned at an edge.
LYSIS_RATE_SPEC = kinetics.ParameterSpec("kl", "1/day", 1e-4, 1e1)

# Grid points per shape constant per sweep, and narrowing passes after the first sweep.
#
# The combination count grows as GRID_POINTS raised to the number of shape constants, so a
# coarse sweep with many refinements dominates a dense sweep with few. Measured on the 90
# training runs with six shape constants: (9 points, 4 refinements) takes 64.6 s and
# (5, 8) takes 3.1 s, for a residual sum of squares identical to five significant figures.
#
# The speed is not a convenience. Screening fits one model per mechanism combination per
# fold, so at a minute per fit there is pressure to trim combinations or folds -- exactly
# the wrong economy when the purpose of that stage is honest selection.
#
# Convergence is checked rather than assumed: against a much denser sweep (21 points, 6
# refinements) these settings agree on the residual sum of squares to 0.000% across M1, M3
# with one mechanism, and M3 with three.
DEFAULT_GRID_POINTS = 5  # 5 | 9 | 13
DEFAULT_SEARCH_REFINEMENTS = 8  # 0 (single sweep) | 4 | 8 | 12

# Combinations evaluated per batch. Bounds peak memory at roughly
# chunk * runs * timepoints floats regardless of how large the grid is.
DEFAULT_CHUNK_SIZE = 4096

# Below this the 2x2 normal equations are treated as singular: the two regressors have
# become effectively parallel and the coefficients are meaningless.
SINGULAR_DETERMINANT_TOLERANCE = 1e-12

Loss = Literal["absolute", "relative"]


@dataclass(frozen=True)
class ModelVariant:
    """Where the environmental factor acts, and how growth is counted.

    Three booleans span the four variants, so adding a fifth placement is a registry entry
    rather than another branch in the fitting code.
    """

    name: str
    description: str
    uses_effective_growth: bool
    factor_on_growth: bool
    factor_on_non_growth: bool

    @property
    def needs_lysis_rate(self) -> bool:
        """M0 regresses on net growth, so it never forms a dead-cell pool."""
        return self.uses_effective_growth

    @property
    def needs_factor(self) -> bool:
        return self.factor_on_growth or self.factor_on_non_growth


VARIANTS: dict[str, ModelVariant] = {
    "M0": ModelVariant(
        "M0",
        "net growth, no environmental factor -- the documented failure, kept as a benchmark",
        uses_effective_growth=False,
        factor_on_growth=False,
        factor_on_non_growth=False,
    ),
    "M1": ModelVariant(
        "M1",
        "effective growth, no environmental factor",
        uses_effective_growth=True,
        factor_on_growth=False,
        factor_on_non_growth=False,
    ),
    "M2": ModelVariant(
        "M2",
        "environmental factor on the non-growth term only",
        uses_effective_growth=True,
        factor_on_growth=False,
        factor_on_non_growth=True,
    ),
    "M3": ModelVariant(
        "M3",
        "environmental factor on the whole rate law",
        uses_effective_growth=True,
        factor_on_growth=True,
        factor_on_non_growth=True,
    ),
}


def resolve_variant(name: str) -> ModelVariant:
    """Look up a model variant, with a readable error for an unknown one."""
    try:
        return VARIANTS[name]
    except KeyError:
        raise KeyError(f"unknown model variant {name!r}; available: {sorted(VARIANTS)}") from None


@dataclass(frozen=True)
class FitDiagnostics:
    """What the fit reveals about how well the parameters are determined.

    ``coefficient_correlation`` is the number to watch. The two regressors measure closely
    related quantities -- a run that makes many cells also accumulates many cell-days -- so
    a magnitude approaching one means ``alpha`` and ``beta`` are not separately identified,
    whatever their point estimates look like. An earlier version of this model reached
    -0.973.

    These standard errors are **conditional** on the fitted shape constants: they are what
    you would get if those were known exactly, so they exclude that uncertainty and are
    optimistic. A bootstrap or profile is what quantifies it, and that is ``uncertainty.py``.

    They are also **post-selection**. Computed after choosing mechanisms on the same data,
    they are optimistically biased: the point estimates are usable, the standard errors are
    not valid as stated.
    """

    alpha_standard_error: float
    beta_standard_error: float
    coefficient_correlation: float
    residual_standard_deviation: float
    residual_sum_of_squares: float
    training_run_count: int
    shape_constant_names: tuple[str, ...]
    shape_constant_values: tuple[float, ...]
    pinned_parameters: tuple[str, ...] = ()
    """Shape constants resting on an edge of their search range.

    A parameter at its bound is not an estimate -- the data wanted to go further and the
    grid stopped it -- so it must be either widened or reported as a bound rather than
    quoted as a fitted value. Two cases look identical here and are told apart by widening:
    a range set too narrow, and a mechanism the fit is switching off by sending its
    constant to infinity, which is a legitimate answer meaning 'this factor does nothing'.
    """


@dataclass(frozen=True)
class LuedekingPiretModel:
    """A fitted rate law, and everything needed to reproduce a prediction.

    The mechanism names and their fitted constants travel with the coefficients, because
    ``alpha`` and ``beta`` were fitted against features computed with that exact mechanism
    set: predicting with a different one would apply them to different quantities.
    """

    variant: str
    alpha: float
    beta: float
    lysis_rate_constant: float
    mechanisms: tuple[str, ...] = ()
    mechanism_parameters: tuple[float, ...] = ()
    loss: Loss = "absolute"
    ridge_penalty: float = 0.0

    def _columns(self, run: ExperimentRun) -> tuple[float, float]:
        quantities = features.run_quantities(run)
        growth, non_growth = design_columns(
            resolve_variant(self.variant),
            [quantities],
            self.lysis_rate_constant,
            kinetics.resolve(self.mechanisms),
            self.mechanism_parameters,
        )
        return float(growth[0]), float(non_growth[0])

    def predict(self, run: ExperimentRun) -> float:
        """Predicted final titre for one experiment.

        One rate-law evaluation per interval plus one sum, however expensive the fit was.
        """
        growth, non_growth = self._columns(run)
        return float(self.alpha * growth + self.beta * non_growth)

    def predict_many(self, runs: Sequence[ExperimentRun]) -> NDArray[np.float64]:
        """Predicted final titres, in the order the runs were given."""
        return np.array([self.predict(run) for run in runs], dtype=np.float64)

    def rate_law(self) -> str:
        """The fitted rate law as an equation with constants and units.

        This is the deliverable the brief asks for -- a mechanistic statement that can be
        checked against literature -- rather than a table of regression coefficients.
        """
        variant = resolve_variant(self.variant)
        growth_symbol = "mu_eff(t)" if variant.uses_effective_growth else "mu_net(t)"
        core = f"({self.alpha:.4g} * {growth_symbol} + {self.beta:.4g})"
        if variant.factor_on_growth:
            body = f"{core} * F(z(t))"
        elif variant.factor_on_non_growth:
            body = f"{self.alpha:.4g} * {growth_symbol} + {self.beta:.4g} * F(z(t))"
        else:
            body = core

        lines = [f"qP(t) = {body}", ""]
        mechanisms = kinetics.resolve(self.mechanisms)
        if mechanisms:
            lines.append("F(z) = " + " * ".join(m.name for m in mechanisms))
            for mechanism in mechanisms:
                lines.append(f"    {mechanism.name}: {mechanism.description}")
            lines.append("")
        lines.append("Fitted constants")
        lines.append(f"    {'alpha':<8} = {self.alpha:>12.4g}   titre per cell synthesised")
        lines.append(f"    {'beta':<8} = {self.beta:>12.4g}   titre per cell-day")
        if variant.needs_lysis_rate:
            lines.append(f"    {'kl':<8} = {self.lysis_rate_constant:>12.4g}   1/day")
        for spec, value in zip(
            kinetics.parameter_specs(mechanisms), self.mechanism_parameters, strict=True
        ):
            lines.append(f"    {spec.name:<8} = {value:>12.4g}   {spec.unit}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serialisable representation, for the inference service artefact."""
        return {
            "variant": self.variant,
            "alpha": self.alpha,
            "beta": self.beta,
            "lysis_rate_constant": self.lysis_rate_constant,
            "mechanisms": list(self.mechanisms),
            "mechanism_parameters": list(self.mechanism_parameters),
            "loss": self.loss,
            "ridge_penalty": self.ridge_penalty,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LuedekingPiretModel":
        """Rebuild a model from :meth:`to_dict`.

        Raises:
            KeyError: if the stored variant or a stored mechanism is not known to this code.
                That means the artefact came from a different version of the package and its
                coefficients no longer correspond to these quantities.
            ValueError: if the stored parameter count does not match the stored mechanisms.
        """
        mechanisms = tuple(payload["mechanisms"])
        parameters = tuple(float(value) for value in payload["mechanism_parameters"])
        resolve_variant(str(payload["variant"]))
        expected = len(kinetics.parameter_specs(kinetics.resolve(mechanisms)))
        if len(parameters) != expected:
            raise ValueError(
                f"artefact stores {len(parameters)} mechanism parameters but its mechanisms "
                f"{list(mechanisms)} take {expected}"
            )
        return cls(
            variant=str(payload["variant"]),
            alpha=float(payload["alpha"]),
            beta=float(payload["beta"]),
            lysis_rate_constant=float(payload["lysis_rate_constant"]),
            mechanisms=mechanisms,
            mechanism_parameters=parameters,
            loss=payload.get("loss", "absolute"),
            ridge_penalty=float(payload.get("ridge_penalty", 0.0)),
        )

    def save(self, artefact_path: Path, provenance: Mapping[str, Any] | None = None) -> None:
        """Write the model to a JSON file, creating parent directories as needed.

        Args:
            artefact_path: where to write.
            provenance: optional record of how the model was produced -- data hash, seed,
                package versions, timestamp. Stored alongside the parameters so a served
                prediction can be traced back to the run that produced it, and ignored by
                :meth:`from_dict`, which reads only the fields it needs.
        """
        payload = self.to_dict()
        if provenance is not None:
            payload["provenance"] = dict(provenance)
        artefact_path.parent.mkdir(parents=True, exist_ok=True)
        artefact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, artefact_path: Path) -> "LuedekingPiretModel":
        """Read a model written by :meth:`save`."""
        return cls.from_dict(json.loads(artefact_path.read_text(encoding="utf-8")))


# --- design columns -------------------------------------------------------------------


def design_columns(
    variant: ModelVariant,
    quantities: Sequence[features.RunQuantities],
    lysis_rate_constant: float,
    mechanisms: Sequence[kinetics.Mechanism],
    parameters: Sequence[float],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The two regressors per run, for any variant.

    Returns:
        ``(growth_column, non_growth_column)``, each one value per run.
    """
    growth = np.empty(len(quantities), dtype=np.float64)
    non_growth = np.empty(len(quantities), dtype=np.float64)

    for index, item in enumerate(quantities):
        per_interval_growth = (
            item.interval_growth(lysis_rate_constant)
            if variant.uses_effective_growth
            else item.interval_viable_change
        )
        if variant.factor_on_growth:
            factor = features.interval_factor(mechanisms, item, parameters)
            growth[index] = float(np.sum(factor * per_interval_growth))
        else:
            growth[index] = float(np.sum(per_interval_growth))

        if variant.factor_on_non_growth:
            # Trapezoidal in the product F*Xv, not the product of separate averages.
            non_growth[index] = float(
                np.sum(features.interval_weighted_cell_days(mechanisms, item, parameters))
            )
        else:
            non_growth[index] = float(np.sum(item.interval_cell_days))
    return growth, non_growth


@dataclass(frozen=True, eq=False)
class _Padded:
    """Ragged per-interval arrays padded to the longest run, for the vectorised search.

    Padding cell-days with zero is safe whatever weight they receive, and the state arrays
    are padded by repeating the last observation so the factor stays finite there rather
    than producing a ``nan`` that would poison the sum.
    """

    cell_days: NDArray[np.float64]  # (runs, intervals)
    viable_change: NDArray[np.float64]  # (runs, intervals)
    growth_base: NDArray[np.float64]  # (runs, intervals)
    lysate_slope_change: NDArray[np.float64]  # (runs, intervals)
    interval_length: NDArray[np.float64]  # (runs, intervals) -- zero where padded
    viable_density: NDArray[np.float64]  # (runs, points)
    state: dict[str, NDArray[np.float64]]  # (runs, points)


def _pad(quantities: Sequence[features.RunQuantities]) -> _Padded:
    run_count = len(quantities)
    intervals = max(item.interval_cell_days.size for item in quantities)
    points = intervals + 1

    cell_days = np.zeros((run_count, intervals), dtype=np.float64)
    viable_change = np.zeros((run_count, intervals), dtype=np.float64)
    growth_base = np.zeros((run_count, intervals), dtype=np.float64)
    slope_change = np.zeros((run_count, intervals), dtype=np.float64)
    interval_length = np.zeros((run_count, intervals), dtype=np.float64)

    series_names = sorted(quantities[0].state)
    state = {name: np.zeros((run_count, points), dtype=np.float64) for name in series_names}

    for index, item in enumerate(quantities):
        width = item.interval_cell_days.size
        cell_days[index, :width] = item.interval_cell_days
        viable_change[index, :width] = item.interval_viable_change
        growth_base[index, :width] = item.interval_growth_base
        slope_change[index, :width] = item.interval_lysate_slope_change
        # Padded intervals get zero length, so whatever the state pads to contributes nothing
        # to a quadrature regardless of the factor evaluated there.
        interval_length[index, :width] = np.diff(item.timestamps)
        for name in series_names:
            values = item.state[name]
            state[name][index, : values.size] = values
            state[name][index, values.size :] = values[-1]

    return _Padded(
        cell_days,
        viable_change,
        growth_base,
        slope_change,
        interval_length,
        state[schema.OBSERVATION_VIABLE_CELL_DENSITY],
        state,
    )


def _mechanism_factor_tables(
    mechanisms: Sequence[kinetics.Mechanism],
    padded: _Padded,
    grids: Sequence[NDArray[np.float64]],
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    """Precompute each mechanism's factor at every point of its own sub-grid.

    ``F`` is a product over mechanisms, so evaluating each mechanism once per point of its
    *own* grid and multiplying lookups costs a small fraction of evaluating the whole
    product once per combination of every mechanism's parameters.

    The tables hold **pointwise** factors, and the quadrature is applied only after the
    product across mechanisms has been formed. Averaging each mechanism over the interval
    first and then multiplying would compute a product of means, which for more than one
    mechanism is not the mean of the product: this search would then be minimising a
    different quantity from the one :func:`design_columns` predicts with, and the fitted
    coefficients would not belong to the model that serves them.

    Returns:
        ``(tables, sub_grids)`` where ``tables[m]`` has shape
        ``(sub_grid_points, runs, points)``.
    """
    tables: list[NDArray[np.float64]] = []
    sub_grids: list[NDArray[np.float64]] = []
    offset = 0
    for mechanism in mechanisms:
        count = mechanism.parameter_count
        combinations = np.array(
            list(itertools.product(*grids[offset : offset + count])), dtype=np.float64
        )
        table = np.empty(
            (combinations.shape[0], *padded.viable_density.shape),
            dtype=np.float64,
        )
        for row, values in enumerate(combinations):
            table[row] = mechanism.evaluate(padded.state, list(values))
        tables.append(table)
        sub_grids.append(combinations)
        offset += count
    return tables, sub_grids


def _solve(
    growth: NDArray[np.float64],
    non_growth: NDArray[np.float64],
    targets: NDArray[np.float64],
    weights: NDArray[np.float64],
    ridge_penalty: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Weighted 2x2 ridge least squares at every candidate at once.

    **The columns are standardised before the penalty is applied.** ``alpha`` multiplies
    cells synthesised and ``beta`` multiplies cell-days, quantities differing by an order of
    magnitude and carrying different units, so an unstandardised penalty would shrink them
    unequally by an accident of scale. Concretely, on this data the unpenalised fit already
    has the smaller raw norm of the two ends of the ridge, so a naive penalty would push
    *away* from the better-extrapolating end rather than towards it.

    With columns scaled to unit weighted mean square, the cross term becomes the weighted
    correlation between the regressors, and ``ridge_penalty`` is a dimensionless multiple of
    the total weight -- so the same value means the same amount of shrinkage regardless of
    the run count, the loss, or the units of titre.

    A penalty of exactly zero recovers ordinary least squares, so it must remain in the
    search grid: the fit has to be able to decline shrinkage rather than be obliged to
    accept it.

    Args:
        growth: ``(candidates, runs)``.
        non_growth: ``(candidates, runs)``.
        targets: ``(runs,)``.
        weights: ``(runs,)``.
        ridge_penalty: shrinkage on the standardised coefficients. Zero is plain OLS.

    Returns:
        ``(alpha, beta, objective)``, each ``(candidates,)``. The objective is the penalised
        weighted sum of squares, which is what the shape constants are chosen to minimise --
        selecting them on the unpenalised residual while the coefficients are shrunk would
        optimise two different criteria against each other. Singular candidates carry an
        infinite objective so they lose the comparison.
    """
    weighted_growth = growth * weights
    weighted_non_growth = non_growth * weights

    growth_squared = np.sum(weighted_growth * growth, axis=1)
    cross = np.sum(weighted_growth * non_growth, axis=1)
    non_growth_squared = np.sum(weighted_non_growth * non_growth, axis=1)
    growth_target = weighted_growth @ targets
    non_growth_target = weighted_non_growth @ targets

    total_weight = float(np.sum(weights))
    positive = (growth_squared > 0.0) & (non_growth_squared > 0.0)
    growth_scale = np.sqrt(np.where(positive, growth_squared, 1.0) / total_weight)
    non_growth_scale = np.sqrt(np.where(positive, non_growth_squared, 1.0) / total_weight)

    # Normal equations in standardised coordinates, where both diagonal entries are the
    # total weight and the off-diagonal is the weighted correlation times that weight.
    penalty = ridge_penalty * total_weight
    diagonal = total_weight + penalty
    standardised_cross = cross / (growth_scale * non_growth_scale)
    standardised_growth_target = growth_target / growth_scale
    standardised_non_growth_target = non_growth_target / non_growth_scale

    determinant = diagonal**2 - standardised_cross**2
    usable = positive & (np.abs(determinant) > SINGULAR_DETERMINANT_TOLERANCE)
    safe = np.where(usable, determinant, 1.0)

    standardised_alpha = (
        diagonal * standardised_growth_target - standardised_cross * standardised_non_growth_target
    ) / safe
    standardised_beta = (
        diagonal * standardised_non_growth_target - standardised_cross * standardised_growth_target
    ) / safe

    alpha = standardised_alpha / growth_scale
    beta = standardised_beta / non_growth_scale

    # Residual computed directly: with a penalty the fit is no longer a projection, so the
    # usual "total minus explained" shortcut does not hold.
    total = float(np.sum(weights * targets**2))
    residual = (
        total
        - 2.0 * (alpha * growth_target + beta * non_growth_target)
        + alpha**2 * growth_squared
        + 2.0 * alpha * beta * cross
        + beta**2 * non_growth_squared
    )
    objective = residual + penalty * (standardised_alpha**2 + standardised_beta**2)
    objective = np.where(usable, np.maximum(objective, 0.0), np.inf)
    return alpha, beta, objective


def fit(
    runs: Sequence[ExperimentRun],
    targets: dict[str, float],
    variant_name: str = "M3",
    mechanism_names: Sequence[str] = (),
    loss: Loss = "absolute",
    ridge_penalty: float = 0.0,
    fixed_shape_constants: Mapping[str, float] | None = None,
    grid_points: int = DEFAULT_GRID_POINTS,  # 5 | 9 | 13
    search_refinements: int = DEFAULT_SEARCH_REFINEMENTS,  # 0 | 2 | 4 | 6
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> tuple[LuedekingPiretModel, FitDiagnostics]:
    """Estimate a variant's parameters from runs and their measured titres.

    The shape constants are chosen to minimise the residual sum of squares **on these runs
    only**. Called inside a cross-validation fold it therefore sees only that fold's
    training data, which is what keeps a held-out estimate honest -- ``kl`` included.
    Reusing a ``kl`` fitted on all the data would leak the held-out runs into every fold. It
    feels structural, which is exactly why it is an easy mistake; it is fitted, so it goes
    inside.

    Args:
        runs: training experiments, each carrying its ``W:`` control profiles.
        targets: measured final titre per experiment identifier.
        variant_name: one of :data:`VARIANTS`.
        mechanism_names: mechanisms composing ``F``. Ignored by M0 and M1, which have no
            factor; passing them there is an error rather than a silent no-op.
        loss: ``"absolute"`` for ordinary least squares, ``"relative"`` for weights
            ``1/y^2``, which is squared relative error.
        grid_points: candidates per shape constant per sweep.
        search_refinements: narrowing passes after the first sweep. The first sweep is
            exhaustive over the whole range; each refinement then assumes the optimum lies
            within the bracket around the current best, which holds provided the first
            sweep resolves the basin.
        chunk_size: combinations evaluated per batch, bounding peak memory.

    Returns:
        The fitted model and its diagnostics.

    Raises:
        KeyError: if a run has no target, or a variant or mechanism name is unknown.
        ValueError: if fewer than three runs are supplied, if a target is non-positive
            under the relative loss, or if mechanisms are given to a variant with no factor.
    """
    variant = resolve_variant(variant_name)
    if len(runs) < 3:
        raise ValueError(
            f"need at least three runs to fit two coefficients and estimate error, got {len(runs)}"
        )
    missing = [run.experiment_id for run in runs if run.experiment_id not in targets]
    if missing:
        raise KeyError(f"no target for {missing}")
    if mechanism_names and not variant.needs_factor:
        raise ValueError(
            f"{variant.name} applies no environmental factor, so mechanisms "
            f"{list(mechanism_names)} would be silently ignored"
        )

    mechanisms = kinetics.resolve(mechanism_names) if variant.needs_factor else ()
    target_vector = np.array([targets[run.experiment_id] for run in runs], dtype=np.float64)
    if loss == "relative":
        if np.any(target_vector <= 0.0):
            raise ValueError("the relative loss needs strictly positive targets")
        weights = 1.0 / target_vector**2
    else:
        weights = np.ones_like(target_vector)

    # Reduce each run once. Nothing here depends on a fitted parameter, so recomputing it
    # per grid point would repeat identical spline fits many thousands of times.
    quantities = [features.run_quantities(run) for run in runs]
    padded = _pad(quantities)

    specs: list[kinetics.ParameterSpec] = []
    if variant.needs_lysis_rate:
        specs.append(LYSIS_RATE_SPEC)
    specs.extend(kinetics.parameter_specs(mechanisms))

    # A pinned constant keeps its place in the parameter vector but is given a
    # single-point grid, so the search optimises everything else *around* it. That is what
    # makes a profile a profile rather than a slice: the remaining parameters are free to
    # compensate, which is the honest test of whether the pinned one is determined.
    pinned = dict(fixed_shape_constants or {})
    unknown = sorted(set(pinned) - {spec.name for spec in specs})
    if unknown:
        raise KeyError(
            f"cannot fix {unknown}: not a shape constant of {variant.name} with mechanisms "
            f"{list(mechanism_names)}; available: {[spec.name for spec in specs]}"
        )

    bounds = [
        (pinned[spec.name], pinned[spec.name])
        if spec.name in pinned
        else (spec.minimum, spec.maximum)
        for spec in specs
    ]
    best_shape: tuple[float, ...] = ()
    best = (np.inf, 0.0, 0.0)  # residual, alpha, beta

    for _sweep in range(search_refinements + 1):
        grids = [
            np.array([pinned[spec.name]], dtype=np.float64)
            if spec.name in pinned
            else _grid_between(spec, low, high, grid_points)
            for spec, (low, high) in zip(specs, bounds, strict=True)
        ]
        candidate = _search(
            variant,
            padded,
            mechanisms,
            specs,
            grids,
            target_vector,
            weights,
            chunk_size,
            ridge_penalty,
        )
        if candidate[0] < best[0]:
            best = candidate[:3]
            best_shape = candidate[3]
        bounds = [
            (pinned[spec.name], pinned[spec.name]) if spec.name in pinned else _bracket(grid, value)
            for spec, grid, value in zip(specs, grids, best_shape, strict=True)
        ]
        if not specs or len(pinned) == len(specs):
            break

    residual, alpha, beta = best
    lysis_rate = best_shape[0] if variant.needs_lysis_rate else float("nan")
    mechanism_values = tuple(best_shape[1:] if variant.needs_lysis_rate else best_shape)

    model = LuedekingPiretModel(
        variant=variant.name,
        alpha=alpha,
        beta=beta,
        lysis_rate_constant=lysis_rate,
        mechanisms=tuple(mechanism_names) if variant.needs_factor else (),
        mechanism_parameters=mechanism_values,
        loss=loss,
        ridge_penalty=ridge_penalty,
    )
    growth, non_growth = design_columns(
        variant, quantities, lysis_rate, mechanisms, mechanism_values
    )
    diagnostics = _diagnostics(growth, non_growth, weights, residual, len(runs), specs, best_shape)
    return model, diagnostics


def _grid_between(
    spec: kinetics.ParameterSpec,
    low: float,
    high: float,
    point_count: int,
) -> NDArray[np.float64]:
    """Candidates between two bounds, spaced as the parameter's own scale demands."""
    if spec.logarithmic:
        return np.logspace(np.log10(low), np.log10(high), point_count)
    return np.linspace(low, high, point_count)


def _bracket(candidates: NDArray[np.float64], chosen: float) -> tuple[float, float]:
    """The neighbours either side of ``chosen``, for the next refinement sweep."""
    index = int(np.argmin(np.abs(candidates - chosen)))
    return (
        float(candidates[max(index - 1, 0)]),
        float(candidates[min(index + 1, candidates.size - 1)]),
    )


def _search(
    variant: ModelVariant,
    padded: _Padded,
    mechanisms: Sequence[kinetics.Mechanism],
    specs: Sequence[kinetics.ParameterSpec],
    grids: Sequence[NDArray[np.float64]],
    targets: NDArray[np.float64],
    weights: NDArray[np.float64],
    chunk_size: int,
    ridge_penalty: float = 0.0,
) -> tuple[float, float, float, tuple[float, ...]]:
    """One exhaustive sweep of the shape-constant grid.

    Returns:
        ``(residual, alpha, beta, shape_constants)`` for the best combination.
    """
    lysis_grid = grids[0] if variant.needs_lysis_rate else np.array([np.nan])
    mechanism_grids = grids[1:] if variant.needs_lysis_rate else grids
    tables, sub_grids = _mechanism_factor_tables(mechanisms, padded, mechanism_grids)

    # Index space: one axis for kl, one per mechanism sub-grid.
    axis_sizes = [lysis_grid.size] + [table.shape[0] for table in tables]
    index_grid = np.array(
        list(itertools.product(*[range(size) for size in axis_sizes])), dtype=np.intp
    )

    best: tuple[float, float, float, tuple[float, ...]] = (np.inf, 0.0, 0.0, ())
    for start in range(0, index_grid.shape[0], chunk_size):
        block = index_grid[start : start + chunk_size]
        lysis_values = lysis_grid[block[:, 0]]

        if variant.needs_factor and tables:
            # Pointwise across mechanisms first; the quadratures below then take the
            # trapezoid of the finished product, matching features.py exactly.
            pointwise_factor = np.ones(
                (block.shape[0], *padded.viable_density.shape), dtype=np.float64
            )
            for position, table in enumerate(tables):
                pointwise_factor = pointwise_factor * table[block[:, position + 1]]
        else:
            pointwise_factor = None

        if variant.uses_effective_growth:
            per_interval_growth = (
                padded.growth_base[None, :, :]
                + padded.lysate_slope_change[None, :, :] / lysis_values[:, None, None]
            )
        else:
            per_interval_growth = padded.viable_change[None, :, :]

        if variant.factor_on_growth and pointwise_factor is not None:
            # dC_j is an endpoint difference, so the interval average of F is its weight.
            interval_factor = 0.5 * (pointwise_factor[:, :, :-1] + pointwise_factor[:, :, 1:])
            growth = np.sum(interval_factor * per_interval_growth, axis=2)
        else:
            growth = np.sum(per_interval_growth, axis=2)
            if growth.shape[0] == 1 and block.shape[0] > 1:
                growth = np.repeat(growth, block.shape[0], axis=0)

        if variant.factor_on_non_growth and pointwise_factor is not None:
            # INT F*Xv dt, trapezoidal in the product rather than in each average.
            product = pointwise_factor * padded.viable_density[None, :, :]
            non_growth = np.sum(
                0.5 * (product[:, :, :-1] + product[:, :, 1:]) * padded.interval_length[None, :, :],
                axis=2,
            )
        else:
            non_growth = np.repeat(
                np.sum(padded.cell_days, axis=1)[None, :], block.shape[0], axis=0
            )

        alpha, beta, residual = _solve(growth, non_growth, targets, weights, ridge_penalty)
        position = int(np.argmin(residual))
        if float(residual[position]) < best[0]:
            shape: list[float] = []
            if variant.needs_lysis_rate:
                shape.append(float(lysis_values[position]))
            for axis, sub_grid in enumerate(sub_grids):
                shape.extend(float(value) for value in sub_grid[block[position, axis + 1]])
            best = (
                float(residual[position]),
                float(alpha[position]),
                float(beta[position]),
                tuple(shape),
            )
    return best


def _diagnostics(
    growth: NDArray[np.float64],
    non_growth: NDArray[np.float64],
    weights: NDArray[np.float64],
    residual_sum_of_squares: float,
    run_count: int,
    specs: Sequence[kinetics.ParameterSpec],
    shape_values: Sequence[float],
) -> FitDiagnostics:
    """Closed-form coefficient uncertainty, conditional on the shape constants."""
    design = np.column_stack([growth, non_growth])
    scaled = design * np.sqrt(weights)[:, None]
    degrees_of_freedom = max(run_count - design.shape[1], 1)
    residual_variance = residual_sum_of_squares / degrees_of_freedom

    try:
        covariance = residual_variance * np.linalg.inv(scaled.T @ scaled)
    except np.linalg.LinAlgError:  # pragma: no cover - guarded by the determinant test
        covariance = np.full((2, 2), np.nan)

    alpha_variance = float(covariance[0, 0])
    beta_variance = float(covariance[1, 1])
    # An exact fit leaves zero residual variance, so the coefficient covariance is zero and
    # the correlation is genuinely undefined rather than zero. It is reported as nan instead
    # of being computed as 0/0, which would otherwise warn and return nan anyway.
    if alpha_variance > 0.0 and beta_variance > 0.0:
        correlation = float(covariance[0, 1] / np.sqrt(alpha_variance * beta_variance))
    else:
        correlation = float("nan")

    pinned = tuple(
        spec.name
        for spec, value in zip(specs, shape_values, strict=True)
        if _is_at_bound(spec, float(value))
    )

    return FitDiagnostics(
        alpha_standard_error=float(np.sqrt(alpha_variance)),
        beta_standard_error=float(np.sqrt(beta_variance)),
        coefficient_correlation=correlation,
        residual_standard_deviation=float(np.sqrt(residual_variance)),
        residual_sum_of_squares=float(residual_sum_of_squares),
        training_run_count=run_count,
        shape_constant_names=tuple(spec.name for spec in specs),
        shape_constant_values=tuple(float(value) for value in shape_values),
        pinned_parameters=pinned,
    )


def _is_at_bound(spec: kinetics.ParameterSpec, value: float, tolerance: float = 1e-6) -> bool:
    """Whether a fitted constant is sitting on an edge of its search range."""
    span = spec.maximum - spec.minimum
    if spec.logarithmic:
        span = np.log10(spec.maximum) - np.log10(spec.minimum)
        return bool(
            abs(np.log10(value) - np.log10(spec.minimum)) <= tolerance * span
            or abs(np.log10(value) - np.log10(spec.maximum)) <= tolerance * span
        )
    return bool(
        abs(value - spec.minimum) <= tolerance * span
        or abs(value - spec.maximum) <= tolerance * span
    )


class MeanTitreModel:
    """Baseline: predict the mean training titre for every run, ignoring all inputs.

    The floor any real model must clear. It is a genuine competitor here rather than a
    formality -- on the leave-duration-out split it beat an earlier version of the
    mechanistic model, which is how that version was identified as broken.
    """

    def __init__(self, mean_titre: float) -> None:
        self.mean_titre = mean_titre

    @classmethod
    def fit(cls, runs: Sequence[ExperimentRun], targets: dict[str, float]) -> "MeanTitreModel":
        """Average the targets of the supplied runs."""
        if not runs:
            raise ValueError("need at least one run to compute a mean titre")
        return cls(float(np.mean([targets[run.experiment_id] for run in runs])))

    def predict(self, run: ExperimentRun) -> float:
        """Ignores ``run`` entirely -- that is what makes it the baseline."""
        del run
        return self.mean_titre

    def predict_many(self, runs: Sequence[ExperimentRun]) -> NDArray[np.float64]:
        return np.full(len(runs), self.mean_titre, dtype=np.float64)
