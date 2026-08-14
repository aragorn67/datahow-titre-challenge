# The model

Assumptions, structure, estimation, and the reasoning behind each choice.

---

## 1. What is being predicted

Final mAb titre — one scalar per experiment, measured at harvest. There is no titre
trajectory anywhere in the data: 100 training runs give 100 numbers.

The difficulty is not the regression but the horizon. All 20 test runs last 14 days, while
only 10 of the 100 training runs reach that. A purely statistical model must extrapolate in
time from ten examples; a model built on rate equations integrates to any horizon by
construction. That is the reason for taking a mechanistic route, and it drives most of what
follows.

## 2. The model

Product formation follows a Luedeking–Piret rate law with specific productivity modulated by
the culture environment:

```
qP(t) = ( α·μ_eff(t) + β ) · F(z(t))

F(z) = f_T(T) · f_G(Glc) · f_Q(Gln) · f_metab(Lac, NH₄) · f_lys(Lysed)

f_G     = Glc / (K_G + Glc)                Monod limitation,   K_G in mM
f_Q     = Gln / (K_Q + Gln)                Monod limitation,   K_Q in mM
f_metab = 1 / (1 + Lac/K_L + NH₄/K_A)      metabolic burden,   K in mM
f_lys   = 1 / (1 + Lysed/K_X)              lysate inhibition
f_T     = exp( θ_T·(T_ref − T) )           temperature
```

Titre is the integral of this against the measured viable-cell curve:

```
P = ∫ qP(t)·Xv(t) dt
```

Every fitted parameter is a rate constant, yield coefficient or half-saturation constant
carrying physical units. `K_L = 40 mM` is a claim that can be checked against literature; a
regression slope is not.

### The identity that makes it computable

From Richelle 2026 Eq. 14, `μ_eff = μ + μ_d + μ_l`. Working this through both papers'
population structures:

```
Frontiers 2022 (sequential):  dXv/dt = (μ_eff − μ_d)·Xv
  ⟹ μ_eff·Xv = dXv/dt + dXd/dt + dXl/dt

CSBJ 2026 (parallel):         dXv/dt = (μ_eff − μ_d − μ_l)·Xv
  ⟹ μ_eff·Xv = dXv/dt + dXd/dt + dXl/dt
```

Both give the same result, so it does not depend on which lysis structure is chosen:

```
∫ μ_eff·Xv dt  =  Δ(Xv + Xd + Xl)  =  total cells ever synthesised
```

This is an **endpoint** quantity. No numerical differentiation of the viable-cell curve is
required anywhere, which matters because differencing a 2%-noise signal over daily samples
produces roughly 20% error on the derivative.

`Xd` is never measured. Under the sequential form it is nonetheless recoverable from the
measured lysate, since `dXl/dt = kl·Xd`:

```
Xd(t) = Xl'(t) / kl
C(t)  = Xv(t) + Xl(t) + Xl'(t)/kl        cells made up to time t
```

### Discretisation

Per observation interval `j`:

```
ΔCⱼ  = C(t_{j+1}) − C(tⱼ)                        growth contribution
γXⱼ  = (Xv(tⱼ) + Xv(t_{j+1}))/2 · Δtⱼ            trapezoidal cell-days
Fⱼ   = (F(zⱼ) + F(z_{j+1}))/2                    factor across the interval
```

with `Σⱼ γXⱼ = γX` and `Σⱼ ΔCⱼ = Δ(Xv+Xd+Xl)` exactly.

### Three nested hypotheses

The placement of `F` is a testable claim, not an assertion:

| | model | integrated form |
|---|---|---|
| **M0** | `qP = α·μ_net + β` | `P = α·ΔXv + β·γX` — documented failure, kept as a benchmark |
| **M1** | `qP = α·μ_eff + β` | `P = α·Δ(Xv+Xd+Xl) + β·γX` |
| **M2** | `qP = α·μ_eff + β·F(z)` | `P = α·Δ(Xv+Xd+Xl) + β·Σⱼ Fⱼ·γXⱼ` |
| **M3** | `qP = (α·μ_eff + β)·F(z)` | `P = Σⱼ Fⱼ·[α·ΔCⱼ + β·γXⱼ]` |

**M3 − M2 has a clean mechanistic interpretation.** M2 says the environment changes the
*rate* of non-growth-associated production. M3 additionally says it changes the *yield of
product per cell synthesised*. Fewer cells made versus less product per cell made — distinct
claims, and the comparison tests exactly that.

All three are linear in `(α, β)` given `F`'s shape constants, so one estimation procedure
serves all of them and the comparison is genuinely nested.

---

## 3. Assumptions

Every assumption, and whether it was forced by the data or chosen.

### Forced by the data

1. **No bioreactor volume is supplied.** Dilution terms `(F/V)` cannot be identified and are
   omitted; feeds are treated as additive concentration source terms, not volumetric flows.
   Both reference papers carry `F/V` explicitly. We cannot.

2. **`X:Lysed` is a cumulative cell density in the same units as VCD.** Inferred, not
   documented — the challenge ships no data dictionary. Evidence: exactly 0.0 at t=0 in all
   120 runs; `Xl/Xv` unbounded so not a fraction; over the post-peak decline it accounts for
   ~3.4% of viable cells lost, the rest sitting in the unmeasured dead pool.

3. **The dead-cell pool is never measured.** Recovered indirectly via the sequential lysis
   form. Only this recoverability depends on sequential-versus-parallel; the `μ_eff` identity
   does not.

4. **Titre is observed once per run, at harvest.** 100 scalars is the entire budget for
   identifying the rate law.

5. **Temperature spans ~3 °C and pH ~1.5 units.** Only one-parameter monotonic forms are
   identifiable over such a window. A Gaussian optimum would place `T_opt` outside the
   observed range and be an awkward parameterisation of a monotonic trend.

### Chosen, with reasons

6. **`kl` is constant** across runs and across time.

7. **No product degradation term.** Unidentifiable against one titre measurement per run.
   Consequence: `f_lys` cannot distinguish *reduced synthesis* from *proteolytic degradation
   of already-made product* — lysed cells release proteases, and both mechanisms look
   identical at a single endpoint. It is fitted and reported as ambiguous rather than named
   "toxicity".

8. **`Xl(t)` is monotone.** A physical constraint on a cumulative pool, not a smoothing
   convenience. Used to fit a smooth monotone curve so `Xd = Xl'/kl` is analytic — which also
   removes the arbitrary slope window whose choice previously swung held-out RMSE between
   1430 and 1648.

9. **`F` is constant within a one-day interval**, taken as the trapezoidal average of its
   endpoint values. Far weaker than assuming it constant over the run. Weakest on the day
   feeding begins, when glucose can move from near zero to several mM and Monod is most
   curved.

10. **Controlled variables are inputs, not states.** `W:temp` and `W:pH` are exact step
    functions of the `Z:` scalars, verified to machine precision across all 1290 rows;
    stirring and DO are per-run constants. The `N_rpm → kLa → DO → qO₂` chain is deliberately
    not modelled: DO is a setpoint handed to us, so the controller has already closed that
    loop, and there is no oxygen measurement to constrain a `kLa`.

11. **Inference is conditioned on observed trajectories.** The `/predict` payload supplies
    them, so growth is measured rather than simulated. **Limitation: this model cannot
    predict an experiment that has not been run.** Forward simulation from design parameters
    alone is a different and harder model, and out of scope for the brief.

12. **Measurement noise is not weighted.** Handled implicitly through the monotone fit rather
    than modelled.

---

## 4. Why each structural choice was made

**Why mechanistic at all.** The duration shift. Ten of 100 training runs reach the test
horizon; 14 of 20 test runs exceed the maximum biomaterial exposure of *any* ≤10-day
training run. Integration handles that by construction; statistical extrapolation does not.

**Why effective rather than net growth.** Luedeking–Piret's `α` couples production to
*biosynthesis*. Using net growth makes decline-phase intervals contribute negatively — the
model would say secreted antibody disappears as cells die. It also reproduces a measured
failure: the naive version fitted `α = −9.9` and scored RMSE 2809, against 1507 once
cells-synthesised was used.

**Why the sequential lysis form.** Both papers' structures give the same `μ_eff` identity,
but only the sequential form makes the unmeasured dead pool recoverable from the measured
lysate. In *both* papers `Xl` is a latent variable — Frontiers states plainly it "acts as a
degree of freedom since no experimental measurements were available", which is why `kl` was
their worst-identified parameter. Here it is measured. That is an advantage over the
published models, and the sequential form is what converts it into one.

**Why saturating kinetic forms rather than linear terms.** Two reasons. Interpretability:
half-saturation constants can be checked against literature. And extrapolation safety: test
runs reach `X:Lysed` of 1.02 against a training maximum of 0.53, and glucose 56.8 against
44.0. Monod and inhibition terms are monotone and bounded, so outside the fitted range they
flatten. A polynomial or linear coefficient would diverge exactly where the test set lives.

**Why cell-weighted averaging for screening features.** Derived, not chosen. From
`q̄P = ∫qP·Xv dt / ∫Xv dt`, if qP depends on `z(t)` then to first order `q̄P` depends on the
**cell-weighted** mean of `z`. Arithmetic means or endpoint values would be the wrong summary
under the mechanism.

**Why a fixed absolute phase split.** Splitting exposure at each run's own midpoint makes
"late" mean days 3.5–7 in a short run and 7–14 in a long one — different physiological
regimes, confounded with duration in precisely the direction that matters. A fixed cut leaves
short runs with little or no late window, which is honest: they never reached that regime,
and that *is* the shift being modelled.

**Why inhibition is two factors, not one or three.** Lysed material is a physically distinct
insult — released proteases, DNA, host-cell protein — while lactate and ammonia act together
through energetics. Their pH-mediated route is closed here because pH is controlled and held
at setpoint, so the combined term reads as metabolic burden rather than acidification.
Caveat: NH₄ and lysate exposure correlate at 0.84, so the two factors may not separate
statistically. They are fitted and their parameter correlation reported — if inseparable,
that is a result, not something to hide behind a lumped term.

---

## 5. Estimation

**Variable projection.** Given the shape constants, all three models are linear in `(α, β)`,
so those are solved in closed form inside a grid search over the shape constants, with
refinement. No starting guess, no convergence criterion, no local minimum to miss.

This is a materially stronger position than the multi-start Nelder–Mead the reference papers
require, because their systems are fully nonlinear in 5–20 parameters. The linearity in
`(α, β)` is a property of Luedeking–Piret, not an empirical shortcut — **the least squares
sits inside the optimiser, not in the model.**

A grid rather than a smooth optimiser also suits an objective made piecewise by clipping.

**Parameter count.** α, β, kl, plus roughly two constants per mechanism: seven to nine
parameters against 100 scalars, with known collinearity. That is pushing the data, and
cross-validated comparison is the arbiter rather than a formality.

**Uncertainty.** Bootstrap over runs for prediction intervals; profile likelihood over each
saturation constant; the full parameter correlation matrix, as the Richelle papers report.
Note `α` and `kl` are partially confounded, since the dead-cell part of the growth term
scales as `α/kl` — separation comes only from `ΔXv` and `ΔXl`, which do not involve `kl`.

**Post-selection caveat.** Coefficients and standard errors computed after selecting
mechanisms on the same data are optimistically biased. Point estimates are usable; the
standard errors are not valid as stated.

---

## 6. Evaluation

**Metrics.** Training titres span 283–4823, a factor of seventeen, and the test set sits in
the upper tail. RMSE is in target units and dominated by high-titre runs; MAPE weights every
run equally and is dominated by low-titre runs. Both are reported: a model that improves one
while worsening the other has not improved, it has changed *which runs it is good at*. R² is
unstable on ten upper-tail runs and is read alongside the absolute errors, never instead.

**The split that matters: leave-duration-out.** Train on runs of ≤10 days, hold out the
14-day runs. This reproduces the real shift. A random split would test the model on the same
durations it was trained on and report a number that says nothing about the actual task. The
held-out group is ten runs; that limitation is real and is reported rather than avoided.

**Supporting split: k-fold within the training runs.** Answers a different question — how the
model does on runs *like* those it trained on. That is interpolation; the task is
extrapolation. K-fold will look better, and **the gap between the two is itself the result**,
separating general model weakness from the cost of extrapolating in time.

Every parameter is re-estimated inside each fold, `kl` included. Reusing a `kl` fitted on all
data would leak the held-out runs into every fold. It feels structural, which is exactly why
it is an easy mistake; it is fitted, so it goes inside.

No grouped splitter is needed: each experiment contributes exactly one sample.

**The fold parameters are a diagnostic, not the deliverable.** Five folds give five
independent estimates; how far they move says whether the data identify them at all. The
shipped model is fitted on all training runs; the fold parameters characterise it.

**Selection discipline.** Mechanisms are selected on cross-validated error *inside* the 90
training runs. The held-out ten are touched exactly once, at the end. An earlier iteration
chose a temperature term partly by looking at held-out error — with one candidate that is
borderline, with a list it is straightforward selection bias.

**Baseline.** `MeanTitreModel` predicts the training mean, ignoring all inputs. Not a
formality: on the leave-duration-out split it **beat** an earlier version of the mechanistic
model, which is how that version was identified as broken.

---

## 7. Known weaknesses

- Seven to nine parameters against 100 scalars, with correlated candidates.
- `f_lys` cannot distinguish reduced synthesis from product degradation.
- `α` and `kl` partially confounded.
- The within-interval constant-`F` assumption is weakest on feed-start days.
- Screening works on `q̄P`, which averages over both production routes, so it identifies
  *which* variables matter but not *where* they act; the M2/M3 comparison answers that.
- The model cannot predict an unrun experiment.

---

## 8. Related documents

- [`analysis.md`](analysis.md) — the high-level account of Part 1: problem, equations,
  references, final model
- [`data-loading.md`](data-loading.md) — the data layer in detail
