# Modelling

The complete account of Part 1: the problem, the equations, the assumptions, how the model was
estimated and evaluated, and what the data do and do not determine.

The data layer is documented separately in [`data-loading.md`](data-loading.md).

---

## 1. The problem

Predict the **final mAb titre** of a simulated fed-batch bioprocess: one scalar per experiment,
from time-series bioprocess data.

The brief specifies only "a simulated bioprocess for monoclonal antibody (mAb) production".
**No cell line is stated, so none is assumed here.** That constraint is load-bearing rather than
pedantic: the reference papers in §3 are CHO studies, and borrowing their organism would licence
plausibility arguments — "that constant is too slow for CHO" — that these data cannot support.
Every claim about a fitted constant below is grounded in a measurement from these runs, or not
made.

### What the data are

100 training experiments and 20 test experiments, one row per (experiment, day) on an exact
one-day grid, with process design scalars (`Z:`), control profiles (`W:`), measured observations
(`X:`) and the target (`Y:`).

Four properties of the dataset shape everything that follows. None of these is an assumption —
they are simply true, and they are listed here so the assumptions in §4 can be read against
them:

- **Titre is observed once per run, at harvest.** There is no product trajectory, so no
  time-resolved information about `qP` beyond what the rate law's form imposes.
- **Dead cells are never measured.** Viable and lysed cells are; the dead pool is not.
- **No bioreactor volume is supplied.**
- **There is no data dictionary**, so units for `X:Lysed` have to be inferred.

### The problem that determines the design

**Every test run lasts 14 days. Only 10 of the 100 training runs reach that horizon.**

| | |
|---|---|
| training durations | 30 at 7 d, 20 each at 8/9/10 d, **10 at 14 d** |
| test durations | 20 runs, **all 14 d** |
| titre range | 283–4823, median 1148; the 14-day subset averages 2380 |
| biomaterial exposure `γX` | training ≤10 d tops out at **242.7**; 14-day runs reach **548.7** |

A model fitted on run-level aggregates must extrapolate in time from ten examples. A model built
on rate equations integrates to any horizon by construction. This is the entire argument for a
mechanistic scaffold, and §7 tests it against data-driven alternatives rather than asserting it.

---

## 2. Equations

An unstructured kinetic model. The cell population is split into viable, dead and lysed cells,
with a catch-all "biomaterial" variable representing accumulated by-products.

### States

```
dXv/dt     = (mu_eff - mu_d) * Xv                    viable cells       [measured]
dXd/dt     =  mu_d * Xv - kl * Xd                    dead cells         [NOT measured]
dXl/dt     =  kl * Xd                                lysed cells        [measured]
dgammaX/dt =  Xv                                     biomaterial        [latent]

dGlc/dt    = -v_Glc * Xv + F_Glc(t)                  glucose            [measured]
dGln/dt    = -v_Gln * Xv + F_Gln(t)                  glutamine          [measured]
dLac/dt    =  v_Lac * Xv                             lactate            [measured]
dNH4/dt    =  v_NH4 * Xv                             ammonia            [measured]

dP/dt      =  q_P * Xv                               product   [measured once, at harvest]
```

### Rates

```
mu_d  = kd + kTd_gam*gammaX + kTd_lac*Lac + kTd_nh4*NH4 + k_shear*Stir
q_P   = (alpha*mu_eff + beta) * F(z)
mu_eff = mu + mu_d + mu_l
```

`F(z)` is the environmental modulation of specific productivity. **Which physiological variables
compose it is not asserted here** — it is selected from the data, in §6. The literature form
carried `f_T(T)·g_pH(pH)`; on this dataset neither temperature nor pH earned a place.

### Not states

Temperature, pH, stirring and dissolved oxygen are **inputs**, not dynamic variables. Temperature
and pH are exact step functions of the `Z:` scalars; stirring and dissolved oxygen are per-run
constants:

```
W:temp(t)    = tempStart    if t <  tempShift  else tempEnd
W:pH(t)      = phStart      if t <  phShift    else phEnd
W:FeedGlc(t) = FeedRateGlc  if FeedStart <= t < FeedEnd  else 0
W:FeedGln(t) = FeedRateGln  if FeedStart <= t < FeedEnd  else 0
```

They enter the rate laws as modulators. Giving them differential equations would mean integrating
quantities already known exactly.

### The identity that makes this computable

This is the central move of the whole model, so it is derived rather than quoted.

The task is to predict titre, not to simulate a bioreactor — and the `/predict` payload supplies
the **full observed trajectories**. So rather than simulating the culture forward, product
formation is integrated *along measured data*. That requires `∫ mu_eff·Xv dt`, and `mu_eff` is not
directly observable. Working the identity `mu_eff = mu + mu_d + mu_l` through either paper's
population structure gives the same result:

```
Frontiers 2022 (sequential):  dXv/dt = (mu_eff - mu_d)*Xv
  => mu_eff*Xv = dXv/dt + mu_d*Xv = dXv/dt + dXd/dt + kl*Xd
               = dXv/dt + dXd/dt + dXl/dt

CSBJ 2026 (parallel):         dXv/dt = (mu_eff - mu_d - mu_l)*Xv
  => mu_eff*Xv = dXv/dt + mu_d*Xv + mu_l*Xv = dXv/dt + dXd/dt + dXl/dt
```

Both yield:

```
INT mu_eff*Xv dt  =  Delta(Xv + Xd + Xl)  =  total cells ever synthesised
```

**That is an endpoint quantity, so no numerical differentiation of the viable-cell curve happens
anywhere.** It matters: differencing a 2%-noise signal over daily samples produces roughly 20%
error on the derivative.

It also explains why net growth is the wrong quantity. Luedeking–Piret's `alpha` couples
production to biosynthesis, not to net population change. Using net `mu` makes decline-phase
intervals contribute *negatively* — the model would say secreted antibody disappears as cells
die. That is not a hypothetical: the naive `alpha*dXv` version fits `alpha = −9.9`, and it is kept
as variant M0 in §5 so the difference is demonstrated rather than claimed.

---

## 3. References, and what was taken from each

Two papers, both from the same group, provide the model structure.

**[1] Richelle et al. (2022).** *Model-based intensification of CHO cell cultures: one-step
strategy from fed-batch to perfusion.* Front. Bioeng. Biotechnol. **10**:948905.
doi:10.3389/fbioe.2022.948905

**[2] Richelle et al. (2026).** *A Hybrid Modeling Framework for Predictive Digital Twins of CHO
Cell Culture.* Comput. Struct. Biotechnol. J. **35**(1):Article 0078. doi:10.34133/csbj.0078

**The three-population split and the biomaterial variable — from [1].** Viable, dead and lysed
cells, with cell death driven by a catch-all by-product variable rather than an explicit list of
inhibitors. This keeps the parameter count low: `gammaX` obeys `dgammaX/dt = Xv` and so costs no
parameters of its own.

**`mu_eff = mu + mu_d + mu_l` — from [2], Eq. 14.** The growth rate obtained from the slope of the
viable-cell curve is a *net* rate. Product formation scales with biomass *synthesis*, so the
Luedeking–Piret term must multiply `mu_eff`, not net `mu`. Over a 14-day run with substantial
death these differ considerably.

**The sequential lysis form — from [1], not [2].** The two papers disagree:

- [1] treats lysed cells as a degradation product of the **dead** pool:
  `dXd/dt = mu_d*Xv − kl*Xd`, `dXl/dt = kl*Xd`
- [2] treats dead and lysed cells as **parallel** products of the viable pool:
  `dXd/dt = mu_d*Xv`, `dXl/dt = mu_l*Xv`

[1] was chosen, for a reason specific to this dataset. **Lysed cells are measured; dead cells
never are.** Under the parallel form, lysis bypasses the dead pool entirely, so the lysed-cell
trajectory says nothing about `Xd` and the dead pool stays invisible. Under the sequential form,
lysed cells come *from* dead cells, so

```
Xd(t) = (1/kl) * dXl/dt
```

and the dead pool becomes readable from measured data, up to the single constant `kl`.

In **both** papers the lysed-cell population is latent. [1] states plainly that it "acts as a
degree of freedom in the model since no experimental measurements were available", which is why
`kl` was by far their worst-identified parameter. Here it is measured, and the sequential form is
what converts that measurement into information about `Xd`.

**A claim not to make from this.** Better *identified* does not mean meaningful as a rate. §8
shows that `kl`'s fitted time constant is 202 days against a 7–14 day observation window, so
lysis is never observed on its own timescale and `kl` functions as a scale factor sizing the
inferred dead pool. An earlier draft of this document claimed `kl` was "a genuine estimate, better
identified than the published models"; the first half stands and the conclusion does not.

**What was not taken.** The chain from agitation through `kLa` to oxygen uptake is not modelled.
Dissolved oxygen is a controlled setpoint handed to us — the controller has already closed that
loop. Building the chain would mean predicting a variable already observed exactly, at the cost of
parameters that no oxygen, OUR or off-gas data could constrain.

---

## 4. Assumptions

Eight assumptions. Each is a choice that could have been made otherwise and on which the model's
validity depends. Facts about the dataset are in §1; how the work was done is in §6 and §8 —
neither is an assumption, and conflating the three is how this list previously reached
twenty-two entries.

### 4.1 Feeds are additive concentration source terms, with no dilution

- **Assumed:** `W:FeedGlc` and `W:FeedGln` add directly to `X:Glc` and `X:Gln`, in the same units
  per day. They are not volumetric flows.
- **Alternative:** carry dilution explicitly, `dC/dt = … + (F/V)·(C_in − C)`, as both reference
  papers do.
- **Why not:** no volume column exists, so `F/V` cannot be identified at all.
- **Consequence if wrong:** every concentration balance is biased, most in runs with the largest
  feed volumes. Since metabolite concentrations feed the two surviving Monod terms, this
  propagates directly into `F(z)`. This is the single most consequential assumption in the model.

### 4.2 `X:Lysed` is a cumulative cell density in the same units as `X:VCD`

- **Assumed:** a cumulative pool, small only because lysis is slow.
- **Alternative:** a fraction, a rate, or different units entirely.
- **Evidence:** exactly 0.0 at `t=0` in all 120 runs — a pool starting empty, not a concentration
  with an initial condition. Not a fraction: `Xl/Xv` is unbounded and one test row exceeds 1.
- **Consequence if wrong:** `Xd = Xl'/kl` is in the wrong units, so `alpha` absorbs a unit
  conversion and stops being interpretable as titre per cell synthesised. Predictions survive;
  the mechanistic reading does not.

### 4.3 The dead pool follows sequential lysis, `Xd = Xl'/kl`

- **Assumed:** lysed cells derive from dead cells, per reference [1].
- **Alternative:** the parallel form of reference [2], which was tried **first**.
- **Why:** under the parallel form the measured lysate carries no information about `Xd`, and
  `Xd` is exactly the quantity needed for `mu_eff`. This was the project's first substantive
  failure and it is documented as one.
- **Consequence if wrong:** the growth regressor is wrong. It is used everywhere — M1, M2 and M3
  all depend on it — so there is no fallback.

### 4.4 Controlled variables are inputs, not states

- **Assumed:** temperature, pH, stirring and dissolved oxygen are exogenous modulators.
- **Alternative:** give them differential equations and predict them.
- **Why:** they reconstruct from the `Z:` scalars to machine precision on all 1290 rows, so
  modelling them would mean integrating quantities already known exactly.
- **Consequence if wrong:** only if the controller failed in some run, which the exact
  reconstruction rules out.

### 4.5 Product formation is integrated along measured trajectories

- **Assumed:** condition on the observed `X:` and `W:` series rather than simulating forward.
- **Alternative:** the full forward model — predict trajectories from `Z:` and initial
  conditions, then integrate.
- **Why:** the `/predict` payload supplies the trajectories. The growth and metabolite equations
  exist to *predict* what we are handed; what survives the substitution is exactly the part no
  measurement pins down, specific productivity.
- **Consequence:** the model predicts titre for a run whose trajectories are known. Forward
  simulation from design parameters alone is a different and harder model.

### 4.6 `Xl(t)` is a smooth monotone cumulative pool, observed with additive noise clipped at zero

- **Assumed:** monotonicity is a physical constraint on a cumulative pool, not a smoothing
  convenience; and the noise on `X:Lysed` is **additive with a fixed absolute size** — sd of
  order 0.009, against a series whose final value has median 0.056 — rather than proportional,
  and clipped at zero. The clipping is visible: every day-0 value is exactly 0.0, as is roughly
  half of days 1–2, and 32% of all rows.
- **Alternative:** finite-difference the measured series directly.
- **Why not:** raw levels make the interval growth contribution negative in roughly 21% of
  intervals, purely from noise. And the clipping biases observations *upward early and not late*,
  so forcing the curve through `Xl(0) = 0` would make it tilt to absorb that bias and corrupt the
  derivative — which is the quantity actually wanted.
- **Consequence if wrong:** if lysate genuinely plateaus and reverses, the monotone fit cannot
  represent it. A handful of runs already get a fitted harvest slope of exactly zero, hence
  `Xd(T) = 0` — implausible physically, but the honest reading of a curve flat at the end relative
  to the noise floor.

### 4.7 `F(z)` is roughly constant within a one-day interval

- **Assumed:** the environmental factor can be represented by its endpoint values across each
  interval.
- **Alternative:** a finer quadrature, which the daily sampling grid does not support.
- **Why it is weak:** metabolites move sharply within a day once feeding starts, so this is
  weakest on feed-start days.
- **Note:** this is a far weaker assumption than holding `F` constant over a whole run, and §5
  discretises it correctly rather than treating `F(mean z)` as `mean F(z)`.

### 4.8 A mechanism helping on 7–10 day runs also helps at 14 days

- **Assumed:** transferability of the selection across the duration shift.
- **Alternative:** select on the 14-day runs specifically.
- **Why not:** there are ten of them, and bootstrapping ten runs gives an RMSE interval wide
  enough to swallow every difference between variants. Selecting on that instrument is selecting
  on noise.
- **Consequence:** **this is untestable at this sample size**, and it is the assumption carrying
  the most risk. Everything in §6 rests on it. It is why the held-out ten are reported separately
  and never used to choose.

---

## 5. The model: four nested hypotheses

Where `F` acts is a testable claim, not an assertion, so four variants are fitted and compared.

| | model | integrated form |
|---|---|---|
| **M0** | `qP = α·μ_net + β` | `P = α·ΔXv + β·γX` — the documented failure, kept as a benchmark |
| **M1** | `qP = α·μ_eff + β` | `P = α·Δ(Xv+Xd+Xl) + β·γX` |
| **M2** | `qP = α·μ_eff + β·F(z)` | `P = α·Δ(Xv+Xd+Xl) + β·Σⱼ (FγX)ⱼ` |
| **M3** | `qP = (α·μ_eff + β)·F(z)` | `P = α·Σⱼ F̄ⱼ·ΔCⱼ + β·Σⱼ (FγX)ⱼ` |

Because `F ≡ 1` when no mechanisms are in force, M2 and M3 collapse onto M1 **exactly**. The
nesting is a property of the arithmetic rather than something arranged, and it is asserted in the
tests.

**All four are linear in `(α, β)`** once the shape constants are fixed, which is a property of
Luedeking–Piret rather than an empirical shortcut. One estimation procedure therefore serves all
four, and the comparison is genuinely nested.

### Discretisation, and two quadratures that are not the same

Per observation interval `j`:

```
ΔCⱼ    = C(t_{j+1}) − C(tⱼ)                        growth contribution
γXⱼ    = (Xv(tⱼ) + Xv(t_{j+1}))/2 · Δtⱼ            trapezoidal cell-days
F̄ⱼ     = (F(zⱼ) + F(z_{j+1}))/2                    factor across the interval
(FγX)ⱼ = (Fⱼ·Xvⱼ + F_{j+1}·Xv_{j+1})/2 · Δtⱼ       ∫F·Xv dt across the interval
```

with `Σⱼ γXⱼ = γX` and `Σⱼ ΔCⱼ = Δ(Xv+Xd+Xl)` exactly, where
`C(t) = Xv(t) + Xl(t) + Xl'(t)/kl`.

**The last two lines are different quantities.** `ΔCⱼ` is an *endpoint difference*, so the
interval average `F̄ⱼ` is the only weight the data support. The non-growth term is an *integral
against the cell curve*, so its integrand is the product `F·Xv` and the trapezoid must be taken of
that product — a mean of products, not a product of means. The two differ by `ΔF·ΔXv/4` per
interval.

An earlier version used `F̄ⱼ · γXⱼ`. The magnitude was small (median −0.07%, maximum +7%) but the
sign was systematic in exactly the regime the model is about: `F` falling as nutrients deplete
while `Xv` rises.

**A worse defect was hiding behind it.** The vectorised fitting search precomputed each
mechanism's factor, averaged it over the interval, and multiplied the averages; the prediction
path multiplied the mechanisms pointwise and *then* averaged. Those agree for one mechanism and
disagree for two or more — **by up to 14% per run here** — so `α` and `β` were being fitted
against columns that no prediction ever used. Both quadratures now live once in `features.py`, the
search re-derives them from the same pointwise factor, and a test asserts the two paths agree for
one, two and three mechanisms. The bug existed because nothing asserted that property.

With `F ≡ 1`, `(FγX)ⱼ` reduces exactly to `γXⱼ`, which is what keeps the nesting exact.

### The environmental factor

```
F(z) = f_T(T) · f_G(Glc) · f_Q(Gln) · f_metab(Lac, NH4) · f_lys(Lysed) · f_pH(pH)

f_G     = Glc / (K_G + Glc)                 Monod limitation
f_Q     = Gln / (K_Q + Gln)                 Monod limitation
f_metab = 1 / (1 + Lac/K_L + NH4/K_A)       combined metabolic burden
f_lys   = 1 / (1 + Lysed/K_X)               debris/protease toxicity
f_T     = exp(theta_T · (T_ref − T))        monotone, not an optimum
f_pH    = exp(theta_p · (pH_ref − pH))      monotone, not an optimum
```

Only factors whose mechanisms survive selection are included; §6 gives the two that did.

**Why saturating forms rather than linear terms.** Two reasons, both about behaviour outside the
fitted range. A half-saturation constant is a physical quantity checkable against literature,
where a regression slope is not. And the test runs reach `X:Lysed` of 1.02 against a training
maximum of 0.53, and glucose 56.8 against 44.0 — the Monod and inhibition forms are monotone and
bounded in `(0, 1]`, so beyond the fitted range they flatten rather than diverge. A polynomial
would run away exactly where the test set lives.

**Inhibition is two factors, not one or three.** Lysed material is a physically distinct insult
(released proteases, DNA, host-cell protein), while lactate and ammonia act together through
energetics — so their burdens share a denominator rather than multiplying as independent routes.

**Temperature and pH are monotone one-parameter forms, deliberately.** Temperature spans ~3 °C
and pH ~1.5 units. A Gaussian optimum needs two parameters and `T_opt` would very likely land
outside the observed range; reporting a `T_opt` of 41 °C fitted from 35–38 °C data would be
indefensible. The exponential is unbounded, which is tolerable only because both variables are
controlled and the test set stays inside the training range.

---

## 6. Estimation and selection

### Variable projection

Given the shape constants, every variant is linear in `(α, β)`, so those are solved in closed form
**inside** a grid search over the shape constants, with successive refinement. No starting guess,
no convergence criterion, no local minimum to miss. **The least squares sits inside the optimiser,
not in the model.**

This is a materially stronger position than the multi-start Nelder–Mead the reference papers
require, because their systems are fully nonlinear in 5–20 parameters. A grid also suits an
objective made piecewise by the non-negativity in the lysate fit.

Convergence is checked rather than assumed: against a much denser sweep (21 points, 6
refinements) the shipped settings (5 points, 8 refinements) agree on the residual sum of squares
to 0.000%.

### The lysate fit

`Xl(t)` is fitted by **non-negative least squares on an integrated B-spline basis**:

```
Xl'(t) = Σₖ cₖ Bₖ(t),  cₖ ≥ 0    ⟹    Xl(t) = c₀ + Σₖ cₖ ∫₀ᵗ Bₖ
```

B-splines are non-negative everywhere, so non-negative coefficients make the derivative
non-negative **everywhere**, not merely at the knots. The alternatives each fail one requirement:
PCHIP is monotone but interpolates, so noise passes straight into `Xl'`; isotonic regression is
monotone but piecewise constant, so its derivative is a train of spikes; an unconstrained
smoothing spline smooths but can turn downwards.

Settings were chosen by measurement, not by eye: leave-one-point-out cross-validation pooled over
all 120 runs, sweeping degree 1–3 against both a fixed knot count and a fixed knot spacing, put a
degree-2 basis with one knot every 5 days lowest (LOO RMS 0.01257). Residual RMS is 0.0068 against
a measured noise sd of about 0.0087 — the fit is not chasing noise. Knot *spacing* rather than
*count* gives a 7-day and a 14-day run the same flexibility per unit time.

This replaces an arbitrary 4-point slope window at harvest whose choice swung held-out RMSE
between 1430 and 1648. A known sensitivity disappears rather than being tuned away.

### Stage 1 — variable screening, as a diagnostic

Screening reports **which measurements carry information** about specific productivity. It does
**not** choose the model; an earlier version let it, and that was a real error (see below).

The target is `q̄P = Y_titer / γX`. Dividing by biomaterial leaves an intensive quantity whose
variation a productivity mechanism has to explain. **The ratio is a screening device only** — the
models fit directly against measured titre, so the artefact that makes anything tracking `γX`
mechanically anti-correlated with `q̄P` never reaches them.

With 100 runs, 25 candidate features and known collinearity, any single ranking on a single fit is
close to arbitrary. So every method runs **inside each fold** and a variable earns its place by
being chosen **repeatedly and by more than one method**. Four methods are used because they fail
differently: correlation sees only marginal association; ElasticNetCV picks one of a correlated
group almost at random; PLS spreads weight across a correlated group instead of choosing within
it; permutation importance is measured on held-out rows and so is the only one answering "does
this help prediction?". Agreement across methods that fail in different directions is evidence.

Features are cell-day-weighted, which is *derived* rather than chosen: if `qP` depends on `z(t)`
then to first order `q̄P` depends on `⟨z⟩_X = ∫z·Xv dt / ∫Xv dt`.

Three screening choices affect only this diagnostic stage and never touch the rate law. The
phase split is at a **fixed absolute day (7)**, not each run's own midpoint — a relative midpoint
would make "late" mean days 3.5–7 in a short run and 7–14 in a long one, different regimes
confounded with duration. Specific growth rate uses the **interval-mean** VCD as denominator,
better conditioned than the left endpoint. The feed window is the one **realised before harvest**,
since a window ending after harvest was never fully applied. Undefined features are reported as
`nan` and **never imputed**: their missingness is perfectly collinear with duration, so filling
them would inject a duration signal disguised as a metabolite effect.

### Stage 2 — mechanism selection, by prediction and by identifiability

**Why selection is not done on the screen.** Whether a variable helps depends on whether the
mechanism assigned to it can represent the **sign** of its association, and a variable screen
cannot see that. Lactate exposure correlates **+0.767** with specific productivity and tops the
screen; its mechanism is an inhibition term that can only bend downwards, so no parameter value
expresses a positive association and the fit switches it off. Glucose correlates +0.719, nearly as
strongly, and is dropped by the screen as collinear with lactate — yet it maps to a Monod term
that bends *upwards* and does the work. **The screen discarded the mechanism that worked and kept
the one that could not.**

The search is forward stepwise on cross-validated error: at most `n(n+1)/2` cross-validations
rather than the `2ⁿ−1` of an exhaustive search, which matters both for cost and for how hard the
cross-validation is being looked at.

**Two admission tests, not one.** Cross-validated error alone is not sufficient, and on this data
it demonstrably is not. After the quadrature correction in §5, `metabolic_burden` cleared the 1%
margin — 174.1 → 168.9, a 3.0% improvement — and then failed to transfer in every other
direction: 287.2 against 286.6 over all 100 runs, and worse on the held-out ten in **98%** of
paired bootstrap resamples. That is what a forward search over 21 candidate sets on one
cross-validation should be expected to produce occasionally.

The fold fits say why, **without consulting held-out data**: across the ten folds its lactate
constant `K_L` is fitted between 1113 and 10 000 mM against a measured lactate range of 0–8 mM,
resting on the search bound in three of them. A half-saturation constant two to three orders of
magnitude above the concentrations that exist means the factor never leaves the neighbourhood of
one — **every fold switched the mechanism off.** Whatever earned the 3%, it was not inhibition by
these metabolites.

So a mechanism must earn its margin **and** have constants the folds determine:

- **No constant may rest on a search bound in more than 20% of folds.** This is the decisive test
  and it is near-definitional: a constant the grid stopped is not an estimate. It rejects both
  mechanisms that are rejected.
- **A log-scaled constant may not move more than 10× across folds** (25% of its search range for
  a linear-scaled one). This is softer and its threshold *is* a judgement, so the sensitivity is
  recorded rather than glossed: `K_G`, which is **accepted**, moves 8.28×, so a threshold of 3
  would also have rejected glucose limitation and left glutamine alone.

It is deliberately a rule rather than a judgement made once about one mechanism, and it costs
nothing — it reuses the fits already performed for scoring. `lysate_inhibition` is rejected by the
same rule (`K_X` moving 52×, on a bound in 30% of folds). Rejected sets are reported with their
evidence so they cannot be mistaken for candidates that simply scored badly.

### One margin for every selection step

Mechanisms, model variant and ridge strength are all held to the same **1%** relative improvement,
through one shared rule. An earlier version applied 1% to mechanisms while accepting a ridge
penalty on a 0.5% gain — one instrument read to two different precisions.

The rule also states what to prefer when the data cannot decide: the simpler or default option
keeps its place unless a challenger clears the margin. A plain argmin has no tie-break and
systematically favours the more elaborate candidate, because a more flexible model has more ways
to score marginally better by luck.

**Variant selection** (CV over the 90 selection runs): M0 365.8, M1 293.9, **M2 174.1**, M3 303.1.

**M2 beats M3, and there is a mechanistic reason rather than only an empirical one.** The growth
term is built from `μ_eff`, derived from the *measured* VCD curve — a curve nutrient limitation
has already depressed. M3 applies `F` to that term a second time and so double-counts the same
physical effect.

### Ridge regularisation: available, and honestly unused

`α` and `β` are anti-correlated, so the training objective is nearly flat along a ridge. Held-out
performance is **not** flat along that same ridge, which makes ridge position a genuine source of
extrapolation error rather than a cosmetic diagnostic.

A penalty was therefore added, acting on **standardised** columns — `α` multiplies cells
synthesised and `β` multiplies cell-days, quantities an order of magnitude apart in different
units, so an unstandardised penalty would shrink them unequally by an accident of scale. Zero is
in the grid, so the fit can decline shrinkage.

**It failed, and that is reported as a result.** The best candidate scores CV RMSE 168.909 against
168.910 for no penalty — a difference of 0.0004%, so **zero is selected**. Worse, plain ridge
pulls towards the origin and the unpenalised fit already has the smaller norm, so shrinkage moves
`α` *down* while the better-extrapolating end of the ridge lies at *higher* `α`. The penalty
pushes the wrong way. The identifiability problem stands unresolved rather than being concealed
behind a regularisation term that happens to be present.

---

## 7. Evaluation and benchmarking

### Metrics

Training titres span 283–4823, a factor of seventeen, and the test set sits in the upper tail.
RMSE is in target units and dominated by high-titre runs; MAPE weights every run equally and is
dominated by low-titre runs. **Both are reported:** a model that improves one while worsening the
other has not improved, it has changed *which runs it is good at*. R² is unstable on ten
upper-tail runs and is read alongside the absolute errors, never instead.

### Three instruments, three jobs, never mixed

| instrument | job | runs |
|---|---|---|
| 10-fold CV inside the short runs | **selection** — mechanisms, variant, ridge | 90 |
| 10-fold CV over everything | **reporting** — matches deployment | 100 |
| leave-duration-out | **held-out diagnostic** — the real shift | 10 |

*Leave-duration-out* trains on runs of ≤10 days and holds out the 14-day runs, reproducing the
shift the task demands. It is the most honest number and it **chooses nothing**.

*The 100-run CV* matches deployment: the shipped model is fitted on all 100 and asked to predict
20 new 14-day runs. It pools 100 predictions rather than 10, so it is the instrument that can
actually resolve differences — but it measures interpolation across mixed durations.

**The gap between the two is itself a result**, separating general model weakness from the cost of
extrapolating in time. The two never appear in one table without that label.

*Selection* runs on the 90 short runs alone so the held-out ten need no caveat about their own
construction. That costs accuracy in the selection and the trade is accepted deliberately: the
duration shift is the whole difficulty of the task.

Every parameter is refitted inside every fold, `kl` included. Reusing a `kl` fitted on all the
data would leak held-out runs into every fold; it feels structural, which is exactly why it is an
easy mistake. No grouped splitter is needed — each experiment contributes exactly one sample.

**The fold parameters are a diagnostic, not the deliverable.** Ten folds give ten independent
estimates and how far they move says whether the data identify them at all. That diagnostic has
since been promoted into the admission rule in §6.

### Results

| | CV over 100 runs | | leave-duration-out (10 × 14-day) | |
|---|---|---|---|---|
| | RMSE | MAPE | RMSE | MAPE |
| mean baseline | 757.4 | 54.7% | 1699.6 | 50.4% |
| M0 — net growth | 759.7 | 36.3% | 2815.7 | 126.2% |
| M1 — effective growth | 577.6 | 26.8% | 1514.7 | 61.1% |
| PLS | 352.8 | 22.6% | 1478.4 | 66.5% |
| gradient boosting | 366.8 | 17.3% | 954.6 | 36.3% |
| M3 — `F` on the whole rate law | 330.8 | 18.0% | 754.8 | 19.5% |
| **M2 — `F` on the non-growth term** | **286.6** | **14.6%** | **721.2** | **19.9%** |

Paired bootstrap on the 100-run CV:

| comparison | RMSE difference | 90% interval | distinguishable? |
|---|---|---|---|
| M1 vs M0 | +182.1 | [+109, +254] | yes |
| M2 vs M1 | +291.1 | [+154, +414] | yes |
| M3 vs M2 | −44.2 | [−94, +9] | **no** |
| M2 vs PLS | +66.3 | [−7, +132] | **no** |
| M2 vs gradient boosting | +80.3 | [+26, +131] | yes |

**M0 reproduces its documented failure** (`α` negative, worse than the mean baseline), which is
why it is kept: the difference between net and effective growth is demonstrated, not asserted.

### The data-driven benchmarks

A mean baseline is a floor, not a comparator; it cannot answer whether mechanistic structure buys
anything a general-purpose learner does not get for free. PLS and gradient boosting are therefore
fitted on the **same** run-level aggregates, from the **same** code, on the **same** folds, with
hyperparameters chosen **inside** each fold. They receive *more* information than the kinetic
model — all 25 always-defined features including `cell_days` and `duration_days` itself, against
two metabolite series.

PLS is not a straw man: it is the established method for this problem shape — many correlated
process variables, few batches — and Johan Trygg co-authored both reference papers *and*
co-invented OPLS. If a mechanistic model could not beat PLS here, the honest conclusion would be
to ship PLS.

**Where the gap sits is the whole argument.** On random folds M2 beats PLS by 66 RMSE and the
paired bootstrap calls that *not distinguishable*. On the duration split the gap is **757**.
Structure buys extrapolation, not raw fit.

**How they fail is more useful than that they fail.** For eight of the ten held-out runs
`cell_days` lies above the training maximum (549 against 243), and the two break in opposite
characteristic ways:

- **PLS overshoots** — 3889 predicted for a run measuring 1790, 4122 for one measuring 1727. Its
  range, 537–4218, is about right; the individual assignments are not.
- **Gradient boosting saturates** — predictions compress into 730–2912 against actuals of
  610–4823. No tree can return a value above its training range, so the highest-titre run is
  capped at 2805. Bounded error, biased low.

The tree scores better precisely *because* it cannot extrapolate; flatness is a cheap form of
safety when the alternative is running away. Neither is a model of the process — they are two ways
of failing to leave the training distribution, which is the failure the kinetic structure exists
to avoid.

The baseline settings were fixed before these numbers were seen and have not been revisited.
Tuning a comparator after seeing that it loses is how a benchmark becomes decoration.

### The selected model

```
qP(t) = 16.62 · μ_eff(t)  +  14.05 · F(z(t))

F(z)  = [ Gln / (0.02185 + Gln) ] · [ Glc / (1.005 + Glc) ]

  α    = 16.62      titre per cell synthesised
  β    = 14.05      titre per cell-day
  kl   = 0.004947   1/day
  K_Q  = 0.02185    mM     glutamine half-saturation
  K_G  = 1.005      mM     glucose half-saturation
```

Five parameters against 100 scalars. Earlier candidate sets reached seven to nine; the
identifiability rule is part of why the shipped count is lower.

**This is hybrid, not purely mechanistic.** The scaffold is a rate law; which physiological
variables modulate it, and how strongly, is learned.

---

## 8. Uncertainty and identifiability

### Parameter uncertainty is not prediction uncertainty

Two different questions, and answering the first where the second was asked understates the real
uncertainty roughly fourfold here.

| | question | shrinks with more data? |
|---|---|---|
| **Confidence interval** | Where does the *parameter* lie? | yes — towards a point |
| **Prediction interval** | How wrong will the *next prediction* be? | **no** |

| | 90% width |
|---|---|
| parameter uncertainty only | 226 |
| **including out-of-fold residual scatter** | **860** |
| residual scatter's share | **73.7%** |

**Three quarters of the predictive uncertainty is model inadequacy, not parameter ignorance.**
That is the most actionable number in the analysis: further effort belongs in the model's
*structure*, not in pinning its constants down. Perfect parameters would remove 226 and leave 860
barely moved.

### How an interval is actually constructed

There is no closed-form expression; it is a simulation, and stating the four steps plainly
matters because the construction is what determines its limits.

1. **Resample and refit, 200 times.** Draw 100 runs with replacement and refit the whole model
   on each draw, giving 200 slightly different sets of `α, β, kl, K_Q, K_G`.
2. **Predict with all 200.** Their spread is *parameter* uncertainty — how much the answer moves
   because the constants are not pinned down. Here that is small: 226 of 860 units.
3. **Pool past errors, as fractions.** From the cross-validation, take the 100 out-of-fold errors
   `(actual − predicted) / predicted` — how wrong the model was on runs it had not seen.
4. **Draw and combine.** Pair each of the 200 predictions with a randomly drawn error fraction
   and apply it multiplicatively. Sort the 200 simulated outcomes; the 5th and 95th percentiles
   are the 90% interval.

**Step 3 is why it works and also where it is fragile.** It assumes the next run's error will
resemble errors already observed — which is precisely the assumption a duration shift breaks. That
is why coverage on the 14-day runs is checked by counting rather than trusted.

**This is not a probability model.** There is no likelihood, no prior and no posterior, so the
result cannot be updated with new data, combined with other sources of uncertainty, or propagated
through a downstream decision. It is a resampling construction that answers one narrow question.
A Bayesian or state-space formulation would give a genuine posterior predictive distribution,
which is a different and better object — see §9 on why that, rather than this, is the route worth
taking next.

### Method

**Runs are resampled, not residuals.** A residual bootstrap holds the fitted structure fixed and
reshuffles noise around it, which assumes the model is correct — precisely the assumption in
question, and the one most likely to be wrong at the 14-day horizon.

**A bootstrap does not correct bias.** It measures how much an estimate moves when the data are
resampled. In a collinear problem the fit can sit away from the generating value along the ridge
and the bootstrap will faithfully report a tight interval around the wrong place. A narrow
bootstrap interval means "stable under resampling", never "close to the truth". This is asserted
in the tests so it cannot quietly be forgotten.

**Search ranges and profile ranges are different things.** A search range must contain the
*optimum* and be resolvable by the grid; a profile range must contain the whole *confidence
interval*. Conflating them gave a wrong answer twice, in an earlier mechanism set that included
`ph_response`: `theta_pH` first fitted hard against a ±3 bound, then at ±6 the optimum sat at 4.28
but the profile still reported the upper bound as unbounded because the interval's true upper end
is 6.19. Widening the *search* range to ±12 fixed the profile but coarsened the first sweep and
cost 5% of cross-validated accuracy (RMSE 440 → 462). So the fit searches ±6 and
`profile_likelihood()` takes its own wider range.

**A profile re-optimises every other parameter** at each pinned value. Holding them fixed would
give a slice, not a profile, and would make almost any parameter look sharply determined by
forbidding the others from compensating. The interval threshold is the standard nonlinear
least-squares region, `RSS(θ) ≤ RSS_min·(1 + F(1,n−p,conf)/(n−p))`, which assumes local linearity
and is indicative rather than exact here.

**The mechanism set is held fixed during the parameter bootstrap**, so parameter intervals are
conditional on having selected these mechanisms and exclude selection uncertainty. That is
quantified separately below rather than blended in.

### Profile likelihood

| constant | unit | fitted | 90% interval | max RSS rise |
|---|---|---|---|---|
| `kl` | 1/day | 0.004947 | **[0.00360, 0.00872]** | 73.5% |
| `K_Q` | mM | 0.02185 | **[0.0113, 0.0369]** | 201.5% |
| `K_G` | mM | 1.005 | **[0.640, 1.566]** | 131.9% |

All three are bounded with a substantial residual rise — a different picture from the earlier
model, where two of four constants ran to infinity because their mechanism was switched off.

### `kl` is well identified and is not a rate

These are separate claims, and conflating them was a real error in an earlier draft.

> `1/kl` = **202 days**, against an observation window of **7–14 days** — 14× to 29× longer than
> the experiment. Over the longest run `exp(−kl·t)` falls only from 1.000 to **0.933**, so just
> **6.7%** of the dead pool ever lyses. The exponential never bends.

In that regime `dXl/dt ≈ kl·Xd` with `Xd` roughly constant, so the fit sees essentially the
*product* — and since `Xd` is *defined* as `Xl'/kl`, the dead pool is whatever `kl` declares it to
be. So **`kl` is a scale factor sizing the inferred dead pool, not a measured lysis rate.** Its
narrow profile reflects how strongly the fit needs a dead pool of that magnitude, not knowledge of
how fast cells lyse. The reading is robust to the uncertainty: at the fast end of the bootstrap,
0.0075 /day, the time constant is still 133 days.

What survives is the dead pool's **magnitude**, which rests on measurement: over the post-peak
decline VCD falls by a median of 3.04 while `X:Lysed` rises 0.115, so lysate accounts for only
~3.4% of the viable cells lost and the rest is unmeasured. A substantial dead pool follows from
the data, with no organism assumed.

### Bootstrap over runs — 200 resamples

| parameter | point | median | 90% interval |
|---|---|---|---|
| α | 16.62 | 17.39 | [11.9, 23.1] |
| β | 14.05 | 13.93 | **[11.3, 15.9]** |
| `kl` | 0.004947 | 0.004975 | [0.00330, 0.00750] |
| `K_Q` | 0.02185 | 0.01991 | [**0.01**, 0.0868] |
| `K_G` | 1.005 | 0.9933 | [0.323, 3.03] |

**β is clearly sign-determined.** In the earlier model it spanned −7.3 to 16.5 and crossed zero,
which undercut M2's structural claim — β is precisely the coefficient the environmental factor
multiplies, so "the environment scales non-growth production" was a statement about a quantity
that might be zero. That objection is gone.

**`K_Q`'s lower bootstrap bound is the search floor, not an estimate.** It starts at exactly 0.01,
the grid minimum, so it reads as "≤ 0.01". The point estimate is sound and *not* grid-limited:
widening the floor from 10⁻² to 10⁻⁶ leaves `K_Q` at 0.0218 with an unchanged residual sum of
squares. Note 15.2% of glutamine measurements lie below 0.01 mM and the series reaches exactly
zero, so this mechanism acts only in the final depletion regime.

**`K_G` is determined only to within about an order of magnitude** — bootstrap [0.32, 3.03], fold
spread 8.28× — and should be quoted that way rather than to four figures. The *prediction* is far
less sensitive than the constant: every fold value from 0.18 to 1.47 mM is small against a glucose
range of 0–44 mM, so the factor stays near one except where glucose is nearly exhausted. A
sensitive constant and an insensitive prediction are different claims, and both are true here.

### Parameter correlation

From the bootstrap draws, so shape-constant uncertainty is included rather than conditioned away.

|  | α | β | `kl` | `K_Q` | `K_G` |
|---|---|---|---|---|---|
| **α** | +1.000 | −0.593 | +0.257 | +0.187 | **+0.619** |
| **β** | −0.593 | +1.000 | +0.352 | −0.044 | +0.048 |
| **`kl`** | +0.257 | +0.352 | +1.000 | +0.193 | +0.268 |
| **`K_Q`** | +0.187 | −0.044 | +0.193 | +1.000 | −0.187 |
| **`K_G`** | **+0.619** | +0.048 | +0.268 | −0.187 | +1.000 |

**α–`K_G` at +0.619** is the largest off-diagonal term. Raising `K_G` weakens the glucose factor
and the fit compensates through the growth coefficient, so neither should be quoted alone.

**α–β at −0.593** is a large improvement on the **−0.97** of the two-term model that motivated
this rework: moving the factor onto the non-growth term genuinely separated the coefficients. It
remains substantial, which is why the closed-form standard errors stay optimistic.

**α–`kl` at +0.257** confirms the structural confounding the model documents — the dead-cell part
of the growth regressor scales as α/`kl` — and shows it is milder than feared.

### Mechanism stability — 25 resamples, the whole selection re-run in each

| mechanism | selected in | shipped? |
|---|---|---|
| `glucose_limitation` | **80%** | yes |
| `temperature_response` | 60% | no |
| `glutamine_limitation` | 56% | yes |
| `ph_response` | 56% | no |
| `metabolic_burden` | 8% | no |
| `lysate_inhibition` | 8% | no |

**The mechanism set is not stable, and this is the honest headline of the uncertainty analysis.**
`temperature_response` is selected *more often* than a mechanism that is in the shipped model. A
different draw of 100 experiments would plausibly have produced a different pair, so the
post-selection caveat is not a formality but a measurement: the shipped constants are point
estimates **conditional** on this mechanism set, and their standard errors, computed after
selecting on the same data, are not valid as stated.

Two things survive it. The gap between the mechanisms the identifiability guard **rejected** (8%)
and those selected (56–80%) is clean, so the guard is not discarding candidates the resamples
liked. And the four clustered at 56–80% are all plausible productivity effects, so the instability
is about *which* of several real effects 90 runs can resolve — not about whether `F(z)` belongs in
the rate law, which the M1→M2 comparison settles separately and decisively.

### Calibration, checked by counting

| coverage check | nominal | achieved |
|---|---|---|
| all 100 training runs | 90% | 93.0% |
| the ten 14-day runs | 90% | **90.0%** |

The in-distribution 93% is a consistency check, not validation — the residuals and the outcomes
come from the same runs. The 14-day figure is the one that matters, and it is at nominal.

**It was not always.** An earlier version pooled *absolute* residuals and covered only **60%** of
the 14-day runs. The diagnosis was that the error is multiplicative rather than additive: across
the duration split the mean absolute residual nearly triples (159 → 459) while the mean relative
residual barely moves (13.7% → 17.7%). Adding a pooled absolute residual therefore gives a
300-unit run and a 4800-unit run the same interval width, which under-covers exactly the runs the
task is about.

**The correction is calibration, not padding**, and the control says so: widened to the *same*
mean width, additive intervals still cover only 50% of long runs, and reaching 80% that way needs
2.2× the width and then over-covers short runs at 99%. Switching to relative residuals moved
14-day coverage from 60% to 90% while the mean width went *down*, 866 → 860. Width was
redistributed to where the error is, not manufactured.

**The honest caveat.** 90% of ten runs is nine of ten, and binomial noise on ten draws is wide in
both directions — with true coverage 90%, seeing eight or fewer covered has probability 0.26. The
claim is "consistent with nominal", not "proven calibrated". A hundred 14-day runs would settle
it; ten cannot.

---

## 9. Known weaknesses

Stated here rather than left to be found.

- **The mechanism set is not stable under resampling**, with a non-selected mechanism chosen more
  often than a selected one. The rate law's *form* is well supported; *which* two physiological
  effects fill it is not settled by 90 runs. This is the largest open weakness.
- **The interval calibration rests on ten runs.** 90% coverage on the 14-day runs is consistent
  with nominal but not established by it; binomial noise at n=10 is wide.
- **`K_G` is determined only to within an order of magnitude**, though predictions are far less
  sensitive to it than the constant is.
- **`kl` is a scale factor, not a rate** — 202-day time constant against a 7–14 day window.
- **α and `kl` are partially confounded** (+0.257); α and `K_G` more so (+0.619).
- **Five parameters against 100 scalars**, with correlated candidates. Cross-validated comparison
  has to be the arbiter rather than a formality.
- **`f_lys` cannot distinguish reduced synthesis from product degradation.** Lysed cells release
  proteases; from one titre measurement per run the two are indistinguishable. It did not enter
  the final model, but the ambiguity would matter if it had.
- **The within-interval constant-`F` assumption is weakest on feed-start days**, when metabolites
  move sharply within a day.
- **Screening works on `q̄P`**, which averages over both production routes, so it identifies
  *which* variables matter but not *where* they act; the M2/M3 comparison answers that.
- **No bioreactor volume**, so no dilution terms; feeds treated as concentration source terms.
- **`X:Lysed` units are inferred, not documented.**
- **The model cannot predict an unrun experiment** — it conditions on measured trajectories.
- **The held-out figure is clean by protocol, not by history.** The pipeline never lets a
  selection step read those ten runs, but they were examined by hand while diagnosing a mechanism
  problem. No code change undoes that, so it is stated.

---

## 10. Future directions

Three, in the order they would pay off.

### Serve the prediction interval

`POST /predict` currently returns a **scalar**, which is what the spec's description asks for and
all it strictly requires — though the response schema is explicitly left to the implementer, so an
interval is permitted.

The machinery exists and is now calibrated at nominal on the runs that matter, so the remaining
question is **cost, and it is an implementation question rather than a modelling one.** Serving an
interval means evaluating the 200 bootstrap parameter sets per request instead of one. The
expensive part of a prediction is the NNLS spline fit in `run_quantities`, and that is *shared*
across all 200 — each additional parameter set costs only a few array operations over the run's
timepoints. Done that way the overhead should be small; done naively, by calling `predict()` 200
times and refitting the spline each time, it would be roughly 200× the work. **This is worth
measuring rather than assuming**, and the artefact would need to carry the bootstrap parameter
sets and the residual pool alongside the point estimates.

### Replace the resampling construction with a probability model

The interval described in §8 is a resampling trick, not a probabilistic model: no likelihood, no
prior, no posterior. It cannot be updated as data arrive, combined with other sources of
uncertainty, or propagated through a downstream decision — and its central assumption, that the
next run's error resembles errors already seen, is exactly what a duration shift breaks.

A Bayesian or state-space formulation of the same rate law would give a genuine posterior
predictive distribution instead. That is a different and better object: it would put uncertainty on
`kl` and the Monod constants *jointly with* the prediction, rather than bolting a residual pool
onto a point fit. It would also handle the mechanism instability below more gracefully, since model
uncertainty could be averaged over rather than conditioned away.

### Resolve which mechanisms belong in `F(z)`

Four mechanisms sit at 56–80% selection frequency across resamples and the data cannot separate
them. More 14-day runs would help most; failing that, designed experiments varying glutamine and
temperature independently would break the collinearity that makes the choice arbitrary.

---

## 11. Reproducing this

```bash
python scripts/screen_and_fit.py                # the pipeline
python scripts/screen_and_fit.py --uncertainty   # plus profiles, bootstrap, stability
```

**The deliverable is the artefacts, not the terminal output.** The pipeline writes
`artefacts/titre_model.json` — which the Part 2 service loads — and
`artefacts/training_report.json`, holding every number in this document in machine-readable form.
Both carry provenance: a hash of the training data, the random seed and package versions, so a
served prediction can be traced to the run that produced it.

---

## 12. Related documents

- [`data-loading.md`](data-loading.md) — the data layer: schema, loading, control reconstruction
- [`../README.md`](../README.md) — the project overview
