# Analysis

How we got from the problem statement to the model.

---

## 1. Problem

Predict the **final mAb titre** of a simulated CHO fed-batch bioprocess: one scalar
per experiment.

The data give 100 training experiments and 20 test experiments, one row per
(experiment, day) on an exact one-day grid. Each row carries process design scalars
(`Z:`), control profiles (`W:`), and measured observations (`X:`). Titre (`Y:`) is
recorded **once per run, at harvest** — there is no titre trajectory anywhere.

The difficulty is not the regression. It is the horizon:

| run length | training runs | test runs |
|---|---|---|
| 7 days | 30 | 0 |
| 8 days | 20 | 0 |
| 9 days | 20 | 0 |
| 10 days | 20 | 0 |
| **14 days** | **10** | **20** |

Every test run lasts 14 days, but only 10 of 100 training runs reach that horizon.
A purely statistical model has to extrapolate in time from ten examples. A model built
on rate equations integrates to any horizon by construction. That is the reason for
taking a mechanistic route.

---

## 2. Equations

An unstructured kinetic model. The cell population is split into viable, dead and
lysed cells, with a catch-all "biomaterial" variable representing accumulated
by-products.

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
q_P   = (alpha*mu_eff + beta) * f_T(T) * g_pH(pH)
```

with the effective growth rate related to the net rate by

```
mu_eff = mu + mu_d + mu_l
```

### Not states

Temperature, pH, stirring and dissolved oxygen are **inputs**, not dynamic variables.
Temperature and pH are exact step functions of the `Z:` scalars, and stirring and
dissolved oxygen are per-run constants:

```
W:temp(t)    = tempStart    if t <  tempShift  else tempEnd
W:pH(t)      = phStart      if t <  phShift    else phEnd
W:FeedGlc(t) = FeedRateGlc  if FeedStart <= t < FeedEnd  else 0
W:FeedGln(t) = FeedRateGln  if FeedStart <= t < FeedEnd  else 0
```

They enter the rate laws as modulators. Giving them differential equations would mean
integrating quantities we already know exactly.

---

## 3. References and reasoning

Two papers, both from the same group, provide the model structure.

**[1] Richelle et al. (2022).** *Model-based intensification of CHO cell cultures:
one-step strategy from fed-batch to perfusion.* Front. Bioeng. Biotechnol. **10**:948905.
doi:10.3389/fbioe.2022.948905

**[2] Richelle et al. (2026).** *A Hybrid Modeling Framework for Predictive Digital
Twins of CHO Cell Culture.* Comput. Struct. Biotechnol. J. **35**(1):Article 0078.
doi:10.34133/csbj.0078

### What we took, and why

**The three-population split and the biomaterial variable — from [1].** Viable, dead
and lysed cells, with cell death driven by a catch-all by-product variable rather than
by an explicit list of inhibitors. This keeps the parameter count low: `gammaX` obeys
`dgammaX/dt = Xv` and so costs no parameters of its own.

**`mu_eff = mu + mu_d + mu_l` — from [2], Eq. 14.** The growth rate obtained from the
slope of the viable-cell curve is a *net* rate. Product formation scales with biomass
*synthesis*, so the Luedeking–Piret term must multiply `mu_eff`, not net `mu`. Over a
14-day run with substantial death these differ considerably.

**The sequential lysis form — from [1], not [2].** The two papers disagree:

- [1] treats lysed cells as a degradation product of the **dead** pool:
  `dXd/dt = mu_d*Xv − kl*Xd`, `dXl/dt = kl*Xd`
- [2] treats dead and lysed cells as **parallel** products of the viable pool:
  `dXd/dt = mu_d*Xv`, `dXl/dt = mu_l*Xv`

We chose [1]. The reason is specific to this dataset. **We measure lysed cells but never
measure dead cells.** Under the parallel form, lysis bypasses the dead pool entirely, so
the lysed-cell trajectory tells us nothing about the dead pool, and `Xd` stays invisible.
Under the sequential form, lysed cells come *from* dead cells, so

```
Xd(t) = (1/kl) * dXl/dt
```

and the dead pool becomes readable from measured data, up to the single constant `kl`.

This is worth stressing: in **both** papers the lysed-cell population is a latent
variable. [1] states plainly that it "acts as a degree of freedom in the model since no
experimental measurements were available", which is why `kl` was by far their worst
identified parameter. Here it is measured. That is an advantage over the published
models, and the sequential form is what converts it into one.

**What we did not take.** The chain from agitation through `kLa` to oxygen uptake is not
modelled. Dissolved oxygen is a controlled setpoint handed to us — the controller has
already closed that loop. Building the chain would mean predicting a variable we already
observe exactly, at the cost of parameters that no oxygen measurement, oxygen uptake
rate or off-gas data could constrain.

**What the data forced.** No bioreactor volume is supplied, so dilution terms `(F/V)`
cannot be identified and are omitted, and feeds are treated as additive concentration
source terms. Both papers carry `F/V` explicitly; we cannot. Full reasoning in
[`../Assumptions.txt`](../Assumptions.txt).

---

## 4. Final model

The task is to predict titre, not to simulate a bioreactor — and the `/predict` payload
supplies the **full observed trajectories**. So rather than simulating the culture
forward, we integrate product formation *along measured data*.

Integrating `dP/dt = (alpha*mu_eff + beta)*Xv` over the run, and substituting
`mu_eff*Xv = dXv/dt + mu_d*Xv`:

```
titre = alpha * (cells made) + beta * (cell-days)
```

where

```
cells made = dXv + Xl_final + Xd_final          Xd_final = (1/kl) * dXl/dt |_end
cell-days  = INT Xv dt   ( = gammaX )
```

Both regressors are computed from measured trajectories. The only unknown quantity,
the dead-cell pool, is recovered from the lysed-cell slope through the single constant
`kl`.

**Consequence: no differential equation is solved at inference.** The model is a set of
quadratures over observed trajectories plus three fitted numbers — `alpha`, `beta` and
`kl`.
