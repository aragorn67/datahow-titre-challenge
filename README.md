# Final mAb Titre Prediction

Predicting the final monoclonal antibody titre of a simulated fed-batch
bioprocess, and serving that model over a REST API.

DataHow Machine Learning Engineer coding challenge.

## The task

**Part 1** — predict final product titre, a single scalar per experiment, from
time-series bioprocess data.

**Part 2** — implement the inference microservice described in
[`inference_server_spec.yml`](inference_server_spec.yml): `GET /health` and
`POST /predict`.

## The data

100 training experiments and 20 test experiments, one row per (experiment, day)
on an exact one-day grid.

| prefix | meaning | count |
|---|---|---|
| `Z:` | process design scalars, constant within a run | 13 |
| `W:` | control profiles over time (temperature, pH, feeds) | 4 |
| `X:` | measured observations (VCD, Glc, Gln, Amm, Lac, Lysed) | 6 |
| `Y:` | target — titre, observed once, at harvest | 1 |

Test targets are supplied at interview time, so the test set informs no modelling
decision. All model selection is done on the training runs.

**The brief names no cell line** — "a simulated bioprocess for monoclonal antibody
(mAb) production" — so none is assumed. That constraint is load-bearing rather than
pedantic: the reference papers are CHO studies, and borrowing their organism would
licence plausibility arguments the data here cannot support.

## The problem that determines the design

**Every test run lasts 14 days. Only 10 of the 100 training runs reach that horizon.**

That single fact drives everything. A model fitted on run-level aggregates has to
extrapolate in time from ten examples; a model built on rate equations integrates to
any horizon by construction. For eight of the ten held-out 14-day runs, cell-day
exposure lies *above the entire training range* — 549 against a maximum of 243.

So the model is **mechanistic in structure and data-driven in content**: a
Luedeking–Piret rate law whose environmental modulation of specific productivity is
selected from the data rather than asserted. Hybrid, not purely mechanistic, and the
docs say so plainly.

## The model

```
qP(t) = 16.62 · μ_eff(t)  +  14.05 · F(z(t))

F(z)  = [ Gln / (0.02185 + Gln) ] · [ Glc / (1.005 + Glc) ]

titre = ∫ qP(t) · Xv(t) dt
```

Because the `/predict` payload supplies the full observed trajectories, product
formation is integrated **along measured data** rather than by simulating the
bioreactor forward — **so no differential equation is solved at inference.** Prediction
is a set of quadratures plus five fitted constants, regardless of how expensive the fit
was.

The dead-cell pool `Xd`, never measured, is recovered from the measured lysed-cell
trajectory through a sequential lysis form, `Xd = Xl'/kl`, with `Xl(t)` fitted by
non-negative least squares on an integrated B-spline basis so its derivative is monotone
by construction rather than by inspection.

Structure follows Richelle et al., *Front. Bioeng. Biotechnol.* **10**:948905 (2022)
and *Comput. Struct. Biotechnol. J.* **35**(1):0078 (2026).

## Results

Two splits measuring two different tasks. **They are never compared to each other.**

| | CV over 100 runs | leave-duration-out (10 × 14-day) |
|---|---|---|
| mean baseline | 757.4 | 1699.6 |
| PLS | 352.8 | 1478.4 |
| gradient boosting | 366.8 | 954.6 |
| **selected model** | **286.6** | **721.2** |

RMSE in titre units; MAPE 14.6% and 19.9% for the selected model.

The data-driven benchmarks read the **same** aggregate features from the **same** code
on the **same** folds, with hyperparameters chosen **inside** each fold, and are given
*more* inputs than the kinetic model uses (25 features including duration itself,
against two metabolite series). PLS is not a straw man: it is the established method for
this problem shape, and Johan Trygg co-authored both Richelle papers *and* co-invented
OPLS.

**Where the gap sits is the whole argument.** On random folds the kinetic model beats
PLS by 66 RMSE and a paired bootstrap calls that *not distinguishable*. On the duration
split the gap is **757**. Structure buys extrapolation, not raw fit.

How the benchmarks fail is more informative than that they fail: PLS **overshoots**
(3889 predicted for a run measuring 1790), gradient boosting **saturates** (compressed
into 730–2912 against actuals of 610–4823, since no tree exceeds its training range).
Two opposite ways of failing to leave the training distribution.

## How decisions were made

Three instruments, three jobs, never mixed:

| instrument | job | runs |
|---|---|---|
| 10-fold CV inside the short runs | **selection** — mechanisms, variant, ridge | 90 |
| 10-fold CV over everything | **reporting** — matches deployment | 100 |
| leave-duration-out | **held-out diagnostic** | 10 |

Every parameter is refitted inside every fold, `kl` included — reusing one fitted on all
the data would leak held-out runs into every fold.

Four decisions worth naming:

- **Screening and mechanism selection are separate stages.** A variable screen ranks
  variables; it cannot see whether the mechanism assigned to a variable can represent
  the *sign* of its association. Lactate tops the screen at +0.767 with productivity,
  and its inhibition term can only bend downwards — so the fit switches it off. Glucose
  is dropped by the screen as collinear, and its Monod term does the work.
- **Where `F` acts was tested, not assumed.** Four nested variants (M0–M3) differing only
  in whether growth is net or effective and where the factor applies. `F ≡ 1` makes M2
  and M3 collapse onto M1 *exactly*, so the comparison is genuinely nested.
- **A mechanism must be identifiable, not merely predictive.** One candidate cleared the
  1% improvement bar and then failed to transfer; every fold had fitted its constant far
  outside the measured concentration range, i.e. switched the mechanism off. Mechanisms
  now need constants the folds agree on as well as a margin.
- **One margin governs every selection step** — mechanisms, variant, ridge — so the same
  measurement is not read to two different precisions.

## Known limitations

Stated here rather than left to be found:

- **The mechanism set is not stable under resampling.** Re-running the whole selection in
  25 resamples, a mechanism that is *not* shipped is selected more often than one that
  is. The rate law's form is well supported; which two physiological effects fill it is
  not settled by 90 runs. This is the largest open weakness.
- **The interval calibration rests on ten runs.** Prediction intervals cover 90% of the
  14-day runs against a nominal 90%, but binomial noise at n=10 is wide in both
  directions — consistent with nominal, not established by it. `/predict` returns a
  scalar; serving the interval is recorded as future work.
- **`kl` is a scale factor, not a rate.** Its time constant is 202 days against a 7–14
  day window, so lysis is never observed on its own timescale.
- **`K_G` is determined only to within an order of magnitude**, though predictions are
  far less sensitive to it than the constant is.
- **No bioreactor volume is supplied**, so dilution terms cannot be identified and feeds
  are treated as additive concentration source terms.
- **The model cannot predict an unrun experiment** — it conditions on measured
  trajectories.

## Layout

```
docs/modelling.md          Part 1 in full: equations, assumptions, estimation,
                           evaluation, benchmarks, uncertainty, weaknesses
docs/data-loading.md       the data layer: schema, loading, control reconstruction
src/titre_predictor/       the package
scripts/screen_and_fit.py  the training pipeline; writes the artefacts
artefacts/                 fitted model + machine-readable training report
data/raw/                  the four supplied CSV files, unmodified
inference_server_spec.yml  the OpenAPI contract for Part 2
```

Two documents, deliberately. Everything about the *model* — including the eight
assumptions it rests on — belongs in one place, because an assumption is only
meaningful next to the choice it constrains. Everything about the *data* is
independent of which model consumes it.

## Getting started

```bash
pip install -e ".[dev]"
```

Reproduce the model and the numbers above:

```bash
python scripts/screen_and_fit.py
```

Add the identifiability work — profiles, bootstrap, mechanism stability (~25 min):

```bash
python scripts/screen_and_fit.py --uncertainty
```

Checks:

```bash
pytest && ruff check . && ruff format --check . && mypy
```

**The deliverable is the artefacts, not the terminal output.** The pipeline writes
`artefacts/titre_model.json` — which the Part 2 service loads — and
`artefacts/training_report.json`, holding every number above in machine-readable form.
Both carry provenance: a hash of the training data, the seed, and package versions, so a
served prediction can be traced to the run that produced it. Printing constants and
expecting them to be copied into the service by hand is how a served model drifts from
the model that was validated.

## Assumptions

Eight assumptions, each stated with the alternative it was chosen over and what breaks
if it is wrong, in [`docs/modelling.md`](docs/modelling.md#4-assumptions).

The most consequential: **no bioreactor volume is supplied**, so dilution terms cannot
be identified and feeds are treated as additive concentration source terms. Both
reference papers carry `F/V` explicitly and we cannot.

The one carrying the most risk: **that a mechanism helping on 7–10 day runs also helps
at 14 days.** Every selection decision rests on it, and it is untestable at this sample
size — isolating the 14-day runs returns to ten points and their noise.
