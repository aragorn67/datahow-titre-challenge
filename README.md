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

Two further decisions worth naming — the other two, on screening and on
identifiability, are covered under [Challenges](#challenges) below:

- **Where `F` acts was tested, not assumed.** Four nested variants (M0–M3) differing only
  in whether growth is net or effective and where the factor applies. `F ≡ 1` makes M2
  and M3 collapse onto M1 *exactly*, so the comparison is genuinely nested rather than a
  comparison of three unrelated models.
- **One margin governs every selection step** — mechanisms, variant, ridge — so the same
  measurement is not read to two different precisions. An earlier version required 1% of
  mechanisms while accepting a ridge penalty on 0.5%.

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

## Part 2 — the inference service

Two endpoints, per [`inference_server_spec.yml`](inference_server_spec.yml).

```bash
docker build -t titre-predictor .
docker run --rm -p 8000:8000 titre-predictor
curl http://localhost:8000/health
```

Interactive API documentation, generated from the code, is at
`http://localhost:8000/docs`.

### `GET /health`

**Readiness, not liveness.** 200 only when a model is loaded and a prediction can
actually be served; 503 with the reason otherwise.

The distinction matters. A process can be perfectly alive and completely unable to
serve — running, listening, no model loaded. A health check that returns 200
unconditionally is *worse than none*, because an orchestrator believes it and routes
traffic to a container that cannot answer.

It also reports **which** model is serving — variant, mechanisms, and the artefact's
provenance including the training-data hash. "The service is up" says nothing about
whether the model behind it is the one that was validated.

A failed load does **not** crash the process. Letting it die would be loud, but the
reason would live only in container logs — and "no artefact at that path", "artefact
from an incompatible build" and "corrupt file" are three problems with three different
fixes. The service comes up and says which one it hit. Readiness keeps it out of the
load balancer either way, so nothing degraded is served.

### `POST /predict`

Takes an experiment's trajectories, returns the predicted final titre:

```json
{
  "predicted_titer": 3505.5192,
  "experiment_id": null,
  "model": {
    "variant": "M2",
    "mechanisms": ["glutamine_limitation", "glucose_limitation"],
    "training_data_sha256": "09b9cf2c..."
  },
  "extrapolation": { "checked": true, "beyond_training_range": [], "detail": [] }
}
```

The specification leaves the response schema to the implementer. This returns an
**object** rather than a bare number so the contract can grow — a prediction interval
is the obvious addition — without breaking callers.

**The extrapolation report is the part worth explaining.** This model exists to
extrapolate — every test run is 14 days against mostly-shorter training runs — so a
service that answers with a bare number, indistinguishable from one it is confident
about, withholds what the user most needs. The artefact records the span of every
quantity over the training runs, and each response says whether the request left it.

It **warns rather than refuses**, because being outside the range does not make a
prediction worthless: the case for a kinetic structure is precisely that it degrades
gracefully where a data-driven fit does not, and refusing would discard the capability
the model was chosen for. The `checked` flag distinguishes "checked and clear" from
"could not check".

One figure is easy to misread, so it is stated explicitly. Against the **shipped**
model, fitted on all 100 runs, the supplied test set is almost entirely *inside* the
recorded ranges — none of the twenty exceed the `cell_days` maximum and exactly one
falls below the minimum. The "eight of ten above the training range" figure quoted in
the modelling docs belongs to the **leave-duration-out validation**, a deliberately
harder setting fitted on the 90 short runs alone. Both are true of different models.

Status codes are chosen rather than defaulted:

| situation | code |
|---|---|
| body does not match the schema | 422 |
| body parses, run is unusable — missing series, wrong length, NaN | 400, naming the variable |
| no model loaded | 503 |

**Validation repairs only what can be repaired exactly.** Missing `W:` control
profiles are reconstructed from the `Z:` scalars, which is legitimate because they are
exact step functions of them, verified to machine precision on all 1290 supplied rows.
Everything else is rejected: interpolating a missing observation would let the service
return a confident number computed partly from invented data.

**Which inputs are required comes from the loaded model**, not a hardcoded list.
`X:VCD` and `X:Lysed` are structural; beyond that it is whatever the mechanisms read.
The shipped model therefore requires `X:Glc` and `X:Gln` and does *not* require
temperature or pH — and a model with a temperature term would require `W:temp` with no
code change.

### Deployment

The fitted model is **baked into the image**, so it is a complete versioned unit:
`docker run` serves predictions with no further setup, and the model that was validated
is the model that ships. `TITRE_MODEL_PATH` stays configurable if you would rather
mount one.

The image installs `.[service]` only — numpy, scipy and the web stack — so it carries
neither pandas nor scikit-learn, **174 MB of packages inference has no use for**. That
split is enforced by a test, not a comment: `tests/test_dependencies.py` imports the
prediction path in a subprocess and fails if the training stack appears.

It runs as a **non-root user**, and `HEALTHCHECK` points at `/health` so the container
reports `healthy` only once the model has loaded.

Verified by building and running, not asserted: image 518 MB, container reports
healthy, and `POST /predict` returns 3505.5192 for the specification's example —
identical to the same prediction computed locally.

## Layout

```
docs/modelling.md          Part 1 in full: equations, assumptions, estimation,
                           evaluation, benchmarks, uncertainty, weaknesses
docs/data-loading.md       the data layer: schema, loading, control reconstruction
src/titre_predictor/       the package
    service/               the inference service: DTOs, translation, endpoints
scripts/screen_and_fit.py  the training pipeline; writes the artefacts
artefacts/                 fitted model + machine-readable training report
data/raw/                  the four supplied CSV files, unmodified
inference_server_spec.yml  the OpenAPI contract for Part 2
Dockerfile                 the service image
.github/workflows/ci.yml   lint, types, tests, and a container smoke test
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

Run the service locally, without Docker:

```bash
uvicorn titre_predictor.service.app:app --reload
```

Checks — the same four commands CI runs, so a local pass is a CI pass:

```bash
pytest && ruff check . && ruff format --check . && mypy
```

CI runs them on every push to `main` and every pull request, and separately builds
the image, starts it and asks it for a prediction. Building is not the same as
working: the commonest containerisation failure is an image that builds and then
cannot start.

**The deliverable is the artefacts, not the terminal output.** The pipeline writes
`artefacts/titre_model.json` — which the Part 2 service loads — and
`artefacts/training_report.json`, holding every number above in machine-readable form.
Both carry provenance: a hash of the training data, the seed, and package versions, so a
served prediction can be traced to the run that produced it. Printing constants and
expecting them to be copied into the service by hand is how a served model drifts from
the model that was validated.

## Challenges

The five that shaped the design. `docs/modelling.md` carries the technical detail behind
each.

### Choosing what kind of model this should be

The first decision, and the one everything else followed from. Three options, none
obviously right:

- **Purely statistical** — regress titre on run-level aggregates. Simple, and it treats a
  14-day run as a point in feature space far from anything in training.
- **Purely mechanistic** — a full bioreactor model, simulating growth, metabolism and
  production forward from the design parameters. Principled, and it needs far more
  parameters than 100 single-scalar observations can identify.
- **Hybrid** — a mechanistic scaffold whose uncertain part is learned.

*Resolved* to the third, for a reason specific to this dataset: **the `/predict` payload
supplies the full measured trajectories.** The growth and metabolite equations of a
forward model exist to *predict* those trajectories — and we are handed them. What
survives that substitution is exactly the part no measurement pins down: specific
productivity. So the model integrates a rate law along measured data and solves no
differential equation at inference.

That settled the scaffold and immediately raised the harder half — **what should the
features be?** The rate law's *shape* comes from theory; which physiological effects
modulate productivity does not. That is empirical, and it is where the difficulty moved
(see the third challenge below).

### The test set occupies a regime the training set barely reaches

All 20 test runs last 14 days; only 10 of 100 training runs do. Biomaterial exposure
tops out at 242.7 across runs of ≤10 days, and **14 of the 20 test runs exceed that**.
Ammonia never sustains a decline in any run of ≤10 days — 0 of 90 — while 30% of 14-day
runs show one. The short runs never enter the state the test runs end in.

*Resolved structurally.* Rate equations integrate to any horizon, so a longer run
accumulates more integral rather than landing in an unvisited part of feature space, and
the saturating forms flatten outside the fitted range instead of diverging. The
benchmarks confirm the argument rather than assert it: on the held-out ten, **721**
against 1478 for PLS and 955 for gradient boosting.

### The quantity the model needs is never measured

Luedeking–Piret couples production to *biosynthesis*, so the growth term needs total
cells synthesised — which requires the dead pool, and dead cells are not in the dataset.

*Resolved by choosing between two published population structures.* Under the parallel
form, lysed cells derive directly from viable cells and the measured lysate says nothing
about the dead pool. Under the sequential form they derive *from* the dead pool, so
`Xd = Xl'/kl` and the unmeasured quantity becomes readable from measured data up to one
constant. I chose the parallel form first because it made the *lysis* parameters easy to
identify — which optimised the wrong thing, since lysis parameters were never the
difficulty.

### A statistical criterion cannot see mechanism

Having settled on a hybrid, the seam between its halves caused trouble twice.

The variable screen's strongest result was **lactate**, correlating +0.767 with
productivity. Its mechanism is an inhibition term, which only bends downwards — no
parameter value expresses a positive association. Meanwhile glucose, dropped by the
screen as collinear with lactate, mapped to a Monod term that bends upwards and did the
actual work. *The screen discarded the mechanism that worked and kept the one that could
not.*

Later, a candidate mechanism improved cross-validated error by 3% while switched off:
every one of the ten folds fitted its inhibition constant above 1000 mM against a lactate
range of 0–8 mM, making the factor `1/(1 + 8/1000) = 0.992`.

*Resolved* by separating screening from selection — screening reports which measurements
carry information, mechanism selection is a separate step judged on prediction — and by
requiring mechanisms to have **identifiable constants**, not merely a better score.

### Rigour applied downstream of a biased choice feels like diligence

A temperature term survived three careful checks: not pinned at a grid boundary, not a
proxy for elapsed time (correlation 0.25), and not overfitting — training fit improved 5%
while held-out error improved 15%, the opposite of the overfitting signature.

Every one of those checks was downstream of a decision already made badly. **The 15%
held-out improvement that made it persuasive had been measured on the same ten runs it
was being selected against.**

*Resolved* by moving all selection — mechanisms, variant, ridge strength — inside the 90
short runs, and reading the ten 14-day runs exactly once, at the end. Under clean
selection, temperature is not chosen.

## Assumptions

Each of these exists because something is **missing from the data**. They are the
substitutions that made a modelling technique applicable at all — not preferences, and
not conclusions. Where an assumption is wrong, the consequence is stated.

Full detail, with the alternative each was chosen over, in
[`docs/modelling.md`](docs/modelling.md#4-assumptions).

| missing | assumed | consequence if wrong |
|---|---|---|
| **Bioreactor volume.** No `V` column exists. | Feeds act as additive concentration source terms; dilution `F/V` is omitted from every balance. | Every concentration balance is biased, worst in the most-fed runs — and metabolite concentrations drive both surviving mechanisms. The most consequential assumption here. |
| **Dead-cell measurement.** `Xv` and `Xl` are measured; `Xd` never is. | Lysis is *sequential* — lysed cells derive from the dead pool — so `Xd = Xl'/kl`, recoverable up to one constant. | The growth regressor is wrong, and it is used by every model variant. No fallback. |
| **A data dictionary.** None is supplied, so `X:Lysed` units are undocumented. | It is a cumulative cell density in the same units as `X:VCD`. | `α` silently absorbs a unit conversion and stops meaning "titre per cell synthesised". Predictions survive; the mechanistic reading does not. |
| **Any titre except at harvest.** One scalar per run, so `qP(t)` cannot be identified. | A rate-law *form* (Luedeking–Piret) holds; only its two coefficients are fitted. | The shape is wrong and no amount of data would reveal it, since there is nothing to compare a trajectory against. |
| **14-day runs in quantity.** Only 10 of 100 reach the test horizon. | A mechanism that helps on 7–10 day runs also helps at 14 days. | Selection is optimising for the wrong regime. **Untestable at this sample size** — isolating the 14-day runs returns to ten points and their noise. The assumption carrying the most risk. |
| **Sub-daily sampling.** Observations are one per day. | `F(z)` is roughly constant within a one-day interval. | Interval quadratures are biased, worst on feed-start days when metabolites move sharply within a day. |

Two further assumptions are methodological rather than data-driven, and are documented
with the code that relies on them: that `Xl(t)` is smooth and monotone (a physical
constraint on a cumulative pool, which justifies fitting rather than differencing), and
that the controlled variables are inputs rather than dynamic states (they reconstruct
from the `Z:` scalars to machine precision, so modelling them would mean integrating
quantities already known exactly).
