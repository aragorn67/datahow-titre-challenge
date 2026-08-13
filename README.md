# Final mAb Titre Prediction

Predicting the final monoclonal antibody titre of a simulated CHO fed-batch
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

## Approach in one paragraph

An unstructured kinetic model of CHO growth, death and lysis, with product formation
described by a Luedeking–Piret rate law. Because the `/predict` payload supplies the
full observed trajectories, product formation is integrated **along measured data**
rather than by simulating the bioreactor forward — so no differential equation is
solved at inference. The dead-cell pool, which is never measured, is recovered from
the measured lysed-cell trajectory using a sequential lysis form.

Structure follows Richelle et al., *Front. Bioeng. Biotechnol.* **10**:948905 (2022)
and *Comput. Struct. Biotechnol. J.* **35**(1):0078 (2026).

## Layout

```
Assumptions.txt            modelling assumptions and why each was forced
inference_server_spec.yml  the OpenAPI contract for Part 2
data/raw/                  the four supplied CSV files, unmodified
pyproject.toml             dependencies and tooling configuration
```

## Getting started

```bash
pip install -e ".[dev]"
```

## Assumptions

Every assumption made, and the reason it was forced by the data, is recorded in
[`Assumptions.txt`](Assumptions.txt). The most consequential: no bioreactor volume is
supplied, so dilution terms cannot be identified and feeds are treated as additive
concentration source terms.

---

*This README grows as the project does. Architecture, evaluation strategy and results
are added once there is something to describe.*
