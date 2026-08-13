# The data layer

How the supplied CSV files become objects the model can use, and why it is built this
way.

*This document will be folded into the main README at the end; it is kept separate
while the data layer is the thing under discussion.*

---

## The one idea behind the design

> **Separate what an experiment *is* from where the data came from.**

Right now the data arrives as CSV files. In Part 2 the same data arrives as a JSON body
posted to `/predict`. If feature calculations reached directly into CSV columns, all of
it would have to be written a second time for the web service — and the two copies would
drift apart. That is the standard way an inference service ends up quietly returning
different numbers from the model that was trained.

So both paths build the **same object**, `ExperimentRun`, and everything downstream —
features, fitting, prediction, the API — only ever sees that object.

```
CSV file  ─┐
           ├─►  ExperimentRun  ─►  features  ─►  model
JSON body ─┘
```

---

## The files

```
src/titre_predictor/
├── domain.py            what "an experiment" is, and what makes one valid
└── data/
    ├── schema.py        every column name, declared once
    ├── loading.py       CSV → ExperimentRun
    └── controls.py      rebuilding W: from Z:
```

### `schema.py` — the column-name dictionary

Every column name in the dataset, defined in one place. No other file in the codebase
contains a literal like `"X:VCD"`.

Same reasoning as declaring parameters at the top of a script rather than typing `0.85`
in fourteen places: if DataHow renames a column, that is one edit, not a search through
the codebase hoping to find every occurrence.

It also exposes `columns_with_prefix(columns, prefix)`, which is how every other module
asks "which columns are the observations?" without hardcoding the list.

### `domain.py` — `ExperimentRun`

One experiment: its identifier, its timestamps, and three dictionaries.

| field | prefix | what it holds |
|---|---|---|
| `design_scalars` | `Z:` | 13 process design values, constant within the run |
| `control_profiles` | `W:` | 4 control profiles, one value per timestamp |
| `observations` | `X:` | 6 measured series, one value per timestamp |

The shape deliberately mirrors the `/predict` request body in
`inference_server_spec.yml` — a timestamp array plus prefixed variable arrays. In Part 2
a request deserialises straight into this, with no second representation to keep in sync.

**It validates itself on construction.** Whenever an `ExperimentRun` is created, it
checks:

- timestamps are one-dimensional, at least two of them, finite, and strictly increasing
- every `W:` and `X:` array is exactly as long as the timestamps
- no `W:` or `X:` value is missing or infinite — and the offending positions are named
- every design scalar is finite

If any check fails it raises `InvalidExperimentRunError` and the object is never created.

This matters more than it looks. It means **there is no such thing as a half-valid
experiment anywhere in the pipeline**. An entire category of bug is removed by
construction rather than by remembering to check. And because the API builds the same
object, those same rules protect it for free: `InvalidExperimentRunError` subclasses
`ValueError` and is what Part 2 will map to HTTP 400. It always means a malformed
request, never a server fault.

The NaN check exists specifically for the API. The CSVs have no gaps in `W:` or `X:`,
but a request can send one — and a single NaN would propagate silently through the
integrals and return a titre of `nan`, which looks like a working service returning a
bad answer.

### `loading.py` — CSV to experiments

Four steps: read the file, forward-fill the design scalars, sort each run by time, build
one `ExperimentRun` per experiment.

Runs are sorted by time explicitly rather than trusting file order, so a reordered file
cannot silently change the model's inputs.

### `controls.py` — rebuilding `W:` from `Z:`

The control profiles are exact step functions of the design scalars:

```
W:temp    = tempStart    while t <  tempShift, else tempEnd
W:pH      = phStart      while t <  phShift,   else phEnd
W:FeedGlc = FeedRateGlc  while FeedStart <= t < FeedEnd, else 0
W:FeedGln = FeedRateGln  while FeedStart <= t < FeedEnd, else 0
```

Note the feed window is closed on the left and open on the right: `FeedEnd = 11` means
fed on days 3 to 10 inclusive, and not on day 11.

This reconstruction reproduces all four `W:` columns to machine precision across all 1290
rows of both files. So **`W:` carries no information beyond `Z:`** — it is a
re-parameterisation, not an independent input. That has two consequences: features must
not double-count them, and in Part 2 the service can derive the controls when a request
omits them, or check them when it supplies them.

---

## Missing values

The raw files contain blanks, but **nothing is actually missing**.

| | train | test |
|---|---|---|
| blanks in `Z:` columns | 11,570 | 3,640 |
| blanks in `W:` columns | 0 | 0 |
| blanks in `X:` columns | 0 | 0 |
| blanks in time / experiment | 0 | 0 |

The `Z:` blanks are a storage convention: design scalars are constant for a run, so they
are written on its first row and left blank on the rest.

**The policy is: propagate what the format implies, then fail loudly if anything is
genuinely absent.** No means, no zeros, no interpolation.

`forward_fill_design_scalars` groups by experiment and forward-fills within each group.
The grouping is the important part — a plain forward-fill across the whole table would
carry one run's design values into the next run's rows, silently attaching the wrong
process conditions to an experiment. Afterwards it checks its own work and raises if any
design cell is still blank, which would mean a run had no value even on its first row.

There is deliberately **no imputation code for `X:` or `W:`**, because there is nothing
to impute. If those had gaps it would be a real modelling decision needing justification;
writing speculative imputation now would be untested code guarding a case that does not
exist.

---

## The tests, and what each is actually for

62 tests, roughly half a second. They are not box-ticking — they encode the facts the
model depends on.

### `test_domain.py` (14) — the validation rules

One test per way an experiment can be malformed: non-increasing timestamps, a single
timepoint, mismatched array lengths, NaN in an observation, infinity in an observation,
NaN in a control profile, NaN in the timestamps, a non-finite design scalar.

Two are worth singling out:

- *missing values are located by index* — the error names which positions are bad, so an
  API caller gets an actionable message rather than "invalid input"
- *missing timestamps are reported as such* — without an explicit check this would
  surface as "not strictly increasing", because every comparison against NaN is false.
  Technically a rejection, but a misleading message that would waste someone's afternoon.

### `test_loading.py` (14) — structure

That the file yields 100 training and 20 test runs; that all 20 test runs last 14 days
while only 10 of 100 training runs do; that the time grid is exactly one day starting at
zero; that `Z:ExpDuration` equals the last timestamp for every run, so the two possible
readings of "harvest time" cannot disagree; that runs come out time-ordered even when the
file is shuffled; that forward-fill does not mutate its input; that a targets file with a
duplicated experiment is rejected.

### `test_controls.py` (7) — the `W:`/`Z:` relationship

The headline test asserts the reconstruction reproduces the supplied columns with
**zero tolerance** — `rtol=0, atol=0` — for every run of both files. It is a guard on a
claim the model relies on. If the control convention ever changes, this fails loudly
instead of quietly altering the model's inputs.

The rest pin the conventions individually: the step happens *at* the shift day; a shift
scheduled beyond the end of a run never takes effect (true for over half the training
runs); the feed window is closed-left, open-right.

### `test_data_fidelity.py` (27) — the numbers are the numbers

Three checks, each covering the others' blind spot:

| check | catches | blind spot |
|---|---|---|
| every cell against a direct `pd.read_csv` (~12,900 cells) | misassigned columns, misordered rows, anything anywhere | if the data file itself changed, both sides move together |
| 14 hardcoded values scattered across all four files | the data file being altered or replaced | only the cells chosen |
| every observation against the file's **raw text**, parsed with Python's own `float()` | rounding — including rounding by pandas | observations only |

The third is the answer to "does anything get rounded?". The first test compares our
loader against `pd.read_csv`; if pandas itself rounded, both sides would round
identically and it would still pass. So this one reads the CSV as *strings* and compares
against the characters on disk. The files carry ten significant figures, float64 holds
about fifteen, and every digit must survive.

Also here: design scalars must come from that run's **own** first row, which is what
catches a forward fill leaking across a run boundary; blank counts pinned exactly; and a
test that the supplied test-targets file is still all `2000`.

That last one **is designed to fail**. When the real test targets arrive at interview
time and are dropped in, it goes red — which is the reminder to wire them into the
evaluation rather than silently scoring against placeholders.

---

## Running it

```bash
pytest
```

```bash
ruff check .
```

```bash
ruff format --check .
```

```bash
mypy
```

To look at the loaded data yourself:

```python
from pathlib import Path
from titre_predictor.data.loading import load_runs

runs = load_runs(Path("data/raw/datahow_interview_train_data.csv"))
run = runs[0]
print(run.experiment_id, run.duration_days)
print(run.observations["X:VCD"])
print(run.design_scalars)
```
