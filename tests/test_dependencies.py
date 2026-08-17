"""The prediction path must not import the training stack.

Why this is a test rather than a note in ``pyproject.toml``
-----------------------------------------------------------
The inference container installs the base dependencies plus ``service``, and
deliberately not ``training``. If someone later adds ``import pandas`` to
``features.py`` for a one-line convenience, three things happen and none of them
is loud: the container stops building, or it builds and crashes on first request,
or -- worst -- someone "fixes" it by adding pandas back to the base dependencies
and the image silently regains 174 MB.

A comment cannot prevent that. This can, and it fails in the repository rather
than in a deployment.

What counts as the prediction path
----------------------------------
Exactly the modules the service needs to turn a payload into a number:

* ``model``     -- loads the artefact, evaluates the rate law
* ``features``  -- the quadratures, shared with training so the served number
                   cannot drift from the fitted one
* ``kinetics``  -- the environmental factor
* ``domain``    -- the run object
* ``data.schema``   -- the ``Z:``/``W:``/``X:`` naming convention
* ``data.controls`` -- reconstructs ``W:`` profiles from ``Z:`` scalars

``data.loading`` is **not** on this list: it reads the supplied CSVs with pandas
and is a training concern. The service receives JSON, not files.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

# Imported by the service to serve a prediction.
PREDICTION_PATH = (
    "titre_predictor.model",
    "titre_predictor.features",
    "titre_predictor.kinetics",
    "titre_predictor.domain",
    "titre_predictor.data.schema",
    "titre_predictor.data.controls",
)

# Packages that exist only to produce a model. None may be reachable from the
# prediction path.
TRAINING_ONLY = ("pandas", "sklearn", "matplotlib")


def _modules_after_importing(modules: tuple[str, ...]) -> set[str]:
    """Top-level packages present in ``sys.modules`` after importing ``modules``.

    Run in a **subprocess** deliberately. Within the pytest process pandas and
    scikit-learn are already imported by other test modules, so any in-process
    check would pass regardless of what the prediction path actually pulls in --
    it would be measuring the test session rather than the import graph.
    """
    program = (
        "import sys, json\n"
        f"sys.path.insert(0, {str(REPOSITORY_ROOT / 'src')!r})\n"
        + "".join(f"import {name}\n" for name in modules)
        + "print(json.dumps(sorted({m.split('.')[0] for m in sys.modules})))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=True,
    )
    return set(json.loads(completed.stdout))


@pytest.mark.parametrize("forbidden", TRAINING_ONLY)
def test_the_prediction_path_does_not_import_the_training_stack(forbidden: str) -> None:
    """The property the container's size depends on."""
    imported = _modules_after_importing(PREDICTION_PATH)

    assert forbidden not in imported, (
        f"{forbidden} is reachable from the prediction path, so the inference image "
        f"would need the training dependencies. Check which of {list(PREDICTION_PATH)} "
        f"imports it."
    )


def test_the_prediction_path_still_imports_what_it_genuinely_needs() -> None:
    """Guards the test above from passing vacuously.

    If a rename broke the imports, ``_modules_after_importing`` would return a set
    with no heavy packages at all and the forbidden-package assertions would pass
    while proving nothing.
    """
    imported = _modules_after_importing(PREDICTION_PATH)

    assert "numpy" in imported
    assert "scipy" in imported, "the lysate spline needs scipy; its absence means a broken import"


def test_loading_csvs_is_a_training_concern_and_may_use_pandas() -> None:
    """The other side of the split, stated so the boundary is unambiguous.

    ``data.loading`` is allowed pandas precisely because the service never calls
    it. If that ever changes, this test is where the decision gets revisited.
    """
    imported = _modules_after_importing(("titre_predictor.data.loading",))

    assert "pandas" in imported
