"""Shared fixtures.

Paths are derived from this file's location rather than hardcoded, so the suite runs
from any working directory and survives the repository being moved or renamed.
"""

from pathlib import Path

import pytest

from titre_predictor.data.loading import load_runs, load_targets
from titre_predictor.domain import ExperimentRun

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIRECTORY = REPOSITORY_ROOT / "data" / "raw"


@pytest.fixture(scope="session")
def raw_data_directory() -> Path:
    return RAW_DATA_DIRECTORY


@pytest.fixture(scope="session")
def train_data_path() -> Path:
    return RAW_DATA_DIRECTORY / "datahow_interview_train_data.csv"


@pytest.fixture(scope="session")
def test_data_path() -> Path:
    return RAW_DATA_DIRECTORY / "datahow_interview_test_data.csv"


@pytest.fixture(scope="session")
def train_targets_path() -> Path:
    return RAW_DATA_DIRECTORY / "datahow_interview_train_targets.csv"


@pytest.fixture(scope="session")
def test_targets_template_path() -> Path:
    """The placeholder targets file. Real test targets arrive at interview time."""
    return RAW_DATA_DIRECTORY / "datahow_interview_test_targets-TEMPLATE.csv"


@pytest.fixture(scope="session")
def train_runs(train_data_path: Path) -> list[ExperimentRun]:
    return load_runs(train_data_path)


@pytest.fixture(scope="session")
def test_runs(test_data_path: Path) -> list[ExperimentRun]:
    return load_runs(test_data_path)


@pytest.fixture(scope="session")
def train_targets(train_targets_path: Path) -> dict[str, float]:
    return load_targets(train_targets_path)
