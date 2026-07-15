"""Shared fixtures: the example spreadsheets used by the Wolfram
``SpreadsheetTrace`` resource-function tests, stored locally under
``tests/data``."""

from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data"


def _data_fixture(number: int):
    @pytest.fixture(scope="session", name=f"file{number}")
    def fixture() -> Path:
        return DATA_DIR / f"example_{number:02d}.xlsx"

    return fixture


for _n in range(1, 11):
    globals()[f"_file{_n}_fixture"] = _data_fixture(_n)
