"""Shared pytest config."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def pytest_collection_modifyitems(config, items):
    """Skip tests marked `slow` unless `--runslow` is passed."""
    import pytest

    run_slow = config.getoption("--runslow", default=False)
    skip_slow = pytest.mark.skip(reason="slow test — pass --runslow to enable")
    for item in items:
        if "slow" in item.keywords and not run_slow:
            item.add_marker(skip_slow)


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False,
                     help="run slow tests")
