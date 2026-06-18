import pytest
import warnings
from ase import Atoms

# Configure pytest-asyncio
#pytest_plugins = ("pytest_asyncio",)

# Test modules that require the optional ``academy`` extra guard themselves with
# ``pytest.importorskip("academy")`` at module top, so they skip cleanly (rather
# than erroring collection) when the extra is not installed.


@pytest.fixture(autouse=True)
def setup_test_env():
    """Setup any test environment variables or configurations needed"""
    # Filter numpy deprecation warnings
    warnings.filterwarnings(
        "ignore",
        message="In future, it will be an error for 'np.bool_' scalars to be interpreted as an index",
        category=DeprecationWarning,
    )
    pass


@pytest.fixture
def simple_h2_molecule():
    """Fixture providing a simple H2 molecule for testing"""
    return Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])

def pytest_addoption(parser):
    parser.addoption(
        "--run-llm", action="store_true", default=False, help="run tests that call LLM APIs"
    )
    parser.addoption(
        "--run-globus-compute", action="store_true", default=False,
        help="run tests that require a live Globus Compute endpoint"
    )

def pytest_collection_modifyitems(config, items):
    skip_llm = None
    if not config.getoption("--run-llm"):
        skip_llm = pytest.mark.skip(reason="need --run-llm option to run")

    skip_globus = None
    if not config.getoption("--run-globus-compute"):
        skip_globus = pytest.mark.skip(reason="need --run-globus-compute option to run")

    for item in items:
        if skip_llm and "llm" in item.keywords:
            item.add_marker(skip_llm)
        if skip_globus and "globus_compute" in item.keywords:
            item.add_marker(skip_globus)