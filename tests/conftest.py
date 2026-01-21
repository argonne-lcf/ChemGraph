import pytest
import warnings
from ase import Atoms

# Configure pytest-asyncio
#pytest_plugins = ("pytest_asyncio",)


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

def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-llm"):
        # --run-llm given in cli: do not skip llm tests
        return
    skip_llm = pytest.mark.skip(reason="need --run-llm option to run")
    for item in items:
        if "llm" in item.keywords:
            item.add_marker(skip_llm)