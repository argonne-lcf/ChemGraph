import pytest
import warnings
import numpy as np
from ase import Atoms

# Configure pytest-asyncio
pytest_plugins = ("pytest_asyncio",)


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
