import pytest
import numpy as np

@pytest.fixture
def sample_config():
    """Basic configuration for testing."""
    return {
        'mesh_size': 32,
        'tolerance': 1e-6,
        'max_iterations': 1000
    }

@pytest.fixture
def sample_data():
    """Generate sample test data."""
    np.random.seed(42)
    return np.random.randn(50, 2)