import numpy as np
import pytest


@pytest.fixture
def toy_2d():
    """Linearly separable 2-D dataset: 3 positive, 3 negative points."""
    X = np.array([
        [2.0, 2.0],
        [2.0, 3.0],
        [3.0, 2.0],
        [-2.0, -2.0],
        [-2.0, -3.0],
        [-3.0, -2.0],
    ], dtype=np.float64)
    y = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0], dtype=np.float64)
    return {"X": X, "y": y}
