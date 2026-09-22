import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phyroute.feeder import build_feeder  # noqa: E402


@pytest.fixture(scope="session")
def feeder():
    return build_feeder()


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(123)
