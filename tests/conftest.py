"""Shared test helpers. See README.md section 6 (Tests)."""

from __future__ import annotations

import numpy as np
import pytest


def call(fn, *args, **kwargs):
    """Run student code; skip (not fail) while the stub is unimplemented.

    Every test that touches _sample/_distance/_embed goes through this, so a
    fresh checkout collects and skips cleanly, and each test flips to a real
    pass or fail as you implement.
    """
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.skip("stub not implemented yet")


@pytest.fixture
def rng() -> np.random.Generator:
    """Fresh deterministic generator per test: results never depend on test order."""
    return np.random.default_rng(0)
