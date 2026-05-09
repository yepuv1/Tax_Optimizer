"""Shared pytest fixtures for the tax_optimizer test suite."""

from __future__ import annotations

import pytest

from tax_optimizer import Config, Inputs


@pytest.fixture
def cfg() -> Config:
    """Plain `Config()` defaults (deterministic 6%, MFJ, horizon 90)."""
    return Config()


@pytest.fixture
def inputs() -> Inputs:
    """Plain `Inputs()` defaults (50/50 ages, 8/6% deferrals, $85k expenses)."""
    return Inputs()
