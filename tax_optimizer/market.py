"""Market return models + asset location.

`MarketModel` is a yearly return source with three implementations:

  * `DeterministicModel` — constant returns. Backward-compatible with
    the old scalar `nominal_growth_rate` (use the same number for both
    equity and bond and you get the original behavior, modulo the
    asset-location split).

  * `LognormalModel` — independent yearly draws from a lognormal
    distribution parameterized by long-run mean and stdev. Cheap and
    captures the dispersion of single-year returns; misses
    autocorrelation and fat tails.

  * `BootstrapModel` — block-bootstrap resampling from a user-supplied
    historical-returns table. Preserves the empirical fat tails and
    short-run autocorrelation of real history. Defaults to a
    1928-2023 S&P-500 + 10y-Treasury sample (Damodaran NYU Stern;
    embedded as constants below for reproducibility).

`AssetLocation` describes the equity / bond split *per account type*.
This is where the "asset location matters" dimension comes in: bonds
in pretax (tax-deferred ordinary), equities in Roth (tax-free
compounding), most growth in taxable + a small bond sleeve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class AssetMix:
    equity_pct: float
    label: str = ""

    @property
    def bond_pct(self) -> float:
        return 1.0 - self.equity_pct

    def annual_return(self, equity_r: float, bond_r: float) -> float:
        return self.equity_pct * equity_r + self.bond_pct * bond_r


@dataclass
class AssetLocation:
    """Per-account equity allocation.

    Defaults reflect a textbook "asset-located" portfolio:
      * pretax  : 40% equity / 60% bond  — bonds get sheltered from the
                  ordinary-income tax line they would otherwise feed.
      * roth    : 100% equity            — maximize tax-free compounding.
      * taxable : 80% equity / 20% bond  — moderate efficiency; muni
                  bonds in the bond sleeve in real life.
      * hsa     : 80% equity             — long-horizon HSA-as-401k
                  treatment; bonds matter less because of the
                  triple-tax-advantaged shelter.
    """

    pretax: AssetMix = field(default_factory=lambda: AssetMix(0.40, "pretax_balanced"))
    roth: AssetMix = field(default_factory=lambda: AssetMix(1.00, "roth_aggressive"))
    taxable: AssetMix = field(default_factory=lambda: AssetMix(0.80, "taxable_growth"))
    hsa: AssetMix = field(default_factory=lambda: AssetMix(0.80, "hsa_growth"))

    @classmethod
    def uniform(cls, equity_pct: float = 0.6) -> "AssetLocation":
        """All accounts at the same equity %. Used to mimic the old
        single-scalar growth-rate behavior."""
        m = AssetMix(equity_pct, f"uniform_{int(equity_pct * 100)}_eq")
        return cls(pretax=m, roth=m, taxable=m, hsa=m)


@runtime_checkable
class MarketModel(Protocol):
    """A source of yearly (equity, bond) returns for the simulator."""

    def begin_path(self, n_years: int, rng: np.random.Generator) -> None:
        """Called once per simulation path to allow stateful models to
        sample a full N-year path up front (e.g. block bootstrap)."""
        ...

    def returns(self, year_offset: int) -> tuple[float, float]:
        """Return (equity_r, bond_r) for the given year of the path."""
        ...


# ---------------------------------------------------------------------------
# 1. Deterministic
# ---------------------------------------------------------------------------


@dataclass
class DeterministicModel:
    """Constant equity / bond returns. Equivalent to the old behavior
    when `equity == bond == cfg.nominal_growth_rate` and asset location
    is uniform."""

    equity: float = 0.07
    bond: float = 0.04

    def begin_path(self, n_years: int, rng: np.random.Generator) -> None:
        return None

    def returns(self, year_offset: int) -> tuple[float, float]:
        return self.equity, self.bond


# ---------------------------------------------------------------------------
# 2. Lognormal
# ---------------------------------------------------------------------------


@dataclass
class LognormalModel:
    """Independent yearly draws.

    Defaults loosely match 1928-2023 trailing US data:
      equity_mu = 9.6%, equity_sigma = 19.5%   (~S&P 500 total return)
      bond_mu   = 4.6%, bond_sigma   = 7.7%   (10y Treasury total return)
    """

    equity_mu: float = 0.07
    equity_sigma: float = 0.18
    bond_mu: float = 0.04
    bond_sigma: float = 0.06

    def __post_init__(self) -> None:
        self._equity_path: np.ndarray | None = None
        self._bond_path: np.ndarray | None = None

    def begin_path(self, n_years: int, rng: np.random.Generator) -> None:
        # Sample arithmetic returns directly from a normal (the lognormal
        # naming is conventional but in practice a normal on returns is
        # what most planners use for one-year horizons).
        self._equity_path = rng.normal(self.equity_mu, self.equity_sigma, size=n_years)
        self._bond_path = rng.normal(self.bond_mu, self.bond_sigma, size=n_years)

    def returns(self, year_offset: int) -> tuple[float, float]:
        assert self._equity_path is not None and self._bond_path is not None
        return float(self._equity_path[year_offset]), float(self._bond_path[year_offset])


# ---------------------------------------------------------------------------
# 3. Block-bootstrap from historical 1928-2023 data
# ---------------------------------------------------------------------------

# Annual total returns, S&P 500 and 10y Treasury, 1928-2023.
# Source: Damodaran, "Annual Returns on Stock, T.Bonds and T.Bills" (NYU Stern),
# accessed Q4 2024. Light rounding to 4 decimals.
_HIST_EQUITY: tuple[float, ...] = (
    0.4381, -0.0830, -0.2512, -0.4384, -0.0864,  0.4998,  -0.0119,  0.4674,  0.3194, -0.3534,  # 1928-37
    0.2928, -0.0110, -0.1067, -0.1277,  0.1917,  0.2506,   0.1903,  0.3582, -0.0843,  0.0520,  # 1938-47
    0.0570,  0.1830,  0.3081,  0.2368,  0.1815,  -0.0121,  0.5256, 0.3260,  0.0744, -0.1046,   # 1948-57
    0.4372,  0.1206,  0.0034,  0.2664, -0.0881,  0.2240,   0.1642, 0.1240, -0.0997,  0.2380,  # 1958-67
    0.1086, -0.0824,  0.0356,  0.1418,  0.1879, -0.1431,  -0.2590, 0.3700,  0.2376, -0.0720,  # 1968-77
    0.0656,  0.1844,  0.3242, -0.0491,  0.2141,  0.2251,   0.0627, 0.3216,  0.1847,  0.0523,  # 1978-87
    0.1681,  0.3149, -0.0317,  0.3055,  0.0762,  0.1008,   0.0132, 0.3758,  0.2296,  0.3336,  # 1988-97
    0.2858,  0.2104, -0.0910, -0.1189, -0.2210,  0.2868,   0.1088, 0.0491,  0.1579,  0.0549,  # 1998-2007
    -0.3700, 0.2646,  0.1506,  0.0211,  0.1600,  0.3239,   0.1369, 0.0138,  0.1196,  0.2183,  # 2008-17
    -0.0438, 0.3149,  0.1840,  0.2871, -0.1811,  0.2629,                                       # 2018-23
)

_HIST_BOND: tuple[float, ...] = (
    0.0084,  0.0420,  0.0454,  -0.0256,  0.0879,  0.0186,   0.0796,  0.0447,  0.0502,  0.0138,  # 1928-37
    0.0428,  0.0449,  0.0298,   0.0055,  0.0298,  0.0204,   0.0282,  0.1085,  -0.0110, -0.0010, # 1938-47
    0.0438,  0.0451,  0.0008,  -0.0407,  0.0117,  0.0364,   0.0082,  -0.0142, -0.0233,  0.0680, # 1948-57
    -0.0220, 0.0552, 0.1175,  0.0192,  0.0405,  0.0114,    0.0421,  0.0265,   0.0062,  -0.0571, # 1958-67
    0.0316,  -0.0097, 0.1668, 0.0989,  0.0265,  0.0535,    0.0613,  0.0788,   0.1487,  0.0196,  # 1968-77
    -0.0076, -0.0117, -0.0395, 0.0185, 0.4080,  0.0028,    0.1531,  0.2548,   0.2419,  -0.0244, # 1978-87
    0.0816,  0.1788,  0.0612,  0.1554,  0.0746,  0.1797,  -0.0780,  0.2393,   0.0136,  0.0959,  # 1988-97
    0.1456,  -0.0837, 0.1672,  0.0556,  0.1525,  0.0036,    0.0428,  0.0288,   0.0196,  0.1018,  # 1998-2007
    0.2016,  -0.1112, 0.0844, 0.1604,  0.0297,  -0.0910,    0.1075,  0.0128,   0.0069,   0.0287, # 2008-17
    -0.0002, 0.0964,  0.1133,  -0.0442, -0.1769, 0.0387,                                          # 2018-23
)


@dataclass
class BootstrapModel:
    """Block-bootstrap resampling from a historical (equity, bond) table.

    `block_size` controls the autocorrelation captured. Block size 1 is
    a plain bootstrap (independent years); a block size of 5-10 is a
    common choice that preserves runs of bull/bear years. Default 5.

    Optionally pass your own `equity_history` / `bond_history` arrays.
    Defaults to embedded 1928-2023 US data.
    """

    block_size: int = 5
    equity_history: tuple[float, ...] = _HIST_EQUITY
    bond_history: tuple[float, ...] = _HIST_BOND

    def __post_init__(self) -> None:
        self._equity_path: np.ndarray | None = None
        self._bond_path: np.ndarray | None = None

    def begin_path(self, n_years: int, rng: np.random.Generator) -> None:
        eq_hist = np.asarray(self.equity_history)
        bd_hist = np.asarray(self.bond_history)
        n_hist = len(eq_hist)
        n_blocks = (n_years + self.block_size - 1) // self.block_size
        starts = rng.integers(0, n_hist - self.block_size + 1, size=n_blocks)
        eq_path = np.concatenate([eq_hist[s : s + self.block_size] for s in starts])
        bd_path = np.concatenate([bd_hist[s : s + self.block_size] for s in starts])
        self._equity_path = eq_path[:n_years]
        self._bond_path = bd_path[:n_years]

    def returns(self, year_offset: int) -> tuple[float, float]:
        assert self._equity_path is not None and self._bond_path is not None
        return float(self._equity_path[year_offset]), float(self._bond_path[year_offset])
