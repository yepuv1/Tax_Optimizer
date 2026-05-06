"""tax_optimizer - retirement tax planner with Monte Carlo, mortality,
asset location, smile-shaped spending, and switchable tax regimes.

Quickstart
----------

>>> from tax_optimizer import Config, Inputs, simulate, simulate_paths
>>> from tax_optimizer.market import LognormalModel
>>> cfg = Config(market=LognormalModel(equity_mu=0.07, equity_sigma=0.18))
>>> mc = simulate_paths(cfg, Inputs(), n_paths=500)
>>> mc.summary()
{'n_paths': 500, 'prob_success': 0.91, ...}

See `tax_optimizer.__main__` for the CLI entry point.
"""

from __future__ import annotations

from .config import Config
from .inputs import (
    CurrentContrib,
    CurrentIncome,
    Inputs,
    PensionInputs,
    SocialSecurity,
    StartingBalances,
)
from .market import (
    AssetLocation,
    AssetMix,
    BootstrapModel,
    DeterministicModel,
    LognormalModel,
    MarketModel,
)
from .metrics import (
    lifetime_irmaa_npv,
    lifetime_tax_npv,
    summarize,
    terminal_after_tax_nw,
)
from .monte_carlo import MonteCarloResult, simulate_paths
from .mortality import Mortality
from .optimizer import (
    BRACKET_CHOICES,
    ObjectiveType,
    optimize_s3,
    x_to_overrides,
)
from .report import build_action_report
from .results import StrategyResult
from .scenario import (
    ScenarioError,
    apply_scenario,
    apply_set_overrides,
    load_scenario_file,
    scenario_to_dict,
)
from .sensitivity import render_actions, render_takeaways, tornado_sensitivity
from .simulator import simulate
from .spending import LongTermCareShock, LumpEvent, SpendingPhase, SpendingProfile
from .tax.regimes import PRE_TCJA_2017, SUNSET_2026, TCJA_EXTENDED, TaxRegime

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "Config",
    "Inputs",
    "StartingBalances",
    "CurrentIncome",
    "CurrentContrib",
    "PensionInputs",
    "SocialSecurity",
    "MarketModel",
    "DeterministicModel",
    "LognormalModel",
    "BootstrapModel",
    "AssetMix",
    "AssetLocation",
    "Mortality",
    "SpendingProfile",
    "SpendingPhase",
    "LumpEvent",
    "LongTermCareShock",
    "TaxRegime",
    "TCJA_EXTENDED",
    "PRE_TCJA_2017",
    "SUNSET_2026",
    "simulate",
    "simulate_paths",
    "MonteCarloResult",
    "optimize_s3",
    "x_to_overrides",
    "BRACKET_CHOICES",
    "ObjectiveType",
    "StrategyResult",
    "summarize",
    "terminal_after_tax_nw",
    "lifetime_tax_npv",
    "lifetime_irmaa_npv",
    "tornado_sensitivity",
    "render_actions",
    "render_takeaways",
    "ScenarioError",
    "load_scenario_file",
    "apply_scenario",
    "apply_set_overrides",
    "scenario_to_dict",
    "build_action_report",
]
