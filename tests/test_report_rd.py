"""Tests for Tier R-D.1 — cross-model robustness check.

R-D.1 added two coupled pieces:

  1. `cross_model_check(cfg, inputs, *, n_paths, seed, models)` — runs
     `simulate_paths` under one or more alternative market models and
     returns a `{name: MonteCarloResult}` dict. Default models cover
     the two non-parametric options (`BootstrapModel`,
     `HistoricalSequenceModel`) which complement the parametric
     `LognormalModel` typically passed as the main `mc=` argument.

  2. `build_action_report(..., extra_mc=...)` — new keyword-only
     parameter that, when supplied alongside `mc=`, appends a
     "Cross-model robustness check" sub-section to §4. The block
     renders the main model row plus one row per `extra_mc` entry,
     and emits a callout when any model's P(success) drops below
     90% or when spread across models is ≥ 5pp.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from tax_optimizer import (
    BootstrapModel,
    Config,
    DeterministicModel,
    HistoricalSequenceModel,
    Inputs,
    LognormalModel,
    StartingBalances,
    cross_model_check,
    simulate,
    simulate_paths,
)


def _rich_inputs() -> Inputs:
    """Default `Inputs()` is just barely viable — small market draws can
    push P(success) below 90%. This factory returns a richly funded
    household so the "robust" callout path can be exercised deterministically."""
    return Inputs(
        starting=StartingBalances(
            spouse_a_pretax_401k=2_000_000,
            spouse_b_pretax_401k=1_000_000,
            spouse_a_roth_ira=500_000,
            spouse_b_roth_ira=300_000,
            taxable_brokerage=1_500_000,
            hsa=100_000,
        ),
    )
from tax_optimizer.metrics import summarize
from tax_optimizer.monte_carlo import MonteCarloResult
from tax_optimizer.report import build_action_report
from tax_optimizer.results import StrategyResult


def _min_args(cfg: Config, inputs: Inputs):
    df = simulate(cfg, inputs)
    results = {
        "S0_baseline": StrategyResult(
            cfg=cfg, inputs=inputs, df=df, summary=summarize(df)
        )
    }
    sens_df = pd.DataFrame([{
        "param": "spouse_a_roth_401k_pct",
        "low_value": 0.0, "high_value": 1.0,
        "delta_low": 1000.0, "delta_high": 0.0,
        "swing": 1000.0,
    }])
    return results, sens_df, 1.0


# =====================================================================
# cross_model_check() — the helper itself
# =====================================================================


class TestCrossModelCheck:
    def test_default_models_returns_two_entries(self) -> None:
        cfg, inp = Config(), Inputs()
        out = cross_model_check(cfg, inp, n_paths=10, seed=0)
        assert set(out.keys()) == {"BootstrapModel", "HistoricalSequenceModel"}
        for v in out.values():
            assert isinstance(v, MonteCarloResult)

    def test_explicit_models_override_defaults(self) -> None:
        cfg, inp = Config(), Inputs()
        out = cross_model_check(
            cfg, inp, n_paths=10, seed=0,
            models=[("Bootstrap", BootstrapModel()),
                    ("Lognormal-alt", LognormalModel(equity_mu=0.06))],
        )
        assert set(out.keys()) == {"Bootstrap", "Lognormal-alt"}

    def test_each_run_uses_its_own_market_model(self) -> None:
        cfg, inp = Config(), Inputs()
        out = cross_model_check(
            cfg, inp, n_paths=10, seed=0,
            models=[("Boot", BootstrapModel()), ("Hist", HistoricalSequenceModel())],
        )
        assert isinstance(out["Boot"].cfg.market, BootstrapModel)
        assert isinstance(out["Hist"].cfg.market, HistoricalSequenceModel)

    def test_deterministic_single_path(self) -> None:
        cfg, inp = Config(), Inputs()
        out = cross_model_check(
            cfg, inp, n_paths=1, seed=0,
            models=[("Det", DeterministicModel())],
        )
        assert out["Det"].n_paths == 1

    def test_does_not_mutate_input_cfg(self) -> None:
        cfg = Config(market=LognormalModel(equity_mu=0.07))
        inp = Inputs()
        _ = cross_model_check(cfg, inp, n_paths=5, seed=0)
        # Caller's cfg.market is unchanged after the call.
        assert isinstance(cfg.market, LognormalModel)
        assert cfg.market.equity_mu == 0.07


# =====================================================================
# Report integration: extra_mc keyword
# =====================================================================


class TestReportExtraMC:
    def test_no_extra_mc_means_no_robustness_section(self) -> None:
        cfg, inp = Config(market=LognormalModel()), Inputs()
        mc = simulate_paths(cfg, inp, n_paths=10, seed=0, keep_paths=False)
        results, sens_df, base = _min_args(cfg, inp)
        report = build_action_report(cfg, inp, results, sens_df, base, mc=mc)
        assert "Cross-model robustness check" not in report

    def test_section_renders_when_extra_mc_provided(self) -> None:
        cfg, inp = Config(market=LognormalModel()), Inputs()
        mc = simulate_paths(cfg, inp, n_paths=10, seed=0, keep_paths=False)
        extra = cross_model_check(cfg, inp, n_paths=10, seed=0)
        results, sens_df, base = _min_args(cfg, inp)
        report = build_action_report(
            cfg, inp, results, sens_df, base, mc=mc, extra_mc=extra
        )
        assert "Cross-model robustness check" in report
        # Main row labels current model with "(current)" tag.
        assert "(current)" in report
        # Each extra model name should appear in the rendered table.
        for name in extra.keys():
            assert name in report

    def test_section_silent_when_mc_is_none_even_with_extra(self) -> None:
        # `extra_mc` requires a `mc` to anchor the "current" row; if
        # mc is None we don't render the block.
        cfg, inp = Config(), Inputs()
        extra = cross_model_check(cfg, inp, n_paths=10, seed=0)
        results, sens_df, base = _min_args(cfg, inp)
        report = build_action_report(
            cfg, inp, results, sens_df, base, mc=None, extra_mc=extra
        )
        assert "Cross-model robustness check" not in report

    def test_robust_callout_when_all_models_safe(self) -> None:
        # Rich starting balances → all market models should agree at
        # P(success)=100% so the "all robust" path is the one that fires.
        cfg, inp = Config(market=LognormalModel()), _rich_inputs()
        mc = simulate_paths(cfg, inp, n_paths=30, seed=0, keep_paths=False)
        extra = cross_model_check(cfg, inp, n_paths=30, seed=0)
        results, sens_df, base = _min_args(cfg, inp)
        report = build_action_report(
            cfg, inp, results, sens_df, base, mc=mc, extra_mc=extra
        )
        assert "All models agree the plan is robust" in report

    def test_model_note_uses_canonical_label_when_known(self) -> None:
        cfg, inp = Config(market=LognormalModel()), Inputs()
        mc = simulate_paths(cfg, inp, n_paths=10, seed=0, keep_paths=False)
        extra = cross_model_check(cfg, inp, n_paths=10, seed=0)
        results, sens_df, base = _min_args(cfg, inp)
        report = build_action_report(
            cfg, inp, results, sens_df, base, mc=mc, extra_mc=extra
        )
        # The notes column should mention "Historical-tail" (for
        # BootstrapModel) and "Preserves return sequences" (for
        # HistoricalSequenceModel).
        assert "Historical-tail" in report
        assert "Preserves return sequences" in report


# =====================================================================
# Public API surface — package export
# =====================================================================


class TestPublicExport:
    def test_cross_model_check_importable_from_package_root(self) -> None:
        # Already imported above; this just confirms the __all__ entry
        # is wired up correctly for downstream users.
        from tax_optimizer import cross_model_check as exported
        assert exported is cross_model_check
