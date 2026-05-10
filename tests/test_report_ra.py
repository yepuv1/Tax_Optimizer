"""Tests for Tier R-A improvements to `build_action_report`.

R-A added five pieces of behavior to the action-plan markdown:

  1. A TL;DR section at the top with verdict + lever diff + risk reading.
  2. An "Assumptions driving this plan" subsection inside §1.
  3. A side-by-side comparison of all 4 canonical strategies in §3.
  4. Honest "at boundary in tested range" labelling for §5 tornado knobs
     whose tested range yields no positive delta on either side.
  5. Near-zero floats render as `—` in §7 (no more confusing `$0` cells).

Each test below pins exactly one of those behaviors so regressions
identify the responsible change.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from tax_optimizer import (
    Config,
    Inputs,
    LognormalModel,
    Mortality,
    SocialSecurity,
    StartingBalances,
    simulate,
)
from tax_optimizer.metrics import summarize
from tax_optimizer.report import (
    _assumptions_block,
    _lever_changes,
    build_action_report,
)
from tax_optimizer.results import StrategyResult


def _four_strategy_results(cfg: Config, inputs: Inputs) -> dict[str, StrategyResult]:
    """Build the canonical S0/S1/S2/S3 result dict. S3 here is a
    hand-picked "moderately different" cfg so the lever-diff path
    exercises rather than collapsing to baseline."""
    base = (cfg, inputs)
    s1 = (cfg, replace(inputs, spouse_a_roth_401k_pct=1.0, spouse_b_roth_401k_pct=1.0))
    s2 = (replace(cfg, roth_conversion_target_bracket=0.22), inputs)
    s3 = (
        replace(cfg, roth_conversion_target_bracket=0.24),
        replace(inputs, spouse_a_roth_401k_pct=0.75),
    )
    out: dict[str, StrategyResult] = {}
    for name, (c, i) in [
        ("S0_baseline", base),
        ("S1_all_roth", s1),
        ("S2_bracket_fill_22", s2),
        ("S3_optimizer", s3),
    ]:
        df = simulate(c, i)
        out[name] = StrategyResult(
            cfg=c, inputs=i, df=df, summary=summarize(df, heir_marginal_rate=c.heir_marginal_rate)
        )
    return out


def _sens_df(deltas_high: list[float], deltas_low: list[float]) -> pd.DataFrame:
    """Build a 3-row tornado frame with caller-controlled deltas so
    each §5 path (both-negative, higher-wins, lower-wins) can be
    exercised directly."""
    rows = []
    names = ["spouse_a_roth_401k_pct", "spouse_b_roth_401k_pct", "inflation"]
    for name, dl, dh in zip(names, deltas_low, deltas_high):
        rows.append({
            "param": name,
            "low_value": 0.0,
            "high_value": 1.0 if "pct" in name else 0.03,
            "delta_low": dl,
            "delta_high": dh,
            "swing": max(abs(dl), abs(dh)),
        })
    return pd.DataFrame(rows)


# =====================================================================
# R-A.1 — TL;DR section
# =====================================================================


class TestTLDRSection:
    def test_tldr_header_present(self) -> None:
        cfg, inp = Config(), Inputs()
        results = _four_strategy_results(cfg, inp)
        sens = _sens_df([100, 100, 100], [-100, -100, -100])
        report = build_action_report(cfg, inp, results, sens, 1.0)
        assert "## TL;DR" in report
        assert "**Verdict:**" in report

    def test_tldr_reports_positive_delta_vs_s0(self) -> None:
        cfg, inp = Config(), Inputs()
        results = _four_strategy_results(cfg, inp)
        winner = max(results, key=lambda n: results[n].summary["terminal_after_tax"])
        sens = _sens_df([100], [100])  # single-row tornado, valid
        sens = _sens_df([100, 100, 100], [-100, -100, -100])
        report = build_action_report(cfg, inp, results, sens, 1.0)
        # Verdict line should contain "beats" when S3 wins by any margin.
        if (
            winner != "S0_baseline"
            and results[winner].summary["terminal_after_tax"]
            > results["S0_baseline"].summary["terminal_after_tax"]
        ):
            assert "beats" in report

    def test_tldr_lists_changed_levers_when_optimizer_differs(self) -> None:
        cfg, inp = Config(), Inputs()
        results = _four_strategy_results(cfg, inp)
        sens = _sens_df([100, 100, 100], [-100, -100, -100])
        report = build_action_report(cfg, inp, results, sens, 1.0)
        winner_name = max(results, key=lambda n: results[n].summary["terminal_after_tax"])
        if winner_name == "S3_optimizer":
            assert "Levers the optimizer wants you to change" in report
            assert "Roth conversion target bracket" in report

    def test_tldr_says_no_changes_when_optimizer_matches_baseline(self) -> None:
        cfg, inp = Config(), Inputs()
        results = {
            "S0_baseline": StrategyResult(
                cfg=cfg, inputs=inp, df=simulate(cfg, inp),
                summary=summarize(simulate(cfg, inp)),
            ),
            "S3_optimizer": StrategyResult(
                cfg=cfg, inputs=inp, df=simulate(cfg, inp),
                summary=summarize(simulate(cfg, inp)),
            ),
        }
        sens = _sens_df([100, 100, 100], [-100, -100, -100])
        report = build_action_report(cfg, inp, results, sens, 1.0)
        assert "No lever changes recommended" in report

    def test_tldr_risk_readings_use_mc_when_provided(self) -> None:
        from tax_optimizer import simulate_paths
        cfg, inp = Config(), Inputs()
        results = _four_strategy_results(cfg, inp)
        sens = _sens_df([100, 100, 100], [-100, -100, -100])
        mc = simulate_paths(
            replace(cfg, market=LognormalModel(equity_bond_corr=0.1)),
            inp,
            n_paths=20,
            seed=0,
            keep_paths=False,
        )
        report = build_action_report(cfg, inp, results, sens, 1.0, mc=mc)
        assert "P(success)" in report
        assert "CVaR(10%)" in report


# =====================================================================
# R-A.2 — Assumptions block
# =====================================================================


class TestAssumptionsBlock:
    def test_block_renders_all_known_knobs(self) -> None:
        cfg = Config(market=LognormalModel(equity_bond_corr=0.10))
        inp = Inputs()
        lines = _assumptions_block(cfg, inp)
        text = "\n".join(lines)
        assert "Assumptions driving this plan" in text
        assert "Heir marginal tax rate" in text
        assert "Market model" in text
        assert "LognormalModel" in text
        assert "Federal tax regime" in text
        assert "State tax regime" in text
        assert "Mortality" in text
        assert "Step-up in basis at first death" in text
        assert "ACA premium tax credit" in text
        assert "IRMAA MAGI lookback" in text

    def test_block_renders_regime_change_when_configured(self) -> None:
        from tax_optimizer import SUNSET_2026
        cfg = Config(regime_change_year_offset=5, regime_change_target=SUNSET_2026)
        text = "\n".join(_assumptions_block(cfg, Inputs()))
        assert "→" in text
        assert "year 5" in text

    def test_block_renders_no_scheduled_change_when_unset(self) -> None:
        cfg = Config(regime_change_year_offset=None, regime_change_target=None)
        text = "\n".join(_assumptions_block(cfg, Inputs()))
        assert "no scheduled change" in text

    def test_lognormal_market_summary_includes_mu_sigma_rho(self) -> None:
        cfg = Config(market=LognormalModel(
            equity_mu=0.07, equity_sigma=0.18,
            bond_mu=0.045, bond_sigma=0.07,
            equity_bond_corr=0.15,
        ))
        text = "\n".join(_assumptions_block(cfg, Inputs()))
        assert "equity μ=7.0%" in text
        assert "ρ=0.15" in text


# =====================================================================
# R-A.3 — §3 four-strategy comparison
# =====================================================================


class TestFourStrategyComparison:
    def test_all_four_strategy_columns_render(self) -> None:
        cfg, inp = Config(), Inputs()
        results = _four_strategy_results(cfg, inp)
        sens = _sens_df([100, 100, 100], [-100, -100, -100])
        report = build_action_report(cfg, inp, results, sens, 1.0)
        # The four-strategy header row uses the human-readable form
        # (underscores → spaces); guard against just one of those.
        for col in ["S0 baseline", "S1 all roth", "S2 bracket fill 22", "S3 optimizer"]:
            assert col in report

    def test_winner_value_is_bolded_in_terminal_row(self) -> None:
        cfg, inp = Config(), Inputs()
        results = _four_strategy_results(cfg, inp)
        sens = _sens_df([100, 100, 100], [-100, -100, -100])
        report = build_action_report(cfg, inp, results, sens, 1.0)
        # Find the terminal-NW row.
        for line in report.splitlines():
            if line.startswith("| Terminal after-tax NW"):
                assert "**" in line  # winning cell wrapped in bold
                break
        else:
            raise AssertionError("Terminal NW row not found")

    def test_single_strategy_falls_back_to_compact_row(self) -> None:
        cfg, inp = Config(), Inputs()
        df = simulate(cfg, inp)
        results = {
            "S3_optimizer": StrategyResult(
                cfg=cfg, inputs=inp, df=df, summary=summarize(df)
            )
        }
        sens = _sens_df([100, 100, 100], [-100, -100, -100])
        report = build_action_report(cfg, inp, results, sens, 1.0)
        # No "Best value" caption when there's only one strategy.
        assert "Best value in each row" not in report


# =====================================================================
# R-A.4 — §5 tornado direction labelling
# =====================================================================


class TestTornadoDirection:
    def test_both_directions_negative_says_at_boundary(self) -> None:
        cfg, inp = Config(), Inputs()
        results = _four_strategy_results(cfg, inp)
        sens = _sens_df(deltas_high=[-100.0, -100.0, -100.0], deltas_low=[-100.0, -100.0, -100.0])
        report = build_action_report(cfg, inp, results, sens, 1.0)
        assert "at boundary in tested range" in report
        # And the misleading "higher (+$0)" must not appear.
        assert "higher (+$0)" not in report

    def test_higher_wins_uses_higher_label(self) -> None:
        cfg, inp = Config(), Inputs()
        results = _four_strategy_results(cfg, inp)
        sens = _sens_df(deltas_high=[5_000.0, 4_000.0, 3_000.0], deltas_low=[-100, -100, -100])
        report = build_action_report(cfg, inp, results, sens, 1.0)
        assert "higher (+$5,000)" in report

    def test_lower_wins_uses_lower_label(self) -> None:
        cfg, inp = Config(), Inputs()
        results = _four_strategy_results(cfg, inp)
        sens = _sens_df(deltas_high=[-100.0, -100, -100], deltas_low=[5_000.0, 4_000, 3_000])
        report = build_action_report(cfg, inp, results, sens, 1.0)
        assert "lower (+$5,000)" in report


# =====================================================================
# R-A.5 — Near-zero floats render as `—` in §7
# =====================================================================


class TestMoneyHelperZeroHandling:
    def test_near_zero_pretax_withdrawal_renders_as_dash(self) -> None:
        """Concretely: configure a scenario where the simulator emits a
        near-zero `pretax_withdrawal` (e.g. retirement years 65-74 in
        example02). Previously these rendered as `$0`; the fixed
        `_money` should render them as `—`."""
        cfg = Config(horizon_age=80)
        inp = Inputs(
            spouse_a_age_start=54,
            spouse_b_age_start=53,
            spouse_a_retire_age=65,
            spouse_b_retire_age=64,
            starting=StartingBalances(
                spouse_a_pretax_401k=1_500_000,
                spouse_b_pretax_401k=200_000,
                taxable_brokerage=500_000,
            ),
            ss=SocialSecurity(
                monthly_spouse_a=3_500,
                monthly_spouse_b=2_000,
                fra_a=67, fra_b=67,
            ),
        )
        df = simulate(cfg, inp)
        results = {
            "S0_baseline": StrategyResult(cfg=cfg, inputs=inp, df=df, summary=summarize(df)),
            "S3_optimizer": StrategyResult(cfg=cfg, inputs=inp, df=df, summary=summarize(df)),
        }
        sens = _sens_df([100, 100, 100], [-100, -100, -100])
        report = build_action_report(cfg, inp, results, sens, 1.0)
        # The §7 withdrawal table should not have `| $0 |` cells.
        # (Other dollar values are fine — we're only asserting that the
        # literal "$0" doesn't appear inside the table portion.)
        section = report.split("## 7.")[1].split("## 8.")[0]
        assert "| $0 |" not in section


# =====================================================================
# Helper / direct unit tests
# =====================================================================


class TestLeverChangesHelper:
    def test_detects_roth_401k_change(self) -> None:
        base_inp = Inputs(spouse_a_roth_401k_pct=0.5)
        new_inp = Inputs(spouse_a_roth_401k_pct=1.0)
        cfg = Config()
        changes = _lever_changes(cfg, base_inp, cfg, new_inp)
        labels = [c["label"] for c in changes]
        assert "Spouse A Roth-401(k) share" in labels

    def test_no_changes_when_inputs_identical(self) -> None:
        cfg, inp = Config(), Inputs()
        assert _lever_changes(cfg, inp, cfg, inp) == []

    def test_detects_ss_claim_age_change(self) -> None:
        base = Inputs(ss=SocialSecurity(start_age_a=67, monthly_spouse_a=3000, monthly_spouse_b=2000, fra_a=67, fra_b=67))
        new = Inputs(ss=SocialSecurity(start_age_a=70, monthly_spouse_a=3000, monthly_spouse_b=2000, fra_a=67, fra_b=67))
        cfg = Config()
        changes = _lever_changes(cfg, base, cfg, new)
        labels = [c["label"] for c in changes]
        assert "Spouse A SS claim age" in labels
