"""Tests for the ``year_table_scope`` knob added to ``build_action_report``.

Before this change, §7 ("Year-by-year withdrawal & conversion plan")
always filtered the simulator's DataFrame down to retirement years
only. Users running the CLI couldn't see pre-retirement AGI / federal-
tax / state-tax behaviour and were missing context for any plan that
front-loads conversions, mega-backdoor contributions, or other working-
year tax events.

The new default is ``"full"`` — the table now spans the entire
simulated horizon, with a visual ``RETIRE @ N`` marker row dividing
accumulation from drawdown. Passing ``year_table_scope="retirement"``
(or ``--year-table-scope retirement`` on the CLI) restores the legacy
compact view.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from tax_optimizer import Config, Inputs, simulate
from tax_optimizer.metrics import summarize
from tax_optimizer.report import build_action_report
from tax_optimizer.results import StrategyResult


def _min_sens_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "param": "spouse_a_roth_401k_pct",
                "low_value": 0.0,
                "high_value": 1.0,
                "delta_low": 1000.0,
                "delta_high": 0.0,
                "swing": 1000.0,
            }
        ]
    )


def _two_strategy_results(
    cfg: Config, inputs: Inputs
) -> dict[str, StrategyResult]:
    df = simulate(cfg, inputs)
    s0 = StrategyResult(cfg=cfg, inputs=inputs, df=df, summary=summarize(df))
    s3_inp = replace(inputs, spouse_a_roth_401k_pct=0.5)
    s3_df = simulate(cfg, s3_inp)
    s3 = StrategyResult(
        cfg=cfg, inputs=s3_inp, df=s3_df, summary=summarize(s3_df)
    )
    return {"S0_baseline": s0, "S3_optimizer": s3}


def _section_7(md: str) -> str:
    return md.split("## 7.")[1].split("## 8.")[0]


def _age_rows(section7: str) -> list[int]:
    """Extract integer ages from §7 data rows.

    Skips the header row, the markdown alignment row, and the
    ``RETIRE @ N`` marker row.
    """
    ages: list[int] = []
    for line in section7.splitlines():
        if not line.startswith("| "):
            continue
        if "---" in line:
            continue
        first_cell = line.split("|")[1].strip()
        if not first_cell or first_cell.startswith("Age") or first_cell.startswith("**"):
            continue
        try:
            ages.append(int(first_cell))
        except ValueError:
            continue
    return ages


# ============================================================== default = full


class TestDefaultIsFullHorizon:
    def test_default_includes_pre_retirement_years(self) -> None:
        cfg = Config()
        # Working horizon: start 50, retire 60 → 10 pre-retirement years.
        inp = Inputs(
            spouse_a_age_start=50,
            spouse_b_age_start=50,
            spouse_a_retire_age=60,
            spouse_b_retire_age=60,
        )
        results = _two_strategy_results(cfg, inp)
        md = build_action_report(cfg, inp, results, _min_sens_df(), 1.0)
        ages = _age_rows(_section_7(md))
        assert ages, "expected at least one §7 data row"
        # Spans full horizon: must include both start age and at least
        # one retirement-era age.
        assert min(ages) == 50
        assert max(ages) >= 60

    def test_default_emits_retirement_marker_row(self) -> None:
        cfg = Config()
        inp = Inputs(
            spouse_a_age_start=50,
            spouse_b_age_start=50,
            spouse_a_retire_age=60,
            spouse_b_retire_age=60,
        )
        results = _two_strategy_results(cfg, inp)
        md = build_action_report(cfg, inp, results, _min_sens_df(), 1.0)
        section7 = _section_7(md)
        # Marker row references the earlier retire age across both spouses.
        assert "**RETIRE @ 60**" in section7

    def test_default_caption_mentions_full_horizon(self) -> None:
        cfg, inp = Config(), Inputs()
        results = _two_strategy_results(cfg, inp)
        md = build_action_report(cfg, inp, results, _min_sens_df(), 1.0)
        section7 = _section_7(md)
        assert "Full horizon" in section7
        assert "Retirement years only" not in section7

    def test_default_caption_advertises_opt_out(self) -> None:
        # The caption should tell the user how to get back to the
        # legacy compact view.
        cfg, inp = Config(), Inputs()
        results = _two_strategy_results(cfg, inp)
        md = build_action_report(cfg, inp, results, _min_sens_df(), 1.0)
        section7 = _section_7(md)
        assert 'year_table_scope="retirement"' in section7
        assert "--year-table-scope retirement" in section7


# ============================================================ retirement scope


class TestRetirementScopeReproducesLegacyBehavior:
    def test_only_retirement_years_present(self) -> None:
        cfg = Config()
        inp = Inputs(
            spouse_a_age_start=50,
            spouse_b_age_start=50,
            spouse_a_retire_age=60,
            spouse_b_retire_age=60,
        )
        results = _two_strategy_results(cfg, inp)
        md = build_action_report(
            cfg, inp, results, _min_sens_df(), 1.0,
            year_table_scope="retirement",
        )
        ages = _age_rows(_section_7(md))
        assert ages, "expected at least one §7 data row"
        # Every visible age must be at or past retirement age.
        assert min(ages) >= 60

    def test_retirement_caption_text(self) -> None:
        cfg, inp = Config(), Inputs()
        results = _two_strategy_results(cfg, inp)
        md = build_action_report(
            cfg, inp, results, _min_sens_df(), 1.0,
            year_table_scope="retirement",
        )
        section7 = _section_7(md)
        assert "Retirement years only" in section7
        assert "Full horizon" not in section7

    def test_retirement_scope_skips_marker_row(self) -> None:
        # The "RETIRE @ N" divider is a full-mode affordance only —
        # there's nothing to divide when only retirement rows are
        # rendered.
        cfg = Config()
        inp = Inputs(
            spouse_a_age_start=50,
            spouse_b_age_start=50,
            spouse_a_retire_age=60,
            spouse_b_retire_age=60,
        )
        results = _two_strategy_results(cfg, inp)
        md = build_action_report(
            cfg, inp, results, _min_sens_df(), 1.0,
            year_table_scope="retirement",
        )
        assert "RETIRE @" not in _section_7(md)

    def test_retirement_count_lt_full_count(self) -> None:
        # Sanity: switching scopes should change the row count on any
        # scenario where the start age is younger than retire age.
        cfg = Config()
        inp = Inputs(
            spouse_a_age_start=50,
            spouse_b_age_start=50,
            spouse_a_retire_age=60,
            spouse_b_retire_age=60,
        )
        results = _two_strategy_results(cfg, inp)
        full = build_action_report(
            cfg, inp, results, _min_sens_df(), 1.0,
            year_table_scope="full",
        )
        retire = build_action_report(
            cfg, inp, results, _min_sens_df(), 1.0,
            year_table_scope="retirement",
        )
        assert len(_age_rows(_section_7(full))) > len(_age_rows(_section_7(retire)))


# ============================================================== input validation


class TestInvalidScope:
    def test_unknown_scope_rejected(self) -> None:
        cfg, inp = Config(), Inputs()
        results = _two_strategy_results(cfg, inp)
        with pytest.raises(ValueError, match="year_table_scope"):
            build_action_report(
                cfg, inp, results, _min_sens_df(), 1.0,
                year_table_scope="weekly",
            )


# ============================================================== CLI flag wiring


class TestCLIFlag:
    """Confirm the argparse surface accepts the new flag."""

    def test_flag_default_is_full(self) -> None:
        from tax_optimizer.__main__ import _build_parser

        args = _build_parser().parse_args([])
        assert args.year_table_scope == "full"

    def test_flag_accepts_retirement(self) -> None:
        from tax_optimizer.__main__ import _build_parser

        args = _build_parser().parse_args(["--year-table-scope", "retirement"])
        assert args.year_table_scope == "retirement"

    def test_flag_rejects_unknown(self) -> None:
        from tax_optimizer.__main__ import _build_parser

        with pytest.raises(SystemExit):
            _build_parser().parse_args(["--year-table-scope", "weekly"])
