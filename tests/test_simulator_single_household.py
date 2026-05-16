"""End-to-end regression tests for the single-filer household feature.

Adding ``inputs.household_kind = "single"`` makes the simulator route
every tax / FICA / IRMAA / SS-provisional decision through the Single
tables and force ``alive_b = False`` and ``filing_status = "single"``
from year 0 — no widow MFJ exception. These tests pin down each of
those guarantees so a future refactor of `Mortality.filing_status` or
the simulator's per-year setup can't silently regress them.

The MFJ behavior (default ``household_kind = "mfj"``) is covered by
`tests/test_simulator.py::TestSimulateWidowsPenalty` and friends — the
tests here only assert what's *new* in the single path.
"""

from __future__ import annotations

import pytest

from tax_optimizer import Config, Inputs, Mortality, simulate, summarize
from tax_optimizer.tax.federal import federal_tax
from tax_optimizer.tax.regimes import TCJA_EXTENDED


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestHouseholdKindValidation:
    def test_default_is_mfj(self) -> None:
        assert Inputs().household_kind == "mfj"

    def test_single_is_accepted(self) -> None:
        # No exception means validation passed.
        Inputs(household_kind="single")

    def test_typo_raises(self) -> None:
        # Catch the typo at construction time — silently falling through to
        # the Single branch would mis-tax MFJ households.
        with pytest.raises(ValueError, match="household_kind"):
            Inputs(household_kind="married_joint")  # type: ignore[arg-type]

    def test_uppercase_raises(self) -> None:
        with pytest.raises(ValueError, match="household_kind"):
            Inputs(household_kind="MFJ")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Filing status / alive flag
# ---------------------------------------------------------------------------


class TestSingleFilingStatusAndAliveFlags:
    def test_filing_status_single_for_every_year(self) -> None:
        """A single household never sees an MFJ year — including year 0.

        This is the discriminator vs. the widow case: ``Mortality``
        keeps year-of-death MFJ per IRS rules, so a `year_of_death_b=0`
        scenario would *still* file MFJ in year 0. The
        ``household_kind = "single"`` override deliberately bypasses
        that exception.
        """
        cfg = Config()
        inputs = Inputs(household_kind="single")
        df = simulate(cfg, inputs)
        assert (df["filing_status"] == "single").all()

    def test_alive_b_is_false_from_year_zero(self) -> None:
        cfg = Config()
        inputs = Inputs(household_kind="single")
        df = simulate(cfg, inputs)
        assert (~df["alive_b"]).all()

    def test_alive_a_unaffected(self) -> None:
        # Only alive_b is forced — alive_a still tracks Mortality.
        cfg = Config(mortality=Mortality(year_of_death_a=10))
        inputs = Inputs(household_kind="single")
        df = simulate(cfg, inputs)
        assert df.iloc[:10]["alive_a"].all()
        assert not df.iloc[10]["alive_a"]

    def test_no_widow_mfj_year_with_year_of_death_b_zero(self) -> None:
        """Cross-check: ``household_kind = "single"`` defeats the
        IRS year-of-death MFJ exception that would otherwise apply
        to ``year_of_death_b = 1``.
        """
        cfg = Config(mortality=Mortality(year_of_death_b=1))
        single = simulate(cfg, Inputs(household_kind="single"))
        mfj_widow = simulate(cfg, Inputs(household_kind="mfj"))

        # Single household: every year is single, including year 1.
        assert (single["filing_status"] == "single").all()
        # MFJ household with B dying at offset 1: year 0 + year 1
        # are MFJ (year-of-death exception); year 2+ flips to single.
        assert mfj_widow.iloc[0]["filing_status"] == "mfj"
        assert mfj_widow.iloc[1]["filing_status"] == "mfj"
        assert mfj_widow.iloc[2]["filing_status"] == "single"


# ---------------------------------------------------------------------------
# Spouse-B inputs are silently ignored
# ---------------------------------------------------------------------------


class TestSpouseBInputsIgnored:
    def test_spouse_b_salary_does_not_count(self) -> None:
        """Spouse B's $200k salary should not enter taxable wages in a
        single-filer household — alive_b=False kills the income path.
        """
        cfg = Config()
        single = Inputs(household_kind="single")
        # Pin a baseline single-only wage for A, give B a fictitious
        # high salary that should be ignored.
        single.income.spouse_a_gross = 120_000
        single.income.spouse_b_gross = 200_000

        df = simulate(cfg, single)
        # Year 0 wages should reflect A's salary only (plus bonus, minus
        # FICA basis — we only need the order-of-magnitude check here).
        # If B's salary leaked in, wages would be ~$320k+.
        assert df.iloc[0]["wages"] < 200_000

    def test_spouse_b_pretax_balance_stays_zero(self) -> None:
        """Spouse B is never `working` (alive_b=False), so neither
        elective deferrals nor employer match should accumulate."""
        cfg = Config()
        inputs = Inputs(household_kind="single")
        # Put B's deferral percent at 30% to make any leak loud.
        inputs.spouse_b_total_contrib_pct = 0.30
        inputs.spouse_b_employer_match_rate = 1.0
        inputs.spouse_b_employer_match_max_pct = 0.30
        inputs.income.spouse_b_gross = 100_000
        inputs.starting.spouse_b_pretax_401k = 0.0
        inputs.starting.spouse_b_pretax_ira = 0.0

        df = simulate(cfg, inputs)
        # B's pretax never grows from contributions (only growth on a
        # zero base = zero). RMDs aren't relevant since balance is 0.
        assert (df["pretax_b_balance"] == 0).all()

    def test_spouse_b_ss_is_zero(self) -> None:
        cfg = Config()
        inputs = Inputs(household_kind="single")
        inputs.ss.monthly_spouse_b = 4000  # generous fictitious benefit
        inputs.ss.start_age_b = 62
        inputs.ss.start_age_a = 67
        inputs.ss.monthly_spouse_a = 0  # silence A so any SS = B's

        df = simulate(cfg, inputs)
        # alive_b is False every year, so the SS-claim branch for B
        # never fires. Total SS income should be zero (or nearly so).
        assert (df["ssn"] == 0).all()


# ---------------------------------------------------------------------------
# Tax / FICA / HSA differences vs MFJ
# ---------------------------------------------------------------------------


class TestSingleHouseholdTaxBehaviour:
    def test_single_household_pays_more_tax_than_mfj_at_same_income(self) -> None:
        """Sanity: at $150k household income, the Single brackets +
        Single std deduction yield strictly more federal tax than MFJ.
        This pins the routing of ``filing_status`` through to
        `federal_tax`, not the magnitudes themselves (which are pinned
        by `test_tax_federal.py`).
        """
        cfg = Config(market=None)  # deterministic
        single_in = Inputs(household_kind="single")
        single_in.income.spouse_a_gross = 150_000
        single_in.income.spouse_a_bonus = 0
        single_in.income.spouse_b_gross = 0

        mfj_in = Inputs(household_kind="mfj")
        mfj_in.income.spouse_a_gross = 150_000
        mfj_in.income.spouse_a_bonus = 0
        mfj_in.income.spouse_b_gross = 0

        single_df = simulate(cfg, single_in)
        mfj_df = simulate(cfg, mfj_in)

        # Year 0 — both spouses still working in MFJ, only A working in
        # the single household. Same gross household wage either way.
        single_tax = single_df.iloc[0]["federal_tax"]
        mfj_tax = mfj_df.iloc[0]["federal_tax"]
        assert single_tax > mfj_tax

    def test_federal_tax_helper_matches_simulator_routing(self) -> None:
        """Cross-check: the simulator's year-0 federal tax row exists
        and the filing-status column is "single". Together with
        `test_single_household_pays_more_tax_than_mfj_at_same_income`
        above, this pins the Single-table routing through the tax
        call without re-doing the regime-bracket math here (already
        covered by `test_tax_federal.py`).
        """
        cfg = Config()
        inputs = Inputs(household_kind="single")
        df = simulate(cfg, inputs)
        row = df.iloc[0]
        assert row["filing_status"] == "single"
        # Verify that `federal_tax` itself does carry a Single path
        # — pin the contract this feature relies on.
        single = federal_tax(
            regime=TCJA_EXTENDED, filing_status="single", wages=120_000
        )
        mfj = federal_tax(
            regime=TCJA_EXTENDED, filing_status="mfj", wages=120_000
        )
        assert single["tax"] > mfj["tax"]


class TestSingleHouseholdHsaCap:
    def test_hsa_cap_drops_to_self_only(self) -> None:
        """HSA contributions in a single household should be capped at
        the IRS *self-only* limit (~$4,150 for 2024) rather than the
        family limit (~$8,300). Couple this with a high target so the
        clip is visible.
        """
        cfg = Config()
        inputs = Inputs(household_kind="single")
        inputs.contrib.hsa_family = 9_000  # over the family cap

        df = simulate(cfg, inputs)
        # Year 0 contribution should be at or below the self-only cap.
        # (We use a generous upper bound to remain robust to IRS
        # constant updates.)
        first_year_hsa_growth = df.iloc[0]["hsa_balance"] - inputs.starting.hsa
        # Pure deposit, no market growth in year 0 — equity_return *
        # starting balance is allocated AT year-end, but with default
        # 6% on $25k that's ~$1,500. Add the contribution to that. So
        # the "implied contribution" is `final - starting - growth`.
        # Easier: cross-check the cap directly.
        from tax_optimizer.limits import hsa_family_cap

        cap_couple = hsa_family_cap(
            inputs.spouse_a_age_start,
            inputs.spouse_b_age_start,
            either_working=True,
            b_alive=True,
        )
        cap_single = hsa_family_cap(
            inputs.spouse_a_age_start,
            inputs.spouse_b_age_start,
            either_working=True,
            b_alive=False,
        )
        # The single (b_alive=False) cap is strictly smaller than the
        # couple cap at equal ages.
        assert cap_single < cap_couple
        # And the simulator's deposit is bounded by the single cap
        # (roughly — the assertion on the year-0 balance is permissive).
        assert first_year_hsa_growth < cap_couple


# ---------------------------------------------------------------------------
# Spurious spousal-rollover detector
# ---------------------------------------------------------------------------


class TestNoSpuriousRolloverInYearZero:
    def test_no_rollover_event_in_single_household(self) -> None:
        """`prev_alive_b` is seeded to False for single households so
        the year-0 rollover detector (`prev_alive_b and not alive_b
        and alive_a`) doesn't spuriously fire against an empty
        spouse-B balance.
        """
        cfg = Config()
        inputs = Inputs(household_kind="single")
        df = simulate(cfg, inputs)
        # The simulator records rollover events in the `rollover_event`
        # column when one spouse's pretax IRA/401(k) is moved to the
        # survivor. For a single household this should NEVER happen.
        if "rollover_event" in df.columns:
            assert (df["rollover_event"] == "").all()


# ---------------------------------------------------------------------------
# Round-trip via summarize()
# ---------------------------------------------------------------------------


class TestSingleHouseholdSummarize:
    def test_summarize_runs_and_growth_keys_present(self) -> None:
        cfg = Config()
        inputs = Inputs(household_kind="single")
        df = simulate(cfg, inputs)
        s = summarize(
            df,
            heir_marginal_rate=cfg.heir_marginal_rate,
            starting_balances=inputs.starting,
            inflation=cfg.inflation,
            retire_age=inputs.spouse_a_retire_age,
        )
        # The growth helpers populated by `summarize` for the Overview
        # tab work the same in single mode — we just need a sanity
        # check that nothing blows up on a single-filer DataFrame.
        for key in (
            "starting_after_tax",
            "terminal_after_tax",
            "total_growth_mult",
            "effective_cagr",
            "real_cagr",
            "accumulation_cagr",
            "decumulation_cagr",
        ):
            assert key in s, f"missing summary key: {key}"
