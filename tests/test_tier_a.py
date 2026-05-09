"""Tests for Tier-A correctness fixes.

Each `Test*` class targets one fix in the original review:

  * T1+T6 — SS COLA + per-spouse claim age + actuarial PIA scaling
  * T2    — Bracket / std-deduction / IRMAA inflation-indexing
  * T3    — FICA on W-2 wages
  * T4    — Spousal pretax rollover on death (survivor's RMDs continue
            on the inherited balance)
  * T5    — Conversion eligibility window expansion + RMD-first ordering

These are deliberately separated from `test_simulator_v2.py` so the
v2-era engine fixes (HSA, caps, deficit cascade) and Tier-A bug fixes
remain independently auditable.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tax_optimizer import (
    Config,
    Inputs,
    Mortality,
    SocialSecurity,
    SpendingProfile,
    StartingBalances,
    simulate,
)
from tax_optimizer.inputs import claim_age_factor
from tax_optimizer.payroll import (
    ADDITIONAL_MEDICARE_RATE,
    ADDITIONAL_MEDICARE_THRESHOLD,
    MEDICARE_RATE,
    OASDI_RATE,
    OASDI_WAGE_BASE_2026,
    fica_employee,
)
from tax_optimizer.tax.regimes import TCJA_EXTENDED


# ---------------------------------------------------------------------------
# T1 + T6 — SS COLA + per-spouse claim age + actuarial PIA scaling
# ---------------------------------------------------------------------------


class TestClaimAgeFactor:
    def test_at_fra_is_unity(self) -> None:
        assert claim_age_factor(67, 67) == pytest.approx(1.0)

    def test_age_70_credits_24pct(self) -> None:
        # 3 years past FRA at +8%/yr = +24% (cap at 70).
        assert claim_age_factor(70, 67) == pytest.approx(1.24)

    def test_above_70_caps_at_70(self) -> None:
        # No further credits beyond age 70 even if claim is delayed.
        assert claim_age_factor(75, 67) == pytest.approx(claim_age_factor(70, 67))

    def test_age_62_haircut_30pct(self) -> None:
        # 5 years early: 36 months × 5/9% + 24 months × 5/12% = 20% + 10% = 30%.
        assert claim_age_factor(62, 67) == pytest.approx(0.70)

    def test_below_62_zero(self) -> None:
        assert claim_age_factor(61, 67) == 0.0

    def test_3_years_early_uses_5_9th_bucket(self) -> None:
        # 36 months × 5/9% = 20% reduction.
        assert claim_age_factor(64, 67) == pytest.approx(0.80)


class TestSocialSecurityCOLA:
    def test_ss_inflates_each_year(self) -> None:
        cfg = Config()
        inp = Inputs()
        df = simulate(cfg, inp)
        post = df[df["spouse_a_age"] >= inp.ss.effective_start_age_a]
        post = post[post["alive_a"] & post["alive_b"]]
        # Successive years (both alive) should differ by exactly the
        # COLA factor — cfg.ss_cola_rate (None) ⇒ cfg.inflation.
        ss_values = post["ssn"].values
        if len(ss_values) >= 2:
            ratio = ss_values[1] / ss_values[0]
            assert ratio == pytest.approx(1 + cfg.inflation)

    def test_ss_cola_zero_recovers_flat_nominal(self) -> None:
        # ss_cola_rate=0.0 reproduces the v1 flat-nominal SS behavior,
        # useful as a regression hedge.
        cfg = Config(ss_cola_rate=0.0)
        inp = Inputs()
        df = simulate(cfg, inp)
        both = df[df["alive_a"] & df["alive_b"]]
        post = both[both["spouse_a_age"] >= inp.ss.effective_start_age_a]
        # In flat-nominal mode, every retirement year has the same SS.
        if len(post) >= 2:
            assert np.isclose(post["ssn"].iloc[0], post["ssn"].iloc[-1])


class TestSocialSecurityClaimAge:
    def test_per_spouse_claim_ages_independent(self) -> None:
        # Spouse A claims at 62 (early), Spouse B at 70 (delayed). The
        # year both have begun collecting, household SS = a*0.7 + b*1.24
        # (times COLA factor).
        cfg = Config(ss_cola_rate=0.0)  # disable COLA for clean math
        inp = Inputs(
            ss=SocialSecurity(
                monthly_spouse_a=2_700.0,
                monthly_spouse_b=2_200.0,
                start_age_a=62,
                start_age_b=70,
                fra_a=67,
                fra_b=67,
            ),
        )
        df = simulate(cfg, inp)
        # Year where both are at claim age and both alive:
        both_collecting = df[
            (df["spouse_a_age"] >= 70)
            & (df["spouse_b_age"] >= 70)
            & df["alive_a"]
            & df["alive_b"]
        ]
        if not both_collecting.empty:
            year0 = both_collecting.iloc[0]
            expected = (
                2_700.0 * 12 * claim_age_factor(62, 67)
                + 2_200.0 * 12 * claim_age_factor(70, 67)
            )
            assert year0["ssn"] == pytest.approx(expected)

    def test_legacy_start_age_still_works(self) -> None:
        # Backward-compat: if start_age_a/b are None, both spouses claim
        # at ss.start_age. The per-spouse properties resolve correctly.
        ss = SocialSecurity(start_age=68)
        assert ss.effective_start_age_a == 68
        assert ss.effective_start_age_b == 68

    def test_spouse_a_starts_drawing_at_claim_age(self) -> None:
        cfg = Config(ss_cola_rate=0.0)
        inp = Inputs(
            ss=SocialSecurity(
                start_age_a=62,
                start_age_b=70,
                fra_a=67,
                fra_b=67,
            ),
        )
        df = simulate(cfg, inp)
        before_a = df[df["spouse_a_age"] < 62]
        between = df[(df["spouse_a_age"] >= 62) & (df["spouse_a_age"] < 70)]
        # Before A's claim age: zero SS.
        assert (before_a["ssn"] == 0.0).all()
        # Between: A is collecting (haircut), B isn't yet.
        if not between.empty:
            both = between[between["alive_a"] & between["alive_b"]]
            if not both.empty:
                expected_a_only = (
                    inp.ss.monthly_spouse_a * 12 * claim_age_factor(62, 67)
                )
                assert both["ssn"].iloc[0] == pytest.approx(expected_a_only)


# ---------------------------------------------------------------------------
# T3 — FICA on W-2 wages
# ---------------------------------------------------------------------------


class TestFICA:
    def test_below_oasdi_base(self) -> None:
        # $50k wages: full OASDI + Medicare, no add'l, no cap binding.
        out = fica_employee(50_000.0)
        assert out["oasdi"] == pytest.approx(50_000 * OASDI_RATE)
        assert out["medicare"] == pytest.approx(50_000 * MEDICARE_RATE)
        assert out["additional_medicare"] == 0.0

    def test_above_oasdi_base_caps_oasdi(self) -> None:
        # $300k wages: OASDI capped at the base; Medicare on full.
        out = fica_employee(300_000.0)
        assert out["oasdi"] == pytest.approx(OASDI_WAGE_BASE_2026 * OASDI_RATE)
        assert out["medicare"] == pytest.approx(300_000 * MEDICARE_RATE)
        # Add'l Medicare: 0.9% on amount over $200k threshold.
        addl_expected = (300_000 - ADDITIONAL_MEDICARE_THRESHOLD) * ADDITIONAL_MEDICARE_RATE
        assert out["additional_medicare"] == pytest.approx(addl_expected)

    def test_zero_wages_zero_fica(self) -> None:
        out = fica_employee(0.0)
        assert out["total"] == 0.0

    def test_simulator_subtracts_fica(self) -> None:
        # FICA should appear as a column and be > 0 every working year
        # (default scenario has $95k + $70k W-2 wages).
        df = simulate(Config(), Inputs())
        working = df[df["wages"] > 0]
        assert (working["fica"] > 0).all()
        # Sanity: FICA ≈ 7.65% of wages while well below OASDI base.
        # With the wage base inflating each year, this ratio is stable.
        first = working.iloc[0]
        approx_rate = first["fica"] / first["wages"]
        assert 0.07 <= approx_rate <= 0.08

    def test_fica_zero_after_retirement(self) -> None:
        # No wages after both spouses retire → no FICA.
        df = simulate(Config(), Inputs())
        post = df[
            (df["spouse_a_age"] >= 65)
            & (df["spouse_b_age"] >= 65)
        ]
        assert (post["fica"] == 0.0).all()


# ---------------------------------------------------------------------------
# T2 — Bracket / std-deduction / IRMAA inflation indexing
# ---------------------------------------------------------------------------


class TestBracketIndexing:
    def test_inflated_zero_offset_returns_self(self) -> None:
        # `factor=1.0` shortcut in `inflated()` skips the rebuild.
        assert TCJA_EXTENDED.inflated(1.0) is TCJA_EXTENDED

    def test_inflated_scales_brackets(self) -> None:
        scaled = TCJA_EXTENDED.inflated(1.10)
        for orig, new in zip(
            TCJA_EXTENDED.ord_brackets_mfj, scaled.ord_brackets_mfj
        ):
            assert new[0] == pytest.approx(orig[0] * 1.10)
            assert new[2] == orig[2]  # rate unchanged

    def test_inflated_does_not_scale_niit(self) -> None:
        # NIIT threshold is statutory (unindexed since 2013). Verify it
        # survives an inflation pass unchanged.
        scaled = TCJA_EXTENDED.inflated(1.50)
        assert scaled.niit_threshold_mfj == TCJA_EXTENDED.niit_threshold_mfj
        assert scaled.niit_threshold_single == TCJA_EXTENDED.niit_threshold_single

    def test_inflated_does_not_scale_ss_provisional(self) -> None:
        # SS provisional thresholds are unindexed since 1993.
        scaled = TCJA_EXTENDED.inflated(1.50)
        assert scaled.ss_provisional_mfj == TCJA_EXTENDED.ss_provisional_mfj

    def test_inflated_scales_std_deduction(self) -> None:
        scaled = TCJA_EXTENDED.inflated(1.10)
        assert scaled.std_deduction_mfj == pytest.approx(
            TCJA_EXTENDED.std_deduction_mfj * 1.10
        )

    def test_inflated_scales_irmaa_thresholds_and_surcharges(self) -> None:
        scaled = TCJA_EXTENDED.inflated(1.20)
        for orig, new in zip(
            TCJA_EXTENDED.irmaa_tiers_mfj, scaled.irmaa_tiers_mfj
        ):
            if orig[0] == float("inf"):
                assert new[0] == float("inf")
            else:
                assert new[0] == pytest.approx(orig[0] * 1.20)
            assert new[1] == pytest.approx(orig[1] * 1.20)
            assert new[2] == pytest.approx(orig[2] * 1.20)

    def test_indexing_zero_freezes_brackets(self) -> None:
        # bracket_indexing_rate=0 ⇒ effective_regime is the raw regime.
        cfg = Config(bracket_indexing_rate=0.0)
        for offset in (0, 5, 20, 40):
            assert (
                cfg.effective_regime(offset).std_deduction_mfj
                == cfg.tax_regime.std_deduction_mfj
            )

    def test_default_indexing_follows_inflation(self) -> None:
        cfg = Config()  # bracket_indexing_rate=None ⇒ cfg.inflation
        scaled = cfg.effective_regime(10)
        expected = cfg.tax_regime.std_deduction_mfj * (1 + cfg.inflation) ** 10
        assert scaled.std_deduction_mfj == pytest.approx(expected)

    def test_indexed_tax_lower_than_unindexed(self) -> None:
        # With indexing on, real tax burden over a long horizon is
        # smaller than with brackets frozen at year-0 nominals (because
        # nominal incomes inflate but indexed brackets keep up). The
        # default scenario should clearly demonstrate this.
        df_indexed = simulate(Config(bracket_indexing_rate=None), Inputs())
        df_frozen = simulate(Config(bracket_indexing_rate=0.0), Inputs())
        assert df_indexed["federal_tax"].sum() < df_frozen["federal_tax"].sum()


# ---------------------------------------------------------------------------
# T4 — Spousal pretax rollover on death
# ---------------------------------------------------------------------------


class TestSpousalRollover:
    def test_rollover_on_a_death(self) -> None:
        # Spouse A dies at year 5. Their pretax should be in B's bucket
        # from year 5 onward, and A's bucket should be 0.
        cfg = Config(mortality=Mortality(year_of_death_a=5))
        inp = Inputs()
        df = simulate(cfg, inp)
        # At year 5, A's pretax balance has been moved.
        post = df[df["spouse_a_age"] >= inp.spouse_a_age_start + 5]
        assert (post["pretax_a_balance"] == 0.0).all()
        # B's pretax should jump up.
        pre_death = df.iloc[4]
        first_widow_year = df.iloc[5]
        assert (
            first_widow_year["pretax_b_balance"]
            >= pre_death["pretax_b_balance"]
        )

    def test_rollover_event_recorded(self) -> None:
        cfg = Config(mortality=Mortality(year_of_death_a=10))
        df = simulate(cfg, Inputs())
        # Exactly one rollover event, on year 10.
        events = df[df["spousal_rollover"] == "a_to_b"]
        assert len(events) == 1
        assert events.index[0] == 10

    def test_no_rollover_when_both_alive(self) -> None:
        df = simulate(Config(), Inputs())
        assert (df["spousal_rollover"] == "").all()

    def test_survivor_rmds_continue_on_combined(self) -> None:
        # A dies before A's RMD age; survivor (B) takes RMDs on the
        # combined balance once B reaches RMD age. Without the rollover
        # fix, A's pretax would compound forever and B's RMD would
        # only reflect B's own balance.
        cfg = Config(
            mortality=Mortality(year_of_death_a=15),  # A dies at 65
            rmd_start_age=75,
        )
        inp = Inputs()
        df = simulate(cfg, inp)
        # B reaches RMD age (75) on year offset 25. Pre-rollover-bug
        # would have B's RMD only covering B's own balance; post-fix
        # B's RMD reflects the combined inherited balance.
        post_rmd = df[
            (df["spouse_b_age"] >= cfg.rmd_start_age)
            & ~df["alive_a"]
            & df["alive_b"]
        ]
        if not post_rmd.empty:
            assert (post_rmd["rmd_b"] > 0).all()
            # And A's RMD remains zero (A is dead and A's pretax was
            # rolled over).
            assert (post_rmd["rmd_a"] == 0).all()


# ---------------------------------------------------------------------------
# T5 — Conversion window expansion + RMD-first ordering
# ---------------------------------------------------------------------------


class TestConversionRMDFirst:
    def test_no_conversion_below_target_bracket(self) -> None:
        # Bracket-fill mode self-regulates: if existing taxable income
        # is already past the target bracket cap, headroom = 0 and no
        # conversion fires regardless of working/retired status. With
        # default Inputs ($165k MFJ wages) and a 12% target, TI is well
        # past the 12% bracket top → no conversion in working years.
        cfg = Config(roth_conversion_target_bracket=0.12)
        df = simulate(cfg, Inputs())
        working = df[
            (df["spouse_a_age"] < Inputs().spouse_a_retire_age)
            & (df["spouse_b_age"] < Inputs().spouse_b_retire_age)
        ]
        assert (working["roth_conversion"] == 0.0).all()

    def test_working_year_conversion_at_high_target(self) -> None:
        # Conversely, when working-year TI is BELOW the target bracket
        # cap (e.g. 24% target with $165k wages, MFJ TI ≈ $133k well
        # below the 24% cap of ~$394k), conversions DO fire in working
        # years — and that's correct behavior. Locking yourself into
        # 24% now beats deferring into 32% later. This is precisely
        # what the gap-only gate was wrong to suppress.
        cfg = Config(roth_conversion_target_bracket=0.24)
        df = simulate(cfg, Inputs())
        working = df[
            (df["spouse_a_age"] < Inputs().spouse_a_retire_age)
            & (df["spouse_b_age"] < Inputs().spouse_b_retire_age)
        ]
        # At least some working years should now convert.
        assert (working["roth_conversion"] > 0).any()

    def test_bracket_fill_active_in_gap_years(self) -> None:
        cfg = Config(roth_conversion_target_bracket=0.12)
        inp = Inputs(
            spouse_a_retire_age=55,
            spouse_b_retire_age=55,
            ss=SocialSecurity(
                monthly_spouse_a=0.0,
                monthly_spouse_b=0.0,
                start_age=70,
                fra_a=67,
                fra_b=67,
            ),
        )
        df = simulate(cfg, inp)
        gap = df[
            (df["spouse_a_age"] >= 55)
            & (df["spouse_a_age"] < cfg.rmd_start_age)
            & (df["pretax_balance"] > 0)
        ]
        # In gap years with low income, conversions should fire.
        assert (gap["roth_conversion"] > 0).any()

    def test_post_rmd_conversion_now_legal(self) -> None:
        # With a high enough bracket target and a still-positive pretax
        # balance after the RMD, the bracket-fill mode now fires past
        # rmd_start_age (the legacy gap-only gate was the bug). We use
        # a lower retire age so most of horizon is post-retirement.
        cfg = Config(
            roth_conversion_target_bracket=0.24,
            rmd_start_age=75,
        )
        inp = Inputs(
            spouse_a_age_start=75,  # start at RMD age
            spouse_b_age_start=75,
            spouse_a_retire_age=75,
            spouse_b_retire_age=75,
            starting=replace(
                Inputs().starting,
                spouse_a_pretax_401k=2_000_000.0,
                spouse_b_pretax_401k=2_000_000.0,
            ),
        )
        df = simulate(cfg, inp)
        # First simulated year is at RMD start age. The legacy code
        # would have refused any conversion (a_in_gap requires
        # age < rmd_start_age). The fix unlocks this.
        first = df.iloc[0]
        assert first["roth_conversion"] > 0

    def test_rmd_eats_bracket_headroom_before_conversion(self) -> None:
        # When the year's RMD is already most of the way through the
        # target bracket, the bracket-fill conversion sizing should be
        # strictly smaller than it would be without RMD. We compare:
        # (a) pre-RMD age scenario at the same bracket → larger conv
        # (b) post-RMD age scenario at the same bracket + same balance
        #     → smaller conv
        cfg = Config(
            roth_conversion_target_bracket=0.22,
            rmd_start_age=75,
        )
        inp_pre_rmd = Inputs(
            spouse_a_age_start=70,
            spouse_b_age_start=70,
            spouse_a_retire_age=70,
            spouse_b_retire_age=70,
            ss=SocialSecurity(
                monthly_spouse_a=0.0,
                monthly_spouse_b=0.0,
                start_age=80,  # push SS out so AGI is just portfolio yield
                fra_a=67,
                fra_b=67,
            ),
            starting=replace(
                Inputs().starting,
                spouse_a_pretax_401k=2_000_000.0,
                spouse_b_pretax_401k=2_000_000.0,
                taxable_brokerage=10_000.0,
            ),
        )
        inp_post_rmd = replace(
            inp_pre_rmd,
            spouse_a_age_start=76,
            spouse_b_age_start=76,
            spouse_a_retire_age=76,
            spouse_b_retire_age=76,
        )
        pre = simulate(cfg, inp_pre_rmd).iloc[0]
        post = simulate(cfg, inp_post_rmd).iloc[0]
        assert post["rmd"] > 0
        assert pre["rmd"] == 0
        # Post-RMD year converts strictly less because RMD ate the
        # bracket headroom first.
        assert post["roth_conversion"] < pre["roth_conversion"]
