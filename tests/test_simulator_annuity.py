"""Behavioral tests for the new annuity account type in ``simulate``.

Covers all four lump-sum/monthly modes for both qualified and
non-qualified contracts, plus the validation guard against
non_qualified + rollover_pretax.
"""

from __future__ import annotations

import pytest

from tax_optimizer import Config, Inputs
from tax_optimizer.inputs import AnnuityInputs, StartingBalances
from tax_optimizer.simulator import simulate
from tax_optimizer.tax.state import CA, NY, STATELESS


def _row(df, age):
    """Return the row where ``spouse_a_age == age`` (single-row select)."""
    sub = df[df["spouse_a_age"] == age]
    assert len(sub) == 1, f"expected exactly one row at age {age}, got {len(sub)}"
    return sub.iloc[0]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestAnnuityValidation:
    def test_non_qualified_with_rollover_pretax_raises(self):
        with pytest.raises(ValueError, match="Non-qualified annuity"):
            Inputs(
                annuity=AnnuityInputs(
                    tax_kind="non_qualified",
                    cost_basis=10_000,
                    lump_sum_mode="rollover_pretax",
                )
            )

    def test_bad_tax_kind_raises(self):
        with pytest.raises(ValueError, match="tax_kind"):
            Inputs(annuity=AnnuityInputs(tax_kind="roth"))  # type: ignore[arg-type]

    def test_bad_lump_sum_mode_raises(self):
        with pytest.raises(ValueError, match="lump_sum_mode"):
            Inputs(
                annuity=AnnuityInputs(lump_sum_mode="dump")  # type: ignore[arg-type]
            )

    def test_zero_payout_years_raises(self):
        with pytest.raises(ValueError, match="expected_payout_years"):
            Inputs(annuity=AnnuityInputs(expected_payout_years=0))

    def test_negative_basis_raises(self):
        with pytest.raises(ValueError, match="cost_basis"):
            Inputs(annuity=AnnuityInputs(cost_basis=-5))


# ---------------------------------------------------------------------------
# Defaults: no annuity → no annuity columns nonzero
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_no_annuity_columns_zero(self):
        df = simulate(Config(), Inputs())
        assert "annuity_balance" in df.columns
        assert (df["annuity_balance"] == 0).all()
        assert (df["annuity_taxable"] == 0).all()
        assert (df["annuity_tax_free"] == 0).all()
        assert (df["annuity_lump_sum"] == 0).all()


# ---------------------------------------------------------------------------
# Monthly mode — qualified
# ---------------------------------------------------------------------------


class TestQualifiedMonthly:
    def test_payments_are_fully_ordinary_at_marginal(self):
        # Single filer at 70 with a $200k qualified annuity paying
        # $12k/yr. Every dollar of the payment should be taxable
        # ordinary income (no tax_free portion).
        inp = Inputs(
            household_kind="single",
            spouse_a_age_start=70,
            annuity=AnnuityInputs(
                balance_today=200_000,
                monthly_at_start=1_000,
                start_age=70,
                tax_kind="qualified",
            ),
        )
        df = simulate(Config(horizon_age=80), inp)
        r70 = _row(df, 70)
        assert r70["annuity_taxable"] == pytest.approx(12_000)
        assert r70["annuity_tax_free"] == 0.0
        assert r70["annuity_lump_sum"] == 0.0


# ---------------------------------------------------------------------------
# Monthly mode — non-qualified
# ---------------------------------------------------------------------------


class TestNonQualifiedMonthly:
    def test_exclusion_ratio_splits_first_year(self):
        # 50k basis, 12k/yr, 20-yr expected → exclusion = 50k/240k = 0.2083.
        # First payment: tax_free = 12k * 0.2083 ≈ 2500;
        # taxable ≈ 9500.
        inp = Inputs(
            household_kind="single",
            spouse_a_age_start=70,
            annuity=AnnuityInputs(
                balance_today=200_000,
                cost_basis=50_000,
                monthly_at_start=1_000,
                start_age=70,
                tax_kind="non_qualified",
                expected_payout_years=20,
            ),
        )
        df = simulate(Config(horizon_age=85), inp)
        r70 = _row(df, 70)
        assert r70["annuity_tax_free"] == pytest.approx(2_500.0, rel=1e-3)
        assert r70["annuity_taxable"] == pytest.approx(9_500.0, rel=1e-3)

    def test_basis_remaining_decreases_while_payments_flow(self):
        inp = Inputs(
            household_kind="single",
            spouse_a_age_start=70,
            annuity=AnnuityInputs(
                balance_today=200_000,
                cost_basis=50_000,
                monthly_at_start=1_000,
                start_age=70,
                tax_kind="non_qualified",
                expected_payout_years=20,
            ),
        )
        df = simulate(Config(horizon_age=92), inp)
        rows = [_row(df, age) for age in range(70, 92)]
        # While the contract is still paying (balance > 0 at start of
        # year), basis must strictly decrease each year — until it
        # hits zero. The contract may exhaust before the basis does
        # in scenarios where the balance is too small relative to
        # the payment stream; we only assert decrease up to that
        # point.
        for prev, cur in zip(rows, rows[1:]):
            paying = prev["annuity_balance"] > 0
            if paying and prev["annuity_basis_remaining"] > 0:
                assert (
                    cur["annuity_basis_remaining"] < prev["annuity_basis_remaining"]
                ), f"basis didn't decrease at age {cur['spouse_a_age']}"
        # First-year payment drained exactly $2,500 of basis (50k/20).
        assert rows[0]["annuity_basis_remaining"] == pytest.approx(47_500.0)

    def test_payments_become_fully_taxable_after_basis_exhausted(self):
        # 50k basis with `expected_payout_years=5` makes the exclusion
        # ratio drain the full basis in 5 payments at ~$10k/yr each.
        # By year 10 every dollar of the (smaller, ongoing) payment
        # is fully taxable.
        inp = Inputs(
            household_kind="single",
            spouse_a_age_start=70,
            annuity=AnnuityInputs(
                balance_today=400_000,
                cost_basis=50_000,
                monthly_at_start=2_000,
                start_age=70,
                tax_kind="non_qualified",
                expected_payout_years=5,
            ),
        )
        df = simulate(Config(horizon_age=82), inp)
        r80 = _row(df, 80)
        # Long after basis is exhausted.
        assert r80["annuity_basis_remaining"] == 0.0
        assert r80["annuity_tax_free"] == 0.0
        assert r80["annuity_taxable"] > 0.0


# ---------------------------------------------------------------------------
# Lump-sum: rollover_pretax (qualified only — non-qualified is forbidden)
# ---------------------------------------------------------------------------


class TestRolloverPretax:
    def test_balance_moves_to_pretax_ira_no_tax(self):
        inp = Inputs(
            household_kind="single",
            spouse_a_age_start=60,
            annuity=AnnuityInputs(
                balance_today=200_000,
                start_age=65,
                tax_kind="qualified",
                lump_sum_mode="rollover_pretax",
            ),
        )
        df = simulate(Config(horizon_age=75), inp)
        r64 = _row(df, 64)
        r65 = _row(df, 65)
        assert r65["annuity_lump_sum"] > 0
        assert r65["annuity_lump_sum_event"] == "rollover_pretax"
        assert r65["annuity_balance"] == 0.0
        assert r65["annuity_taxable"] == 0.0
        assert r65["early_distribution_penalty"] == 0.0
        # Pretax IRA jumped substantially due to the rollover (net of
        # whatever spending withdrawal happens that year). We don't
        # require the full lump-sum amount because some flows out for
        # spending; the qualitative jump is what we're testing.
        delta = r65["pretax_a_balance"] - r64["pretax_a_balance"]
        assert delta > 100_000

    def test_no_subsequent_payments(self):
        # Once the contract is rolled, no future annuity income.
        inp = Inputs(
            household_kind="single",
            spouse_a_age_start=60,
            annuity=AnnuityInputs(
                balance_today=200_000,
                monthly_at_start=1_000,  # would normally pay $12k/yr
                start_age=65,
                tax_kind="qualified",
                lump_sum_mode="rollover_pretax",
            ),
        )
        df = simulate(Config(horizon_age=75), inp)
        post = df[df["spouse_a_age"] >= 66]
        assert (post["annuity_taxable"] == 0).all()
        assert (post["annuity_balance"] == 0).all()


# ---------------------------------------------------------------------------
# Lump-sum: cash, qualified
# ---------------------------------------------------------------------------


class TestCashQualified:
    def test_full_balance_ordinary_no_penalty_post_60(self):
        inp = Inputs(
            household_kind="single",
            spouse_a_age_start=70,
            annuity=AnnuityInputs(
                balance_today=200_000,
                start_age=70,
                tax_kind="qualified",
                lump_sum_mode="cash",
            ),
        )
        df = simulate(Config(horizon_age=80), inp)
        r70 = _row(df, 70)
        assert r70["annuity_lump_sum"] == pytest.approx(200_000)
        assert r70["annuity_taxable"] == pytest.approx(200_000)
        assert r70["annuity_tax_free"] == 0.0
        assert r70["early_distribution_penalty"] == 0.0
        assert r70["annuity_lump_sum_event"] == "cash"

    def test_pre_60_triggers_10_percent_surtax_on_full_balance(self):
        inp = Inputs(
            household_kind="single",
            spouse_a_age_start=55,
            annuity=AnnuityInputs(
                balance_today=200_000,
                start_age=55,
                tax_kind="qualified",
                lump_sum_mode="cash",
            ),
        )
        df = simulate(Config(horizon_age=75), inp)
        r55 = _row(df, 55)
        assert r55["annuity_taxable"] == pytest.approx(200_000)
        # 10% on full taxable principal ($200k) = $20k.
        assert r55["early_distribution_penalty"] == pytest.approx(20_000)


# ---------------------------------------------------------------------------
# Lump-sum: cash, non-qualified
# ---------------------------------------------------------------------------


class TestCashNonQualified:
    def test_only_gain_taxable_basis_returned_tax_free_post_60(self):
        inp = Inputs(
            household_kind="single",
            spouse_a_age_start=70,
            annuity=AnnuityInputs(
                balance_today=200_000,
                cost_basis=50_000,
                start_age=70,
                tax_kind="non_qualified",
                lump_sum_mode="cash",
            ),
        )
        df = simulate(Config(horizon_age=80), inp)
        r70 = _row(df, 70)
        assert r70["annuity_lump_sum"] == pytest.approx(200_000)
        assert r70["annuity_taxable"] == pytest.approx(150_000)  # gain
        assert r70["annuity_tax_free"] == pytest.approx(50_000)  # basis
        assert r70["early_distribution_penalty"] == 0.0

    def test_pre_60_surtax_on_gain_only(self):
        # Pre-59½: §72(q) hits the gain only. Basis return is exempt.
        inp = Inputs(
            household_kind="single",
            spouse_a_age_start=55,
            annuity=AnnuityInputs(
                balance_today=200_000,
                cost_basis=50_000,
                start_age=55,
                tax_kind="non_qualified",
                lump_sum_mode="cash",
            ),
        )
        df = simulate(Config(horizon_age=75), inp)
        r55 = _row(df, 55)
        assert r55["annuity_taxable"] == pytest.approx(150_000)
        assert r55["annuity_tax_free"] == pytest.approx(50_000)
        # 10% on the $150k gain only = $15k.
        assert r55["early_distribution_penalty"] == pytest.approx(15_000)


# ---------------------------------------------------------------------------
# Pre-payout growth
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# State-tax routing for annuity income (Critical fix:
# `state-tax-annuity-routing`).
#
# Pre-fix the simulator routed `annuity_taxable` through the federal
# solver but `_state_tax_fn` and the conversion-capacity / post-cascade
# state-tax recomputes ignored it. CA's `state_regime` would therefore
# under-tax a household whose only ordinary income that year was an
# annuity payment; the marginal state rate also dropped out of the
# Roth-conversion liquidity guard. NY's $20k per-filer exclusion
# (§612(c)(3-a)) likewise excluded annuity from the eligible pool.
# ---------------------------------------------------------------------------


class TestAnnuityStateTaxRouting:
    def test_ca_taxes_annuity_as_ordinary_income(self):
        # Single filer, CA resident, no other income. A $12k qualified
        # annuity payment is fully taxable as ordinary income at both
        # the federal AND state level (CA RTC §17071 conformity to IRC
        # §61(a)(9)). Pre-fix the state-tax line stayed at $0 because
        # `annuity_taxable` was never threaded into `state_tax`.
        inp = Inputs(
            household_kind="single",
            spouse_a_age_start=70,
            annuity=AnnuityInputs(
                balance_today=200_000,
                monthly_at_start=1_000,
                start_age=70,
                tax_kind="qualified",
            ),
        )
        cfg = Config(horizon_age=80, state_regime=CA)
        df = simulate(cfg, inp)
        r70 = _row(df, 70)
        # Federal regression: annuity is recognized.
        assert r70["annuity_taxable"] == pytest.approx(12_000)
        # State tax must be > 0 for a CA resident with $12k of annuity
        # income (after CA standard deduction the $12k - ~$5,540 ≈
        # $6,460 of taxable ordinary; CA's 1% slab alone yields ~$65
        # at minimum).
        assert r70["state_tax"] > 0.0, (
            "CA state tax must be non-zero on annuity income"
        )

    def test_stateless_unchanged(self):
        # STATELESS regime is the no-tax control: state_tax stays at 0
        # regardless of annuity income.
        inp = Inputs(
            household_kind="single",
            spouse_a_age_start=70,
            annuity=AnnuityInputs(
                balance_today=200_000,
                monthly_at_start=1_000,
                start_age=70,
                tax_kind="qualified",
            ),
        )
        cfg = Config(horizon_age=80, state_regime=STATELESS)
        df = simulate(cfg, inp)
        r70 = _row(df, 70)
        assert r70["state_tax"] == 0.0


class TestNYRetirementExclusionWithAnnuity:
    def test_ny_per_filer_exclusion_covers_annuity(self):
        # Direct unit test of the state-tax engine with annuity
        # routing. We call `state_tax` twice with identical income
        # except for `annuity_taxable`; under NY's $20k exclusion the
        # second call should return $0 of state tax (the entire
        # annuity payment falls inside the per-filer cap), while
        # the first call (no annuity) is also $0 — and a third call
        # with $30k of annuity should owe NY tax on $10k (the
        # un-excluded slice).
        #
        # This isolates the routing fix from spending-driven taxable
        # withdrawals that pollute the simulator-level state tax in
        # an end-to-end run.
        from tax_optimizer.tax.state import state_tax

        kw_common = dict(
            regime=NY,
            filing_status="single",
            wages_box1=0.0,
            interest=0.0,
            ordinary_div=0.0,
            qualified_div=0.0,
            ltcg=0.0,
            pension=0.0,
            pretax_withdrawal=0.0,
            roth_conversion=0.0,
            social_security=0.0,
            ss_taxable_federal=0.0,
            age_a=70,
            alive_a=True,
        )
        no_annuity = state_tax(annuity_taxable=0.0, **kw_common)
        small_annuity = state_tax(
            annuity_taxable=20_000.0,
            annuity_per_spouse=(20_000.0, 0.0),
            **kw_common,
        )
        big_annuity = state_tax(
            annuity_taxable=30_000.0,
            annuity_per_spouse=(30_000.0, 0.0),
            **kw_common,
        )
        # No annuity → no state tax (only $0 of state taxable).
        assert no_annuity["state_tax"] == 0.0
        # $20k of annuity is fully excluded ($20k cap, age 70 ≥ 60).
        assert small_annuity["state_tax"] == 0.0
        # $30k of annuity → only $10k flows through the NY rate
        # schedule. After the NY single-filer std-deduction ($8k)
        # this lands at $2k of state-taxable income — taxed at the
        # NY 4% slab → $80 of NY tax.
        assert big_annuity["state_tax"] == pytest.approx(80.0, abs=5.0)

    def test_ny_exclusion_per_spouse_independent(self):
        # Two-filer (MFJ) sanity check: each spouse gets their own
        # $20k cap. Spouse A gets $20k of annuity, spouse B gets
        # $20k of pension — both should be fully excluded.
        from tax_optimizer.tax.state import state_tax

        result = state_tax(
            regime=NY,
            filing_status="mfj",
            wages_box1=0.0,
            interest=0.0,
            ordinary_div=0.0,
            qualified_div=0.0,
            ltcg=0.0,
            pension=20_000.0,
            pension_per_spouse=(0.0, 20_000.0),
            pretax_withdrawal=0.0,
            roth_conversion=0.0,
            social_security=0.0,
            ss_taxable_federal=0.0,
            annuity_taxable=20_000.0,
            annuity_per_spouse=(20_000.0, 0.0),
            age_a=70,
            age_b=70,
            alive_a=True,
            alive_b=True,
        )
        # Both buckets fully sheltered → no NY tax (after std
        # deduction the $0 of remaining ordinary income → $0 tax).
        assert result["state_tax"] == 0.0


# ---------------------------------------------------------------------------
# Survivor balance scaling (Critical fix: `annuity-survivor-balance`).
#
# Pre-fix, after spouse A's death the contract paid `payment *
# pension_survivor_pct` (correct) but `state.annuity_balance` was
# drained by the full `payment` (wrong). With a 50% J&S election the
# contract balance silently exhausted twice as fast as the survivor
# was actually receiving — a half-year mortality scenario could
# misrepresent terminal contract value by tens of thousands of
# dollars.
# ---------------------------------------------------------------------------


class TestAnnuitySurvivorBalance:
    def test_balance_drains_at_survivor_pace(self):
        # Joint household with spouse A dying at year_offset=2. The
        # contract pays $12k/yr until A dies, then $6k/yr (50% J&S)
        # thereafter. Pre-fix the balance dropped at $12k/yr in
        # every post-death year too. We assert the balance drains
        # at the post-death rate by comparing the year-over-year
        # delta in the contract balance.
        from tax_optimizer.mortality import Mortality

        inp = Inputs(
            spouse_a_age_start=70,
            spouse_b_age_start=68,
            spouse_a_retire_age=70,
            spouse_b_retire_age=68,
            annuity=AnnuityInputs(
                balance_today=200_000,
                monthly_at_start=1_000,
                start_age=70,
                tax_kind="qualified",
            ),
        )
        cfg = Config(
            horizon_age=85,
            mortality=Mortality(year_of_death_a=2, pension_survivor_pct=0.5),
        )
        df = simulate(cfg, inp)
        # Year 1 (both alive): balance drops by ~$12k.
        # Year 2 (A dies, B survives): payment = $6k, balance drop ~$6k.
        # Year 3 (B alone, A dead): payment = $6k, balance drop ~$6k.
        balances = df.set_index("spouse_a_age")["annuity_balance"]
        # Year 0 → 1: full $12k drop (both alive).
        delta_y1 = balances.iloc[0] - balances.iloc[1]
        assert delta_y1 == pytest.approx(12_000, abs=1)
        # Year 1 → 2: A dies in year 2, payment scales to 50% → $6k drop.
        delta_y2 = balances.iloc[1] - balances.iloc[2]
        assert delta_y2 == pytest.approx(6_000, abs=1), (
            f"survivor-year balance drop must equal scaled payment, got {delta_y2:.2f}"
        )
        # Year 2 → 3: still $6k/yr (post-death survivor pace).
        delta_y3 = balances.iloc[2] - balances.iloc[3]
        assert delta_y3 == pytest.approx(6_000, abs=1)

    def test_total_dollars_paid_match_balance_drained(self):
        # Stronger invariant: across the full payout horizon, the
        # cumulative cash actually received by the household must
        # equal the total amount drained from the contract balance.
        # (Pre-fix the balance was over-drained by 2× the survivor
        # haircut every post-death year.)
        from tax_optimizer.mortality import Mortality

        inp = Inputs(
            spouse_a_age_start=70,
            spouse_b_age_start=68,
            spouse_a_retire_age=70,
            spouse_b_retire_age=68,
            annuity=AnnuityInputs(
                balance_today=200_000,
                monthly_at_start=1_000,
                start_age=70,
                tax_kind="qualified",
            ),
        )
        cfg = Config(
            horizon_age=95,
            mortality=Mortality(year_of_death_a=2, pension_survivor_pct=0.5),
        )
        df = simulate(cfg, inp)
        # Cumulative payments = sum of the annuity_taxable column
        # (contract is qualified, so no tax-free split).
        total_paid = df["annuity_taxable"].sum()
        # Total drained = starting balance minus ending balance.
        drained = (
            df["annuity_balance"].iloc[0] + 12_000
            - df["annuity_balance"].iloc[-1]
        )
        # The +$12k offset accounts for the year-0 in-flow (year 0
        # already shows the post-payment balance in the dataframe).
        # We just compare total_paid vs drained directly.
        total_drained_raw = (
            inp.annuity.balance_today - df["annuity_balance"].iloc[-1]
        )
        # Conservation: dollars out of the contract = dollars
        # received by the household.
        assert total_paid == pytest.approx(total_drained_raw, abs=1.0)


class TestPrePayoutGrowth:
    def test_balance_grows_at_configured_rate(self):
        inp = Inputs(
            household_kind="single",
            spouse_a_age_start=60,
            annuity=AnnuityInputs(
                balance_today=100_000,
                start_age=70,
                growth_rate=0.05,
                tax_kind="qualified",
            ),
        )
        df = simulate(Config(horizon_age=75), inp)
        r60 = _row(df, 60)
        r61 = _row(df, 61)
        # End-of-year balance should be ~5% higher.
        assert r61["annuity_balance"] == pytest.approx(
            r60["annuity_balance"] * 1.05, rel=1e-6
        )
