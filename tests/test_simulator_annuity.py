"""Behavioral tests for the new annuity account type in ``simulate``.

Covers all four lump-sum/monthly modes for both qualified and
non-qualified contracts, plus the validation guard against
non_qualified + rollover_pretax.
"""

from __future__ import annotations

import pytest

from tax_optimizer import Config, Inputs
from tax_optimizer.inputs import AnnuityInputs
from tax_optimizer.simulator import simulate


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
