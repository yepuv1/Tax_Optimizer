"""Regression tests for the v6.6 mega-backdoor auto-spillover.

Pre-v6.6 the simulator silently dropped any elective-deferral target
above the §402(g) cap. The household kept that excess as taxable
paycheck cash. v6.6 changes that behavior: when
``spouse_*_mega_backdoor_enabled`` is True, the excess auto-routes
into the after-tax 401(k) bucket up to the §415(c) ceiling, and the
explicit ``spouse_*_after_tax_401k_pct`` is then capped to whatever
§415(c) room remains.

Three new diagnostic columns surface the mechanic:

  * ``excess_deferral_{a,b}``         raw target − §402(g) cap
  * ``mega_backdoor_spillover_{a,b}`` auto-routed portion only
  * ``after_tax_target_uncovered_{a,b}`` explicit % that didn't fit

The existing ``mega_backdoor_{a,b}`` column keeps its meaning:
total after-tax routed to Roth = auto + explicit.
"""

from __future__ import annotations

import pytest

from tax_optimizer.config import Config
from tax_optimizer.inputs import (
    CurrentContrib,
    CurrentIncome,
    HealthPremiums,
    Inputs,
    SocialSecurity,
    StartingBalances,
)
from tax_optimizer.limits import (
    ELECTIVE_DEFERRAL_LIMIT,
    SECTION_415C_LIMIT,
    elective_deferral_cap,
)
from tax_optimizer.simulator import simulate


# ---------------------------------------------------------------------------
# Helper: a working couple with a salary high enough to make
# `total_contrib_pct × salary` overshoot the §402(g) cap, and an
# employer match that meaningfully consumes §415(c) room.
# ---------------------------------------------------------------------------


def _single_earner_setup(
    *,
    age=45,
    salary=300_000.0,
    total_contrib_pct=0.30,
    roth_401k_pct=0.5,
    after_tax_pct=0.0,
    mega_enabled=True,
    match_rate=1.0,
    match_max_pct=0.07,
):
    """Spouse A is the only earner. Bonus, HSA, health premiums all
    zeroed so the §415(c) / §402(g) math is the only moving piece."""
    inp = Inputs(
        spouse_a_age_start=age,
        spouse_b_age_start=age,
        spouse_a_retire_age=65,
        spouse_b_retire_age=65,
        spouse_a_total_contrib_pct=total_contrib_pct,
        spouse_b_total_contrib_pct=0.0,
        spouse_a_roth_401k_pct=roth_401k_pct,
        spouse_a_mega_backdoor_enabled=mega_enabled,
        spouse_a_after_tax_401k_pct=after_tax_pct,
        spouse_a_employer_match_rate=match_rate,
        spouse_a_employer_match_max_pct=match_max_pct,
        income=CurrentIncome(
            spouse_a_gross=salary,
            spouse_b_gross=0.0,
            spouse_a_bonus=0.0,
            interest=0.0,
            capital_gains=0.0,
            dividends=0.0,
        ),
        contrib=CurrentContrib(hsa_family=0.0),
        health_premiums=HealthPremiums(),
        ss=SocialSecurity(start_age_a=70, start_age_b=70),
        starting=StartingBalances(taxable_brokerage=100_000.0),
    )
    cfg = Config(horizon_age=age + 5, start_year=2026)
    return cfg, inp


# ---------------------------------------------------------------------------
# 1. Disabled gate: excess stays as cash, no auto-spillover.
# ---------------------------------------------------------------------------


class TestDisabledGate:
    """When `mega_backdoor_enabled = False`, the excess elective-
    deferral target remains as taxable cash (pre-v6.6 behavior).
    The `excess_deferral_a` column still surfaces the raw gap so
    users can see the silent-cap; but `mega_backdoor_spillover_a`
    is zero."""

    def test_excess_visible_but_not_spilled(self) -> None:
        cfg, inp = _single_earner_setup(mega_enabled=False)
        df = simulate(cfg, inp)
        row = df.iloc[0]
        # Target $90k, cap $23.5k → excess $66.5k visible.
        assert row["excess_deferral_a"] == pytest.approx(66_500.0)
        # No spillover (plan doesn't support mega-backdoor).
        assert row["mega_backdoor_spillover_a"] == 0.0
        assert row["mega_backdoor_a"] == 0.0
        assert row["after_tax_target_uncovered_a"] == 0.0
        # Elective deferral correctly capped.
        assert row["elective_deferral_a"] == pytest.approx(23_500.0)


# ---------------------------------------------------------------------------
# 2. Basic spillover: no explicit pct, excess fits in §415(c) room.
# ---------------------------------------------------------------------------


class TestBasicSpillover:
    """Plan supports mega-backdoor, user set no explicit after-tax pct.
    Excess elective-deferral target spills into the after-tax bucket
    up to §415(c) room. Mega_backdoor_a equals the spillover."""

    def test_spillover_equals_min_of_excess_and_room(self) -> None:
        # Use a moderate over-target so excess < §415(c) room.
        # salary $200k × 17% = $34k target; cap $23.5k → excess $10.5k.
        # §415(c) room = $70k - $23.5k - match. Match at 7%/100% on
        # the post-cap deferral pct: effective_pct = $23.5k/$200k = 11.75%,
        # capped at 7% → match = $200k × 7% × 1.0 = $14k.
        # So room = $70k - $23.5k - $14k = $32.5k.
        # excess $10.5k < room $32.5k → spillover = $10.5k.
        cfg, inp = _single_earner_setup(
            salary=200_000.0,
            total_contrib_pct=0.17,
            after_tax_pct=0.0,
            mega_enabled=True,
        )
        df = simulate(cfg, inp)
        row = df.iloc[0]
        assert row["excess_deferral_a"] == pytest.approx(10_500.0, abs=1.0)
        assert row["mega_backdoor_spillover_a"] == pytest.approx(10_500.0, abs=1.0)
        assert row["mega_backdoor_a"] == pytest.approx(10_500.0, abs=1.0)
        assert row["after_tax_target_uncovered_a"] == 0.0


# ---------------------------------------------------------------------------
# 3. Spillover capped at §415(c) room when excess > room.
# ---------------------------------------------------------------------------


class TestSpilloverCappedByRoom:
    """When excess exceeds the §415(c) room, spillover fills the room
    exactly; the remainder stays as taxable cash. This matches the
    pre-v6.6 silent-cap behavior for the un-spillover-able portion."""

    def test_spillover_clips_at_room(self) -> None:
        # Worked example from the plan: salary $300k × 30% = $90k target,
        # cap $23.5k → excess $66.5k. With match $21k (7% × $300k × 1.0):
        # room = $70k - $23.5k - $21k = $25.5k.
        # spillover = min($66.5k, $25.5k) = $25.5k.
        # Remaining excess $41k stays as taxable cash.
        cfg, inp = _single_earner_setup(
            salary=300_000.0,
            total_contrib_pct=0.30,
            after_tax_pct=0.0,
            mega_enabled=True,
        )
        df = simulate(cfg, inp)
        row = df.iloc[0]
        assert row["excess_deferral_a"] == pytest.approx(66_500.0, abs=1.0)
        assert row["mega_backdoor_spillover_a"] == pytest.approx(25_500.0, abs=1.0)
        assert row["mega_backdoor_a"] == pytest.approx(25_500.0, abs=1.0)
        # Invariant: spillover never exceeds §415(c) ceiling - base - match.
        room = SECTION_415C_LIMIT - 23_500.0 - 21_000.0
        assert row["mega_backdoor_spillover_a"] <= room + 1.0


# ---------------------------------------------------------------------------
# 4. Stacking with explicit pct (excess + explicit fits in room).
# ---------------------------------------------------------------------------


class TestStackingExplicit:
    """Auto-spillover stacks on top of the user's explicit
    `after_tax_401k_pct`, capped at the §415(c) ceiling."""

    def test_explicit_stacks_on_top_when_room_allows(self) -> None:
        # salary $200k × 13% = $26k target → excess $2.5k.
        # match = $200k × 7% × 1.0 = $14k. room = $70k - $23.5k - $14k = $32.5k.
        # spillover = $2.5k; room_left = $30k.
        # explicit = 10% × $200k = $20k, fits in $30k → explicit = $20k.
        # total mega_backdoor_a = $22.5k.
        cfg, inp = _single_earner_setup(
            salary=200_000.0,
            total_contrib_pct=0.13,
            after_tax_pct=0.10,
            mega_enabled=True,
        )
        df = simulate(cfg, inp)
        row = df.iloc[0]
        assert row["excess_deferral_a"] == pytest.approx(2_500.0, abs=1.0)
        assert row["mega_backdoor_spillover_a"] == pytest.approx(2_500.0, abs=1.0)
        assert row["mega_backdoor_a"] == pytest.approx(22_500.0, abs=1.0)
        assert row["after_tax_target_uncovered_a"] == 0.0


# ---------------------------------------------------------------------------
# 5. Crowd-out: spillover consumes all the room, explicit gets clamped.
# ---------------------------------------------------------------------------


class TestExplicitCrowdedOut:
    """When the auto-spillover fills the entire §415(c) room, the
    user's explicit `after_tax_401k_pct` target gets clamped to zero.
    The crowded-out dollars surface in `after_tax_target_uncovered_a`
    and stay as taxable cash (no auto-rerouting elsewhere)."""

    def test_explicit_clamped_when_room_exhausted(self) -> None:
        # Match the plan's worked example exactly.
        cfg, inp = _single_earner_setup(
            salary=300_000.0,
            total_contrib_pct=0.30,
            after_tax_pct=0.10,  # explicit target = $30k
            mega_enabled=True,
        )
        df = simulate(cfg, inp)
        row = df.iloc[0]
        # Room exhausted by the spillover.
        assert row["mega_backdoor_spillover_a"] == pytest.approx(25_500.0, abs=1.0)
        # Explicit target was $30k but no room left → 0 explicit.
        assert row["mega_backdoor_a"] == pytest.approx(25_500.0, abs=1.0)
        # The crowd-out is fully surfaced.
        assert row["after_tax_target_uncovered_a"] == pytest.approx(30_000.0, abs=1.0)


# ---------------------------------------------------------------------------
# 6. Catch-up sits outside §415(c).
# ---------------------------------------------------------------------------


class TestCatchUpExclusion:
    """Age-50+ catch-up ($7,500) raises the §402(g) cap but does NOT
    reduce §415(c) room. So a 55-y-o with $23.5k base + $7.5k catch-up
    still has the same after-tax room as a 45-y-o at $23.5k base."""

    def test_catchup_does_not_consume_415c_room(self) -> None:
        # Age 55, $300k salary, 15% target.
        # cap = $23.5k + $7.5k = $31k; target = $45k → excess = $14k.
        # §415(c) room calc uses BASE deferral only: min($45k, $23.5k) = $23.5k.
        # match (effective pct = $31k/$300k = 10.33%, capped 7%): $21k.
        # room = $70k - $23.5k - $21k = $25.5k.
        # spillover = min($14k, $25.5k) = $14k.
        cfg, inp = _single_earner_setup(
            age=55,
            salary=300_000.0,
            total_contrib_pct=0.15,
            after_tax_pct=0.0,
            mega_enabled=True,
        )
        df = simulate(cfg, inp)
        row = df.iloc[0]
        # The simulator's elective_deferral_a includes catch-up.
        assert row["elective_deferral_a"] == pytest.approx(31_000.0, abs=1.0)
        assert row["excess_deferral_a"] == pytest.approx(14_000.0, abs=1.0)
        assert row["mega_backdoor_spillover_a"] == pytest.approx(14_000.0, abs=1.0)
        # Sanity: §415(c) base-only deferral leaves room as expected.
        assert elective_deferral_cap(55) == ELECTIVE_DEFERRAL_LIMIT + 7_500.0


# ---------------------------------------------------------------------------
# 7. Employer match consumes §415(c) room.
# ---------------------------------------------------------------------------


class TestEmployerMatchConsumesRoom:
    """A larger employer match leaves less §415(c) room for the
    spillover. Capacity drops proportionally."""

    def test_bigger_match_shrinks_spillover(self) -> None:
        # Same salary/target/age; vary the match rate.
        # NOTE: employer match is sized on the POST-CAP effective
        # deferral pct, not the user's target pct. With a $23.5k cap
        # on a $300k salary the effective pct is 7.833%, which is the
        # ceiling for the match-max-pct calc.
        cfg_lo, inp_lo = _single_earner_setup(
            salary=300_000.0,
            total_contrib_pct=0.30,
            after_tax_pct=0.0,
            mega_enabled=True,
            match_rate=0.5,
            match_max_pct=0.06,  # 6% cap is < 7.833% effective
                                 # → match = $300k × 6% × 0.5 = $9k
        )
        cfg_hi, inp_hi = _single_earner_setup(
            salary=300_000.0,
            total_contrib_pct=0.30,
            after_tax_pct=0.0,
            mega_enabled=True,
            match_rate=1.0,
            match_max_pct=0.10,  # 10% cap > 7.833% effective
                                 # → match = $300k × 7.833% × 1.0 = $23.5k
        )
        df_lo = simulate(cfg_lo, inp_lo)
        df_hi = simulate(cfg_hi, inp_hi)

        # Lower match → more room → larger spillover.
        spill_lo = df_lo.iloc[0]["mega_backdoor_spillover_a"]
        spill_hi = df_hi.iloc[0]["mega_backdoor_spillover_a"]
        assert spill_lo > spill_hi
        # lo: room = $70k - $23.5k - $9k = $37.5k → spillover $37.5k.
        # hi: room = $70k - $23.5k - $23.5k = $23k → spillover $23k.
        assert spill_lo == pytest.approx(37_500.0, abs=1.0)
        assert spill_hi == pytest.approx(23_000.0, abs=1.0)


# ---------------------------------------------------------------------------
# 8. Cash-flow consistency: spillover dollars actually leave the
#    household's cash and end up in the Roth balance.
# ---------------------------------------------------------------------------


class TestCashFlowConsistency:
    """Auto-spillover takes cash from the post-tax paycheck (just like
    the existing explicit-pct path). End-of-year Roth balance should
    rise by the mega_backdoor_total; taxable balance should fall by
    roughly the same amount (modulo tax-on-the-cash-that-stayed)."""

    def test_spillover_grows_roth_and_reduces_taxable(self) -> None:
        # Zero out portfolio returns so the cash-flow math is exact
        # (balances at year-end equal contributions in for the first
        # year, no growth-overlay).
        cfg_off, inp_off = _single_earner_setup(
            salary=300_000.0,
            total_contrib_pct=0.30,
            after_tax_pct=0.0,
            mega_enabled=False,  # baseline: silent cap, excess to cash
        )
        cfg_on, inp_on = _single_earner_setup(
            salary=300_000.0,
            total_contrib_pct=0.30,
            after_tax_pct=0.0,
            mega_enabled=True,
        )
        cfg_off = Config(
            horizon_age=50, start_year=2026,
            nominal_growth_rate=0.0, taxable_drag=0.0,
        )
        cfg_on = Config(
            horizon_age=50, start_year=2026,
            nominal_growth_rate=0.0, taxable_drag=0.0,
        )
        df_off = simulate(cfg_off, inp_off)
        df_on = simulate(cfg_on, inp_on)

        row_off = df_off.iloc[0]
        row_on = df_on.iloc[0]

        spillover = row_on["mega_backdoor_spillover_a"]
        assert spillover > 0

        # Roth balance grows more in the spillover case (no portfolio
        # returns → delta equals exact spillover contribution).
        roth_delta = row_on["roth_balance"] - row_off["roth_balance"]
        assert roth_delta == pytest.approx(spillover, abs=1.0)

        # Cash conservation: the silent-cap path (off) and the
        # spillover path (on) book the SAME federal / state / FICA
        # tax (Box 1 uses the *capped* deferral in both cases — the
        # un-deferred excess never enters wages_box1). So the only
        # difference between the two scenarios is where the spillover
        # dollars land: Roth (on) vs taxable (off). The taxable delta
        # must therefore equal the Roth delta in magnitude.
        taxable_delta = row_off["taxable_balance"] - row_on["taxable_balance"]
        assert taxable_delta == pytest.approx(spillover, abs=1.0)

        # And federal tax is unchanged (sanity-check the conservation).
        assert row_on["federal_tax"] == pytest.approx(
            row_off["federal_tax"], abs=1.0
        )


# ---------------------------------------------------------------------------
# 9. Per-spouse independence: A enabled, B disabled.
# ---------------------------------------------------------------------------


class TestPerSpouseIndependence:
    """Spillover decisions are per-spouse and gated by each spouse's
    own `mega_backdoor_enabled` flag."""

    def test_only_enabled_spouse_spills(self) -> None:
        inp = Inputs(
            spouse_a_age_start=45,
            spouse_b_age_start=45,
            spouse_a_retire_age=65,
            spouse_b_retire_age=65,
            spouse_a_total_contrib_pct=0.30,
            spouse_b_total_contrib_pct=0.30,
            spouse_a_roth_401k_pct=0.0,
            spouse_b_roth_401k_pct=0.0,
            spouse_a_mega_backdoor_enabled=True,   # spills
            spouse_b_mega_backdoor_enabled=False,  # does not
            spouse_a_after_tax_401k_pct=0.0,
            spouse_b_after_tax_401k_pct=0.0,
            spouse_a_employer_match_rate=1.0,
            spouse_a_employer_match_max_pct=0.07,
            spouse_b_employer_match_rate=1.0,
            spouse_b_employer_match_max_pct=0.07,
            income=CurrentIncome(
                spouse_a_gross=300_000.0,
                spouse_b_gross=300_000.0,
                spouse_a_bonus=0.0,
            ),
            contrib=CurrentContrib(hsa_family=0.0),
            health_premiums=HealthPremiums(),
            ss=SocialSecurity(start_age_a=70, start_age_b=70),
            starting=StartingBalances(taxable_brokerage=100_000.0),
        )
        cfg = Config(horizon_age=50, start_year=2026)
        df = simulate(cfg, inp)
        row = df.iloc[0]
        # Both spouses have the same excess gap.
        assert row["excess_deferral_a"] == pytest.approx(row["excess_deferral_b"])
        # But only A spills.
        assert row["mega_backdoor_spillover_a"] > 0
        assert row["mega_backdoor_spillover_b"] == 0.0


# ---------------------------------------------------------------------------
# 10. Invariant sweep: across a range of contribution percentages,
#     spillover + explicit ≤ §415(c) room (minus base + match).
# ---------------------------------------------------------------------------


class TestRoomInvariant:
    """For every working year, the total after-tax dollars routed to
    Roth (auto + explicit) must never exceed the §415(c) ceiling
    minus the base deferral minus the employer match."""

    @pytest.mark.parametrize(
        "total_pct,after_tax_pct",
        [
            (0.05, 0.0),
            (0.10, 0.05),
            (0.15, 0.10),
            (0.20, 0.10),
            (0.30, 0.10),
            (0.50, 0.20),
        ],
    )
    def test_room_invariant(self, total_pct, after_tax_pct) -> None:
        cfg, inp = _single_earner_setup(
            salary=300_000.0,
            total_contrib_pct=total_pct,
            after_tax_pct=after_tax_pct,
            mega_enabled=True,
        )
        df = simulate(cfg, inp)
        row = df.iloc[0]

        base_deferral = min(
            300_000.0 * total_pct, ELECTIVE_DEFERRAL_LIMIT
        )
        match = row["employer_match_a"]
        room = max(0.0, SECTION_415C_LIMIT - base_deferral - match)

        mega = row["mega_backdoor_a"]
        # Total routed never exceeds §415(c) room.
        assert mega <= room + 1.0
        # Spillover is the auto-portion; never exceeds room.
        assert row["mega_backdoor_spillover_a"] <= room + 1.0
        # mega_backdoor_a == spillover + explicit; uncovered tracks
        # the explicit-target dollars that didn't fit.
        explicit = mega - row["mega_backdoor_spillover_a"]
        explicit_target = 300_000.0 * after_tax_pct
        assert (
            row["after_tax_target_uncovered_a"]
            == pytest.approx(max(0.0, explicit_target - explicit), abs=1.0)
        )
