"""Estate / heir-mode regression: with both spouses dead the simulator
must NOT phantom-spend by liquidating the taxable account against an
age-driven smile profile that no living person can consume.

Bug history: prior to this guard, the simulator treated mortality as a
strict income-side gate (wages / pension / SS / RMD all zero out when
both `alive_*` flip False) but the spending profile and withdrawal
cascade had no parallel guard. The smile profile would keep producing a
non-zero `net_need` for the surviving estate years, the cascade would
fall into the retirement branch (`a_working or b_working` is False), and
`withdraw_for_need` would liquidate the taxable account to net out a
spending need that nobody was actually spending. That silently
understated the inheritance balance and produced a confusing all-zeros
income row alongside a non-zero `taxable_withdrawal`.

The fix collapses `net_need` (and any scheduled lump events) to zero in
years where neither spouse is alive. Tax on residual portfolio yield is
still funded via the normal deficit cascade.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from tax_optimizer import (
    Config,
    Inputs,
    Mortality,
    SpendingProfile,
    simulate,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _both_dead_mask(cfg: Config, inputs: Inputs, df):
    """Return a boolean Series flagging rows where both spouses are dead.

    Mortality uses simulation year offsets, so we recompute it from the
    `cfg.mortality` config rather than introspecting any specific column
    of the result frame.
    """
    a_offset = df["spouse_a_age"] - inputs.spouse_a_age_start
    death_a = cfg.mortality.year_of_death_a
    death_b = cfg.mortality.year_of_death_b
    if death_a is None or death_b is None:
        # If either spouse never dies the both-dead set is empty.
        return a_offset != a_offset  # all-False series
    return a_offset >= max(death_a, death_b)


# ---------------------------------------------------------------------------
# Core invariants
# ---------------------------------------------------------------------------


class TestBothSpousesDeadCollapsesSpending:
    """When both spouses are dead, the smile spending profile must not
    drive any consumption-funded withdrawals."""

    def _scenario(self) -> tuple[Config, Inputs]:
        # Both spouses die at year offset 30. Horizon stretches well past
        # that so we have a clean post-death window to inspect.
        cfg = Config(
            horizon_age=90,
            mortality=Mortality(year_of_death_a=30, year_of_death_b=30),
            spending=SpendingProfile(
                base_spending=100_000.0,
                inflation=0.03,
            ),
        )
        inputs = Inputs()
        return cfg, inputs

    def test_post_death_window_exists(self) -> None:
        cfg, inputs = self._scenario()
        df = simulate(cfg, inputs)
        mask = _both_dead_mask(cfg, inputs, df)
        # Sanity check: the test only proves something if there's at
        # least one estate-mode row. With death year 30 and horizon 90
        # (Spouse A starts at 50 by Inputs() default → simulates 41
        # rows), we expect ~10 post-death rows.
        assert mask.sum() >= 5

    def test_spending_need_is_zero_post_death(self) -> None:
        cfg, inputs = self._scenario()
        df = simulate(cfg, inputs)
        mask = _both_dead_mask(cfg, inputs, df)
        post = df[mask]
        # The smile spending profile is age-driven and would otherwise
        # report ~$200k+ in this window. The guard collapses it to 0.
        assert post["spending_need"].sum() == pytest.approx(0.0, abs=1e-6)

    def test_no_consumption_withdrawals_post_death(self) -> None:
        cfg, inputs = self._scenario()
        df = simulate(cfg, inputs)
        mask = _both_dead_mask(cfg, inputs, df)
        post = df[mask]

        # Pretax / Roth withdrawals must be zero: there's no RMD (both
        # dead → `alive_*` False blocks RMD), no spending need, and no
        # conversion in estate years.
        assert post["pretax_withdrawal"].sum() == pytest.approx(0.0, abs=1e-6)
        assert post["roth_withdrawal"].sum() == pytest.approx(0.0, abs=1e-6)

        # Taxable withdrawal can be very slightly non-zero in the year
        # immediately after death because portfolio yield (dividends /
        # interest on the still-alive taxable balance) generates a tiny
        # tax liability that the deficit cascade funds. Keep the bound
        # tight enough to catch any regression to the old phantom-spend
        # behavior (which produced ~$200k+ taxable_withdrawal per year).
        assert (post["taxable_withdrawal"] < 5_000.0).all(), (
            f"Estate-mode taxable_withdrawal too large; phantom-spend regression?\n"
            f"{post[['spouse_a_age', 'spending_need', 'taxable_withdrawal']]}"
        )

    def test_income_lines_remain_zero_post_death(self) -> None:
        """Sanity: the existing alive-gates on income are unchanged by
        the spending guard."""
        cfg, inputs = self._scenario()
        df = simulate(cfg, inputs)
        mask = _both_dead_mask(cfg, inputs, df)
        post = df[mask]
        for col in ("wages", "pension", "ssn", "rmd"):
            assert post[col].sum() == pytest.approx(0.0, abs=1e-6), (
                f"Income column {col} non-zero in estate-mode rows"
            )


class TestSurvivorStillSpends:
    """The guard must NOT touch the single-survivor case: when one
    spouse is alive, the smile profile and withdrawal cascade should
    continue to function exactly as before."""

    def test_one_alive_spouse_keeps_spending(self) -> None:
        # Spouse A dies at year 20, Spouse B survives to horizon.
        cfg = Config(
            horizon_age=90,
            mortality=Mortality(year_of_death_a=20, year_of_death_b=None),
            spending=SpendingProfile(
                base_spending=80_000.0,
                inflation=0.0,
            ),
        )
        inputs = Inputs()
        df = simulate(cfg, inputs)

        # Pick rows after A's death where B is still alive AND past
        # retirement, so the smile is in steady-state retirement mode.
        a_offset = df["spouse_a_age"] - inputs.spouse_a_age_start
        survivor_rows = df[
            (a_offset >= 20)
            & (a_offset < 30)  # well before any plausible B death
            & (df["spouse_a_age"] >= max(inputs.spouse_a_retire_age, inputs.spouse_b_retire_age))
        ]
        assert len(survivor_rows) >= 3
        # The survivor still consumes, so spending_need must be > 0
        # for at least one row in this window.
        assert (survivor_rows["spending_need"] > 0).any()


class TestBothAliveUnchanged:
    """Default `Mortality()` (both alive through horizon) must produce
    identical results to before the guard — no rows match the both-dead
    mask, so `net_need` is never touched."""

    def test_default_simulation_unchanged_shape(self) -> None:
        cfg = Config()  # default mortality: both survive
        inputs = Inputs()
        df = simulate(cfg, inputs)
        mask = _both_dead_mask(cfg, inputs, df)
        assert mask.sum() == 0
        # Spending need should be > 0 in retirement years.
        retired = df[df["spouse_a_age"] >= inputs.spouse_a_retire_age]
        assert (retired["spending_need"] > 0).any()
