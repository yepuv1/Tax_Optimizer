"""Tests for `tax_optimizer.mortality.Mortality`."""

from __future__ import annotations

from tax_optimizer.mortality import Mortality


class TestMortalityDefaults:
    """`Mortality()` with no death years means both spouses survive to horizon."""

    def test_alive_a_default_is_always_true(self) -> None:
        m = Mortality()
        assert m.alive_a(0) is True
        assert m.alive_a(50) is True
        assert m.alive_a(1_000) is True

    def test_alive_b_default_is_always_true(self) -> None:
        m = Mortality()
        assert m.alive_b(0) is True
        assert m.alive_b(75) is True

    def test_filing_status_default_is_mfj(self) -> None:
        m = Mortality()
        assert m.filing_status(0) == "mfj"
        assert m.filing_status(99) == "mfj"

    def test_survivor_label_none_when_both_alive(self) -> None:
        m = Mortality()
        assert m.survivor_label(0) is None
        assert m.survivor_label(50) is None

    def test_default_pension_survivor_pct_is_50pct(self) -> None:
        assert Mortality().pension_survivor_pct == 0.50

    def test_default_ss_survivor_keeps_higher(self) -> None:
        assert Mortality().ss_survivor_keeps_higher is True


class TestSpouseADies:
    """Spouse A dies at year 20 → A alive for 0..19, dead from 20+."""

    def test_alive_a_boundary(self) -> None:
        m = Mortality(year_of_death_a=20)
        assert m.alive_a(0) is True
        assert m.alive_a(19) is True
        assert m.alive_a(20) is False  # dies at start of year 20
        assert m.alive_a(50) is False

    def test_alive_b_unaffected(self) -> None:
        m = Mortality(year_of_death_a=20)
        assert m.alive_b(20) is True
        assert m.alive_b(50) is True

    def test_filing_status_flips_to_single(self) -> None:
        # IRS year-of-death rule: the calendar year a spouse dies is
        # still MFJ; the survivor only flips to single starting the
        # following year.
        m = Mortality(year_of_death_a=20)
        assert m.filing_status(19) == "mfj"
        assert m.filing_status(20) == "mfj"   # year of death — still MFJ
        assert m.filing_status(21) == "single"

    def test_is_year_of_death_helper(self) -> None:
        m = Mortality(year_of_death_a=20, year_of_death_b=30)
        assert m.is_year_of_death(20) is True
        assert m.is_year_of_death(30) is True
        assert m.is_year_of_death(19) is False
        assert m.is_year_of_death(25) is False

    def test_survivor_label_b(self) -> None:
        m = Mortality(year_of_death_a=20)
        assert m.survivor_label(19) is None
        assert m.survivor_label(20) == "b"

    def test_both_alive_helper(self) -> None:
        m = Mortality(year_of_death_a=20)
        assert m.both_alive(19) is True
        assert m.both_alive(20) is False


class TestSpouseBDies:
    def test_survivor_label_a(self) -> None:
        m = Mortality(year_of_death_b=15)
        assert m.survivor_label(14) is None
        assert m.survivor_label(15) == "a"


class TestBothDie:
    def test_survivor_label_neither(self) -> None:
        m = Mortality(year_of_death_a=10, year_of_death_b=20)
        assert m.survivor_label(9) is None         # both alive
        assert m.survivor_label(10) == "b"          # a dead, b alive
        assert m.survivor_label(20) == "neither"   # both dead
