"""Tests for spending-related deprecation / consistency warnings.

Covers two related but distinct UX guardrails:

1. ``Inputs.annual_expenses`` is a dead field — the simulator never
   reads it (the spending base comes from ``cfg.resolved_spending()``).
   We keep the dataclass field for backward compatibility with old
   scenarios, but setting it to a non-default value emits a
   ``DeprecationWarning`` so users migrate to the correct knob.

2. When a scenario JSON sets both ``config.annual_expenses_today``
   (the scalar fallback) AND ``config.spending.base_spending``
   (the active knob) to **different** values, the scalar is silently
   ignored. The loader emits a ``UserWarning`` to surface the
   misconfiguration. Matching values stay silent (redundant but not
   misleading).
"""

from __future__ import annotations

import warnings

import pytest

from tax_optimizer import Config, Inputs
from tax_optimizer.scenario import apply_scenario, apply_set_overrides


# --------------------------------------------------------- Inputs.annual_expenses


class TestInputsAnnualExpensesDeprecation:
    def test_default_construction_is_silent(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Inputs()
        assert not [
            w for w in caught if issubclass(w.category, DeprecationWarning)
        ]

    def test_non_default_value_warns(self) -> None:
        with pytest.warns(DeprecationWarning, match="annual_expenses"):
            Inputs(annual_expenses=120_000.0)

    def test_warning_message_points_at_replacement(self) -> None:
        with pytest.warns(DeprecationWarning) as record:
            Inputs(annual_expenses=99_999.0)
        msg = str(record[0].message)
        # Replacement knobs both named in the message so the user can
        # self-fix without reading the source.
        assert "Config.annual_expenses_today" in msg
        assert "Config.spending.base_spending" in msg

    def test_setting_default_value_explicitly_is_silent(self) -> None:
        # Explicitly setting it to the legacy default value is not a
        # migration signal — keep the loader/test machinery quiet so
        # round-tripping a deserialized "85000" doesn't spam warnings.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Inputs(annual_expenses=85_000.0)
        assert not [
            w for w in caught if issubclass(w.category, DeprecationWarning)
        ]

    def test_loader_propagates_warning(self) -> None:
        # The scenario loader calls Inputs(**patch), which fires the
        # post-init warning. Verify it bubbles through apply_scenario.
        with pytest.warns(DeprecationWarning, match="annual_expenses"):
            apply_scenario(
                Config(),
                Inputs(),
                {"inputs": {"annual_expenses": 110_000}},
            )

    def test_set_override_propagates_warning(self) -> None:
        # `--set inputs.annual_expenses=...` should also fire the warning.
        with pytest.warns(DeprecationWarning, match="annual_expenses"):
            apply_set_overrides(
                Config(), Inputs(), ["inputs.annual_expenses=130000"]
            )


# ------------------------------------------ config.annual_expenses_today vs spending


class TestSpendingInconsistencyWarning:
    def test_matching_values_silent(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            apply_scenario(
                Config(),
                Inputs(),
                {
                    "config": {
                        "annual_expenses_today": 100_000,
                        "spending": {
                            "kind": "smile",
                            "base_spending": 100_000,
                        },
                    }
                },
            )
        assert not [
            w for w in caught if issubclass(w.category, UserWarning)
        ]

    def test_mismatched_values_warn(self) -> None:
        with pytest.warns(UserWarning, match="annual_expenses_today.*base_spending"):
            apply_scenario(
                Config(),
                Inputs(),
                {
                    "config": {
                        "annual_expenses_today": 85_000,
                        "spending": {
                            "kind": "smile",
                            "base_spending": 100_000,
                        },
                    }
                },
            )

    def test_warning_quotes_both_values(self) -> None:
        with pytest.warns(UserWarning) as record:
            apply_scenario(
                Config(),
                Inputs(),
                {
                    "config": {
                        "annual_expenses_today": 75_000,
                        "spending": {
                            "kind": "smile",
                            "base_spending": 110_000,
                        },
                    }
                },
            )
        msg = str(record[0].message)
        assert "$75,000" in msg
        assert "$110,000" in msg

    def test_only_scalar_set_silent(self) -> None:
        # `config.annual_expenses_today` alone (no spending block) is
        # the documented fallback path; nothing to reconcile.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            apply_scenario(
                Config(),
                Inputs(),
                {"config": {"annual_expenses_today": 120_000}},
            )
        assert not [
            w for w in caught if issubclass(w.category, UserWarning)
        ]

    def test_only_spending_block_set_silent(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            apply_scenario(
                Config(),
                Inputs(),
                {
                    "config": {
                        "spending": {"kind": "flat", "base_spending": 90_000}
                    }
                },
            )
        assert not [
            w for w in caught if issubclass(w.category, UserWarning)
        ]

    def test_spending_null_is_silent(self) -> None:
        # Explicit `spending: null` resets to the deterministic-fallback
        # path. Even with an explicit `annual_expenses_today` we treat
        # the scalar as actively used → no warning.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            apply_scenario(
                Config(),
                Inputs(),
                {
                    "config": {
                        "annual_expenses_today": 100_000,
                        "spending": None,
                    }
                },
            )
        assert not [
            w for w in caught if issubclass(w.category, UserWarning)
        ]

    def test_spending_without_base_spending_silent(self) -> None:
        # A spending block that doesn't carry an explicit
        # `base_spending` field (e.g. a future custom-only shape) can't
        # contradict the scalar; stay silent.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                apply_scenario(
                    Config(),
                    Inputs(),
                    {
                        "config": {
                            "annual_expenses_today": 100_000,
                            # Intentionally missing required keys — the
                            # coercion step will raise; we only want to
                            # see that the inconsistency check itself
                            # doesn't fire a UserWarning beforehand.
                            "spending": {"kind": "smile"},
                        }
                    },
                )
            except Exception:
                pass
        assert not [
            w for w in caught if issubclass(w.category, UserWarning)
        ]
