"""Tests for the Dash app's "Download HTML report" feature.

Covers the on-demand action-plan builder that backs the
``report-download-btn`` button in :mod:`dash_app.app`. The Dash
callback layer itself is exercised indirectly: we call the public
helpers (``cache_run`` / ``get_cached`` / ``build_html_payload``)
that the callback delegates to, plus a smoke test that runs the full
pipeline end-to-end on a small scenario.

These tests exercise the *server-side* report generation. They do not
spin up a browser or a Dash dev server.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

# `dash_app` is an optional install (only available when the user does
# `pip install -e ".[dash]"`). Skip the entire module if dash isn't
# importable so the base test suite still passes on minimal installs.
pytest.importorskip("dash")

from tax_optimizer import Config, Inputs

from dash_app.report_builder import (  # noqa: E402  - import after skip
    _MAX_CACHED_RUNS,
    build_html_payload,
    cache_run,
    clear_cache,
    get_cached,
)
from dash_app.runner import run_scenario  # noqa: E402


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_cache_around_each_test() -> None:
    """Each test starts with an empty LRU cache."""
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def small_scenario() -> tuple[Config, Inputs]:
    """A short-horizon scenario so the optimizer / sensitivity runs
    fast in CI.

    11-year horizon (start age 50 → 60) keeps every per-knob
    perturbation in :func:`tax_optimizer.sensitivity.tornado_sensitivity`
    well under 100ms, and the optimizer's differential-evolution loop
    converges in a couple of seconds even at the runner's
    ``maxiter=20, popsize=10`` defaults.
    """
    return Config(horizon_age=60), Inputs()


# ---------------------------------------------------------------------
# LRU cache
# ---------------------------------------------------------------------


class TestRunCache:
    def test_round_trip_returns_same_triple(self, small_scenario) -> None:
        cfg, inputs = small_scenario
        rr = run_scenario(cfg, inputs, mode="single")
        run_id = cache_run(cfg, inputs, rr)
        cached = get_cached(run_id)
        assert cached is not None
        c2, i2, r2 = cached
        assert c2 is cfg
        assert i2 is inputs
        assert r2 is rr

    def test_get_cached_returns_none_for_unknown_id(self) -> None:
        assert get_cached("never-cached-id") is None
        assert get_cached(None) is None
        assert get_cached("") is None

    def test_lru_eviction_at_capacity(self, small_scenario) -> None:
        cfg, inputs = small_scenario
        rr = run_scenario(cfg, inputs, mode="single")

        # Fill the cache to capacity, then add one more — the oldest
        # entry must be evicted while everything else stays.
        ids = [cache_run(cfg, inputs, rr) for _ in range(_MAX_CACHED_RUNS)]
        oldest_id = ids[0]
        newest_before_overflow = ids[-1]

        # Sanity: every id is currently retrievable.
        for run_id in ids:
            assert get_cached(run_id) is not None, run_id

        # One more push tips the cache over the limit.
        overflow_id = cache_run(cfg, inputs, rr)
        assert get_cached(oldest_id) is None, "oldest run should have been evicted"
        assert get_cached(newest_before_overflow) is not None
        assert get_cached(overflow_id) is not None

    def test_get_cached_promotes_lru_position(self, small_scenario) -> None:
        cfg, inputs = small_scenario
        rr = run_scenario(cfg, inputs, mode="single")

        ids = [cache_run(cfg, inputs, rr) for _ in range(_MAX_CACHED_RUNS)]
        # Touch the oldest id to bump it to the front.
        assert get_cached(ids[0]) is not None
        # Adding one more entry should now evict the *second*-oldest
        # (since the original oldest got promoted to most-recently-used).
        cache_run(cfg, inputs, rr)
        assert get_cached(ids[0]) is not None, "promoted id should survive"
        assert get_cached(ids[1]) is None, "second-oldest should be evicted now"


# ---------------------------------------------------------------------
# build_html_payload
# ---------------------------------------------------------------------


class TestBuildHtmlPayload:
    def test_returns_content_and_filename(self, small_scenario) -> None:
        cfg, inputs = small_scenario
        rr = run_scenario(cfg, inputs, mode="single")
        payload = build_html_payload(cfg, inputs, rr)
        assert set(payload.keys()) == {"content", "filename"}
        assert isinstance(payload["content"], str) and payload["content"]
        assert payload["filename"].startswith("tax-optimizer-report-")
        assert payload["filename"].endswith(".html")

    def test_content_is_full_html_document(self, small_scenario) -> None:
        cfg, inputs = small_scenario
        rr = run_scenario(cfg, inputs, mode="single")
        html = build_html_payload(cfg, inputs, rr)["content"]
        # The CLI's HTML template wraps the report; check both the
        # outer scaffolding and the inner report header.
        assert html.lstrip().startswith("<!DOCTYPE html>")
        assert "<html" in html and "</html>" in html
        assert "<title>" in html
        # Every report has the same H1.
        assert "Retirement Tax Optimization" in html

    def test_content_includes_canonical_report_sections(
        self, small_scenario
    ) -> None:
        cfg, inputs = small_scenario
        rr = run_scenario(cfg, inputs, mode="single")
        html = build_html_payload(cfg, inputs, rr)["content"]
        # Sections are emitted as ``## N. Title`` markdown which the
        # renderer turns into ``<h2>N. Title</h2>``. Hit a few stable
        # ones spanning the whole report so a regression in the report
        # builder breaks here too.
        for section in (
            "1. Household snapshot",
            "3. Expected outcomes",
            "7. Year-by-year withdrawal &amp; conversion plan",
            "9. Caveats",
        ):
            assert section in html, f"missing report section: {section!r}"

    def test_scenario_path_is_surfaced_in_header(self, small_scenario) -> None:
        cfg, inputs = small_scenario
        rr = run_scenario(cfg, inputs, mode="single")
        html = build_html_payload(
            cfg, inputs, rr, scenario_path="scenarios/example.json"
        )["content"]
        assert "scenarios/example.json" in html

    def test_year_table_scope_retirement_shrinks_year_table(
        self, small_scenario
    ) -> None:
        cfg, inputs = small_scenario
        rr = run_scenario(cfg, inputs, mode="single")
        full = build_html_payload(
            cfg, inputs, rr, year_table_scope="full"
        )["content"]
        retired = build_html_payload(
            cfg, inputs, rr, year_table_scope="retirement"
        )["content"]
        # Both should contain the section header; the retirement-scope
        # version trims pre-retirement rows so the HTML is materially
        # shorter.
        assert "Year-by-year withdrawal &amp; conversion plan" in full
        assert "Year-by-year withdrawal &amp; conversion plan" in retired
        assert len(retired) < len(full)

    def test_invalid_year_table_scope_raises(self, small_scenario) -> None:
        cfg, inputs = small_scenario
        rr = run_scenario(cfg, inputs, mode="single")
        with pytest.raises(ValueError, match="year_table_scope"):
            build_html_payload(cfg, inputs, rr, year_table_scope="bogus")


# ---------------------------------------------------------------------
# Round-trip through the cache (mirrors the production callback flow)
# ---------------------------------------------------------------------


class TestEndToEnd:
    """Full path: run → cache → fetch → render."""

    def test_single_mode_round_trip(self, small_scenario) -> None:
        cfg, inputs = small_scenario
        rr = run_scenario(cfg, inputs, mode="single")
        run_id = cache_run(cfg, inputs, rr)

        cached = get_cached(run_id)
        assert cached is not None
        cfg_back, inp_back, rr_back = cached

        payload = build_html_payload(cfg_back, inp_back, rr_back)
        assert payload["content"]
        assert "Retirement Tax Optimization" in payload["content"]

    def test_four_strategies_mode_round_trip(self, small_scenario) -> None:
        """Sanity that the report builder also handles the four-strategy
        dict shape (S0/S1/S2/S3). Triggers the differential-evolution
        optimizer at the runner's ``maxiter=20, popsize=10`` defaults;
        the very short horizon below keeps the whole thing under a
        couple of seconds even on slow hardware.
        """
        cfg, inputs = small_scenario
        # Drop horizon further so the optimizer's ~200 fitness
        # evaluations stay fast.
        cfg = replace(cfg, horizon_age=58)
        rr = run_scenario(cfg, inputs, mode="four_strategies", seed=0)
        assert set(rr.strategies.keys()) == {
            "S0_baseline",
            "S1_all_roth_401k",
            "S2_bracket_fill_22",
            "S3_optimized",
        }
        run_id = cache_run(cfg, inputs, rr)

        cfg_back, inp_back, rr_back = get_cached(run_id)
        payload = build_html_payload(cfg_back, inp_back, rr_back)
        html = payload["content"]
        # Four-strategy reports surface the verdict block.
        assert "S0_baseline" in html
        assert "S3_optimized" in html
