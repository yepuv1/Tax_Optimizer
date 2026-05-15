"""Tests for the Dash app's action-plan report rendering.

Two surfaces share the same memoized HTML source:

* ``report-download-btn`` — saves the report as a self-contained HTML
  file (callback ``_download_report``).
* The Report tab's ``<iframe>`` — auto-renders when the user
  activates the tab (callback ``_render_report_tab``).

These tests exercise the *server-side* helpers that back both
callbacks (``cache_run`` / ``get_cached`` / ``build_html_payload`` /
``build_inline_html``) plus a smoke test that ``make_app()`` builds
the full Dash callback graph without duplicate-Output errors.
They do not spin up a browser or a Dash dev server.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import replace
from unittest.mock import patch

import pytest


@contextmanager
def _silence_dash_deprecations():
    """Suppress noisy ``DeprecationWarning``s emitted while building
    the Dash layout (e.g. ``dash_table.DataTable`` will be removed in
    a future major version). The pytest config treats warnings as
    errors, so we have to scope-silence them around ``make_app()``
    instead of polluting the global filter list.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        yield

# `dash_app` is an optional install (only available when the user does
# `pip install -e ".[dash]"`). Skip the entire module if dash isn't
# importable so the base test suite still passes on minimal installs.
pytest.importorskip("dash")

from tax_optimizer import Config, Inputs

from dash_app.report_builder import (  # noqa: E402  - import after skip
    _MAX_CACHED_RUNS,
    CachedRun,
    build_html_payload,
    build_inline_html,
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
    def test_round_trip_returns_same_objects(self, small_scenario) -> None:
        cfg, inputs = small_scenario
        rr = run_scenario(cfg, inputs, mode="single")
        run_id = cache_run(cfg, inputs, rr)
        cached = get_cached(run_id)
        assert isinstance(cached, CachedRun)
        # Identity preserved through the cache so the report builder
        # can walk the original Config / Inputs without copies.
        assert cached.cfg is cfg
        assert cached.inputs is inputs
        assert cached.rr is rr

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

    def test_three_arg_legacy_call_still_works(self, small_scenario) -> None:
        """Backward compat: ``build_html_payload(cfg, inputs, rr)`` is
        the original public shape. Make sure adding the
        single-``CachedRun`` overload didn't break it (existing
        notebooks / external callers may still use the 3-arg form).
        """
        cfg, inputs = small_scenario
        rr = run_scenario(cfg, inputs, mode="single")
        payload = build_html_payload(cfg, inputs, rr)
        assert payload["content"]
        assert payload["filename"].startswith("tax-optimizer-report-")

    def test_cached_run_overload(self, small_scenario) -> None:
        """``build_html_payload(cached_run)`` is the preferred shape
        used by the Dash callbacks — it shares the memoized markdown
        with :func:`build_inline_html` so both surfaces stay cheap."""
        cfg, inputs = small_scenario
        rr = run_scenario(cfg, inputs, mode="single")
        run_id = cache_run(cfg, inputs, rr)
        cached = get_cached(run_id)
        payload = build_html_payload(cached)
        assert payload["content"]
        assert "Retirement Tax Optimization" in payload["content"]

    def test_two_arg_call_raises(self, small_scenario) -> None:
        """Catch obvious mis-uses of the overloaded signature early."""
        cfg, inputs = small_scenario
        with pytest.raises(TypeError, match="CachedRun or"):
            build_html_payload(cfg, inputs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------
# build_inline_html  (Report-tab iframe srcDoc)
# ---------------------------------------------------------------------


class TestBuildInlineHtml:
    def test_returns_full_html_document(self, small_scenario) -> None:
        cfg, inputs = small_scenario
        rr = run_scenario(cfg, inputs, mode="single")
        cached = CachedRun(cfg=cfg, inputs=inputs, rr=rr)
        html = build_inline_html(cached)
        assert html.lstrip().startswith("<!DOCTYPE html>")
        assert "Retirement Tax Optimization" in html

    def test_inline_html_matches_download_content(self, small_scenario) -> None:
        """The Download HTML button and the Report tab share their
        source so users see the *same* document on both surfaces.
        """
        cfg, inputs = small_scenario
        rr = run_scenario(cfg, inputs, mode="single")
        cached = CachedRun(cfg=cfg, inputs=inputs, rr=rr)
        inline = build_inline_html(cached)
        download = build_html_payload(cached)["content"]
        assert inline == download

    def test_second_call_hits_cache(self, small_scenario) -> None:
        """The expensive piece is the tornado-sensitivity sweep run
        inside ``_build_markdown``. Re-rendering the same cached run
        a second time must short-circuit through the per-CachedRun
        markdown cache so tab re-visits during a session don't pay
        the sweep again.
        """
        cfg, inputs = small_scenario
        rr = run_scenario(cfg, inputs, mode="single")
        cached = CachedRun(cfg=cfg, inputs=inputs, rr=rr)

        with patch(
            "dash_app.report_builder.tornado_sensitivity",
            wraps=__import__(
                "tax_optimizer.sensitivity", fromlist=["tornado_sensitivity"]
            ).tornado_sensitivity,
        ) as m:
            first = build_inline_html(cached)
            second = build_inline_html(cached)
            third = build_html_payload(cached)["content"]

        assert first == second == third
        # Tornado sweep ran exactly once across the three renders even
        # though they hit two different entry points.
        assert m.call_count == 1

    def test_different_scopes_cached_separately(self, small_scenario) -> None:
        """Switching ``year_table_scope`` must not return a stale
        cached HTML keyed under the other scope.
        """
        cfg, inputs = small_scenario
        rr = run_scenario(cfg, inputs, mode="single")
        cached = CachedRun(cfg=cfg, inputs=inputs, rr=rr)
        full = build_inline_html(cached, year_table_scope="full")
        retired = build_inline_html(cached, year_table_scope="retirement")
        assert full != retired
        assert len(retired) < len(full)


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

        payload = build_html_payload(cached)
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

        cached = get_cached(run_id)
        payload = build_html_payload(cached)
        html = payload["content"]
        # Four-strategy reports surface the verdict block.
        assert "S0_baseline" in html
        assert "S3_optimized" in html


# ---------------------------------------------------------------------
# Dash app smoke test (callback graph)
# ---------------------------------------------------------------------


class TestAppCallbackGraph:
    """Ensure the new Report-tab callback registers cleanly.

    We don't run the dev server — building the Dash app already
    validates every ``@app.callback`` decorator (duplicate Outputs
    without ``allow_duplicate=True`` raise at registration time).
    Importing :func:`dash_app.app.make_app` and calling it covers all
    the callback wiring in one shot.
    """

    def test_make_app_builds_without_duplicate_output_errors(self) -> None:
        from dash_app.app import make_app

        with _silence_dash_deprecations():
            app = make_app()
        cb_map = app.callback_map
        # Sanity: the report iframe srcDoc should be the output of at
        # least one registered callback (the new ``_render_report_tab``).
        outputs = []
        for spec in cb_map.values():
            outs = spec.get("output", [])
            if not isinstance(outs, list):
                outs = [outs]
            for out in outs:
                outputs.append(getattr(out, "component_id", None))
        assert "report-iframe" in outputs, (
            "expected report-iframe to be the output of the report-tab "
            "callback"
        )

    def test_report_iframe_has_at_most_one_writer(self) -> None:
        """The Report-tab refactor moved the iframe srcDoc output off
        the Download callback so only one callback writes to it.
        Multiple writers would force ``allow_duplicate=True`` and
        introduce ordering hazards (whichever fires last wins).
        """
        from dash_app.app import make_app

        with _silence_dash_deprecations():
            app = make_app()
        writers = 0
        for spec in app.callback_map.values():
            outs = spec.get("output", [])
            if not isinstance(outs, list):
                outs = [outs]
            for o in outs:
                if getattr(o, "component_id", None) == "report-iframe":
                    writers += 1
        assert writers == 1, (
            f"report-iframe.srcDoc has {writers} writer callbacks; "
            "expected exactly 1 (the Report-tab renderer)."
        )
