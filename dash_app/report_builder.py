"""On-demand HTML action-plan builder for the Dash app.

The Dash app stores Monte-Carlo aggregates and per-strategy
`DataFrame`s in a browser-side `dcc.Store` so the figure callbacks have
local data, but the canonical `tax_optimizer.report.build_action_report`
needs the *full* `Config` / `Inputs` / `StrategyResult` Python objects
plus a tornado-sensitivity sweep — none of which serialize cleanly to
JSON. We solve that with a small module-level LRU cache keyed by a
fresh UUID per run; the Dash run-result Store carries only the UUID,
and the report-download callback re-hydrates the run from the cache.

Public API:

* :func:`cache_run`        — register a `(cfg, inputs, RunResult)` triple
  and return its run-id. Bounded by ``_MAX_CACHED_RUNS`` to avoid
  unbounded growth in long-running app sessions.
* :func:`get_cached`       — fetch the triple by run-id (touches LRU
  ordering).
* :func:`build_html_payload` — re-runs the tornado sweep against the
  base ``(cfg, inputs)``, calls
  :func:`tax_optimizer.report.build_action_report` to get the markdown,
  and wraps it via :func:`tax_optimizer.render.render_html` into a
  self-contained HTML document. The return value is the dict shape
  expected by ``dcc.Download``.

The HTML produced is the same Letter-paged document the CLI emits for
``python -m tax_optimizer --report report.html``, so the user can open
it in any browser or print it to PDF via the browser's "Save as PDF"
dialog. We deliberately do not depend on WeasyPrint here so that the
"Download report" button works on plain ``pip install
tax-optimizer[dash]`` installs without dragging in the pango/cairo
system libraries.
"""

from __future__ import annotations

import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from tax_optimizer.config import Config
from tax_optimizer.inputs import Inputs
from tax_optimizer.render import render_html
from tax_optimizer.report import build_action_report
from tax_optimizer.sensitivity import tornado_sensitivity

from .runner import RunResult


# ---------------------------------------------------------------------
# LRU cache of full RunResult objects, keyed by a UUID per Run.
# ---------------------------------------------------------------------

_MAX_CACHED_RUNS = 5


@dataclass
class CachedRun:
    """Server-side state for a single Dash run.

    We hold the original Python objects (``cfg`` / ``inputs`` / ``rr``)
    so the report renderer can walk them on demand without re-running
    the simulator. The two ``*_cache`` fields memoize the heavy build
    paths so the user can hop between the Report tab and the Download
    button without paying the tornado-sensitivity sweep twice — both
    paths share the same markdown source.
    """

    cfg: Config
    inputs: Inputs
    rr: RunResult
    # Lazy: built on first request via `_build_markdown`.
    _markdown_cache: dict[tuple[str, Optional[str]], str] = field(
        default_factory=dict
    )
    # Lazy: built on first request via `_build_html`. Keyed the same
    # way as `_markdown_cache` (year_table_scope, scenario_path).
    _html_cache: dict[tuple[str, Optional[str]], str] = field(
        default_factory=dict
    )


# Keys are UUID hex strings; values are CachedRun. `OrderedDict` lets
# us evict the oldest entry when the cache overflows.
_RUN_CACHE: "OrderedDict[str, CachedRun]" = OrderedDict()


def cache_run(cfg: Config, inputs: Inputs, rr: RunResult) -> str:
    """Register a finished run and return a fresh ``run_id`` for it.

    The Dash run-result ``dcc.Store`` carries this id alongside the
    JSON-serialized strategy frames. Both the Download HTML callback
    and the Report-tab renderer use the id to recover the original
    Python objects.

    LRU bound: when the cache exceeds :data:`_MAX_CACHED_RUNS`, the
    oldest entry is evicted. That keeps memory bounded in long-lived
    app sessions while still letting the user jump back to a few
    recent runs.
    """
    run_id = uuid.uuid4().hex
    _RUN_CACHE[run_id] = CachedRun(cfg=cfg, inputs=inputs, rr=rr)
    while len(_RUN_CACHE) > _MAX_CACHED_RUNS:
        _RUN_CACHE.popitem(last=False)
    return run_id


def get_cached(run_id: Optional[str]) -> Optional[CachedRun]:
    """Fetch the cached run for ``run_id`` and bump its LRU position.

    Returns ``None`` if the id is missing or has been evicted.
    """
    if not run_id or run_id not in _RUN_CACHE:
        return None
    _RUN_CACHE.move_to_end(run_id)
    return _RUN_CACHE[run_id]


def clear_cache() -> None:
    """Drop every cached run. Mostly for tests / explicit memory release."""
    _RUN_CACHE.clear()


# ---------------------------------------------------------------------
# Report rendering (memoized at the markdown layer)
# ---------------------------------------------------------------------


def _build_markdown(
    cached: CachedRun,
    *,
    scenario_path: Optional[str],
    year_table_scope: str,
) -> str:
    """Return the action-plan markdown for ``cached``, building once.

    Cache key is ``(year_table_scope, scenario_path)`` so a user that
    happens to flip between scopes still benefits from
    short-circuiting on repeats. The expensive piece is
    :func:`tornado_sensitivity` (~1-2s for the default scenario),
    not the markdown formatting itself.
    """
    key = (year_table_scope, scenario_path)
    if key in cached._markdown_cache:
        return cached._markdown_cache[key]
    sens_df, base_terminal = tornado_sensitivity(cached.cfg, cached.inputs)
    md = build_action_report(
        cfg=cached.cfg,
        inputs=cached.inputs,
        results=cached.rr.strategies,
        sens_df=sens_df,
        base_terminal=base_terminal,
        mc=cached.rr.mc,
        scenario_path=scenario_path,
        year_table_scope=year_table_scope,
    )
    cached._markdown_cache[key] = md
    return md


def _build_html(
    cached: CachedRun,
    *,
    scenario_path: Optional[str],
    year_table_scope: str,
) -> str:
    """Return the rendered HTML for ``cached``, memoized."""
    key = (year_table_scope, scenario_path)
    if key in cached._html_cache:
        return cached._html_cache[key]
    md = _build_markdown(
        cached, scenario_path=scenario_path, year_table_scope=year_table_scope
    )
    html_text = render_html(md)
    cached._html_cache[key] = html_text
    return html_text


def build_html_payload(
    cfg_or_cached,
    inputs: Optional[Inputs] = None,
    rr: Optional[RunResult] = None,
    *,
    scenario_path: Optional[str] = None,
    year_table_scope: str = "full",
) -> dict[str, str]:
    """Re-render the action plan as a self-contained HTML document.

    Two call shapes are supported:

    * ``build_html_payload(cached_run)`` — preferred; uses the
      memoized markdown / HTML attached to the
      :class:`CachedRun`.
    * ``build_html_payload(cfg, inputs, rr)`` — legacy 3-arg shape
      kept for backward compatibility with tests / external callers.
      Builds an ephemeral :class:`CachedRun` so the work isn't shared
      with the in-process LRU cache.

    Returns
    -------
    dict with keys ``content`` (the HTML text) and ``filename``
    (``tax-optimizer-report-YYYYMMDD-HHMMSS.html``). The shape is the
    one :class:`dash.dcc.Download` expects, so the callback can
    return this dict directly.
    """
    if isinstance(cfg_or_cached, CachedRun):
        cached = cfg_or_cached
    else:
        if inputs is None or rr is None:
            raise TypeError(
                "build_html_payload() requires either a CachedRun or "
                "the (cfg, inputs, rr) triple."
            )
        cached = CachedRun(cfg=cfg_or_cached, inputs=inputs, rr=rr)

    html_text = _build_html(
        cached, scenario_path=scenario_path, year_table_scope=year_table_scope
    )
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return {
        "content": html_text,
        "filename": f"tax-optimizer-report-{stamp}.html",
    }


def build_inline_html(
    cached: CachedRun,
    *,
    scenario_path: Optional[str] = None,
    year_table_scope: str = "full",
) -> str:
    """Return just the HTML string for the Report tab's iframe.

    Same content as :func:`build_html_payload` but skips the
    download-payload wrapping. Memoized via :class:`CachedRun`'s
    ``_html_cache``, so the second visit to the Report tab during
    the same run re-renders instantly.
    """
    return _build_html(
        cached, scenario_path=scenario_path, year_table_scope=year_table_scope
    )


__all__ = [
    "cache_run",
    "get_cached",
    "clear_cache",
    "build_html_payload",
    "build_inline_html",
    "CachedRun",
]
