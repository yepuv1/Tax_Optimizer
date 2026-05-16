# Changelog

All notable feature additions, modeling enhancements, and bug fixes
to `tax_optimizer` are tracked here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

The package version constant lives in `tax_optimizer/__init__.py`.
Section labels (`v2`, `v3 — Tier A`, `v3 — Tier B`, `v4`) match the
"What changed" recap at the bottom of the standalone notebooks.

> **Maintenance rule:** **Every** new feature, behavior change, or
> bug fix added to this repo MUST get a one-line entry in the
> `[Unreleased]` section below before the change is committed.
> Entries graduate from `[Unreleased]` to a versioned section when
> a release is cut.
>
> When in doubt, prefer a smaller, more specific entry over a vague
> sweeping one. The point is for someone six months from now (or
> the next contributor) to be able to scan this file and know
> exactly what shipped, when, and where to look for it in code.

Categories used:

- **Added** — new user-visible feature or knob
- **Changed** — change to existing behavior (often a default-value
  shift or API rename)
- **Fixed** — bug fix
- **Tests** — new test coverage worth calling out
- **Docs** — documentation/notebook updates

---

## [Unreleased]

### Added — Single-filer households (`inputs.household_kind`)

**Why it matters:** the simulator was previously
married-filing-jointly only — every scenario had to model a
two-spouse household, even for users who file Single. Adding a
single-filer flag lets a never-married / divorced filer plan in
the right tax tables (Single brackets, Single std deduction),
the right FICA thresholds ($200k Additional-Medicare vs $250k
MFJ), the right IRMAA tiers, and the right HSA self-only limit
without faking it via mortality.

**What changed:**

- New ``Inputs.household_kind`` field with values ``"mfj"``
  (default, back-compat) and ``"single"``. Validated in
  ``__post_init__`` so a typo raises immediately rather than
  silently falling through to the wrong branch.
- ``simulator.simulate`` overrides ``alive_b = False`` and
  ``filing_status = "single"`` from year 0 when
  ``household_kind == "single"``. This bypasses the IRS
  year-of-death MFJ exception (which would otherwise apply if
  a user faked single via ``year_of_death_b = 0``) and routes
  every downstream tax / FICA / IRMAA / SS-provisional call
  through the existing Single tables in ``TaxRegime``.
- ``simulator.simulate`` Social Security branch now
  distinguishes never-married single filers from widows: the
  former gets only their own benefit (no fictitious survivor
  benefit on a non-existent spouse's record); the latter
  retains the existing widow survivor logic.
- ``hsa_family_cap`` accepts a new ``b_alive`` keyword (default
  ``True`` for back-compat). Single households drop to the
  IRS self-only HSA limit instead of the family limit.
- New scenario file ``scenarios/example_single.json`` showing
  a complete single-filer household.
- ``scenarios/template.json`` now includes the
  ``household_kind`` field at its default value.

**Dash app:**

- New "Filing status" selector at the top of the simple-tier
  Ages & retirement section (``inputs.household_kind``).
- New ``FormField.couple_only`` flag tagging every spouse-B
  field (ages, income, 401(k), starting balances, SS, IRA,
  mega-backdoor, mortality, FRA, health premiums).
- New callback ``_toggle_couple_only_inputs`` disables every
  ``couple_only`` input when ``household_kind == "single"``
  (and re-enables them on flip back to "mfj"). Values stay in
  the form so toggling the discriminator doesn't blow away
  user-entered data.
- New CSS (``fira-code.css``) gives disabled inputs / dropdowns
  / switches a faded look so users can see at a glance which
  fields the simulator will ignore.

**Tests:**

- ``tests/test_simulator_single_household.py`` — 16 tests
  pinning filing status, ``alive_b``, spouse-B input
  suppression, single tax > MFJ tax at the same income, no
  widow MFJ exception, the SS no-survivor-benefit guarantee,
  the HSA self-only cap, and the no-spurious-rollover
  invariant.
- ``tests/test_dash_household_kind.py`` — 14 tests pinning
  the form schema (selector exists, in simple tier, two
  options, has help, isn't itself ``couple_only``), the
  ``couple_only`` flag coverage (every spouse-B field tagged,
  no household-level field tagged), the toggle callback
  semantics (mfj enables everything, single disables exactly
  the couple-only inputs), and the live ``make_app()``
  callback registration.

### Changed — Dash app: every Plotly chart now renders in Fira Code

**Why it matters:** the dashboard chrome (form labels, KPI
tiles, DataTable, headings) is themed via
``dash_app/assets/fira-code.css`` to use Fira Code as the
primary monospace font. Plotly figures, however, render their
text inside the chart's SVG and do NOT inherit ``font-family``
from the surrounding HTML. Every chart was therefore rendering
in Plotly's default sans-serif (``Open Sans, verdana, arial``),
which looked visually disconnected from the rest of the
dashboard — chart titles, axis labels, tick labels, legend
entries, hover tooltips, and percentile callouts were all in a
different typeface than the surrounding UI.

**Fix:** add a single ``_FIRA_FONT_FAMILY`` constant to
``dash_app/figures.py`` mirroring the same fallback chain the
CSS uses (``'Fira Code', 'Fira Mono', 'JetBrains Mono',
'SFMono-Regular', ui-monospace, Menlo, Monaco, Consolas,
'Liberation Mono', monospace``), and wire it into the shared
``_LAYOUT`` fragment as ``font=dict(family=_FIRA_FONT_FAMILY,
size=12)``. Every figure builder that spreads ``**_LAYOUT``
into ``update_layout`` now picks up the family automatically;
``empty_figure`` (the only builder that doesn't use
``_LAYOUT``) sets the same family directly.

Per-element ``font=dict(size=N, color=…)`` overrides scattered
through this module continue to work — Plotly merges the
partial dicts with the layout-level default, leaving family
inherited unless explicitly clobbered.

**Tests:** new ``tests/test_dash_figure_font.py`` pins the
contract — every public figure builder
(``balance_stack``, ``taxes_panel``, ``conversion_panel``,
``multi_strategy_taxes_panel``, ``multi_strategy_conversion_panel``,
``multi_strategy_growth_panel``, ``strategy_comparison``,
``strategy_compare_panel``, ``mc_terminal_histogram``,
``mc_fan_chart``, ``empty_figure``) must produce a figure
whose ``layout.font.family`` contains "Fira Code". Two
extra tests verify that per-element annotation overrides
don't accidentally clobber the family.

### Fixed — Dash app: Monte Carlo histogram P50 callout overlapping the chart title

**Why it matters:** the `mc_terminal_histogram` figure pushed
the median (P50) percentile callout up by ``yshift=22`` to give
it a separate vertical row from the side labels (P10 / P90).
The intent was anti-overlap protection in the pathological
"P10 ≈ P50" case — but the side effect was that the P50 box
landed inside the chart-title margin band, where it visibly
collided with the title text on every typical run (see the
screenshot in the conversation thread).

**Fix:** drop ``yshift`` to 0 for all three callouts. The
existing horizontal-anchor differentiation (P10 anchored
LEFT, P50 CENTER, P90 RIGHT) is sufficient overlap protection
on its own — the three callouts land in three different
horizontal zones for any plausible percentile set, so the
extra vertical staggering was redundant. The 80 px top margin
is retained so the title still has comfortable breathing room
above the top-of-plot callout row.

**Tests:** `test_dash_mc_histogram.py` —
`test_p50_label_is_vertically_offset` (which previously pinned
``yshifts["P50"] > yshifts["P10"]``) renamed to
`test_no_callout_pierces_the_title_margin` and rewritten as
the *opposite* assertion: every callout's ``yshift`` must be
``<= 0`` so a future change can't reintroduce the
title-overlap regression. The top-margin test had its rationale
comment updated to reflect the new design.

### Fixed — Dash app: run-control card layout + persistent loaded-scenario filename

**Why it matters:** the run-control card at the top of the
sidebar had two related UX defects:

1. **Filename was overwritten.** The "Loaded scenario from
   foo.json" message was written to the shared ``run-status``
   div, which the Run callback then clobbered with the run
   summary ("Ran 'four_plus_mc' in 1.5s. Winner: …"). After
   one click of Run the user lost track of which scenario was
   currently loaded into the form.
2. **Inconsistent column heights.** The run-mode radios sat
   in a single ``md=5`` column with vertically-stacked labels.
   "Four + Monte Carlo (~10-30s)" wrapped onto two lines on
   typical viewports, making that column visually taller than
   the MC paths / Seed / Run column to its right. The Run
   button ended up wedged mid-height, which read as a layout
   bug.

**Fix:**

- Added a new ``scenario-loaded-name`` div directly below the
  upload widget, with its own callback output. The filename
  ("📄 Loaded: foo.json") persists there independently of
  ``run-status``, so subsequent runs no longer overwrite it.
- Reorganized ``top_bar()`` into three rows: scenario I/O
  (upload + download + the new loaded-name indicator), a
  full-width horizontal run-mode radio strip (no more wrap),
  and a balanced ``md=4 / md=4 / md=4`` row for MC paths /
  Seed / Run with shared baseline alignment.
- New CSS: ``.scenario-loaded-name:empty { display: none; }``
  collapses the indicator's vertical space until a file is
  loaded, and ``.run-mode-radios label`` keeps the horizontal
  radio strip readable on narrow viewports.

### Fixed — Dash app: duplicate hint tooltips on hover (Overview tiles + form fields)

**Why it matters:** every hint surface in the dashboard
(Overview KPI tiles + scenario-form ⓘ icons) was wired with
*two* tooltip layers — a native HTML ``title`` attribute (which
the browser renders immediately on hover) AND a
``dbc.Tooltip`` (the styled Bootstrap popover, with a 150 ms
show delay). The result on hover was a flicker: the browser's
native tooltip flashed first, then the Bootstrap one rendered
on top. The user's perception was "two duplicate tooltips, one
quick and one delayed".

**Fix:** drop the native ``title`` attribute on every
hint-bearing element. The single visible tooltip is the
``dbc.Tooltip`` popover. Accessibility / screen-reader support
is retained via the ⓘ icon's ``aria-label`` attribute, which
assistive technology prefers to ``title`` anyway. Touched both
the form-panel pattern (`dash_app/layout.py:_help_components`,
`_field_row`) and the Overview-tile pattern
(`dash_app/app.py:_build_kpi_tile`).

**Tests:** `test_dash_form_help.py` and
`test_dash_overview_growth.py` updated — the two tests that
previously *required* a native ``title`` attribute now assert
the opposite (no ``title`` anywhere in the rendered tree),
pinning the contract so a future change can't reintroduce the
duplicate-tooltip bug.

### Added — Dash app: Overview-tab KPI tiles get tooltip hints

**Why it matters:** the Overview tab has 11+ tiles (5–7
Outcomes + 6 Growth) packed with terms-of-art labels — "Peak
marginal rate", "CVaR (10%)", "Effective CAGR", "Accumulation
CAGR" — that don't necessarily mean the same thing to every
viewer. A clinician-style label without explanation invites
misinterpretation (e.g. reading Effective CAGR as a pure
investment return when it actually bundles contributions +
withdrawals + tax drag).

This change attaches help text to every Overview tile via the
same ⓘ-icon + Bootstrap-tooltip pattern already used by the
scenario-form fields:

- ``figures.overview_kpis`` now emits 3-tuple tiles
  ``(label, value, hint)``. Hints are sourced from a single
  ``_OVERVIEW_TILE_HINTS`` dict (one source of truth, makes
  adding a tile a single-place edit).
- ``_build_kpi_tiles`` (in ``dash_app/app.py``) renders each
  tile with a small ⓘ icon next to the label, a ``dbc.Tooltip``
  for the rich popover, and a native ``title`` attribute on an
  inner wrapper as the no-JS / screen-reader fallback. The
  cursor on the tile body changes to ``cursor: help`` so the
  affordance is obvious.
- Hints surface the bequest-tax-aware Terminal-NW formula, the
  Lifetime-NPV discount rate, the IRMAA tier mechanics, the
  Effective-CAGR caveat (NOT a pure investment return), and the
  Fisher-equation rule of thumb on Real CAGR.
- New ``.kpi-hint-icon`` and ``.kpi-hint-tooltip`` CSS rules in
  ``dash_app/assets/fira-code.css`` mirror the form-input
  styling so the dashboard's help vocabulary stays consistent
  across all tabs.
- Back-compat: ``_build_kpi_tiles`` still accepts the legacy
  2-tuple ``(label, value)`` shape — those tiles render without
  a tooltip surface.

**Tests:** ``test_dash_overview_growth.py`` extended with a
new ``TestOverviewTileHints`` class pinning that every tile in
both sections (and the MC-only tiles) carries a non-trivial
hint (≥ 40 chars), plus two new ``TestBuildKpiTiles`` cases
verifying the rendered ``dbc.Tooltip`` + native-``title``
fallback structure.

### Added — Dash app: Overview tab — growth-rate metrics + multi-strategy growth chart

**Why it matters:** the Overview tab used to show one bottom-line
tile, "Terminal after-tax NW", with no context for *what's in
that number* or *how the plan got there*. Two compounding gaps:

1. The Terminal after-tax NW formula is **bequest-tax-aware**
   (pretax + HSA discounted by `cfg.heir_marginal_rate`, Roth
   tax-free, taxable face-value via step-up basis at death) but
   that detail lived only in the metrics module's docstring.
2. There was no growth surface at all — no starting-NW
   comparison, no CAGR, no accumulation-vs-decumulation split,
   no per-year wealth trajectory.

This change adds:

- **Five new metric helpers** in `tax_optimizer/metrics.py`:
  `starting_after_tax_nw` (mirrors `terminal_after_tax_nw`
  symmetrically against `StartingBalances` so growth ratios
  aren't apples-to-oranges), `total_growth_multiplier`,
  `effective_cagr` (handles negative growth via signed-power
  root), `real_cagr` (Fisher-equation-based inflation
  adjustment), `stage_cagr` (splits at `retire_age`,
  matching `report.py`'s convention), and `nw_after_tax_series`
  (vectorized per-row version of the bequest-tax-aware terminal
  formula).
- `summarize()` extended with three optional kwargs
  (`starting_balances`, `inflation`, `retire_age`) and six new
  result keys (`starting_after_tax`, `total_growth_mult`,
  `effective_cagr`, `real_cagr`, `accumulation_cagr`,
  `decumulation_cagr`). Legacy callers without the new kwargs
  still work — the keys come back as `None`.
- **Sectioned KPI tiles**: `figures.overview_kpis` now returns
  `[(section_label, [(label, value), ...]), ...]` with two
  sections — **Outcomes** (Terminal NW, lifetime tax / IRMAA,
  peak marginal, MC stats) and **Growth** (Starting NW, Total
  growth multiplier "3.15×", Effective / Real / Accumulation /
  Decumulation CAGR with explicit `+` / `-` signs).
- **New multi-strategy growth panel**
  (`figures.multi_strategy_growth_panel`): two stacked
  subplots — after-tax NW per year (top) + YoY growth %
  (bottom, with a 0 % reference line) — for all four canonical
  strategies overlaid. Reuses the existing `_add_strategy_lines`
  helper so the chart inherits Okabe-Ito palette + marker +
  dash redundancy + winner-line emphasis from the Taxes /
  Conversions tabs.
- Inline documentation in `overview_kpis`'s docstring spelling
  out the bequest-tax-aware Terminal NW formula so a curious
  user can trace the number end-to-end without diving into
  `metrics.py`.
- `_cfg_summary` now carries `heir_marginal_rate` + `inflation`
  on the run payload so the Overview chart can reproduce the
  Terminal-NW lens consistently from a deserialized payload.

**Tests:**
- `tests/test_metrics_growth.py` — pure-function regression for
  every helper plus `summarize()` with full / partial / no
  kwargs.
- `tests/test_dash_overview_growth.py` — sectioned KPI shape,
  six growth tiles in canonical order, `+`/`-` sign rendering,
  NaN graceful fallback, palette / marker / dash redundancy on
  the chart, winner-line thickness, 0 % reference line, and an
  end-to-end `run_scenario` smoke test.

**Caveat surfaced in the tile labels:** the "Effective CAGR"
tile bundles market returns + contributions + withdrawals + tax
drag into one effective compounding rate — it's the right
answer to "at what rate did the plan compound?" but the wrong
answer to "what return did my portfolio earn?". The
accumulation-vs-decumulation split helps separate "growth from
contributions + market" from "drawdown from withdrawals".

### Changed — Dash app: Year-by-year table reorganized into functional groups + per-group coloring

**Why it matters:** the Year-by-year drill-down DataTable grew
organically over time. The 22-column list was *implicitly*
grouped via source-line breaks but the UI gave the user zero
visual cue for the groupings, and the order had drifted in a few
spots: `medicare_base_premium` came AFTER `irmaa` (the
surcharge-on-top read backwards), `spending_need` was sandwiched
between healthcare columns despite being a household-cashflow
target, and `agi` and `federal_tax` were split by `state_tax`
even though all three are tax-outcome metrics. The simulator
also emits a handful of high-value diagnostic columns
(`taxable_income`, `unfunded`, `qualified_dividends`,
`interest_income`) that the dashboard wasn't surfacing.

The drill-down table is now structured around eight functional
groups, each with a color-blind-safe Okabe-Ito tint and a
colored left-border divider on the first column of the group:

| # | Group | Columns | Hue |
|---|-------|---------|-----|
| 1 | Identity        | year, spouse_a_age, filing_status                                         | slate (default)         |
| 2 | Income          | wages, pension, ssn, qualified_dividends, interest_income                 | Okabe blue `#0072B2`    |
| 3 | Pretax events   | rmd, roth_conversion                                                      | Okabe orange `#E69F00`  |
| 4 | Withdrawals     | pretax_withdrawal, roth_withdrawal, taxable_withdrawal                    | Okabe yellow `#F0E442`  |
| 5 | Spending        | spending_need, unfunded                                                   | vermillion `#D55E00`    |
| 6 | Tax outcome     | agi, taxable_income, federal_tax, state_tax, bracket_pct                  | reddish-purple `#CC79A7`|
| 7 | Healthcare      | medicare_base_premium, irmaa                                              | sky-blue `#56B4E9`      |
| 8 | Balances (EOY)  | pretax_balance, roth_balance, taxable_balance, hsa_balance                | bluish-green `#009E73`  |

Total: 26 columns (was 22). The four newly-surfaced columns are
all already in the simulator output, no engine changes:

- **`taxable_income`** (Tax group) — distinct from AGI; this is
  the post-deduction line that actually drives the bracket.
  Useful for spotting "AGI looks fine but standard deduction
  is being phased out so taxable income is higher than expected".
- **`unfunded`** (Spending group) — the deficit-indicator
  diagnostic. Critical for spotting failed years where the
  withdrawal cascade couldn't fund the spending target. The
  most-requested missing column.
- **`qualified_dividends`** (Income group) — LTCG-rate portfolio
  yield, distinct from interest. Helps explain AGI build-up
  in years with no wages.
- **`interest_income`** (Income group) — taxable bond / cash
  yield. Same rationale as qualified_dividends; surfaces
  why the asset-location strategy matters.

#### Implementation details

- **`dash_app/figures.py`** — new structured declaration
  `_YEARLY_COLUMN_GROUPS` (8-tuple list of
  `(group_id, color_key, columns)`) is the single source of
  truth. `detail_columns()` now derives from this list (with
  `bracket_pct → marginal` translated via `_DISPLAY_TO_NATIVE`
  for filtering against the simulator-native frame).
  `filter_to_detail_cols()` is unchanged in behavior — same
  `marginal × 100 → bracket_pct` rename and dollar rounding.

- **`dash_app/figures.py:yearly_table_styles()`** — new helper
  that derives three Dash DataTable conditional-style arrays
  from the same `_YEARLY_COLUMN_GROUPS` declaration:
  - `style_header_conditional` — 23 rules (one per non-identity
    column) at `rgba(<hue>, 0.22)`. Strong enough that the eight
    group blocks are obvious at a glance, light enough that the
    column-name text stays readable.
  - `style_data_conditional` — 23 rules (mirror) at
    `rgba(<hue>, 0.08)`. The light tint carries the group cue
    down through every data row without competing with the
    digit text.
  - `style_cell_conditional` — 7 rules (one per non-identity
    group's first column) with
    `borderLeft="2px solid rgba(<hue>, 0.6)"`. Vertical break
    between groups without needing empty separator columns
    (which would have broken sort/filter).

- **`dash_app/figures.py:_hex_to_rgba()`** — tiny helper that
  converts the canonical Okabe-Ito hex codes to the rgba alpha
  tints used by the conditional styles. Avoids hand-mixing
  pastels and keeps the palette as the single source of truth.

- **`dash_app/layout.py:yearly_tab()`** — wires the conditional
  styles into the existing `dash_table.DataTable` via the three
  `style_*_conditional` props. The styles are computed once at
  layout-build time and Dash gracefully ignores rules whose
  `column_id` isn't in the data, so it's safe to declare every
  rule even though `_populate_yearly_table` may filter to a
  subset of the columns at runtime.

#### Tests — `tests/test_dash_yearly_table.py`

New test file (40 tests) pinning the contract so a future
refactor can't silently regress column order, group membership,
the simulator-native rename, or palette membership:

- **Group declaration invariants**: 8 groups in the documented
  order; only the identity group has no color; every color key
  resolves through `_OKABE_ITO`; each color used at most once;
  no column appears in two groups; per-group spot-check via
  `parametrize`.
- **`detail_columns()`**: returns simulator-native names (with
  `bracket_pct` translated back to `marginal`); total = 26;
  positional invariants (`year` first, balances last,
  `unfunded` immediately after `spending_need`,
  `medicare_base_premium` before `irmaa`, `agi` before
  `federal_tax`); 4 new columns present.
- **`filter_to_detail_cols`**: against a synthetic
  simulator-shaped DataFrame, the output preserves group order,
  renames `marginal → bracket_pct`, drops simulator extras
  (`fica`, `rmd_a`), rounds dollar columns, and gracefully
  handles missing columns.
- **`yearly_table_styles`**: returns the three conditional
  lists; header / data rules cover exactly the 23 non-identity
  columns; cell-border rules cover exactly 7 group-first
  columns; header alpha (0.22) is strictly greater than data
  alpha (0.08); every rgba color used in any rule parses back
  to an Okabe-Ito hex code (palette-membership pin); the
  first-column-of-each-group border uses the group's own
  color.
- **`_hex_to_rgba`**: parameterized known-input cases (case
  insensitivity, optional leading `#`, edge-case alpha values)
  + ValueError on invalid lengths.
- **`_DISPLAY_TO_NATIVE`**: `bracket_pct → marginal`; every
  display-side key actually appears in some group.
- **End-to-end**: a real `run_scenario(mode="four_strategies")`
  → `filter_to_detail_cols(...)` produces a DataFrame whose
  columns match the union of `_YEARLY_COLUMN_GROUPS` in display
  order — catches drift between the simulator's emitted columns
  and the dashboard's filter list.

### Fixed — Strategies tab: bar value labels truncated

**Why it matters:** the 3-subplot strategy comparison chart at
the bottom of the Strategies tab showed bar values like
`$17,4` (truncated `$17,454,663`), `$1` (truncated
`$18,626,834`), and `$8` (truncated `$83,378`). At realistic
Strategies-tab widths the full-precision dollar text simply
didn't fit alongside the bars, and the previous layout used
`textposition="outside"` with no x-axis padding so Plotly
clipped the labels at the subplot's right edge.

The fix applies the same anti-overlap pattern used for the
Monte Carlo histogram annotations:

- **`dash_app/figures.py:strategy_compare_panel`**:
  - **Abbreviated labels via `_abbrev_dollars`**:
    `$17,454,663` → `$17.5M`, `$83,378` → `$83K`. Cuts label
    width by ~60% so most labels now fit *inside* their bar.
  - **`textposition="auto"`** (was `"outside"`) so Plotly
    decides per-bar whether to render the label inside or
    outside based on bar width.
  - **X-axis range padded 18%** beyond each subplot's max
    value so any label that does spill outside has a place
    to land.
  - **`cliponaxis=False`** as belt-and-suspenders against
    very small bars (e.g. zero rows) whose label has to
    render outside.
  - **`horizontal_spacing=0.10`** between subplots so the
    longer subplot titles ("Lifetime federal tax (NPV)")
    don't crash into adjacent subplots.
  - **Font sizes pinned**: subplot title font = 13, bar
    text font = 12, chart title font = 15. Previously the
    subplot titles inherited Plotly's default (~12) which
    looked smaller than the dashboard's table chrome.
  - Total height bumped 380 → 400 px to accommodate the
    larger subplot titles.

- **`dash_app/figures.py:strategy_comparison`** (the
  single-metric variant used by Monte Carlo and ad-hoc
  callsites): same fix — abbreviated labels (with the
  `(+X.X%)` baseline-delta suffix preserved),
  `textposition="auto"`, `cliponaxis=False`, x-axis padded
  22% (slightly more than the panel's 18% to accommodate the
  percentage suffix), explicit chart title font.

### Tests — Strategies-tab bar chart regression suite

- `tests/test_dash_strategy_bars.py` (new file, 16 tests)
  pinning the anti-truncation invariants for both bar
  chart builders. The exact dollar values from the
  regression screenshot — `$17,454,663`, `$18,626,834`,
  `$1,978,147`, `$1,713,631`, `$83,378`, `$39,457` — are
  replayed as a fixture, so the specific case the user
  reported is now guarded.
  - **Abbreviated text invariant**: full-precision strings
    (`$17,454,663` etc.) are explicitly NOT present in any
    bar label; abbreviated forms (`$17.5M`, `$83K`) ARE.
  - **`textposition="auto"`** on every trace.
  - **`cliponaxis=False`** on every trace.
  - **X-axis range** extends 15-25% beyond the largest
    value on each subplot.
  - **Subplot title fonts** = 13, **chart title font** =
    15, **bar text font** = 12.
  - **NaN / None defensive case**: `strategy_comparison`
    handles missing-summary values without choking on the
    abbreviation helper.

### Fixed — Dash app: Monte Carlo histogram percentile labels overlap

**Why it matters:** the Monte-Carlo terminal-NW histogram has
three percentile callouts (P10/P50/P90) attached to vertical
guide lines. With long fully-formatted dollar amounts
(`$5,826,124`, `$18,189,053`) the callouts collided horizontally
and rendered as one tangled string at the top of the chart.
For typical retirement-planning runs the median sits well below
the mean (skewed distribution), so the P10 and P50 lines are
near each other in x — exactly the case the old layout failed.

- **`dash_app/figures.py`**:

  - New `_abbrev_dollars(v)` helper formats large dollar
    amounts compactly: `$5,826,124` → `$5.8M`,
    `$1.234B` → `$1.2B`, `$50,000` → `$50K`. Sub-$10K values
    fall through to full-precision (`$9,999`) because at that
    magnitude there's nothing to abbreviate. Used by the
    histogram annotations + chart title.

  - **Anti-overlap layout for the P10/P50/P90 annotations**
    in `mc_terminal_histogram`:
    - **Distinct horizontal anchors** — leftmost line
      (P10) anchors `xanchor=right` (label extends LEFT
      away from the line); middle (P50) anchors `center`
      (label sits ABOVE); rightmost (P90) anchors
      `xanchor=left` (label extends RIGHT). The three
      callouts now land in three different horizontal
      zones and never compete for the same slot.
    - **Vertical staggering** — P50 also gets
      `yshift=22` so it sits in a higher row than the side
      labels. Even in pathological cases (e.g. P10 ≈ P50
      lines very close in x), the P50 label slips above
      the side labels.
    - **Per-annotation `bgcolor`** at 88% white, plus a
      colored 1px border in each percentile's hue, so the
      callout text is legible against the histogram bars
      and the chart title above. The colored border doubles
      as a visual link between the label and its line.
    - Compact font (`size=11`) and slightly larger top
      margin (`t=80` vs the default 50) so the bumped-up
      P50 doesn't crash into the chart title.

  - The chart title now also abbreviates the CVaR(10%)
    readout via `_abbrev_dollars` so titles like
    `CVaR(10%)=$3,823,858` shorten to
    `CVaR(10%)=$3.8M` and stay on a single line at narrow
    viewport widths.

### Tests — Monte Carlo histogram regression suite

- `tests/test_dash_mc_histogram.py` (new file, 19 tests)
  pinning the anti-overlap invariants so a future refactor
  can't regress the layout:

  - **`_abbrev_dollars` parameter sweep** — millions
    (`$5.8M`), billions (`$1.2B`), tens-of-thousands
    (`$50K`), sub-$10K full-precision (`$9,999`),
    zero (`$0`), negatives (`$-5.8M`).
  - **Three percentile annotations present** with
    abbreviated text and the canonical un-abbreviated
    full-precision strings (`$5,826,124` etc.) explicitly
    NOT present.
  - **`xanchor` spread** — three different anchors
    (`right` / `center` / `left`) so the labels splay
    out into different horizontal zones.
  - **`yshift` staggering** — P50's yshift is strictly
    greater than P10's and P90's so the middle label
    sits in a different vertical row.
  - **`bgcolor`** is set on every annotation (legibility).
  - **Title CVaR is abbreviated** (`$3.8M` not
    `$3,823,858`).
  - **Top margin extended** to ≥70 px so the bumped-up P50
    callout has clearance from the chart title.

  The screenshot from the bug report — P10=$5,826,124,
  P50=$18,189,053, P90=$43,815,711 — is replayed verbatim
  as a fixture, so the exact regression case is now
  guarded.

### Changed — Dash app: color-blind-friendly palette + shape redundancy

**Why it matters:** ~8% of men and ~0.5% of women have some form
of color blindness. The previous palettes — Tailwind-style
`green-500` / `orange-500` / `violet-500` / `slate-500` for
strategies, plus assorted `red-500` / `amber-500` accents
elsewhere — fail multiple times for the most common form
(deuteranopia): green vs orange vs red are easily confused, and
several charts (Monte Carlo percentile lines, single-strategy
tax/conversion panels) used the canonical "red = bad / green =
good" mapping that a deuteranope simply cannot resolve. We've
reworked color usage across every figure in the dashboard to
follow color-universal-design (CUD) principles.

- **Adopted the Okabe-Ito palette** (Okabe & Ito 2008,
  https://jfly.uni-koeln.de/color/) as the canonical color set
  for every figure. Okabe-Ito is the de-facto scientific-
  publishing standard for accessibility — its eight hues are
  pairwise-distinguishable across deuteranopia, protanopia,
  AND tritanopia, and each color also carries a distinct
  *luminance* so the palette also degrades gracefully under
  achromatopsia and B&W printing.

- **Shape redundancy on multi-strategy charts**
  (`dash_app/figures.py`):
  Strategy is now encoded with **three redundant cues**, not
  one. Even users with total color blindness can distinguish
  the four canonical strategies:
  - **Color** (Okabe-Ito hue): S0 = blue, S1 = reddish-purple,
    S2 = orange, S3 (winner) = bluish-green.
  - **Marker symbol**: S0 = circle, S1 = square, S2 = diamond,
    S3 = triangle-up. Each shape stays unambiguous at the
    4.5 px marker size we use.
  - **Line dash pattern**: S0 = solid, S1 = dash, S2 = dot,
    S3 = solid. The two anchor strategies (baseline + winner)
    stay solid so the eye can track them first; the
    alternatives use distinct broken patterns so they're
    distinguishable from the anchors and from each other.
  - With these three cues a deuteranope, a protanope, a
    tritanope, AND someone printing the dashboard on a B&W
    printer can all read the chart correctly.
  - Three new helpers — `_color_for`, `_marker_for`,
    `_dash_for` — pull from `_STRATEGY_COLORS`,
    `_STRATEGY_MARKER`, `_STRATEGY_DASH` respectively, with
    safe fallbacks for non-canonical strategy names.

- **Migrated every other palette in `figures.py` to Okabe-Ito**:
  - `balance_stack` (account-type palette): pretax = blue, Roth
    = bluish-green (matches the optimizer hue, since tax-free
    growth is the household's most valuable bucket dollar-for-
    dollar), taxable = orange (warm = drag from annual yield),
    HSA = reddish-purple. Replaces sky-500 / green-500 /
    amber-500 / purple-500.
  - `taxes_panel` (single-strategy tax components): AGI = blue
    (solid), federal tax = vermillion (solid, replaces the
    previous `#ef4444` red — vermillion is more
    deuteranopia-distinguishable from green), state tax =
    orange (dashed), IRMAA = reddish-purple (dotted), marginal
    bracket = black. Each component also has a distinct dash
    pattern as a redundant cue.
  - `conversion_panel` (single-strategy bars): conversion =
    bluish-green ("the proactive move"), RMD = orange ("the
    forced move"), liquidity-cap = vermillion ✕ marker. The
    `✕` symbol is itself a non-color cue (no other trace on
    that chart uses it).
  - `mc_terminal_histogram` percentile vertical lines: P10 =
    vermillion (worst case), P50 = black (median), P90 =
    bluish-green (best case). Replaces the previous
    red/black/green which fails for deuteranopia. Dash style
    on each line was already distinct.
  - `mc_fan_chart`: P50 = black solid, P10 = vermillion +
    `dot` dash, P90 = bluish-green + `dash` dash, P10-P90
    band = Okabe-Ito blue at 18% alpha. Distinct dash patterns
    are the second redundant cue — even in monochrome the
    user can tell which percentile is which.
  - `strategy_comparison` and `strategy_compare_panel`
    horizontal bars: now use `_color_for` per strategy, so
    every strategy keeps the same color across every chart.
    Previously the comparison bars were a special case
    ("blue if not S3, green if S3") which broke palette
    consistency between the strategy table and the per-year
    timelines.

- **Strategy comparison table icon prefixes**
  (`dash_app/assets/fira-code.css`):
  The table cells used yellow tint for "knob changed" and green
  tint for "optimizer override". For an achromatope both pastels
  read as off-white. Each cell class now also gets a CSS
  `::before` icon prefix:
  - `.strategy-cell-changed::before` → ▲ in amber-800
  - `.strategy-cell-optimized::before` → ★ in emerald-800

  The icons render via `::before` so they show in the visual
  table but aren't part of the cell's text content — exporting
  / copying the table data still produces clean strings. The
  pre-existing 2 px left-border + font-weight step (500 → 600)
  remain as additional non-color cues.

### Tests — color-blind regression suite

- `tests/test_dash_taxes_multi.py` — 5 new tests pinning the
  CB-redundancy invariants so a future refactor can't silently
  regress accessibility:
  - **Palette membership**: every entry in `_STRATEGY_COLORS`
    must be drawn from `_OKABE_ITO`, and the canonical
    Okabe-Ito hex codes themselves are spot-checked
    (`#0072B2` blue, `#009E73` bluish-green, `#E69F00`
    orange, `#CC79A7` reddish-purple, `#D55E00` vermillion).
  - **Marker uniqueness**: every canonical strategy must have
    a distinct marker symbol; the four pinned shapes
    (circle / square / diamond / triangle-up) must all
    appear.
  - **Dash uniqueness**: baseline and winner must be solid;
    the two alternative strategies must each be non-solid
    AND distinct from each other (so the four lines are
    unambiguous in monochrome).
  - **Rendered traces apply per-strategy marker**:
    `multi_strategy_taxes_panel` actually puts the right
    `marker.symbol` on every Plotly trace per panel, not
    just stores it in a dict. Every panel for a given
    strategy renders with the *same* symbol (so the
    legend stays consistent), and the four canonical
    strategies render with *four different* symbols (so
    the legend is unambiguous in monochrome).
  - **Rendered traces apply per-strategy dash**: same idea
    for `line.dash`.

### Changed — Dash app: Taxes tab plots all four strategies

**Why it matters:** the Taxes tab previously rendered AGI / federal
tax / state tax / IRMAA / marginal-bracket only for the **winning**
strategy. To compare strategies the user had to switch between the
Year-by-year tab and re-pick from the dropdown. The natural visual
question — "in year X, which strategy paid less federal tax?" —
required eyeballing four separate runs. The Strategies tab already
has a side-by-side outcomes table, but the Taxes tab is where the
*per-year* curves live, and curves only really compare when they're
overlaid on the same axis.

The Taxes tab now overlays **all four canonical strategies on
every panel**, color-coded by strategy, with the optimizer (S3)
pulling the strongest hue (green) so the winner pops on first
glance.

- **`dash_app/figures.py`** — two new builders:
  - `multi_strategy_taxes_panel(strategies, *, title=...)` —
    stacked subplots, one per metric (AGI / federal tax /
    optionally state tax / optionally IRMAA / marginal bracket),
    with one color-coded line per strategy. Subplots that have
    zero signal across every strategy (state tax for stateless
    households, IRMAA for pre-65 horizons) are silently skipped
    so the figure stays compact (~180px per active panel +
    ~80px chrome). Marginal bracket renders as a percentage
    (`y_scale=100, ticksuffix="%"`).
  - `multi_strategy_conversion_panel(strategies, *, title=...)`
    — two stacked subplots (Roth conversion + RMD), one line
    per strategy each. The liquidity-cap markers (small ✕) now
    render per-strategy in that strategy's color so the user
    can tell which strategy was constrained by the
    `tax_paying_capacity` guard rather than just "somebody got
    capped here".
  - Helper functions `_color_for`, `_stable_strategy_order`,
    `_x_for`, `_add_strategy_lines` sit at module scope so the
    tests can pin them. The strategy → color palette is a
    single `_STRATEGY_COLORS` dict (slate / violet / orange /
    green) chosen for color-blindness-safe contrast.
  - Both builders accept either a deserialized DataFrame
    (notebook callers) or the runner's serialized payload
    (Dash callbacks) on each strategy's `df` field. Detection
    is `isinstance(s["df"], pd.DataFrame)`; deserialization is
    deferred via a local import of
    `dash_app.runner.deserialize_strategy_df` so `figures.py`
    stays importable without Dash.
  - Legend grouping (`legendgroup=name`) ties every trace for
    a strategy to a single legend entry — clicking the
    "S3_optimized" name in the legend toggles all of S3's
    panels at once. Without this, 12+ legend entries would
    overflow the chart and clicking one wouldn't visually
    isolate "the optimizer".
- **`dash_app/app.py:_render_taxes`** — now passes the full
  `strategies` dict to the multi-strategy builders. The winner's
  name still surfaces in the chart titles (`Taxes & marginal
  bracket — winner: S3_optimized`) so the user sees the verdict
  even before parsing the colors.
- The single-strategy `taxes_panel` and `conversion_panel`
  builders are kept unchanged — they're still the right shape
  for notebook contexts and the canvas-style "I want one
  strategy in detail" surfaces. The Dash app just doesn't use
  them anymore.
- **`dash_app/layout.py:taxes_tab`** — wraps each `dcc.Graph`
  in a `tax-figure my-3` className so the new
  `assets/fira-code.css:.tax-figure` rule gives both panels a
  light slate-200 (`#e2e8f0`) border + 6px radius + 6px
  internal padding + white background. Matches the card
  vocabulary already used by the report iframe and strategy
  comparison table headers, so the Taxes tab now reads as
  "two cards" rather than "two SVGs floating on the page".
  An `html.Hr` styled via `.tax-figure-divider` (slate-300,
  `1.5rem` vertical margin) sits between the two cards as a
  deliberate section break — the two charts answer different
  questions ("what tax was paid?" vs "what conversion / RMD
  activity drove that?") and benefit from being separated
  rather than read as one tall scroll region.

- **Readability pass on the multi-strategy charts**
  (`dash_app/figures.py`):

  - **Metric labels moved to y-axis titles**, replacing the
    old `subplot_titles` annotations. Annotations sit at
    y≈1.0 of each subplot and were colliding with peak trace
    values in late retirement (e.g. AGI / federal tax / Roth
    conversion plateau). Y-axis titles live in the left
    gutter where there's always room. New labels:
    `AGI ($)`, `Federal tax ($)`, optional
    `State tax ($)` / `IRMAA ($)`, `Marginal (%)`,
    `Roth conversion ($)`, `RMD ($)`.
  - **Vertical spacing** bumped (`taxes_panel`: 0.045 → 0.085;
    `conversion_panel`: 0.07 → 0.10) so adjacent panels read
    as distinct without the annotation crutch.
  - **Per-panel height** bumped: taxes panel from 180 → 220
    px per panel (~1200 px total at 5 panels) and conversion
    panel from 560 → 660 px total. Year-over-year deltas are
    now legible at the 30+ year horizons typical for
    retirement plans.
  - **Mode is `lines+markers`** instead of `lines` only, with
    a 4.5 px marker. Pins individual years visually — useful
    around the RMD ramp-up at age 75 and around years where
    multiple strategies overlap on the same path (without
    markers, S0 + S1 flat at zero in the conversion panel
    look like one strategy).
  - **Winner gets a heavier stroke** (3.25 px vs 2.25 px for
    alternatives) and is **drawn last** so the winning trace
    sits on top of the z-stack rather than being buried under
    whichever strategy it tracks closely (frequently S2 in
    the conversion panel). The callback now passes
    `winner_name` through to both builders.
  - **Liquidity-cap markers** scaled up from size 9 → 11 with
    a slightly heavier outline, so the `✕` reads at chart
    scale rather than getting lost in the line.
  - Wider left margin (60 → 80 px) to accommodate the new
    y-axis titles + dollar tick labels without crowding the
    chart area.

### Tests — Dash multi-strategy Taxes-tab regressions

- `tests/test_dash_taxes_multi.py` — 18 new tests, gated on
  `pytest.importorskip("dash")`. Coverage:
  - **Palette / ordering**: each canonical strategy has a
    unique color; unknown strategy names fall through to a
    neutral fallback hue (so we can spot custom runs visually);
    `_stable_strategy_order` pins canonical strategies S0→S3
    before any custom keys regardless of input iteration order
    and is idempotent.
  - **`multi_strategy_taxes_panel`**: empty payload returns the
    placeholder; one trace per strategy per active panel (4×3
    in the default fixture); state-tax / IRMAA panels add
    themselves only when at least one strategy has a non-zero
    column; zero-signal panels are silently skipped (no flat
    zero-line clutter); the legend shows exactly N strategies,
    not N×panels (so legend clicks toggle every panel for that
    strategy at once); `marginal` is rendered as a percentage
    (× 100, suffix `%`); every trace for a given strategy
    shares one color.
  - **`multi_strategy_conversion_panel`**: two-panel × N-strategy
    line layout; liquidity-cap markers render only on years
    where the guard fired and only for the affected strategy
    (S3 in the fixture); no markers when no strategy was
    capped; legend shows once per strategy.
  - **End-to-end**: a real
    `run_scenario(mode="four_strategies", seed=0)` →
    `serialize_run_result` → builders walk to catch column-name
    drift between simulator output and the column constants
    hard-coded in the figure builders (`agi`, `federal_tax`,
    `marginal`, `roth_conversion`, `rmd`,
    `roth_conv_capped_by_liquidity`).

### Docs — Dash app: how-to-run guide

**Why it matters:** the README's "Web app (Plotly Dash)" section
was a single nine-line code block — fine as a quick-reference
once you knew the dashboard, but useless for a first-time user
who needs to know what each tab does, how the run modes differ,
which workflow to follow for "I want a printed action plan", or
how to debug the inevitable "address already in use" port clash.
The README also referenced a `pip install -e ".[dash]"` extra
that doesn't actually exist in `pyproject.toml` (`dash`,
`dash-bootstrap-components`, and `plotly` are listed under the
base `dependencies`), so users following the README would `pip`-
install with a silently-ignored extra and get the same result
as a plain `pip install -e .`.

- **`docs/dashboard.md`** (new, ~485 lines) — full how-to-run
  guide covering:
  - **Quickstart** — clone / venv / `pip install -e .` /
    `python -m dash_app`.
  - **Launching the dashboard** — module entry point, console
    script (`tax-optimizer-app`), `--host` / `--port` / `--debug`
    flags, and the `DASH_HOST` / `DASH_PORT` environment
    variables.
  - **The UI at a glance** — ASCII diagram of the two-column
    sidebar / results-panel layout.
  - **Sidebar walkthrough** — top bar (Load JSON drop zone,
    Download JSON, run-mode dropdown, Run button, status banner),
    Simple / Advanced form tabs (with a per-field help-tooltip
    explainer and the decimal-percent convention call-out), and
    the run-mode selector with a wall-time table.
  - **Results panel walkthrough** — Overview, Taxes, Strategies
    (including the per-knob optimizer-overrides table), Monte
    Carlo, Year-by-year, and Report tabs with a description of
    what each surfaces.
  - **Common workflows** — "I just want to play with my numbers",
    "I have a saved scenario JSON", "I want to share / archive
    the current scenario", "I want a printed action plan", "I
    want to debug a specific year", "I want to develop on the
    dashboard itself".
  - **Tips and shortcuts** — per-field hints, monospace UI,
    persistence (the form does NOT auto-save across reloads),
    run cache, decimal-percent reminder, run-button stickiness.
  - **Troubleshooting** — port-in-use, run-cache eviction, blank
    iframe (tornado sweep still in flight), red form text, charts
    not auto-updating, Dash deprecation warnings, `No module
    named dash_app` (forgot to `pip install -e .`).
- **`README.md`** — the "Web app (Plotly Dash)" section now:
  - Drops the misleading `pip install -e ".[dash]"` reference
    in favor of the correct `pip install -e .` (Dash is a base
    dep).
  - Adds a short "useful flags" cheat-sheet (`--port`, `--host`,
    `--debug`, `tax-optimizer-app`).
  - Calls out the **Report** tab and **Download HTML** button
    explicitly so users know the action plan is available
    inline, not just as a download.
  - Links to `docs/dashboard.md` for the deep dive.
- **`README.md`** — the Documentation table at the bottom of the
  file now lists `docs/dashboard.md` alongside the other docs
  (architecture, scenario_guide, market_models, roth_conversion,
  CHANGELOG).

### Added — Dash app: per-field help tooltips on every form input

**Why it matters:** the scenario form has 118 fields spanning
household basics, contribution mechanics, tax-regime knobs, market
parameters, mortality, asset location, spending profile, IRA /
mega-backdoor, pension, and health premiums. Many of them use
domain abbreviations (PIA, FRA, IRMAA, RMD, NRD, MAGI, FICA, CAPE,
J&S, MFJ, S125) that mean nothing to a first-time user, and the
labels are short by design (≤ ~30 chars) so they fit the sidebar.
Without tooltips the only way to learn a field's meaning was to
read the source.

The form now exposes each field's help text in three layers:

1. A small ⓘ icon next to every label — the discoverable
   surface that signals "hover for help".
2. A Bootstrap `dbc.Tooltip` anchored to the icon. Instant-show
   (150ms delay), styled with a wider 360px max-width and 1.45
   line-height so two-sentence hints read cleanly.
3. The native HTML `title=` attribute on both the label and the
   icon — a screen-reader / no-JS fallback that surfaces the same
   text via the browser's default tooltip.

- **`dash_app/forms.py` — populated `help=` on all 118
  `FormField` entries.** Hints are concise (≤ 360 chars, single
  sentence preferred) and explain *what the field is*, *what unit
  it's in*, and *the effect of changing it*. Domain abbreviations
  are spelled out at first use ("Primary Insurance Amount at Full
  Retirement Age", "Required Minimum Distribution", "Modified
  Adjusted Gross Income", etc.). Decimal-percent fields explicitly
  call out the convention ("0.10 = 10%") because the form already
  appends "(decimal, e.g. 0.07)" to percent labels.
- **`dash_app/layout.py` — new `_help_components(fld)` helper**
  returns a 3-element list (NBSP separator, ⓘ span, `dbc.Tooltip`)
  to splat into the label's children, or `[]` if the field has
  no help. The icon's id is built by `_hint_id(path)` which
  replaces dots with double underscores so the resulting CSS
  selector (`#hint-config__market__equity_mu`) is universally
  valid (raw dotted ids trip Bootstrap's tooltip on some
  browsers). `_field_row` keeps the existing `title=help`
  attribute on the label as a no-JS fallback.
- **`dash_app/assets/fira-code.css` — new `.form-hint-icon` and
  `.form-hint-tooltip` rules.** Icon is muted gray with a
  `cursor: help` affordance and a subtle blue hover state to
  signal interactivity. Tooltip max-width bumped from
  Bootstrap's default 200px to 360px and line-height to 1.45 so
  paragraph-shaped help reads cleanly. Tooltip text drops Fira
  Code in favor of the system sans-serif stack — body copy
  reads materially faster in proportional fonts than monospace,
  and tooltip text is purely prose (no aligned numbers).

### Tests — Dash form-help coverage

- `tests/test_dash_form_help.py` — 9 new tests, gated on
  `pytest.importorskip("dash")`. Coverage:
  - **Schema-level**: every `FormField` in `FIELD_SCHEMA` has a
    non-empty `help`; no hint exceeds 360 characters (Bootstrap
    tooltip max-width budget); no hint is a verbatim copy of
    its label (the most common no-info anti-pattern).
  - **Rendering**: `_help_components` produces exactly one icon
    + one tooltip pair with matching ids; the icon's
    `title=help` fallback is set; `_hint_id` strips dots from
    dotted paths.
  - **Negative case**: a hand-rolled field with `help=None` does
    NOT render an icon — defensive guard against shipping
    orphaned tooltip targets that throw a console error.
  - **Label fallback**: the underlying `<label>` element exposes
    `title=help` for the no-JS / screen-reader path.
  - **Smoke**: `_field_row` doesn't blow up on any of the 118
    schema entries (catches future kwarg drift on `dbc.Tooltip`
    / `html.Span`); every hint id is globally unique.

### Added — Dash app: Strategies tab shows optimizer overrides

**Why it matters:** the Strategies tab previously displayed the
winner's seven decision-axis values in a one-line "Optimizer picks"
callout, but the user couldn't see at a glance *which* knobs the
optimizer chose to override versus the canonical reference
strategies (S0 baseline, S1 all-Roth-401(k), S2 fill-to-22%-bracket).
The bar chart below the callout shows outcome deltas (terminal NW,
lifetime tax, lifetime IRMAA) but says nothing about the *inputs*
that produced them.

The tab now renders a per-knob × per-strategy comparison table that
makes overrides visually obvious, then shows the existing outcome
chart below it. Cells whose value differs from the baseline column
get a yellow tint; cells in the optimizer column that differ from
baseline get a stronger green tint and bold weight so "what did the
optimizer change?" pops on first read.

- **`dash_app/app.py`** — new `_strategy_compare_table` helper
  builds a `dbc.Table` with two row groups:
  - **Decision parameters** (7 rows): Roth-401(k) % per spouse,
    Roth conversion target bracket, after-tax 401(k) % per spouse,
    SS claim age per spouse. Cell-level diff highlight against the
    baseline column.
  - **Outcomes** (4 rows): terminal after-tax NW, lifetime federal
    tax NPV, lifetime IRMAA NPV, peak marginal rate. Diff
    highlighting *not* applied here — every strategy's NW differs
    in $ so coloring would just paint every cell. Outcome rows are
    purely for visual correlation with the parameter rows above.
  Headers use brief friendly labels (`Baseline`, `All Roth`,
  `Fill to 22%`, `Optimizer ★`) with the canonical strategy id as
  a small subtitle so users can still match against the CLI / docs.
  The optimizer's column gets a green star next to the label. The
  raw key is used for any custom strategy name not in the
  canonical four (no silent label substitution).
  Helper functions `_fmt_value`, `_approx_equal` (1e-9 tolerance
  to absorb float wobble through the JSON round-trip), and
  `_pick_baseline` (prefers `S0_baseline` → `S1_all_roth_401k` →
  `S2_bracket_fill_22` → first column) are also exposed at module
  scope so the tests can pin them.
- **`dash_app/app.py`** — the existing `_strategy_callout`
  (one-line "Optimizer picks" summary) is kept as the
  **single-strategy fallback**: when the run mode is `single`,
  the comparison table degenerates to a single column and there's
  nothing to diff against, so we render the old callout instead.
  No regressions for users running single-mode reports.
- **`dash_app/layout.py`** — the `strategies-callout` Div was
  renamed to `strategies-comparison` (and its docstring updated
  to describe the new two-section layout). The callback Output
  follows.
- **`dash_app/assets/fira-code.css`** — added two cell-shading
  classes (`strategy-cell-changed` for the yellow tint and
  `strategy-cell-optimized` for the green tint), a section-header
  divider rule for the "Decision parameters" / "Outcomes" group
  separators, and tighter row padding so the full 11-row table
  fits above the fold on a 1366×768 laptop. Background colors
  are pulled from Bootstrap's `bg-warning-subtle` / `bg-success-
  subtle` palettes so the highlights read consistently against
  the rest of the dashboard's neutral grays.

### Tests — Dash strategy-compare table

- `tests/test_dash_strategy_compare.py` — 28 new tests, gated on
  `pytest.importorskip("dash")`. Coverage includes:
  - Pure helpers: `_fmt_value` (percent / money / int / None /
    pass-through), `_approx_equal` (within-tolerance equality),
    `_pick_baseline` (fallback ordering S0 → S1 → S2 → first
    column).
  - Table builder: empty placeholder, single-strategy fallback to
    the callout, four-strategy table headers / row labels,
    cell-highlight counts (3 'changed' body cells + 1 legend chip
    for the synthetic fixture; 2 'optimized' body cells + 1
    legend chip), and the regression guard that outcome rows
    don't get diff-highlighted (would paint every NW cell).
  - Winner-following: the green optimized class follows the
    runner's `winner_name`, not a hardcoded `S3_optimized`.
  - Custom strategy names: unknown keys render with the raw key
    rather than a silent canonical substitution.
  - End-to-end: a real `run_scenario(mode="four_strategies")` →
    `serialize_run_result` → `_strategy_compare_table` walk to
    catch drift between the runner's `_cfg_summary` keys and the
    parameter rows hard-coded in the table builder.
  Tests inspect the component tree by serializing through
  `to_plotly_json` so they don't need a Dash dev server or a
  browser.

### Changed — Dash app: switch the dashboard UI to Fira Code (monospace)

**Why it matters:** the dashboard renders a lot of money / percent /
year values across the form, KPI tiles, and Year-by-year DataTable.
With the previous proportional system stack (`-apple-system, Segoe
UI, …`), digit columns shifted around as the user typed and the
KPI rows didn't visually line up. Switching to a monospace ("fixed
font") face — Fira Code — makes every figure column-align by glyph
position, so the dashboard reads like a spreadsheet instead of a
prose document.

- **`dash_app/app.py`** — added the Fira Code Google Fonts URL to
  `external_stylesheets` (weights 400/500/600 only, ``display=swap``
  to avoid a flash of invisible text) and rewrote `app.index_string`
  to inject `<link rel="preconnect">` for `fonts.googleapis.com` and
  `fonts.gstatic.com`. Without the preconnect the browser opens
  TCP+TLS to gstatic only after parsing the stylesheet, which adds
  ~200ms to first paint on a cold load.
- **`dash_app/assets/fira-code.css`** (new) — Dash auto-loads any
  CSS file under `dash_app/assets/`, so this is where the actual
  font rules live. Applies the stack to `body`, `.form-control`,
  `.btn`, `.nav-tabs`, `.dash-table-container`, `.kpi-tile`, etc.,
  with sensible monospace fallbacks (`Fira Mono` → `JetBrains Mono`
  → `SFMono-Regular` → `ui-monospace` → `Menlo` → … → `monospace`).
  Disables Fira Code's programming ligatures (`>=` becoming `≥`) via
  `font-variant-ligatures: none` and `font-feature-settings: "calt"
  0` because they're distracting in form labels and KPI text.
  Enables tabular-numerals (`tnum`) on numeric inputs and
  DataTables for belt-and-suspenders digit alignment. Bumps
  `line-height` slightly because Fira Code's taller x-height makes
  the default Bootstrap row spacing feel cramped, and trims KPI
  tile font-size by a hair so monospace dollar amounts don't wrap.
- **Inline iframe placeholder srcDoc** in
  `dash_app/app.py:_placeholder_srcdoc` and the empty-state srcDoc
  in `dash_app/layout.py:report_tab` — both inline-load Fira Code
  from Google Fonts with their own `<link>` tags. Iframes are
  document-isolated so they don't inherit the parent page's
  `external_stylesheets`; the inline imports keep cold-state
  messages typographically continuous with the surrounding
  dashboard.

**What's NOT styled by this change:** the action-plan report
rendered inside the Report-tab iframe. That HTML carries its own
`<style>` block from `tax_optimizer/render.py` (sans-serif body +
SFMono code), unchanged so the in-dashboard view and the
downloaded HTML / PDF look identical to the CLI's output. The
`fira-code.css` rules can't reach the iframe document, so this
separation is enforced by the browser's same-origin styling
boundary.

### Added — Dash app: Report tab auto-renders the action plan

**Why it matters:** the previous flow had users click a "Download HTML"
button to get the action-plan report, which produced a downloaded file
that they then had to open in a separate browser tab. Most users
expect to *see* the report in the dashboard the moment they click the
"Report" tab — the download is for archiving / sharing later.

The Report tab now renders the action plan in an inline `<iframe>` the
moment the user activates it, with no extra click required. The
"Download HTML" button stays in the same tab for users who want to
save a copy, but it no longer drives the iframe state.

- **dash_app/app.py — new `_render_report_tab` callback** triggered
  by `results-tabs.active_tab` *and* `run-result.data` so:
  - Switching to the Report tab right after a Run shows the fresh
    report immediately.
  - Re-running while the tab is already active swaps the iframe to
    the new run's report.
  - Switching away to a different tab and back doesn't pay for a
    rebuild — see the caching note below.
  The callback gates on `active_tab == "tab-report"` and returns
  `no_update` for every other tab so we don't burn cycles on the
  tornado-sensitivity sweep when nobody is looking.
- **dash_app/app.py — `_download_report` simplified.** The previous
  version had three Outputs (download payload + status banner +
  iframe srcDoc); the iframe Output moved to the new tab callback so
  exactly one callback writes to `report-iframe.srcDoc`. No more
  `allow_duplicate=True` ordering hazard.
- **dash_app/report_builder.py — `CachedRun` dataclass with memoized
  markdown / HTML.** The expensive piece of the report build is the
  tornado-sensitivity sweep (~1-2s on the default scenario, since it
  re-simulates the household for every perturbed knob). Now the sweep
  runs once per cached run, and both the iframe-render path and the
  download-file path share the cached HTML through
  `_build_markdown` / `_build_html` keyed by
  `(year_table_scope, scenario_path)`.
  - `cache_run(cfg, inputs, rr)` now stores a `CachedRun` instead of
    a raw 3-tuple. Per-run state (cfg / inputs / rr) is unchanged;
    the dataclass just hangs the two memoization dicts on the same
    object so the LRU eviction reclaims them together.
  - `get_cached(run_id)` returns the `CachedRun` directly. (Tests
    that previously used tuple unpacking — `cfg, inputs, rr =
    cached` — were updated to `cached.cfg`, `cached.inputs`,
    `cached.rr`.)
  - `build_html_payload(...)` accepts either the new `CachedRun`
    (preferred — shares the cache) or the legacy 3-arg shape
    `(cfg, inputs, rr)` (kept for backward compatibility with
    notebooks / external callers; builds an ephemeral `CachedRun`).
  - `build_inline_html(cached)` is the new entry point used by the
    iframe callback. Returns the same HTML string the download
    payload's `content` field carries, so users get an identical
    document on both surfaces.
- **dash_app/layout.py — `report_tab()` reorganized.** The "Download
  HTML" button kept its id / behavior. Above the iframe we added a
  small grey hint ("The action plan renders automatically below
  when you switch to this tab after a Run.") so the auto-render
  is discoverable without forcing the user to read the changelog.
  The iframe sits inside `dcc.Loading` so the user sees a spinner
  during the first render's tornado sweep instead of staring at a
  blank pane.

### Tests — Dash report-tab rendering

- `tests/test_dash_report_download.py` extended (was 12 tests, now
  21):
  - `TestBuildInlineHtml::test_inline_html_matches_download_content`
    — proves the iframe and the download share their source.
  - `TestBuildInlineHtml::test_second_call_hits_cache` — patches
    `dash_app.report_builder.tornado_sensitivity` and asserts the
    sweep runs *exactly once* across two `build_inline_html` calls
    + one `build_html_payload` call. Catches a regression where
    the cache key drift would silently re-run the sweep on tab
    re-visits.
  - `TestBuildInlineHtml::test_different_scopes_cached_separately`
    — `year_table_scope="full"` and `="retirement"` cache under
    distinct keys so toggling scopes doesn't return stale HTML.
  - `TestAppCallbackGraph` — boots `make_app()` and walks the
    `callback_map` to verify (a) the Report-tab callback is
    registered, (b) `report-iframe.srcDoc` has *exactly one*
    writer (regression guard for the duplicate-Output trap that
    forced `allow_duplicate=True`).
  - `TestBuildHtmlPayload::test_cached_run_overload` and
    `test_two_arg_call_raises` cover the new overloaded
    signature.
  Existing tests that did `cfg, inputs, rr = get_cached(run_id)`
  were updated to use the `CachedRun` attributes; the public LRU
  semantics are otherwise unchanged.

### Fixed — Dash app: input boxes turn red on legitimate values

**Why it matters:** percent and currency fields rendered with red text
in the browser for inputs the user knew were valid (e.g. a Roth-401(k)
percentage of `0.07`, or a taxable balance of `460,270.90`). Looked
like a validation error in the simulator, but every value re-loaded
fine, ran fine, and produced the expected report — the red was purely
a visual lie.

The cause is HTML5's `<input type="number">` step-validation rule:
the browser treats the `step` attribute as a *constraint* rather than
a spinner-increment hint, and an input fails `:invalid` when
`(value - min) % step != 0`. JavaScript / IEEE-754 floats turn
that math into a minefield:

- **Percent fields** rendered with `step=0.005` to give arrow-key
  spinners a 0.5% increment. But `0.07 / 0.005 = 14.000000000000002`
  in float arithmetic, not 14, so the browser flagged 0.07 as
  off-step. Same for 0.135, 0.0125, etc. Any percent field whose
  decimal representation isn't an exact binary fraction tripped it.
- **Number fields without an explicit step** fell through to the
  HTML5 default of `step=1`, marking every dollar amount with a
  fractional component as `:invalid`. Even an explicit `step=1000`
  (used in the schema as a UX hint for "round-thousand spinner")
  did not save us — real-world balances like `taxable_brokerage =
  $460,270.90` and `pension_balance = $416,741.12` are obviously
  not exact multiples of $1k.

Browsers (Chrome, Safari, Firefox) all paint `:invalid` numeric
inputs with a red ring; combined with the `form-control form-control-sm`
Bootstrap class, the visible result was red text on every plausibly
populated form.

- **Fix (dash_app/layout.py:_input_for)** — render `percent` and
  `number` kinds with `step="any"`, which disables HTML5 step
  validation entirely while keeping `min` / `max` bounds intact.
  `int` kinds keep `step=1` (or the schema's explicit step) because
  integer-only is intentional there (ages, year offsets, horizon).
  The schema's per-field `step=1000` / `step=100` / `step=50`
  values are now what they should always have been — advisory
  spinner hints, not validation rules — and the renderer ignores
  them for `number` kinds in favor of `step="any"`. Spinner
  arrow-key increments default to 1, which is fine for currency
  entry where most users type the value rather than spinning.
- **Bounds preserved** — `min` / `max` still pass through to the
  rendered input, so e.g. `inputs.spouse_a_age_start` (min=18,
  max=100) still flags 5 or 200 as invalid. Only the off-step
  `:invalid` triggers were defused.

### Tests — Dash form-input regression suite

- `tests/test_dash_form_inputs.py` — 10 new tests, gated on
  `pytest.importorskip("dash")`. The test harness invokes the
  `_input_for` renderer directly per schema field rather than
  standing up `make_app()` (which would drag in `dash_table.DataTable`
  and emit a `DeprecationWarning` that the project's `filterwarnings
  = ["error"]` would promote to a failure):
  - `TestPercentInputsAllowAnyStep` (2) — every `percent` field
    renders with the literal string `step="any"` (catches a
    regression to `step=0.005` or any float).
  - `TestNumberInputsAllowAnyStep` (1) — every `number` field
    renders with `step="any"`, including those whose schema
    declares an explicit dollar-rounded `step` (currency hints
    are intentionally ignored at render time).
  - `TestIntInputsKeepIntegerStep` (2) — `int` fields preserve
    `step=1` (or the schema's explicit step), so the browser
    correctly rejects e.g. "65.5" for a spouse age. Plus a
    sanity check that the schema actually contains int fields.
  - `TestBoundsArePreserved` (1) — pins down that `min` / `max`
    still flow through (e.g. `spouse_a_age_start` still has
    18-100 bounds), so the fix didn't accidentally strip them.
  - `TestKnownInvalidValuesNowAccepted` (4) — a parametrized
    sanity check that `0.07` / `460270.90` / `1549863.75` /
    `416741.12` all *would* have failed the old step rules
    (proving the fix is non-vacuous and pinning down the exact
    values the user reported).

Full suite: 615 passing (605 pre-existing + 10 new).

### Fixed — Dash app: "Download JSON" persistence

**Why it matters:** the Dash app's Save → JSON button silently rewrote
the user's scenario in two ways that broke "round-trip" expectations:

1. **Spending profile downgraded `kind="smile"` → `kind="custom"`.**
   The save callback called
   ``scenario_to_dict(*apply_form_values(values))``, which funnels
   every spending profile through ``_spending_to_dict`` — and that
   helper unconditionally emits ``kind="custom"`` with an explicit
   ``phases`` array, ``lump_events``, and ``ltc_shock`` block, even
   for profiles built via :meth:`SpendingProfile.retirement_smile`.
   A user who loaded ``scenarios/example02.local.json`` (smile
   profile, ``ltc_years=4``, ``ltc_annual_today=$150k``) and clicked
   Save received a JSON file with the verbose custom shape — fields
   like ``ltc_years`` and ``ltc_annual_today`` vanished, replaced by
   a phases array and a separate ``ltc_shock`` block. Re-uploading
   the file still simulated correctly (the form-side
   ``_normalize_spending_block`` defensively demoted ``custom`` back
   to ``smile``), but the saved JSON was unrecognizable as the
   user's original intent and the diff against the on-disk file was
   massively noisy.
2. **Deprecated / legacy paths leaked into saved JSON.**
   ``_inputs_to_dict`` walked every dataclass field and surfaced
   ``inputs.annual_expenses`` (deprecated; replaced by
   ``config.annual_expenses_today`` /
   ``config.spending.base_spending``) and ``inputs.ss.start_age``
   (legacy single-spouse SS claim age, replaced by ``start_age_a``
   / ``start_age_b``). Both are in ``_HIDDEN_PATHS`` for the form
   schema, so the form never authors them — but the save was
   re-injecting them every time.

There was also a related ``tax_optimizer.scenario`` round-trip bug
that surfaced through the same path:

3. **``LognormalModel.cape_long_run`` silently dropped when
   ``cape_today`` was ``None``.** ``_market_to_dict`` only emitted
   ``cape_long_run`` *inside* the ``cape_today is not None`` branch.
   A user who set ``"cape_long_run": 22.0`` with ``"cape_today":
   null`` (the normal "no scaling, but I want this constant for
   future sensitivity sweeps" pattern) lost their value on every
   ``scenario_to_dict`` round-trip. This affected both the Dash
   form-load path (the form's ``cape_long_run`` field came back
   blank) and any direct ``scenario_to_dict`` consumer.

- **Fix 1 (dash_app/app.py:_save_scenario)** — switch the save
  callback to emit ``form_values_to_scenario(values)`` instead of
  ``scenario_to_dict(*apply_form_values(values))``. The form-shaped
  dict already excludes ``_HIDDEN_PATHS`` and respects the active
  ``spending.kind`` discriminator (``flat`` / ``smile`` only — the
  form doesn't author ``custom``), so the saved JSON now matches
  what the user actually entered. We still call
  ``apply_form_values(values)`` first as a *validation* step so any
  malformed form input fails loudly with a ``ScenarioError`` here
  rather than producing a half-baked file the user only notices on
  re-load.
- **Fix 2 (tax_optimizer/scenario.py:_market_to_dict)** — emit
  ``cape_long_run`` unconditionally for ``LognormalModel``. Only
  ``cape_today`` stays gated on being non-``None`` (its absence is
  the documented "no scaling" sentinel; emitting a literal ``null``
  would round-trip the same).
- **Drift reduction** — for
  ``scenarios/example02.local.json``, the diff between the on-disk
  scenario and the saved-from-form scenario went from **11 drift
  entries** (smile-vs-custom split, plus injected
  ``annual_expenses`` / ``start_age`` / ``phases`` / ``lump_events``
  / ``ltc_shock`` / ``cape_long_run`` loss) to **2 drift entries**
  (``roth_conversion_amount=0.0`` and
  ``section125_reduces_fica_wages=True``, both Config defaults the
  form correctly emits because that's what its values are after a
  load). The remaining two are semantically equivalent — they
  round-trip cleanly and yield byte-identical JSON on the next
  Save → Reload → Save cycle.
- **Save → Reload → Save fixed point** — the new
  ``test_two_saves_byte_identical`` test pins this down: loading a
  scenario, saving it, re-uploading the saved JSON, and saving
  again produces a byte-identical second save. That's the property
  the user actually relies on when they iterate on a plan.

### Tests — Dash save round-trip regression suite

- ``tests/test_dash_save_roundtrip.py`` — 13 new tests, gated on
  ``pytest.importorskip("dash")``:
  - **``TestSmileProfilePreserved``** (3) —
    ``kind="smile"`` survives the round-trip with
    ``base_spending`` / ``inflation`` / ``ltc_years`` /
    ``ltc_annual_today`` intact; ``phases`` / ``lump_events`` /
    ``ltc_shock`` do *not* leak into the saved file (otherwise the
    decoder's ``_check_keys('spending(smile)', ...)`` would reject
    on re-upload); the saved JSON re-loads cleanly and produces a
    simulator output equal to the original on first-year and
    last-year balance vectors.
  - **``TestFlatProfilePreserved``** (1) — same shape guard for
    ``kind="flat"``.
  - **``TestHiddenPathsNotEmitted``** (2) —
    ``inputs.annual_expenses`` and ``inputs.ss.start_age`` do not
    appear in the saved JSON; ``inputs.ss.start_age_a`` /
    ``start_age_b`` do.
  - **``TestMarketRoundTrip``** (2) — ``kind="lognormal"`` carries
    only lognormal-shaped fields; ``kind="deterministic"`` carries
    only ``equity`` / ``bond`` (no lognormal-only keys leak in).
  - **``TestCapeLongRunPreserved``** (3) —
    ``cape_long_run`` is emitted by ``scenario_to_dict`` even when
    ``cape_today`` is unset; round-trips through
    ``apply_scenario`` to the same value; the original both-set
    branch still works.
  - **``TestSaveReloadFixedPoint``** (2) — Save → Reload → Save
    reaches a byte-identical fixed point;
    ``apply_form_values(values)`` validates the save payload
    (decoder accepts it without error and yields a simulator
    output equal to the original).

Full suite: 605 passing (592 pre-existing + 13 new).

### Added — Dash app: download action-plan report (v6.8)

**Why it matters:** the Dash app already produced the same simulation
artifacts the CLI's `--report report.html` flag exports (winner pick,
KPI tiles, year-by-year table, MC fan / histogram), but to actually
walk away with a *document* the user had to drop back to the
command line. This adds a one-click "Download HTML" button next to
the existing "Download JSON" so the optimizer's recommendations —
TL;DR / household snapshot / recommended levers / expected outcomes /
top tornado sensitivities / year-1 actions / year-by-year withdrawal
& conversion table / hygiene / caveats — can be saved straight from
the browser. The output is the same Letter-paged, self-styled HTML
template `tax_optimizer.render.write_html` already produces, so the
file opens cleanly in any browser and prints to PDF via the browser's
"Save as PDF" dialog (no WeasyPrint / pango dependency).

- **New module — `dash_app/report_builder.py`** with three public
  helpers:
  - `cache_run(cfg, inputs, RunResult) -> run_id` — registers a
    finished run in a module-level `OrderedDict` keyed by a fresh
    UUID, bounded to the last 5 runs (LRU eviction). The Dash
    `run-result` `dcc.Store` carries only the run-id alongside its
    JSON-safe figure data; the actual `Config` / `Inputs` /
    per-strategy DataFrames live in the cache so the report
    renderer can walk them without re-running the simulator.
  - `get_cached(run_id)` — fetches the triple and bumps its LRU
    position. Returns `None` for unknown / evicted ids so the
    download callback can show a friendly "click Run again"
    message instead of throwing.
  - `build_html_payload(cfg, inputs, RunResult, *, scenario_path,
    year_table_scope)` — runs `tornado_sensitivity` against the base
    `(cfg, inputs)` and feeds the strategy dict + sensitivity sweep
    into `tax_optimizer.report.build_action_report`, then wraps the
    markdown via `tax_optimizer.render.render_html` into a
    self-contained HTML document. Returns the `{"content", "filename"}`
    shape that `dcc.Download.data` expects, with a timestamped filename
    (`tax-optimizer-report-YYYYMMDD-HHMMSS.html`).
- **Wiring** — `dash_app/app.py:_run` now calls `cache_run(...)` after
  every successful `run_scenario` and tucks the run-id into the
  serialized `run-result` payload. A new
  `_download_report` callback reads `run-result.data["run_id"]`,
  re-hydrates the cached `(cfg, inputs, rr)` triple, and feeds it
  to `build_html_payload`. The "Run a scenario first" / "cache
  expired" / generic-render-error edges all surface via the
  existing `run-status` Span. The button is `outline=True` blue so
  it visually distinguishes from the secondary "Download JSON".
- **Layout** — `dash_app/layout.py:top_bar` reflows from
  `[upload(md=8), save(md=4)]` to `[upload(md=6), save(md=3),
  report(md=3)]`. The new column has its own `dcc.Download` (id
  `report-download`) and a `<title>` tooltip that explains the
  PDF-via-browser-print workflow.
- **Mode coverage** — works for all three run modes (`single`,
  `four_strategies`, `four_plus_mc`). Single-mode reports degrade
  gracefully (the "Verdict" block is empty when there's no
  S0_baseline / S3_optimized contrast); four-strategy reports
  render the full optimizer narrative; `four_plus_mc` reports
  additionally pick up the Monte Carlo block via the cached
  `RunResult.mc`. No new Config knobs.
- **No new system deps** — the Dash app only emits HTML, never PDF,
  so the existing `[dash]` extra (`dash + dash-bootstrap-components +
  plotly`) covers it. PDF output is still available via the CLI
  (`python -m tax_optimizer --report report.pdf` with the `[pdf]`
  extra), and from the downloaded HTML via the browser's print
  dialog.

### Tests — Dash report-download regression suite

- `tests/test_dash_report_download.py` — 12 new tests, gated on
  `pytest.importorskip("dash")` so a minimal install without the
  `[dash]` extra still passes the rest of the suite:
  - **`TestRunCache`** (4 tests) — `cache_run` / `get_cached`
    round-trip, `None` / unknown-id handling, LRU eviction at
    capacity (`_MAX_CACHED_RUNS = 5`), and the bump-on-touch
    behavior (touching the oldest id promotes it so the *next*
    eviction takes the second-oldest instead).
  - **`TestBuildHtmlPayload`** (6 tests) — payload shape (`content`
    + timestamped filename), full-HTML-document scaffolding
    (`<!DOCTYPE html>` / `<title>` / report H1), presence of every
    canonical report section (1. Household snapshot through 9.
    Caveats), `scenario_path` propagation into the header,
    `year_table_scope="retirement"` shrinkage vs `"full"`, and the
    `ValueError` raise on an invalid scope (the latter is what
    surfaces in the UI as `Report build failed: …`).
  - **`TestEndToEnd`** (2 tests) — full `run → cache_run → get_cached
    → build_html_payload` flow for both `single` and
    `four_strategies` modes (the latter at `horizon_age=58` to keep
    the differential-evolution loop fast enough for CI).

Full suite: 592 passing (580 pre-existing + 12 new).

### Fixed — v6.8 estate-mode phantom spending (post-mortality drain)

**Why it matters:** When both spouses' `mortality.year_of_death_*` had
elapsed but the simulation horizon hadn't, the simulator silently
liquidated the taxable account year after year to fund a `spending_need`
that nobody alive could consume. On the user's `example02.local.json`
(`year_of_death_a = year_of_death_b = 30`, `horizon_age = 90`, smile
profile with `base_spending = 100k`), the post-death window produced
`spending_need ≈ $273–290k/yr` driving `taxable_withdrawal` of the same
order; the taxable balance fell from \$1.84M → \$1.26M over the last
two years even though every income column (`wages`, `pension`, `ssn`,
`rmd`) correctly reported \$0. The understated terminal taxable balance
flowed straight into `terminal_after_tax_nw` and any heir-bequest
metric, biasing the inheritance number for any scenario whose horizon
extends past `max(year_of_death_a, year_of_death_b)`.

The bug surfaced from a notebook eyeball: "the last 3 years the pension
and ssn columns are 0, why?" — and on closer look, so were `wages` and
`rmd`, but `taxable_withdrawal` was very much not zero.

- **Root cause** — mortality was modeled as a strict income-side gate
  (each of `wages`, `pension`, `ssn_income`, `rmd` checks `alive_a` /
  `alive_b`) but the spending profile and withdrawal cascade had no
  parallel guard. `spending.amount_for(...)` is age-driven by design
  (smile profile, manual phases, lump events), so it kept emitting a
  non-zero `net_need` in estate years. The retirement branch in
  `simulator.py` (`else: withdraws = withdraw_for_need(...)`) only
  toggles on `a_working or b_working` being False — which is True both
  in normal retirement *and* in estate mode — so the cascade fell
  through and drained taxable to net out the phantom need.
- **Fix** — one block in `tax_optimizer/simulator.py`, immediately
  after `net_need` and `lump_total` are built from
  `spending.amount_for(...)`:

  ```python
  if not (alive_a or alive_b):
      net_need = 0.0
      lump_total = 0.0
  ```

  With `net_need = 0`, the `conventional` / `proportional` /
  `bracket_fill` cascades in `withdrawals.py` short-circuit on their
  `if remaining > 0 and state.taxable > 0` guards and `withdraw_for_need`
  returns all-zero buckets. The deficit cascade can still pull a few
  thousand dollars from taxable to fund federal/state tax on residual
  portfolio yield (dividends + interest on the still-alive taxable
  balance), which is the correct behavior for an estate.
- **Downstream effect on inheritance metrics** — on the user's
  `example02.local.json`, the last-row `taxable_balance` rises from
  \$1.26M (pre-fix) to \$5.64M (post-fix); `pretax_balance` goes from
  \$6.33M → \$8.42M; `roth_balance` from \$11.53M → \$6.08M (the Roth
  drop is a different effect — pre-fix the cascade was protecting Roth
  by draining taxable first, so Roth grew unchecked; post-fix neither
  bucket gets touched, so the Roth's relative share is smaller because
  the taxable bucket isn't being artificially shrunk). The `single`
  filing-status label in estate-mode rows is left as-is — it's a
  fallback in `Mortality.filing_status` (no "neither" enum) and is
  cosmetic when both `alive_*` are False.
- **No new Config knobs** — the guard is purely defensive and fires
  automatically whenever `cfg.mortality.year_of_death_a` and
  `year_of_death_b` are both set and the simulation horizon stretches
  past `max(year_of_death_*)`. Default `Mortality()` (both survive to
  horizon) is unaffected — the both-dead set is empty.
- **Year-of-death rows unaffected** — the IRS year-of-death MFJ rule
  in `Mortality.filing_status` and the income-side `alive_*` flips are
  unchanged. Only the *post-death* rows (years strictly after both
  spouses' deaths) collapse to zero spending.

### Tests — v6.8 estate-phantom-spend regression suite

- `tests/test_estate_phantom_spend.py` — 6 new regression tests:
  - `test_post_death_window_exists` — sanity that the `Mortality(year_of_death_a=30, year_of_death_b=30)` + `horizon_age=90` scenario actually has post-death rows to inspect (≥5).
  - `test_spending_need_is_zero_post_death` — the core invariant: `df.loc[both_dead, "spending_need"].sum() == 0`.
  - `test_no_consumption_withdrawals_post_death` — `pretax_withdrawal == 0`, `roth_withdrawal == 0`, and `taxable_withdrawal < $5k/yr` (catches any regression that re-introduces the \$200k+ phantom drain; the \$5k bound leaves room for the deficit cascade to fund yield-related tax).
  - `test_income_lines_remain_zero_post_death` — guards the existing alive-gates (wages / pension / ssn / rmd) against accidental coupling to the spending fix.
  - `test_one_alive_spouse_keeps_spending` — single-survivor case (`year_of_death_a=20, year_of_death_b=None`) still has `spending_need > 0` in the survivor window. The guard must not fire when at least one spouse is alive.
  - `test_default_simulation_unchanged_shape` — default `Mortality()` produces an empty both-dead mask, so the guard is a no-op and the smile profile drives `spending_need > 0` in retirement as before.

Full suite: 580 passing (574 pre-existing + 6 new).

### Fixed — v6.7 Roth-401(k) cash-flow double-count

**Why it matters:** Every dollar a spouse routed into the Roth 401(k)
bucket was silently counted twice — once correctly (added to
`state.roth`) and once incorrectly (carried through as taxable cash
surplus into `state.taxable`). On the user's `example02.local.json`
($34.6k/yr of Roth-401(k) deferrals, 100% Roth split), this added
~$0.4M of phantom wealth over a 10-year working window — and silently
weakened the v6.5 Roth-conversion liquidity guard by the same amount
every year.

The bug surfaced via a wealth-conservation probe: flipping
`spouse_*_roth_401k_pct` from `1.0` to `0.0` on an otherwise-identical
deterministic scenario (zero growth/drag, no conversion, no mega-
backdoor) **increased** total household wealth by ~$30k, whereas the
correct behavior is for wealth to **decrease** by ≈ `deferral ×
marginal_rate` (the extra tax paid today for Roth treatment instead
of paying it later on pretax withdrawals).

- **Root cause** — `earned_cash` (used to derive `tax_paying_capacity`
  for the v6.5 Roth-conversion liquidity guard) and `cash_inflow`
  (used in the year's final `delta` that lands in `state.taxable`)
  both omitted `a_roth_contrib + b_roth_contrib` as a paycheck
  outflow. Pretax 401(k) is naturally subtracted via `wages_box1`
  reduction; mega-backdoor is explicitly subtracted via
  `mega_backdoor_total`; Roth 401(k) deferrals had no equivalent
  outflow term despite being added to `state.roth` at the
  contribution step.
- **Fix** — subtract `a_roth_contrib + b_roth_contrib` from both
  `earned_cash` and `cash_inflow` in `simulator.py`. Two-line change.
- **Downstream effect on Roth conversion** — `tax_paying_capacity`
  now correctly excludes Roth-401(k) deferrals. The v6.5 liquidity
  guard fires slightly more aggressively in tight-capacity scenarios
  (e.g. high-bracket-fill targets with low `conversion_taxable_use_ratio`).
  In the user's `example02.local.json` at 32% bracket-fill, capacity
  is still well above the conversion's marginal tax, so the visible
  conversion amount is unchanged — but the underlying capacity
  number in the `roth_conv_tax_capacity` diagnostic column drops by
  the elective-deferral total, matching textbook cash-flow.

### Tests — v6.7 wealth-conservation regression suite

- `tests/test_roth_401k_cashflow_v67.py` — 4 new regression tests:
  - `test_terminal_wealth_drops_by_tax_only_under_roth_election` —
    toggling Roth-401(k) pct between 0 and 1 must change terminal
    wealth by exactly `-extra_federal_tax_paid_today`. Pre-v6.7 the
    delta was opposite-signed and dominated by the missing deferral
    outflow.
  - `test_pretax_and_roth_bucket_movements_offset` — orthogonal
    "money goes to the right buckets" check; was already correct
    pre-v6.7.
  - `test_taxable_difference_equals_extra_tax_with_opposite_sign` —
    pins down where the bug lived (the `taxable_balance` Δ between
    the two cases).
  - `test_capacity_does_not_climb_under_roth_election` — `roth_conv_tax_capacity`
    diagnostic column must not gain the elective-deferral total when
    Roth-401(k) toggles on.

Full suite: 574 passing (570 pre-existing + 4 new).

### Fixed — `scenarios/example02.json` typo

- Renamed `roth_cap_conversion_by_liquidity` → `cap_conversion_by_liquidity`
  in `scenarios/example02.json`. The misspelled key was an invalid
  Config field, so the scenario parser correctly raised
  `ScenarioError` on load — but that meant the example file itself
  was unusable. Loud failure, but worth fixing so the example loads.

### Added — v6.6 mega-backdoor auto-spillover

**Why it matters:** Pre-v6.6, setting `spouse_*_total_contrib_pct *
salary > §402(g) cap` silently dropped the excess from the 401(k)
buckets and left it as taxable paycheck cash. Users frequently set
"target" percentages thinking the simulator would do the right thing
with the overflow; instead, every dollar above the cap got taxed at
ordinary rates and parked in the brokerage. v6.6 changes this when
the plan supports a mega-backdoor: the excess auto-routes into the
after-tax 401(k) bucket up to the §415(c) ceiling, then immediately
converts to Roth (existing same-day in-plan conversion path). Result:
no more silent leakage to taxable for households whose plans support
"Spillover After-Tax" (Vanguard) or equivalent (Fidelity / Schwab).

- **New behavior gated by existing knob** — when
  `spouse_*_mega_backdoor_enabled = True`, the simulator computes
  `excess = max(0, target - §402(g) cap)` per spouse and routes
  `min(excess, §415(c) room)` into the after-tax bucket. The
  explicit `spouse_*_after_tax_401k_pct` then stacks on top, capped
  at whatever §415(c) room remains.
- **Crowd-out is fully auditable** — if the auto-spillover consumes
  all the §415(c) room, the user's explicit pct gets clamped and
  the uncovered dollars stay as taxable cash (same fate as excess
  beyond the §415(c) ceiling — no silent re-routing).
- **No new Config knobs** — gating is purely on the existing
  per-spouse `mega_backdoor_enabled` flag. The §415(c) and §402(g)
  constants in `tax_optimizer/limits.py` are unchanged.

#### New diagnostic columns on the simulation DataFrame

- `excess_deferral_a` / `excess_deferral_b` — raw `target − §402(g) cap`.
  Always emitted (even when mega-backdoor is off), so users can see
  today's silent-cap behavior for the first time.
- `mega_backdoor_spillover_a` / `mega_backdoor_spillover_b` — the
  auto-routed portion only. Subset of `mega_backdoor_{a,b}`.
- `after_tax_target_uncovered_a` / `after_tax_target_uncovered_b` —
  explicit-pct target dollars that didn't fit into the remaining
  §415(c) room. Non-zero when the auto-spillover crowded out the
  user's explicit `after_tax_401k_pct`.
- (Existing `mega_backdoor_a` / `mega_backdoor_b` keeps its meaning:
  total after-tax routed to Roth = auto + explicit.)

#### Tests — `tests/test_mega_backdoor_spillover_v66.py` (15 new)

- Disabled gate: `mega_backdoor_enabled=False` with excess → excess
  surfaces in diagnostic column, no spillover (pre-v6.6 behavior).
- Basic spillover when excess < §415(c) room.
- Spillover clipped at §415(c) room when excess > room.
- Explicit pct stacks on top when room allows.
- Explicit crowded out when auto consumes all room; uncovered
  diagnostic shows the gap.
- Age-50+ catch-up correctly excluded from §415(c) room.
- Employer match consumes §415(c) room proportionally.
- Cash conservation: Roth grows by spillover, taxable falls by the
  same dollars; federal tax unchanged (Box 1 uses capped deferral
  in both paths).
- Per-spouse independence: only the enabled spouse spills.
- Parametrized invariant: total mega_backdoor ≤ §415(c) ceiling for
  a sweep of contribution percentages.

#### Behavior change to watch

- Scenarios with `mega_backdoor_enabled=True` AND
  `total_contrib_pct × salary > §402(g) cap` will see incremental
  Roth contributions where pre-v6.6 there was silent cash leakage.
  None of the bundled scenarios (template / example01 / example02)
  trigger this — their `total_contrib_pct × salary` values are all
  below the §402(g) cap. To recover legacy silent-drop behavior on
  a per-spouse basis, set `spouse_*_mega_backdoor_enabled = False`
  (which also disables the explicit `after_tax_401k_pct` knob).

---

### Added — v6.6 Section 125 cafeteria-plan health premiums

**Why it matters:** Medical / dental / vision insurance premiums paid
under a §125 cafeteria plan are pre-tax for federal income tax, FICA
(OASDI + Medicare), and state income tax. Pre-v6.6 the model had no
way to express the M/D/V employee share, so households with $10k–$20k
of annual premium were over-stating both Box 1 wages and FICA wages,
which propagated to over-stated taxes and Social Security earnings
records throughout the working-year horizon. v6.6 closes the gap.

- **New `inputs.health_premiums`** — `HealthPremiums` dataclass with
  six annual-dollar fields:
  `spouse_a_medical`, `spouse_a_dental`, `spouse_a_vision`,
  `spouse_b_medical`, `spouse_b_dental`, `spouse_b_vision`.
  All default to `0.0` (no behavior change for unset scenarios).
  Includes `total_a`, `total_b`, `total` convenience properties.
  Exported from the top-level `tax_optimizer` package.
- **New `Config.section125_reduces_fica_wages`** (default `True`) —
  controls whether §125 cafeteria deductions (M/D/V premiums **and**
  the HSA contribution) also reduce FICA wages and CA SDI wages.
  When `True` the simulator computes FICA on post-§125 wages
  (matching real Box 3 / Box 5 payroll); when `False` it falls back
  to the pre-v6.6 approximation of FICA-on-gross. The Box 1 / federal
  / state tax reductions are always on (those have never been
  approximated).
- **Per-spouse gating** — a spouse's premium is dropped when that
  spouse isn't working (no W-2 to deduct from). Combined per-spouse
  premium is clamped at that spouse's gross wages so Box 1 never
  goes negative.
- **HSA also benefits** — the v6.6 §125 path also fixes the
  long-standing FICA-on-gross-HSA approximation documented in
  `payroll.py`. Households with an HSA contribution now correctly
  save 7.65% × HSA on FICA in addition to the federal/state savings.

#### New diagnostic columns on the simulation DataFrame

- `health_premium_a` — annual §125 deduction for spouse A
- `health_premium_b` — annual §125 deduction for spouse B
- `health_premium_total` — sum of both (for quick visibility)

#### Tests — `tests/test_health_premiums_v66.py` (10 new)

- Federal tax reduction ≈ marginal × premium (within ±$200 across
  brackets).
- FICA reduction = 7.65% × premium when flag on; **no** FICA delta
  when flag off (back-compat path).
- CA state tax + SDI reductions when state regime is CA.
- Per-spouse gating: premium zeroes out the year a spouse retires;
  non-working spouse's premium is ignored even if set.
- Clamp at gross wages (you can't §125-deduct more than you earn).
- Cash-flow consistency: end-of-year taxable balance drops by
  ~(premium − tax savings), proving the cash actually leaves the
  household.

#### Behavior change to watch

- The CA-SDI baseline test in `tests/test_tax_v64.py` was updated to
  explicitly zero `inputs.contrib.hsa_family` (the default HSA family
  contribution would otherwise reduce CA SDI base by ~$94/yr under
  the new default-on §125 treatment). The fix is real-world correct
  (HSA dollars come out of post-§125 wages); existing scenarios with
  HSA contributions will see a small (single-digit-dollar) FICA + SDI
  reduction. Set `cfg.section125_reduces_fica_wages = False` to
  reproduce pre-v6.6 numbers exactly.

#### Template / scenarios

- `scenarios/template.json` gains the `inputs.health_premiums` block
  (all six fields zeroed) and the new `cfg.section125_reduces_fica_wages`
  knob. Drift tests in `tests/test_scenario_template.py` extended to
  cover `HealthPremiums`.

---

### Added — Scenario template + drift tests

- **New `scenarios/template.json`** — canonical reference listing every
  available `Config` / `Inputs` knob (and every field of the nested
  `StartingBalances` / `CurrentIncome` / `CurrentContrib` /
  `PensionInputs` / `SocialSecurity` / `Mortality` dataclasses) with
  its default value. Copy as the starting point for a new scenario.
  Includes the polymorphic `market` (lognormal) and `spending` (smile)
  shapes populated with defaults, plus the per-bucket `asset_location`
  form.
- **New `scenarios/README.md`** — documents the directory's purpose,
  the maintenance contract for new knobs, all polymorphic block
  shapes (`market.kind`, `spending.kind`), and helpful CLI patterns.
- **New `tests/test_scenario_template.py`** — 24 drift tests that
  fail loudly when a dataclass field is added without also being
  mirrored into `template.json`. Covers:
  - structural sanity (file exists, valid JSON, parses through
    `apply_scenario` without `ScenarioError`);
  - field-coverage drift (`Config`, `Inputs`, each nested input
    block, `Mortality`) with informative failure messages;
  - default-value drift (every value in the template equals the
    dataclass default);
  - end-to-end smoke (template loads via the path API and simulates
    cleanly).

  Maintenance contract is documented in the test docstring and the
  scenarios README — when you add a knob, also add it to
  `template.json` (and to `example01.json` / `example02.json` if the
  knob matters for those narrative scenarios).

### Fixed — v6.5 Roth-conversion liquidity guards

- **HIGH — Roth conversion now sized by tax-paying capacity, not just
  bracket headroom**
  ([`tax_optimizer/conversion.py`](tax_optimizer/conversion.py),
  [`tax_optimizer/simulator.py`](tax_optimizer/simulator.py),
  [`tax_optimizer/config.py`](tax_optimizer/config.py)). Pre-v6.5,
  `planned_roth_conversion` capped only at `pretax_balance - rmd`
  and never checked whether the household had cash to pay the
  conversion's marginal tax. An aggressive
  `roth_conversion_target_bracket` (or any fixed conversion >
  liquid cash) silently triggered the deficit cascade and pulled
  tax dollars out of the **just-funded Roth bucket** — defeating
  the strategy and, under IRS rules, potentially incurring the
  10% penalty on Roth-conversion principal when the holder is < 59½
  or the 5-year clock hasn't matured (neither tracked by the model).
  New knob `cfg.cap_conversion_by_liquidity` (default True) feeds
  the sizer a per-year `tax_paying_capacity` estimate (earned cash
  + pension + SS + RMD net of FICA/SDI/spending/healthcare/IRA &
  MBR contributions, plus `cfg.conversion_taxable_use_ratio` × the
  taxable-brokerage balance — default 50%). The sizer bisects the
  conversion total down so its federal + state marginal tax delta
  fits inside capacity. New row columns
  `roth_conv_capped_by_liquidity` /
  `roth_conv_bracket_target` / `roth_conv_tax_capacity` surface the
  bind so reports can flag it. `planned_roth_conversion` now returns
  a `ConversionPlan` NamedTuple instead of a `(conv_a, conv_b)`
  tuple to carry the diagnostic flags.
- **HIGH — Deficit cascade excludes Roth in conversion years**
  ([`tax_optimizer/withdrawals.py`](tax_optimizer/withdrawals.py),
  [`tax_optimizer/simulator.py`](tax_optimizer/simulator.py)).
  `cover_deficit` accepts a new `roth_available: bool = True`
  parameter; the simulator passes `False` whenever a conversion
  fires AND `cfg.protect_roth_in_conversion_years=True` (the new
  default). Any shortfall left after taxable / HSA-after-65 /
  pretax legs now surfaces as `unfunded` instead of silently
  raiding the Roth. Pre-v6.5 a conversion-tax shortfall in a
  gap year would withdraw from the *same* Roth bucket the
  conversion just funded — leaving the user with $0 of new Roth
  after paying tax with Roth dollars.
- **MEDIUM — Spending need + healthcare costs hoisted ahead of
  conversion sizer** ([`tax_optimizer/simulator.py`](tax_optimizer/simulator.py)).
  `years_until_horizon`, `net_need`, the HSA LTC pay-down,
  `health_pre65`, `medicare_base_premium`, and lagged-AGI IRMAA
  all compute *before* the conversion now, so the liquidity guard
  sees the household's full obligation stack. The `_state_tax_fn`
  closure used by the cascade solvers also hoists up so the
  bisection captures CA / NY / MA state-tax marginal bite. When
  `cfg.irmaa_lookback_years <= 0` the hoisted estimate uses
  pre-conversion AGI and the simulator refreshes IRMAA after
  final tax — minor numerical refinement for users on the
  zero-lookback setting.
- **Knobs added** (all on `Config`):
  - `cap_conversion_by_liquidity: bool = True`
  - `protect_roth_in_conversion_years: bool = True`
  - `conversion_taxable_use_ratio: float = 0.5`

### Tests — v6.5 (new file)

- `tests/test_conversion_liquidity_v65.py` — 11 new regression tests
  covering: liquidity-cap binds when aggressive bracket-fill meets
  small taxable balance; capped conversion doesn't raid Roth;
  `conversion_taxable_use_ratio` knob varies the cap monotonically;
  Roth protection surfaces unfunded vs raiding when enabled;
  both knobs off reproduces pre-v6.5 behavior;
  `planned_roth_conversion` direct unit tests for bisection
  monotonicity, CA-vs-NV state-tax tightening, and marginal-tax
  vs capacity invariant.

### Fixed — v6.4 Tax-module correctness sweep

- **HIGH — Age-65+ additional standard deduction is now modeled**
  ([`tax_optimizer/tax/regimes.py`](tax_optimizer/tax/regimes.py),
  [`tax_optimizer/simulator.py`](tax_optimizer/simulator.py)). The
  base TCJA-extended std deduction ($32,200 MFJ) is now augmented
  per §63(f): +$1,600/spouse 65+ for MFJ, +$2,000/filer 65+ for
  single. The 2025 OBBBA "senior bonus" of +$6,000/filer 65+ also
  layers on for tax years 2025–2028. Pre-fix, retirees lost up to
  $15,200/yr of deduction headroom, which over-stated federal tax
  and under-stated Roth-conversion bracket-fill room every
  retirement year. `TaxRegime.effective_std_deduction()` is the
  new public helper.
- **MEDIUM — NY retirement-income exclusion is now per-spouse, not
  pool-wide** ([`tax_optimizer/tax/state.py`](tax_optimizer/tax/state.py),
  [`tax_optimizer/simulator.py`](tax_optimizer/simulator.py)). NY's
  §612(c)(3-a) grants each filer 59½+ a $20k exclusion against
  their *own* pension / IRA / Roth-conv distributions. Pre-fix the
  model pooled distributions and applied the combined cap, which
  over-excluded by up to $20k in any year with lopsided spouse
  distributions (e.g. only spouse A drawing from pretax). The
  simulator now threads `pension_per_spouse` / `pretax_per_spouse`
  / `roth_conv_per_spouse` tuples into `state_tax`.
- **MEDIUM — Portfolio dividends split into qualified vs ordinary**
  ([`tax_optimizer/config.py`](tax_optimizer/config.py),
  [`tax_optimizer/simulator.py`](tax_optimizer/simulator.py)). New
  knob `cfg.taxable_equity_qualified_fraction` (default 0.85)
  routes the non-qualified slice through `ordinary_div` so it's
  taxed at ordinary rates per IRS §1(h). Pre-fix every portfolio
  dividend was treated as qualified, under-stating tax by ~10 bp
  of dividend yield on the ordinary-rate slice. Two new row
  columns (`qualified_dividends`, `ordinary_dividends`) for reports.
- **MEDIUM — AMT (Alternative Minimum Tax, §55) is now computed
  in parallel** ([`tax_optimizer/tax/regimes.py`](tax_optimizer/tax/regimes.py),
  [`tax_optimizer/tax/federal.py`](tax_optimizer/tax/federal.py)).
  Adds `amt_exemption_*`, `amt_phaseout_*`, `amt_28pct_threshold_*`,
  `amt_std_deduction_addback` to `TaxRegime`. AMT is effectively
  dormant under TCJA-extended (high $137k MFJ exemption, $1.25M
  phaseout) but can bite under SUNSET (pre-TCJA mechanics: lower
  exemption + std-deduction add-back) during large Roth-conversion
  years. `federal_tax` returns `amt` / `amti` / `tmt` for the
  reports. LTCG/QDIV preserve their preferential rates inside AMT.
- **MEDIUM — CA SDI (State Disability Insurance) modeled in payroll**
  ([`tax_optimizer/payroll.py`](tax_optimizer/payroll.py),
  [`tax_optimizer/tax/state.py`](tax_optimizer/tax/state.py),
  [`tax_optimizer/simulator.py`](tax_optimizer/simulator.py)). CA's
  1.1% rate is **uncapped** since SB 951 (2024). New `state_sdi()`
  helper + `StateTaxRegime.sdi_rate` / `sdi_wage_cap` fields. CA
  presets get `sdi_rate=0.011, sdi_wage_cap=math.inf`; other
  bundled states default to 0. Subtracted from cash inflow during
  working years (cash-flow cost only — not a federal/state
  taxable-income reducer).
- **LOW — `_marginal_rate` boundary semantics**
  ([`tax_optimizer/tax/federal.py`](tax_optimizer/tax/federal.py)).
  Changed inner comparison from `>` to `>=` so the function reports
  the higher bracket's rate at exact bracket boundaries (where the
  next dollar actually goes). In practice rarely matters (real
  taxable income doesn't land on cents), but was a latent
  bracket-fill-conversion edge case.
- **LOW — Stale IRMAA docstring updated**
  ([`tax_optimizer/tax/irmaa.py`](tax_optimizer/tax/irmaa.py)). The
  old doc claimed "we approximate with current-year AGI"; in
  practice the simulator threads a proper 2-year-lookback MAGI
  (default `cfg.irmaa_lookback_years = 2`).

### Tests — v6.4

- **`tests/test_tax_v64.py`**: 26 new regression tests covering all
  v6.4 fixes — senior std deduction (OBBBA window + post-window
  expiry), `_marginal_rate` boundary, dividend qualified/ordinary
  split, NY per-spouse exclusion (lopsided vs balanced
  distributions, back-compat for pool callers), AMT (TCJA-extended
  dormant, SUNSET firing on large conversions, regime-disabled
  short-circuit, std-deduction add-back), and CA SDI per-spouse +
  cash-flow impact.

### Fixed — v6.3 BP RAP fidelity follow-ups (retire_age + bonus eligibility)

- **HIGH — Annual incentive payments (bonus) now count as eligible
  earnings for pension pay credits**
  ([`tax_optimizer/simulator.py`](tax_optimizer/simulator.py)). Per
  BP RAP page 9, "Payments made under an annual incentive plan at
  the business unit or stream level" are pension-eligible.
  Pre-fix the simulator passed only `spouse_a_salary` to
  `pension_annual_credit`, **excluding `spouse_a_bonus`**. For
  participants with material bonuses (e.g. $100k/yr) this understated
  pension accrual by 30-50% over a working career. The IRS
  §401(a)(17) comp cap inside `pension_annual_credit` still applies
  to the combined base+bonus figure, so high-earner ceiling still
  binds.
- **MEDIUM — Projector reference now honors `retire_age`**
  ([`tax_optimizer/pension.py`](tax_optimizer/pension.py)).
  `project_pension_balance` accepted no retire-age argument and
  silently credited pay-credits all the way to NRD. For users
  whose `spouse_a_retire_age < pension.start_age`, the projector
  overstated the reference balance by ~4% per year of gap, scaling
  the user's `monthly_at_nrd` input DOWN by the same margin
  (because the simulator's annuity = input × actual / projected).
  The simulator now passes `retire_age=inputs.spouse_a_retire_age`,
  bringing actual / projected ≈ 100% and using the user's
  `monthly_at_nrd` input directly.

Both fixes are required for BP RAP scenarios to honor the user's
`monthly_at_nrd` (from BP NetBenefits) without silent scaling. New
regression tests in `tests/test_pension.py::TestProjectorRespectsRetireAge`
and `TestSimulatorIncludesBonusInPension`.

### Fixed — v6.3 BP RAP fidelity (pension module rewrite)

Cross-checked the pension module against the "How the plan works"
section (pages 8-13) of the BP Retirement Accumulation Plan (RAP)
Summary Plan Description (August 2023 edition). Three real
correctness gaps were found and fixed:

- **HIGH — Pay-credit tiers now respect age AND years-of-service**
  ([`tax_optimizer/pension.py`](tax_optimizer/pension.py)). Pre-v6.3
  `pension_annual_credit` hardcoded the **top** tier (6% / 11%) for
  every participant, overstating accrual for early-career employees
  by 33-57%. The function now implements the BP RAP page-9 table:
  the rate is selected by the higher of the age band (<40 / 40-49 /
  50+) and the service band (<10 / 10-19 / 20+). New helper
  `pay_credit_rates(age, years_of_service)` returns
  ``(low, high)`` per tier.
- **HIGH — IRS §401(a)(17) compensation limit applied** to eligible
  earnings ([`tax_optimizer/pension.py`](tax_optimizer/pension.py)).
  Pre-v6.3 a $500k earner accrued credits on the full salary; the
  SPD caps eligible earnings at the IRS limit ($350k in 2025,
  inflation-indexed). The simulator and the projector now index
  the cap forward with wage growth.
- **MEDIUM — Interest credit honors pre-2016 / post-2016 floor**
  ([`tax_optimizer/pension.py`](tax_optimizer/pension.py)). New
  `effective_interest_rate(rate, *, pre_2016_participant)` applies
  `max(rate, floor)` where floor = 5% (pre-2016) or 2% (post-2016)
  per BP RAP page 10. The simulator's annual rate is now this
  floored value rather than a bare 4.8% constant. The default
  ``pre_2016_participant=False`` preserves pre-v6.3 behavior for
  the existing test corpus; BP RAP participants who were eligible
  before January 1, 2016 should flip the knob to ``True``.

API additions:

- **`PensionInputs`** gained four fields (all default to preserve
  pre-v6.3 behavior): `years_of_service_today`,
  `pre_2016_participant`, `interest_rate`,
  `irs_comp_limit_today`. The JSON scenario loader handles them
  automatically via dataclass-field introspection.
- **`pension_annual_credit`** and **`project_pension_balance`**
  gained keyword arguments for age, YoS, comp limit, and interest
  rate (all optional / backward-compatible).

Regression coverage in [`tests/test_pension.py`](tests/test_pension.py)
(18 new tests across `TestPayCreditTiers`,
`TestBpRapWorkedExample1`, `TestIrsCompensationLimit`,
`TestInterestRateFloor`, `TestProjectorTierTransition`). The
worked-example test reproduces Example 1 on page 12 of the SPD
within $100 (annual-aggregate approximation vs SPD's monthly
compounding).

### Docs — v6.2 functional review (batch 4: modeling-scope clarifications)

- **`pension.py` module docstring** now explicitly calls out the
  single-spouse-A modeling limitation (F11) and the recommended
  combined-monthly-at-NRD workaround for dual-pension households.
- **`ira.allocate_ira_contributions` pro-rata comment** now
  documents that pre-existing Form-8606 (nondeductible Traditional)
  basis is NOT tracked (F14), and that the simplification is
  conservative (overstates backdoor-conversion tax).
- **`LognormalModel` docstring** explicitly notes the name is a
  misnomer for backward compat (F17) — the class actually samples
  arithmetic-return draws from a bivariate normal, not log-returns
  from a true lognormal. Pointers to `BootstrapModel` /
  `HistoricalSequenceModel` for bounded-return analyses.
- **README "What's not modeled (yet)"** expanded to include four
  newly-documented scope items: the Roth 5-year clock (F15), Form-
  8606 nondeductible IRA basis (F14), dual cash-balance pensions
  (F11), and arithmetic-vs-true-lognormal return draws (F17).

### Fixed — v6.2 functional review (batch 3: defensive / API)

- **LOW — `BootstrapModel` rejects invalid construction inputs at
  build time** ([`tax_optimizer/market.py`](tax_optimizer/market.py)).
  Now raises `ValueError` if `block_size <= 0`,
  `block_size > len(equity_history)`, or
  `len(equity_history) != len(bond_history)`. `begin_path(n_years=0)`
  is also rejected. Previously these surfaced as opaque numpy errors
  (or silently truncated histories) deep inside path generation.
- **LOW — `simulate_paths` rejects `n_paths < 1`**
  ([`tax_optimizer/monte_carlo.py`](tax_optimizer/monte_carlo.py)).
  Previously accepted zero/negative and returned a zero-row result
  that silently poisoned every downstream summary (NaN percentiles,
  CVaR, ruin rate). Now raises `ValueError`.
- **LOW — `Mortality.survivor_label` docstring matches code**
  ([`tax_optimizer/mortality.py`](tax_optimizer/mortality.py)). The
  old docstring claimed `None` for both "both alive" and "both
  dead"; the code actually returns the string `"neither"` for
  both-dead. The docstring now explicitly enumerates all four return
  values.
- **LOW — Pension `PENSION_QTR_SSWB` kink indexes forward with wage
  growth in `project_pension_balance`** ([`tax_optimizer/pension.py`](tax_optimizer/pension.py)).
  The SSWB tracks the National Average Wage Index in real life;
  freezing the 2025 value silently pushed more earnings into the 11%
  high band each decade. `pension_annual_credit` gained an optional
  `qtr_sswb` override; the projector grows the kink at the supplied
  `wage_growth` each year.

### Fixed — v6.2 functional review (batch 2: MEDIUM-severity correctness)

- **MED — HSA family cap downshifts to self-only when one spouse is on
  Medicare** ([`tax_optimizer/limits.py`](tax_optimizer/limits.py)).
  Previously the model kept the full HSA family limit ($8,550 + $1k
  catch-up) until both spouses hit 65. In staggered-Medicare
  households (one spouse 65+, other still working under HDHP) the
  IRS rules actually downshift to self-only ($4,300 + applicable
  catch-up) because the Medicare-enrolled spouse is no longer HDHP-
  eligible. Pre-v6.2 overstated capacity by ~$4.3k/year in these
  scenarios. Added `HSA_SELF_LIMIT` constant.
- **MED — `_solve_taxable_for_net` clamps `basis_frac` to [0, 1]**
  ([`tax_optimizer/withdrawals.py`](tax_optimizer/withdrawals.py)).
  A `basis_frac > 1.0` (taxable account at an unrealized loss with
  basis > FMV) used to flow through to `gain = mid * (1 - basis_frac)`
  as a negative number, producing a phantom AGI reduction in
  `federal_tax`. The simulator already clamps live ratios, but the
  public solver entry point is now defensive too.
- **MED — LTC shock anchors to end-of-life, not end-of-simulation**
  ([`tax_optimizer/spending.py`](tax_optimizer/spending.py),
  [`tax_optimizer/simulator.py`](tax_optimizer/simulator.py)).
  `SpendingProfile.amount_for` now takes an optional
  `years_until_death` argument. When provided (the simulator always
  passes it now, derived from `Mortality.year_of_death_a/b`), the
  shock fires in the last `ltc_shock.years` years *of life*. Falls
  back to `years_until_horizon` for legacy callers. Pre-v6.2 the
  shock fired in the last N years of the *simulation*, which was
  wrong whenever the horizon exceeded the mortality date.
- **MED — `metrics.summarize.min_balance` and
  `monte_carlo._ruin_year_offset` include HSA when at least one
  spouse is 65+** ([`tax_optimizer/metrics.py`](tax_optimizer/metrics.py),
  [`tax_optimizer/monte_carlo.py`](tax_optimizer/monte_carlo.py)).
  After 65 the HSA becomes a stealth IRA (any-purpose withdrawals
  at ordinary rate, no penalty); excluding it from the liquidity
  metric was misleading for HSA-heavy plans. Before 65 the HSA is
  restricted to qualified medical, so it's still excluded.

### Fixed — v6.2 functional review (batch 1: HIGH-severity math/state)

- **HIGH — `pension.pension_annual_credit` no longer silently keeps the
  high-rate band dormant for typical salaries**
  ([`tax_optimizer/pension.py`](tax_optimizer/pension.py)). The
  pre-v6.2 function divided annual earnings by 12 before comparing to
  `PENSION_QTR_SSWB` (which is one quarter of the *annual* SS wage
  base), then multiplied the per-month credit back up by 12. The two
  cancelled algebraically only for the all-low-band leg — the high-
  rate (11%) band did not fire until annual salary exceeded ~$553k.
  Effect: 30-40% underprediction of accrued pension for salaries
  between $50k and $300k. Now both legs evaluate against annual
  earnings against an annual threshold.
  ([`tests/test_pension.py`](tests/test_pension.py),
  [`tests/test_review_fixes.py::TestPensionCreditHighBandFires`](tests/test_review_fixes.py)
  for regression coverage.)
- **HIGH — Deficit cascade now grosses up for **state income tax** in
  addition to federal** ([`tax_optimizer/withdrawals.py`](tax_optimizer/withdrawals.py),
  [`tax_optimizer/simulator.py`](tax_optimizer/simulator.py)). The
  `_solve_pretax_for_net` / `_solve_taxable_for_net` helpers used to
  only account for federal-tax delta when computing the gross
  withdrawal needed to net a target. Households in CA/MA/OR/NY came
  up 6-12% short on every cascade leg, surfacing as silent unfunded
  gaps or post-hoc rebalancing artifacts. The solvers now accept an
  optional `state_tax_fn` callable; the simulator wires it to the
  household's resolved state regime so the gross-up is correct in all
  regimes (including STATELESS, where the closure returns 0 and
  behavior is unchanged).
  ([`tests/test_review_fixes.py::TestCascadeIncorporatesStateTax`](tests/test_review_fixes.py).)
- **HIGH — Pension annuity initializes when simulation starts at-or-
  past NRD** ([`tax_optimizer/simulator.py`](tax_optimizer/simulator.py)).
  Pre-v6.2 the initialization guard was `a_age == inputs.pension.start_age`,
  which silently dropped the pension for already-retired-with-pension
  households whose simulation horizon begins after the pension's
  Normal Retirement Date. The check is now `a_age >=
  inputs.pension.start_age`. When the user starts at-or-after NRD with
  no accumulated cash-balance, we honor `monthly_at_nrd` as the
  baseline (instead of scaling by the zero accumulator).
  ([`tests/test_review_fixes.py::TestPensionAnnuityInitWhenAlreadyAtNRD`](tests/test_review_fixes.py).)

### Fixed

- **HIGH — `inputs.income.*` (interest, capital gains, dividends) no
  longer drops to zero when only spouse A retires**
  ([`tax_optimizer/simulator.py`](tax_optimizer/simulator.py)). The
  pre-fix gate keyed the non-portfolio income extras on `a_working`
  alone, so a household where A retires at 62 while B keeps earning
  silently lost those streams for the staggered window. The gate now
  follows the in-code comment: `anyone_working = a_working or b_working`.
  Discovered by the package-wide error review (review finding #2).
- **HIGH — `--seed N` on the CLI now actually reseeds Monte Carlo
  objectives** ([`tax_optimizer/__main__.py`](tax_optimizer/__main__.py)).
  `_run_objective_optimizer` was only threading `args.seed` into the
  differential-evolution sampler (`seed=`). The MC draws scored by
  `--mc-objective {cvar,p_success}` kept their built-in default of
  42 regardless of the CLI flag, so two distinct seeds produced
  identical MC objective curves. Fix threads `args.seed` into BOTH
  `seed=` and `mc_seed=`. Discovered by the review (finding #3).
- **HIGH — `Config(rmd_start_age < 72)` is now rejected; `rmd_amount`
  defends against under-72 ages** ([`tax_optimizer/config.py`](tax_optimizer/config.py),
  [`tax_optimizer/rmd.py`](tax_optimizer/rmd.py)). Previously, any
  `rmd_start_age` below the youngest IRS Uniform Lifetime Table entry
  (72) caused `rmd_amount` to fall through to `divisor = 1.0` and
  return the **entire** pretax balance as the year's RMD. `Config.__post_init__`
  now raises `ValueError` at construction, and `rmd_amount` returns
  `0.0` (not `balance / 1.0`) for any age below the table's youngest
  key as a belt-and-suspenders. Discovered by the review (finding #4).
- **HIGH — Post-deficit-cascade IRMAA now respects
  `cfg.irmaa_lookback_years`** ([`tax_optimizer/simulator.py`](tax_optimizer/simulator.py)).
  The initial-pass IRMAA correctly honored the 2-year MAGI lookback
  (TC-5), but the post-cascade recompute always reset IRMAA to the
  current-year cascade-inflated AGI. This silently inflated IRMAA in
  any year the cascade fired under a positive lookback (which is the
  SSA-realistic setting). The recompute now only re-fires IRMAA when
  `irmaa_lookback_years <= 0`; under any positive lookback the
  cascade's effect on future-year IRMAA still flows through the
  `state.prior_agi` / `state.agi_lag_2` chain. Discovered by the
  review (finding #6).
- **MEDIUM — Report's market-model row no longer prints `NoneType` for
  the default `Config()`** ([`tax_optimizer/report.py`](tax_optimizer/report.py)).
  `_market_summary`, the cross-model robustness table, and the §4
  Monte Carlo blurb now route through `cfg.resolved_market()`, so a
  `Config()` with `market=None` correctly renders the deterministic
  fallback the simulator actually uses. Discovered by the review
  (findings #5, #18).

### Changed

- **`scenarios/example.json` renamed to `scenarios/example01.json`**
  to match the canonical name already used in `README.md`,
  `docs/scenario_guide.md`, and `CHANGELOG.md` references. The smoke
  test that loads each shipped scenario (`tests/test_simulator.py`)
  was previously `pytest.skip`-ing on the missing `example01.json`
  path; the test now `assert`s existence so a future rename can never
  silently regress. Notebook (`retirement_planning_demo.ipynb`)
  source cells are updated too. Discovered by the review (finding #1).
- **Minor cleanup of dead locals and unused imports**
  (`tax_optimizer/report.py`, `tax_optimizer/sensitivity.py`,
  `tax_optimizer/simulator.py`, `tax_optimizer/tax/state.py`). Removed
  the unused `pre_rmd_rows` and `base_sum` locals flagged by ruff
  F841, plus four F401 imports never referenced by their modules.
- **§7 "Year-by-year withdrawal & conversion plan" now defaults to the
  full horizon** ([`tax_optimizer/report.py`](tax_optimizer/report.py)).
  Pre-v7, the §7 table silently filtered the simulator DataFrame down
  to retirement years only — pre-retirement AGI / federal-tax / state-
  tax / any-pre-retirement-conversion detail was hidden, which left
  CLI users without context for plans that front-load conversions or
  use mega-backdoor contributions during working years. The table now
  spans every simulated year by default, with a visual
  `**RETIRE @ N**` marker row dividing accumulation from drawdown.
  The legacy compact view is still available via the new
  `year_table_scope="retirement"` kwarg on `build_action_report` or
  the matching `--year-table-scope retirement` CLI flag. The §7
  caption advertises the opt-out in both directions.

### Added

- **CLI `--year-table-scope {full,retirement}` flag**
  ([`tax_optimizer/__main__.py`](tax_optimizer/__main__.py)). Mirrors
  the `year_table_scope` kwarg on `build_action_report`. Defaults to
  `full`. Pass `retirement` to reproduce the pre-v7 compact view.

### Deprecated

- **`Inputs.annual_expenses` is now a no-op with a `DeprecationWarning`**
  ([`tax_optimizer/inputs.py`](tax_optimizer/inputs.py)). The simulator
  has not read this field in any v2+ release — `Config.resolved_spending()`
  picks `Config.spending.base_spending` first and falls back to
  `Config.annual_expenses_today`, never touching `Inputs.annual_expenses`.
  Setting it to a non-default value now emits a `DeprecationWarning`
  in `__post_init__` pointing the user at the two correct knobs. The
  field is retained on the dataclass so legacy scenario JSON / `--set`
  overrides keep round-tripping; only the simulator's behavior is
  unchanged. Removed from `scenarios/example01.json` and
  `scenarios/example02.json` (the values were ignored anyway).

### Added

- **Scenario-loader consistency warning for spending knobs**
  ([`tax_optimizer/scenario.py`](tax_optimizer/scenario.py)
  `_warn_spending_inconsistency`). When a scenario JSON sets both
  `config.annual_expenses_today` and `config.spending.base_spending`
  to **different** values, the loader now emits a `UserWarning`
  explaining that `base_spending` wins and `annual_expenses_today` is
  ignored in that case. Matching values stay silent (redundant but not
  misleading; the bundled example scenarios use the matched-value
  pattern as a reference template).
- **CLI `--cross-model [MODELS]` flag**
  ([`tax_optimizer/__main__.py`](tax_optimizer/__main__.py)). Surfaces
  the v6 `cross_model_check()` Python API on the command line. Pass
  the flag bare (e.g. `tax-optimizer --monte-carlo 1000 --cross-model`)
  to use the built-in defaults (`BootstrapModel` +
  `HistoricalSequenceModel`), or with a comma-separated list (e.g.
  `--cross-model bootstrap,vanguard_2025,jpm_ltcma_2025`) to specify a
  custom model menu. Recognized names: the built-in kinds (`lognormal`,
  `bootstrap`, `historical_sequence`, alias `historical`) plus every
  key in `CMA_PRESETS`. Companion flag `--cross-model-paths N`
  (default 200) tunes the per-alternative-model Monte Carlo path
  count. Requires `--monte-carlo > 0`; otherwise emits a clean error.
  The resolved `{name: MonteCarloResult}` dict is threaded into
  `build_action_report(..., extra_mc=...)` so the existing
  "Cross-model robustness check" sub-section auto-renders inside §4.
- **CMA-preset entries in `_MODEL_NOTES`**
  ([`tax_optimizer/report.py`](tax_optimizer/report.py)). The
  cross-model table's "Note" column now describes Vanguard / JPM /
  Horizon / historical CMA presets instead of falling back to `—`,
  matching the human-readable notes already shown for the built-in
  kinds.

### Tests

- 20 new tests in
  [`tests/test_cli_cross_model.py`](tests/test_cli_cross_model.py)
  covering: argparse acceptance of `--cross-model` (bare, with value,
  and the `--cross-model-paths` companion); `_parse_cross_model_arg`
  resolution of built-in kinds, the `historical` alias, case-insensitive
  input, whitespace tolerance, CMA presets, and unknown-name error
  messaging that lists known options; subprocess smoke tests for the
  end-to-end CLI happy path (default menu and custom CMA-mixed menu
  both render the section, both required-flag guards reject early
  with non-zero exit codes).
- 13 new tests in
  [`tests/test_spending_deprecation.py`](tests/test_spending_deprecation.py)
  covering: `Inputs.annual_expenses` deprecation
  (default-construction silent, non-default-construction warns,
  warning message names the replacement knobs, explicit-default-value
  silent, `apply_scenario` / `apply_set_overrides` propagate the
  warning); `_warn_spending_inconsistency` (matching values silent,
  mismatched values warn, message quotes both dollar amounts,
  scalar-only / spending-only / `spending: null` / spending-without-
  `base_spending` all silent).
- 12 new tests in
  [`tests/test_report_year_scope.py`](tests/test_report_year_scope.py)
  covering: default `"full"` scope spans the entire horizon, emits the
  `RETIRE @ N` marker row, advertises the opt-out in the §7 caption;
  `"retirement"` scope reproduces the legacy compact view (ages ≥
  retire only, no marker row, retirement-only caption text); explicit
  scope validation rejects unknown values; the new
  `--year-table-scope` CLI flag's argparse contract (default, valid
  values, rejection of unknown values).
- 12 new tests in
  [`tests/test_review_fixes.py`](tests/test_review_fixes.py) — regression
  coverage for each of the six HIGH review findings: `example01.json`
  rename guard, staggered-retirement income preservation, CLI
  `mc_seed=` thread, `rmd_start_age` validator + `rmd_amount` defense,
  `_market_summary` routing through `resolved_market()`, and
  post-cascade IRMAA respecting the lookback knob.
- [`tests/test_simulator.py`](tests/test_simulator.py) — replaced the
  silent `pytest.skip` with a hard `assert` on `scenarios/example01.json`
  so the canonical example scenario's existence is a tested invariant.
- [`tests/test_tier_c.py::TestConversionReservesRMD`](tests/test_tier_c.py)
  — bumped the test's `rmd_start_age` from 70 → 72 to match the new
  `Config.__post_init__` validator. Still verifies the same TC-7
  behavior (a fixed conversion can't truncate the year's RMD), just
  at the lowest age the IRS table publishes a divisor for.
  Test count: **380 → 437 passing**.

### Docs

- [`README.md`](README.md) gains two `--cross-model` CLI examples in
  the usage block, a cross-reference under the v6 helper bullet, and
  a callout note under the scenario JSON example pointing at the
  three spending knobs and the new deprecation. The §7 row in the
  "What the action report contains" table now documents the
  full-horizon default + `year_table_scope` opt-out, and the CLI
  usage block adds a `--year-table-scope retirement` example.
- [`docs/market_models.md`](docs/market_models.md) gains a new "From
  the CLI" subsection under §6 showing the flag's recognized model
  names, the `--cross-model-paths` companion, and the seed-sharing
  guarantee with `--monte-carlo`.
- [`docs/scenario_guide.md`](docs/scenario_guide.md) gains a new
  "Spending knobs — three names, one effective value" section that
  documents `Config.resolved_spending()`'s precedence, the
  loader-emitted warnings, and a migration table for legacy scenarios.
  The stale `inputs.annual_expenses` reference in the
  `median_ruin_year_offset` lever list is updated to point at
  `config.spending.base_spending`.

---

## [v6] — Tier R (action-report polish) (2026-05-10)

A 4-batch overhaul of the markdown action plan emitted by
`build_action_report`, turning it from a "what the optimizer chose"
summary into a decision document with quantified rationale, multi-model
robustness checks, and concrete next-12-month actions.

Total ≈ 1,050 LoC of source + ≈ 950 LoC of tests, all in one file
([`tax_optimizer/report.py`](tax_optimizer/report.py)). Test count:
**318 → 380 passing** (the 62 new tests live in
[`tests/test_report_ra.py`](tests/test_report_ra.py),
[`tests/test_report_rb.py`](tests/test_report_rb.py),
[`tests/test_report_rc.py`](tests/test_report_rc.py), and
[`tests/test_report_rd.py`](tests/test_report_rd.py)).

The pre-existing 9-section report layout was preserved; new sections
slot in alongside the old ones so external consumers see only
additions. No `build_action_report(...)` argument changed default
behaviour — every R-tier feature is opt-in or auto-detected from
existing state.

### Added *(Tier R-A — headline + transparency)*

- **R-A.1 — TL;DR header**
  ([`tax_optimizer/report.py`](tax_optimizer/report.py) `_tldr_section`).
  New top-of-report block: one-line verdict ("S3 optimizer beats S0
  by $X / +Y%"), a bullet list of levers the optimizer wants to change
  (driven by the new `_lever_changes` helper), and key risk readings
  (P(success), CVaR, peak marginal, lifetime tax / IRMAA Δ vs S0).
  When the winner matches the user's inputs on every axis, renders
  "No lever changes recommended" instead of forcing a diff table.
- **R-A.2 — Assumptions block**
  ([`tax_optimizer/report.py`](tax_optimizer/report.py)
  `_assumptions_block`). A new sub-table inside §1 disclosing the
  consequential assumptions: heir marginal rate, inflation, nominal
  growth, market model (with μ/σ/ρ for `LognormalModel`), federal
  regime + change year, state regime, mortality, step-up flag, ACA
  flag, healthcare costs, IRMAA lookback. Pinning these to one place
  lets advisors audit the recommendation without re-reading the JSON.
- **R-A.3 — Side-by-side strategy comparison**
  ([`tax_optimizer/report.py`](tax_optimizer/report.py) §3). §3 now
  renders all 4 canonical strategies (S0/S1/S2/S3) as columns rather
  than just "S3 vs S0", with the winning value bolded per row. Reveals
  cases like "S0 = S1 = S2 because the JSON already at 100% Roth"
  that the old format hid.

### Fixed *(Tier R-A — pre-existing report bugs)*

- **R-A.4 — Tornado direction labelling**
  ([`tax_optimizer/report.py`](tax_optimizer/report.py) §5). The §5
  tornado table previously emitted misleading `"higher (+$0)"`
  recommendations for knobs already at their tested-range boundary.
  Now detects `delta_high <= 0 and delta_low <= 0` and renders
  `"at boundary in tested range (—)"`. Knobs that *do* improve in one
  direction are labelled honestly with the positive delta.
- **R-A.5 — Near-zero `$0` cells**
  ([`tax_optimizer/report.py`](tax_optimizer/report.py) `_money`).
  Floats that round to zero (e.g. `1e-9` cascade artifacts) now render
  as `—` to match the legend's "no activity" convention. Previous
  `float(v) == 0.0` check missed near-zero values, producing confusing
  `$0` cells next to `—` cells for what are logically the same outcome.

### Added *(Tier R-B — actionable content + risk callouts)*

- **R-B.1 — "This year's concrete actions"**
  ([`tax_optimizer/report.py`](tax_optimizer/report.py)
  `_this_year_actions`). New §6 sub-section listing year-1 dollar
  amounts the user should set up immediately: 401(k) employee
  deferral (with Roth split), employer match (flagged as free money),
  mega-backdoor, backdoor IRA, HSA, Roth conversion (if any),
  upcoming RMD (if applicable), and the expected year-1 tax bill
  (federal + state + FICA).
- **R-B.2 — Widow's-penalty paragraph**
  ([`tax_optimizer/report.py`](tax_optimizer/report.py)
  `_widow_paragraph`). Auto-renders when one spouse predeceases the
  other inside the horizon. Reads the last-MFJ vs first-single rows
  from `w_df` and shows the AGI / marginal-rate / federal-tax delta
  at the filing-status switch. Silent for households with no mortality
  configured or with simultaneous death.
- **R-B.3 — Regime-change paragraph**
  ([`tax_optimizer/report.py`](tax_optimizer/report.py)
  `_sunset_paragraph`). Auto-renders when `cfg.regime_change_year_offset`
  is set. Shows the AGI / marginal / federal-tax delta at the boundary
  year so the reader sees the bracket jump in dollars and percentage
  points, not just regime names.
- **R-B.4 — Conditional state-tax column in §7**
  ([`tax_optimizer/report.py`](tax_optimizer/report.py) §7). The
  year-by-year withdrawal table adds a `State tax` column when
  `cfg.state_regime.name != "stateless"`. Avoids uniformly-zero
  columns for stateless scenarios.
- **R-B.5 — Conditional healthcare column in §7**
  ([`tax_optimizer/report.py`](tax_optimizer/report.py) §7). New
  `Health $` column collapsing `medicare_base_premium + health_pre65
  - aca_apt_credit` per row, with the legend extended to explain that
  IRMAA stays as its own column because it's a discrete AGI-cliff
  lever. Auto-omitted when no row has nonzero healthcare costs.

### Added *(Tier R-C — diagnostics + multi-scenario)*

- **R-C.1 — Optimizer-rationale paragraph**
  ([`tax_optimizer/report.py`](tax_optimizer/report.py)
  `_optimizer_rationale`). Five heuristic pattern detectors generate
  1-3 plain-English bullets explaining *why* the recommended plan
  looks the way it does:
  - heir-rate framing (heir vs avg-retirement marginal)
  - sunset front-loading (regime change + conversions chosen)
  - bucket imbalance (pretax / Roth ratio > 5×)
  - mega-backdoor activity
  - peak-marginal sticker shock (≥ 32%)
- **R-C.2 — Heir-rate sensitivity sweep**
  ([`tax_optimizer/report.py`](tax_optimizer/report.py)
  `_heir_rate_sensitivity`). Re-scores the winning plan at
  `heir_marginal_rate` ∈ {10%, 22%, 32%, 37%} (plus the current value
  if not already in the set) and renders a Δ-vs-current table. The
  plan never changes; only the bequest-haircut score does, so the
  reader sees the robustness of the verdict to that one assumption.
- **R-C.3 — `compare_scenarios()` public API**
  ([`tax_optimizer/report.py`](tax_optimizer/report.py)
  `compare_scenarios`). New top-level function rendering a side-by-side
  markdown diff of N independent household configurations. Returns
  `# Scenario comparison` markdown with three sub-tables: outcome
  metrics (bolded best per row), scenario assumptions, and optional
  Monte Carlo overlay when `mc=` dict is supplied. Useful for
  "what if I move to CA vs NY", "what if I retire at 62 vs 67", etc.
  Exported from package root as `tax_optimizer.compare_scenarios`.

### Added *(Tier R-D.1 — cross-model robustness)*

- **R-D.1 — Cross-model robustness check + `cross_model_check()` API**
  ([`tax_optimizer/report.py`](tax_optimizer/report.py)
  `cross_model_check`, `_cross_model_table`, and `build_action_report`
  `extra_mc=` kwarg). New helper `cross_model_check(cfg, inputs, *,
  n_paths=200, seed=42, models=None)` re-runs `simulate_paths` under
  alternative market models and returns `{name: MonteCarloResult}`.
  Defaults to `BootstrapModel` + `HistoricalSequenceModel` (the two
  non-parametric options that complement the typical parametric
  `LognormalModel`). `build_action_report(..., extra_mc=...)` then
  renders a new `### Cross-model robustness check` table inside §4
  showing each model's P(success), median terminal NW, and CVaR(10%),
  with auto-emitted callouts: model-risk flag if any model drops below
  90%, neutral note for ≥ 5pp spread, "all robust" confirmation when
  all models agree. Exported from package root as
  `tax_optimizer.cross_model_check`. ~3-5 seconds of additional compute
  at default settings.

### Tests

- 19 new tests in [`tests/test_report_ra.py`](tests/test_report_ra.py)
  covering TL;DR, assumptions block, 4-strategy comparison, tornado
  direction fix, near-zero rendering, and `_lever_changes` helper.
- 15 new tests in [`tests/test_report_rb.py`](tests/test_report_rb.py)
  covering this-year actions, widow paragraph (both-live / same-year /
  A-first / B-first), sunset paragraph, state-tax column conditional,
  healthcare column conditional.
- 17 new tests in [`tests/test_report_rc.py`](tests/test_report_rc.py)
  covering rationale pattern detectors, heir-rate sensitivity sweep
  (canonical rates + custom rate + monotonicity), and `compare_scenarios`
  (single/multi-scenario, bolding, duplicate-name validation, empty
  input, MC overlay, None-mortality handling).
- 11 new tests in [`tests/test_report_rd.py`](tests/test_report_rd.py)
  covering `cross_model_check` (default models, override, market actually
  applied, deterministic edge case, no mutation), `extra_mc` integration
  (no-section default, renders when provided, silent when `mc=None`,
  "all robust" callout, model-note column), and the public-export contract.

---

## [v5] — Tier C (correctness + healthcare + optimizer scope) (2026-05-10)

A 17-item batch organized into three sub-tiers:
- **C-A** (9 correctness bugs)
- **C-B** (5 high-ROI modeling additions, including ACA premium tax credit)
- **C-C** (3 optimizer-scope extensions)

Total ≈ 1,100 LoC of source + ≈ 700 LoC of tests. Test count: **288 → 318
passing** (the 30 new tests live in [`tests/test_tier_c.py`](tests/test_tier_c.py)).

### Fixed *(Tier C-A — correctness bugs)*
- **TC-1 — Year-of-death MFJ filing status**
  ([`tax_optimizer/mortality.py`](tax_optimizer/mortality.py)). The
  IRS treats the calendar year a spouse dies as MFJ on the joint
  return. Pre-Tier-C, `Mortality.filing_status()` flipped to single
  immediately at `year_of_death`. New helper
  `Mortality.is_year_of_death()` lets `filing_status()` keep MFJ for
  exactly the year of death.
- **TC-2 — Survivor SS at age 60 from deceased's record**
  ([`tax_optimizer/simulator.py`](tax_optimizer/simulator.py) ~370–
  395). SSA lets a widow(er) collect survivor benefits on the
  deceased's record starting at age 60, regardless of their own
  claim age. Pre-Tier-C, `ssn_income` was forced to $0 if the
  survivor hadn't reached their own claim age — silently dropping
  5–10 years of benefits in any scenario where the higher-earner
  died first. Now uses `max(own_benefit_if_eligible,
  survivor_of_deceased_if_60+)`.
- **TC-3 — Deficit cascade tax base**
  ([`tax_optimizer/simulator.py`](tax_optimizer/simulator.py)
  ~720–740). `cover_deficit` is now passed `final_kwargs` (with
  primary withdrawals + conversions stacked) instead of
  `base_kwargs`. The internal `_solve_pretax_for_net` /
  `_solve_taxable_for_net` solvers compute marginal tax via a
  base-tax delta; passing pre-stack kwargs **understated** the
  marginal bracket on every cascade leg in any year with a primary
  RMD or conversion.
- **TC-4 — Additional Medicare 0.9% MFJ joint threshold**
  ([`tax_optimizer/payroll.py`](tax_optimizer/payroll.py),
  [`tax_optimizer/simulator.py`](tax_optimizer/simulator.py)). New
  `fica_household(wages_a, wages_b, *, filing_status)` reconciles
  the Form-8959 Additional Medicare 0.9% surcharge at the
  household level using the correct filing-status threshold
  ($250k MFJ vs. $200k single). Pre-Tier-C, two $180k W-2s
  produced $0 of Additional Medicare (each below the per-W-2
  $200k withholding threshold) when the actual joint liability
  was 0.9% × ($360k - $250k) = $990. `fica_employee()` is
  preserved for employer-withholding modeling.
- **TC-5 — Roth IRA MAGI estimate uses prior-year AGI**
  ([`tax_optimizer/simulator.py`](tax_optimizer/simulator.py)
  ~245–255, [`tax_optimizer/state.py`](tax_optimizer/state.py)).
  New `state.prior_agi` field carries last year's AGI forward.
  The IRA MAGI phase-out estimate now uses
  `max(state.prior_agi, current_year_wages)` instead of wages
  alone, fixing the case where a high-portfolio retiree's
  $300k+ MAGI silently allowed direct Roth contributions.
- **TC-6 — Backdoor Roth pro-rata is IRA-aggregate, not IRA + 401(k)**
  ([`tax_optimizer/state.py`](tax_optimizer/state.py),
  [`tax_optimizer/ira.py`](tax_optimizer/ira.py),
  [`tax_optimizer/simulator.py`](tax_optimizer/simulator.py)). New
  `state.spouse_*_pretax_ira` IRA-only sub-balances correctly model
  IRC §408(d)(2): pro-rata aggregates only IRA balances, NOT 401(k).
  Pre-Tier-C, a spouse with $0 pretax IRA but $500k 401(k) had a
  99% taxable backdoor; now correctly $0 taxable. Drains
  (RMDs, conversions, withdrawals) reduce the IRA-only sub-balance
  pro-rata to its share of the total pretax bucket.
- **TC-7 — Fixed Roth conversion reserves RMD bucket**
  ([`tax_optimizer/conversion.py`](tax_optimizer/conversion.py)).
  `planned_roth_conversion()` now caps each spouse's conversion
  at `pretax_balance - rmd_amount` so a fixed
  `roth_conversion_amount` can no longer leave the pretax bucket
  too low for `withdraw_for_need` to satisfy the RMD. New `rmd_a`
  / `rmd_b` kwargs.
- **TC-8 — Action report shows per-spouse SS start_age**
  ([`tax_optimizer/report.py`](tax_optimizer/report.py) ~95–105).
  Household snapshot now uses `inputs.ss.effective_start_age_a` /
  `effective_start_age_b` instead of the legacy
  `inputs.ss.start_age` (which was the same value for both
  spouses).
- **TC-9 — Caveats block doesn't claim "federal-only"**
  ([`tax_optimizer/report.py`](tax_optimizer/report.py) ~415–435).
  Updated to reflect Tier B (state tax shipped) and Tier C-B
  (Medicare + ACA + step-up shipped).

### Added *(Tier C-B — high-ROI modeling)*
- **TC-10 — Base Medicare Part B + Part D premiums**
  ([`tax_optimizer/config.py`](tax_optimizer/config.py),
  [`tax_optimizer/simulator.py`](tax_optimizer/simulator.py)). New
  `cfg.medicare_base_b_d_premium` (default $2,500/yr today's
  dollars per enrolled spouse). Inflated by `cfg.inflation`. New
  DataFrame column `medicare_base_premium`.
- **TC-11 — IRMAA 2-year MAGI lookback**
  ([`tax_optimizer/state.py`](tax_optimizer/state.py),
  [`tax_optimizer/simulator.py`](tax_optimizer/simulator.py)). New
  `cfg.irmaa_lookback_years` (default `2`, the SSA-published
  rule). The simulator maintains `state.prior_agi` /
  `state.agi_lag_2` and reads the appropriate lag for IRMAA
  computation. Set to `0` to recover the pre-Tier-C (current-year
  AGI) behavior. New DataFrame column `irmaa_lookback_agi`.
- **TC-12 — Pre-Medicare healthcare cost knob**
  ([`tax_optimizer/config.py`](tax_optimizer/config.py),
  [`tax_optimizer/simulator.py`](tax_optimizer/simulator.py)). New
  `cfg.health_pre65_today` (default 0). Charged each year either
  spouse is alive AND below 65; per-spouse share is
  `health_pre65_today × (n_pre_medicare / 2)`. Bypassed when ACA
  is enabled (the benchmark premium replaces it). New DataFrame
  column `health_pre65`.
- **TC-13 — ACA premium tax credit**
  ([`tax_optimizer/config.py`](tax_optimizer/config.py),
  [`tax_optimizer/simulator.py`](tax_optimizer/simulator.py)). New
  `cfg.aca_enabled` (default `false`),
  `cfg.aca_benchmark_premium_per_adult` (default $14,000/yr),
  `cfg.aca_max_contrib_pct` (default 0.085 — post-IRA-2022 8.5%
  cap). Models the **post-IRA-2022 enhanced subsidies**: cap
  premium contribution at 8.5% of MAGI; credit
  `max(0, benchmark - cap)` against cash outflow. Cliff-free for
  all incomes ≥ 150% FPL. Approximations: MAGI ≈ AGI; no FPL ×
  household-size lookup (Tier D). New DataFrame columns
  `aca_benchmark_premium`, `aca_apt_credit`.
- **TC-14 — Step-up in basis on first spouse's death**
  ([`tax_optimizer/config.py`](tax_optimizer/config.py),
  [`tax_optimizer/simulator.py`](tax_optimizer/simulator.py)). New
  `cfg.stepup_at_first_death` (default `false`). When true,
  resets `state.cumulative_basis = state.taxable` at the start of
  the year of first death (community-property full step-up;
  CA/WA/ID/NM/AZ/LA/NV/TX/WI). Common-law half-step-up is in
  Tier D. New DataFrame column `cumulative_basis`.

### Added *(Tier C-C — optimizer scope)*
- **TC-15 — Mega-backdoor 401(k) % in optimizer decision vector**
  ([`tax_optimizer/optimizer.py`](tax_optimizer/optimizer.py)). New
  `_build_decision_vector_meta(cfg, inputs)` discovers active
  decision axes dynamically. When
  `inputs.spouse_*_mega_backdoor_enabled=True`, the decision vector
  grows with `mega_backdoor_pct_a` / `mega_backdoor_pct_b`
  (continuous, 0..1 each).
- **TC-16 — Per-spouse SS claim-age axis**
  ([`tax_optimizer/optimizer.py`](tax_optimizer/optimizer.py),
  [`tax_optimizer/config.py`](tax_optimizer/config.py)). New
  `cfg.optimize_ss_claim_age` flag (default `false`). When true,
  adds `ss_claim_age_a` / `ss_claim_age_b` axes (discrete grid
  `{62, 65, 67, 70}`). Major lever for asymmetric couples.
- **TC-17 — `mc_seed` thread-through in MC objectives**
  ([`tax_optimizer/optimizer.py`](tax_optimizer/optimizer.py)). New
  `mc_seed` parameter on `make_objective`, `optimize_household`
  (formerly hard-coded to `42` inside `_cvar_objective` /
  `_p_success_objective`). Pin a single value for reproducibility;
  sweep to detect overfitting to one MC draw.

### Changed
- `Config` gained 8 new fields (see Tier C-B/C entries above).
  Scenario JSON loader auto-recognizes them via the existing
  `fields(Config)` reflection — no `scenario.py` changes needed.
- `optimize_s3` is preserved as a backward-compat alias to the
  newly-renamed `optimize_household`.
- `simulator.py` row dict gained columns: `medicare_base_premium`,
  `health_pre65`, `aca_benchmark_premium`, `aca_apt_credit`,
  `irmaa_lookback_agi`, `cumulative_basis`,
  `fica_oasdi`, `fica_medicare`, `fica_additional_medicare`.
  Removed: `fica_a`, `fica_b` (replaced by household-level
  reconciliation; per-W-2 data still available via
  `fica_employee()`).

### Tests
- `tests/test_tier_c.py` (NEW) — 30 tests covering every Tier C-A
  bug fix, every Tier C-B feature, and every Tier C-C optimizer
  axis.
- `tests/test_mortality.py` updated for TC-1 (year-of-death MFJ).
- `tests/test_simulator.py` updated for TC-1 (year-of-death MFJ).

### Docs
- `docs/scenario_guide.md`: 7 new sections covering all Tier C
  config knobs (Medicare base premium, pre-65 healthcare, IRMAA
  lookback, ACA suite, step-up, optimize_ss_claim_age).
- `scenarios/example01.json` and `scenarios/example02.json`:
  added Tier C config block (set to backward-compat defaults so
  existing scenarios behave identically).

---

## [v4] — Market-modeling deep dive (2026-05-09, commit `3ac192f`)

### Added
- **Equity-bond correlation in `LognormalModel`** — new field
  `equity_bond_corr: float = 0.10`. `begin_path()` now draws from a
  bivariate normal via `rng.multivariate_normal`, so equity and bond
  returns are no longer treated as independent. Validates `[-1, 1]`
  in `__post_init__`. Marginals (μ, σ) are unchanged by correlation
  — verified empirically with N=100k draws.
- **CAPE conditioning** on `LognormalModel.equity_mu` — new fields
  `cape_today: float | None = None` and `cape_long_run: float = 16.5`,
  plus public `effective_equity_mu()` for diagnostics. When set,
  scales `equity_mu` by `cape_long_run / cape_today` (Shiller's
  starting-yield approximation). Vol is intentionally left untouched.
- **`HistoricalSequenceModel`** (4th `MarketModel`) — picks one
  contiguous N-year slice of 1928–2023 history per Monte Carlo path
  (Bengen / FIRECalc methodology). Distinct from `BootstrapModel`,
  which stitches short blocks. Yields 67 distinct possible 30-year
  paths from 96 years of data. Raises a clean `ValueError` when
  horizon exceeds available history.
- **`CMA_PRESETS` table** + **`lognormal_from_cma(name, **overrides)`**
  helper — five canned forward-looking assumption profiles:
  `vanguard_2025`, `jpm_ltcma_2025`, `horizon_2025`,
  `historical_1928_2023`, `historical_1985_2023`. Each preset bundles
  μ_eq, σ_eq, μ_b, σ_b, ρ(eq,b). Per-knob overrides via kwargs.
- Public API exports in `tax_optimizer/__init__.py`:
  `HistoricalSequenceModel`, `CMA_PRESETS`, `lognormal_from_cma`.
- Scenario JSON loader/serializer (`tax_optimizer/scenario.py`):
  - `kind: "historical_sequence"` for the new model.
  - `cma: "vanguard_2025"` shortcut (implies `kind: "lognormal"`,
    pulls preset, allows per-knob overrides).
  - Round-trips `equity_bond_corr`, `cape_today`, `cape_long_run`.
  - Helpful error for unknown CMA presets and `cma` paired with a
    non-lognormal `kind`.

### Tests
- 33 new tests in `tests/test_market_models.py` covering correlation
  marginals + propagation, CAPE math (including no-op cases),
  historical-sequence contiguity & bounds, preset table integrity,
  and full scenario JSON round-trips for every new field.

---

## [v3 — Tier B] — High-ROI modeling gaps (2026-05-09, commit `b64bbbb`)

### Added
- **State income tax module** (`tax_optimizer/tax/state.py`):
  - `StateTaxRegime` dataclass — encapsulates state-specific tax
    parameters (brackets MFJ/Single, std deductions, LTCG treatment,
    SS taxability, retirement-income exclusions, HSA deductibility).
  - `state_tax(...)` — computes state income tax for a given year,
    handling state-specific adjustments.
  - Bundled presets: `STATELESS`, `CA`, `NY`, `IL`, `MA`.
  - `lookup(name)` for retrieving regimes by name.
  - `Config.state_regime`, `state_regime_change_year_offset`,
    `state_regime_change_target` for state-to-state moves mid-horizon.
  - State brackets/deductions inflate annually via
    `bracket_indexing_rate`.
- **Bequest tax in terminal-NW objective** —
  `Config.heir_marginal_rate: float = 0.22` discounts terminal pretax
  AND HSA balances when projecting to heirs (HSA inherited by
  non-spouses is taxed like pretax). Threaded through
  `metrics.py`, `optimizer.py`, `sensitivity.py`, `monte_carlo.py`,
  and `__main__.py`. `terminal_after_tax_nw()` keeps a `marginal_rate=`
  alias for backward compatibility.
- **IRA contributions** (`tax_optimizer/ira.py`) — three priority-
  ordered paths per spouse, capped by the annual IRA limit:
  - Deductible Traditional IRA (reduces `wages_box1`).
  - Direct Roth IRA (auto-phases out by MAGI).
  - Backdoor Roth (non-deductible Traditional + immediate conversion,
    pro-rata aware over the existing pretax balance).
  - New `Inputs` fields: `spouse_a/b_traditional_ira_contrib`,
    `_roth_ira_contrib`, `_backdoor_roth`.
  - New `limits.py` constants: `IRA_CONTRIBUTION_LIMIT`,
    `IRA_CATCH_UP_50`, `IRA_CATCH_UP_AGE`, plus Roth IRA MAGI
    phase-out thresholds and helpers `ira_contribution_cap(age)`,
    `roth_ira_phaseout_factor(magi, filing_status)`.
- **Mega-backdoor Roth** (after-tax 401k → in-plan Roth conversion):
  - `Inputs.spouse_a/b_mega_backdoor_enabled` and `_after_tax_401k_pct`.
  - Capped by §415(c) overall annual additions limit
    (`SECTION_415C_LIMIT = 70_000.0` in `limits.py`), minus other
    401(k) contributions.
  - Routes directly to `state.roth`; reduces `cash_inflow`.
- **HSA after 65** — once either spouse hits 65, HSA becomes a 4th
  retirement bucket. The deficit cascade now draws taxable →
  Roth → HSA (post-65, ordinary income, no penalty) → pretax.
  `cover_deficit()` accepts `hsa_already` and `hsa_unlocked` kwargs.
- Example scenarios `scenarios/example01.json` and `example02.json`
  populated with all new Tier B knobs.
- Documentation: full sections added to `docs/scenario_guide.md`
  for every new config field.

### Tests
- 44 new tests in `tests/test_tier_b.py` covering state regimes
  (lookup, per-state rules, simulator integration, regime change),
  bequest tax with various heir rates, IRA cap arithmetic, IRA
  allocation (eligibility, cap priority, phase-out, pro-rata
  backdoor), simulator integration of all IRA paths, mega-backdoor
  enablement and §415(c) cap, HSA-after-65 deficit-cascade behavior,
  and a kitchen-sink integration test with all knobs enabled.
- Total test count: 211 → 255.

### Fixed
- Illinois `state_tax` was double-zeroing `ss_taxable_federal` in the
  retirement-income full-exclusion branch when SS was already exempt
  by `regime.ss_taxable_fraction == 0`. Now uses `state_ss_taxable`
  in the exclusion arithmetic.

---

## [v3 — Tier A] — Correctness fixes & contribution-cap accuracy (2026-05-09, commits `0d191e4` + `ef4e345` + `00db500`)

### Added
- **`limits.py` module** (commit `00db500`) — IRS 401(k) elective
  deferral limit, age-50 catch-up, age-60 super-catch-up (SECURE 2.0),
  and HSA family-coverage limit + age-55 catch-up. Helper
  `elective_deferral_cap(age)` and `hsa_contribution_cap(age)`.
- **Tax-efficient deficit cascade** (commit `00db500`) — new
  `cover_deficit()` in `withdrawals.py` draws from accounts in
  order: taxable → Roth → pretax (Roth conversion math is
  pro-rata-aware). Replaces the prior "drain pretax first" behavior
  which over-stated lifetime tax in many scenarios.
- **Employer 401(k) match modeling** (commit `ef4e345`) — new
  `Inputs` fields per spouse: `employer_match_rate` and
  `employer_match_max_pct` (i.e. "100% match up to 6% of salary").
  Match is added to the pretax balance during working years; action
  report displays the per-spouse breakdown.
- **Social Security annual COLA** (commit `0d191e4`) — `Config.ss_cola_rate`
  inflates each spouse's PIA annually after their claim year.
- **Per-spouse SS claim age & FRA** (commit `0d191e4`) — `Inputs.ss`
  upgraded from a single `claim_age` / `fra` to per-spouse fields.
  Actuarial scaling for early/delayed claims applies to each spouse
  independently.
- **Bracket indexing rate** (commit `0d191e4`) — `Config.bracket_indexing_rate`
  inflates federal ordinary-income brackets, LTCG brackets, std
  deduction, AND IRMAA tier thresholds annually. Closes the bug
  where retirees in 2055 were being taxed off 2026 nominal brackets.
- **Explicit yields on taxable-account holdings** (commit `0d191e4`)
  — `Config.taxable_equity_div_yield` and
  `Config.taxable_bond_interest_yield` make the tax drag on a taxable
  brokerage realistic instead of relying on price-only returns.
- **FICA / payroll tax modeling** (commit `0d191e4`) — employee-side
  OASDI (capped at SS wage base, indexed annually), Medicare, and
  Additional Medicare on W-2 wages. Threaded through `simulate()`
  and the action report.
- **Spousal pretax rollover on death** (commit `0d191e4`) — when a
  spouse dies, their `state.spouse_*_pretax` balance rolls over to
  the surviving spouse's pretax bucket, preserving the asset and
  rebasing future RMDs to the survivor's age.
- **Roth conversion eligibility expansion** (commit `0d191e4`) — the
  bracket-fill conversion strategy is no longer restricted to the
  pre-RMD "gap years" between retirement and SS/RMD start. The
  optimizer now sees conversions as available across the full
  horizon (still RMD-first within any given year).

### Changed
- `simulator.py` enforces all contribution caps from `limits.py`
  rather than silently letting the user over-contribute.
- README updated with new example-scenario paths and CLI usage notes
  (commit `00db500`).

### Tests
- New tests for `limits.py`, the deficit cascade, employer match,
  SS COLA + per-spouse claim, bracket indexing, and FICA. Total
  pre-Tier-B test count: ~211.

---

## [v2] — Stochastic baseline (pre-existing)

The capabilities below were already in the package before this
changelog started; they're the foundation that everything above
sits on top of. Kept here as a reference for the notebook recap.

### Already present
- `LognormalModel` (univariate, pre-correlation), `BootstrapModel`,
  `DeterministicModel` market models with `simulate_paths()` Monte
  Carlo and CVaR / `prob_success` metrics.
- `Mortality(year_of_death_a, year_of_death_b, ...)` with widow's-
  penalty filing-status flip, SS / pension scaling on death.
- `AssetLocation` per-account equity/bond split (bonds-in-pretax,
  equities-in-Roth) and `AssetLocation.uniform(equity_pct=...)`.
- `SpendingProfile.retirement_smile(...)` with per-age multipliers,
  `LumpEvent`s, and `LongTermCareShock`.
- `TCJA_EXTENDED` / `PRE_TCJA_2017` / `SUNSET_2026` regime swaps via
  `Config.tax_regime`, `regime_change_year_offset`,
  `regime_change_target`.
- `optimize_s3(cfg, inputs, objective='terminal' | 'cvar' | 'p_success')`
  via differential evolution over 401(k) Roth split + Roth
  conversion target bracket + bracket-fill target.
- `tornado_sensitivity()` for one-knob-at-a-time leverage analysis.
- `build_action_report()` Markdown/HTML/PDF report builder shared
  between the notebook and the `python -m tax_optimizer` CLI.

---

## How to update this changelog

1. **When you add a feature or fix a bug:** add a one-liner under
   the appropriate `### Added` / `### Changed` / `### Fixed` /
   `### Tests` / `### Docs` heading inside `[Unreleased]`. Mention
   the new symbol/field/file by name and (in one phrase) what it does.
2. **When cutting a release:** rename the `[Unreleased]` section
   header to `[vX.Y.Z] — <short-name> (YYYY-MM-DD, commit <sha>)`,
   then create a fresh empty `[Unreleased]` block at the top.
3. **Don't reorder old entries.** History should read in commit
   order (newest at top).
4. **Be specific about *where* the change lives** — file path or
   module name. The point is for a future reader to grep their way
   to the implementation in seconds.
