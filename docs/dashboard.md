# Running the Dash dashboard

The Plotly Dash web UI is the visual companion to `python -m
tax_optimizer`. It wraps every scenario knob in a form, runs the
simulator on demand, and renders the same action-plan report the CLI
emits — but interactively, with live charts and per-field help
tooltips. This page covers everything you need to run it.

> Looking for the Python API or the CLI? See the
> [main README](../README.md). For the scenario JSON schema, see
> [`docs/scenario_guide.md`](scenario_guide.md).

## Contents

- [Quickstart](#quickstart)
- [Launching the dashboard](#launching-the-dashboard)
  - [Module entry point](#module-entry-point)
  - [Console script](#console-script)
  - [Host / port / debug options](#host--port--debug-options)
  - [Environment variables](#environment-variables)
- [The UI at a glance](#the-ui-at-a-glance)
- [Sidebar walkthrough](#sidebar-walkthrough)
  - [Top bar](#top-bar)
  - [Scenario form (Simple / Advanced)](#scenario-form-simple--advanced)
  - [Run mode selector](#run-mode-selector)
- [Results panel walkthrough](#results-panel-walkthrough)
  - [Overview tab](#overview-tab)
  - [Taxes tab](#taxes-tab)
  - [Strategies tab](#strategies-tab)
  - [Monte Carlo tab](#monte-carlo-tab)
  - [Year-by-year tab](#year-by-year-tab)
  - [Report tab](#report-tab)
- [Common workflows](#common-workflows)
- [Tips and shortcuts](#tips-and-shortcuts)
- [Troubleshooting](#troubleshooting)

---

## Quickstart

```bash
git clone https://github.com/vijayyepuri/Tax_Optimizer.git
cd Tax_Optimizer
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .                   # dash, plotly, etc. are base deps
python -m dash_app                 # opens http://127.0.0.1:8050
```

Then open <http://127.0.0.1:8050> in your browser. The form starts
populated with the package defaults (a 54/53-year-old MFJ couple
with $371k combined gross W-2). Click **Run** to simulate; the
charts and the action plan render in seconds.

> The `dash`, `dash-bootstrap-components`, and `plotly` packages are
> all listed under the base `dependencies` in `pyproject.toml`, so
> a plain `pip install -e .` is enough — there's no separate `[dash]`
> extra to remember.

## Launching the dashboard

You can boot the app three ways.

### Module entry point

```bash
python -m dash_app
```

This is the canonical form. It works from any directory as long as
the `tax_optimizer` package is installed (or you're inside the repo
with the `.venv` activated). Hits the same code path as the console
script below.

### Console script

`pip install -e .` registers a `tax-optimizer-app` shim on your
`PATH` (declared in `pyproject.toml`'s `[project.scripts]` block).
After install:

```bash
tax-optimizer-app                  # same as `python -m dash_app`
tax-optimizer-app --port 9000      # serve on a different port
```

### Host / port / debug options

The entry point accepts three flags:

| Flag | Default | Purpose |
|---|---|---|
| `--host HOST` | `127.0.0.1` | Bind address. Use `0.0.0.0` to expose to the local network. |
| `--port PORT` | `8050` | TCP port to listen on. |
| `--debug` | off | Run in Dash debug mode: hot-reload on Python file changes, plus the in-page dev console for callback inspection. |

```bash
# Local-only (default).
python -m dash_app

# Expose to phones / tablets on the same Wi-Fi for a quick demo.
python -m dash_app --host 0.0.0.0 --port 8050

# Develop with hot-reload on every save.
python -m dash_app --debug
```

### Environment variables

Both options also read environment variables, which is convenient
for `systemd`, Docker, or `direnv` setups:

| Variable | Same as |
|---|---|
| `DASH_HOST` | `--host` |
| `DASH_PORT` | `--port` |

CLI flags take precedence over environment variables.

---

## The UI at a glance

```
┌────────────────────────────────────┬─────────────────────────────────────┐
│  Sidebar                           │  Results panel                      │
│                                    │                                     │
│  ┌────────────────────────────┐   │  ┌─Overview─Taxes─Strategies─MC─    │
│  │ Top bar:                   │   │  │  Year-by-year─Report────────┐    │
│  │  Load JSON / Save JSON /   │   │  │                              │    │
│  │  Run mode / Run button     │   │  │  Charts + tables for the     │    │
│  └────────────────────────────┘   │  │  active tab                  │    │
│                                    │  │                              │    │
│  ┌─Simple──Advanced─┐              │  │                              │    │
│  │                   │              │  │                              │    │
│  │  Form sections    │              │  │                              │    │
│  │  (accordion):     │              │  │                              │    │
│  │   Ages & retire   │              │  │                              │    │
│  │   Current income  │              │  │                              │    │
│  │   401(k) contribs │              │  │                              │    │
│  │   Starting balance│              │  │                              │    │
│  │   Social Security │              │  │                              │    │
│  │   Pension         │              │  │                              │    │
│  │   Macro & horizon │              │  │                              │    │
│  │   ...             │              │  └──────────────────────────────┘   │
│  └───────────────────┘              │                                     │
└────────────────────────────────────┴─────────────────────────────────────┘
```

Two columns: the **sidebar** on the left holds every input plus the
controls that drive a run; the **results panel** on the right shows
the simulation output across six tabs. The page renders responsively
— on narrow displays the panels stack vertically.

---

## Sidebar walkthrough

### Top bar

```
┌─────────────────────────────────────────────────────────────────────┐
│  Load scenario JSON                          Save scenario           │
│  ┌──────────────────────────────────────┐   ┌────────────────────┐  │
│  │  📁  Drag & drop or browse...        │   │  Download JSON     │  │
│  └──────────────────────────────────────┘   └────────────────────┘  │
│                                                                      │
│  Run mode                                    Run                     │
│  ┌──────────────────────────────────────┐   ┌────────────────────┐  │
│  │ Four strategies + Monte Carlo  ▾     │   │       ▶ Run         │  │
│  └──────────────────────────────────────┘   └────────────────────┘  │
│                                                                      │
│  Status: Ran 'four_strategies_mc' in 18.7s. Winner: S3_optimized.   │
└─────────────────────────────────────────────────────────────────────┘
```

- **Load scenario JSON** — drag & drop a JSON file (or click to
  browse) to populate the entire form from a saved scenario. Same
  file format as `python -m tax_optimizer --scenario PATH.json`.
  Unknown keys raise a friendly error; deprecated keys (e.g. the
  old `inputs.annual_expenses`) are migrated silently with a
  warning in the status banner.
- **Download JSON** — saves the current form state as a scenario
  JSON. The file is byte-identical to what you'd hand to the CLI,
  so you can iterate in the dashboard and then run the headless
  CLI for regression / archival.
- **Run mode** — controls how the **Run** button drives the
  simulator. See [Run mode selector](#run-mode-selector) below.
- **Run** — kicks off the simulation. Disabled (greyed out) while
  a run is in progress; the status line below shows progress, the
  elapsed time, and the winning strategy when it completes.

### Scenario form (Simple / Advanced)

The form is generated from
[`dash_app/forms.py:FIELD_SCHEMA`](../dash_app/forms.py) and split
into two tabs:

- **Simple** — the ~30 fields you'll touch on most runs: ages,
  retirement ages, gross W-2 incomes, contribution rates, starting
  balances, Social Security PIAs, pension data, and the headline
  macro knobs (start year, horizon, inflation, tax regime).
- **Advanced** — the remaining ~85 fields covering Roth conversion
  strategy, market-model parameters, asset location, spending
  profile (flat / smile / custom phases), mortality, healthcare
  premiums, IRA mechanics (mega-backdoor, backdoor Roth, direct
  contributions), pension internals, and ACA / IRMAA tuning.

Both tabs are organized as a Bootstrap accordion. **Simple** opens
the first section by default; **Advanced** starts collapsed so the
sidebar isn't overwhelming on first load.

#### Per-field help tooltips

Every field shows a small ⓘ icon next to its label. Hover the icon
to see a one-sentence hint that explains the unit, expands the
abbreviation (PIA, FRA, IRMAA, MAGI, FICA, RMD, NRD, …), and notes
the practical effect of changing the value:

```
Spouse A PIA at FRA ($/mo)  ⓘ  ┐
                                │  Spouse A's monthly Primary
                                │  Insurance Amount at Full
                                │  Retirement Age, in today's
                                │  dollars (read off your SSA
                                │  statement).
                                ┘
```

The icon also has a native HTML `title=` attribute, so the hint is
discoverable for screen readers and works without JavaScript.

#### Decimal-percent convention

Percent fields are stored as **decimals** (0.07, not 7), so the
JSON round-trips byte-for-byte with the CLI scenario format. The
form labels every percent field with `(decimal, e.g. 0.07)` to
make the convention obvious. If you type `7` thinking "7%", the
simulator will treat it as 700% and the run will look very strange
— always type `0.07`.

### Run mode selector

| Mode | What runs | Approx wall time† |
|---|---|---|
| **Single sim** | One deterministic path through the household. Fastest. Useful for "did I enter the form correctly?". | < 1 s |
| **Four strategies** | The four canonical strategies side-by-side: `S0_baseline`, `S1_all_roth_401k`, `S2_bracket_fill_22`, `S3_optimized` (differential evolution over the optimizer's decision axes). | 2-5 s |
| **Four strategies + Monte Carlo** | The four strategies above, **plus** a 200-path Monte Carlo for sequence-of-returns risk. Adds the Monte Carlo tab and the risk-picture rows in the report. | 15-30 s |

† at default `horizon_age=95` with the package defaults; longer
horizons and larger Monte Carlo path counts scale roughly linearly.

The first time you click **Run** the simulator caches the result
server-side under a UUID; the cache holds the last 5 runs (LRU).
That cache backs the **Report** tab and the **Download HTML**
button — switching back to a recent run won't trigger a recompute.

---

## Results panel walkthrough

### Overview tab

Headline KPI tiles (terminal after-tax NW, lifetime federal tax
NPV, peak marginal rate, Monte Carlo probability of success) plus
a stacked-area chart of every account balance over the horizon for
the **winning** strategy. The first place to look post-run.

### Taxes tab

Two charts:
- A breakdown of federal vs. state vs. NIIT vs. IRMAA vs. FICA per
  year for the winning strategy.
- The Roth conversion + RMD profile per year, so you can see when
  the optimizer chose to do conversions and how RMDs eventually
  take over.

### Strategies tab

Top: a per-knob × per-strategy comparison table that highlights
**which decision parameters each strategy overrode** relative to
the baseline. The optimizer's overrides get a green tint; other
strategies' overrides get a yellow tint. Outcome rows (terminal
NW, lifetime tax, lifetime IRMAA, peak marginal) sit at the bottom
of the same table for visual correlation.

Below: bar charts of the headline outcome metrics across the four
strategies, colored so the optimizer (S3) pops.

In **single-sim** mode there's nothing to compare — the table
collapses to a single-line "Optimizer picks" callout instead.

### Monte Carlo tab

Only populated when you ran in **Four strategies + Monte Carlo**
mode. Two charts:
- Histogram of terminal NW across paths, with P10 / P50 / P90 cuts.
- Fan chart of liquid NW per year (P10 / P50 / P90 envelope) so
  you can see when the spread widens (typically at retirement +
  during the LTC shock).

### Year-by-year tab

A sortable, filterable Dash DataTable of the full per-year
DataFrame for any one strategy (drop-down at the top selects
which). ~80 columns: balances, contributions, income, RMDs,
conversions, withdrawals, federal / state / NIIT tax, IRMAA tier,
ACA credit, etc. Use the column filters to narrow down a specific
year or value range.

### Report tab

The full action-plan report — the same document `python -m
tax_optimizer --report report.html` produces — rendered inline in
an `<iframe>`. Auto-renders when you switch to the tab after a
run; re-renders when you run again. The **Download HTML** button
saves a copy as a self-contained HTML file you can email, print,
or convert to PDF via your browser's "Print to PDF".

The first render of the report runs a tornado-sensitivity sweep
(~1-2 s on the default scenario). Switching away from the tab and
back is instant — the rendered HTML is memoized server-side
alongside the cached run.

---

## Common workflows

### "I just want to play with my numbers"

1. `python -m dash_app`
2. Open <http://127.0.0.1:8050>
3. On the **Simple** tab, edit your ages, salaries, and starting
   balances.
4. Set **Run mode** to **Four strategies** (good starting point).
5. Click **Run**.
6. Inspect the **Strategies** tab to see what the optimizer
   overrode versus the baseline. Iterate on the Simple tab and
   click **Run** again.

### "I have a saved scenario JSON"

1. Click the **Load scenario JSON** drop zone, drag your file in
   (or click to browse).
2. The form populates from the file. Status banner confirms the
   load and lists any deprecated-key migrations.
3. Click **Run** (the run mode is independent of the file).

### "I want to share / archive the current scenario"

1. Tweak the form to your liking.
2. Click **Download JSON** in the top bar.
3. The browser saves `scenario.json` to your downloads folder.
4. Run that file headlessly with `python -m tax_optimizer
   --scenario scenario.json` to get the same numbers without the
   dashboard.

### "I want a printed action plan"

1. Set **Run mode** to **Four strategies** (or **Four + MC** if
   you want the risk-picture section).
2. Click **Run** and wait for it to finish.
3. Switch to the **Report** tab. The full action-plan HTML renders
   inline in seconds.
4. Click **Download HTML** for a saved copy, or use your browser's
   **Print → Save as PDF** while viewing the iframe (the report's
   embedded `@page` CSS produces nicely paginated output).

### "I want to debug a specific year"

1. Run the simulator (any mode).
2. Switch to the **Year-by-year** tab.
3. Pick the strategy of interest from the drop-down.
4. Use the column filters to narrow to the year(s) you care about.
   The full ~80 columns are exposed (every diagnostic the simulator
   computes), so you can audit RMDs, conversion sizing, IRMAA
   thresholds, etc.

### "I want to develop on the dashboard itself"

1. `python -m dash_app --debug`
2. The Dash dev console appears in the bottom-right of the page
   (callback graph, error traces, performance breakdown).
3. Edit any file under `dash_app/` — the page hot-reloads on save.
4. Editing `tax_optimizer/*.py` also triggers a reload (the import
   graph is watched).

---

## Tips and shortcuts

- **Per-field hints** — every input has an ⓘ icon next to its
  label with a one-sentence hint. Hover to read.
- **Monospace UI** — the dashboard renders in
  [Fira Code](https://github.com/tonsky/FiraCode) so currency
  columns line up by digit position. The action-plan report
  inside the **Report** tab keeps its CLI typography (sans-serif
  body + monospace code) so the iframe document looks identical
  to the downloaded HTML.
- **Persistence** — the form state is **not** auto-saved across
  reloads. Use **Download JSON** before closing the browser if
  you want to come back to the same scenario.
- **Run cache** — the last 5 runs stay in memory keyed by UUID.
  The **Download HTML** button and the **Report** tab share that
  cache, so the report's tornado sweep runs at most once per
  scenario.
- **Browser tab survives a run** — if you close the browser and
  re-open <http://127.0.0.1:8050>, you'll see an empty form even
  though the server is still up. Re-loading a scenario JSON is
  the fastest way to get back to where you were.
- **Decimal-percent fields** — always enter `0.07` for 7%. The
  form label `(decimal, e.g. 0.07)` is the cue.
- **The Run button is sticky** — it stays disabled while a run is
  in progress so accidental double-clicks don't queue a second
  run. Watch the status line below it for completion.

---

## Troubleshooting

### `Address already in use` on port 8050

Another Dash app (or a previous instance that didn't shut down
cleanly) still owns the port. Either:

```bash
# Use a different port
python -m dash_app --port 8051

# Or find and kill the previous process
lsof -ti:8050 | xargs kill -9      # macOS / Linux
```

### "Run results have expired from the cache"

The dashboard's run cache is bounded to the last 5 runs. If you've
clicked **Run** more than 5 times in this session, the older runs
get evicted and their **Download HTML** / **Report** tab become
unavailable. Click **Run** again to refresh.

### Report tab shows a blank iframe

This usually means the iframe is still rendering the first call
to `tornado_sensitivity` (~1-2 s). The `dcc.Loading` spinner
should be visible while the build is in flight. If the iframe
stays blank > 30 s, check the browser dev console for callback
errors and re-run.

### Form fields render with red text

This was a pre-v6.8 bug from HTML5 step-validation flagging
legitimate decimal values as `:invalid`. Fixed in the current
release — `step="any"` is now used on every percent / number
field. If you still see red text, you're running an older build
(check `tax_optimizer.__version__`).

### Charts don't update after editing the form

Editing a form field doesn't auto-rerun the simulator (would be
expensive). You have to click **Run**. The status banner is your
confirmation that a run completed.

### Dash deprecation warnings on startup

`dash.dash_table.DataTable` is being deprecated in favor of
`dash-ag-grid`. The warning is benign and the dashboard works fine
on the current Dash 4.x; we'll migrate when AG-Grid's column-filter
ergonomics catch up to DataTable's.

### `python -m dash_app` says `No module named dash_app`

The package isn't installed. `cd` into the repo root and run
`pip install -e .`, then try again. Don't `cd` into `dash_app/`
itself — Python module resolution needs the parent directory on
`PYTHONPATH`.

---

For dashboard-specific bugs, open an issue on
<https://github.com/vijayyepuri/Tax_Optimizer/issues> with:

- Your OS + Python version (`python --version`).
- The output of `pip show dash dash-bootstrap-components plotly`.
- Steps to reproduce (a scenario JSON via **Download JSON**
  attached to the issue is ideal).
- Any tracebacks from the terminal that booted `python -m dash_app`.
