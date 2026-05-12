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
