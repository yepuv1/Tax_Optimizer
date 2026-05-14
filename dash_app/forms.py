"""Declarative schema for every editable scenario field.

`FIELD_SCHEMA` is the single source of truth used to:

  * build the Simple and Advanced form tabs,
  * map form values back into a scenario JSON dict,
  * map a loaded scenario dict back into form values.

Each `FormField` declares:

  * `path`     - dotted path inside the scenario JSON, e.g.
                 "config.market.equity_mu" or
                 "inputs.starting.spouse_a_pretax_401k".
  * `label`    - human-readable form label.
  * `kind`     - one of "number", "percent", "int", "bool", "select", "text".
                 "percent" is just a number rendered with a "%" suffix; the
                 stored / loaded value is always a float (0.07, not 7).
  * `options`  - for "select", a list of `(value, label)` tuples.
  * `group`    - section heading inside the form.
  * `tier`     - "simple" or "advanced".
  * `step`     - optional step for `dcc.Input(type="number")`.
  * `min` / `max` - optional bounds (display-only; not enforced server-side).
  * `help`     - optional one-line description shown as the input's tooltip.

Markets and spending profiles are *discriminated unions* keyed on `kind`.
The schema lists every block's fields; `state.py` shapes them into the
nested JSON the simulator expects based on the active discriminator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class FormField:
    path: str
    label: str
    kind: str = "number"
    options: tuple[tuple[Any, str], ...] = ()
    group: str = "Misc"
    tier: str = "advanced"
    step: float | None = None
    min: float | None = None
    max: float | None = None
    help: str | None = None

    @property
    def component_id(self) -> dict[str, str]:
        """ID used for Dash pattern-matching callbacks.

        We use a `{type, path}` dict so a single callback can read every
        form input via `Input({"type": "form-input", "path": ALL}, "value")`.
        """
        return {"type": "form-input", "path": self.path}


# --- Choice tables ---------------------------------------------------

_TAX_REGIMES: tuple[tuple[str, str], ...] = (
    ("tcja", "TCJA (current law, 2018-2025)"),
    ("sunset", "Sunset / 2026 brackets"),
    ("pre_tcja", "Pre-TCJA (2017 brackets)"),
)

_STATE_REGIMES: tuple[tuple[str, str], ...] = (
    ("stateless", "Stateless (FL, TX, WA, NV, ...)"),
    ("ca", "California"),
    ("ny", "New York"),
    ("il", "Illinois"),
    ("ma", "Massachusetts"),
)

_REGIMES_NULLABLE: tuple[tuple[str | None, str], ...] = (
    (None, "(no regime swap)"),
    ("tcja", "TCJA"),
    ("sunset", "Sunset"),
    ("pre_tcja", "Pre-TCJA"),
)

_STATE_REGIMES_NULLABLE: tuple[tuple[str | None, str], ...] = (
    (None, "(no state regime swap)"),
    ("stateless", "Stateless"),
    ("ca", "California"),
    ("ny", "New York"),
    ("il", "Illinois"),
    ("ma", "Massachusetts"),
)

_WITHDRAWAL_STRATEGIES: tuple[tuple[str, str], ...] = (
    ("conventional", "Conventional (taxable -> pretax -> Roth)"),
    ("proportional", "Proportional"),
    ("bracket_fill", "Bracket-fill"),
)

_MARKET_KINDS: tuple[tuple[str, str], ...] = (
    ("lognormal", "Lognormal (stochastic)"),
    ("deterministic", "Deterministic (constant returns)"),
    ("bootstrap", "Bootstrap (block-resample empirical)"),
    ("historical_sequence", "Historical sequence"),
)

_SPENDING_KINDS: tuple[tuple[str, str], ...] = (
    ("flat", "Flat (inflation-indexed)"),
    ("smile", "Retirement smile (slow / go-go / no-go)"),
)


# --- Schema ---------------------------------------------------------

def _f(*args: Any, **kwargs: Any) -> FormField:
    return FormField(*args, **kwargs)


FIELD_SCHEMA: tuple[FormField, ...] = (
    # --- Household: ages & retirement (Simple) ---
    _f("inputs.spouse_a_age_start", "Spouse A current age", "int",
       group="Ages & retirement", tier="simple", min=18, max=100),
    _f("inputs.spouse_b_age_start", "Spouse B current age", "int",
       group="Ages & retirement", tier="simple", min=18, max=100),
    _f("inputs.spouse_a_retire_age", "Spouse A retire age", "int",
       group="Ages & retirement", tier="simple", min=40, max=100),
    _f("inputs.spouse_b_retire_age", "Spouse B retire age", "int",
       group="Ages & retirement", tier="simple", min=40, max=100),

    # --- Household: income (Simple) ---
    _f("inputs.income.spouse_a_gross", "Spouse A gross W-2", "number",
       group="Current income", tier="simple", step=1000),
    _f("inputs.income.spouse_b_gross", "Spouse B gross W-2", "number",
       group="Current income", tier="simple", step=1000),
    _f("inputs.income.spouse_a_bonus", "Spouse A bonus", "number",
       group="Current income", tier="simple", step=1000),
    _f("inputs.income.interest", "Taxable interest", "number",
       group="Current income", tier="simple", step=100),
    _f("inputs.income.capital_gains", "Realized capital gains", "number",
       group="Current income", tier="simple", step=100),
    _f("inputs.income.dividends", "Dividends", "number",
       group="Current income", tier="simple", step=100),

    # --- Household: contributions (Simple) ---
    _f("inputs.spouse_a_total_contrib_pct", "Spouse A total 401(k) %",
       "percent", group="401(k) contributions", tier="simple"),
    _f("inputs.spouse_b_total_contrib_pct", "Spouse B total 401(k) %",
       "percent", group="401(k) contributions", tier="simple"),
    _f("inputs.spouse_a_roth_401k_pct", "Spouse A Roth-401(k) %",
       "percent", group="401(k) contributions", tier="simple"),
    _f("inputs.spouse_b_roth_401k_pct", "Spouse B Roth-401(k) %",
       "percent", group="401(k) contributions", tier="simple"),
    _f("inputs.spouse_a_employer_match_rate", "Spouse A match rate",
       "percent", group="401(k) contributions", tier="simple"),
    _f("inputs.spouse_a_employer_match_max_pct", "Spouse A match max %",
       "percent", group="401(k) contributions", tier="simple"),
    _f("inputs.spouse_b_employer_match_rate", "Spouse B match rate",
       "percent", group="401(k) contributions", tier="simple"),
    _f("inputs.spouse_b_employer_match_max_pct", "Spouse B match max %",
       "percent", group="401(k) contributions", tier="simple"),
    _f("inputs.contrib.hsa_family", "HSA family contribution", "number",
       group="401(k) contributions", tier="simple", step=100),

    # --- Household: starting balances (Simple) ---
    _f("inputs.starting.spouse_a_pretax_401k",
       "Spouse A pretax 401(k)", "number",
       group="Starting balances", tier="simple", step=1000),
    _f("inputs.starting.spouse_b_pretax_401k",
       "Spouse B pretax 401(k)", "number",
       group="Starting balances", tier="simple", step=1000),
    _f("inputs.starting.spouse_a_pretax_ira",
       "Spouse A pretax IRA", "number",
       group="Starting balances", tier="simple", step=1000),
    _f("inputs.starting.spouse_b_pretax_ira",
       "Spouse B pretax IRA", "number",
       group="Starting balances", tier="simple", step=1000),
    _f("inputs.starting.spouse_a_roth_ira",
       "Spouse A Roth IRA", "number",
       group="Starting balances", tier="simple", step=1000),
    _f("inputs.starting.spouse_b_roth_ira",
       "Spouse B Roth IRA", "number",
       group="Starting balances", tier="simple", step=1000),
    _f("inputs.starting.hsa", "HSA balance", "number",
       group="Starting balances", tier="simple", step=1000),
    _f("inputs.starting.taxable_brokerage",
       "Taxable brokerage", "number",
       group="Starting balances", tier="simple", step=1000),
    _f("inputs.starting.pension_balance",
       "Pension cash balance", "number",
       group="Starting balances", tier="simple", step=1000),

    # --- Household: Social Security (Simple) ---
    _f("inputs.ss.monthly_spouse_a", "Spouse A PIA at FRA ($/mo)",
       "number", group="Social Security", tier="simple", step=50),
    _f("inputs.ss.monthly_spouse_b", "Spouse B PIA at FRA ($/mo)",
       "number", group="Social Security", tier="simple", step=50),
    _f("inputs.ss.start_age_a", "Spouse A claim age", "int",
       group="Social Security", tier="simple", min=62, max=70),
    _f("inputs.ss.start_age_b", "Spouse B claim age", "int",
       group="Social Security", tier="simple", min=62, max=70),

    # --- Household: pension (Simple) ---
    _f("inputs.pension.balance_today", "Pension balance today",
       "number", group="Pension", tier="simple", step=1000),
    _f("inputs.pension.monthly_at_nrd", "Pension annuity at NRD ($/mo)",
       "number", group="Pension", tier="simple", step=50),
    _f("inputs.pension.start_age", "Pension start age (NRD)", "int",
       group="Pension", tier="simple", min=50, max=80),

    # --- Macro / horizon (Simple) ---
    _f("config.start_year", "Start year", "int",
       group="Macro & horizon", tier="simple", min=2020, max=2050),
    _f("config.horizon_age", "Horizon age", "int",
       group="Macro & horizon", tier="simple", min=70, max=110),
    _f("config.annual_expenses_today", "Annual expenses today",
       "number", group="Macro & horizon", tier="simple", step=1000),
    _f("config.inflation", "Inflation", "percent",
       group="Macro & horizon", tier="simple"),
    _f("config.nominal_growth_rate",
       "Nominal growth rate (deterministic fallback)", "percent",
       group="Macro & horizon", tier="simple"),
    _f("config.tax_regime", "Federal tax regime", "select",
       options=_TAX_REGIMES,
       group="Macro & horizon", tier="simple"),
    _f("config.state_regime", "State tax regime", "select",
       options=_STATE_REGIMES,
       group="Macro & horizon", tier="simple"),

    # --- Macro / horizon (Advanced) ---
    _f("config.wage_growth", "Wage growth", "percent",
       group="Macro (advanced)", tier="advanced"),
    _f("config.taxable_drag", "Taxable account drag", "percent",
       group="Macro (advanced)", tier="advanced"),
    _f("config.ss_cola_rate",
       "SS COLA rate (blank = follow inflation)", "percent",
       group="Macro (advanced)", tier="advanced"),
    _f("config.bracket_indexing_rate",
       "Bracket indexing rate (blank = inflation)", "percent",
       group="Macro (advanced)", tier="advanced"),
    _f("config.regime_change_year_offset",
       "Regime swap year offset (blank = none)", "int",
       group="Macro (advanced)", tier="advanced", min=0, max=60),
    _f("config.regime_change_target", "Regime swap target", "select",
       options=_REGIMES_NULLABLE,
       group="Macro (advanced)", tier="advanced"),
    _f("config.state_regime_change_year_offset",
       "State regime swap year offset (blank = none)", "int",
       group="Macro (advanced)", tier="advanced", min=0, max=60),
    _f("config.state_regime_change_target", "State regime swap target",
       "select", options=_STATE_REGIMES_NULLABLE,
       group="Macro (advanced)", tier="advanced"),
    _f("config.section125_reduces_fica_wages",
       "S125 reduces FICA wages", "bool",
       group="Macro (advanced)", tier="advanced"),

    # --- Withdrawal & conversion strategy (Advanced) ---
    _f("config.withdrawal_strategy", "Withdrawal strategy", "select",
       options=_WITHDRAWAL_STRATEGIES,
       group="Withdrawal & conversion", tier="advanced"),
    _f("config.bracket_fill_target", "Bracket-fill target",
       "percent", group="Withdrawal & conversion", tier="advanced"),
    _f("config.roth_conversion_target_bracket",
       "Roth conversion target bracket", "percent",
       group="Withdrawal & conversion", tier="advanced"),
    _f("config.roth_conversion_amount",
       "Fixed Roth conversion amount", "number",
       group="Withdrawal & conversion", tier="advanced", step=1000),
    _f("config.cap_conversion_by_liquidity",
       "Cap conversion by liquidity", "bool",
       group="Withdrawal & conversion", tier="advanced"),
    _f("config.conversion_taxable_use_ratio",
       "Conversion taxable-use ratio", "percent",
       group="Withdrawal & conversion", tier="advanced"),
    _f("config.protect_roth_in_conversion_years",
       "Protect Roth in conversion years", "bool",
       group="Withdrawal & conversion", tier="advanced"),
    _f("config.rmd_start_age", "RMD start age", "int",
       group="Withdrawal & conversion", tier="advanced", min=72, max=80),
    _f("config.cap_gains_basis_fraction",
       "Capital gains cost-basis fraction", "percent",
       group="Withdrawal & conversion", tier="advanced"),
    _f("config.heir_marginal_rate", "Heir marginal rate", "percent",
       group="Withdrawal & conversion", tier="advanced"),
    _f("config.optimize_ss_claim_age",
       "Optimize SS claim age (S3)", "bool",
       group="Withdrawal & conversion", tier="advanced"),

    # --- Healthcare & Medicare (Advanced) ---
    _f("config.medicare_base_b_d_premium",
       "Medicare Part B+D base premium / yr", "number",
       group="Healthcare & Medicare", tier="advanced", step=100),
    _f("config.health_pre65_today", "Pre-Medicare health cost / yr",
       "number", group="Healthcare & Medicare", tier="advanced", step=100),
    _f("config.irmaa_lookback_years", "IRMAA lookback years", "int",
       group="Healthcare & Medicare", tier="advanced", min=0, max=3),
    _f("config.aca_enabled", "ACA premium tax credit enabled", "bool",
       group="Healthcare & Medicare", tier="advanced"),
    _f("config.aca_benchmark_premium_per_adult",
       "ACA benchmark premium / adult / yr", "number",
       group="Healthcare & Medicare", tier="advanced", step=500),
    _f("config.aca_max_contrib_pct", "ACA max contrib % of MAGI",
       "percent", group="Healthcare & Medicare", tier="advanced"),
    _f("config.stepup_at_first_death",
       "Step-up basis at first death", "bool",
       group="Healthcare & Medicare", tier="advanced"),

    # --- Taxable yields (Advanced) ---
    _f("config.taxable_equity_div_yield",
       "Taxable equity dividend yield", "percent",
       group="Taxable yields", tier="advanced"),
    _f("config.taxable_bond_interest_yield",
       "Taxable bond interest yield", "percent",
       group="Taxable yields", tier="advanced"),
    _f("config.taxable_equity_qualified_fraction",
       "Qualified-dividend fraction", "percent",
       group="Taxable yields", tier="advanced"),

    # --- Mortality (Advanced) ---
    _f("config.mortality.year_of_death_a",
       "Year of death - Spouse A", "int",
       group="Mortality", tier="advanced", min=1, max=80),
    _f("config.mortality.year_of_death_b",
       "Year of death - Spouse B", "int",
       group="Mortality", tier="advanced", min=1, max=80),
    _f("config.mortality.pension_survivor_pct",
       "Pension survivor %", "percent",
       group="Mortality", tier="advanced"),
    _f("config.mortality.ss_survivor_keeps_higher",
       "SS survivor keeps higher PIA", "bool",
       group="Mortality", tier="advanced"),

    # --- Market model (Advanced) ---
    _f("config.market.kind", "Market model", "select",
       options=_MARKET_KINDS,
       group="Market model", tier="advanced"),
    # Lognormal block
    _f("config.market.equity_mu", "Equity mu", "percent",
       group="Market model", tier="advanced"),
    _f("config.market.equity_sigma", "Equity sigma", "percent",
       group="Market model", tier="advanced"),
    _f("config.market.bond_mu", "Bond mu", "percent",
       group="Market model", tier="advanced"),
    _f("config.market.bond_sigma", "Bond sigma", "percent",
       group="Market model", tier="advanced"),
    _f("config.market.equity_bond_corr", "Equity-bond correlation",
       "number", group="Market model", tier="advanced", step=0.01),
    _f("config.market.cape_today",
       "CAPE today (blank = no scaling)", "number",
       group="Market model", tier="advanced", step=0.5),
    _f("config.market.cape_long_run", "CAPE long-run mean",
       "number", group="Market model", tier="advanced", step=0.5),
    # Deterministic block
    _f("config.market.equity",
       "Deterministic equity return (only if kind=deterministic)",
       "percent", group="Market model", tier="advanced"),
    _f("config.market.bond",
       "Deterministic bond return (only if kind=deterministic)",
       "percent", group="Market model", tier="advanced"),
    # Bootstrap block
    _f("config.market.block_size",
       "Bootstrap block size (only if kind=bootstrap)", "int",
       group="Market model", tier="advanced", min=1, max=20),

    # --- Asset location (Advanced) ---
    _f("config.asset_location.pretax_equity_pct",
       "Pretax equity %", "percent",
       group="Asset location", tier="advanced"),
    _f("config.asset_location.roth_equity_pct",
       "Roth equity %", "percent",
       group="Asset location", tier="advanced"),
    _f("config.asset_location.taxable_equity_pct",
       "Taxable equity %", "percent",
       group="Asset location", tier="advanced"),
    _f("config.asset_location.hsa_equity_pct",
       "HSA equity %", "percent",
       group="Asset location", tier="advanced"),

    # --- Spending profile (Advanced) ---
    _f("config.spending.kind", "Spending profile", "select",
       options=_SPENDING_KINDS,
       group="Spending profile", tier="advanced"),
    _f("config.spending.base_spending",
       "Base spending (today $)", "number",
       group="Spending profile", tier="advanced", step=1000),
    _f("config.spending.inflation",
       "Spending inflation", "percent",
       group="Spending profile", tier="advanced"),
    _f("config.spending.ltc_years",
       "LTC shock years (smile only)", "int",
       group="Spending profile", tier="advanced", min=0, max=15),
    _f("config.spending.ltc_annual_today",
       "LTC annual today $ (smile only)", "number",
       group="Spending profile", tier="advanced", step=1000),

    # --- IRA / Mega-backdoor (Advanced) ---
    _f("inputs.spouse_a_after_tax_401k_pct",
       "Spouse A after-tax 401(k) %", "percent",
       group="IRA & Mega-backdoor", tier="advanced"),
    _f("inputs.spouse_b_after_tax_401k_pct",
       "Spouse B after-tax 401(k) %", "percent",
       group="IRA & Mega-backdoor", tier="advanced"),
    _f("inputs.spouse_a_mega_backdoor_enabled",
       "Spouse A mega-backdoor enabled", "bool",
       group="IRA & Mega-backdoor", tier="advanced"),
    _f("inputs.spouse_b_mega_backdoor_enabled",
       "Spouse B mega-backdoor enabled", "bool",
       group="IRA & Mega-backdoor", tier="advanced"),
    _f("inputs.spouse_a_traditional_ira_contrib",
       "Spouse A Trad IRA contribution", "number",
       group="IRA & Mega-backdoor", tier="advanced", step=500),
    _f("inputs.spouse_b_traditional_ira_contrib",
       "Spouse B Trad IRA contribution", "number",
       group="IRA & Mega-backdoor", tier="advanced", step=500),
    _f("inputs.spouse_a_roth_ira_contrib",
       "Spouse A direct Roth IRA contribution", "number",
       group="IRA & Mega-backdoor", tier="advanced", step=500),
    _f("inputs.spouse_b_roth_ira_contrib",
       "Spouse B direct Roth IRA contribution", "number",
       group="IRA & Mega-backdoor", tier="advanced", step=500),
    _f("inputs.spouse_a_backdoor_roth",
       "Spouse A backdoor Roth", "bool",
       group="IRA & Mega-backdoor", tier="advanced"),
    _f("inputs.spouse_b_backdoor_roth",
       "Spouse B backdoor Roth", "bool",
       group="IRA & Mega-backdoor", tier="advanced"),

    # --- Social Security (Advanced) ---
    _f("inputs.ss.fra_a", "Spouse A FRA", "int",
       group="Social Security (advanced)", tier="advanced", min=62, max=70),
    _f("inputs.ss.fra_b", "Spouse B FRA", "int",
       group="Social Security (advanced)", tier="advanced", min=62, max=70),

    # --- Pension (Advanced) ---
    _f("inputs.pension.years_of_service_today",
       "Years of service today", "int",
       group="Pension (advanced)", tier="advanced", min=0, max=60),
    _f("inputs.pension.pre_2016_participant",
       "Pre-2016 participant (5% floor)", "bool",
       group="Pension (advanced)", tier="advanced"),
    _f("inputs.pension.interest_rate",
       "Interest rate (blank = floor)", "percent",
       group="Pension (advanced)", tier="advanced"),
    _f("inputs.pension.irs_comp_limit_today",
       "IRS comp limit today $", "number",
       group="Pension (advanced)", tier="advanced", step=1000),

    # --- Health premiums (Advanced) ---
    _f("inputs.health_premiums.spouse_a_medical",
       "Spouse A medical premium", "number",
       group="Health premiums", tier="advanced", step=100),
    _f("inputs.health_premiums.spouse_a_dental",
       "Spouse A dental premium", "number",
       group="Health premiums", tier="advanced", step=50),
    _f("inputs.health_premiums.spouse_a_vision",
       "Spouse A vision premium", "number",
       group="Health premiums", tier="advanced", step=50),
    _f("inputs.health_premiums.spouse_b_medical",
       "Spouse B medical premium", "number",
       group="Health premiums", tier="advanced", step=100),
    _f("inputs.health_premiums.spouse_b_dental",
       "Spouse B dental premium", "number",
       group="Health premiums", tier="advanced", step=50),
    _f("inputs.health_premiums.spouse_b_vision",
       "Spouse B vision premium", "number",
       group="Health premiums", tier="advanced", step=50),
)


def fields_by_tier(tier: str) -> tuple[FormField, ...]:
    """Return the form fields belonging to `tier` ('simple' or 'advanced')."""
    return tuple(f for f in FIELD_SCHEMA if f.tier == tier)


def fields_by_group(fields: Iterable[FormField]) -> dict[str, list[FormField]]:
    """Group fields by their `group` for sectioned form rendering."""
    out: dict[str, list[FormField]] = {}
    for f in fields:
        out.setdefault(f.group, []).append(f)
    return out


_FIELDS_BY_PATH: dict[str, FormField] = {f.path: f for f in FIELD_SCHEMA}


def get_field(path: str) -> FormField | None:
    return _FIELDS_BY_PATH.get(path)


def all_paths() -> tuple[str, ...]:
    return tuple(f.path for f in FIELD_SCHEMA)
