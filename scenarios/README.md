# Scenarios

This directory holds JSON scenario files consumed by `tax_optimizer`.
A scenario is just a `{ "config": {...}, "inputs": {...} }` dict that
overrides defaults on `Config` / `Inputs`. The CLI loads them with
`--scenario PATH`, and the library API uses `apply_scenario(cfg,
inputs, scenario_dict)`.

## Files

| File              | Purpose                                                                                              |
| ----------------- | ---------------------------------------------------------------------------------------------------- |
| `template.json`   | **Canonical reference**: every available knob, with its default value. Copy this to start a new plan. |
| `example01.json`  | Mid-career dual-income household with a widow stress test.                                            |
| `example02.json`  | Late-career household with mega-backdoor Roth, pension, and aggressive savings.                       |

## Maintenance contract for new knobs

**Whenever a new field is added to `Config`, `Inputs`, or one of the
nested input dataclasses (`StartingBalances`, `CurrentIncome`,
`CurrentContrib`, `PensionInputs`, `SocialSecurity`) or to `Mortality`,
the contributor MUST also:**

1. Add the field (with its default value) to `template.json`.
2. If the new knob is meaningful for the two narrative scenarios,
   reflect it in `example01.json` / `example02.json` too (else they
   silently fall back to defaults).
3. Re-run the drift tests:

   ```
   pytest tests/test_scenario_template.py
   ```

   These tests **will fail loudly** if `template.json` is missing a
   new dataclass field or has a value that disagrees with the
   field's default. The failure message points at the exact field
   and which side to update.

`scenarios/example02.local.json` is a developer's personal scenario,
listed in `.gitignore`. Do not commit personal financial data.

## Polymorphic blocks

`config.market` and `config.spending` are tagged-union shapes. The
template renders ONE representative form (lognormal + smile) so the
file remains a useful copy-paste starting point. Other shapes the
loader accepts:

### `config.market`

```jsonc
// kind: "deterministic" — constant returns
{ "kind": "deterministic", "equity": 0.07, "bond": 0.04 }

// kind: "lognormal" — independent yearly draws (default; CAPE-aware)
{ "kind": "lognormal", "equity_mu": 0.07, "equity_sigma": 0.18,
  "bond_mu": 0.04, "bond_sigma": 0.06, "equity_bond_corr": 0.10,
  "cape_today": null, "cape_long_run": 16.5 }

// kind: "lognormal" with a CMA preset shortcut
{ "kind": "lognormal", "cma": "vanguard_2025" }

// kind: "bootstrap" — block-bootstrap from history
{ "kind": "bootstrap", "block_size": 5 }

// kind: "historical_sequence" — replay contiguous N-year slices
{ "kind": "historical_sequence" }
```

### `config.spending`

```jsonc
// kind: "flat" — constant real spending
{ "kind": "flat", "base_spending": 85000, "inflation": 0.025 }

// kind: "smile" — Blanchett "retirement smile" + LTC shock at end of life
{ "kind": "smile", "base_spending": 85000, "inflation": 0.025,
  "ltc_years": 3, "ltc_annual_today": 80000 }

// kind: "custom" — arbitrary phases / lump events
{ "kind": "custom", "base_spending": 85000, "inflation": 0.025,
  "phases": [{ "age_lo": 65, "age_hi": 200, "multiplier": 1.0,
               "label": "retirement" }],
  "lump_events": [{ "year_offset": 5, "amount_today": 50000,
                    "label": "kitchen-remodel" }],
  "ltc_shock": { "years": 3, "annual_cost_today": 80000 } }
```

### `config.state_regime`

String code: `"stateless"` (default), `"CA"`, `"NY"`, `"IL"`, `"MA"`.

### `config.tax_regime` / `config.regime_change_target`

String code: `"tcja"` (alias `"tcja_extended"`), `"pre_tcja"` (alias
`"pre_tcja_2017"`), `"sunset"` (alias `"sunset_2026"`).

### `config.asset_location`

Either `{ "uniform_equity_pct": 0.7 }` (every account allocated the
same way) OR the per-bucket form shown in `template.json`.

## Useful CLI patterns

Generate a fresh defaults dump (newer than `template.json` if you've
been editing on a feature branch):

```
tax-optimizer --print-defaults > /tmp/defaults.json
```

Run a scenario:

```
tax-optimizer --scenario scenarios/example01.json
```

Override a single value without editing the file:

```
tax-optimizer --scenario scenarios/example01.json \
  --set config.roth_conversion_target_bracket=0.22
```
