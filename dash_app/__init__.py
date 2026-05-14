"""Plotly Dash front-end for the tax optimizer.

The package wraps `tax_optimizer`'s public APIs (`scenario`, `simulator`,
`optimizer`, `monte_carlo`) with a single-page Dash dashboard:

  * a Simple/Advanced form that round-trips the full scenario JSON,
  * a run-mode selector (single sim / four strategies / four + Monte Carlo),
  * an interactive results panel built from Plotly figures.

Nothing in this package modifies `tax_optimizer/*`; we only consume it.
"""
