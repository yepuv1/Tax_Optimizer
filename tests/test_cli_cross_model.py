"""Tests for the `--cross-model` CLI flag.

Exercises both the argparse surface and the `_parse_cross_model_arg`
helper that maps user-supplied model names to concrete `MarketModel`
instances. End-to-end report rendering is already covered by
`tests/test_report_rd.py`; here we only verify the CLI plumbing.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tax_optimizer.__main__ import (
    CrossModelError,
    _build_parser,
    _parse_cross_model_arg,
)
from tax_optimizer.market import (
    BootstrapModel,
    HistoricalSequenceModel,
    LognormalModel,
)


# --------------------------------------------------------------------- argparse


class TestParserAcceptsCrossModel:
    def test_flag_absent_means_none(self):
        args = _build_parser().parse_args([])
        assert args.cross_model is None

    def test_flag_bare_means_empty_string(self):
        # `--cross-model` with no value collapses to the const sentinel "".
        args = _build_parser().parse_args(["--cross-model"])
        assert args.cross_model == ""

    def test_flag_with_value(self):
        args = _build_parser().parse_args(
            ["--cross-model", "bootstrap,historical_sequence"]
        )
        assert args.cross_model == "bootstrap,historical_sequence"

    def test_cross_model_paths_default(self):
        args = _build_parser().parse_args([])
        assert args.cross_model_paths == 200

    def test_cross_model_paths_override(self):
        args = _build_parser().parse_args(["--cross-model-paths", "500"])
        assert args.cross_model_paths == 500


# ---------------------------------------------------------------- model parser


class TestParseCrossModelArg:
    def test_none_returns_none(self):
        assert _parse_cross_model_arg(None) is None

    def test_empty_string_returns_none(self):
        # Empty / whitespace means "use cross_model_check defaults".
        assert _parse_cross_model_arg("") is None
        assert _parse_cross_model_arg("   ") is None

    def test_builtin_kinds_resolve(self):
        resolved = _parse_cross_model_arg("lognormal,bootstrap,historical_sequence")
        assert resolved is not None
        labels = [lbl for lbl, _m in resolved]
        models = [m for _lbl, m in resolved]
        assert labels == ["LognormalModel", "BootstrapModel", "HistoricalSequenceModel"]
        assert isinstance(models[0], LognormalModel)
        assert isinstance(models[1], BootstrapModel)
        assert isinstance(models[2], HistoricalSequenceModel)

    def test_historical_alias(self):
        # `historical` is an alias for `historical_sequence` for ergonomics.
        resolved = _parse_cross_model_arg("historical")
        assert resolved is not None
        assert resolved[0][0] == "HistoricalSequenceModel"
        assert isinstance(resolved[0][1], HistoricalSequenceModel)

    def test_case_insensitive(self):
        resolved = _parse_cross_model_arg("BOOTSTRAP,Lognormal")
        assert resolved is not None
        labels = [lbl for lbl, _m in resolved]
        assert labels == ["BootstrapModel", "LognormalModel"]

    def test_whitespace_tolerant(self):
        resolved = _parse_cross_model_arg(" bootstrap , vanguard_2025 ")
        assert resolved is not None
        labels = [lbl for lbl, _m in resolved]
        assert labels == ["BootstrapModel", "vanguard_2025"]

    def test_cma_preset_resolves_to_lognormal(self):
        resolved = _parse_cross_model_arg("vanguard_2025")
        assert resolved is not None
        label, model = resolved[0]
        assert label == "vanguard_2025"
        assert isinstance(model, LognormalModel)
        # Vanguard CMA equity μ = 5.5%.
        assert model.equity_mu == pytest.approx(0.055)

    def test_unknown_name_raises_clear_error(self):
        with pytest.raises(CrossModelError, match="unknown model 'frobnicate'"):
            _parse_cross_model_arg("frobnicate")

    def test_unknown_name_lists_known_options(self):
        # Error message should list the known options so the user can self-fix.
        with pytest.raises(CrossModelError) as exc_info:
            _parse_cross_model_arg("bogus")
        msg = str(exc_info.value)
        assert "bootstrap" in msg
        assert "historical_sequence" in msg
        assert "vanguard_2025" in msg

    def test_mixed_valid_with_invalid_still_raises(self):
        # First bad name should stop resolution.
        with pytest.raises(CrossModelError, match="unknown model 'oops'"):
            _parse_cross_model_arg("bootstrap,oops,vanguard_2025")

    def test_trailing_commas_skipped(self):
        resolved = _parse_cross_model_arg("bootstrap,,")
        assert resolved is not None
        assert [lbl for lbl, _m in resolved] == ["BootstrapModel"]


# ------------------------------------------------ end-to-end (subprocess smoke)


def _python_module(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "tax_optimizer", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=120,
    )


REPO_ROOT = Path(__file__).resolve().parent.parent


class TestCrossModelCLISmoke:
    """Slow-but-real subprocess tests so we know the flag wires up end-to-end."""

    def test_cross_model_requires_monte_carlo(self):
        result = _python_module("--cross-model", cwd=REPO_ROOT)
        assert result.returncode == 2
        assert "--cross-model requires --monte-carlo" in result.stderr

    def test_unknown_model_exits_cleanly(self):
        result = _python_module(
            "--monte-carlo", "10",
            "--cross-model", "no_such_model",
            cwd=REPO_ROOT,
        )
        assert result.returncode == 2
        assert "unknown model 'no_such_model'" in result.stderr

    def test_default_models_emit_cross_model_section(self, tmp_path: Path):
        out_path = tmp_path / "report.html"
        result = _python_module(
            "--scenario", "scenarios/example02.json",
            "--monte-carlo", "30",
            "--cross-model",
            "--cross-model-paths", "20",
            "--report", str(out_path),
            "--quiet",
            cwd=REPO_ROOT,
        )
        assert result.returncode == 0, result.stderr
        html = out_path.read_text()
        assert "Cross-model robustness check" in html
        assert "BootstrapModel" in html
        assert "HistoricalSequenceModel" in html

    def test_custom_model_list_with_cma_preset(self, tmp_path: Path):
        out_path = tmp_path / "report.html"
        result = _python_module(
            "--scenario", "scenarios/example02.json",
            "--monte-carlo", "30",
            "--cross-model", "bootstrap,vanguard_2025",
            "--cross-model-paths", "20",
            "--report", str(out_path),
            "--quiet",
            cwd=REPO_ROOT,
        )
        assert result.returncode == 0, result.stderr
        html = out_path.read_text()
        assert "Cross-model robustness check" in html
        assert "BootstrapModel" in html
        assert "vanguard_2025" in html
        # HistoricalSequenceModel must NOT appear — it wasn't in the list.
        # (LognormalModel is the "current" row, so we can't filter on that.)
        # Limit to the cross-model table rows by searching for the label cells.
        assert "<code>HistoricalSequenceModel</code>" not in html
