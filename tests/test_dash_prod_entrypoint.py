"""Smoke tests for ``dash_app.prod`` (the ``tax-optimizer-app-prod``
console entry point).

We don't actually boot waitress here — that would block — we just
verify:

  * the ``--help`` path exits 0 and writes usage text;
  * the friendly-error path fires when ``waitress`` is missing;
  * the package exposes ``main`` for the ``[project.scripts]`` mapping.
"""

from __future__ import annotations

import builtins
import sys

import pytest

import dash_app.prod as prod


def test_main_help_exits_clean(capsys):
    with pytest.raises(SystemExit) as excinfo:
        prod.main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "tax-optimizer-app-prod" in out
    assert "--threads" in out


def test_main_friendly_error_when_waitress_missing(monkeypatch, capsys):
    """If the user installed plain `tax-optimizer` without the prod extra
    and tries to run ``tax-optimizer-app-prod``, they should get a clear
    "install the prod extra" message and exit code 1 — NOT an opaque
    ImportError traceback."""
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "waitress":
            raise ImportError("simulated missing waitress")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    # Also evict any cached waitress module so the import attempt
    # actually re-runs.
    monkeypatch.delitem(sys.modules, "waitress", raising=False)

    rc = prod.main(["--port", "0"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "prod" in err  # references the [prod] extra


def test_entry_point_callable_signature_matches_pyproject():
    # The pyproject.toml mapping is `tax-optimizer-app-prod = "dash_app.prod:main"`,
    # so `main` must be a top-level callable accepting an optional argv list.
    assert callable(prod.main)
    # Calling with --help raises SystemExit; the smoke test above covers
    # the actual exit path. Here we just confirm the callable surface.
