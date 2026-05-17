"""Production entry point for the Dash app.

The default ``tax-optimizer-app`` console script runs under Flask /
Werkzeug's built-in development server, which prints

    WARNING: This is a development server. Do not use it in a
    production deployment. Use a production WSGI server instead.

For local single-user planning that warning is harmless. For anything
beyond that — multi-user access, machine-to-machine calls, sustained
load — use ``tax-optimizer-app-prod`` instead, which serves
``app.server`` (the underlying Flask WSGI app) through
`waitress <https://github.com/Pylons/waitress>`_, a production-grade
pure-Python WSGI server.

Usage::

    pip install -e ".[prod]"
    tax-optimizer-app-prod                         # 127.0.0.1:8050
    tax-optimizer-app-prod --host 0.0.0.0 --port 9000
    tax-optimizer-app-prod --threads 8

Environment variables ``DASH_HOST`` and ``DASH_PORT`` are honored as
defaults (matching the dev entry point).

Why ``waitress`` rather than ``gunicorn``? Waitress is pure Python and
runs on every platform (including Windows); gunicorn is fork-based and
only supports POSIX. For a small, self-hosted planner where the
operational target is "any laptop", waitress hits the right
simplicity/portability point.
"""

from __future__ import annotations

import argparse
import os
import sys

from .app import make_app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tax-optimizer-app-prod",
        description=(
            "Production WSGI launcher for the Dash app (uses waitress). "
            "Silences the development-server warning printed by the default "
            "tax-optimizer-app entry point."
        ),
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("DASH_HOST", "127.0.0.1"),
        help="Bind host (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("DASH_PORT", "8050")),
        help="Bind port (default: 8050).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help=(
            "Number of worker threads waitress should use. "
            "Default 4 — adequate for single-user / small-team use."
        ),
    )
    args = parser.parse_args(argv)

    # Defer the import so a missing ``waitress`` doesn't break the
    # package import surface (the dev entry point keeps working
    # without the prod extra installed).
    try:
        from waitress import serve  # type: ignore[import-not-found]
    except ImportError:
        sys.stderr.write(
            "tax-optimizer-app-prod requires the 'prod' extra. Install it with:\n"
            '    pip install -e ".[prod]"\n'
        )
        return 1

    app = make_app()
    sys.stdout.write(
        f"Serving Dash on http://{args.host}:{args.port} "
        f"(waitress, {args.threads} threads). Press Ctrl+C to stop.\n"
    )
    serve(
        app.server,
        host=args.host,
        port=args.port,
        threads=args.threads,
        # `ident` is what waitress reports in the Server: header; set
        # something neutral rather than leaking the waitress version.
        ident="tax-optimizer-app",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
