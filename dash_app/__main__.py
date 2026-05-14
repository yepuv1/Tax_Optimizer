"""`python -m dash_app` entry point.

Boots the Dash app on http://127.0.0.1:8050 by default. Override host
and port via the `DASH_HOST` / `DASH_PORT` environment variables, or
via CLI arguments (`--host`, `--port`).
"""

from __future__ import annotations

import argparse
import os
import sys

from .app import make_app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tax-optimizer-app",
        description="Interactive Plotly Dash front-end for tax_optimizer.",
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
        "--debug",
        action="store_true",
        help="Run in Dash debug mode (hot-reload, dev console).",
    )
    args = parser.parse_args(argv)

    app = make_app()
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    sys.exit(main())
