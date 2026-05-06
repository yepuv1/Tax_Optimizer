"""Pretty-printing the action plan to terminal, HTML, and PDF.

The CLI used to dump raw markdown to stdout and write `.md` files. That
worked but was ugly. This module replaces that with three rendering
backends:

  * `render_terminal(md)` -- pretty-print via Rich when stdout is a TTY,
    fall back to raw markdown when stdout is piped/redirected (so you
    can still pipe into `glow`, `pandoc`, `mdcat`, etc.).
  * `write_html(md, path)` -- markdown -> standalone HTML document with
    a clean built-in stylesheet.
  * `write_pdf(md, path)` -- the same HTML rendered to PDF via
    WeasyPrint. WeasyPrint is an optional dependency (`tax-optimizer[pdf]`)
    because it needs system libraries (pango, cairo); we raise a friendly
    install hint if it's not available.

`write_report(md, path)` dispatches based on the file extension. Only
`.html`, `.htm`, and `.pdf` are accepted.
"""

from __future__ import annotations

import html as _html
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

OutputFormat = Literal["html", "pdf"]


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    @page {{
      size: Letter;
      margin: 0.7in;
      @bottom-right {{
        content: counter(page) " / " counter(pages);
        font-size: 9pt;
        color: #94a3b8;
      }}
      @bottom-left {{
        content: "tax_optimizer  ·  {generated}";
        font-size: 9pt;
        color: #94a3b8;
      }}
    }}
    html {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
    body {{
      font-family: -apple-system, "Helvetica Neue", "Segoe UI", Roboto, Arial, sans-serif;
      color: #0f172a;
      line-height: 1.55;
      max-width: 920px;
      margin: 2.2rem auto;
      padding: 0 1.5rem 2rem;
      font-size: 11pt;
    }}
    h1 {{
      color: #0f172a;
      font-size: 1.7rem;
      border-bottom: 3px solid #0ea5e9;
      padding-bottom: 0.4rem;
      margin: 0 0 0.4rem;
    }}
    h2 {{
      color: #0f172a;
      font-size: 1.18rem;
      margin: 1.6rem 0 0.4rem;
      border-bottom: 1px solid #e2e8f0;
      padding-bottom: 0.2rem;
      page-break-after: avoid;
    }}
    h3 {{ color: #0f172a; font-size: 1.02rem; margin: 1.1rem 0 0.3rem; }}
    p {{ margin: 0.45rem 0; }}
    em {{ color: #475569; }}
    strong {{ color: #0f172a; }}
    code {{
      background: #f1f5f9;
      padding: 0.08rem 0.35rem;
      border-radius: 4px;
      font-size: 92%;
      font-family: "SFMono-Regular", Menlo, Consolas, monospace;
    }}
    table {{
      border-collapse: collapse;
      margin: 0.6rem 0 1rem;
      width: 100%;
      page-break-inside: auto;
    }}
    tr {{ page-break-inside: avoid; page-break-after: auto; }}
    th, td {{
      border: 1px solid #cbd5e1;
      padding: 0.36rem 0.65rem;
      text-align: left;
      vertical-align: top;
    }}
    th {{ background: #f1f5f9; font-weight: 600; color: #0f172a; }}
    tr:nth-child(even) td {{ background: #fafbfc; }}
    td[align="right"], th[align="right"] {{ text-align: right; font-variant-numeric: tabular-nums; }}
    td[align="center"], th[align="center"] {{ text-align: center; }}
    blockquote {{
      border-left: 4px solid #94a3b8;
      padding-left: 1rem;
      color: #475569;
      margin: 0.6rem 0 0.6rem 0;
    }}
    hr {{ border: 0; border-top: 1px solid #e2e8f0; margin: 1.6rem 0; }}
    ul, ol {{ padding-left: 1.4rem; }}
    li {{ margin: 0.18rem 0; }}
    .meta {{
      color: #64748b;
      font-size: 0.85em;
      margin: 0.2rem 0 1.4rem;
    }}
    .meta code {{ background: transparent; padding: 0; }}
  </style>
</head>
<body>
  <p class="meta">Generated {generated} from <code>tax_optimizer</code> CLI</p>
{body}
</body>
</html>
"""


_PDF_INSTALL_HINT = (
    "PDF output requires WeasyPrint and its system libraries.\n"
    "  pip install 'tax-optimizer[pdf]'\n"
    "  macOS:          brew install pango\n"
    "  Debian/Ubuntu:  sudo apt install libpango-1.0-0 libpangoft2-1.0-0\n"
    "Or write to an HTML file instead (any path ending in .html)."
)


# ---------------------------------------------------------------------------
# Terminal
# ---------------------------------------------------------------------------


def render_terminal(md_text: str, *, force_pretty: bool | None = None) -> None:
    """Print the report to stdout.

    When stdout is a TTY (and `force_pretty` is not False), render with
    Rich for color, headings, and table styling. When piped/redirected,
    emit the raw markdown unchanged so it can be fed to `glow`, `pandoc`,
    `mdcat`, etc., or saved by shell redirection.
    """
    pretty = sys.stdout.isatty() if force_pretty is None else force_pretty
    if not pretty:
        sys.stdout.write(md_text)
        if not md_text.endswith("\n"):
            sys.stdout.write("\n")
        return

    try:
        from rich.console import Console
        from rich.markdown import Markdown
    except ImportError:
        sys.stdout.write(md_text)
        sys.stdout.write("\n")
        return

    Console().print(Markdown(md_text, hyperlinks=True))


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


def render_html(
    md_text: str,
    *,
    title: str = "Retirement Tax Optimization — Action Plan",
) -> str:
    """Convert the report markdown to a standalone, styled HTML document."""
    import markdown as _md

    body = _md.markdown(
        md_text,
        extensions=["tables", "fenced_code", "smarty", "sane_lists"],
        output_format="html",
    )
    return _HTML_TEMPLATE.format(
        title=_html.escape(title),
        generated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        body=body,
    )


def write_html(md_text: str, path: Path, *, title: str | None = None) -> None:
    kwargs = {"title": title} if title else {}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(md_text, **kwargs), encoding="utf-8")


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------


def write_pdf(md_text: str, path: Path, *, title: str | None = None) -> None:
    # WeasyPrint loads native pango/cairo/gobject libraries during import
    # via ctypes/cffi, so a missing system lib surfaces as OSError at the
    # `import` statement itself, not as ImportError. Catch both.
    try:
        from weasyprint import HTML  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(_PDF_INSTALL_HINT) from e
    except OSError as e:
        raise RuntimeError(f"{_PDF_INSTALL_HINT}\n(WeasyPrint: {e})") from e

    kwargs = {"title": title} if title else {}
    html_text = render_html(md_text, **kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        HTML(string=html_text).write_pdf(str(path))
    except OSError as e:
        raise RuntimeError(f"{_PDF_INSTALL_HINT}\n(WeasyPrint: {e})") from e


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def detect_format(path: Path) -> OutputFormat:
    suffix = path.suffix.lower()
    if suffix in (".html", ".htm"):
        return "html"
    if suffix == ".pdf":
        return "pdf"
    raise ValueError(
        f"--report path must end in .html or .pdf (got '{path.suffix or path.name}'). "
        "Markdown file output is no longer supported by the CLI -- use the "
        "`build_action_report()` Python API to get raw markdown."
    )


def write_report(
    md_text: str,
    path: Path,
    fmt: OutputFormat | None = None,
    *,
    title: str | None = None,
) -> OutputFormat:
    fmt = fmt or detect_format(path)
    if fmt == "html":
        write_html(md_text, path, title=title)
    elif fmt == "pdf":
        write_pdf(md_text, path, title=title)
    else:  # pragma: no cover - guarded by detect_format / Literal
        raise ValueError(f"unknown report format {fmt!r}")
    return fmt
