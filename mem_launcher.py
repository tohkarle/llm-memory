"""Console-script shim for the ``mem`` command.

An editable install normally works because the installer drops a ``.pth``
into site-packages that appends the project directory to ``sys.path``. That
is not always honored: on macOS an installer may write the file with the
``UF_HIDDEN`` filesystem flag, and CPython >= 3.12 skips hidden ``.pth``
files outright (see ``site.addpackage``). The package then imports fine from
inside the project directory, where the CWD is already on the path, but the
installed ``mem`` script fails anywhere else.

A console script imports its entry point before any package code can run, so
this shim is a top-level module rather than part of ``memory``: it repairs
the path first, then hands off. In a normal install the repair is a no-op.
"""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_importable() -> bool:
    """Put the directory containing ``memory/`` on ``sys.path`` if missing.

    Returns True if an entry was added. Safe to call repeatedly.
    """
    try:
        import memory  # noqa: F401
    except ModuleNotFoundError:
        pass
    else:
        return False

    # Look beside this module first (a regular install puts the package
    # there), then follow the editable install's own path record, which names
    # the project directory even when the .pth that would have loaded it was
    # skipped.
    here = Path(__file__).resolve().parent
    candidates = [here, here.parent]
    candidates.extend(_editable_roots(here))

    for candidate in candidates:
        if (candidate / "memory" / "__init__.py").is_file():
            entry = str(candidate)
            if entry not in sys.path:
                sys.path.insert(0, entry)
            return True
    return False


def _editable_roots(site_dir: Path) -> list[Path]:
    """Project directories named by this package's editable install records.

    Reads the ``.pth`` contents directly rather than relying on the
    interpreter to have processed them.
    """
    roots: list[Path] = []
    for pth in sorted(site_dir.glob("*.pth")):
        if "personal_ai_memory" not in pth.name and "memory" not in pth.name:
            continue
        try:
            lines = pth.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line or line.startswith(("#", "import ")):
                continue
            roots.append(Path(line))
    return roots


def _require_package(prog: str) -> None:
    """Repair sys.path, or exit with the fix if the package is truly absent."""
    if ensure_importable():
        return
    try:
        import memory  # noqa: F401
    except ModuleNotFoundError:
        sys.exit(
            f"{prog}: cannot import the 'memory' package.\n"
            "\n"
            "This happens when an editable install's .pth path hook was "
            "not honored -- on macOS, installers may write it with the "
            "UF_HIDDEN flag, which CPython >= 3.12 skips.\n"
            "\n"
            "Fix with a regular install:\n"
            "    uv pip install .\n"
            "or run it through the project environment instead:\n"
            "    uv run python -m memory --help\n"
        )


def run() -> None:
    _require_package("mem")
    from memory.cli import main

    main()


def run_mcp() -> None:
    """Entry point for the ``mem-mcp`` script: the stdio MCP server.

    Separate from ``mem`` so an MCP client config can name a command rather
    than an interpreter path. The ``mcp`` extra is optional, so a missing
    dependency is reported here as the install that fixes it -- on stderr,
    because stdout is the client's transport and must stay clean.
    """
    _require_package("mem-mcp")
    try:
        from memory.mcp_server import main as mcp_main
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.split(".")[0] == "mcp":
            sys.exit(
                "mem-mcp: the MCP server needs the 'mcp' extra.\n"
                "\n"
                "    uv pip install 'personal-ai-memory[mcp]'\n"
            )
        raise

    mcp_main()


if __name__ == "__main__":  # pragma: no cover
    run()
