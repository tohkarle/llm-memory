"""Local-first memory system for LLM agents.

The filesystem under ``data/`` is authoritative; SQLite is a rebuildable
index. See :class:`memory.api.MemoryAPI` for the public API, or the ``mem``
command line.
"""

from __future__ import annotations

__version__ = "0.1.2"

__all__ = ["__version__"]
