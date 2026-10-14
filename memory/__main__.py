"""Entry point for ``python -m memory``.

Equivalent to the ``mem`` console script, and preferable in environments
where an editable install's path hook is not honored.
"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    main()
