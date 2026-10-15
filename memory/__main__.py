"""Entry point for ``python -m memory``.

Equivalent to the ``mem`` console script. Reaching this module means the
package already imported, so no path repair is needed here -- that concern
belongs to :mod:`mem_launcher`, which the console script uses.
"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    main()
