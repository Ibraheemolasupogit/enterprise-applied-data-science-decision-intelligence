"""Make the package importable without depending on editable-install path injection.

Hatchling's editable install writes a `.pth` file that some platforms (observed:
macOS) mark with the OS "hidden" flag. CPython's site.py intentionally skips
hidden `.pth` files, which can silently stop `pip install -e .` from putting
`src/` on `sys.path` — non-deterministically, depending on the install
environment. Inserting `src/` directly here keeps test collection correct
regardless of that interaction.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
