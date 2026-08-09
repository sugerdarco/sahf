"""Make the repo root importable so `pytest` works on a fresh clone.

Installing the package is not required — a clone plus the dependencies in
requirements.txt is enough to run the suite.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
