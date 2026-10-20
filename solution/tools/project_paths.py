"""Shared paths for the repository and the separately supplied challenge data."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOLUTION_ROOT = REPO_ROOT / "solution"
BUILD = SOLUTION_ROOT / "build"
DATA = Path(os.environ.get(
    "EDA_DATA_DIR",
    str(REPO_ROOT.parents[1] / "EDA-PANGO-SRB-Arch-Delay" / "EDA-PANGO-SRB-Arch-Delay"),
)).resolve()
