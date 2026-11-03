"""Launch the interactive Aletheia Governance & Explainability Dashboard."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "src" / "aletheia" / "dashboard" / "app.py"


def main() -> int:
    cmd = [sys.executable, "-m", "streamlit", "run", str(APP_PATH)]
    print(f"Launching Aletheia Dashboard: {' '.join(cmd)}")
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())

