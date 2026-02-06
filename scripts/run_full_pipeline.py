"""
Run the full Fresh Flow pipeline from the command line.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
PIPELINE_DIR = ROOT_DIR / "src" / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.append(str(PIPELINE_DIR))

from pipeline_runner import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Fresh Flow pipeline")
    parser.add_argument("--data-dir", default="data/Inventory_Management", help="Directory containing CSVs")
    parser.add_argument("--output-dir", default="reports", help="Output directory for reports")
    parser.add_argument("--forecast-horizon", default="daily", help="Forecast horizon")
    parser.add_argument("--prefer-advanced", action="store_true", help="Try advanced forecaster if available")
    parser.add_argument("--prep-days", type=int, default=1, help="Prep horizon in days")

    args = parser.parse_args()
    run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        forecast_horizon=args.forecast_horizon,
        prefer_advanced=args.prefer_advanced,
        prep_horizon_days=args.prep_days,
    )


if __name__ == "__main__":
    main()
