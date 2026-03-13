"""
scripts/run_real_backtests.py
==============================
Standalone backtest runner.  For the full pipeline (backtest + submission),
use run_pipeline_2026.py instead.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.backtest import rolling_backtest, save_backtest_outputs
from src.config import CONFIG


def main():
    print("Starting rolling backtest...")
    results = rolling_backtest(
        seasons=CONFIG["backtest_seasons"],
        cfg=CONFIG,
        genders=("M", "W"),
        calibrate=True,
        calibrator_methods=CONFIG["calibration_methods"],
    )

    save_backtest_outputs(results, "backtest_results")

    print("\n===== SUMMARY =====")
    print(results["summary"].to_string(index=False))

    print("\n===== CALIBRATION TABLE =====")
    print(results["calibration_table"].to_string(index=False))

    print("\n===== PROBABILITY BANDS =====")
    print(results["probability_bands"].to_string(index=False))


if __name__ == "__main__":
    main()
