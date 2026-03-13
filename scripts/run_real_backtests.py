
import sys
import os
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.backtest import rolling_backtest, save_backtest_outputs
from src.config import CONFIG

def main():
    # Ensure data_dir is correct for the environment
    # CONFIG["data_dir"] = "data" 
    
    print("Starting rolling backtest...")
    results = rolling_backtest(
        seasons=CONFIG["backtest_seasons"],
        cfg=CONFIG,
        genders=("M", "W"),
        calibrate=True,
        calibrator_methods=CONFIG["calibration_methods"],
    )

    output_dir = "backtest_results"
    save_backtest_outputs(results, output_dir)

    print("\n===== SUMMARY =====")
    print(results["summary"].to_string(index=False))

    print("\n===== CALIBRATION TABLE =====")
    print(results["calibration_table"].to_string(index=False))

    print("\n===== PROBABILITY BANDS =====")
    print(results["probability_bands"].to_string(index=False))

if __name__ == "__main__":
    main()
