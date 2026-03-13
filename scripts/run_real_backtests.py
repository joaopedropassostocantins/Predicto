from src.backtest import rolling_backtest, save_backtest_outputs
from src.config import CONFIG


def main():
    results = rolling_backtest(
        seasons=CONFIG["backtest_seasons"],
        cfg=CONFIG,
        genders=("M", "W"),
        calibrate=True,
        calibrator_methods=CONFIG["calibration_methods"],
    )

    save_backtest_outputs(results, "/kaggle/working")

    print("\n===== SUMMARY =====")
    print(results["summary"].to_string(index=False))

    print("\n===== CALIBRATION TABLE =====")
    print(results["calibration_table"].to_string(index=False))

    print("\n===== PROBABILITY BANDS =====")
    print(results["probability_bands"].to_string(index=False))


if __name__ == "__main__":
    main()
