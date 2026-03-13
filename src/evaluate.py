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

    print("\n=== SUMMARY ===")
    print(results["summary"])

    print("\n=== CALIBRATION TABLE ===")
    print(results["calibration_table"])

    print("\nArquivos salvos em /kaggle/working:")
    print("- summary.csv")
    print("- predictions.csv")
    print("- calibration_table.csv")
    print("- probability_bands.csv")


if __name__ == "__main__":
    main()
