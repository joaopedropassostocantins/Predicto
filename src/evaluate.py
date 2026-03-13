# src/evaluate.py — Kaggle notebook entry point

from __future__ import annotations

from src.backtest import rolling_backtest, save_backtest_outputs
from src.config import CONFIG


def main(output_dir: str = "/kaggle/working", cfg: dict | None = None):
    if cfg is None:
        cfg = CONFIG

    results = rolling_backtest(
        seasons=cfg["backtest_seasons"],
        cfg=cfg,
        genders=("M", "W"),
        calibrate=True,
        calibrator_methods=cfg["calibration_methods"],
    )

    save_backtest_outputs(results, output_dir)

    print("\n=== SUMMARY ===")
    print(results["summary"].to_string(index=False))

    print("\n=== CALIBRATION TABLE ===")
    print(results["calibration_table"].to_string(index=False))

    return results


if __name__ == "__main__":
    main()
