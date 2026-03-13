"""
scripts/generate_2024_submission.py
====================================
Generate a submission for the 2024 season (kept for reference).
For the current competition (2026), use run_pipeline_2026.py instead.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import CONFIG
from src.submit import generate_submission


def main():
    cfg = dict(CONFIG)
    cfg["target_season"] = 2024

    submission = generate_submission(cfg, output_path="submission_2024.csv")
    print(f"Done. {len(submission)} rows written to submission_2024.csv")


if __name__ == "__main__":
    main()
