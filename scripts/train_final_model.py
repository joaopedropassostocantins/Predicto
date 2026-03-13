"""
scripts/train_final_model.py
============================
Train the final model and generate a submission.
This script is superseded by run_pipeline_2026.py but kept for compatibility.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import CONFIG
from src.submit import generate_submission


def main():
    submission = generate_submission(CONFIG, output_path="final_submission.csv")
    print(f"Done. {len(submission)} rows written to final_submission.csv")


if __name__ == "__main__":
    main()
