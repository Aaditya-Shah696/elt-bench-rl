"""where ELT-Bench lives on disk.

it stays outside this repo (17 GB of data), right next to it by default. set ELT_BENCH_DIR
to point somewhere else
"""

import os
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
ELT_BENCH_DIR = Path(os.environ.get("ELT_BENCH_DIR", REPO_DIR.parent / "ELT-Bench")).resolve()

SETUP_DIR = ELT_BENCH_DIR / "setup"
SOURCE_DATA_DIR = SETUP_DIR / "data"
SCHEMAS_DIR = ELT_BENCH_DIR / "elt-bench" / "schemas"
EVAL_DIR = ELT_BENCH_DIR / "evaluation"
GROUND_TRUTH_DIR = ELT_BENCH_DIR / "ground_truth"


def task_dir(task: str, destination: str = "snowflake") -> Path:
    """folder with one task's config.yaml and data_model.yaml"""
    return ELT_BENCH_DIR / "elt-bench" / destination / task
