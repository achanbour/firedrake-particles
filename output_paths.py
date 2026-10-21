"""Shared locations for generated experiment output."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"


def experiment_results_dir(script_file):
    """Return (and create) the results directory for an experiment script."""
    experiment_name = Path(script_file).stem.removeprefix("test_")
    result_dir = RESULTS_DIR / experiment_name
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir
