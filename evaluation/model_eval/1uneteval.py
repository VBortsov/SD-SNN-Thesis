import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    """Run the command-line entry point."""
    from decomposers.ML_methods.NN_based.experiment_catalog import get_experiment_spec
    from evaluation.model_eval.eval_common import evaluate_experiment

    evaluate_experiment(get_experiment_spec("unet1d"), PROJECT_ROOT)


if __name__ == "__main__":
    main()
