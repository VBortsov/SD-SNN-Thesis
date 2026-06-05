from __future__ import annotations

import argparse
import sys
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decomposers.ML_methods.NN_based.experiment_catalog import evaluation_specs, get_experiment_spec
from evaluation.model_eval.eval_common import evaluate_experiment


def main() -> int:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description="Unified model evaluation launcher.")
    parser.add_argument("model", help="Model key to evaluate, or 'all'.")
    parser.add_argument("--signal-length", type=int, default=1024)
    parser.add_argument("--fs", type=int, default=256)
    parser.add_argument("--test-samples", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--disable-permutation-invariant", action="store_true")
    args = parser.parse_args()

    def run_spec(spec) -> int:
        try:
            evaluate_experiment(
                spec,
                PROJECT_ROOT,
                signal_length=args.signal_length,
                fs=args.fs,
                num_test_samples=args.test_samples,
                batch_size=args.batch_size,
                permutation_invariant=not args.disable_permutation_invariant,
            )
            return 0
        except Exception as exc:
            print(f"[{spec.key}] evaluation failed: {exc}", file=sys.stderr)
            return 1

    if args.model == "all":
        statuses = {}
        for spec in evaluation_specs():
            print(f"\n[{spec.key}] evaluating {spec.display_name}")
            statuses[spec.key] = run_spec(spec)
        failed = {key: code for key, code in statuses.items() if code != 0}
        if failed:
            print(f"\nFailed evaluations: {failed}", file=sys.stderr)
            return 1
        return 0

    spec = get_experiment_spec(args.model)
    if not spec:
        available = ", ".join(sorted(item.key for item in evaluation_specs()))
        print(f"Unknown model '{args.model}'. Available: {available}", file=sys.stderr)
        return 2
    return run_spec(spec)


if __name__ == "__main__":
    raise SystemExit(main())
