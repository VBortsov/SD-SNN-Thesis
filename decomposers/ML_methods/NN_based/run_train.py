from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decomposers.ML_methods.NN_based.experiment_catalog import get_experiment_spec, training_specs
from decomposers.ML_methods.NN_based.training_common import run_training_pipeline


def _load_train_module(script: Path, model_key: str):
    module_name = f"_nn_train_{model_key}"
    spec = importlib.util.spec_from_file_location(module_name, script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import train script: {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _checkpoint_stem(default_checkpoint: str, fallback: str) -> str:
    name = Path(default_checkpoint).name
    if name.endswith("_best.pt"):
        return name[: -len("_best.pt")]
    if name.endswith(".pt"):
        return name[:-3]
    return fallback


def main() -> int:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description="Unified model training launcher.")
    parser.add_argument("model", help="Model key to train.")
    args, forwarded_args = parser.parse_known_args()

    spec = get_experiment_spec(args.model)
    if not spec:
        available = ", ".join(sorted(item.key for item in training_specs()))
        print(f"Unknown model '{args.model}'. Available: {available}", file=sys.stderr)
        return 2
    if not spec.train_script:
        print(f"Model '{args.model}' has no train script configured.", file=sys.stderr)
        return 2

    script = PROJECT_ROOT / spec.train_script
    if not script.exists():
        print(f"Train script not found: {script}", file=sys.stderr)
        return 2

    train_module = _load_train_module(script, args.model)
    sys.argv = [str(script), *forwarded_args]
    cfg = train_module.parse_args()
    run_training_pipeline(
        cfg=cfg,
        project_root=PROJECT_ROOT,
        model_name=_checkpoint_stem(spec.default_checkpoint, spec.key),
        create_model_fn=train_module.create_model,
        component_names=getattr(train_module, "COMPONENT_NAMES", None),
        include_sample_metrics=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
