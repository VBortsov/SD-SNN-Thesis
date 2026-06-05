from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from app.services.paths import REPO_ROOT
from decomposers.ML_methods.NN_based.experiment_catalog import get_experiment_spec

UNIFIED_TRAIN_RUNNER = REPO_ROOT / "decomposers" / "ML_methods" / "NN_based" / "run_train.py"


@dataclass
class TrainingRequest:
    """Request data for a workflow."""
    model_name: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    seed: int
    fs: int
    duration: float
    train_samples: int
    val_samples: int
    test_samples: int
    noise_level: float
    device: str
    output_dir: str
    component_types: list[str] | None = None
    component_loss_weight: float = 1.0
    reconstruction_loss_weight: float = 0.5
    spectral_loss_weight: float = 0.05
    experiment_name: str = ""
    supports_depth_expansion: bool = False
    extra_conv_layers: int = 0
    extra_conv_kernel_size: int = 3
    extra_conv_channels: int = 0
    extra_conv_dilation: int = 1
    extra_conv_activation: str = "gelu"
    extra_conv_norm: str = "groupnorm"
    extra_conv_dropout: float = 0.1
    extra_conv_num_groups: int = 8
    extra_conv_residual: bool = False


@dataclass
class TrainingJob:
    """Runtime state for a background job."""
    model: dict
    request: TrainingRequest
    process: subprocess.Popen
    command: list[str]
    output_queue: queue.Queue
    started_at: float


def training_script_path(model_name: str) -> Path | None:
    """Return the training script path.
    
    Args:
        model_name: Name used for lookup or display.
    """
    spec = get_experiment_spec(model_name)
    if not spec or not spec.train_script:
        return None
    return REPO_ROOT / spec.train_script


def build_command(req: TrainingRequest) -> list[str]:
    """Build command.
    
    Args:
        req: Training request settings.
    """
    signal_length = max(64, int(req.duration * req.fs))
    component_count = len(req.component_types or ["harmonic", "amfm", "chirp"])
    cmd = [
        sys.executable,
        "-u",
        str(UNIFIED_TRAIN_RUNNER),
        req.model_name,
        "--epochs",
        str(req.epochs),
        "--batch-size",
        str(req.batch_size),
        "--learning-rate",
        str(req.learning_rate),
        "--weight-decay",
        str(req.weight_decay),
        "--component-loss-weight",
        str(req.component_loss_weight),
        "--reconstruction-loss-weight",
        str(req.reconstruction_loss_weight),
        "--spectral-loss-weight",
        str(req.spectral_loss_weight),
        "--seed",
        str(req.seed),
        "--fs",
        str(req.fs),
        "--signal-length",
        str(signal_length),
        "--train-samples",
        str(req.train_samples),
        "--val-samples",
        str(req.val_samples),
        "--test-samples",
        str(req.test_samples),
        "--noise-min",
        str(req.noise_level),
        "--noise-max",
        str(req.noise_level),
        "--out-channels",
        str(component_count),
        "--output-dir",
        req.output_dir,
    ]
    if req.experiment_name:
        cmd.extend(["--experiment-name", req.experiment_name])
    if req.supports_depth_expansion:
        cmd.extend(
            [
                "--extra-conv-layers",
                str(req.extra_conv_layers),
                "--extra-conv-kernel-size",
                str(req.extra_conv_kernel_size),
                "--extra-conv-channels",
                str(req.extra_conv_channels),
                "--extra-conv-dilation",
                str(req.extra_conv_dilation),
                "--extra-conv-activation",
                req.extra_conv_activation,
                "--extra-conv-norm",
                req.extra_conv_norm,
                "--extra-conv-dropout",
                str(req.extra_conv_dropout),
                "--extra-conv-num-groups",
                str(req.extra_conv_num_groups),
            ]
        )
        if req.extra_conv_residual:
            cmd.append("--extra-conv-residual")
    return cmd


def expected_checkpoint_path(model_name: str, output_dir: str) -> str:
    """Return the expected checkpoint path.
    
    Args:
        output_dir: Directory where outputs are written.
        model_name: Name used for lookup or display.
    """
    spec = get_experiment_spec(model_name)
    if spec and spec.default_checkpoint:
        name = Path(spec.default_checkpoint).name
    else:
        name = f"{model_name}_best.pt"
    return f"{output_dir}/{name}"


EPOCH_RE = re.compile(
    r"epoch\s+(?P<epoch>\d+)\s+\|\s+train_loss=(?P<train>[0-9.eE+-]+)\s+\|\s+val_loss=(?P<val>[0-9.eE+-]+)"
)
BATCH_RE = re.compile(r"batch\s+(?P<batch>\d+)\s+\|\s+loss=(?P<loss>[0-9.eE+-]+)")
VAL_CORR_RE = re.compile(r"val_corr=(?P<val_corr>[0-9.eE+-]+)")
VAL_SNR_RE = re.compile(r"val_snr_db=(?P<val_snr_db>[0-9.eE+-]+)")
MIXTURE_MSE_RE = re.compile(r"mixture_mse=(?P<mixture_mse>[0-9.eE+-]+)")


def run_training(req: TrainingRequest):
    """Run training.
    
    Args:
        req: Training request settings.
    """
    script = training_script_path(req.model_name)
    if script is None or not script.exists():
        raise FileNotFoundError(f"No training script mapped for model '{req.model_name}'.")

    cmd = build_command(req)
    env = None
    if req.device.lower() == "cpu":
        env = dict(**os.environ, CUDA_VISIBLE_DEVICES="")
    if env is None:
        env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    if req.component_types:
        env = dict(os.environ if env is None else env)
        env["NN_COMPONENT_TYPES"] = ",".join(req.component_types)

    process = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    return process, cmd


def _enqueue_output(process: subprocess.Popen, output_queue: queue.Queue) -> None:
    if process.stdout is None:
        return
    for line in process.stdout:
        output_queue.put(line)


def start_training_job(model: dict, req: TrainingRequest) -> TrainingJob:
    """Start training job.
    
    Args:
        model: Model instance to run or inspect.
        req: Training request settings.
    """
    process, cmd = run_training(req)
    output_queue: queue.Queue = queue.Queue()
    thread = threading.Thread(target=_enqueue_output, args=(process, output_queue), daemon=True)
    thread.start()
    return TrainingJob(
        model=model,
        request=req,
        process=process,
        command=cmd,
        output_queue=output_queue,
        started_at=time.monotonic(),
    )


def drain_training_output(job: TrainingJob) -> list[str]:
    """Drain training output.
    
    Args:
        job: Project value for this call.
    """
    lines: list[str] = []
    while True:
        try:
            lines.append(job.output_queue.get_nowait())
        except queue.Empty:
            return lines


def elapsed_seconds(job: TrainingJob) -> float:
    """Elapsed seconds.
    
    Args:
        job: Project value for this call.
    """
    return time.monotonic() - job.started_at


def parse_progress_line(line: str) -> dict | None:
    """Parse progress line.
    
    Args:
        line: Log line to parse.
    """
    lowered = line.lower()
    match = EPOCH_RE.search(lowered)
    if not match:
        return None
    parsed = {
        "epoch": int(match.group("epoch")),
        "train_loss": float(match.group("train")),
        "val_loss": float(match.group("val")),
    }
    for regex, key in [
        (VAL_CORR_RE, "val_corr"),
        (VAL_SNR_RE, "val_snr_db"),
        (MIXTURE_MSE_RE, "mixture_mse"),
    ]:
        metric_match = regex.search(lowered)
        if metric_match:
            parsed[key] = float(metric_match.group(key))
    return parsed


def parse_batch_line(line: str) -> dict | None:
    """Parse batch line.
    
    Args:
        line: Log line to parse.
    """
    match = BATCH_RE.search(line.lower())
    if not match:
        return None
    return {
        "batch": int(match.group("batch")),
        "loss": float(match.group("loss")),
    }


def read_final_report(output_dir: str) -> dict:
    """Read final report.
    
    Args:
        output_dir: Directory where outputs are written.
    """
    path = REPO_ROOT / output_dir / "final_test_report.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
