import itertools
from typing import Dict, List, Tuple

import numpy as np

from .metrics import evaluate_signal, mse


class DecompositionEvaluationError(ValueError):
    """Raised when decomposition metrics cannot be computed cleanly."""
    pass



def _to_components(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise DecompositionEvaluationError(
            "Expected component array with shape (n_components, signal_length)."
        )
    return x



def find_best_permutation(y_true: np.ndarray, y_pred: np.ndarray, metric: str = "mse") -> Tuple[Tuple[int, ...], np.ndarray]:
    """Find best permutation.
    
    Args:
        y_true: Target component signals.
        y_pred: Predicted component signals.
        metric: Metric column or metric name to use.
    """
    y_true = _to_components(y_true)
    y_pred = _to_components(y_pred)

    if y_true.shape != y_pred.shape:
        raise DecompositionEvaluationError("y_true and y_pred must have the same shape.")

    n_components = y_true.shape[0]
    if metric != "mse":
        raise DecompositionEvaluationError("Only 'mse' is currently supported for assignment.")

    cost = np.zeros((n_components, n_components), dtype=float)
    for i in range(n_components):
        for j in range(n_components):
            cost[i, j] = mse(y_true[i], y_pred[j])

    best_perm = None
    best_cost = float("inf")
    for perm in itertools.permutations(range(n_components)):
        total = sum(cost[i, perm[i]] for i in range(n_components))
        if total < best_cost:
            best_cost = total
            best_perm = perm

    return best_perm, cost



def reorder_prediction(y_pred: np.ndarray, permutation: Tuple[int, ...]) -> np.ndarray:
    """Reorder prediction.
    
    Args:
        y_pred: Predicted component signals.
        permutation: Component-order mapping.
    """
    y_pred = _to_components(y_pred)
    return y_pred[list(permutation)]



def evaluate_decomposition(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    observed_mixture: np.ndarray = None,
    permutation_invariant: bool = True,
    assignment_metric: str = "mse",
) -> Dict[str, object]:
    """Evaluate decomposition.
    
    Args:
        y_true: Target component signals.
        y_pred: Predicted component signals.
    """
    y_true = _to_components(y_true)
    y_pred = _to_components(y_pred)

    if y_true.shape != y_pred.shape:
        raise DecompositionEvaluationError("y_true and y_pred must have the same shape.")

    if permutation_invariant:
        permutation, cost_matrix = find_best_permutation(y_true, y_pred, metric=assignment_metric)
        aligned_pred = reorder_prediction(y_pred, permutation)
    else:
        permutation = tuple(range(y_true.shape[0]))
        cost_matrix = None
        aligned_pred = y_pred

    component_reports: List[Dict[str, float]] = []
    for true_comp, pred_comp in zip(y_true, aligned_pred):
        component_reports.append(evaluate_signal(true_comp, pred_comp))

    metric_names = component_reports[0].keys() if component_reports else []
    macro_average = {
        metric: float(np.mean([report[metric] for report in component_reports]))
        for metric in metric_names
    }
    std_average = {
        metric: float(np.std([report[metric] for report in component_reports]))
        for metric in metric_names
    }

    # Component-wise metrics (and caller-side test_loss) evaluate source separation quality.
    clean_sum_true = np.sum(y_true, axis=0)
    clean_sum_pred = np.sum(aligned_pred, axis=0)
    clean_sum_report = evaluate_signal(clean_sum_true, clean_sum_pred)

    # Observed mixture metrics evaluate reconstruction of the actual model input mixture.
    observed_mixture_report = None
    if observed_mixture is not None:
        observed_mixture = np.asarray(observed_mixture, dtype=float).reshape(-1)
        if observed_mixture.shape != clean_sum_pred.shape:
            raise DecompositionEvaluationError(
                "observed_mixture must have shape (signal_length,) matching summed components."
            )
        observed_mixture_report = evaluate_signal(observed_mixture, clean_sum_pred)

    return {
        "permutation": permutation,
        "cost_matrix": cost_matrix,
        "component_metrics": component_reports,
        "macro_average": macro_average,
        "std_average": std_average,
        "clean_sum_metrics": clean_sum_report,
        "observed_mixture_metrics": observed_mixture_report,
        # Backward compatibility shim for older call sites/reports.
        "mixture_metrics": clean_sum_report,
    }



def format_report(report: Dict[str, object], component_names=None) -> str:
    """Format report.
    
    Args:
        report: Metric report dictionary.
        component_names: Names for predicted or target components.
    """
    component_metrics = report["component_metrics"]
    permutation = report["permutation"]
    macro = report["macro_average"]
    clean_sum = report["clean_sum_metrics"]
    observed_mixture = report.get("observed_mixture_metrics")

    lines = []
    lines.append("Decomposition evaluation")
    lines.append("=" * 26)
    lines.append(f"Best permutation: {permutation}")
    lines.append("")

    if component_names is None:
        component_names = [f"component_{i}" for i in range(len(component_metrics))]

    for name, metrics in zip(component_names, component_metrics):
        lines.append(f"[{name}]")
        for key, value in metrics.items():
            lines.append(f"  {key}: {value:.6f}")
        lines.append("")

    lines.append("[macro_average]")
    for key, value in macro.items():
        lines.append(f"  {key}: {value:.6f}")
    lines.append("")

    lines.append("[clean_sum_metrics]")
    for key, value in clean_sum.items():
        lines.append(f"  {key}: {value:.6f}")

    if observed_mixture is not None:
        lines.append("")
        lines.append("[observed_mixture_metrics]")
        for key, value in observed_mixture.items():
            lines.append(f"  {key}: {value:.6f}")

    return "\n".join(lines)
