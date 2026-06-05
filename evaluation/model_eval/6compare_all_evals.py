import argparse
import json
import subprocess
import sys
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decomposers.ML_methods.NN_based.experiment_catalog import evaluation_specs

MODELS = [
    {"name": spec.key, "eval_script": spec.eval_script, "report": spec.report}
    for spec in evaluation_specs()
]


def maybe_run_eval(script_rel: str) -> bool:
    """Maybe run eval.
    
    Args:
        script_rel: Evaluation script path relative to the project root.
    """
    script = PROJECT_ROOT / script_rel
    result = subprocess.run([sys.executable, str(script)], cwd=PROJECT_ROOT)
    return result.returncode == 0


def load_report(report_rel: str):
    """Load report.
    
    Args:
        report_rel: Report path relative to the project root.
    """
    p = PROJECT_ROOT / report_rel
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def format_table(rows: list[dict]) -> str:
    """Format table.
    
    Args:
        rows: Rows to format or export.
    """
    headers = [
        "model",
        "test_loss",
        "macro_corr",
        "macro_snr_db",
        "clean_sum_mse",
        "observed_mixture_mse",
        "training_time_sec",
        "total_samples",
    ]
    line = "| " + " | ".join(headers) + " |"
    sep = "|" + "|".join(["---"] * len(headers)) + "|"
    out = [line, sep]
    for row in rows:
        out.append(
            "| {model} | {test_loss:.6f} | {macro_corr:.6f} | {macro_snr_db:.6f} | {clean_sum_mse:.6f} | {observed_mixture_mse:.6f} | {training_time_sec:.2f} | {total_samples:.0f} |".format(**row)
        )
    return "\n".join(out)


def format_model_label(label: str, *, max_len: int = 44, wrap_width: int = 20) -> str:
    """Format model label.
    
    Args:
        label: Model or axis label.
        max_len: Maximum label length before truncation.
        wrap_width: Line width for wrapped labels.
    """
    text = str(label).strip().replace("_", " ")
    if len(text) > max_len:
        text = text[: max_len - 3].rstrip() + "..."
    return fill(text, width=wrap_width)


def save_bar_comparison(names: list[str], values: list[float], ylabel: str, title: str, output_path: Path) -> None:
    """Save bar comparison.
    
    Args:
        title: Chart title.
        output_path: File-system location.
    """
    labels = [format_model_label(name) for name in names]
    positions = list(range(len(labels)))
    fig_height = max(4.0, 0.45 * len(labels) + 1.4)
    fig, ax = plt.subplots(figsize=(9, fig_height))
    ax.barh(positions, values)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description="Run all model eval scripts and build a comparison table + plots.")
    parser.add_argument("--skip-run", action="store_true", help="Skip executing eval scripts and only read report files.")
    parser.add_argument("--output-dir", default="evaluation/model_eval/artifacts")
    args = parser.parse_args()

    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_status = {}
    if not args.skip_run:
        for spec in MODELS:
            ok = maybe_run_eval(spec["eval_script"])
            eval_status[spec["name"]] = ok
            print(f"[{spec['name']}] eval {'OK' if ok else 'FAILED'}")

    rows = []
    missing_reports = []
    for spec in MODELS:
        report = load_report(spec["report"])
        if report is None:
            missing_reports.append({"model": spec["name"], "report": spec["report"]})
            continue
        summary = report.get("test_summary", {})
        macro = summary.get("macro_average", {})
        clean_sum = summary.get("clean_sum_metrics") or summary.get("mixture_metrics", {})
        observed_mixture = summary.get("observed_mixture_metrics", {})
        metadata = report.get("training_metadata", {})
        rows.append(
            {
                "model": spec["name"],
                "test_loss": float(report.get("test_loss", float("nan"))),
                "macro_corr": float(macro.get("corr", float("nan"))),
                "macro_snr_db": float(macro.get("snr_db", float("nan"))),
                "clean_sum_mse": float(clean_sum.get("mse", float("nan"))),
                "observed_mixture_mse": float(observed_mixture.get("mse", float("nan"))),
                "training_time_sec": float(metadata.get("elapsed_seconds", float("nan"))) if isinstance(metadata, dict) else float("nan"),
                "total_samples": float(metadata.get("total_samples", float("nan"))) if isinstance(metadata, dict) else float("nan"),
            }
        )

    if not rows:
        print("No final_test_report.json files found. Train models first.")
        return

    rows.sort(key=lambda r: r["test_loss"])
    table_md = format_table(rows)
    print("\n" + table_md)

    if missing_reports:
        print("\nMissing reports (model omitted from table):")
        for item in missing_reports:
            status = eval_status.get(item["model"])
            if status is True:
                suffix = " (eval script ran OK; report file still missing)"
            elif status is False:
                suffix = " (eval script failed)"
            else:
                suffix = ""
            print(f"- {item['model']}: {item['report']}{suffix}")

    (output_dir / "model_comparison_table.md").write_text(table_md + "\n", encoding="utf-8")

    names = [r["model"] for r in rows]
    corr = [r["macro_corr"] for r in rows]
    snr = [r["macro_snr_db"] for r in rows]
    loss = [r["test_loss"] for r in rows]

    save_bar_comparison(names, corr, "Macro corr", "Model comparison: macro correlation", output_dir / "macro_corr_comparison.png")
    save_bar_comparison(names, snr, "Macro SNR (dB)", "Model comparison: macro SNR", output_dir / "macro_snr_comparison.png")
    save_bar_comparison(names, loss, "Test MSE loss", "Model comparison: test loss", output_dir / "test_loss_comparison.png")

    if eval_status:
        (output_dir / "eval_status.json").write_text(json.dumps(eval_status, indent=2), encoding="utf-8")
    if missing_reports:
        (output_dir / "missing_reports.json").write_text(json.dumps(missing_reports, indent=2), encoding="utf-8")

    print(f"\nSaved artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
