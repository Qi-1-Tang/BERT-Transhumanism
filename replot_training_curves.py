"""
Re-plot training curves (loss, F1, precision/recall, class balance)
from an existing metrics JSON, without re-running training.

Usage example:
  python replot_training_curves.py \
      --metrics training_output_stage2_classweights/reports/metrics_stage2_classweights.json \
      --output training_output_stage2_classweights/reports/training_curves_stage2_classweights.png
(非必要运行代码)
"""

import argparse
import json
import os

import matplotlib.pyplot as plt


def plot_training_curves_from_metrics(metrics_path: str, output_path: str):
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    epoch_metrics = metrics.get("epoch_metrics")
    if not epoch_metrics:
        raise ValueError(f"'epoch_metrics' not found in {metrics_path}")

    epochs = [m["epoch"] for m in epoch_metrics]
    train_losses = [m["train_loss"] for m in epoch_metrics]
    val_losses = [m["val_loss"] for m in epoch_metrics]
    f1_macros = [m["val_f1_macro"] for m in epoch_metrics]
    f1_class_0_list = [m["f1_class_0"] for m in epoch_metrics]
    f1_class_1_list = [m["f1_class_1"] for m in epoch_metrics]
    precisions = [m["val_precision"] for m in epoch_metrics]
    recalls = [m["val_recall"] for m in epoch_metrics]
    f1_diff = [abs(f1_class_0_list[i] - f1_class_1_list[i]) for i in range(len(f1_class_0_list))]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss curves
    axes[0, 0].plot(epochs, train_losses, label="Train loss", marker="o")
    axes[0, 0].plot(epochs, val_losses, label="Val loss", marker="s")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training & validation loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # F1 curves
    axes[0, 1].plot(epochs, f1_macros, label="Macro F1", marker="o")
    axes[0, 1].plot(epochs, f1_class_0_list, label="Class 0 F1", marker="s")
    axes[0, 1].plot(epochs, f1_class_1_list, label="Class 1 F1", marker="^")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("F1-score")
    axes[0, 1].set_title("F1-score over epochs")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Precision & Recall curves
    axes[1, 0].plot(epochs, precisions, label="Precision", marker="o")
    axes[1, 0].plot(epochs, recalls, label="Recall", marker="s")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].set_title("Precision & Recall")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # F1 difference (class balance)
    axes[1, 1].plot(epochs, f1_diff, label="F1 gap", marker="o", color="red")
    axes[1, 1].axhline(y=0.1, color="green", linestyle="--", label="Target threshold (0.1)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("|F1_class0 - F1_class1|")
    axes[1, 1].set_title("Per-class F1 gap (smaller is better)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Training curves re-plotted and saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Re-plot training curves from metrics JSON")
    parser.add_argument(
        "--metrics",
        type=str,
        required=True,
        help="Path to metrics_*.json produced by training",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output PNG path; default: training_curves_<suffix>.png next to metrics",
    )
    args = parser.parse_args()

    metrics_path = args.metrics
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    if args.output:
        output_path = args.output
    else:
        base_dir = os.path.dirname(metrics_path)
        base_name = os.path.basename(metrics_path)
        suffix = base_name.replace("metrics_", "").replace(".json", "")
        output_path = os.path.join(base_dir, f"training_curves_{suffix}.png")

    plot_training_curves_from_metrics(metrics_path, output_path)


if __name__ == "__main__":
    main()



