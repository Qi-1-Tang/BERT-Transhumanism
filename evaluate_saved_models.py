"""
离线评估脚本：使用已训练好的 best 模型重新跑最终评估/可视化，
无需重新训练。
"""

import os
import argparse
import glob
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split

# 复用训练脚本中的工具和常量
from train_bert import (
    TextDataset,
    evaluate,
    plot_confusion_matrix,
    plot_roc_pr_curves,
    print_detailed_metrics,
    DATASET_FILE,
    MAX_LENGTH,
    BASE_OUTPUT_DIR,
    DEVICE,
    RANDOM_STATE,
)


def load_data_with_label_map(label_map):
    """按照已保存的 label_map 读取并编码数据，保持与训练时一致。"""
    if not os.path.exists(DATASET_FILE):
        raise FileNotFoundError(f"找不到数据集文件: {DATASET_FILE}")

    df = pd.read_csv(DATASET_FILE, encoding="utf-8-sig")
    if "source" not in df.columns or "text" not in df.columns:
        raise ValueError("数据集缺少必需字段: source 或 text")

    # 按照 checkpoint 中的 label_map 进行编码，确保顺序一致
    df["label"] = df["source"].map(label_map)
    if df["label"].isnull().any():
        missing = df[df["label"].isnull()]["source"].unique()
        raise ValueError(f"数据集中出现未在 label_map 中的类别: {missing}")

    texts = df["text"].values
    labels = df["label"].astype(int).values

    _, X_val, _, y_val = train_test_split(
        texts,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=RANDOM_STATE,
    )
    return X_val, y_val


def build_val_loader(tokenizer, X_val, y_val, batch_size, max_length):
    val_dataset = TextDataset(X_val, y_val, tokenizer, max_length)
    return torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )


def evaluate_single_model(model_path, reports_dir, batch_size, save_suffix=None):
    checkpoint = torch.load(model_path, map_location=DEVICE)
    cfg = checkpoint.get("config", {})

    label_map = cfg.get("label_map")
    class_names = cfg.get("class_names")
    model_name = cfg.get("model_name")
    max_length = cfg.get("max_length", MAX_LENGTH)
    if label_map is None or class_names is None or model_name is None:
        raise ValueError(f"模型缺少必要配置字段: {model_path}")

    # 准备数据
    X_val, y_val = load_data_with_label_map(label_map)
    # 使用本地模型，避免联网
    tokenizer = BertTokenizer.from_pretrained(model_name, local_files_only=True)
    model = BertForSequenceClassification.from_pretrained(
        model_name, num_labels=len(label_map), local_files_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    val_loader = build_val_loader(tokenizer, X_val, y_val, batch_size, max_length)

    val_loss, val_preds, val_labels, val_probs = evaluate(
        model, val_loader, criterion, DEVICE
    )

    # 生成文件名后缀
    suffix = save_suffix or os.path.splitext(os.path.basename(model_path))[0]
    report_path = os.path.join(reports_dir, f"classification_report_{suffix}.txt")
    cm_path = os.path.join(reports_dir, f"confusion_matrix_{suffix}.png")
    metrics_json_path = os.path.join(reports_dir, f"metrics_{suffix}.json")

    metrics_dict = print_detailed_metrics(
        val_labels, val_preds, val_probs, class_names, report_path, X_val
    )
    metrics_dict["val_loss"] = float(val_loss)
    metrics_dict["model_path"] = model_path

    plot_confusion_matrix(val_labels, val_preds, class_names, cm_path)
    plot_roc_pr_curves(val_labels, val_probs, class_names, reports_dir, suffix)

    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)

    print(f"评估完成: {model_path}")
    print(f"  报告: {report_path}")
    print(f"  混淆矩阵: {cm_path}")
    print(f"  指标JSON: {metrics_json_path}")
    return metrics_dict


def parse_args():
    parser = argparse.ArgumentParser(
        description="对已保存的 best 模型进行离线评估和可视化"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="逗号分隔的模型路径列表；留空则自动匹配 best_model_epoch_*.pt",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="验证 batch size")
    parser.add_argument("--mode", type=str, choices=["stage1", "stage2"], default="stage2",
                        help="用于推断默认输出目录（training_output_<mode>）")
    parser.add_argument("--output-dir", type=str, default="",
                        help="自定义输出目录（包含 models/ 和 reports/），留空则按 mode 使用 training_output_<mode>")
    return parser.parse_args()


def main():
    args = parse_args()

    mode_suffix = args.mode
    output_dir = args.output_dir.strip() or f"{BASE_OUTPUT_DIR}_{mode_suffix}"
    model_dir = os.path.join(output_dir, "models")
    reports_dir = os.path.join(output_dir, "reports")

    os.makedirs(reports_dir, exist_ok=True)

    if args.models:
        model_paths = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        model_paths = sorted(glob.glob(os.path.join(model_dir, "best_model_epoch_*.pt")))

    if not model_paths:
        raise FileNotFoundError(f"未找到可评估的模型，请检查 {model_dir} 或 --models 参数。")

    all_metrics = []
    for mp in model_paths:
        metrics = evaluate_single_model(mp, reports_dir, batch_size=args.batch_size)
        all_metrics.append(metrics)

    summary_path = os.path.join(reports_dir, "metrics_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"所有模型评估结果汇总: {summary_path}")


if __name__ == "__main__":
    main()


