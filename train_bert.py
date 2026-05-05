import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    average_precision_score,
    accuracy_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
    precision_score,
    recall_score
)
import json
from datetime import datetime
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ================= 配置区域 =================

DATASET_FILE = "bert_training_dataset.csv"
MODEL_NAME = "bert-base-uncased-local" # 使用本地模型，避免联网下载
MAX_LENGTH = 512
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
TEST_SIZE = 0.2
RANDOM_STATE = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 输出目录
BASE_OUTPUT_DIR = "training_output"


def build_output_dirs(args):
    """构建输出目录，包含模式和关键训练选项的后缀，避免互相覆盖。"""
    suffix_parts = [args.mode]
    if args.downsample:
        suffix_parts.append("downsample")
    if args.use_class_weights:
        suffix_parts.append("classweights")
    if args.use_focal_loss:
        suffix_parts.append("focal")
    suffix = "_".join(suffix_parts)
    output_dir = f"{BASE_OUTPUT_DIR}_{suffix}"
    model_dir = os.path.join(output_dir, "models")
    reports_dir = os.path.join(output_dir, "reports")
    return suffix, output_dir, model_dir, reports_dir

# ================= 数据集类 =================

class TextDataset(Dataset):
    """BERT文本分类数据集"""
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ================= Focal Loss（可选） =================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ================= 训练函数 =================

def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        loss = criterion(logits, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        total_loss += loss.item()
        
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        predictions.extend(preds)
        true_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, predictions, true_labels

def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    prediction_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())
            prediction_probs.extend(probs)
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, predictions, true_labels, np.array(prediction_probs)

# ================= 评估和可视化 =================

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot confusion matrix (English labels to avoid font issues)."""
    cm = confusion_matrix(y_true, y_pred)
    
    # 计算百分比
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    
    # 添加百分比标注
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = ax.text(j+0.5, i+0.7, f'({cm_percent[i, j]:.1f}%)',
                          ha="center", va="center", color="red", fontsize=9)
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵已保存: {save_path}")

def plot_roc_pr_curves(y_true, y_proba, class_names, save_dir, mode_suffix):
    """Plot ROC & PR curves (English labels to avoid font issues)."""
    # ROC曲线
    fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
    roc_auc = roc_auc_score(y_true, y_proba[:, 1])
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # PR曲线
    precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
    pr_auc = average_precision_score(y_true, y_proba[:, 1])
    
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR (AUC = {pr_auc:.4f})')
    baseline = np.sum(y_true) / len(y_true)
    plt.axhline(y=baseline, color='navy', linestyle='--', label=f'Baseline ({baseline:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    curve_path = os.path.join(save_dir, f"roc_pr_curves_{mode_suffix}.png")
    plt.savefig(curve_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC/PR curves saved: {curve_path}")

def print_detailed_metrics(y_true, y_pred, y_proba, class_names, save_path, X_val=None):
    """打印详细的评估指标并保存到文件"""
    # 确保传入的数据是numpy数组，避免布尔比较得到单个标量
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_proba = np.array(y_proba)
    
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # 计算每个类别的指标
    cm = confusion_matrix(y_true, y_pred)
    
    # 计算PR-AUC和ROC-AUC
    try:
        pr_auc = average_precision_score(y_true, y_proba[:, 1])
        roc_auc = roc_auc_score(y_true, y_proba[:, 1])
    except Exception as e:
        pr_auc = None
        roc_auc = None
        print(f"警告: 无法计算AUC指标: {e}")
    
    # 计算每个类别的详细指标
    chinese_mask = (y_true == 0)
    western_mask = (y_true == 1)
    
    # 注意：precision/recall/f1 需要明确正类标签，否则默认正类为1，会导致类0指标为0
    chinese_precision = precision_score(y_true[chinese_mask], y_pred[chinese_mask], pos_label=0, zero_division=0) if chinese_mask.sum() > 0 else 0
    chinese_recall = recall_score(y_true[chinese_mask], y_pred[chinese_mask], pos_label=0, zero_division=0) if chinese_mask.sum() > 0 else 0
    chinese_f1 = f1_score(y_true[chinese_mask], y_pred[chinese_mask], pos_label=0, zero_division=0) if chinese_mask.sum() > 0 else 0
    
    western_precision = precision_score(y_true[western_mask], y_pred[western_mask], pos_label=1, zero_division=0) if western_mask.sum() > 0 else 0
    western_recall = recall_score(y_true[western_mask], y_pred[western_mask], pos_label=1, zero_division=0) if western_mask.sum() > 0 else 0
    western_f1 = f1_score(y_true[western_mask], y_pred[western_mask], pos_label=1, zero_division=0) if western_mask.sum() > 0 else 0
    
    # 计算预测置信度统计
    max_probs = np.max(y_proba, axis=1)
    avg_confidence = np.mean(max_probs)
    low_confidence_mask = max_probs < 0.7
    high_confidence_mask = max_probs >= 0.9
    
    def fmt(v):
        return f"{v:.4f}" if v is not None else "N/A"
    
    output = f"""
{'='*70}
详细评估报告
{'='*70}

分类报告 (Classification Report):
              Precision    Recall    F1-Score    Support
{class_names[0]:<20} {report[class_names[0]]['precision']:.4f}     {report[class_names[0]]['recall']:.4f}     {report[class_names[0]]['f1-score']:.4f}     {int(report[class_names[0]]['support'])}
{class_names[1]:<20} {report[class_names[1]]['precision']:.4f}     {report[class_names[1]]['recall']:.4f}     {report[class_names[1]]['f1-score']:.4f}     {int(report[class_names[1]]['support'])}
{'准确率 (Accuracy)':<20} {'':>10} {'':>10} {report['accuracy']:.4f}     {len(y_true)}

混淆矩阵 (Confusion Matrix):
真实\\预测          {class_names[0]:<20} {class_names[1]:<20}
{class_names[0]:<20} {cm[0,0]:<20} {cm[0,1]:<20}
{class_names[1]:<20} {cm[1,0]:<20} {cm[1,1]:<20}

总体指标:
- 总体准确率: {accuracy_score(y_true, y_pred):.4f}
- ROC-AUC: {fmt(roc_auc)}
- PR-AUC: {fmt(pr_auc)}
- 宏平均F1: {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}
- 加权平均F1: {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}

每个类别的详细指标:
{class_names[0]}:
  - 精确率 (Precision): {chinese_precision:.4f}
  - 召回率 (Recall): {chinese_recall:.4f}
  - F1-Score: {chinese_f1:.4f}
  - 支持样本数: {chinese_mask.sum()}

{class_names[1]}:
  - 精确率 (Precision): {western_precision:.4f}
  - 召回率 (Recall): {western_recall:.4f}
  - F1-Score: {western_f1:.4f}
  - 支持样本数: {western_mask.sum()}

关键观察:
- {class_names[1]} 被误判为 {class_names[0]} 的数量: {cm[1,0]} ({cm[1,0]/western_mask.sum()*100:.2f}%的{class_names[1]}样本)
- {class_names[0]} 被误判为 {class_names[1]} 的数量: {cm[0,1]} ({cm[0,1]/chinese_mask.sum()*100:.2f}%的{class_names[0]}样本)
- F1-Score差异: {abs(chinese_f1 - western_f1):.4f} {'(理想: < 0.1)' if abs(chinese_f1 - western_f1) < 0.1 else '(需要改进)'}

预测置信度分析:
- 平均置信度: {avg_confidence:.4f}
- 低置信度样本 (<0.7): {low_confidence_mask.sum()} ({low_confidence_mask.sum()/len(y_true)*100:.2f}%)
- 高置信度样本 (>=0.9): {high_confidence_mask.sum()} ({high_confidence_mask.sum()/len(y_true)*100:.2f}%)

{'='*70}
"""
    
    print(output)
    
    # 保存到文件
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(output)
    
    print(f"详细报告已保存: {save_path}")
    
    # 返回指标字典用于后续分析（全部转为可JSON序列化的基础类型）
    def to_float(v):
        return float(v) if v is not None else None
    
    return {
        'overall_accuracy': to_float(accuracy_score(y_true, y_pred)),
        'roc_auc': to_float(roc_auc),
        'pr_auc': to_float(pr_auc),
        'macro_f1': to_float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'weighted_f1': to_float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'class_0': {'precision': to_float(chinese_precision), 'recall': to_float(chinese_recall), 'f1': to_float(chinese_f1)},
        'class_1': {'precision': to_float(western_precision), 'recall': to_float(western_recall), 'f1': to_float(western_f1)},
        'confusion_matrix': cm.tolist(),
        'avg_confidence': to_float(avg_confidence)
    }

def save_error_samples(X_val, y_true, y_pred, y_proba, class_names, save_path):
    """保存错误分类的样本"""
    errors = []
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            errors.append({
                'text': X_val[i][:200] + '...' if len(X_val[i]) > 200 else X_val[i],
                'true_label': class_names[y_true[i]],
                'predicted_label': class_names[y_pred[i]],
                'confidence': float(np.max(y_proba[i])),
                'prob_class_0': float(y_proba[i][0]),
                'prob_class_1': float(y_proba[i][1])
            })
    
    if errors:
        df_errors = pd.DataFrame(errors)
        df_errors.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"错误样本已保存: {save_path} (共{len(errors)}个)")
    else:
        print("完美！没有错误分类的样本")

# ================= 主函数 =================

def main():
    parser = argparse.ArgumentParser(description='BERT文本分类训练')
    parser.add_argument('--mode', type=str, choices=['stage1', 'stage2'], 
                       default='stage2', help='训练模式：stage1(快速验证) 或 stage2(优化模型)')
    parser.add_argument('--downsample', action='store_true', 
                       help='阶段1：启用下采样（平衡数据）')
    parser.add_argument('--use_class_weights', action='store_true', 
                       help='阶段2：使用类别权重（保留所有数据）')
    parser.add_argument('--use_focal_loss', action='store_true',
                       help='使用Focal Loss（可选）')
    parser.add_argument('--model_name', type=str, default=MODEL_NAME,
                       help=f'模型名称（默认: {MODEL_NAME}）')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                       help=f'批次大小（默认: {BATCH_SIZE}）')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                       help=f'训练轮数（默认: {NUM_EPOCHS}）')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                       help=f'学习率（默认: {LEARNING_RATE}）')
    
    args = parser.parse_args()
    
# 根据模式自动设置参数
    if args.mode == 'stage1':
        args.downsample = True
        print("阶段1模式：快速验证（下采样）")
    elif args.mode == 'stage2':
        args.use_class_weights = True
        print("阶段2模式：优化模型（加权损失函数）")
    
    mode_suffix, output_dir, model_save_dir, reports_dir = build_output_dirs(args)
    
    # 创建输出目录
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    print(f"使用设备: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"模型: {args.model_name}")
    print(f"数据集: {DATASET_FILE}")
    print("-" * 60)
    
    # 1. 加载数据
    print("加载数据集...")
    if not os.path.exists(DATASET_FILE):
        print(f"错误：找不到数据集文件 {DATASET_FILE}")
        print("  请先运行 python 3.py 生成数据集")
        return
    
    df = pd.read_csv(DATASET_FILE, encoding='utf-8-sig')
    print(f"加载完成，共 {len(df)} 个样本")
    
# 显示数据分布
    print("\n 数据分布:")
    source_counts = df['source'].value_counts()
    print(source_counts)
    imbalance_ratio = source_counts.max() / source_counts.min()
    print(f"  类别比例: {imbalance_ratio:.2f}:1")
    
    # 2. 数据预处理
    print("\n数据预处理...")
    
    # 标签编码
    label_map = {label: idx for idx, label in enumerate(df['source'].unique())}
    reverse_label_map = {idx: label for label, idx in label_map.items()}
    df['label'] = df['source'].map(label_map)
    
    class_names = [reverse_label_map[i] for i in sorted(reverse_label_map.keys())]
    print(f"  类别映射: {label_map}")
    
    # 下采样（阶段1）
    if args.downsample:
        print("\n执行下采样（阶段1模式）...")
        min_count = source_counts.min()
        balanced_dfs = []
        
        for label in df['source'].unique():
            label_df = df[df['source'] == label].copy()
            if len(label_df) > min_count:
                label_df = label_df.sample(n=min_count, random_state=RANDOM_STATE)
                print(f"  [{label}] 下采样: {source_counts[label]} → {len(label_df)}")
            else:
                print(f"  [{label}] 保留: {len(label_df)}")
            balanced_dfs.append(label_df)
        
        df = pd.concat(balanced_dfs, ignore_index=True)
        print(f"下采样完成，共 {len(df)} 个样本（平衡后）")
    
# 准备数据
    texts = df['text'].values
    labels = df['label'].values
    
    # 3. 分层采样划分训练/验证集
    print("\n划分训练/验证集（分层采样）...")
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels,
        test_size=TEST_SIZE,
        stratify=labels, # 关键：保持类别比例
        random_state=RANDOM_STATE
    )
    
    print(f"  训练集: {len(X_train)} 个样本")
    print(f"  验证集: {len(X_val)} 个样本")
    
    # 显示训练集和验证集的类别分布
    train_dist = pd.Series(y_train).value_counts().sort_index()
    val_dist = pd.Series(y_val).value_counts().sort_index()
    print(f"\n训练集类别分布:")
    for idx, count in train_dist.items():
        print(f"  {class_names[idx]}: {count}")
    print(f"验证集类别分布:")
    for idx, count in val_dist.items():
        print(f"  {class_names[idx]}: {count}")
    
    # 4. 计算类别权重（阶段2）
    class_weights = None
    if args.use_class_weights:
        print("\n计算类别权重（阶段2模式）...")
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
        print(f"  类别权重: {dict(zip(class_names, class_weights.cpu().numpy()))}")
    
    # 5. 初始化模型和tokenizer
    # 确保使用本地模型路径，避免联网
    model_path = args.model_name
    if not os.path.isabs(model_path) and not os.path.exists(model_path):
        # 如果是相对路径且不存在，尝试在当前目录查找
        local_path = os.path.join(os.path.dirname(__file__), model_path)
        if os.path.exists(local_path):
            model_path = local_path
    
    print(f"\n初始化模型: {model_path}...")
    tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True)
    model = BertForSequenceClassification.from_pretrained(
        model_path,
        num_labels=len(label_map),
        local_files_only=True
    )
    model.to(DEVICE)
    
    # 6. 创建数据加载器
    print("\n创建数据加载器...")
    train_dataset = TextDataset(X_train, y_train, tokenizer, MAX_LENGTH)
    val_dataset = TextDataset(X_val, y_val, tokenizer, MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 7. 设置优化器和损失函数
    print("\n设置优化器和损失函数...")
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # 选择损失函数
    if args.use_focal_loss:
        criterion = FocalLoss(alpha=1, gamma=2)
        print("使用 Focal Loss")
    elif class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("使用加权 CrossEntropyLoss")
    else:
        criterion = nn.CrossEntropyLoss()
        print("使用标准 CrossEntropyLoss")
    
# 8. 保存训练配置
    config = {
        'model_name': args.model_name,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'epochs': args.epochs,
        'max_length': MAX_LENGTH,
        'mode': args.mode,
        'downsample': args.downsample,
        'use_class_weights': args.use_class_weights,
        'use_focal_loss': args.use_focal_loss,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'class_names': class_names,
        'label_map': label_map,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    config_path = os.path.join(reports_dir, f"training_config_{mode_suffix}.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"训练配置已保存: {config_path}")
    
    # 8. 训练循环
    print(f"\n开始训练（{args.epochs} 个epoch）...")
    print("-" * 60)
    
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    train_losses = []
    val_losses = []
    epoch_metrics = []  # 存储每个epoch的详细指标
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # 训练
        train_loss, _, _ = train_epoch(model, train_loader, criterion, optimizer, DEVICE, scheduler)
        train_losses.append(train_loss)
        
        # 验证
        val_loss, val_preds, val_labels, val_probs = evaluate(model, val_loader, criterion, DEVICE)
        val_losses.append(val_loss)
        
        # 确保列表转换为numpy数组，避免布尔比较返回单个标量
        val_labels_arr = np.array(val_labels)
        val_preds_arr = np.array(val_preds)
        
        # 计算每个epoch的详细指标
        val_f1_macro = f1_score(val_labels_arr, val_preds_arr, average='macro', zero_division=0)
        val_f1_weighted = f1_score(val_labels_arr, val_preds_arr, average='weighted', zero_division=0)
        val_precision = precision_score(val_labels_arr, val_preds_arr, average='macro', zero_division=0)
        val_recall = recall_score(val_labels_arr, val_preds_arr, average='macro', zero_division=0)
        
        # 每个类别的F1
        f1_class_0 = f1_score(val_labels_arr == 0, val_preds_arr == 0, zero_division=0)
        f1_class_1 = f1_score(val_labels_arr == 1, val_preds_arr == 1, zero_division=0)
        
        epoch_metrics.append({
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'val_f1_macro': float(val_f1_macro),
            'val_f1_weighted': float(val_f1_weighted),
            'val_precision': float(val_precision),
            'val_recall': float(val_recall),
            'f1_class_0': float(f1_class_0),
            'f1_class_1': float(f1_class_1)
        })
        
        epoch_time = time.time() - epoch_start_time
        
        print(f"  训练损失: {train_loss:.4f}")
        print(f"  验证损失: {val_loss:.4f}")
        print(f"  验证F1 (宏平均): {val_f1_macro:.4f}")
        print(f"  验证F1 (加权): {val_f1_weighted:.4f}")
        print(f"  {class_names[0]} F1: {f1_class_0:.4f}")
        print(f"  {class_names[1]} F1: {f1_class_1:.4f}")
        print(f"  耗时: {epoch_time:.2f}秒")
        
        # 保存最佳模型（基于验证损失或F1）
        is_best = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            is_best = True
        if val_f1_macro > best_val_f1:
            best_val_f1 = val_f1_macro
            is_best = True
        
        if is_best:
            model_save_path = os.path.join(model_save_dir, f"best_model_epoch_{epoch+1}.pt")
            # 保存完整模型（包括tokenizer信息）
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_f1_macro': val_f1_macro,
                'config': config
            }, model_save_path)
            print(f"  保存最佳模型: {model_save_path} (F1: {val_f1_macro:.4f}, Loss: {val_loss:.4f})")
    
    total_time = time.time() - start_time
    print(f"\n总训练时间: {total_time/60:.2f}分钟 ({total_time:.2f}秒)")
    
# 9. 最终评估
    print("\n" + "=" * 60)
    print("最终评估")
    print("=" * 60)
    
    # 加载最佳模型进行评估
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    val_loss, val_preds, val_labels, val_probs = evaluate(model, val_loader, criterion, DEVICE)
    
    # 生成报告（mode_suffix已在前面定义）
    report_path = os.path.join(reports_dir, f"classification_report_{mode_suffix}.txt")
    cm_path = os.path.join(reports_dir, f"confusion_matrix_{mode_suffix}.png")
    error_samples_path = os.path.join(reports_dir, f"error_samples_{mode_suffix}.csv")
    metrics_json_path = os.path.join(reports_dir, f"metrics_{mode_suffix}.json")
    
    # 详细指标分析
    metrics_dict = print_detailed_metrics(val_labels, val_preds, val_probs, class_names, report_path, X_val)
    metrics_dict['total_training_time'] = total_time
    metrics_dict['epoch_metrics'] = epoch_metrics
    
    # 保存指标到JSON
    with open(metrics_json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)
    
    # 可视化
    plot_confusion_matrix(val_labels, val_preds, class_names, cm_path)
    plot_roc_pr_curves(val_labels, val_probs, class_names, reports_dir, mode_suffix)
    
    # 保存错误样本
    save_error_samples(X_val, val_labels, val_preds, val_probs, class_names, error_samples_path)
    
    # Plot training curves (loss, F1, precision/recall, balance)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss curves
    axes[0, 0].plot(range(1, args.epochs + 1), train_losses, label='Train loss', marker='o')
    axes[0, 0].plot(range(1, args.epochs + 1), val_losses, label='Val loss', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training & validation loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # F1 curves
    f1_macros = [m['val_f1_macro'] for m in epoch_metrics]
    f1_class_0_list = [m['f1_class_0'] for m in epoch_metrics]
    f1_class_1_list = [m['f1_class_1'] for m in epoch_metrics]
    axes[0, 1].plot(range(1, args.epochs + 1), f1_macros, label='Macro F1', marker='o')
    axes[0, 1].plot(range(1, args.epochs + 1), f1_class_0_list, label=f'{class_names[0]} F1', marker='s')
    axes[0, 1].plot(range(1, args.epochs + 1), f1_class_1_list, label=f'{class_names[1]} F1', marker='^')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1-Score')
    axes[0, 1].set_title('F1-score over epochs')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision & Recall curves
    precisions = [m['val_precision'] for m in epoch_metrics]
    recalls = [m['val_recall'] for m in epoch_metrics]
    axes[1, 0].plot(range(1, args.epochs + 1), precisions, label='Precision', marker='o')
    axes[1, 0].plot(range(1, args.epochs + 1), recalls, label='Recall', marker='s')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Precision & Recall')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # F1 difference (measure class balance)
    f1_diff = [abs(f1_class_0_list[i] - f1_class_1_list[i]) for i in range(len(f1_class_0_list))]
    axes[1, 1].plot(range(1, args.epochs + 1), f1_diff, label='F1 gap', marker='o', color='red')
    axes[1, 1].axhline(y=0.1, color='green', linestyle='--', label='Target threshold (0.1)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('|F1_0 - F1_1|')
    axes[1, 1].set_title('Per-class F1 gap (smaller is better)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    curve_path = os.path.join(reports_dir, f"training_curves_{mode_suffix}.png")
    plt.savefig(curve_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved: {curve_path}")
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"模型保存在: {model_save_dir}")
    print(f"报告保存在: {reports_dir}")
    print("\n 下一步建议:")
    if args.mode == 'stage1':
        print("  1. 检查混淆矩阵和分类报告")
        print("  2. 如果模型能区分两类，进入阶段2（使用加权损失函数）")
        print("  3. 运行: python train_bert.py --mode stage2")
    else:
        print("  1. 检查每个类别的F1-score是否接近")
        print("  2. 如果少数类召回率仍低，尝试Focal Loss")
        print("  3. 运行: python train_bert.py --mode stage2 --use_focal_loss")

if __name__ == "__main__":
    main()

