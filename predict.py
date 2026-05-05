"""
BERT 文本分类推理脚本
使用训练好的模型对新文本进行分类预测

使用方法：
1. 单条文本预测：
   python predict.py --text "Your text here"

2. 批量文件预测：
   python predict.py --file input.txt --output results.csv

3. 交互式预测：
   python predict.py --interactive

4. 指定模型路径：
   python predict.py --model path/to/model.pt --text "Your text"
"""

import os
import argparse
import warnings

# 在导入 transformers 之前抑制 FutureWarning（PyTorch 版本兼容性警告，不影响功能）
warnings.filterwarnings('ignore', category=FutureWarning, module='transformers')
warnings.filterwarnings('ignore', message='.*torch.utils._pytree._register_pytree_node.*')

import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from train_bert import DEVICE, MAX_LENGTH

# 默认最佳模型路径
DEFAULT_MODEL_PATH = "training_output_stage2_classweights_focal/models/best_model_epoch_3.pt"


def load_model(model_path):
    """加载训练好的模型和配置"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    print(f"加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=DEVICE)
    config = checkpoint.get('config', {})
    
    # 获取配置信息
    model_name = config.get('model_name', 'bert-base-uncased-local')
    label_map = config.get('label_map', {})
    class_names = config.get('class_names', [])
    
    if not label_map or not class_names:
        raise ValueError("模型配置中缺少 label_map 或 class_names")
    
    # 创建反向映射（label -> class_name）
    reverse_label_map = {v: k for k, v in label_map.items()}
    class_names_ordered = [reverse_label_map[i] for i in sorted(reverse_label_map.keys())]
    
    # 处理模型路径：如果配置中的路径不存在，尝试使用相对路径
    # 1. 首先尝试使用配置中的路径（可能是绝对路径）
    model_path_to_use = model_name
    
    # 2. 如果路径不存在，尝试相对于当前脚本目录
    if not os.path.exists(model_path_to_use):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        relative_path = os.path.join(script_dir, model_name)
        if os.path.exists(relative_path):
            model_path_to_use = relative_path
            print(f"  使用相对路径: {model_path_to_use}")
        else:
            # 3. 尝试只使用路径的最后一部分（目录名）
            model_dir_name = os.path.basename(model_name)
            relative_path = os.path.join(script_dir, model_dir_name)
            if os.path.exists(relative_path):
                model_path_to_use = relative_path
                print(f"  使用本地模型目录: {model_path_to_use}")
            else:
                # 4. 如果都不存在，尝试使用默认的本地模型路径
                default_local_path = os.path.join(script_dir, 'bert-base-uncased-local')
                if os.path.exists(default_local_path):
                    model_path_to_use = default_local_path
                    print(f"  使用默认本地模型路径: {model_path_to_use}")
                else:
                    raise FileNotFoundError(
                        f"无法找到BERT模型文件。尝试过的路径：\n"
                        f"  1. {model_name}\n"
                        f"  2. {relative_path}\n"
                        f"  3. {default_local_path}\n"
                        f"请确保 bert-base-uncased-local 目录存在于项目根目录。"
                    )
    
    # 加载 tokenizer 和模型
    print(f"  加载BERT模型: {model_path_to_use}")
    tokenizer = BertTokenizer.from_pretrained(model_path_to_use, local_files_only=True)
    model = BertForSequenceClassification.from_pretrained(
        model_path_to_use, 
        num_labels=len(label_map),
        local_files_only=True
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    print(f"模型加载完成")
    print(f"  类别: {class_names_ordered}")
    print(f"  设备: {DEVICE}")
    
    return model, tokenizer, class_names_ordered


def predict_single(model, tokenizer, text, class_names, return_probs=False):
    """对单条文本进行预测"""
    # Tokenize
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)
    
    # 预测
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_label = np.argmax(probs)
        confidence = float(probs[pred_label])
    
    result = {
        'predicted_class': class_names[pred_label],
        'confidence': confidence,
        'probabilities': {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    }
    
    if return_probs:
        return result
    else:
        return result['predicted_class'], result['confidence']


def predict_batch(model, tokenizer, texts, class_names):
    """批量预测"""
    results = []
    for text in texts:
        pred_class, confidence = predict_single(model, tokenizer, text, class_names)
        results.append({
            'text': text[:100] + '...' if len(text) > 100 else text,
            'predicted_class': pred_class,
            'confidence': confidence
        })
    return results


def main():
    parser = argparse.ArgumentParser(description='BERT文本分类推理')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH,
                       help=f'模型路径（默认: {DEFAULT_MODEL_PATH}）')
    parser.add_argument('--text', type=str, default=None,
                       help='要预测的单条文本')
    parser.add_argument('--file', type=str, default=None,
                       help='包含文本的文件路径（每行一条）')
    parser.add_argument('--output', type=str, default='predictions.csv',
                       help='输出CSV文件路径（默认: predictions.csv）')
    parser.add_argument('--interactive', action='store_true',
                       help='交互式模式，可以连续输入文本进行预测')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='置信度阈值，低于此值会显示警告（默认: 0.5）')
    
    args = parser.parse_args()
    
    # 加载模型
    model, tokenizer, class_names = load_model(args.model)
    
    # 单条文本预测
    if args.text:
        print("\n" + "="*60)
        print("预测结果")
        print("="*60)
        result = predict_single(model, tokenizer, args.text, class_names, return_probs=True)
        print(f"文本: {args.text[:200]}..." if len(args.text) > 200 else f"文本: {args.text}")
        print(f"预测类别: {result['predicted_class']}")
        print(f"置信度: {result['confidence']:.4f}")
        if result['confidence'] < args.threshold:
            print(f"  警告: 置信度较低 ({result['confidence']:.4f} < {args.threshold})")
        print("\n各类别概率:")
        for cls, prob in result['probabilities'].items():
            print(f"  {cls}: {prob:.4f}")
        print("="*60)
    
    # 文件批量预测
    elif args.file:
        if not os.path.exists(args.file):
            print(f"错误: 文件不存在 {args.file}")
            return
        
        print(f"\n读取文件: {args.file}")
        with open(args.file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        print(f"共 {len(texts)} 条文本，开始预测...")
        results = predict_batch(model, tokenizer, texts, class_names)
        
        # 保存结果
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False, encoding='utf-8-sig')
        print(f"预测完成，结果已保存到: {args.output}")
        print(f"\n预测统计:")
        print(df['predicted_class'].value_counts())
        low_confidence = df[df['confidence'] < args.threshold]
        if len(low_confidence) > 0:
            print(f"\n  低置信度样本 ({len(low_confidence)} 个):")
            print(low_confidence[['text', 'confidence']].head(10))
    
    # 交互式模式
    elif args.interactive:
        print("\n" + "="*60)
        print("交互式预测模式")
        print("输入文本进行预测，输入 'quit' 或 'exit' 退出")
        print("="*60)
        
        while True:
            text = input("\n请输入文本: ").strip()
            if text.lower() in ['quit', 'exit', 'q']:
                print("退出交互模式")
                break
            
            if not text:
                continue
            
            pred_class, confidence = predict_single(model, tokenizer, text, class_names)
            print(f"预测类别: {pred_class}")
            print(f"置信度: {confidence:.4f}")
            if confidence < args.threshold:
                print(f"  警告: 置信度较低")
    
    else:
        print("请指定 --text、--file 或 --interactive 参数")
        print("使用 --help 查看帮助信息")


if __name__ == "__main__":
    main()