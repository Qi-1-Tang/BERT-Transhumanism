"""
检查GPU环境的脚本
在租用GPU服务器后，先运行此脚本确认环境是否正确
"""

import sys

def check_gpu():
    print("=" * 60)
    print("GPU环境检查")
    print("=" * 60)
    
    # 检查PyTorch
    try:
        import torch
        print(f"✓ PyTorch版本: {torch.__version__}")
    except ImportError:
        print("✗ PyTorch未安装")
        print("  请运行: pip install torch")
        return False
    
    # 检查CUDA
    cuda_available = torch.cuda.is_available()
    print(f"✓ CUDA可用: {cuda_available}")
    
    if cuda_available:
        print(f"✓ CUDA版本: {torch.version.cuda}")
        print(f"✓ GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    else:
        print("  警告: 未检测到GPU，将使用CPU训练（速度很慢）")
        print("  如果确定有GPU，请检查:")
        print("  1. GPU驱动是否正确安装")
        print("  2. CUDA版本是否匹配")
        print("  3. PyTorch是否支持CUDA")
    
    # 检查其他依赖
    print("\n检查其他依赖...")
    dependencies = {
        'transformers': 'transformers',
        'pandas': 'pandas',
        'numpy': 'numpy',
        # sklearn's import name is "sklearn" (pip package scikit-learn)
        'sklearn': 'sklearn',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'tqdm': 'tqdm'
    }
    
    all_ok = True
    for name, package in dependencies.items():
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} 未安装")
            all_ok = False
    
    # 检查数据集
    print("\n检查数据集...")
    import os
    if os.path.exists("bert_training_dataset.csv"):
        import pandas as pd
        df = pd.read_csv("bert_training_dataset.csv", encoding='utf-8-sig')
        print(f"✓ 数据集文件存在")
        print(f"  总样本数: {len(df)}")
        print(f"  类别分布:")
        print(df['source'].value_counts())
    else:
        print("  数据集文件不存在")
        print("  请先运行: python 3.py 生成数据集")
        all_ok = False
    
    print("\n" + "=" * 60)
    if cuda_available and all_ok:
        print("✓ 环境检查通过！可以开始训练")
        print("  运行: python train_bert.py --mode stage1")
    elif all_ok:
        print("  环境检查通过，但未检测到GPU")
        print("  可以使用CPU训练，但速度会很慢")
    else:
        print("✗ 环境检查未通过，请先解决上述问题")
    print("=" * 60)
    
    return cuda_available and all_ok

if __name__ == "__main__":
    check_gpu()

