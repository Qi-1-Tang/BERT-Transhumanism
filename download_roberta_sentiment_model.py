"""
在本地电脑上下载 RoBERTa-base-sentiment 模型
下载后需要上传到服务器使用

使用方法：
python download_roberta_sentiment_model.py
"""

import os
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from pathlib import Path
import time

# 模型配置
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
MODEL_DESCRIPTION = "RoBERTa-base-sentiment (Twitter情感分析模型)"
SAVE_DIR = "roberta-base-sentiment-local"


def check_dependencies():
    """检查必要的依赖包"""
    required_packages = {
        'transformers': 'transformers',
        'torch': 'torch'
    }

    missing_packages = []
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)

    if missing_packages:
        print(" 缺少以下依赖包:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\n请运行以下命令安装:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False

    return True


def setup_mirror():
    """设置 HuggingFace 镜像（适用于国内用户）"""
    # 尝试使用国内镜像
    mirrors = [
        "https://hf-mirror.com",  # HuggingFace 镜像
    ]
    
    # 设置环境变量
    original_hf_endpoint = os.environ.get('HF_ENDPOINT', '')
    
    for mirror in mirrors:
        try:
            print(f"尝试使用镜像: {mirror}")
            os.environ['HF_ENDPOINT'] = mirror
            return mirror, original_hf_endpoint
        except:
            continue
    
    return None, original_hf_endpoint


def download_model(max_retries=3, timeout=60):
    """下载模型和分词器到本地目录，带重试机制"""
    print("=" * 60)
    print(f"下载模型: {MODEL_DESCRIPTION}")
    print(f"模型名称: {MODEL_NAME}")
    print(f"保存到: {SAVE_DIR}")
    print("=" * 60)

    # 检查依赖
    if not check_dependencies():
        sys.exit(1)

    # 设置镜像
    mirror, original_endpoint = setup_mirror()
    if mirror:
        print(f"\n✓ 使用镜像站点: {mirror}")
    else:
        print("\n⚠ 使用默认 HuggingFace 站点（如果连接慢，建议使用代理或镜像）")

    # 重试下载
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                wait_time = 2 ** attempt  # 指数退避
                print(f"\n重试 {attempt}/{max_retries-1}，等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            
            print(f"\n正在下载模型（尝试 {attempt + 1}/{max_retries}，这可能需要几分钟）...")
            print("  提示：如果连接超时，可以：")
            print("    1. 使用 VPN 或代理")
            print("    2. 使用 HuggingFace 镜像站点")
            print("    3. 手动下载模型文件")

            # 下载分词器
            print("\n步骤 1/2: 下载分词器 (Tokenizer)...")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            print("✓ 分词器下载完成")

            # 下载模型
            print("\n步骤 2/2: 下载模型 (Model)...")
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
            print("✓ 模型下载完成")

            # 保存到本地目录
            print(f"\n保存模型到本地目录: {SAVE_DIR}...")
            os.makedirs(SAVE_DIR, exist_ok=True)
            tokenizer.save_pretrained(SAVE_DIR)
            model.save_pretrained(SAVE_DIR)
            print("✓ 模型保存完成")

            # 验证保存的文件
            config_file = os.path.join(SAVE_DIR, "config.json")
            if os.path.exists(config_file):
                # 恢复原始环境变量
                if original_endpoint:
                    os.environ['HF_ENDPOINT'] = original_endpoint
                elif 'HF_ENDPOINT' in os.environ:
                    del os.environ['HF_ENDPOINT']
                
                # 显示模型信息
                print("\n" + "=" * 60)
                print("模型信息:")
                print("=" * 60)

                # 计算模型大小
                try:
                    model_size = sum(p.numel() * p.element_size() for p in model.parameters())
                    model_size_mb = model_size / (1024 * 1024)
                    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()):,}")
                    print(f"  模型大小: {model_size_mb:.2f} MB")
                except:
                    pass

                print(f"  词汇表大小: {len(tokenizer.get_vocab()):,}")
                print(f"  最大序列长度: {tokenizer.model_max_length}")
                print(f"  保存位置: {os.path.abspath(SAVE_DIR)}")
                
                print("\n✓ 模型下载并保存成功！")
                print(f"\n下一步:")
                print(f"  请将整个 '{SAVE_DIR}' 目录上传到服务器:")
                print(f"    /root/autodl-tmp/clean/{SAVE_DIR}")
                return True
            else:
                print(f" 警告: 保存的模型可能不完整")
                if attempt < max_retries - 1:
                    continue
                return False

        except Exception as e:
            error_msg = str(e)
            print(f"\n✗ 下载失败 (尝试 {attempt + 1}/{max_retries}): {error_msg}")
            
            if "timeout" in error_msg.lower() or "connection" in error_msg.lower():
                print("  原因: 网络连接超时")
                if attempt < max_retries - 1:
                    print("  将自动重试...")
                    continue
                else:
                    print("\n建议解决方案:")
                    print("  1. 检查网络连接")
                    print("  2. 使用 VPN 或代理")
                    print("  3. 设置环境变量使用镜像:")
                    print("     export HF_ENDPOINT=https://hf-mirror.com")
                    print("  4. 手动下载模型（见下方说明）")
            else:
                # 其他错误，不重试
                if attempt < max_retries - 1:
                    continue
                break
    
    # 恢复原始环境变量
    if original_endpoint:
        os.environ['HF_ENDPOINT'] = original_endpoint
    elif 'HF_ENDPOINT' in os.environ:
        del os.environ['HF_ENDPOINT']
    
    return False


if __name__ == "__main__":
    print("="*60)
    print("RoBERTa-base-sentiment 模型本地下载工具")
    print("="*60)
    print("\n此脚本将在本地下载模型，然后需要上传到服务器")
    print(f"\n模型: {MODEL_DESCRIPTION}")
    print(f"模型名称: {MODEL_NAME}")

    # 检查是否设置了镜像
    hf_endpoint = os.environ.get('HF_ENDPOINT', '')
    if hf_endpoint:
        print(f"\n检测到环境变量 HF_ENDPOINT={hf_endpoint}")
        print("将使用此镜像站点下载模型")

    success = download_model(max_retries=5, timeout=120)

    if success:
        print("\n" + "="*60)
        print("下载完成！")
        print("="*60)
        print("\n上传说明:")
        print("1. 使用 scp (Linux/Mac):")
        print(f"   scp -r {SAVE_DIR} root@your-server:/root/autodl-tmp/clean/")
        print("\n2. 使用 WinSCP (Windows): 拖拽上传")
        print("\n3. 使用 autodl 文件管理: 通过网页界面上传")
    else:
        print("\n" + "="*60)
        print("自动下载失败")
        print("="*60)
        print("\n替代方案:")
        print("\n方案1: 使用镜像站点（推荐）")
        print("  在运行脚本前设置环境变量:")
        print("  Linux/Mac:")
        print("    export HF_ENDPOINT=https://hf-mirror.com")
        print("    python download_roberta_sentiment_model.py")
        print("  Windows (CMD):")
        print("    set HF_ENDPOINT=https://hf-mirror.com")
        print("    python download_roberta_sentiment_model.py")
        print("  Windows (PowerShell):")
        print("    $env:HF_ENDPOINT='https://hf-mirror.com'")
        print("    python download_roberta_sentiment_model.py")
        
        print("\n方案2: 使用 VPN 或代理")
        print("  配置系统代理后重新运行脚本")
        
        print("\n方案3: 手动下载（如果自动下载一直失败）")
        print(f"  1. 访问: https://huggingface.co/{MODEL_NAME}")
        print(f"  2. 下载所有文件到本地目录")
        print(f"  3. 将目录重命名为: {SAVE_DIR}")
        
        print("\n方案4: 使用 git lfs（如果已安装）")
        print(f"  git lfs install")
        print(f"  git clone https://huggingface.co/{MODEL_NAME}")
        print(f"  mv {MODEL_NAME.split('/')[-1]} {SAVE_DIR}")
