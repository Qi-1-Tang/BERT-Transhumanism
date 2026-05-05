"""
在本地电脑上下载 SentenceTransformer 模型
下载后需要上传到服务器使用

使用方法：
python download_model_local.py
"""

from sentence_transformers import SentenceTransformer
import os
from pathlib import Path
import time

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


def download_model(model_name, save_dir, max_retries=3, timeout=60):
    """下载模型到本地，带重试机制"""
    print(f"\n{'='*60}")
    print(f"下载模型: {model_name}")
    print(f"保存到: {save_dir}")
    print(f"{'='*60}\n")

    # 设置镜像
    mirror, original_endpoint = setup_mirror()
    if mirror:
        print(f"✓ 使用镜像站点: {mirror}")
    else:
        print(" 使用默认 HuggingFace 站点（如果连接慢，建议使用代理或镜像）")

    # 重试下载
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                wait_time = 2 ** attempt  # 指数退避
                print(f"\n重试 {attempt}/{max_retries-1}，等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            
            print(f"正在下载模型（尝试 {attempt + 1}/{max_retries}，这可能需要几分钟）...")
            print("  提示：如果连接超时，可以：")
            print("    1. 使用 VPN 或代理")
            print("    2. 使用 HuggingFace 镜像站点")
            print("    3. 手动下载模型文件")
            
            # 下载模型（SentenceTransformer 会自动处理超时）
            model = SentenceTransformer(model_name)

            print(f"\n保存模型到本地目录...")
            model.save(save_dir)

            # 验证
            modules_json = os.path.join(save_dir, "modules.json")
            if os.path.exists(modules_json):
                # 恢复原始环境变量
                if original_endpoint:
                    os.environ['HF_ENDPOINT'] = original_endpoint
                elif 'HF_ENDPOINT' in os.environ:
                    del os.environ['HF_ENDPOINT']
                
                print(f"\n✓ 模型下载并保存成功！")
                print(f"  保存位置: {os.path.abspath(save_dir)}")
                print(f"\n下一步:")
                print(f"  请将整个 '{save_dir}' 目录上传到服务器:")
                print(f"    /root/autodl-tmp/clean/{save_dir}")
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
                break
    
    # 恢复原始环境变量
    if original_endpoint:
        os.environ['HF_ENDPOINT'] = original_endpoint
    elif 'HF_ENDPOINT' in os.environ:
        del os.environ['HF_ENDPOINT']
    
    return False


if __name__ == "__main__":
    print("="*60)
    print("SentenceTransformer 模型本地下载工具")
    print("="*60)
    print("\n此脚本将在本地下载模型，然后需要上传到服务器")
    print("\n推荐模型: all-MiniLM-L6-v2 (速度快，效果好，约90MB)")

    # 检查是否设置了镜像
    hf_endpoint = os.environ.get('HF_ENDPOINT', '')
    if hf_endpoint:
        print(f"\n检测到环境变量 HF_ENDPOINT={hf_endpoint}")
        print("将使用此镜像站点下载模型")

    # 下载推荐的模型
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    save_dir = 'all-MiniLM-L6-v2-local'

    success = download_model(model_name, save_dir, max_retries=5, timeout=120)

    if success:
        print("\n" + "="*60)
        print("下载完成！")
        print("="*60)
        print("\n上传说明:")
        print("1. 使用 scp (Linux/Mac):")
        print(f"   scp -r {save_dir} root@your-server:/root/autodl-tmp/clean/")
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
        print("    python download_model_local.py")
        print("  Windows (CMD):")
        print("    set HF_ENDPOINT=https://hf-mirror.com")
        print("    python download_model_local.py")
        print("  Windows (PowerShell):")
        print("    $env:HF_ENDPOINT='https://hf-mirror.com'")
        print("    python download_model_local.py")
        
        print("\n方案2: 使用 VPN 或代理")
        print("  配置系统代理后重新运行脚本")
        
        print("\n方案3: 手动下载（如果自动下载一直失败）")
        print("  1. 访问: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2")
        print("  2. 下载所有文件到本地目录")
        print("  3. 将目录重命名为: all-MiniLM-L6-v2-local")
        
        print("\n方案4: 使用 git lfs（如果已安装）")
        print("  git lfs install")
        print("  git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2")
        print("  mv all-MiniLM-L6-v2 all-MiniLM-L6-v2-local")

