import os
import pandas as pd
import re
import argparse

# ================= 配置区域 =================

INPUT_DIRS = {
    "Deep_Cleaned_English": "Western_SciFi", # 深度清洗后的西方语料（200本）
    "Deep_Cleaned_Chinese": "Chinese_Xianxia" # 深度清洗后的中国修仙语料（20本）
}

OUTPUT_FILE = "bert_training_dataset.csv"

# 切分参数 (BERT 限制)
# BERT 通常限制在 512 tokens，考虑到 tokenization 后可能变长，这里用字符数估算
# 英文：约 4 字符 = 1 token，所以 400 tokens ≈ 1600 字符
# 中文：约 1.5 字符 = 1 token，所以 400 tokens ≈ 600 字符
# 为了统一处理，使用字符数作为切分标准
WINDOW_SIZE_CHARS = 1200 # 每段约 1200 字符（约 300-400 tokens）
OVERLAP_CHARS = 150 # 保留 150 字符的重叠，防止语义中断
MIN_CHUNK_SIZE = 200 # 最小片段长度（太短的片段可能没有意义）

# 数据平衡选项
ENABLE_BALANCE = False # 设为 True 可以平衡两类数据（下采样多数类）
MAX_SAMPLES_PER_CLASS = None # 如果设置，每个类别最多保留这么多样本（None表示不限制）
RANDOM_STATE = 42 # 随机种子，用于下采样和shuffle的可重复性

# 顺序处理选项
KEEP_ORDER = True # 保持文件处理顺序，不进行shuffle（设为False会打乱数据）

# 文本归一化选项
KEEP_PARAGRAPH_SEP = False # 是否保留段落分隔（双换行符）
# 对于BERT训练，建议设为False：


# ================= 切分核心逻辑 =================

def normalize_text(text):
    """
    文本预处理：将单换行符替换为空格
    根据配置决定是否保留段落分隔（双换行符）
    例如：Introduction\nIn the final... -> Introduction In the final...
    """
# 使用临时标记来保护段落分隔
    TEMP_MARKER = '\uE000PARAGRAPH_SEP\uE000'
    
    if KEEP_PARAGRAPH_SEP:
# 保留段落分隔：先标记段落分隔，再替换单换行符
# 先保护段落分隔（双换行符或更多）
        text = re.sub(r'\n{2,}', TEMP_MARKER, text)
# 将剩余的单换行符替换为空格
        text = text.replace('\n', ' ')
# 恢复段落分隔
        text = text.replace(TEMP_MARKER, '\n\n')
# 清理段落分隔前后的多余空格
        text = re.sub(r' +(\n\n) +', r'\1', text)
        text = re.sub(r'(\n\n) +', r'\1', text)
        text = re.sub(r' +(\n\n)', r'\1', text)
    else:
# 不保留段落分隔：所有换行符都替换为空格，文本完全连续
        text = text.replace('\n', ' ')
    
# 清理多余的空格（多个连续空格替换为单个空格）
    text = re.sub(r' +', ' ', text)
    
# 清理行首行尾空格
    text = text.strip()
    
    return text


def is_mostly_english(text):
    """
    判断文本是否主要是英文
    通过统计中文字符比例来判断
    """
    if not text:
        return True
    
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    total_chars = len(text)
    chinese_ratio = chinese_chars / total_chars if total_chars > 0 else 0
    
# 如果中文字符占比小于30%，认为是英文文本
    return chinese_ratio < 0.3


def find_word_boundary(text, position, direction='backward'):
    """
    在指定位置附近查找单词边界（空格、标点等）
    direction: 'backward' 向前查找, 'forward' 向后查找
    """
    if direction == 'backward':
# 向前查找最近的单词边界
        for i in range(position, max(0, position - 100), -1):
            if i < len(text) and text[i] in ' \n\t.,!?;:':
                return i
        return max(0, position - 50) # 如果找不到，至少回退50个字符
    else:
# 向后查找最近的单词边界
        for i in range(position, min(len(text), position + 100)):
            if i < len(text) and text[i] in ' \n\t.,!?;:':
                return i + 1 # 包含边界字符
        return min(len(text), position + 50) # 如果找不到，至少前进50个字符


def sliding_window_segment(text, window_size_chars, overlap_chars, min_chunk_size):
    """
    将长文本按字符数切分为小片段 (Chunking)
    修复：确保英文单词不会被切断，在单词边界处切分
    """
    text = text.strip()
    
# 如果文本太短，直接返回
    if len(text) <= window_size_chars:
        return [text] if len(text) >= min_chunk_size else []

# 判断文本类型
    mostly_english = is_mostly_english(text)
    
    chunks = []
    step = window_size_chars - overlap_chars # 步长
    i = 0
    
    while i < len(text):
# 计算窗口结束位置
        window_end = min(i + window_size_chars, len(text))
        
# 如果已经到文本末尾
        if window_end >= len(text):
            chunk = text[i:].strip()
            if len(chunk) >= min_chunk_size:
                chunks.append(chunk)
            break
        
# 取出窗口内的文本
        chunk = text[i:window_end]
        chunk_length = len(chunk)
        
# 对于英文文本，确保不在单词中间切断
        if mostly_english:
# 优先：尝试在句子边界处切分
            sentence_ends = [
                chunk.rfind('.'),
                chunk.rfind('!'),
                chunk.rfind('?'),
                chunk.rfind('\n\n'), # 段落分隔
                chunk.rfind('\n')
            ]
            sentence_end = max(sentence_ends) if any(e >= 0 for e in sentence_ends) else -1
            
# 如果找到句子边界且在合理范围内（最后200字符内）
            if sentence_end >= 0 and sentence_end > chunk_length - 200 and sentence_end >= min_chunk_size:
                chunk = chunk[:sentence_end + 1].strip()
                i = i + sentence_end + 1
            else:
# 次优：在单词边界（空格）处切分
# 从窗口末尾向前查找最近的空格
                space_pos = chunk.rfind(' ')
                
                if space_pos >= min_chunk_size:
# 找到空格，在空格处切分（不包含空格）
                    chunk = chunk[:space_pos].strip()
                    i = i + space_pos + 1 # 跳过空格
                else:
# 如果找不到空格（可能是超长单词或特殊格式），尝试其他分隔符
                    other_seps = ['\t', '-', '—', '–']
                    found_sep = -1
                    for sep in other_seps:
                        sep_pos = chunk.rfind(sep)
                        if sep_pos >= min_chunk_size:
                            found_sep = max(found_sep, sep_pos)
                    
                    if found_sep >= min_chunk_size:
                        chunk = chunk[:found_sep + 1].strip()
                        i = i + found_sep + 1
                    else:
# 实在找不到分隔符，只能硬切（这种情况应该很少，比如超长URL或代码）
# 但至少确保不会无限循环
                        chunk = chunk.strip()
                        i = window_end
        else:
# 中文文本，可以在字符边界切分，但也要尽量在句子边界
            sentence_ends = [
                chunk.rfind('。'),
                chunk.rfind('！'),
                chunk.rfind('？'),
                chunk.rfind('.\n'), # 英文句号+换行
                chunk.rfind('.'),
                chunk.rfind('!'),
                chunk.rfind('?'),
                chunk.rfind('\n\n'), # 段落分隔
                chunk.rfind('\n')
            ]
            sentence_end = max(sentence_ends) if any(e >= 0 for e in sentence_ends) else -1
            
            if sentence_end >= 0 and sentence_end > chunk_length - 200 and sentence_end >= min_chunk_size:
                chunk = chunk[:sentence_end + 1].strip()
                i = i + sentence_end + 1
            else:
# 中文可以在字符边界切分（中文字符本身就是词）
                chunk = chunk.strip()
                i = window_end
        
# 过滤掉太短的片段
        if len(chunk) >= min_chunk_size:
            chunks.append(chunk)
        
# 防止无限循环：确保i至少前进一步
        if i <= 0 or i == window_end - overlap_chars:
            i = window_end
    
    return chunks


def split_oversized_chunk(chunk, max_chars_for_512_tokens=1800):
    """
    将超过 512 token 限制的片段进一步切分
    修复：确保英文单词不会被切断，在单词边界处切分
    max_chars_for_512_tokens: 估算的 512 tokens 对应的最大字符数
    """
    has_chinese = bool(re.search(r'[\u4e00-\u9fff]', chunk))
    mostly_english = is_mostly_english(chunk)
# 保守估算：英文 1800 字符 ≈ 450 tokens，中文 768 字符 ≈ 512 tokens
    max_chars = 768 if has_chinese else max_chars_for_512_tokens
    
    if len(chunk) <= max_chars:
        return [chunk]
    
# 如果超过限制，进一步切分
    sub_chunks = []
    step = max_chars - 100 # 小重叠
    i = 0
    
    while i < len(chunk):
        window_end = min(i + max_chars, len(chunk))
        
        if window_end >= len(chunk):
            sub_chunk = chunk[i:].strip()
            if len(sub_chunk) >= MIN_CHUNK_SIZE:
                sub_chunks.append(sub_chunk)
            break
        
        sub_chunk = chunk[i:window_end]
        
# 对于英文文本，确保不在单词中间切断
        if mostly_english:
# 优先：在句子边界切分
            sentence_ends = [
                sub_chunk.rfind('.'),
                sub_chunk.rfind('!'),
                sub_chunk.rfind('?'),
                sub_chunk.rfind('\n')
            ]
            sentence_end = max(sentence_ends) if any(e >= 0 for e in sentence_ends) else -1
            
            if sentence_end >= 0 and sentence_end > len(sub_chunk) - 150 and sentence_end >= MIN_CHUNK_SIZE:
                sub_chunk = sub_chunk[:sentence_end + 1].strip()
                i = i + sentence_end + 1
            else:
# 次优：在单词边界（空格）处切分
                space_pos = sub_chunk.rfind(' ')
                if space_pos >= MIN_CHUNK_SIZE:
                    sub_chunk = sub_chunk[:space_pos].strip()
                    i = i + space_pos + 1
                else:
# 找不到空格，尝试其他分隔符
                    other_seps = ['\t', '-', '—', '–']
                    found_sep = -1
                    for sep in other_seps:
                        sep_pos = sub_chunk.rfind(sep)
                        if sep_pos >= MIN_CHUNK_SIZE:
                            found_sep = max(found_sep, sep_pos)
                    
                    if found_sep >= MIN_CHUNK_SIZE:
                        sub_chunk = sub_chunk[:found_sep + 1].strip()
                        i = i + found_sep + 1
                    else:
# 实在找不到，只能硬切（但这种情况应该很少）
                        sub_chunk = sub_chunk.strip()
                        i = window_end
        else:
# 中文文本，可以在字符边界切分，但优先句子边界
            sentence_ends = [
                sub_chunk.rfind('。'),
                sub_chunk.rfind('！'),
                sub_chunk.rfind('？'),
                sub_chunk.rfind('.'),
                sub_chunk.rfind('!'),
                sub_chunk.rfind('?'),
                sub_chunk.rfind('\n')
            ]
            sentence_end = max(sentence_ends) if any(e >= 0 for e in sentence_ends) else -1
            
            if sentence_end >= 0 and sentence_end > len(sub_chunk) - 150 and sentence_end >= MIN_CHUNK_SIZE:
                sub_chunk = sub_chunk[:sentence_end + 1].strip()
                i = i + sentence_end + 1
            else:
# 中文可以在字符边界切分
                sub_chunk = sub_chunk.strip()
                i = window_end
        
        if len(sub_chunk) >= MIN_CHUNK_SIZE:
            sub_chunks.append(sub_chunk)
        
# 防止无限循环
        if i <= 0 or i == window_end - 100:
            i = window_end
    
    return sub_chunks if sub_chunks else [chunk[:max_chars]]


# ================= 主程序 =================

def process_file_chunks(file_path, file_name, label):
    """
    处理单个文件：读取、切分、处理超长片段
    返回处理后的数据列表
    """
# 1. 读取清洗好的 TXT
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except:
        print(f"读取失败: {file_name}")
        return []

# 2. 文本预处理：将单换行符替换为空格，保留段落分隔
    content = normalize_text(content)

# 3. 滑动窗口切分（基于字符数，适用于中英文）
    chunks = sliding_window_segment(
        content, 
        WINDOW_SIZE_CHARS, 
        OVERLAP_CHARS, 
        MIN_CHUNK_SIZE
    )

# 4. 处理每个chunk，检查并处理超长片段
    processed_data = []
    split_count = 0
    
    for idx, chunk in enumerate(chunks):
# 估算token数
        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', chunk))
        estimated_tokens = len(chunk) / (1.5 if has_chinese else 4)
        
# 如果超过 512 tokens，进一步切分
        if estimated_tokens > 512:
            sub_chunks = split_oversized_chunk(chunk)
            split_count += len(sub_chunks) - 1
            
            for sub_idx, sub_chunk in enumerate(sub_chunks):
                sub_has_chinese = bool(re.search(r'[\u4e00-\u9fff]', sub_chunk))
                sub_estimated_tokens = len(sub_chunk) / (1.5 if sub_has_chinese else 4)
                
                processed_data.append({
                    "text": sub_chunk,
                    "source": label,
                    "book_name": file_name,
                    "chunk_id": f"{file_name}_{idx}_sub{sub_idx}",
                    "char_count": len(sub_chunk),
                    "estimated_tokens": int(sub_estimated_tokens)
                })
        else:
            processed_data.append({
                "text": chunk,
                "source": label,
                "book_name": file_name,
                "chunk_id": f"{file_name}_{idx}",
                "char_count": len(chunk),
                "estimated_tokens": int(estimated_tokens)
            })
    
    if split_count > 0:
        print(f"✓ 生成 {len(chunks)} 个片段，其中 {split_count} 个超长片段已进一步切分 → 共 {len(processed_data)} 个片段")
    else:
        print(f"✓ 生成 {len(processed_data)} 个片段")
    
    return processed_data


def apply_downsampling(df):
    """
    应用下采样功能：平衡两类数据
    """
    print("\n  执行下采样（数据平衡）...")
    balanced_dfs = []
    
    source_counts = df['source'].value_counts()
    min_count = source_counts.min()
    target_count = MAX_SAMPLES_PER_CLASS if MAX_SAMPLES_PER_CLASS else min_count
    
    for label in df['source'].unique():
        label_df = df[df['source'] == label].copy()
        original_count = len(label_df)
        
        if len(label_df) > target_count:
# 下采样（随机采样，更公平地代表整体数据分布）
            label_df = label_df.sample(n=target_count, random_state=RANDOM_STATE)
            print(f" [{label}] 下采样: {original_count} → {len(label_df)} (随机采样)")
        else:
            print(f" [{label}] 保留: {len(label_df)}")
        
        balanced_dfs.append(label_df)
    
    df_balanced = pd.concat(balanced_dfs, ignore_index=True)
    
# 重新保存平衡后的数据
    df_balanced.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    return df_balanced


def main():
# 解析命令行参数
    parser = argparse.ArgumentParser(
        description='构建BERT训练数据集',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python 3.py # 生成所有片段（默认）
  python 3.py --function downsampling # 生成下采样后的平衡数据集
        """
    )
    parser.add_argument(
        '--function', 
        type=str, 
        choices=['downsampling'],
        help='可选功能：downsampling（下采样平衡数据）'
    )
    args = parser.parse_args()
    
# 根据命令行参数决定是否启用下采样
    enable_downsampling = (args.function == 'downsampling')
    
    print("开始构建 BERT 训练数据集...")
    print(f"输出文件: {OUTPUT_FILE}")
    if enable_downsampling:
        print("下采样模式：将自动平衡两类数据")
    
# 如果输出文件已存在，先删除（从头开始）
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        print(f" 已删除旧的输出文件")
    
    total_files_processed = 0
    total_chunks = 0
    is_first_write = True # 标记是否是第一次写入（用于控制表头）
    
# 按顺序处理每个文件夹
    for folder_path, label in INPUT_DIRS.items():
        if not os.path.exists(folder_path):
            print(f"跳过：找不到文件夹 {folder_path} (请确认路径是否正确)")
            continue

# 按文件名排序，确保顺序一致
        files = sorted([f for f in os.listdir(folder_path) if f.endswith('.txt')])
        print(f"\n 正在处理 [{label}]: {folder_path}, 共 {len(files)} 本书")

# 按顺序处理每个文件
        for file_idx, file_name in enumerate(files, 1):
            print(f" [{file_idx}/{len(files)}] 处理中: {file_name}...", end=" ")
            file_path = os.path.join(folder_path, file_name)

# 处理文件并获取数据
            file_data = process_file_chunks(file_path, file_name, label)
            
            if not file_data:
                continue
            
# 立即追加到CSV文件（保持顺序）
            df_file = pd.DataFrame(file_data)
            
# 第一次写入包含表头，之后追加不包含表头
            df_file.to_csv(OUTPUT_FILE, mode='a', index=False, header=is_first_write, encoding='utf-8-sig')
            is_first_write = False # 后续写入不再包含表头
            
            total_files_processed += 1
            total_chunks += len(file_data)

# 读取最终CSV进行统计和可选的数据平衡
    if total_chunks > 0:
        print(f"\n 处理完成！共处理 {total_files_processed} 个文件，生成 {total_chunks} 个片段")
        print(f"读取最终数据集进行统计...")
        
        df = pd.read_csv(OUTPUT_FILE, encoding='utf-8-sig')
        
# 数据平衡（通过命令行参数或配置选项）
        if enable_downsampling or ENABLE_BALANCE or MAX_SAMPLES_PER_CLASS:
            df = apply_downsampling(df)
        
# 可选：打乱数据（如果不需要保持顺序）
        if not KEEP_ORDER:
            print("\n 打乱数据顺序...")
            df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
            df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

        print("-" * 50)
        print(f"数据集构建完成！")
        print(f"文件已保存为: {OUTPUT_FILE}")
        if KEEP_ORDER:
            print(f"数据已按文件夹和文件顺序保存（未打乱）")
        print(f"\n 数据统计:")
        print(df['source'].value_counts())
        print(f"\n 总体信息:")
        print(f" - 总片段数: {len(df)}")
        print(f" - 平均字符数: {df['char_count'].mean():.1f}")
        print(f" - 平均估算token数: {df['estimated_tokens'].mean():.1f}")
        print(f" - 最大token数: {df['estimated_tokens'].max()}")
        print(f" - 最小token数: {df['estimated_tokens'].min()}")
        
# 检查是否有超过BERT限制的片段
        over_limit = (df['estimated_tokens'] > 512).sum()
        if over_limit > 0:
            print(f"\n  警告: 仍有 {over_limit} 个片段估算token数超过512")
            print(f"  这些片段可能需要手动检查或使用更严格的切分参数")
        else:
            print(f"\n 所有片段都在BERT的512 token限制内")
        
# 数据平衡建议
        source_counts = df['source'].value_counts()
        if len(source_counts) > 1:
            max_count = source_counts.max()
            min_count = source_counts.min()
            imbalance_ratio = max_count / min_count
            if imbalance_ratio > 5:
                print(f"\n 数据不平衡提示:")
                print(f"  类别比例约为 {imbalance_ratio:.1f}:1")
                print(f"  如果训练时遇到问题，可以在配置中启用 ENABLE_BALANCE 或设置 MAX_SAMPLES_PER_CLASS")
        
        print("-" * 50)
        print("下一步：")
        print(" 1. 检查 CSV 文件中的数据质量")
    else:
        print("没有生成任何数据，请检查输入文件夹路径。")


if __name__ == "__main__":
    main()