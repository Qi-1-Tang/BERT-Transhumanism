"""
主题建模脚本 - 提取小说中的主题并统计频率
使用BERTopic进行无监督主题建模，支持中方西方对比

使用方法：
1. 使用CSV文件（推荐，支持中方西方对比）：
   python topic_modeling.py --input_csv bert_training_dataset.csv --output topic_analysis.csv

2. 分析所有小说（从目录）：
   python topic_modeling.py --input_dir Cleaned_English_V2 --output topic_analysis.csv

3. 分析单个文件：
   python topic_modeling.py --input_file clean_Dune.txt --output dune_topics.csv

4. 指定主题数量：
   python topic_modeling.py --input_csv bert_training_dataset.csv --num_topics 10
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired
    from sentence_transformers import SentenceTransformer
    import sentence_transformers
    from sklearn.feature_extraction.text import CountVectorizer
except ImportError:
    print("错误: 需要安装 bertopic 和 sentence-transformers")
    print("请运行: pip install bertopic sentence-transformers>=5.0.0")
    exit(1)

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ================= 停用词列表（从 stopwords.py 导入） =================
from stopwords import CUSTOM_STOPWORDS  # noqa: E402


def save_stop_words_to_file(stop_words_list, output_file="stop_words.txt"):
    """保存停用词列表到文件"""
    if stop_words_list is None or len(stop_words_list) == 0:
        return
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for word in sorted(stop_words_list):
                f.write(f"{word}\n")
        print(f"  ✓ 停用词列表已保存到: {output_file}")
        print(f"    共 {len(stop_words_list)} 个停用词")
    except Exception as e:
        print(f"  警告: 无法保存停用词列表: {e}")


def load_stop_words_from_file(input_file):
    """从文件加载停用词列表"""
    if not os.path.exists(input_file):
        return None
    
    try:
        stop_words = set()
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().lower()
                if word:  # 忽略空行
                    stop_words.add(word)
        print(f"  ✓ 从文件加载了 {len(stop_words)} 个停用词: {input_file}")
        return list(stop_words)
    except Exception as e:
        print(f"  警告: 无法加载停用词文件: {e}")
        return None


def print_stop_words(stop_words_list, max_display=20):
    """打印停用词列表（用于查看）"""
    if stop_words_list is None or len(stop_words_list) == 0:
        print("  停用词列表为空")
        return
    
    sorted_words = sorted(stop_words_list)
    print(f"\n  当前停用词列表（共 {len(sorted_words)} 个）:")
    print("  " + "-" * 60)
    
    # 显示前 max_display 个
    for i, word in enumerate(sorted_words[:max_display], 1):
        print(f"  {i:3d}. {word}")
    
    if len(sorted_words) > max_display:
        print(f"  ... 还有 {len(sorted_words) - max_display} 个")
    print("  " + "-" * 60)

# 检查GPU可用性
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        print(f"✓ 检测到GPU: {torch.cuda.get_device_name(0)}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print(" 未检测到GPU，将使用CPU（速度较慢）")
        print("  建议：使用GPU可以显著加速文本嵌入过程")
except ImportError:
    GPU_AVAILABLE = False
    print(" PyTorch未安装，无法检测GPU")


def load_texts_from_csv(csv_path):
    """从CSV文件中加载文本数据，支持中方西方对比"""
    if not os.path.exists(csv_path):
        raise ValueError(f"CSV文件不存在: {csv_path}")
    
    print(f"从CSV文件加载数据: {csv_path}")
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    
    # 检查必要的列
    if 'text' not in df.columns:
        raise ValueError("CSV文件必须包含 'text' 列")
    
    # 过滤掉空文本
    df = df[df['text'].notna() & (df['text'].str.strip() != '')]
    
    texts = df['text'].tolist()
    
    # 获取文件名称（如果有book_name列，使用它；否则使用chunk_id）
    if 'book_name' in df.columns:
        file_names = df['book_name'].tolist()
    elif 'chunk_id' in df.columns:
        file_names = df['chunk_id'].tolist()
    else:
        file_names = [f"document_{i}" for i in range(len(texts))]
    
    # 获取来源信息（用于中方西方对比）
    sources = None
    if 'source' in df.columns:
        sources = df['source'].tolist()
        source_counts = df['source'].value_counts()
        print(f"数据来源分布:")
        for source, count in source_counts.items():
            print(f"  {source}: {count} 个文本段落")
    
    # 获取时间戳信息（用于动态主题建模 Topics over Time）
    timestamps = None
    if 'Timestamps' in df.columns:
        timestamps = df['Timestamps'].tolist()
        valid_ts = [t for t in timestamps if pd.notna(t)]
        print(f"时间戳覆盖: {len(valid_ts)}/{len(timestamps)} 个文本段落有时间戳")
        if valid_ts:
            print(f"  时间范围: {int(min(valid_ts))} - {int(max(valid_ts))}")
    else:
        print("提示: CSV中未找到 'Timestamps' 列，跳过动态主题建模")
        print("  运行 add_timestamps.py 可添加时间戳列")
    
    print(f"共加载 {len(texts)} 个文本段落")
    return texts, file_names, sources, timestamps


def load_texts_from_directory(directory):
    """从目录中加载所有文本文件"""
    texts = []
    file_names = []
    
    directory = Path(directory)
    text_files = list(directory.glob("*.txt"))
    
    if not text_files:
        raise ValueError(f"目录 {directory} 中没有找到 .txt 文件")
    
    print(f"找到 {len(text_files)} 个文本文件")
    
    for file_path in tqdm(text_files, desc="加载文件"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # 按段落分割（空行分隔）
                paragraphs = [p.strip() for p in content.split('\n\n') if p.strip() and len(p.strip()) > 50]
                texts.extend(paragraphs)
                file_names.extend([file_path.name] * len(paragraphs))
        except Exception as e:
            print(f"警告: 无法读取文件 {file_path}: {e}")
    
    print(f"共加载 {len(texts)} 个文本段落")
    return texts, file_names, None


def load_text_from_file(file_path):
    """从单个文件加载文本"""
    texts = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        # 按段落分割
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip() and len(p.strip()) > 50]
        texts.extend(paragraphs)
    
    print(f"从 {file_path} 加载了 {len(texts)} 个文本段落")
    return texts, [os.path.basename(file_path)] * len(texts), None


def segment_text(text, chunk_size=500, overlap=50):
    """将长文本分割成较小的块"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk.strip()) > 100:  # 只保留足够长的块
            chunks.append(chunk)
    
    return chunks


def perform_topic_modeling(texts, num_topics=None, min_topic_size=10, 
                          use_custom_stopwords=True):
    """执行主题建模"""
    print("\n开始主题建模...")
    print(f"文本数量: {len(texts)}")
    
    # 使用停用词列表（从 stopwords.py 导入）
    stop_words_list = None
    if use_custom_stopwords:
        stop_words_list = CUSTOM_STOPWORDS.copy()
        print(f"\n使用停用词列表（共 {len(stop_words_list)} 个词，来源: stopwords.py）")
        print_stop_words(stop_words_list, max_display=30)
    
    # 检查GPU并提示
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ 使用GPU加速: {torch.cuda.get_device_name(0)}")
            gpu_available = True
        else:
            print(" 使用CPU模式（速度较慢，建议使用GPU）")
            gpu_available = False
    except ImportError:
        gpu_available = False
        print(" PyTorch未安装，将使用CPU")
    
    # 使用本地SentenceTransformer模型（如果可用）或使用默认模型
    try:
        # ================= 修改开始：优先使用 AutoDL 指定路径 =================
        autodl_path = "/root/autodl-tmp/clean/all-MiniLM-L6-v2-local"
        local_st_model = "all-MiniLM-L6-v2-local"
        
        # 1. 优先检查 AutoDL 的绝对路径
        if os.path.exists(autodl_path):
            print(f"使用 AutoDL 本地 SentenceTransformer 模型: {autodl_path}")
            embedding_model = SentenceTransformer(autodl_path)
            if gpu_available:
                print("  → 模型将使用GPU进行文本嵌入")

        # 2. 其次检查当前目录下的本地模型
        elif os.path.exists(local_st_model):
            print(f"使用本地 SentenceTransformer 模型: {local_st_model}")
            embedding_model = SentenceTransformer(local_st_model)
            if gpu_available:
                print("  → 模型将使用GPU进行文本嵌入")

        # 3. 再尝试本地 BERT 模型
        elif os.path.exists("bert-base-uncased-local"):
            model_path = "bert-base-uncased-local"
            print(f"使用本地BERT模型: {model_path}")
            embedding_model = SentenceTransformer(model_path)
            if gpu_available:
                print("  → 模型将使用GPU进行文本嵌入")
        
        # 4. 最后尝试在线下载
        else:
            print("使用默认 SentenceTransformer 模型: all-MiniLM-L6-v2")
            print("  （将从网络下载，如果已有本地模型建议使用）")
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            if gpu_available:
                print("  → 模型将使用GPU进行文本嵌入")
        # ================= 修改结束 =================

    except Exception as e:
        error_msg = str(e)
        if "version" in error_msg.lower() or "created with version" in error_msg.lower():
            print(f"\n✗ 版本兼容性错误: {error_msg}")
            print("\n解决方案:")
            print("  请升级 sentence-transformers 到 5.0.0 或更高版本:")
            print("  pip install --upgrade sentence-transformers>=5.0.0")
            print("\n或者重新下载模型（使用新版本）:")
            print("  python download_model_local.py")
            raise
        else:
            print(f"警告: 无法加载本地模型，尝试使用默认模型: {e}")
            try:
                embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                if gpu_available:
                    print("  → 模型将使用GPU进行文本嵌入")
            except Exception as e2:
                print(f"✗ 无法加载任何模型: {e2}")
                raise
    
    # 创建BERTopic模型
    # 使用KeyBERT来改进主题表示
    representation_model = KeyBERTInspired()
    
    # 设置停用词（通过 CountVectorizer）
    vectorizer_model = None
    if stop_words_list:
        # BERTopic 通过 CountVectorizer 来设置停用词
        vectorizer_model = CountVectorizer(stop_words=stop_words_list)
        print(f"\n  已设置 {len(stop_words_list)} 个停用词（通过 CountVectorizer）")
    
    # 构建BERTopic参数
    topic_model_params = {
        'embedding_model': embedding_model,
        'representation_model': representation_model,
        'min_topic_size': min_topic_size,
        'verbose': True
    }
    
    # 添加 vectorizer_model（如果设置了停用词）
    if vectorizer_model:
        topic_model_params['vectorizer_model'] = vectorizer_model
    
    # nr_topics 用于减少主题数量（可选）
    # 如果指定了 num_topics，使用它来减少主题
    if num_topics is not None:
        topic_model_params['nr_topics'] = num_topics
    
    topic_model = BERTopic(**topic_model_params)
    
    # 执行主题建模（无监督，不需要训练，但需要计算文本嵌入和聚类）
    print("正在处理文本并提取主题（这可能需要一些时间）...")
    print("  注意：这是无监督方法，不需要训练模型，但需要计算文本嵌入和聚类")
    topics, probs = topic_model.fit_transform(texts)
    
    # 统计主题数量（排除噪声主题 -1）
    unique_topics = set(topics)
    num_topics_found = len(unique_topics) - (1 if -1 in unique_topics else 0)
    noise_count = topics.count(-1) if isinstance(topics, list) else (topics == -1).sum()
    
    print(f"\n发现 {num_topics_found} 个主题")
    print(f"未分类文档数（噪声）: {noise_count}")
    
    return topic_model, topics, probs


def analyze_topics(topic_model, topics, texts, file_names=None, sources=None):
    """分析主题并生成统计信息，支持中方西方对比"""
    print("\n分析主题...")
    
    # 获取主题信息
    try:
        topic_info = topic_model.get_topic_info()
    except Exception as e:
        print(f"警告: 无法获取主题信息: {e}")
        topic_info = pd.DataFrame()
    
    # 将 topics 转换为列表（如果是 numpy array）
    if isinstance(topics, np.ndarray):
        topics = topics.tolist()
    
    # 统计每个主题的频率
    topic_counts = pd.Series(topics).value_counts().sort_index()
    
    # 创建结果DataFrame
    results = []
    
    # 获取所有主题ID（从 topic_info 或从 topics 中提取）
    if not topic_info.empty and 'Topic' in topic_info.columns:
        topic_ids = topic_info['Topic'].values
    else:
        # 如果没有 topic_info，从 topics 中提取唯一值
        topic_ids = sorted([t for t in set(topics) if t != -1])
    
    for topic_id in topic_ids:
        if topic_id == -1:
            continue  # 噪声主题稍后单独处理
        
        # 获取主题的关键词和 c-TF-IDF 分数
        try:
            topic_words = topic_model.get_topic(topic_id)
            if topic_words and len(topic_words) > 0:
                # 暂时不使用NER后处理过滤，只使用BERTopic的stop_words过滤
                # 这样可以确保没有被误删
                # 保存关键词（前10个）
                keywords = ', '.join([word for word, _ in topic_words[:10]])
                # 保存关键词和分数的完整信息（用于详细分析）
                keywords_with_scores = '; '.join([f"{word}({score:.4f})" for word, score in topic_words[:10]])
            else:
                keywords = "N/A"
                keywords_with_scores = "N/A"
        except Exception as e:
            print(f"警告: 无法获取主题 {topic_id} 的关键词: {e}")
            keywords = "N/A"
            keywords_with_scores = "N/A"
        
        # 统计该主题的文档数
        count = int(topic_counts.get(topic_id, 0))
        percentage = (count / len(texts)) * 100 if len(texts) > 0 else 0
        
        # 找出包含该主题的文件
        topic_indices = [i for i, t in enumerate(topics) if t == topic_id]
        if file_names and len(topic_indices) > 0:
            files = [file_names[i] for i in topic_indices]
            unique_files = list(set(files))
            file_list = ', '.join(unique_files[:5])  # 最多显示5个文件
            if len(unique_files) > 5:
                file_list += f" ... (共{len(unique_files)}个文件)"
        else:
            file_list = "N/A"
        
        # 统计该主题在不同来源中的分布（中方西方对比）
        source_distribution = "N/A"
        if sources and len(topic_indices) > 0:
            topic_sources = [sources[i] for i in topic_indices]
            source_counts = pd.Series(topic_sources).value_counts()
            source_distribution = ', '.join([f"{source}: {count}" for source, count in source_counts.items()])
        
        results.append({
            'topic_id': topic_id,
            'topic_keywords': keywords,
            'topic_keywords_with_scores': keywords_with_scores,  # 包含 c-TF-IDF 分数
            'frequency': count,
            'percentage': round(percentage, 2),
            'source_distribution': source_distribution,
            'files': file_list
        })
    
    # 添加噪声主题（如果存在）
    if -1 in topics:
        noise_count = int(topic_counts.get(-1, 0))
        noise_indices = [i for i, t in enumerate(topics) if t == -1]
        noise_source_dist = "N/A"
        if sources and len(noise_indices) > 0:
            noise_sources = [sources[i] for i in noise_indices]
            noise_source_counts = pd.Series(noise_sources).value_counts()
            noise_source_dist = ', '.join([f"{source}: {count}" for source, count in noise_source_counts.items()])
        
        results.append({
            'topic_id': -1,
            'topic_keywords': 'Noise/Outlier (未分类文档)',
            'topic_keywords_with_scores': 'N/A',  # 噪声主题没有关键词
            'frequency': noise_count,
            'percentage': round((noise_count / len(texts)) * 100, 2) if len(texts) > 0 else 0,
            'source_distribution': noise_source_dist,
            'files': 'N/A'
        })
    
    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results = df_results.sort_values('frequency', ascending=False)
    
    return df_results, topic_info


def visualize_topics(topic_model, topics, output_dir="topic_analysis_output", sources=None, texts=None, timestamps=None):
    """可视化主题"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n生成可视化图表...")
    
    # 计算实际主题数量（用于限制可视化）
    unique_topics = set(topics) if isinstance(topics, (list, np.ndarray)) else set(topics.tolist())
    num_topics_actual = len(unique_topics) - (1 if -1 in unique_topics else 0)
    top_n = min(20, num_topics_actual) if num_topics_actual > 0 else 10
    
    try:
        # Topic Keyword Barchart
        fig = topic_model.visualize_barchart(top_n_topics=top_n, n_words=10)
        if fig is not None:
            fig.update_layout(title="Topic Keywords (c-TF-IDF Scores)")
            fig.write_html(os.path.join(output_dir, "topic_barchart.html"))
            print(f"  ✓ Saved topic barchart: {output_dir}/topic_barchart.html")
        else:
            print(f"  Warning: topic barchart returned None")
    except Exception as e:
        print(f"  Warning: cannot generate topic barchart: {e}")
    
    try:
        # Intertopic Distance Map
        fig = topic_model.visualize_topics()
        if fig is not None:
            fig.update_layout(title="Intertopic Distance Map")
            fig.write_html(os.path.join(output_dir, "topic_distance.html"))
            print(f"  ✓ Saved intertopic distance map: {output_dir}/topic_distance.html")
        else:
            print(f"  Warning: intertopic distance map returned None")
    except Exception as e:
        print(f"  Warning: cannot generate intertopic distance map: {e}")
    
    try:
        # Topic Hierarchy (Dendrogram)
        fig = topic_model.visualize_hierarchy()
        if fig is not None:
            fig.update_layout(title="Hierarchical Topic Clustering")
            fig.write_html(os.path.join(output_dir, "topic_hierarchy.html"))
            print(f"  ✓ Saved topic hierarchy: {output_dir}/topic_hierarchy.html")
        else:
            print(f"  Warning: topic hierarchy returned None")
    except Exception as e:
        print(f"  Warning: cannot generate topic hierarchy: {e}")
    
    # 主题相似度矩阵 (Topic Similarity Matrix Heatmap)
    try:
        fig = topic_model.visualize_heatmap(top_n_topics=top_n)
        if fig is not None:
            fig.update_layout(title="Topic Similarity Matrix")
            fig.write_html(os.path.join(output_dir, "topic_similarity_matrix.html"))
            print(f"  ✓ Saved topic similarity matrix: {output_dir}/topic_similarity_matrix.html")
        else:
            print(f"  Warning: topic similarity matrix returned None")
    except Exception as e:
        print(f"  Warning: cannot generate topic similarity matrix: {e}")

    # 文档与主题空间分布 (Documents and Topics UMAP scatter plot)
    try:
        fig = topic_model.visualize_documents(
            texts, topics=topics, hide_document_hover=True
        )
        if fig is not None:
            # 重新添加悬停信息：只显示主题归属，不显示文本内容
            for trace in fig.data:
                if trace.name and trace.name != "":
                    trace.hovertemplate = f"<b>{trace.name}</b><extra></extra>"
            fig.update_layout(
                title="Documents and Topics (UMAP Projection)",
                xaxis_title="UMAP Dimension 1",
                yaxis_title="UMAP Dimension 2",
            )
            fig.write_html(os.path.join(output_dir, "documents_and_topics.html"))
            print(f"  ✓ Saved documents & topics plot: {output_dir}/documents_and_topics.html")
        else:
            print(f"  Warning: documents & topics plot returned None")
    except Exception as e:
        print(f"  Warning: cannot generate documents & topics plot: {e}")

    # 动态主题演化图 (Topics over Time)
    if timestamps is not None:
        try:
            import pandas as _pd
            # 将年份转为字符串时间戳（BERTopic 需要的格式）
            # 过滤掉 NaN 时间戳的文档
            valid_mask = [_pd.notna(t) for t in timestamps]
            valid_count = sum(valid_mask)
            
            if valid_count > 0:
                valid_texts = [text for text, v in zip(texts, valid_mask) if v]
                valid_topics = [topic for topic, v in zip(topics, valid_mask) if v]
                # 直接转为 pandas Timestamp 对象，避免 BERTopic 内部调用
                # pd.to_datetime(..., infer_datetime_format=True) 在 pandas>=2.0 中已移除该参数
                valid_timestamps = [
                    _pd.Timestamp(year=int(t), month=1, day=1)
                    for t, v in zip(timestamps, valid_mask) if v
                ]
                
                print(f"\n生成动态主题演化图（{valid_count} 个有时间戳的文档）...")
                
                # 兼容性修补：pandas>=2.0 移除了 infer_datetime_format 参数
                # BERTopic 内部仍使用该参数，需临时修补 pd.to_datetime
                _original_to_datetime = _pd.to_datetime
                def _patched_to_datetime(*args, **kwargs):
                    kwargs.pop('infer_datetime_format', None)
                    return _original_to_datetime(*args, **kwargs)
                _pd.to_datetime = _patched_to_datetime
                
                try:
                    topics_over_time = topic_model.topics_over_time(
                        valid_texts,
                        valid_timestamps,
                        nr_bins=None,           # 不合并时间桶，按原始年份
                        evolution_tuning=True,  # 启用演化调优
                        global_tuning=True      # 启用全局调优
                    )
                finally:
                    # 恢复原始 pd.to_datetime
                    _pd.to_datetime = _original_to_datetime
                
                if topics_over_time is not None and len(topics_over_time) > 0:
                    # 获取频率最高的 top N 主题用于可视化
                    from collections import Counter
                    topic_counts = Counter(valid_topics)
                    # 排除噪声主题 -1，取前20个主题
                    top_topics_list = [
                        t for t, _ in topic_counts.most_common()
                        if t != -1
                    ][:20]
                    
                    fig = topic_model.visualize_topics_over_time(
                        topics_over_time,
                        topics=top_topics_list,
                        normalize_frequency=False,  # 使用实际频率（文档数量）
                    )
                    if fig is not None:
                        # 自定义悬停信息：Topic（ID_关键词）、Timestamp（年份）、Frequency（文档数量）
                        # 构建主题ID到标签的映射（格式：0_keyword1_keyword2_keyword3_keyword4）
                        _topic_info = topic_model.get_topic_info()
                        _topic_label_map = {}
                        for _, _row in _topic_info.iterrows():
                            _tid = _row['Topic']
                            if _tid == -1:
                                continue
                            _tw = topic_model.get_topic(_tid)
                            if _tw and len(_tw) > 0:
                                _top_kw = '_'.join([w for w, _ in _tw[:4]])
                                _topic_label_map[_tid] = f"{_tid}_{_top_kw}"
                            else:
                                _topic_label_map[_tid] = f"Topic_{_tid}"
                        
                        # 更新每条 trace 的 name 和 hovertemplate
                        for trace in fig.data:
                            # 尝试从 trace.name 中提取主题ID，替换为完整标签
                            _trace_name = trace.name if trace.name else ""
                            # BERTopic trace.name 可能是 "0_word1_word2..." 或 "Topic 0" 等格式
                            # 尝试提取数字ID
                            _matched_tid = None
                            for _tid_key in _topic_label_map:
                                # 检查 trace.name 是否以该主题ID开头
                                if _trace_name.startswith(f"{_tid_key}_") or _trace_name == str(_tid_key):
                                    _matched_tid = _tid_key
                                    break
                                # 也检查 "Topic N" 格式
                                if _trace_name.strip() == f"Topic {_tid_key}":
                                    _matched_tid = _tid_key
                                    break
                            
                            if _matched_tid is not None:
                                trace.name = _topic_label_map[_matched_tid]
                            
                            # 设置自定义 hovertemplate
                            # %{fullData.name} = 主题标签, %{x|%Y} = 年份, %{y} = 频率
                            trace.hovertemplate = (
                                "<b>Topic</b>: %{fullData.name}<br>"
                                "<b>Timestamp</b>: %{x|%Y}<br>"
                                "<b>Frequency</b>: %{y}"
                                "<extra></extra>"
                            )
                        
                        fig.update_layout(
                            title="Topics over Time (Publication/Completion Year)",
                            xaxis_title="Year",
                            yaxis_title="Frequency",
                        )
                        fig.write_html(os.path.join(output_dir, "topics_over_time.html"))
                        print(f"  ✓ Saved topics over time: {output_dir}/topics_over_time.html")
                    else:
                        print(f"  Warning: topics over time plot returned None")
                    
                    # 同时保存 topics_over_time 数据为 CSV
                    tot_csv_path = os.path.join(output_dir, "topics_over_time.csv")
                    topics_over_time.to_csv(tot_csv_path, index=False, encoding='utf-8-sig')
                    print(f"  ✓ Saved topics over time data: {tot_csv_path}")
                else:
                    print(f"  Warning: topics_over_time returned empty result")
            else:
                print("  跳过动态主题演化图：没有有效时间戳")
        except Exception as e:
            print(f"  Warning: cannot generate topics over time: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n  跳过动态主题演化图：无时间戳数据（运行 add_timestamps.py 可添加）")

    # 生成 c-TF-IDF 值可视化
    try:
        visualize_ctfidf_scores(topic_model, topics, output_dir)
    except Exception as e:
        print(f"  警告: 无法生成 c-TF-IDF 可视化: {e}")


def visualize_ctfidf_scores(topic_model, topics, output_dir="topic_analysis_output", top_n_topics=20, n_words=15):
    """可视化每个主题的 c-TF-IDF 分数"""
    print("\n生成 c-TF-IDF 分数可视化...")
    
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # 获取所有主题ID（排除噪声主题 -1）
        unique_topics = sorted([t for t in set(topics) if t != -1])
        
        if len(unique_topics) == 0:
            print("  跳过 c-TF-IDF 可视化：没有有效主题")
            return
        
        # 限制主题数量
        topics_to_plot = unique_topics[:top_n_topics]
        
        # 为每个主题创建子图
        fig = make_subplots(
            rows=len(topics_to_plot), 
            cols=1,
            subplot_titles=[f"Topic {tid}" for tid in topics_to_plot],
            vertical_spacing=0.02,
            shared_xaxes=True
        )
        
        # 使用颜色映射，确保每个主题都有清晰的颜色
        import plotly.colors as pc
        # 使用不同的颜色方案，确保对比度
        color_palette = pc.qualitative.Set3 + pc.qualitative.Pastel + pc.qualitative.Dark2
        
        for idx, topic_id in enumerate(topics_to_plot, 1):
            try:
                # 获取主题的关键词和 c-TF-IDF 分数
                topic_words = topic_model.get_topic(topic_id)
                if topic_words and len(topic_words) > 0:
                    # 取前 n_words 个词
                    words_data = topic_words[:n_words]
                    words = [word for word, _ in words_data]
                    scores = [score for _, score in words_data]
                    
                    # 使用颜色映射，循环使用颜色方案，确保每个主题都有清晰的颜色
                    color = color_palette[(idx - 1) % len(color_palette)]
                    
                    # 创建条形图
                    fig.add_trace(
                        go.Bar(
                            x=scores,
                            y=words,
                            orientation='h',
                            name=f"Topic {topic_id}",
                            showlegend=False,
                            marker=dict(
                                color=color,
                                opacity=0.8,
                                line=dict(color='rgba(0,0,0,0.3)', width=0.5)  # 添加边框增强对比度
                            )
                        ),
                        row=idx, col=1
                    )
                    
                    # 设置 y 轴标签
                    fig.update_yaxes(
                        title_text="Keywords",
                        row=idx, col=1,
                        autorange="reversed"  # 反转顺序，让分数高的在上面
                    )
            except Exception as e:
                print(f"  警告: 无法获取主题 {topic_id} 的数据: {e}")
                continue
        
        # 设置 x 轴标签（只在最后一个子图显示）
        fig.update_xaxes(title_text="c-TF-IDF Score", row=len(topics_to_plot), col=1)
        
        # 更新布局
        fig.update_layout(
            title={
                'text': f'Topic Keywords c-TF-IDF Scores (Top {len(topics_to_plot)} Topics)',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            height=100 * len(topics_to_plot) + 200,  # 动态调整高度
            width=1000,
            template='plotly_white'
        )
        
        # 保存为HTML
        output_path = os.path.join(output_dir, "ctfidf_scores.html")
        fig.write_html(output_path)
        print(f"  ✓ 保存 c-TF-IDF 分数图: {output_path}")
        
    except ImportError:
        # 如果没有 plotly，使用 matplotlib
        try:
            # 获取所有主题ID（排除噪声主题 -1）
            unique_topics = sorted([t for t in set(topics) if t != -1])
            topics_to_plot = unique_topics[:top_n_topics]
            
            # 计算需要的子图数量
            n_topics = len(topics_to_plot)
            n_cols = 2
            n_rows = (n_topics + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
            if n_topics == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            for idx, topic_id in enumerate(topics_to_plot):
                try:
                    topic_words = topic_model.get_topic(topic_id)
                    if topic_words and len(topic_words) > 0:
                        words_data = topic_words[:n_words]
                        words = [word for word, _ in words_data]
                        scores = [score for _, score in words_data]
                        
                        ax = axes[idx]
                        # 使用颜色映射，确保每个主题都有清晰的颜色
                        colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                                      '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
                                      '#c49c94', '#f7b6d3', '#c7c7c7', '#dbdb8d', '#9edae5']
                        # 循环使用颜色，确保每个主题都有清晰的颜色
                        color = colors_list[idx % len(colors_list)]
                        ax.barh(words, scores, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
                        ax.set_xlabel('c-TF-IDF Score')
                        ax.set_ylabel('Keywords')
                        ax.set_title(f'Topic {topic_id}', fontweight='bold')
                        ax.invert_yaxis()  # 反转顺序
                        ax.grid(axis='x', alpha=0.3)  # 添加网格线增强可读性
                except Exception as e:
                    continue
            
            # 隐藏多余的子图
            for idx in range(len(topics_to_plot), len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            output_path = os.path.join(output_dir, "ctfidf_scores.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ 保存 c-TF-IDF 分数图: {output_path}")
        except Exception as e:
            print(f"  警告: 无法生成 c-TF-IDF 可视化: {e}")


def main():
    parser = argparse.ArgumentParser(description='主题建模 - 提取小说中的主题并统计频率')
    parser.add_argument('--input_csv', type=str, default='bert_training_dataset.csv',
                        help='输入CSV文件路径（默认: bert_training_dataset.csv）')
    parser.add_argument('--input_dir', type=str, default=None,
                        help='输入目录（包含多个txt文件）')
    parser.add_argument('--input_file', type=str, default=None,
                        help='输入文件（单个txt文件）')
    parser.add_argument('--output', type=str, default='topic_analysis.csv',
                        help='输出CSV文件路径（默认: topic_analysis.csv）')
    parser.add_argument('--num_topics', type=int, default=None,
                        help='主题数量（默认: 自动确定）')
    parser.add_argument('--min_topic_size', type=int, default=None,
                        help='最小主题大小（默认: 根据数据量自动推断）')
    parser.add_argument('--chunk_size', type=int, default=500,
                        help='文本块大小（默认: 500词）')
    parser.add_argument('--output_dir', type=str, default='topic_analysis_output',
                        help='输出目录（用于保存可视化，默认: topic_analysis_output）')
    parser.add_argument('--no_stop_words', action='store_true', default=False,
                        help='不使用停用词过滤（默认: 使用硬编码的停用词列表）')
    
    args = parser.parse_args()
    
    # 加载文本（优先使用CSV文件）
    sources = None
    timestamps = None
    if args.input_csv and os.path.exists(args.input_csv):
        try:
            texts, file_names, sources, timestamps = load_texts_from_csv(args.input_csv)
        except Exception as e:
            print(f"错误: 无法从CSV加载数据: {e}")
            return
    elif args.input_file:
        if not os.path.exists(args.input_file):
            print(f"错误: 文件不存在 {args.input_file}")
            return
        texts, file_names, sources = load_text_from_file(args.input_file)
    elif args.input_dir:
        if not os.path.exists(args.input_dir):
            print(f"错误: 目录不存在 {args.input_dir}")
            return
        texts, file_names, sources = load_texts_from_directory(args.input_dir)
    else:
        # 默认尝试使用 bert_training_dataset.csv
        if os.path.exists('bert_training_dataset.csv'):
            print("使用默认CSV文件: bert_training_dataset.csv")
            try:
                texts, file_names, sources, timestamps = load_texts_from_csv('bert_training_dataset.csv')
            except Exception as e:
                print(f"错误: 无法从CSV加载数据: {e}")
                return
        else:
            print("错误: 请指定 --input_csv, --input_dir 或 --input_file")
            parser.print_help()
            return
    
    if len(texts) == 0:
        print("错误: 没有加载到任何文本")
        return
    
    # 如果文本太长，进行分割
    if args.chunk_size > 0:
        print(f"\n检查文本长度（块大小: {args.chunk_size}词）...")
        new_texts = []
        new_file_names = []
        new_timestamps = [] if timestamps else None
        for i, (text, file_name) in enumerate(zip(texts, file_names)):
            words = text.split()
            if len(words) > args.chunk_size * 2:  # 如果文本太长，进行分割
                chunks = segment_text(text, args.chunk_size)
                new_texts.extend(chunks)
                new_file_names.extend([file_name] * len(chunks))
                if new_timestamps is not None:
                    new_timestamps.extend([timestamps[i]] * len(chunks))
            else:
                new_texts.append(text)
                new_file_names.append(file_name)
                if new_timestamps is not None:
                    new_timestamps.append(timestamps[i])
        texts = new_texts
        file_names = new_file_names
        if new_timestamps is not None:
            timestamps = new_timestamps
        print(f"分割后文本数量: {len(texts)}")
    
    # 自动推断 min_topic_size（如果用户未指定）
    if args.min_topic_size is None:
        n = len(texts)
        if n >= 50000:
            auto_min_size = 150
        elif n >= 20000:
            auto_min_size = 100
        elif n >= 5000:
            auto_min_size = 50
        else:
            auto_min_size = 10
        print(f"\n自动推断 min_topic_size = {auto_min_size}（基于 {n} 个文本片段）")
        print(f"  如需调整，请使用 --min_topic_size 参数")
        args.min_topic_size = auto_min_size

    # 执行主题建模
    topic_model, topics, probs = perform_topic_modeling(
        texts, 
        num_topics=args.num_topics,
        min_topic_size=args.min_topic_size,
        use_custom_stopwords=not args.no_stop_words
    )
    
    # 分析主题
    df_results, topic_info = analyze_topics(topic_model, topics, texts, file_names, sources)
    
    # 保存结果
    df_results.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f"\n主题分析结果已保存到: {args.output}")
    
    # 打印摘要
    print("\n" + "="*80)
    print("主题频率统计")
    print("="*80)
    print(df_results.to_string(index=False))
    print("="*80)
    
    # 生成可视化
    visualize_topics(topic_model, topics, args.output_dir, sources, texts, timestamps)
    
    # 保存详细的主题信息
    topic_info_path = args.output.replace('.csv', '_detailed_topics.csv')
    topic_info.to_csv(topic_info_path, index=False, encoding='utf-8-sig')
    print(f"\n详细主题信息已保存到: {topic_info_path}")
    
    # 保存每个文档的主题分配
    doc_topics_path = args.output.replace('.csv', '_document_topics.csv')
    
    # 处理 probs（可能是 None 或 numpy array）
    if probs is not None:
        if isinstance(probs, np.ndarray):
            # 如果是多维数组，取最大概率
            if len(probs.shape) > 1:
                topic_probs = probs.max(axis=1).tolist() if probs.shape[1] > 0 else [None] * len(texts)
            else:
                topic_probs = probs.tolist()
        else:
            topic_probs = probs
    else:
        topic_probs = [None] * len(texts)
    
    # 确保 topics 是列表
    if isinstance(topics, np.ndarray):
        topics_list = topics.tolist()
    else:
        topics_list = list(topics)
    
    doc_df_dict = {
        'document_id': range(len(texts)),
        'text_preview': [t[:200] + '...' if len(t) > 200 else t for t in texts],  # 预览（用于显示）
        'text': texts,  # 完整文本（用于情感分析等后续处理）
        'topic_id': topics_list,
        'topic_probability': topic_probs,
        'source_file': file_names
    }
    
    # 如果有来源信息，添加到文档主题分配表中
    if sources:
        doc_df_dict['source'] = sources
    
    # 如果有时间戳信息，添加到文档主题分配表中
    if timestamps:
        doc_df_dict['timestamp'] = timestamps
    
    doc_df = pd.DataFrame(doc_df_dict)
    doc_df.to_csv(doc_topics_path, index=False, encoding='utf-8-sig')
    print(f"文档主题分配已保存到: {doc_topics_path}")


if __name__ == "__main__":
    main()