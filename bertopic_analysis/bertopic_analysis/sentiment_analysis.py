"""
脚本名称: sentiment_analysis.py
功能: 实现"细粒度情感量化分析"
核心模型: RoBERTa-base-sentiment
输入: topic_modeling.py 生成的 _document_topics.csv
输出: 带有情感得分的 CSV 和 情感-主题象限图

主要功能:
1. 使用 RoBERTa-base-sentiment 模型为每段文本计算情感极性分（-1 到 1 之间）
2. 计算每个主题的情感均分（加权平均或简单平均）
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.express as px
import os
import warnings
warnings.filterwarnings('ignore')

# ================= 配置区域 =================
# 输入文件：topic_modeling.py 生成的文档-主题 CSV 文件
INPUT_FILE = "topic_analysis_document_topics.csv"  
OUTPUT_FILE = "topic_sentiment_final.csv"
# 使用本地 RoBERTa-base-sentiment 模型
MODEL_NAME = "/root/autodl-tmp/clean/roberta-base-sentiment-local"
# 批处理大小（24GB GPU 显存，可以设置较大值以提高速度）
BATCH_SIZE = 32
# 是否保存中间结果文件（文档级情感得分），设为False可节省空间
SAVE_TEMP_FILE = False  # 默认不保存，如需调试可设为True

# ================= 1. 初始化情感模型 =================
def init_sentiment_model():
    """
    初始化 RoBERTa-base-sentiment 模型
    返回: tokenizer, model, device
    """
    print(f"正在加载情感模型: {MODEL_NAME} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        
        # 检查 GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()  # 设置为评估模式
        print(f"✓ 模型加载成功，运行设备: {device}")
        if device == "cuda":
            print(f"  GPU 型号: {torch.cuda.get_device_name(0)}")
        return tokenizer, model, device
    except Exception as e:
        print(f"  模型加载失败: {e}")
        print("请尝试运行: pip install transformers torch scipy")
        raise

# ================= 2. 计算单条文本情感 =================
def get_sentiment_score(text, tokenizer, model, device):
    """
    输入文本，返回归一化的情感极性分 [-1, 1]
    
    RoBERTa-base-sentiment 输出三个 logit: [Negative, Neutral, Positive]
    计算逻辑: 情感极性分 = Prob(Positive) - Prob(Negative)
    确保结果严格在 -1 到 1 之间
    
    Args:
        text: 输入文本
        tokenizer: 分词器
        model: 情感分析模型
        device: 运行设备
    
    Returns:
        float: 情感极性分，范围 [-1, 1]
            -1: 极负面
            0: 中性
            1: 极正面
    """
    if not text or pd.isna(text) or str(text).strip() == "":
        return 0.0  # 空文本返回中性分数
    
    try:
        # 截断过长的文本以适应 RoBERTa (max_length=512)
        encoded_input = tokenizer(
            str(text), 
            return_tensors='pt', 
            truncation=True, 
            max_length=512,
            padding=True
        ).to(device)
        
        with torch.no_grad():
            output = model(**encoded_input)
        
        scores = output.logits[0].cpu().numpy()
        probs = softmax(scores)  # 转换为概率分布
        
        # RoBERTa-base-sentiment 的标签顺序: ['negative', 'neutral', 'positive']
        # probs[0] = negative, probs[1] = neutral, probs[2] = positive
        
        # 计算情感极性分：正向概率减去负向概率
        # 结果范围在 -1 (极负) 到 1 (极正) 之间
        sentiment_polarity = probs[2] - probs[0]
        
        # 确保结果在 [-1, 1] 范围内（理论上已经在范围内，但保险起见）
        sentiment_polarity = np.clip(sentiment_polarity, -1.0, 1.0)
        
        return float(sentiment_polarity)
    except Exception as e:
        print(f"警告: 处理文本时出错: {e}")
        return 0.0  # 出错时返回中性分数

# ================= 2.1 批量计算情感得分（提高效率）=================
def get_sentiment_scores_batch(texts, tokenizer, model, device, batch_size=BATCH_SIZE):
    """
    批量计算情感得分，提高处理效率
    
    Args:
        texts: 文本列表
        tokenizer: 分词器
        model: 情感分析模型
        device: 运行设备
        batch_size: 批处理大小
    
    Returns:
        list: 情感得分列表
    """
    scores = []
    
    # 过滤空文本
    valid_texts = []
    valid_indices = []
    for i, text in enumerate(texts):
        if text and not pd.isna(text) and str(text).strip() != "":
            valid_texts.append(str(text))
            valid_indices.append(i)
        else:
            scores.append(0.0)
    
    # 批量处理
    for i in tqdm(range(0, len(valid_texts), batch_size), desc="批量计算情感得分"):
        batch_texts = valid_texts[i:i+batch_size]
        
        try:
            # 批量编码
            encoded_input = tokenizer(
                batch_texts,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**encoded_input)
            
            # 处理每个结果
            batch_scores = outputs.logits.cpu().numpy()
            for score in batch_scores:
                probs = softmax(score)
                sentiment_polarity = probs[2] - probs[0]  # positive - negative
                sentiment_polarity = np.clip(sentiment_polarity, -1.0, 1.0)
                scores.append(float(sentiment_polarity))
        except Exception as e:
            print(f"警告: 批处理出错: {e}，改用单条处理")
            # 如果批处理失败，回退到单条处理
            for text in batch_texts:
                score = get_sentiment_score(text, tokenizer, model, device)
                scores.append(score)
    
    # 将结果映射回原始索引
    result = [0.0] * len(texts)
    for idx, score in zip(valid_indices, scores):
        result[idx] = score
    
    return result

# ================= 3. 批量处理与计算 =================
def main():
    # 1. 读取数据
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 找不到输入文件 {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"加载数据成功，共 {len(df)} 条文本片段")
    print(f"  列名: {df.columns.tolist()}")

    # 标准化列名（适配不同的列名格式）
    # 优先使用完整文本列：'text'（完整文本）> 'Document' > 'text_preview'（预览，可能截断）
    doc_column = None
    # 首先检查是否有完整文本列 'text'
    if 'text' in df.columns:
        doc_column = 'text'
        print("  ✓ 检测到完整文本列 'text'，将使用完整文本进行情感分析")
    else:
        # 如果没有 'text'，按优先级查找其他列
        for col in ['Document', 'document', 'Document_Text', 'text_preview']:
            if col in df.columns:
                doc_column = col
                if col == 'text_preview':
                    print("    警告: 只找到 'text_preview' 列（可能是截断的预览文本）")
                    print("     建议: 重新运行 topic_modeling.py 以生成包含完整 'text' 列的文件")
                    # 检查是否真的是截断的
                    sample_text = str(df[col].iloc[0]) if len(df) > 0 else ""
                    if len(sample_text) <= 203 and sample_text.endswith('...'):
                        print("     确认: text_preview 确实是截断的，情感分析可能不准确")
                break
    
    if doc_column is None:
        print("错误: 找不到文档内容列，请确保 CSV 文件包含以下列名之一:")
        print("  text (完整文本，推荐), Document, text_preview, document, Document_Text")
        return
    
    # 主题ID列：可能是 'Topic', 'topic_id', 'Topic_ID' 等
    topic_column = None
    for col in ['Topic', 'topic_id', 'Topic_ID', 'topic']:
        if col in df.columns:
            topic_column = col
            break
    
    if topic_column is None:
        print("错误: 找不到主题ID列，请确保 CSV 文件包含以下列名之一:")
        print("  Topic, topic_id, Topic_ID, topic")
        return
    
    # 统一列名为标准格式
    df = df.rename(columns={
        doc_column: 'Document',
        topic_column: 'Topic'
    })
    
    print(f"  使用列: Document (原: {doc_column}), Topic (原: {topic_column})")

    # 2. 加载模型
    tokenizer, model, device = init_sentiment_model()

    # 3. 计算每一行的情感得分
    print("开始计算情感得分 (这可能需要一些时间)...")
    print(f"  总文本数: {len(df)}")
    
    texts = df['Document'].astype(str).tolist()
    
    # 使用批量处理提高效率
    try:
        scores = get_sentiment_scores_batch(texts, tokenizer, model, device, batch_size=BATCH_SIZE)
    except Exception as e:
        print(f"批量处理失败，改用单条处理: {e}")
        scores = []
        for text in tqdm(texts, desc="单条计算情感得分"):
            score = get_sentiment_score(text, tokenizer, model, device)
            scores.append(score)
    
    df['sentiment_score'] = scores
    
    # 打印统计信息
    print(f"\n情感得分统计:")
    print(f"  平均值: {df['sentiment_score'].mean():.4f}")
    print(f"  标准差: {df['sentiment_score'].std():.4f}")
    print(f"  最小值: {df['sentiment_score'].min():.4f}")
    print(f"  最大值: {df['sentiment_score'].max():.4f}")
    print(f"  正面文本 (>0): {(df['sentiment_score'] > 0).sum()} ({(df['sentiment_score'] > 0).sum()/len(df)*100:.2f}%)")
    print(f"  负面文本 (<0): {(df['sentiment_score'] < 0).sum()} ({(df['sentiment_score'] < 0).sum()/len(df)*100:.2f}%)")
    print(f"  中性文本 (=0): {(df['sentiment_score'] == 0).sum()} ({(df['sentiment_score'] == 0).sum()/len(df)*100:.2f}%)")
    
    # 保存中间结果 (防止后面崩了白跑，可选)
    if SAVE_TEMP_FILE:
        df.to_csv("temp_sentiment_docs.csv", index=False, encoding='utf-8-sig')
        print(f"  ✓ 中间结果已保存: temp_sentiment_docs.csv")
    else:
        print(f"    跳过保存中间结果文件（如需保存，请设置 SAVE_TEMP_FILE = True）")

    # ================= 4. 计算主题情感均分 =================
    # 如果存在 topic_probability 列，使用加权平均: E_k = Sum(p * S) / Sum(p)
    # 否则使用简单平均: E_k = Mean(S)
    
    print("\n正在计算主题情感均分...")
    
    topic_stats = []
    
    # 检查是否有 topic_probability 列
    has_prob = 'topic_probability' in df.columns
    if has_prob:
        print("  使用加权平均计算（基于 topic_probability）")
    else:
        print("  使用简单平均计算（未检测到 topic_probability 列）")
        df['topic_probability'] = 1.0

    # 按主题分组计算
    unique_topics = sorted(df['Topic'].unique())
    
    for topic_id in unique_topics:
        if topic_id == -1: 
            continue  # 跳过噪音主题

        subset = df[df['Topic'] == topic_id]
        
        if len(subset) == 0:
            continue
        
        # 计算加权平均情感
        if has_prob:
            # 加权平均: E_k = Sum(p * S) / Sum(p)
            numerator = (subset['topic_probability'] * subset['sentiment_score']).sum()
            denominator = subset['topic_probability'].sum()
            topic_sentiment = numerator / denominator if denominator != 0 else 0.0
        else:
            # 简单平均: E_k = Mean(S)
            topic_sentiment = subset['sentiment_score'].mean()
        
        # 确保结果在 [-1, 1] 范围内
        topic_sentiment = np.clip(topic_sentiment, -1.0, 1.0)
        
        # 统计信息
        freq = len(subset)
        sentiment_std = subset['sentiment_score'].std()  # 情感得分标准差
        
        topic_stats.append({
            "Topic": topic_id,
            "Sentiment_Mean": topic_sentiment,  # 主题情感均分
            "Sentiment_Std": sentiment_std,     # 情感得分标准差
            "Frequency": freq,                  # 该主题的文档数量
            "Positive_Count": (subset['sentiment_score'] > 0).sum(),  # 正面文本数
            "Negative_Count": (subset['sentiment_score'] < 0).sum(),  # 负面文本数
            "Neutral_Count": (subset['sentiment_score'] == 0).sum()   # 中性文本数
        })
    
    stats_df = pd.DataFrame(topic_stats)
    
    # 融合之前的关键词信息 (可选，为了画图好看)
    # 这里我们简单读取一下之前生成的 topic_analysis.csv 如果存在
    try:
        topic_info = pd.read_csv("topic_analysis.csv")
        if 'Topic_ID' in topic_info.columns and 'Keywords' in topic_info.columns:
            stats_df = pd.merge(stats_df, topic_info[['Topic_ID', 'Keywords']], 
                              left_on='Topic', right_on='Topic_ID', how='left')
        elif 'Topic' in topic_info.columns and 'Keywords' in topic_info.columns:
            stats_df = pd.merge(stats_df, topic_info[['Topic', 'Keywords']], 
                              on='Topic', how='left')
        else:
            # topic_analysis.csv 存在但没有 Keywords 列
            stats_df['Keywords'] = "Topic " + stats_df['Topic'].astype(str)
    except Exception as e:
        print(f"  提示: 无法加载 topic_analysis.csv，使用默认关键词: {e}")
        stats_df['Keywords'] = "Topic " + stats_df['Topic'].astype(str)
    
    # 确保 Keywords 列存在（防止合并失败等情况）
    if 'Keywords' not in stats_df.columns:
        stats_df['Keywords'] = "Topic " + stats_df['Topic'].astype(str)

    # 保存最终统计表
    stats_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n✓ 主题情感统计结果已保存: {OUTPUT_FILE}")
    print(f"  共 {len(stats_df)} 个主题")
    
    # 打印主题情感均分统计
    print(f"\n主题情感均分统计:")
    print(f"  平均情感均分: {stats_df['Sentiment_Mean'].mean():.4f}")
    print(f"  最正面主题: Topic {stats_df.loc[stats_df['Sentiment_Mean'].idxmax(), 'Topic']} "
          f"(得分: {stats_df['Sentiment_Mean'].max():.4f})")
    print(f"  最负面主题: Topic {stats_df.loc[stats_df['Sentiment_Mean'].idxmin(), 'Topic']} "
          f"(得分: {stats_df['Sentiment_Mean'].min():.4f})")

    # ================= 5. 绘制情感-主题象限图 =================
    print("\n正在绘制情感-主题象限图...")
    
    # 创建交互式散点图
    fig = px.scatter(
        stats_df,
        x="Frequency",
        y="Sentiment_Mean",
        size="Frequency",           # 点的大小代表热度
        color="Sentiment_Mean",     # 颜色代表情感
        hover_name="Keywords",      # 鼠标悬停显示关键词
        hover_data={
            "Topic": True,
            "Sentiment_Mean": ":.4f",
            "Frequency": True,
            "Positive_Count": True,
            "Negative_Count": True
        },
        color_continuous_scale="RdBu_r",  # 红蓝配色 (红=正, 蓝=负，_r表示反转)
        title="Sentiment-Topic Quadrant",
        labels={
            "Sentiment_Mean": "Topic Sentiment Score (Negative <-> Positive)",
            "Frequency": "Topic Frequency (Number of Documents)"
        }
    )

    # 添加辅助线 (十字象限)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                  annotation_text="Neutral Line", annotation_position="right")
    
    # 计算中位数频率作为参考线
    median_freq = stats_df['Frequency'].median()
    fig.add_vline(x=median_freq, line_dash="dash", line_color="gray",
                  annotation_text="Median Frequency", annotation_position="top")
    
    # 优化布局
    fig.update_layout(
        plot_bgcolor="white",
        width=1000,
        height=700,
        xaxis=dict(
            showgrid=True, 
            gridcolor='lightgray',
            title="Topic Frequency (Number of Documents)"
        ),
        yaxis=dict(
            showgrid=True, 
            gridcolor='lightgray', 
            range=[-1, 1],  # 情感分限制在 -1 到 1
            title="Topic Sentiment Score"
        ),
        coloraxis_colorbar=dict(
            title="Sentiment",
            tickmode='linear',
            tick0=-1,
            dtick=0.5
        )
    )
    
    output_html = "sentiment_quadrant_chart.html"
    fig.write_html(output_html)
    print(f"✓ 情感象限图已生成: {output_html}")
    
    print("\n" + "="*60)
    print("情感分析完成！")
    print("="*60)
    print(f"输出文件:")
    print(f"  1. {OUTPUT_FILE} - 主题情感统计表")
    if SAVE_TEMP_FILE:
        print(f"  2. temp_sentiment_docs.csv - 文档级情感得分（中间结果）")
        print(f"  3. {output_html} - 情感-主题象限图")
    else:
        print(f"  2. {output_html} - 情感-主题象限图")
        print(f"  (中间结果文件未保存，如需保存请设置 SAVE_TEMP_FILE = True)")
    print("="*60)

if __name__ == "__main__":
    main()