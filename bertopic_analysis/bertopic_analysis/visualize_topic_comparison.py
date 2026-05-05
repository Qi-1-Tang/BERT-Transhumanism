"""
中西方主题占比对比可视化（水平分组柱状图）

独立脚本，直接读取 topic_modeling.py 生成的 CSV 文件，无需重新运行主题建模。

使用方法：
  python visualize_topic_comparison.py
  python visualize_topic_comparison.py --all          # 显示全部主题
  python visualize_topic_comparison.py --top_n 30     # 显示前 30 个主题

可选参数：
  --topic_csv       主题分析结果 CSV（默认: topic_analysis.csv）
  --doc_csv         文档主题分配 CSV（默认: topic_analysis_document_topics.csv）
  --output_dir      输出目录（默认: topic_analysis_output）
  --top_n           展示前 N 个主题（默认: 20，排除噪声主题）
  --all             显示所有主题（忽略 --top_n）
  --sort_by         排序方式: topic_id / diff / frequency（默认: topic_id）
"""

import os
import argparse
import pandas as pd
import numpy as np

try:
    import plotly.graph_objects as go
except ImportError:
    print("错误: 需要安装 plotly")
    print("请运行: pip install plotly")
    exit(1)


def load_data(topic_csv, doc_csv):
    """加载主题分析 CSV 和文档主题分配 CSV"""
    if not os.path.exists(topic_csv):
        raise FileNotFoundError(f"找不到主题分析文件: {topic_csv}")
    if not os.path.exists(doc_csv):
        raise FileNotFoundError(f"找不到文档主题分配文件: {doc_csv}")

    df_topics = pd.read_csv(topic_csv, encoding="utf-8-sig")
    df_docs = pd.read_csv(doc_csv, encoding="utf-8-sig")

    print(f"加载主题分析: {topic_csv}  ({len(df_topics)} 个主题)")
    print(f"加载文档分配: {doc_csv}  ({len(df_docs)} 条记录)")

    if "source" not in df_docs.columns:
        raise ValueError("文档 CSV 中缺少 'source' 列，无法进行中西方对比")

    source_counts = df_docs["source"].value_counts()
    print("数据来源分布:")
    for src, cnt in source_counts.items():
        print(f"  {src}: {cnt} 条")

    return df_topics, df_docs


def compute_group_proportions(df_topics, df_docs, top_n=20, sort_by="diff", show_all=False):
    """计算每个主题在中方 / 西方各自组内的占比"""

    # 排除噪声主题 -1
    df_docs_valid = df_docs[df_docs["topic_id"] != -1].copy()

    # 获取两个来源的名称
    sources = sorted(df_docs_valid["source"].dropna().unique())
    if len(sources) < 2:
        raise ValueError(f"需要至少两个来源，当前只有: {sources}")

    n_valid_topics = df_docs_valid["topic_id"].nunique()
    print(f"\n对比分组: {sources[0]}  vs  {sources[1]}")
    print(f"有效主题数（排除噪声 -1）: {n_valid_topics}")
    print(f"有效文档数（排除噪声 -1）: {len(df_docs_valid)}")

    # 每个来源的文档总数（排除噪声后）
    group_totals = df_docs_valid.groupby("source").size()
    for src in sources:
        print(f"  {src} 有效文档: {group_totals[src]}")

    # 每个来源在每个主题下的文档数
    cross = df_docs_valid.groupby(["topic_id", "source"]).size().unstack(fill_value=0)

    # 转换为组内占比 (%)
    proportions = cross.div(group_totals, axis=1) * 100

    # 确保两个来源列都存在
    for src in sources:
        if src not in proportions.columns:
            proportions[src] = 0.0

    # 构建汇总表
    topic_keywords_map = {}
    if "topic_keywords" in df_topics.columns and "topic_id" in df_topics.columns:
        for _, row in df_topics.iterrows():
            tid = row["topic_id"]
            if tid == -1:
                continue
            kw = str(row["topic_keywords"])
            # 取前 4 个关键词作为短标签
            short_kw = ", ".join([w.strip() for w in kw.split(",")[:4]])
            topic_keywords_map[tid] = short_kw

    records = []
    for tid in proportions.index:
        pct_a = proportions.loc[tid, sources[0]]
        pct_b = proportions.loc[tid, sources[1]]
        kw = topic_keywords_map.get(tid, "")
        label = f"Topic {tid}: {kw}" if kw else f"Topic {tid}"
        records.append({
            "topic_id": tid,
            "label": label,
            sources[0]: round(pct_a, 2),
            sources[1]: round(pct_b, 2),
            "diff": round(abs(pct_a - pct_b), 2),
            "total_freq": int(cross.loc[tid].sum()),
        })

    df_comp = pd.DataFrame(records)

    # 排序（水平柱状图 y 轴从下往上，descending 使 Topic 0 在最顶部）
    if sort_by == "topic_id":
        df_comp = df_comp.sort_values("topic_id", ascending=False)
    elif sort_by == "diff":
        df_comp = df_comp.sort_values("diff", ascending=True)
    else:
        df_comp = df_comp.sort_values("total_freq", ascending=True)

    # 显示全部或只保留 top_n
    if show_all:
        print(f"\n显示全部 {len(df_comp)} 个主题")
    else:
        df_comp = df_comp.tail(top_n)
        print(f"\n显示前 {len(df_comp)} 个主题（共 {n_valid_topics} 个）")

    return df_comp, sources


def plot_grouped_horizontal_bar(df_comp, sources, output_dir):
    """生成分组水平柱状图（Plotly 交互式 HTML）"""

    os.makedirs(output_dir, exist_ok=True)

    src_a, src_b = sources[0], sources[1]

    # 显示友好名称
    display_names = {
        "Chinese_Xianxia": "Chinese Xianxia",
        "Western_SciFi": "Western Sci-Fi",
    }
    name_a = display_names.get(src_a, src_a)
    name_b = display_names.get(src_b, src_b)

    # 颜色方案
    color_a = "rgba(227, 74, 51, 0.82)"   # 中方 - 红/橙
    color_b = "rgba(44, 123, 182, 0.82)"   # 西方 - 蓝

    fig = go.Figure()

    # 西方（先添加，在下层）
    fig.add_trace(go.Bar(
        y=df_comp["label"],
        x=df_comp[src_b],
        name=name_b,
        orientation="h",
        marker=dict(color=color_b, line=dict(color="rgba(44,123,182,1)", width=0.6)),
        text=[f"{v:.1f}%" for v in df_comp[src_b]],
        textposition="outside",
        textfont=dict(size=11),
        hovertemplate=(
            f"<b>{name_b}</b><br>"
            "Topic: %{y}<br>"
            "Proportion: %{x:.2f}%<extra></extra>"
        ),
    ))

    # 中方
    fig.add_trace(go.Bar(
        y=df_comp["label"],
        x=df_comp[src_a],
        name=name_a,
        orientation="h",
        marker=dict(color=color_a, line=dict(color="rgba(227,74,51,1)", width=0.6)),
        text=[f"{v:.1f}%" for v in df_comp[src_a]],
        textposition="outside",
        textfont=dict(size=11),
        hovertemplate=(
            f"<b>{name_a}</b><br>"
            "Topic: %{y}<br>"
            "Proportion: %{x:.2f}%<extra></extra>"
        ),
    ))

    n_topics = len(df_comp)
    if n_topics > 80:
        bar_height = 22
        tick_font_size = 8
    elif n_topics > 40:
        bar_height = 28
        tick_font_size = 9
    else:
        bar_height = max(38, 60 - n_topics)
        tick_font_size = 11
    chart_height = max(600, n_topics * bar_height + 250)

    fig.update_layout(
        title=dict(
            text="Topic Proportion Comparison: Chinese Xianxia vs Western Sci-Fi",
            x=0.5,
            xanchor="center",
            font=dict(size=18),
        ),
        xaxis=dict(
            title="Proportion within Group (%)",
            gridcolor="rgba(200,200,200,0.4)",
            zeroline=True,
            zerolinecolor="rgba(150,150,150,0.5)",
        ),
        yaxis=dict(
            title="",
            tickfont=dict(size=tick_font_size),
            automargin=True,
        ),
        barmode="group",
        bargap=0.25,
        bargroupgap=0.08,
        height=chart_height,
        width=1100,
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=13),
        ),
        margin=dict(l=20, r=80, t=80, b=50),
    )

    output_path = os.path.join(output_dir, "topic_comparison_bar.html")
    fig.write_html(output_path)
    print(f"\n✓ 分组水平柱状图已保存: {output_path}")

    return output_path


def plot_radar(df_comp, sources, output_dir):
    """生成雷达图（主题数 <= 12 时自动生成）"""

    os.makedirs(output_dir, exist_ok=True)

    src_a, src_b = sources[0], sources[1]
    display_names = {
        "Chinese_Xianxia": "Chinese Xianxia",
        "Western_SciFi": "Western Sci-Fi",
    }
    name_a = display_names.get(src_a, src_a)
    name_b = display_names.get(src_b, src_b)

    # 雷达图用简短标签
    short_labels = []
    for label in df_comp["label"]:
        parts = label.split(": ", 1)
        if len(parts) == 2:
            tid_part = parts[0].replace("Topic ", "T")
            kw_part = ", ".join(parts[1].split(", ")[:2])
            short_labels.append(f"{tid_part}: {kw_part}")
        else:
            short_labels.append(label)

    # 闭合雷达图（首尾相连）
    values_a = list(df_comp[src_a]) + [df_comp[src_a].iloc[0]]
    values_b = list(df_comp[src_b]) + [df_comp[src_b].iloc[0]]
    labels = short_labels + [short_labels[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values_a,
        theta=labels,
        fill="toself",
        fillcolor="rgba(227, 74, 51, 0.15)",
        line=dict(color="rgba(227, 74, 51, 0.9)", width=2),
        name=name_a,
        hovertemplate=f"<b>{name_a}</b><br>%{{theta}}<br>%{{r:.2f}}%<extra></extra>",
    ))

    fig.add_trace(go.Scatterpolar(
        r=values_b,
        theta=labels,
        fill="toself",
        fillcolor="rgba(44, 123, 182, 0.15)",
        line=dict(color="rgba(44, 123, 182, 0.9)", width=2),
        name=name_b,
        hovertemplate=f"<b>{name_b}</b><br>%{{theta}}<br>%{{r:.2f}}%<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text="Topic Proportion Radar: Chinese Xianxia vs Western Sci-Fi",
            x=0.5,
            xanchor="center",
            font=dict(size=18),
        ),
        polar=dict(
            radialaxis=dict(
                visible=True,
                ticksuffix="%",
                gridcolor="rgba(200,200,200,0.5)",
            ),
            angularaxis=dict(
                tickfont=dict(size=10),
            ),
        ),
        height=750,
        width=850,
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.12,
            xanchor="center",
            x=0.5,
            font=dict(size=13),
        ),
    )

    output_path = os.path.join(output_dir, "topic_comparison_radar.html")
    fig.write_html(output_path)
    print(f"✓ 雷达图已保存: {output_path}")

    return output_path


def save_comparison_csv(df_comp, sources, output_dir):
    """保存对比数据为 CSV，方便后续引用"""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "topic_comparison_data.csv")
    df_comp.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"✓ 对比数据已保存: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="中西方主题占比对比可视化（水平分组柱状图 + 雷达图）"
    )
    parser.add_argument(
        "--topic_csv", type=str, default="topic_analysis.csv",
        help="主题分析结果 CSV（默认: topic_analysis.csv）",
    )
    parser.add_argument(
        "--doc_csv", type=str, default="topic_analysis_document_topics.csv",
        help="文档主题分配 CSV（默认: topic_analysis_document_topics.csv）",
    )
    parser.add_argument(
        "--output_dir", type=str, default="topic_analysis_output",
        help="输出目录（默认: topic_analysis_output）",
    )
    parser.add_argument(
        "--top_n", type=int, default=20,
        help="展示前 N 个主题（默认: 20）",
    )
    parser.add_argument(
        "--sort_by", type=str, default="topic_id",
        choices=["topic_id", "diff", "frequency"],
        help="排序方式: topic_id=按主题编号, diff=按差异大小, frequency=按总频率（默认: topic_id）",
    )
    parser.add_argument(
        "--all", action="store_true", default=False,
        help="显示所有主题（忽略 --top_n）",
    )
    parser.add_argument(
        "--radar", action="store_true", default=False,
        help="强制生成雷达图（默认仅在主题数 <= 12 时自动生成）",
    )
    args = parser.parse_args()

    # 1. 加载数据
    df_topics, df_docs = load_data(args.topic_csv, args.doc_csv)

    # 2. 计算组内占比
    df_comp, sources = compute_group_proportions(
        df_topics, df_docs, top_n=args.top_n, sort_by=args.sort_by,
        show_all=args.all,
    )

    # 打印对比表
    n_show = len(df_comp)
    display_cols = ["topic_id", "label", sources[0], sources[1], "diff"]
    if args.sort_by == "topic_id":
        df_display = df_comp[display_cols].sort_values("topic_id", ascending=True)
    else:
        df_display = df_comp[display_cols].sort_values("diff", ascending=False)

    print("\n" + "=" * 90)
    print(f"主题占比对比（组内百分比） — 共 {n_show} 个主题")
    print("=" * 90)
    if n_show > 40:
        print("【前 20 个差异最大的主题】")
        print(df_display.head(20).to_string(index=False))
        print(f"\n... 省略中间 {n_show - 40} 个主题（完整数据见 CSV 输出）...\n")
        print("【后 20 个差异最小的主题】")
        print(df_display.tail(20).to_string(index=False))
    else:
        print(df_display.to_string(index=False))
    print("=" * 90)

    # 3. 生成水平分组柱状图
    plot_grouped_horizontal_bar(df_comp, sources, args.output_dir)

    # 4. 雷达图（主题数 <= 12 或用户强制指定）
    n_topics_show = len(df_comp)
    if args.radar or n_topics_show <= 12:
        # 雷达图按频率排序，取 top 12
        df_radar = df_comp.sort_values("total_freq", ascending=False).head(12)
        plot_radar(df_radar, sources, args.output_dir)
    else:
        print(f"\n提示: 当前展示 {n_topics_show} 个主题，雷达图可读性较差，已跳过。")
        print(f"  添加 --radar 参数可强制生成，或用 --top_n 12 减少主题数。")

    # 5. 保存对比数据 CSV
    save_comparison_csv(df_comp, sources, args.output_dir)


if __name__ == "__main__":
    main()