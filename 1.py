import os
import re
import pandas as pd
from collections import Counter

# ================= 配置区域 =================

INPUT_FOLDER = "英文西方" # 原 TXT 文件夹
OUTPUT_FOLDER = "Cleaned_English_V2" # 输出到新文件夹
REPORT_FILE = "deleted_report_v2.csv"

# 阈值调整：更宽容
# 只有当一句话长度超过 30 个字，且重复出现超过 20 次时
# 这样能保护 "Yes, sir." 和 "Princess Irulan"
REPEAT_THRESHOLD = 20
MIN_LENGTH_FOR_REPEAT_CHECK = 30


# ================= 判断逻辑 =================

def get_junk_reason_safe(line, global_counter, line_idx=0, total_lines=0):
    line = line.strip()
    if not line: return "Empty Line"
    lower_line = line.lower()

# 如果一行是以引号开头的，说明是人物对话，不判为系统广告
    if line.startswith('“') or line.startswith('"'):
        return None

# --- 1. 严格的垃圾特征 ---

# URL
    if re.search(r'https?://', line) or re.search(r'www\.[a-zA-Z0-9-]+\.', line):
        return "包含网址 (URL)"

    if re.search(r'\b[a-zA-Z0-9-]+\.(com|net|org|io|co|me|sk|info|xyz)\b', lower_line):
        return "包含域名"

    if "please go to" in lower_line and ("read" in lower_line or "chapter" in lower_line):
        return "阅读推广链接"

    if "table of contents" in lower_line:
        return "目录导航"

# 广告
#  "You're reading my mind" 这种正文就不误删了
    if "boxnovel" in lower_line:
        return "BoxNovel广告"
    if "you’re reading" in lower_line and "novel" in lower_line:
        return "阅读提示广告"

    if re.match(r'^(Translator|Editor|Epub|Credits?):\s*', line, re.IGNORECASE):
        return "翻译/编辑名单"
    if re.search(r'\b(Translator|Editor|Epub):\s*[A-Za-z]', line):
        return "翻译/编辑名单"

    if "copyright" in lower_line and ("all rights reserved" in lower_line or "reserved" in lower_line):
        return "版权声明"

    if re.match(r'^Story Description:', line, re.IGNORECASE):
        return "故事描述"
    if re.match(r'^Original Story can be found here:', line, re.IGNORECASE):
        return "原文链接"

    if re.match(r'^["\']', line) and len(line) < 200 and line_idx < 100:
        if any(marker in lower_line for marker in ['outstanding', 'brilliant', 'superb', 'excellent', 'dazzling', 'tautly', 'homage']):
            if any(marker in lower_line for marker in ['times', 'magazine', 'guardian', 'locus', 'sfx', 'review', 'hamilton', 'macleod']):
                return "书评/推荐语"

    if re.search(r'chapter.*release.*push.*back|promote.*new.*book|unlock.*chapter.*coin', lower_line):
        return "章节推广信息"

    if re.search(r'this chapter.*release|final chapter.*release.*tomorrow', lower_line):
        return "章节发布信息"

    clean_chars = re.sub(r'\s+', '', line)
    if len(clean_chars) > 3 and not any(c.isalnum() for c in clean_chars):
        return "纯分割线"

    if re.match(r'^Chapter \d+:', line, re.IGNORECASE) and len(line) < 50:
        if global_counter[line] > 5:
            return f"重复章节标题 ({global_counter[line]}次)"

    if len(line) < MIN_LENGTH_FOR_REPEAT_CHECK:
        return None

    if global_counter[line] > REPEAT_THRESHOLD:
        return f"长句高频重复 ({global_counter[line]}次)"

    if total_lines > 0 and line_idx >= total_lines - 20:
        if re.search(r'^(translator|editor|epub|credit|thoughts?):', lower_line):
            return "结尾致谢/名单"
        if re.search(r'continue.*support|thank.*support|next.*project', lower_line):
            return "结尾推广信息"

    return None


# ================= 主程序 =================

def main():
    if not os.path.exists(INPUT_FOLDER):
        print(f"错误：找不到文件夹 '{INPUT_FOLDER}'")
        return

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    print("Step 1: 全局扫描 (统计重复率)...")
    all_lines_buffer = []
    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.txt')]

    for file_name in files:
        path = os.path.join(INPUT_FOLDER, file_name)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except:
            try:
                with open(path, 'r', encoding='latin-1') as f:
                    lines = f.readlines()
            except:
                continue

        for line in lines:
            line = line.strip()
            if line: all_lines_buffer.append(line)

    global_counter = Counter(all_lines_buffer)
    print(f"统计完成。")


    print("Step 2: 执行安全清洗...")
    audit_records = []

    for file_name in files:
        file_path = os.path.join(INPUT_FOLDER, file_name)
        output_path = os.path.join(OUTPUT_FOLDER, "clean_" + file_name)
        clean_lines = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_lines = f.readlines()
        except:
            with open(file_path, 'r', encoding='latin-1') as f:
                raw_lines = f.readlines()

        total_lines = len(raw_lines)
        for idx, line in enumerate(raw_lines):
            original = line.strip()
            reason = get_junk_reason_safe(original, global_counter, idx, total_lines)

            if reason is None:
                clean_lines.append(original)
            elif reason == "Empty Line":
                continue
            else:
                audit_records.append({
                    "文件名": file_name,
                    "行号": idx + 1,
                    "内容": original[:50],
                    "原因": reason
                })

# 这里的 join "\n\n" 会让段落间更清晰
        with open(output_path, 'w', encoding='utf-8') as f_out:
            f_out.write("\n\n".join(clean_lines))

# ----------------------------------------------------

    if audit_records:
        df = pd.DataFrame(audit_records)
        df.to_csv(REPORT_FILE, index=False, encoding='utf-8-sig')
        print(f"V2 报告已生成: {REPORT_FILE}")
        print(f"文件已保存在: {OUTPUT_FOLDER}/")
        print(f"删除了 {len(df)} 行。请再次检查 CSV 确认 'Yes, sir' 是否已被保护。")
    else:
        print("没有发现需要删除的垃圾。")


if __name__ == "__main__":
    main()