import os
import re
import pandas as pd

# ================= 配置区域 =================

TASKS = [
    {
        "input_folder": "Cleaned_English_V2",
        "output_folder": "Deep_Cleaned_English",
    },
    {
        "input_folder": "Cleaned_Chinese_Trans_V3",
        "output_folder": "Deep_Cleaned_Chinese",
    },
]
REPORT_FILE = "deep_clean_report.csv"


# ================= 深度清洗规则 =================

def is_symbol_only_line(stripped):
    """
    规则1: 判断一行是否只有特殊符号（* ! # | - 等），应被删除。
    例外保留：
      - 纯句点/省略号行（"......." 或 "……"）
      - 含引号的行（对话相关）
    """
    # 含引号 → 不删
    if any(c in stripped for c in '"\u201c\u201d\u2018\u2019\''):
        return False

    # 纯句点/省略号 → 保留
    dots_only = stripped.replace(' ', '')
    if dots_only and all(c in '.\u2026' for c in dots_only):
        return False

    # 如果存在任何字母/数字/中文 → 不是纯符号行
    for c in stripped:
        if c.isalnum():  # Python 3: 英文字母、数字、中文都算
            return False

    return True


def is_watermark_pattern(stripped):
    """
    规则4: 检测水印文字。
    特征：大部分"词"是单个字母（可能带句点），如：
      "w w"  "c o c o"  "B B Y Y. B B Y Y."  "w m w m w .A w .A"
    """
    words = stripped.split()
    if len(words) < 2 or len(stripped) > 50:
        return False
    # 每个词是 可选点+单字母+可选点，如 w  A  .A  Y.
    single_letter = sum(1 for w in words if re.match(r'^\.?[a-zA-Z]\.?$', w))
    return single_letter >= len(words) * 0.7


def deep_clean_line(raw_line):
    """
    对单行执行深度清洗。
    返回 (处理后文本, 删除原因 or None, 是否修复了乱码)。
    """
    line = raw_line
    fixed_fffd = False

    # Step 0: 去掉乱码字符 U+FFFD（只删字符，不删整行）
    if '\ufffd' in line:
        line = line.replace('\ufffd', '')
        fixed_fffd = True

    stripped = line.strip()
    if not stripped:
        return '', 'Empty', fixed_fffd

    # 对话保护：以引号开头的行不做删除判断
    if stripped[0] in '"\u201c\u201d\'':
        return stripped, None, fixed_fffd

    # 规则1: 纯符号/分割线（保留省略号，保留含引号行）
    if is_symbol_only_line(stripped):
        return '', '纯符号/分割线', fixed_fffd

    # 规则2: 页码行 "Page 362"
    if re.match(r'^page\s+\d+\s*$', stripped, re.IGNORECASE):
        return '', '页码(Page XXX)', fixed_fffd

    # 规则4: 水印字母间距 "w w" "c o c o" "B B Y Y."
    if is_watermark_pattern(stripped):
        return '', '水印文字(字母间距)', fixed_fffd

    return stripped, None, fixed_fffd


# ================= 主程序 =================

def main():
    all_audit = []

    for task in TASKS:
        input_folder = task["input_folder"]
        output_folder = task["output_folder"]

        if not os.path.exists(input_folder):
            print(f"跳过：找不到 '{input_folder}'")
            continue

        os.makedirs(output_folder, exist_ok=True)
        files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
        print(f"\n处理 {input_folder}（{len(files)} 个文件）...")

        for file_name in files:
            file_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name)

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_lines = f.readlines()
            except:
                with open(file_path, 'r', encoding='latin-1') as f:
                    raw_lines = f.readlines()

            clean_lines = []

            for idx, raw_line in enumerate(raw_lines):
                original = raw_line.strip()
                if not original:
                    continue  # 空行跳过（输出时用 \n\n 重新分隔）

                cleaned, reason, fixed_fffd = deep_clean_line(raw_line)

                if reason == 'Empty':
                    continue
                elif reason is not None:
                    # 整行被删除
                    all_audit.append({
                        "来源文件夹": input_folder,
                        "文件名": file_name,
                        "行号": idx + 1,
                        "原内容": original[:50],
                        "原因": reason,
                    })
                else:
                    # 行被保留（可能修复了乱码字符）
                    if fixed_fffd:
                        all_audit.append({
                            "来源文件夹": input_folder,
                            "文件名": file_name,
                            "行号": idx + 1,
                            "原内容": original[:50],
                            "原因": "乱码字符(�)清除",
                        })
                    if cleaned:
                        clean_lines.append(cleaned)

            with open(output_path, 'w', encoding='utf-8') as f_out:
                f_out.write("\n\n".join(clean_lines))

    # ---- 输出报告 ----
    if all_audit:
        df = pd.DataFrame(all_audit)
        df.to_csv(REPORT_FILE, index=False, encoding='utf-8-sig')
        print(f"\n深度清洗报告已生成: {REPORT_FILE}")
        print(f"共处理 {len(df)} 条记录。")

        reason_counts = df["原因"].value_counts()
        print("\n--- 删除/修复原因统计 ---")
        for reason, count in reason_counts.items():
            print(f"  {reason}: {count} 条")
    else:
        print("\n没有发现需要深度清洗的内容。")


if __name__ == "__main__":
    main()
