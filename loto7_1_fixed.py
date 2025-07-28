import pandas as pd
import re

def parse_number_string_auto(number_str, expected_count):
    if pd.isna(number_str):
        return []
    number_str = str(number_str).strip("[]()\"'")
    parts = re.split(r'[\s,、\t]+', number_str)
    numbers = [int(p) for p in parts if p.isdigit()]
    return numbers if len(numbers) == expected_count else []

def fix_loto7_csv(input_file="loto7_1.csv", output_file="loto7_1_fixed.csv"):
    try:
        df = pd.read_csv(input_file, encoding="utf-8-sig")
    except Exception as e:
        print(f"[ERROR] CSV読み込み失敗: {e}")
        return

    fixed_rows = []
    skipped_count = 0

    for _, row in df.iterrows():
        draw_date = row.get("抽せん日", "")
        main_numbers = parse_number_string_auto(row.get("本数字", ""), 7)
        bonus_numbers = parse_number_string_auto(row.get("ボーナス数字", ""), 2)

        if len(main_numbers) == 7 and len(bonus_numbers) == 2:
            fixed_rows.append({
                "抽せん日": draw_date,
                "本数字": " ".join(f"{n:02d}" for n in main_numbers),
                "ボーナス数字": " ".join(f"{n:02d}" for n in bonus_numbers)
            })
        else:
            skipped_count += 1
            print(f"[SKIP] 不正な形式: 本数字={row.get('本数字')}, ボーナス数字={row.get('ボーナス数字')}")

    fixed_df = pd.DataFrame(fixed_rows)
    fixed_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"[OK] 修正済みCSVを {output_file} に保存しました。スキップ行数: {skipped_count}")

# 実行
fix_loto7_csv()
