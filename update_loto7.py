import pandas as pd
import subprocess
import shutil
import os

def update_loto7():
    loto7_file = "loto7.csv"
    loto7_1_file = "loto7_1.csv"
    predictions_file = "loto7_predictions.csv"
    temp_file = "loto7_1_temp.csv"

    # 元のデータを読み込み
    df_loto7 = pd.read_csv(loto7_file)
    df_loto7_1 = pd.read_csv(loto7_1_file)

    for i in range(5, len(df_loto7)):  # 6行目 (index 5) から開始
        print(f"Processing row {i+1} from {loto7_file}...")

        # 追加する行
        new_row = df_loto7.iloc[i]
        df_loto7_1 = pd.concat([df_loto7_1, pd.DataFrame([new_row])], ignore_index=True)

        # 一時ファイルに保存
        df_loto7_1.to_csv(temp_file, index=False)

        # 元のloto7_1.csvをバックアップして置き換え
        shutil.move(temp_file, loto7_1_file)

        # test2.py を実行
        try:
            subprocess.run(["python", "test2.py"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] test2.py の実行に失敗しました: {e}")
            break

        # 予測結果の日付を変更
        if os.path.exists(predictions_file):
            df_predictions = pd.read_csv(predictions_file)

            if not df_predictions.empty:
                if i + 1 < len(df_loto7):  # 次の行が存在する場合
                    next_date = df_loto7.iloc[i + 1, 0]
                    df_predictions.iloc[-1, 0] = next_date  # 最終行の日付を更新
                    df_predictions.to_csv(predictions_file, index=False)
                    print(f"Updated {predictions_file} with date from row {i+2} of {loto7_file}.")
                else:
                    print("No next date available to update predictions.")
            else:
                print("[WARNING] df_predictions が空のため日付更新をスキップしました")
        else:
            print(f"[ERROR] {predictions_file} が見つかりません")

if __name__ == "__main__":
    update_loto7()
