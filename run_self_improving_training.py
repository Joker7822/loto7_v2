
def run_self_improving_training(data_path="loto7.csv", cycles=5):
    """
    高一致データと自己予測を利用して、モデルを継続的に進化させる
    """
    import pandas as pd
    import os
    from datetime import datetime
    from lottery_prediction import (
        set_global_seed,
        LotoPredictor,
        verify_predictions,
        save_self_predictions,
        evaluate_prediction_accuracy_with_bonus
    )

    set_global_seed(42)

    data = pd.read_csv(data_path, encoding="utf-8-sig")
    data["抽せん日"] = pd.to_datetime(data["抽せん日"], errors='coerce')
    data = data.sort_values("抽せん日").reset_index(drop=True)

    # 過去予測精度を取得（ある場合）
    accuracy_df = None
    if os.path.exists("loto7_prediction_evaluation_with_bonus.csv"):
        accuracy_df = pd.read_csv("loto7_prediction_evaluation_with_bonus.csv", encoding="utf-8-sig")

    for cycle in range(cycles):
        print(f"\n=== 🔁 強化学習サイクル {cycle+1}/{cycles} ===")

        predictor = LotoPredictor(input_size=50, hidden_size=128, output_size=7)  # 入力次元は仮。正確には前処理後に調整。
        predictor.train_model(data.copy(), accuracy_results=accuracy_df)

        # 最新データ10件から予測
        latest_data = data.tail(10)
        predictions, confidences = predictor.predict(latest_data)

        # 高一致予測だけを再学習用に保存
        verified_predictions = verify_predictions(list(zip(predictions, confidences)), data)

        if verified_predictions:
            save_self_predictions(verified_predictions)
            print(f"[Cycle {cycle+1}] 自己予測 {len(verified_predictions)} 件保存")

        # 評価データ更新
        evaluate_prediction_accuracy_with_bonus("loto7_predictions.csv", "loto7.csv")
