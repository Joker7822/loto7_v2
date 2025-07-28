# === stacking_model.py ===
import numpy as np
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier


def convert_number_list_to_vector(numbers, vector_size=37):
    vec = np.zeros(vector_size)
    for n in numbers:
        if 1 <= n <= vector_size:
            vec[n - 1] = 1
    return vec


def train_stacking_model(lstm_preds, automl_preds, gan_preds, ppo_preds, true_labels):
    """
    各モデルの出力を融合して最終モデルを学習する
    """
    X = []
    y = []

    for lstm, automl, gan_vec, ppo_vec, label in zip(
        lstm_preds, automl_preds, gan_preds, ppo_preds, true_labels
    ):
        lstm_vec = convert_number_list_to_vector(lstm)
        automl_vec = convert_number_list_to_vector(automl)
        label_vec = convert_number_list_to_vector(label)

        fused_input = np.concatenate([lstm_vec, automl_vec, gan_vec, ppo_vec])
        X.append(fused_input)
        y.append(label_vec)

    X = np.array(X)
    y = np.array(y)

    model = MultiOutputClassifier(
        XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.1,
            verbosity=0,
            use_label_encoder=False,
            base_score=0.5  # ★ ここを追加
        )
    )

    model.fit(X, y)

    print("[INFO] スタッキング融合モデルの学習完了")
    return model


def predict_with_stacking(stack_model, lstm, automl, gan_vec, ppo_vec):
    lstm_vec = convert_number_list_to_vector(lstm)
    automl_vec = convert_number_list_to_vector(automl)
    input_vec = np.concatenate([lstm_vec, automl_vec, gan_vec, ppo_vec]).reshape(1, -1)

    pred_proba = stack_model.predict_proba(input_vec)
    score_vec = np.array([p[1] if isinstance(p, np.ndarray) and len(p) > 1 else 0.0 for p in pred_proba])
    top7 = np.argsort(score_vec)[-7:] + 1
    return sorted(top7), score_vec
