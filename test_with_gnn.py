import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor, RandomForestRegressor, StackingRegressor,
    GradientBoostingClassifier, RandomForestClassifier, HistGradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
)
from genetic_algorithm import evolve_candidates
from sklearn.linear_model import Ridge, LogisticRegression, RidgeClassifier, Perceptron, PassiveAggressiveClassifier, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import StackingRegressor
from statsmodels.tsa.arima.model import ARIMA
from stable_baselines3 import PPO
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
import matplotlib.pyplot as plt
import aiohttp
import asyncio
import warnings
import re
import platform
import gym
import sys
import os
import random
from sklearn.metrics import precision_score, recall_score, f1_score
from neuralforecast.models import TFT
from neuralforecast import NeuralForecast
import onnxruntime
import streamlit as st
from autogluon.tabular import TabularPredictor
import torch.backends.cudnn
from datetime import datetime 
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from models.set_transformer import SetTransformer
from stacking_model import (
    train_stacking_model,
    predict_with_stacking,
    convert_number_list_to_vector
)
import traceback  # 上部でインポートしておいてください
import subprocess

# Windows環境のイベントループポリシーを設定
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

warnings.filterwarnings("ignore")

SEED = int(time.time()) % (2**32)  # 動的なシード値
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = False  # 多様性を優先するなら False
torch.backends.cudnn.benchmark = True       # 性能最適化を有効に
def git_commit_and_push(file_path, message):
    try:
        subprocess.run(["git", "add", file_path], check=True)
        diff = subprocess.run(["git", "diff", "--cached", "--quiet"])
        if diff.returncode != 0:
            subprocess.run(["git", "config", "--global", "user.name", "github-actions"], check=True)
            subprocess.run(["git", "config", "--global", "user.email", "github-actions@github.com"], check=True)
            subprocess.run(["git", "commit", "-m", message], check=True)
            subprocess.run(["git", "push"], check=True)
        else:
            print(f"[INFO] No changes in {file_path}")
    except Exception as e:
        print(f"[WARNING] Git commit/push failed: {e}")

# 実行対象ファイル（複数）
targets = [
    "loto7_predictions.csv",
    "loto7_prediction_evaluation_with_bonus.csv",
    "loto7_evaluation_summary.txt",
    "self_predictions.csv"
]

for file in targets:
    git_commit_and_push(file, f"Auto update {file} [skip ci]")

class LotoEnv(gym.Env):
    def __init__(self, historical_numbers):
        super(LotoEnv, self).__init__()
        self.historical_numbers = historical_numbers
        self.action_space = spaces.Box(low=0, high=1, shape=(37,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(37,), dtype=np.float32)

    def reset(self):
        return np.zeros(37, dtype=np.float32)

    def step(self, action):
        action = np.clip(action, 0, 1)
        selected_numbers = np.argsort(action)[-7:] + 1  # 上位7個

        # 直近データからランダムに本物データを選んで「対戦」
        winning_numbers = set(np.random.choice(self.historical_numbers, 7, replace=False))

        main_match = len(set(selected_numbers) & winning_numbers)

        reward = main_match / 7  # 本数字一致数でスコア
        done = True
        obs = np.zeros(37, dtype=np.float32)

        return obs, reward, done, {}

class LotoGAN(nn.Module):
    def __init__(self, noise_dim=100):
        super(LotoGAN, self).__init__()
        self.generator = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 37),
            nn.Sigmoid()
        )
        self.discriminator = nn.Sequential(
            nn.Linear(37, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.noise_dim = noise_dim

    def generate_samples(self, num_samples):
        noise = torch.randn(num_samples, self.noise_dim)
        with torch.no_grad():
            samples = self.generator(noise)
        return samples.numpy()

def create_advanced_features(dataframe):
    import numpy as np
    import pandas as pd
    import re
    from itertools import combinations

    def convert_to_number_list(x):
        if isinstance(x, str):
            cleaned = re.sub(r"[^\d\s]", " ", x)
            return [int(n) for n in cleaned.split() if n.isdigit()]
        elif isinstance(x, list):
            return [int(n) for n in x if isinstance(n, (int, float)) and not pd.isna(n)]
        return []

    # データ前処理
    dataframe['本数字'] = dataframe['本数字'].apply(convert_to_number_list)
    dataframe['ボーナス数字'] = dataframe['ボーナス数字'].apply(convert_to_number_list)
    dataframe['抽せん日'] = pd.to_datetime(dataframe['抽せん日'], errors='coerce')
    dataframe = dataframe.dropna(subset=['抽せん日'])

    valid_mask = (dataframe['本数字'].apply(len) == 7) & (dataframe['ボーナス数字'].apply(len) == 2)
    dataframe = dataframe[valid_mask].copy()
    if dataframe.empty:
        print("[ERROR] 有効な抽せんデータが存在しません。")
        return pd.DataFrame()

    # 数字配列展開
    try:
        nums_array = np.vstack(dataframe['本数字'].values)
    except Exception as e:
        print(f"[ERROR] 数字のvstackに失敗: {e}")
        return pd.DataFrame()

    sorted_nums = np.sort(nums_array, axis=1)
    diffs = np.diff(sorted_nums, axis=1)

    # 特徴量生成
    features = pd.DataFrame(index=dataframe.index)
    features['奇数比'] = (nums_array % 2 != 0).sum(axis=1) / 7
    features['偶数比'] = (nums_array % 2 == 0).sum(axis=1) / 7
    features['本数字合計'] = nums_array.sum(axis=1)
    features['レンジ'] = sorted_nums[:, -1] - sorted_nums[:, 0]
    features['標準偏差'] = np.std(nums_array, axis=1)
    features['中央値'] = np.median(nums_array, axis=1)
    features['数字平均'] = np.mean(nums_array, axis=1)
    features['連番数'] = (diffs == 1).sum(axis=1)
    features['最小間隔'] = diffs.min(axis=1)
    features['最大間隔'] = diffs.max(axis=1)
    features['曜日'] = dataframe['抽せん日'].dt.dayofweek
    features['月'] = dataframe['抽せん日'].dt.month
    features['年'] = dataframe['抽せん日'].dt.year

    # 出現間隔平均
    last_seen = {}
    gaps = []
    for idx, nums in dataframe['本数字'].items():
        gap = [idx - last_seen.get(n, idx) for n in nums]
        gaps.append(np.mean(gap))
        for n in nums:
            last_seen[n] = idx
    features['出現間隔平均'] = gaps

    # 出現頻度スコア
    all_numbers = [n for nums in dataframe['本数字'] for n in nums]
    freq_dict = pd.Series(all_numbers).value_counts().to_dict()
    features['出現頻度スコア'] = dataframe['本数字'].apply(
        lambda nums: sum(freq_dict.get(n, 0) for n in nums) / len(nums)
    )

    pair_freq = {}
    triple_freq = {}
    quad_freq = {}  # ✅ 追加

    for nums in dataframe['本数字']:
        for pair in combinations(sorted(nums), 2):
            pair_freq[pair] = pair_freq.get(pair, 0) + 1
        for triple in combinations(sorted(nums), 3):
            triple_freq[triple] = triple_freq.get(triple, 0) + 1
        for quad in combinations(sorted(nums), 4):  # ✅ 追加
            quad_freq[quad] = quad_freq.get(quad, 0) + 1

    # --- 特徴量列を追加 ---
    features['ペア出現頻度'] = dataframe['本数字'].apply(
        lambda nums: sum(pair_freq.get(tuple(sorted((nums[i], nums[j]))), 0)
                        for i in range(7) for j in range(i+1, 7))
    )

    features['トリプル出現頻度'] = dataframe['本数字'].apply(
        lambda nums: sum(triple_freq.get(tuple(sorted((nums[i], nums[j], nums[k]))), 0)
                        for i in range(7) for j in range(i+1, 7) for k in range(j+1, 7))
    )

    features['クワッド出現頻度'] = dataframe['本数字'].apply(  # ✅ 追加
        lambda nums: sum(quad_freq.get(tuple(sorted((nums[i], nums[j], nums[k], nums[l]))), 0)
                        for i in range(7) for j in range(i+1, 7)
                        for k in range(j+1, 7) for l in range(k+1, 7))
    )

    # 直近5回出現率
    past_5_counts = []
    for i, row in dataframe.iterrows():
        current_date = row['抽せん日']
        recent = dataframe[dataframe['抽せん日'] < current_date].tail(5)
        recent_nums = [n for nums in recent['本数字'] for n in nums]
        match_count = sum(n in recent_nums for n in row['本数字'])
        past_5_counts.append(match_count / 7)
    features['直近5回出現率'] = past_5_counts

    # 列順を安定化（オプション）
    features = features[sorted(features.columns, key=lambda x: x.encode('utf-8'))]

    # 結合して返す
    return pd.concat([dataframe.reset_index(drop=True), features.reset_index(drop=True)], axis=1)

def preprocess_data(data):
    """データの前処理: 特徴量の作成 & スケーリング"""
    
    processed_data = create_advanced_features(data)
    if processed_data.empty:
        print("エラー: 特徴量生成後のデータが空です。データのフォーマットを確認してください。")
        return None, None, None, None

    print("=== 特徴量作成後のデータ ===")
    print(processed_data.head())

    # 数値特徴量の選択
    numeric_features = processed_data.select_dtypes(include=[np.number]).columns.tolist()
    X = processed_data[numeric_features].fillna(0)

    print(f"数値特徴量の数: {len(numeric_features)}, サンプル数: {X.shape[0]}")
    if X.empty:
        print("エラー: 数値特徴量が作成されず、データが空になっています。")
        return None, None, None, None

    # スケーリング
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    print("=== スケーリング後のデータ ===")
    print(X_scaled[:5])

    # 目標変数の作成
    try:
        y = np.array([list(map(int, nums)) for nums in processed_data['本数字']])
    except Exception as e:
        print(f"エラー: 目標変数の作成時に問題が発生しました: {e}")
        return None, None, None, None

    return X_scaled, y, scaler, numeric_features  # ← 特徴量名を追加して返す

def convert_numbers_to_binary_vectors(data):
    """
    本数字を0/1ベクトル化する
    例：[1,5,7,22,28,30,36] → [1,0,0,0,1,0,1, ..., 0,1]
    """
    vectors = []
    for numbers in data['本数字']:
        vec = np.zeros(37)
        for n in numbers:
            if 1 <= n <= 37:
                vec[n-1] = 1
        vectors.append(vec)
    return np.array(vectors)

def calculate_prediction_errors(predictions, actual_numbers):
    """予測値と実際の当選結果の誤差を計算し、特徴量として保存"""
    errors = []
    for pred, actual in zip(predictions, actual_numbers):
        pred_numbers = set(pred[0])
        actual_numbers = set(actual)
        error_count = len(actual_numbers - pred_numbers)
        errors.append(error_count)
    
    return np.mean(errors)

def save_self_predictions(predictions, file_path="self_predictions.csv", max_records=100):
    """予測結果をCSVに保存し、保存件数を最大max_recordsに制限し、世代ファイルも保存"""

    # 🔒 空の予測は保存しない
    if not predictions:
        print("[WARNING] 保存対象の予測データが空です。保存をスキップします。")
        return

    rows = []
    for numbers, confidence in predictions:
        rows.append(numbers.tolist())

    # --- 既存データがあれば読み込み ---
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        try:
            existing = pd.read_csv(file_path, header=None).values.tolist()
            rows = existing + rows
        except pd.errors.EmptyDataError:
            print(f"[WARNING] {file_path} が空または無効な形式のため、新規作成します。")
    else:
        print(f"[INFO] {file_path} が存在しないか空のため、新規作成します。")

    # 最新max_records件だけ残す
    rows = rows[-max_records:]
    df = pd.DataFrame(rows)

    # --- メイン保存 ---
    df.to_csv(file_path, index=False, header=False)
    print(f"[INFO] 自己予測結果を {file_path} に保存しました（最大{max_records}件）")

    # --- 世代ファイルとして保存 ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "self_predictions_history"
    os.makedirs(output_dir, exist_ok=True)

    generation_file = os.path.join(output_dir, f"self_predictions_gen_{timestamp}.csv")
    df.to_csv(generation_file, index=False, header=False)
    print(f"[INFO] 世代別に自己予測も保存しました: {generation_file}")

def load_self_predictions(file_path="self_predictions.csv", min_match_threshold=3, true_data=None):
    if not os.path.exists(file_path):
        print(f"[INFO] 自己予測ファイル {file_path} が見つかりません。")
        return None

    try:
        # 🔥 高速版に置き換え！
        df = pd.read_csv(file_path, header=None)
        df = df.dropna()
        df = df[df.apply(lambda row: all(1 <= x <= 37 for x in row), axis=1)]
        numbers_list = df.values.tolist()

        if true_data is not None:
            scores = evaluate_self_predictions(numbers_list, true_data)
            filtered_rows = [r for r, s in zip(numbers_list, scores) if s >= min_match_threshold]
            print(f"[INFO] 一致数{min_match_threshold}以上の自己予測データ: {len(filtered_rows)}件")
            return filtered_rows
        else:
            return numbers_list

    except Exception as e:
        print(f"[ERROR] 自己予測データ読み込みエラー: {e}")
        return None

def evaluate_self_predictions(self_predictions, true_data):
    """
    自己予測リストと本物データを比較して一致数を評価
    :param self_predictions: [[5,12,17,22,30,34,37], ...]
    :param true_data: 過去の本物本数字データ（data['本数字'].tolist()）
    :return: 各自己予測に対応する最大一致数リスト
    """
    scores = []
    true_sets = [set(nums) for nums in true_data]

    for pred in self_predictions:
        pred_set = set(pred)
        max_match = 0
        for true_set in true_sets:
            match = len(pred_set & true_set)
            if match > max_match:
                max_match = match
        scores.append(max_match)

    return scores

def update_features_based_on_results(data, accuracy_results):
    """過去の予測結果と実際の結果の比較から特徴量を更新"""
    
    for result in accuracy_results:
        event_date = result["抽せん日"]
        max_matches = result["最高一致数"]
        avg_matches = result["平均一致数"]
        confidence_avg = result["信頼度平均"]

        # 過去のデータに予測精度を組み込む
        data.loc[data["抽せん日"] == event_date, "過去の最大一致数"] = max_matches
        data.loc[data["抽せん日"] == event_date, "過去の平均一致数"] = avg_matches
        data.loc[data["抽せん日"] == event_date, "過去の予測信頼度"] = confidence_avg

    # 特徴量がない場合は0で埋める
    data["過去の最大一致数"] = data["過去の最大一致数"].fillna(0)
    data["過去の平均一致数"] = data["過去の平均一致数"].fillna(0)
    data["過去の予測信頼度"] = data["過去の予測信頼度"].fillna(0)

    return data

class LotoLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LotoLSTM, self).__init__()
        self.lstm = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden*2]
        attn_weights = torch.softmax(self.attn(lstm_out).squeeze(-1), dim=1)  # [batch, seq_len]
        context = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)  # [batch, hidden*2]
        out = self.fc(context)
        return out

def train_lstm_model(X_train, y_train, input_size, device):
    
    torch.backends.cudnn.benchmark = True  # ★これを追加
    
    model = LotoLSTM(input_size=input_size, hidden_size=128, output_size=7).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=2)  # ★変更

    scaler = torch.cuda.amp.GradScaler()  # ★Mixed Precision追加

    model.train()
    for epoch in range(50):
        total_loss = 0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():  # ★ここもMixed Precision
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        print(f"[LSTM] Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

    # ONNXエクスポート
    dummy_input = torch.randn(1, 1, input_size).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        "lstm_model.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=12
    )
    print("[INFO] LSTM モデルのトレーニングが完了")
    return model

def extract_high_accuracy_combinations(evaluation_df, threshold=6):
    high_matches = evaluation_df[evaluation_df["本数字一致数"] >= threshold]
    return high_matches

def convert_hit_combos_to_training_data(hit_combos, original_data):
    temp_df = original_data.copy()
    new_rows = []
    for _, row in hit_combos.iterrows():
        temp = {
            "抽せん日": row["抽せん日"],
            "本数字": row["予測番号"],
            "ボーナス数字": row["当選ボーナス"]
        }
        new_rows.append(temp)
    if not new_rows:
        return None, None
    temp_df = pd.DataFrame(new_rows)
    return preprocess_data(temp_df)[:2]

class LotoPredictor:
    def __init__(self, input_size, hidden_size, output_size):
        print("[INFO] モデルを初期化")
        device = torch.device("cpu")
        self.lstm_model = LotoLSTM(input_size, hidden_size, output_size)
        self.regression_models = [None] * 7
        self.scaler = None
        self.onnx_session = None
        self.gan_model = None
        self.ppo_model = None
        self.set_model = SetTransformer().to(device)  # ✅ ここで使っている device
        self.stacking_model = None

        # --- GANモデルロード（存在すれば） ---
        if os.path.exists("gan_model.pth"):
            self.gan_model = LotoGAN()
            self.gan_model.load_state_dict(torch.load("gan_model.pth"))
            self.gan_model.eval()
            print("[INFO] GANモデルをロードしました")

        # --- PPOモデルロード（存在すれば） ---
        if os.path.exists("ppo_model.zip"):
            self.ppo_model = PPO.load("ppo_model")
            print("[INFO] PPOモデルをロードしました")

    def load_onnx_model(self, onnx_path="lstm_model.onnx"):
        print("[INFO] ONNX モデルを読み込みます")
        self.onnx_session = onnxruntime.InferenceSession(
            onnx_path,
            providers=['CPUExecutionProvider']
        )

    def predict_with_onnx(self, X):
        if self.onnx_session is None:
            print("[ERROR] ONNX モデルがロードされていません")
            return None

        input_name = self.onnx_session.get_inputs()[0].name
        output = self.onnx_session.run(None, {input_name: X.astype(np.float32)})
        return output[0]

    def train_model(self, data, accuracy_results=None):
        if accuracy_results is not None:
            print("[DEBUG] accuracy_results に記録を行います")

        now = pd.Timestamp.now()
        past_data = data[data["抽せん日"] < now].copy()
        true_numbers = past_data["本数字"].tolist()

        self_data = load_self_predictions(
            file_path="self_predictions.csv",
            min_match_threshold=6,
            true_data=true_numbers
        )
        high_match_combos = extract_high_match_patterns(past_data, min_match=6)

        if self_data or high_match_combos:
            print("[INFO] 過去の高一致自己予測＋高一致本物データを追加します")
            new_rows = []
            for nums in (self_data or []):
                new_rows.append({'抽せん日': now, '回号': 9999, '本数字': nums, 'ボーナス数字': [0, 0]})
            for nums in (high_match_combos or []):
                new_rows.append({'抽せん日': now, '回号': 9999, '本数字': nums, 'ボーナス数字': [0, 0]})
            if new_rows:
                new_data = pd.DataFrame(new_rows)
                past_data = pd.concat([past_data, new_data], ignore_index=True)

        X, y, self.scaler, self.feature_names = preprocess_data(past_data)
        if X is None or y is None or len(X) == 0:
            print("[ERROR] 前処理後のデータが空です")
            return False

        try:
            X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
            input_size = X_train.shape[1]
            device = torch.device("cpu")

            # --- LSTM モデル ---
            X_train_tensor = torch.tensor(X_train.reshape(-1, 1, input_size), dtype=torch.float32).to(device)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
            self.lstm_model = train_lstm_model(X_train_tensor, y_train_tensor, input_size, device)

            # ONNX 保存
            dummy_input = torch.randn(1, 1, input_size)
            torch.onnx.export(
                self.lstm_model, dummy_input, "lstm_model.onnx",
                input_names=["input"], output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                opset_version=12
            )
            self.load_onnx_model("lstm_model.onnx")
            if self.onnx_session is None:
                print("[ERROR] ONNXモデルのロードに失敗しました")
                return False
            # --- AutoGluon モデル ---
            self.regression_models = [None] * 7
            for i in range(7):
                try:
                    df_train = pd.DataFrame(X_train)
                    df_train['target'] = y_train[:, i]
                    predictor = TabularPredictor(label='target', path=f'autogluon_model_pos{i}').fit(
                        df_train,
                        excluded_model_types=['KNN', 'NN_TORCH'],
            hyperparameters={
                'GBM': {'device': 'cpu', 'num_boost_round': 300},
                'XGB': {'tree_method': 'hist', 'n_estimators': 300},
                'CAT': {'task_type': 'CPU', 'iterations': 300},
                'RF': {'n_estimators': 200}
            },
            num_gpus=0
                    )
                    self.regression_models[i] = predictor
                    print(f"[DEBUG] AutoGluon モデル {i+1}/7 の学習完了")
                except Exception as e:
                    print(f"[ERROR] AutoML モデル {i+1} の学習に失敗しました: {e}")
                    traceback.print_exc()

            if any(model is None for model in self.regression_models):
                print("[ERROR] 一部の AutoML モデルが未学習です")
                return False
            # --- TabNet モデル ---
            try:
                from tabnet_module import train_tabnet
                self.tabnet_model = train_tabnet(X_train, y_train)
                print("[INFO] TabNet モデルの学習完了")
            except Exception as e:
                print(f"[ERROR] TabNet モデルの学習に失敗: {e}")
                traceback.print_exc()
                self.tabnet_model = None
            # --- SetTransformer ---
            try:
                all_numbers = past_data['本数字'].tolist()
                input_tensor = torch.tensor(all_numbers, dtype=torch.long).to(device)
                label_tensor = torch.tensor(
                    [convert_number_list_to_vector(x) for x in all_numbers],
                    dtype=torch.float32
                ).to(device)

                self.set_model = SetTransformer().to(device)
                optimizer = torch.optim.Adam(self.set_model.parameters(), lr=0.001)
                loss_fn = nn.BCELoss()
                self.set_model.train()
                for epoch in range(300):
                    optimizer.zero_grad()
                    output = self.set_model(input_tensor)
                    loss = loss_fn(output, label_tensor)
                    loss.backward()
                    optimizer.step()
                print("[INFO] SetTransformer モデルの学習完了")
            except Exception as e:
                print(f"[ERROR] SetTransformer モデルの学習に失敗: {e}")
                traceback.print_exc()
                return False
            # --- BNN モデル ---
            try:
                from bnn_module import train_bayesian_regression
                self.bnn_model, self.bnn_guide = train_bayesian_regression(
                    X_train, y_train, in_features=input_size, out_features=7, num_steps=500
                )
                print("[INFO] BNN モデルの学習完了")
            except Exception as e:
                print(f"[ERROR] BNN モデルの学習に失敗: {e}")
                traceback.print_exc()
                self.bnn_model = None
                self.bnn_guide = None
            # --- TFT モデル ---
            try:
                from neuralforecast.models import TFT
                from neuralforecast import NeuralForecast
                df_tft = past_data.copy()
                df_tft['ds'] = pd.to_datetime(df_tft['抽せん日'])
                df_tft['unique_id'] = 'loto'
                df_tft['y'] = df_tft['本数字'].apply(lambda x: sum(x) if isinstance(x, list) else 0)
                df_tft = df_tft[['unique_id', 'ds', 'y']].dropna().sort_values('ds')

                if len(df_tft) < 15:
                    print(f"[WARNING] TFTモデルの学習データ不足 ({len(df_tft)} 件)。スキップします。")
                    self.tft_model = None
                else:
                    self.tft_model = NeuralForecast(models=[TFT(input_size=5, h=1)], freq='W')
                    self.tft_model.fit(df_tft)
                    print("[INFO] TFT モデルの学習完了")
            except Exception as e:
                print(f"[ERROR] TFT モデルの学習に失敗: {e}")
                traceback.print_exc()
                self.tft_model = None
            # --- GNN モデル ---
            try:
                import networkx as nx
                from torch_geometric.data import Data
                from torch_geometric.nn import GCNConv

                class LotoGNN(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.conv1 = GCNConv(37, 64)
                        self.conv2 = GCNConv(64, 1)

                    def forward(self, data):
                        x = self.conv1(data.x, data.edge_index).relu()
                        return self.conv2(x, data.edge_index)

                G = nx.Graph()
                for nums in past_data['本数字']:
                    for i in range(len(nums)):
                        for j in range(i + 1, len(nums)):
                            G.add_edge(nums[i] - 1, nums[j] - 1)
                edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
                x = torch.eye(37)
                graph_data = Data(x=x, edge_index=edge_index)

                self.gnn_model = LotoGNN().to(device)
                optimizer = torch.optim.Adam(self.gnn_model.parameters(), lr=0.01)
                self.gnn_model.train()
                for epoch in range(100):
                    optimizer.zero_grad()
                    out = self.gnn_model(graph_data.to(device))
                    loss = out.mean()
                    loss.backward()
                    optimizer.step()
                print("[INFO] GNN モデルの学習完了")
            except Exception as e:
                print(f"[ERROR] GNN モデルの学習に失敗: {e}")
                traceback.print_exc()
                return False
            # --- Diffusion モデル（DDPM） ---
            try:
                from diffusion_module import train_diffusion_ddpm
                vectors = [convert_number_list_to_vector(nums) for nums in past_data["本数字"]]
                self.diffusion_model, self.diffusion_betas, self.diffusion_alphas = train_diffusion_ddpm(
                    np.array(vectors), timesteps=100, epochs=500, batch_size=64
                )
                print("[INFO] Diffusion モデルの学習完了")
            except Exception as e:
                print(f"[ERROR] Diffusion モデルの学習に失敗: {e}")
                traceback.print_exc()
                self.diffusion_model = None

            # --- stacking モデル ---
            try:
                from stacking_model import train_stacking_model

                X_tensor = torch.tensor(X_train.reshape(-1, 1, input_size), dtype=torch.float32).to(device)
                self.lstm_model.eval()
                with torch.no_grad():
                    lstm_preds = self.lstm_model(X_tensor).cpu().numpy().astype(int).tolist()

                automl_preds = []
                for i in range(7):
                    preds = self.regression_models[i].predict(pd.DataFrame(X_train))
                    automl_preds.append(np.round(preds).astype(int).tolist())
                automl_preds = list(map(list, zip(*automl_preds)))

                gan_preds = [self.gan_model.generate_samples(1)[0] if self.gan_model else np.random.rand(37)
                            for _ in range(len(X_train))]
                ppo_preds = []
                for _ in range(len(X_train)):
                    obs = np.zeros(37, dtype=np.float32)
                    action, _ = self.ppo_model.predict(obs, deterministic=True) if self.ppo_model else (np.random.rand(37), None)
                    ppo_preds.append(np.array(action))

                self.stacking_model = train_stacking_model(
                    lstm_preds, automl_preds, gan_preds, ppo_preds, y_train.tolist()
                )
                print("[INFO] stacking_model の学習完了")
            except Exception as e:
                print(f"[ERROR] stacking_model の学習に失敗: {e}")
                traceback.print_exc()
                return False
            # --- Stacking 最適化（Optuna） ---
            try:
                from stacking_optuna import optimize_stacking
                pred_dict = {
                    "lstm": lstm_preds,
                    "automl": automl_preds,
                    "gan": [list(g) for g in gan_preds],
                    "ppo": [list(p) for p in ppo_preds],
                }
                self.best_stacking_weights = optimize_stacking(pred_dict, y_train.tolist())
                print(f"[INFO] Stacking 重み最適化完了: {self.best_stacking_weights}")
            except Exception as e:
                print(f"[ERROR] Stacking 重み最適化に失敗: {e}")
                traceback.print_exc()
                self.best_stacking_weights = None

            return True

        except Exception as e:
            print(f"[ERROR] モデル学習中に例外が発生しました: {e}")
            traceback.print_exc()
            return False

    def predict(self, latest_data, num_candidates=50):
        print(f"[INFO] 予測を開始（候補数: {num_candidates}）")
        X, _, _, _ = preprocess_data(latest_data)
    
        if X is None or len(X) == 0:
            print("[ERROR] 予測用データが空です")
            return None, None
    
        if not self.regression_models or any(m is None for m in self.regression_models):
            print("[ERROR] 回帰モデル（AutoML）が未学習です")
            return None, None
    
        if self.onnx_session is None:
            print("[ERROR] ONNXモデル（LSTM）が未ロードです")
            return None, None
    
        if self.gnn_model is None:
            print("[ERROR] GNNモデルが未学習です")
            return None, None
    
        if self.stacking_model is None:
            print("[ERROR] stacking_model が未セットです")
            return None, None
    
        print(f"[DEBUG] 予測用データの shape: {X.shape}")
        try:
            latest_draw_date = latest_data['抽せん日'].max()
            past_data = latest_data[latest_data['抽せん日'] < latest_draw_date]
            if past_data.empty:
                past_data = latest_data.copy()
    
            freq_score = calculate_number_frequencies(past_data)
            cycle_score = calculate_number_cycle_score(past_data)
    
            # --- GNNスコア計算 ---
            import networkx as nx
            from torch_geometric.data import Data
            G = nx.Graph()
            for nums in past_data['本数字']:
                for i in range(len(nums)):
                    for j in range(i + 1, len(nums)):
                        G.add_edge(nums[i] - 1, nums[j] - 1)
            edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
            x = torch.eye(37)
            graph_data = Data(x=x, edge_index=edge_index)
    
            device = torch.device("cpu")
            self.gnn_model.eval()
            with torch.no_grad():
                gnn_scores = self.gnn_model(graph_data.to(device)).squeeze().cpu().numpy()
    
            # --- 特徴量整形 ---
            expected_features = self.regression_models[0].feature_metadata.get_features()
            X_df = pd.DataFrame(X, columns=self.feature_names)
    
            missing_cols = [col for col in expected_features if col not in X_df.columns]
            if missing_cols:
                print(f"[WARNING] AutoML用の不足特徴量を0で補完: {missing_cols}")
                for col in missing_cols:
                    X_df[col] = 0.0
            X_df = X_df[expected_features]
    
            lstm_input_size = self.lstm_model.lstm.input_size
            if X_df.shape[1] < lstm_input_size:
                print(f"[WARNING] LSTM向け特徴量が不足: {X_df.shape[1]} → {lstm_input_size} に補完")
                for i in range(lstm_input_size - X_df.shape[1]):
                    X_df[f'_pad_{i}'] = 0.0
            elif X_df.shape[1] > lstm_input_size:
                print(f"[WARNING] LSTM向け特徴量が多すぎます: {X_df.shape[1]} → {lstm_input_size} に切り詰め")
                X_df = X_df.iloc[:, :lstm_input_size]
    
            # --- GAベース候補生成 ---
            from genetic_algorithm import evolve_candidates
            base_candidates = evolve_candidates(self, X_df, generations=10, population_size=num_candidates)
    
            if not base_candidates or not isinstance(base_candidates, list) or len(base_candidates) < 1:
                print("[ERROR] 候補生成に失敗しました（base_candidatesが空または不正）")
                return None, None
    
            all_predictions = []
            for idx, (lstm_vec, automl_vec, gan_vec, ppo_vec) in enumerate(base_candidates):
                try:
                    numbers, _ = predict_with_stacking(self.stacking_model, lstm_vec, automl_vec, gan_vec, ppo_vec)
    
                    flat_numbers = []
                    for n in numbers:
                        if isinstance(n, list):
                            flat_numbers.extend(n)
                        else:
                            flat_numbers.append(n)
    
                    # --- 候補をクリーンに整形 ---
                    flat_numbers = list(set(np.round(flat_numbers).astype(int)))
                    flat_numbers = [n for n in flat_numbers if 1 <= n <= 37]
                    flat_numbers = sorted(flat_numbers)[:7]  # 上位7個のみ
    
                    if len(flat_numbers) != 7:
                        continue  # ❌ 不正な候補は除外
    
                    # --- スコア計算 ---
                    score_freq = sum(freq_score.get(n, 0) for n in flat_numbers)
                    score_cycle = sum(cycle_score.get(n, 0) for n in flat_numbers)
                    score_gnn = sum(gnn_scores[n - 1] for n in flat_numbers)
                    score = score_freq - score_cycle + score_gnn
    
                    if np.isnan(score) or np.isinf(score):
                        continue  # 異常値スキップ
    
                    confidence = 1.0 + (score / 500.0)
                    all_predictions.append((flat_numbers, confidence))
    
                except Exception as e:
                    print(f"[WARNING] stacking予測中にエラー (index {idx}): {e}")
                    traceback.print_exc()
                    continue
    
            print(f"[INFO] 総予測候補数: {len(all_predictions)} 件")
    
            if not all_predictions:
                print("[WARNING] 有効な予測候補が生成されませんでした")
                return None, None
    
            numbers_only = [pred[0] for pred in all_predictions]
            confidence_scores = [pred[1] for pred in all_predictions]
            return numbers_only, confidence_scores
    
        except Exception as e:
            print(f"[ERROR] 予測中にエラー発生: {e}")
            traceback.print_exc()
            return None, None

# 予測結果の評価
def evaluate_predictions(predictions, actual_numbers):
    matches = []
    for pred in predictions:
        match_count = len(set(pred[0]) & set(actual_numbers))
        matches.append(match_count)
    return {
        'max_matches': max(matches),
        'avg_matches': np.mean(matches),
        'predictions_with_matches': list(zip(predictions, matches))
    }
# 追加: 最新の抽せん日を取得する関数
official_url = "https://www.takarakuji-official.jp/ec/loto7/?kujiprdShbt=61&knyschm=0"

async def fetch_drawing_dates():
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(official_url, timeout=10) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    drawing_dates = []
                    date_elements = soup.select("dl.m_param.m_thumbSet_row")
                    for dl in date_elements:
                        dt_element = dl.find("dt", string="抽せん日")
                        if dt_element:
                            dd_element = dt_element.find_next_sibling("dd")
                            if dd_element:
                                formatted_date = dd_element.text.strip().replace("/", "-")
                                drawing_dates.append(formatted_date)
                    
                    return drawing_dates
                else:
                    print(f"HTTPエラー {response.status}: {official_url}")
        except Exception as e:
            print(f"抽せん日取得エラー: {e}")
    return []

async def get_latest_drawing_dates():
    dates = await fetch_drawing_dates()
    return dates

def parse_number_string(number_str):
    """
    予測番号や当選番号の文字列をリスト化する関数
    - スペース / カンマ / タブ 区切りに対応
    - "07 15 20 28 29 34 36" → [7, 15, 20, 28, 29, 34, 36]
    - "[7, 15, 20, 28, 29, 34, 36]" → [7, 15, 20, 28, 29, 34, 36]
    """
    if pd.isna(number_str):
        return []  # NaN の場合は空リストを返す
    
    # 不要な記号を削除（リスト形式の場合）
    number_str = number_str.strip("[]").replace("'", "").replace('"', '')

    # スペース・カンマ・タブで分割し、整数変換
    numbers = re.split(r'[\s,]+', number_str)

    # 数字のみにフィルタリングして整数変換
    return [int(n) for n in numbers if n.isdigit()]

def classify_rank(main_match, bonus_match):
    """本数字一致数とボーナス一致数からLoto7の等級を判定"""
    if main_match == 7:
        return "1等"
    elif main_match == 6 and bonus_match >= 1:
        return "2等"
    elif main_match == 6:
        return "3等"
    elif main_match == 5:
        return "4等"
    elif main_match == 4:
        return "5等"
    elif main_match == 3 and bonus_match >= 1:
        return "6等"
    else:
        return "該当なし"
    
def calculate_precision_recall_f1(evaluation_df):
    y_true = []
    y_pred = []

    for _, row in evaluation_df.iterrows():
        actual = set(row["当選本数字"])
        predicted = set(row["予測番号"])
        for n in range(1, 38):
            y_true.append(1 if n in actual else 0)
            y_pred.append(1 if n in predicted else 0)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("\n=== 評価指標 ===")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")

def evaluate_prediction_accuracy_with_bonus(predictions_file="loto7_predictions.csv", results_file="loto7.csv"):
    """
    予測結果と実際の当選結果を比較し、ボーナス数字を考慮して精度を評価し、等級も判定する
    """
    try:
        # === ✅ 空ファイルや構文エラーでも安全に読み込む ===
        try:
            predictions_df = pd.read_csv(predictions_file, encoding='utf-8-sig')
            if predictions_df.empty or predictions_df.shape[0] == 0 or "抽せん日" not in predictions_df.columns:
                print(f"[WARNING] 予測ファイルが空か無効です: {predictions_file}")
                return None
        except Exception as read_err:
            print(f"[WARNING] 予測ファイルの読み込み失敗: {read_err}")
            return None

        results_df = pd.read_csv(results_file, encoding='utf-8-sig')
        evaluation_results = []

        for index, row in predictions_df.iterrows():
            draw_date = row["抽せん日"]
            actual_row = results_df[results_df["抽せん日"] == draw_date]
            if actual_row.empty:
                continue

            actual_numbers = parse_number_string(actual_row.iloc[0]["本数字"])
            actual_bonus = parse_number_string(actual_row.iloc[0]["ボーナス数字"])

            for i in range(1, 6):  # 予測1〜5
                pred_col = f"予測{i}"
                if pred_col not in row or pd.isna(row[pred_col]):
                    continue

                try:
                    predicted_numbers = set(parse_number_string(row[pred_col]))
                    main_match = len(predicted_numbers & set(actual_numbers))
                    bonus_match = len(predicted_numbers & set(actual_bonus))
                    rank = classify_rank(main_match, bonus_match)

                    evaluation_results.append({
                        "抽せん日": draw_date,
                        "予測番号": list(predicted_numbers),
                        "当選本数字": actual_numbers,
                        "当選ボーナス": actual_bonus,
                        "本数字一致数": main_match,
                        "ボーナス一致数": bonus_match,
                        "信頼度": row.get(f"信頼度{i}", None),
                        "等級": rank
                    })

                except Exception as e:
                    print(f"予測データ処理エラー (行 {index}, 予測 {i}): {e}")
                    continue

        evaluation_df = pd.DataFrame(evaluation_results)
        evaluation_df.to_csv("loto7_prediction_evaluation_with_bonus.csv", index=False, encoding='utf-8-sig')
        print("予測精度の評価結果を保存しました: loto7_prediction_evaluation_with_bonus.csv")

        # 統計出力
        print("\n=== 予測精度の統計情報 ===")
        if not evaluation_df.empty:
            print(f"最大本数字一致数: {evaluation_df['本数字一致数'].max()}")
            print(f"平均本数字一致数: {evaluation_df['本数字一致数'].mean():.2f}")
            print(f"最大ボーナス一致数: {evaluation_df['ボーナス一致数'].max()}")
            print(f"平均ボーナス一致数: {evaluation_df['ボーナス一致数'].mean():.2f}")
            print("\n--- 等級の分布 ---")
            print(evaluation_df['等級'].value_counts())
            # ✅ 評価指標を追加で表示
            calculate_precision_recall_f1(evaluation_df)
        else:
            print("評価データがありません。")
                # === 評価結果をテキストファイルに出力 ===
        try:
            with open("loto7_evaluation_summary.txt", "w", encoding="utf-8") as f:
                f.write("=== 予測精度の統計情報 ===\n")
                f.write(f"最大本数字一致数: {evaluation_df['本数字一致数'].max()}\n")
                f.write(f"平均本数字一致数: {evaluation_df['本数字一致数'].mean():.2f}\n")
                f.write(f"最大ボーナス一致数: {evaluation_df['ボーナス一致数'].max()}\n")
                f.write(f"平均ボーナス一致数: {evaluation_df['ボーナス一致数'].mean():.2f}\n\n")

                f.write("--- 等級の分布 ---\n")
                f.write(f"{evaluation_df['等級'].value_counts().to_string()}\n\n")

                # Precision/Recall/F1
                y_true = []
                y_pred = []

                for _, row in evaluation_df.iterrows():
                    actual = set(row["当選本数字"])
                    predicted = set(row["予測番号"])
                    for n in range(1, 38):
                        y_true.append(1 if n in actual else 0)
                        y_pred.append(1 if n in predicted else 0)

                precision = precision_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)

                f.write("=== 評価指標 ===\n")
                f.write(f"Precision: {precision:.3f}\n")
                f.write(f"Recall:    {recall:.3f}\n")
                f.write(f"F1 Score:  {f1:.3f}\n")
            print("評価結果を loto7_evaluation_summary.txt に保存しました。")
        except Exception as e:
            print(f"[WARNING] テキストファイル出力失敗: {e}")

        return evaluation_df

    except Exception as e:
        print(f"予測精度の評価エラー: {e}")
        return None

# 予測結果をCSVファイルに保存する関数
def save_predictions_to_csv(predictions, drawing_date, filename="loto7_predictions.csv"):
    row = {"抽せん日": drawing_date}

    for i, (numbers, confidence) in enumerate(predictions[:5], 1):
        row[f"予測{i}"] = ', '.join(map(str, numbers))
        row[f"信頼度{i}"] = round(confidence, 3)

    df = pd.DataFrame([row])

    if os.path.exists(filename):
        try:
            existing_df = pd.read_csv(filename, encoding='utf-8-sig')
            if "抽せん日" not in existing_df.columns:
                print(f"警告: CSVに'抽せん日'列が見つかりません。新規作成します。")
                existing_df = pd.DataFrame(columns=["抽せん日"] + [f"予測{i}" for i in range(1, 6)] + [f"信頼度{i}" for i in range(1, 6)])
            existing_df = existing_df[existing_df["抽せん日"] != drawing_date]
            df = pd.concat([existing_df, df], ignore_index=True)
        except Exception as e:
            print(f"CSVファイルの読み込みエラー: {e}。新規作成します。")
            df = pd.DataFrame([row])

    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"予測結果を {filename} に保存しました。")

def is_running_with_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except ImportError:
        return False

def main_with_improved_predictions():
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    drawing_dates = asyncio.run(get_latest_drawing_dates())
    latest_drawing_date = drawing_dates[0] if drawing_dates else "不明"
    print("最新の抽せん日:", latest_drawing_date)

    try:
        data = pd.read_csv("loto7.csv")
        print("データ読み込み完了")
    except Exception as e:
        print(f"データ読み込みエラー: {e}")
        return

    accuracy_results = evaluate_prediction_accuracy_with_bonus("loto7_predictions.csv", "loto7.csv")
    if accuracy_results is not None and not accuracy_results.empty:
        print("過去の予測精度を評価しました。")

    X, _, _ = preprocess_data(data)
    input_size = X.shape[1] if X is not None else 10
    hidden_size = 128
    output_size = 7

    predictor = LotoPredictor(input_size, hidden_size, output_size)

    try:
        print("モデルの学習を開始...")
        predictor.train_model(data)
        print("モデルの学習完了")
    except Exception as e:
        print(f"モデル学習エラー: {e}")
        return

    if is_running_with_streamlit():
        st.title("ロト7予測AI")
        if st.button("予測を実行"):
            try:
                latest_data = data.tail(10)
                predictions, confidence_scores = predictor.predict(latest_data)

                if predictions is None:
                    print("[ERROR] 予測に失敗したため処理を中断します。")
                    return  # ⬅️ ここで強制終了させると安全

                verified_predictions = verify_predictions(list(zip(predictions, confidence_scores)), data)

                for i, (numbers, confidence) in enumerate(verified_predictions[:5], 1):
                    st.write(f"予測 {i}: {numbers} (信頼度: {confidence:.3f})")

                save_predictions_to_csv(verified_predictions, latest_drawing_date)

            except Exception as e:
                st.error(f"予測エラー: {e}")
    else:
        print("[INFO] Streamlit以外の実行環境検出。通常のコンソール出力で予測を実行します。")
        try:
            latest_data = data.tail(10)
            predictions, confidence_scores = predictor.predict(latest_data)

            if predictions is None:
                print("[ERROR] 予測に失敗したため処理を中断します。")
                return  # ⬅️ ここで強制終了させると安全

            verified_predictions = verify_predictions(list(zip(predictions, confidence_scores)), data)

            print("\n=== 予測結果 ===")
            for i, (numbers, confidence) in enumerate(verified_predictions[:5], 1):
                print(f"予測 {i}: {numbers} (信頼度: {confidence:.3f})")

            save_predictions_to_csv(verified_predictions, latest_drawing_date)

        except Exception as e:
            print(f"予測エラー: {e}")
    
def calculate_pattern_score(numbers, historical_data=None):
    score = 0
    odd_count = sum(1 for n in numbers if n % 2 != 0)
    if 2 <= odd_count <= 5:
        score += 1
    gaps = [numbers[i+1] - numbers[i] for i in range(len(numbers)-1)]
    if min(gaps) >= 2:
        score += 1
    total = sum(numbers)
    if 100 <= total <= 150:
        score += 1
    if max(numbers) - min(numbers) >= 15:
        score += 1
    return score

def plot_prediction_analysis(predictions, historical_data):
    plt.figure(figsize=(15, 10))
    
    # 予測番号の分布
    plt.subplot(2, 2, 1)
    all_predicted_numbers = [num for pred in predictions for num in pred[0]]
    plt.hist(all_predicted_numbers, bins=37, range=(1, 38), alpha=0.7)
    plt.title('予測番号の分布')
    plt.xlabel('数字')
    plt.ylabel('頻度')
    
    # 信頼度スコアの分布
    plt.subplot(2, 2, 2)
    confidence_scores = [pred[1] for pred in predictions]
    plt.hist(confidence_scores, bins=20, alpha=0.7)
    plt.title('信頼度スコアの分布')
    plt.xlabel('信頼度')
    plt.ylabel('頻度')
    
    # 過去の当選番号との比較
    plt.subplot(2, 2, 3)
    historical_numbers = [num for numbers in historical_data['本数字'] for num in numbers]
    plt.hist(historical_numbers, bins=37, range=(1, 38), alpha=0.5, label='過去の当選')
    plt.hist(all_predicted_numbers, bins=37, range=(1, 38), alpha=0.5, label='予測')
    plt.title('予測 vs 過去の当選')
    plt.xlabel('数字')
    plt.ylabel('頻度')
    plt.legend()
    
    # パターン分析
    plt.subplot(2, 2, 4)
    pattern_scores = [calculate_pattern_score(pred[0]) for pred in predictions]
    plt.scatter(range(len(pattern_scores)), pattern_scores, alpha=0.5)
    plt.title('予測パターンスコア')
    plt.xlabel('予測インデックス')
    plt.ylabel('パターンスコア')
    
    plt.tight_layout()
    plt.savefig('prediction_analysis.png')
    plt.close()

def generate_evolution_graph(log_file="evolution_log.txt", output_file="evolution_graph.png"):
    """
    evolution_log.txtを読み込んで進化グラフを生成・保存する
    """
    if not os.path.exists(log_file):
        print(f"[WARNING] 進化ログ {log_file} が見つかりません")
        return

    dates = []
    counts = []

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                parts = line.strip().split(":")
                date_part = parts[0].strip()
                count_part = parts[2].strip()

                date = pd.to_datetime(date_part)
                count = int(count_part.split()[0])

                dates.append(date)
                counts.append(count)
            except Exception as e:
                print(f"[WARNING] ログパース失敗: {e}")
                continue

    if not dates:
        print("[WARNING] 進化ログに有効なデータがありません")
        return

    # --- グラフ描画 ---
    plt.figure(figsize=(10, 6))
    plt.plot(dates, counts, marker='o', linestyle='-', color='blue')
    plt.title("自己進化履歴（自己予測データ件数推移）")
    plt.xlabel("日時")
    plt.ylabel("自己予測データ件数")
    plt.grid(True)
    plt.tight_layout()

    # --- 保存 ---
    plt.savefig(output_file)
    plt.close()
    print(f"[INFO] 進化履歴グラフを保存しました: {output_file}")

def verify_predictions(predictions, historical_data, top_k=5):
    def check_number_constraints(numbers):
        """予測数字配列の制約チェック"""
        if len(numbers) != 7:
            return False
        if len(np.unique(numbers)) != 7:
            return False
        if not np.all((numbers >= 1) & (numbers <= 37)):
            return False
        if not np.issubdtype(numbers.dtype, np.integer):
            return False
        return True

    print(f"[INFO] 予測候補をフィルタリング中...（総数: {len(predictions)}）")

    # --- 有効な予測だけ抽出 ---
    valid_predictions = []
    for pred, conf in predictions:
        try:
            numbers = np.array(pred)
            numbers = np.unique(np.round(numbers).astype(int))
            numbers = np.sort(numbers)
            if check_number_constraints(numbers):
                valid_predictions.append((numbers, conf))
        except Exception as e:
            print(f"[WARNING] 予測整形中にエラー: {e}")
            continue

    print(f"[INFO] 有効な予測数: {len(valid_predictions)} 件")

    if not valid_predictions:
        print("[WARNING] 有効な予測がありません")
        return []

    # --- 上位候補を選定 ---
    valid_predictions.sort(key=lambda x: x[1], reverse=True)
    candidates = valid_predictions[:max(100, top_k)]

    # --- カバレッジ最大化による選抜 ---
    selected = []
    used_numbers = set()
    used_flags = [False] * len(candidates)

    while len(selected) < max(top_k - 2, 1):
        best_score = -1
        best_idx = -1

        for idx, (numbers_set, conf) in enumerate(candidates):
            if used_flags[idx]:
                continue
            combined = used_numbers.union(numbers_set)
            coverage_score = len(combined)
            random_boost = random.uniform(0, 1) * 0.1
            total_score = (coverage_score * 0.6) + (conf * 0.2) + random_boost

            if total_score > best_score:
                best_score = total_score
                best_idx = idx

        if best_idx == -1:
            break

        selected.append(candidates[best_idx])
        used_numbers.update(candidates[best_idx][0])
        used_flags[best_idx] = True

    # --- 強制6本構成を追加 ---
    try:
        historical = historical_data.copy()
        historical['本数字'] = historical['本数字'].apply(lambda x: list(map(int, x)) if isinstance(x, list) else [])

        high_match_rows = []
        for idx1, row1 in historical.iterrows():
            nums1 = set(row1['本数字'])
            for idx2, row2 in historical.iterrows():
                if idx1 >= idx2:
                    continue
                nums2 = set(row2['本数字'])
                if len(nums1 & nums2) >= 6:
                    high_match_rows.append(list(nums1))

        if high_match_rows:
            added_templates = set()
            attempts = 0
            while len(added_templates) < 2 and attempts < 10:
                template = tuple(sorted(random.choice(high_match_rows)))
                if template in added_templates:
                    attempts += 1
                    continue
                template_set = set(template)
                available_numbers = list(set(range(1, 38)) - template_set)
                if available_numbers:
                    removed = random.choice(list(template_set))
                    added = random.choice(available_numbers)
                    template_set.remove(removed)
                    template_set.add(added)
                final_combo = sorted(template_set)
                selected.append((np.array(final_combo), 1.0))
                added_templates.add(template)
            print("[INFO] 強制6本構成を追加しました")
        else:
            print("[WARNING] 過去に6本一致パターンが見つかりませんでした")

    except Exception as e:
        print(f"[WARNING] 強制構成作成エラー: {e}")

    print("[INFO] 最終選択された予測数:", len(selected))
    return selected
# --- 🔥 新規追加関数 ---
def extract_high_match_patterns(dataframe, min_match=6):
    """過去データから高一致パターンだけ抽出"""
    high_match_combos = []
    for idx1, row1 in dataframe.iterrows():
        nums1 = set(row1['本数字'])
        for idx2, row2 in dataframe.iterrows():
            if idx1 >= idx2:
                continue
            nums2 = set(row2['本数字'])
            if len(nums1 & nums2) >= min_match:
                high_match_combos.append(sorted(nums1))
    return high_match_combos

def calculate_number_frequencies(dataframe):
    """過去データから番号出現頻度スコアを計算"""
    all_numbers = [num for nums in dataframe['本数字'] for num in nums]
    freq = pd.Series(all_numbers).value_counts().to_dict()
    return freq

def calculate_number_cycle_score(dataframe):
    """番号ごとの出現周期スコアを計算"""
    number_last_seen = {n: None for n in range(1, 38)}
    number_cycle = {n: [] for n in range(1, 38)}
    
    for i, nums in enumerate(dataframe['本数字']):
        for n in range(1, 38):
            if n in nums:
                if number_last_seen[n] is not None:
                    cycle = i - number_last_seen[n]
                    number_cycle[n].append(cycle)
                number_last_seen[n] = i

    avg_cycle = {n: np.mean(cycles) if cycles else 999 for n, cycles in number_cycle.items()}
    return avg_cycle

def bulk_predict_all_past_draws():
    df = pd.read_csv("loto7.csv")
    df["抽せん日"] = pd.to_datetime(df["抽せん日"])
    df = df.sort_values("抽せん日").reset_index(drop=True)

    predictions_file = "loto7_predictions.csv"

    # ✅ 既存の予測日付を読み取り
    predicted_dates = set()
    if os.path.exists(predictions_file):
        try:
            pred_df = pd.read_csv(predictions_file, encoding='utf-8-sig')
            pred_df["抽せん日"] = pd.to_datetime(pred_df["抽せん日"], errors='coerce')
            predicted_dates = set(pred_df["抽せん日"].dropna().dt.date)
            print(f"[INFO] 予測済み日付: {len(predicted_dates)} 件")
        except Exception as e:
            print(f"[WARNING] 予測済みファイルの読み込み失敗: {e}")

    total = len(df)
    predictor = None
    input_size = 0

    for i in range(10, total):
        test_row = df.iloc[i]
        test_date = test_row["抽せん日"]
        test_date_str = test_date.strftime("%Y-%m-%d")

        # ✅ すでに予測済みならスキップ
        if test_date.date() in predicted_dates:
            print(f"[SKIP] 既に予測済み: {test_date_str}")
            continue

        print(f"\n=== {test_date_str} の予測を開始（{i}/{total - 1}） ===")
        latest_data = df.iloc[i - 10:i]
        train_data = df.iloc[:i]

        # ✅ 50件ごとに再学習 または 初回
        if predictor is None or (i - 10) % 50 == 0:
            print(f"[INFO] モデルを再学習中...（index={i}）")
            try:
                X_tmp, _, _, _ = preprocess_data(train_data)
                if X_tmp is None or X_tmp.shape[1] == 0:
                    print(f"[WARNING] {test_date_str} の特徴量が不正です。スキップします。")
                    continue
                input_size = X_tmp.shape[1]
                predictor = LotoPredictor(input_size, 128, 7)
                success = predictor.train_model(train_data)
                if not success:
                    print(f"[ERROR] {test_date_str} モデル学習に失敗しました。スキップします。")
                    continue
            except Exception as e:
                print(f"[ERROR] {test_date_str} モデル学習例外: {e}")
                traceback.print_exc()
                continue

        # 予測
        try:
            predictions, confidence_scores = predictor.predict(latest_data)
            if predictions is None:
                print(f"[ERROR] {test_date_str} の予測に失敗しました。スキップします。")
                continue
        except Exception as e:
            print(f"[ERROR] {test_date_str} の予測中に例外: {e}")
            traceback.print_exc()
            continue

        # 検証・保存
        try:
            verified_predictions = verify_predictions(list(zip(predictions, confidence_scores)), train_data)
            save_self_predictions(verified_predictions)
            save_predictions_to_csv(verified_predictions, test_date_str)
        except Exception as e:
            print(f"[ERROR] {test_date_str} の予測結果保存中にエラー: {e}")
            traceback.print_exc()

        # 評価
        try:
            print(f"[INFO] {test_date_str} の予測精度を評価中...")
            evaluate_prediction_accuracy_with_bonus("loto7_predictions.csv", "loto7.csv")
        except Exception as e:
            print(f"[ERROR] {test_date_str} 評価中にエラー: {e}")
            traceback.print_exc()

    print(f"\n=== 一括予測と評価が完了しました [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ===")

if __name__ == "__main__":
    # main_with_improved_predictions()
    bulk_predict_all_past_draws()
