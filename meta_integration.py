
# -*- coding: utf-8 -*-
# meta_integration.py
# Utility to boost prediction pipeline: calibration, number-lift, pattern boost, meta-learner, enhanced selection, and reward shaping.

from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from collections import Counter
import warnings

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def _parse_listish(x):
    if isinstance(x, list):
        return [int(v) for v in x]
    if pd.isna(x):
        return []
    s = str(x).strip().strip("[](){}").replace("'", "").replace('"', "")
    parts = [p for p in s.replace("、"," ").replace("，"," ").replace(",", " ").split() if p.strip().isdigit()]
    return [int(p) for p in parts]

def _combo_features(nums):
    nums = sorted(nums)
    if not nums:
        return {"sum": np.nan, "odd_ratio": np.nan, "range": np.nan, "consecutive": np.nan, "min_gap": np.nan, "max_gap": np.nan}
    gaps = [nums[i+1]-nums[i] for i in range(len(nums)-1)]
    odd = sum(1 for n in nums if n%2==1)
    return {
        "sum": sum(nums),
        "odd_ratio": odd/len(nums),
        "range": max(nums)-min(nums),
        "consecutive": sum(1 for g in gaps if g==1),
        "min_gap": min(gaps) if gaps else np.nan,
        "max_gap": max(gaps) if gaps else np.nan,
    }

@dataclass
class TrainingAssets:
    conf_bins: List[float]
    conf_bin_rates: List[float]
    lift_by_number: Dict[int, float]
    lift_mean: float
    lift_std: float
    pattern_corr: Dict[str, float]
    meta_pipeline: Optional[Pipeline] = None

    def calibrated_conf(self, conf: float) -> float:
        if conf is None or (isinstance(conf, float) and np.isnan(conf)):
            return float(np.mean(self.conf_bin_rates)) if self.conf_bin_rates else 0.5
        for edge, rate in zip(self.conf_bins, self.conf_bin_rates):
            if conf <= edge:
                return float(rate)
        return float(self.conf_bin_rates[-1] if self.conf_bin_rates else 0.5)

def train_assets_from_eval(eval_csv: str) -> TrainingAssets:
    df = pd.read_csv(eval_csv, encoding="utf-8")
    colmap = {
        "本数字一致数": "main_match",
        "ボーナス一致数": "bonus_match",
        "等級": "rank",
        "信頼度": "confidence",
        "予測番号": "predicted",
        "当選本数字": "actual_main",
        "当選ボーナス": "actual_bonus",
        "抽せん日": "draw_date",
    }
    for k,v in colmap.items():
        if k in df.columns: df.rename(columns={k:v}, inplace=True)
    df["predicted"] = df["predicted"].apply(_parse_listish)
    df["actual_main"] = df["actual_main"].apply(_parse_listish)
    df["actual_bonus"] = df["actual_bonus"].apply(_parse_listish)
    df["main_match"] = pd.to_numeric(df["main_match"], errors="coerce").fillna(0).astype(int)
    df["bonus_match"] = pd.to_numeric(df["bonus_match"], errors="coerce").fillna(0).astype(int)
    if "confidence" not in df.columns:
        cand = [c for c in df.columns if "信頼" in c]
        if cand:
            df["confidence"] = pd.to_numeric(df[cand[0]], errors="coerce")
        else:
            df["confidence"] = np.nan

    tmp = df.copy()
    tmp["success4"] = (tmp["main_match"] >= 4).astype(int)
    tmp["confidence"] = tmp["confidence"].astype(float)
    tmp["confidence"].fillna(tmp["confidence"].median(), inplace=True)
    try:
        bins = pd.qcut(tmp["confidence"], q=10, duplicates="drop")
        grp = tmp.groupby(bins)["success4"].mean().reset_index()
        bin_edges = [b.right for b in grp["confidence"]]
        bin_rates = grp["success4"].tolist()
    except Exception:
        bin_edges = [tmp["confidence"].max()]
        bin_rates = [tmp["success4"].mean()]

    # number lift
    rows = []
    for _, r in df.iterrows():
        pred = r["predicted"]; m = r["main_match"]
        for n in range(1, 38):
            rows.append({"number": n, "included": 1 if n in pred else 0, "main_match": m})
    pernum = pd.DataFrame(rows)
    impact = pernum.groupby(["number","included"]).agg(avg_match=("main_match","mean"),
                                                      count=("main_match","count")).reset_index()
    pivot = impact.pivot(index="number", columns="included", values="avg_match").reset_index()
    pivot.columns = ["number","avg_excluded","avg_included"]
    pivot["lift"] = pivot["avg_included"] - pivot["avg_excluded"]
    lift_by_number = {int(n): float(l) for n,l in zip(pivot["number"], pivot["lift"])}
    lift_vals = np.array(list(lift_by_number.values()))
    lift_mean = float(np.nanmean(lift_vals))
    lift_std  = float(np.nanstd(lift_vals) + 1e-8)

    # pattern corr
    feat = df["predicted"].apply(_combo_features).apply(pd.Series)
    feat["main_match"] = df["main_match"]
    corr = feat.corr(numeric_only=True)["main_match"].drop("main_match", errors="ignore")
    pattern_corr = {k: float(v) for k,v in corr.to_dict().items()}

    # meta learner
    X_rows, y_rows = [], []
    for _, r in df.iterrows():
        nums = r["predicted"]
        feats = _combo_features(nums)
        lift_sum = float(np.sum([lift_by_number.get(int(n), 0.0) for n in nums])) if nums else 0.0
        high_lift_cnt = int(np.sum([1 for n in nums if lift_by_number.get(int(n), 0.0) > 0]))
        row = {"confidence": float(r.get("confidence", np.nan)), "lift_sum": lift_sum, "high_lift_cnt": high_lift_cnt, **feats}
        X_rows.append(row); y_rows.append(1 if int(r["main_match"]) >= 4 else 0)
    X = pd.DataFrame(X_rows).fillna(0.0)
    y = np.array(y_rows, dtype=int) if len(y_rows)>0 else np.array([0])

    meta = None
    try:
        base = GradientBoostingClassifier(random_state=42)
        pipe = Pipeline([("scaler", StandardScaler(with_mean=False)), ("clf", base)])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        meta = CalibratedClassifierCV(pipe, method="isotonic", cv=cv).fit(X, y)
    except Exception as e:
        warnings.warn(f"Meta learner training failed ({e}); proceeding without it.")
        meta = None

    return TrainingAssets(
        conf_bins=bin_edges,
        conf_bin_rates=bin_rates,
        lift_by_number=lift_by_number,
        lift_mean=lift_mean,
        lift_std=lift_std,
        pattern_corr=pattern_corr,
        meta_pipeline=meta
    )

def _pattern_boost(nums: List[int], pattern_corr: Dict[str,float]) -> float:
    f = _combo_features(nums)
    val = 0.0
    for k, corr in pattern_corr.items():
        v = f.get(k, np.nan)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        x = float(v)
        if k == "odd_ratio":
            x_norm = x
        elif k in ("sum","range","consecutive","min_gap","max_gap"):
            x_norm = x / (x + 50.0)
        else:
            x_norm = x
        val += corr * x_norm
    return float(val)

def _number_lift_boost(nums: List[int], lift_by_number: Dict[int,float], mean: float, std: float) -> float:
    s = np.sum([lift_by_number.get(int(n), 0.0) for n in nums])
    z = (s - mean*7.0) / (std*np.sqrt(7.0) + 1e-8)
    return float(np.tanh(0.5 * z))

def _meta_prob(nums: List[int], conf: float, assets: TrainingAssets) -> float:
    if assets.meta_pipeline is None:
        return 0.5
    feats = _combo_features(nums)
    lift_sum = float(np.sum([assets.lift_by_number.get(int(n), 0.0) for n in nums])) if nums else 0.0
    high_lift_cnt = int(np.sum([1 for n in nums if assets.lift_by_number.get(int(n), 0.0) > 0]))
    row = {"confidence": float(conf if conf is not None else np.nan), "lift_sum": lift_sum, "high_lift_cnt": high_lift_cnt, **feats}
    X = pd.DataFrame([row]).fillna(0.0)
    try:
        proba = assets.meta_pipeline.predict_proba(X)[0,1]
        return float(proba)
    except Exception:
        return 0.5

def score_candidate(nums: List[int], conf: float, assets: TrainingAssets,
                    w_cal=0.4, w_meta=0.4, w_lift=0.12, w_pat=0.08) -> float:
    cal  = assets.calibrated_conf(conf)
    meta = _meta_prob(nums, conf, assets)
    liftb= _number_lift_boost(nums, assets.lift_by_number, assets.lift_mean, assets.lift_std)
    patb = float(np.tanh(_pattern_boost(nums, assets.pattern_corr)))
    return float(w_cal*cal + w_meta*meta + w_lift*liftb + w_pat*patb)

def _coverage_gain(nums: List[int], used: set) -> float:
    return float(len(set(nums) - used) / 7.0)

def enhance_predictions(predictions: List[Tuple[List[int], float]],
                        history_df: pd.DataFrame,
                        assets: TrainingAssets,
                        top_k: int = 5) -> List[Tuple[List[int], float]]:
    clean = []
    for nums, conf in predictions:
        arr = sorted(set(int(n) for n in nums if 1 <= int(n) <= 37))
        if len(arr) != 7: continue
        s = score_candidate(arr, conf, assets)
        clean.append((arr, float(s)))
    if not clean: return []

    clean.sort(key=lambda x: x[1], reverse=True)
    pool = clean[:150]

    selected, used = [], set()
    alpha, beta = 0.65, 0.35
    while len(selected) < max(top_k-2, 1) and pool:
        best_i, best_val = -1, -1e9
        for i,(nums,s) in enumerate(pool):
            val = alpha*s + beta*_coverage_gain(nums, used)
            if val > best_val:
                best_val, best_i = val, i
        if best_i < 0: break
        nums, s = pool.pop(best_i)
        selected.append((nums, s))
        used |= set(nums)

    # add 1-2 forced 6-overlap templates from history
    try:
        if "本数字" in history_df.columns:
            hist_sets = [set(v) for v in history_df["本数字"] if isinstance(v, list) and v]
            templates = set()
            for i in range(len(hist_sets)):
                for j in range(i+1, len(hist_sets)):
                    inter = hist_sets[i] & hist_sets[j]
                    if len(inter) >= 6:
                        templates.add(tuple(sorted(list(inter))[:6]))
            templates = list(templates)[:10]
            for t in templates[:2]:
                base = set(t)
                allnums = list(range(1,38))
                allnums.sort(key=lambda n: assets.lift_by_number.get(n, 0.0), reverse=True)
                pick = next((n for n in allnums if n not in base), None)
                if pick is None: continue
                cand = sorted(list(base | {pick}))
                s = score_candidate(cand, 1.0, assets)
                selected.append((cand, s))
    except Exception:
        pass

    selected.sort(key=lambda x: x[1], reverse=True)
    return selected[:top_k]

def shaped_reward(chosen_numbers: List[int], main_match: int, bonus_match: int, assets: TrainingAssets) -> float:
    if main_match == 7: base = 100.0
    elif main_match == 6 and bonus_match >= 1: base = 40.0
    elif main_match == 6: base = 20.0
    elif main_match == 5: base = 8.0
    elif main_match == 4: base = 3.0
    elif main_match == 3 and bonus_match >= 1: base = 1.0
    else: base = 0.0
    liftb = _number_lift_boost(chosen_numbers, assets.lift_by_number, assets.lift_mean, assets.lift_std)
    patb = float(np.tanh(_pattern_boost(chosen_numbers, assets.pattern_corr)))
    return float(base + 2.0*liftb + 1.0*patb)
