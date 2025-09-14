#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optuna search (resumable) using backfill_predictions.csv to score TopK & max_sim.

Flow:
 1) Load backfill_predictions.csv (from backfill_forecasts.py)
 2) Load ground-truth CSV (loto7_prediction_evaluation_with_bonus.csv)
 3) For each trial (topk, max_sim), re-filter backfill candidates in order,
    compute per-draw best hits, and maximize the average(best_hits).
 4) Save best params to optuna_results/best_params.json (overwrite if exists)

Usage example:
  python optuna_search_resumable.py \
    --backfill backfill_predictions.csv \
    --csv loto7_prediction_evaluation_with_bonus.csv \
    --n_trials 80 \
    --out optuna_results/best_params.json \
    --storage sqlite:///optuna_results/optuna_study.db \
    --study_name loto7_optuna
"""
from __future__ import annotations
import argparse, json, os, sys, shutil, re
from pathlib import Path
from typing import List, Optional

import optuna
import pandas as pd


# ----------------------------- IO helpers -----------------------------------
def _read_csv_any(path: str) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)


# ------------------------- Diversity re-filter -------------------------------
def _enforce_final_diversity(seq_of_cands: List[List[int]], max_sim: float) -> List[List[int]]:
    """Order-preserving Jaccard diversity filter."""
    def jacc(a, b):
        A, B = set(a), set(b)
        return len(A & B) / float(len(A | B) + 1e-9)
    out: List[List[int]] = []
    for cand in seq_of_cands:
        ok = True
        for s in out:
            if jacc(cand, s) > max_sim:
                ok = False; break
        if ok:
            out.append(cand)
    return out


# -------------------------- Parse predictions --------------------------------
def _parse_pred_cell(cell: str) -> List[List[int]]:
    """
    backfill_predictions.csv の predictions 列は
      "a1/a2/.../a7; b1/b2/.../b7; ..." 形式を想定。
    セミコロンで候補区切り、スラッシュで7数字。
    """
    parts = [p.strip() for p in str(cell).split(";") if str(p).strip()]
    out: List[List[int]] = []
    for p in parts:
        nums = [int(x) for x in re.findall(r"\d+", p)]
        nums = [n for n in nums if 1 <= n <= 37]
        if len(nums) >= 7:
            cand = sorted(nums[:7])
            out.append(cand)
    return out


# ---------------------------- Ground truth -----------------------------------
def _extract_winners(row: pd.Series) -> Optional[List[int]]:
    """
    本数字7個を抽出。列名の揺れにそこそこ頑健。
    優先：本数字1..7 / 本 数字1..7 / n1..n7 / num1..num7
    最後の手段として行全体から 1..37 を走査してユニーク先頭7個。
    """
    names = list(row.index)
    items = []
    for k in names:
        m = re.match(r'^(?:本\s*数字|本数字|n|num)\s*([1-7])$', k, flags=re.I)
        if m:
            items.append((int(m.group(1)), k))
    if len(items) == 7:
        items.sort()
        try:
            vals = [int(row[k]) for _, k in items]
            if all(1 <= v <= 37 for v in vals):
                return sorted(vals)
        except Exception:
            pass
    # fallback
    vals = [int(x) for x in re.findall(r"\d+", " ".join(map(str, row.values)))]
    vals = [v for v in vals if 1 <= v <= 37]
    seen, out = set(), []
    for v in vals:
        if v not in seen:
            seen.add(v); out.append(v)
        if len(out) >= 7:
            return sorted(out[:7])
    return None


# ---------------------------- Data loaders -----------------------------------
def load_backfill(backfill_path: str) -> pd.DataFrame:
    bf = _read_csv_any(backfill_path)
    if "抽せん日" not in bf.columns or "predictions" not in bf.columns:
        raise RuntimeError("backfill_predictions.csv に '抽せん日' / 'predictions' 列が必要です。")
    bf["抽せん日"] = pd.to_datetime(bf["抽せん日"], errors="coerce")
    bf = bf.dropna(subset=["抽せん日"]).sort_values("抽せん日").reset_index(drop=True)
    bf["cand_list"] = bf["predictions"].map(_parse_pred_cell)
    # keep only non-empty candidates
    bf = bf[bf["cand_list"].map(len) > 0].reset_index(drop=True)
    return bf[["抽せん日", "cand_list"]]


def load_truth(csv_path: str) -> pd.DataFrame:
    df = _read_csv_any(csv_path)
    if "抽せん日" not in df.columns:
        for alt in ("抽選日", "date", "Date", "draw_date"):
            if alt in df.columns:
                df = df.rename(columns={alt: "抽せん日"}); break
    if "抽せん日" not in df.columns:
        raise RuntimeError("正解CSVに '抽せん日' 列が見つかりません。")
    df["抽せん日"] = pd.to_datetime(df["抽せん日"], errors="coerce")
    df = df.dropna(subset=["抽せん日"]).sort_values("抽せん日").reset_index(drop=True)
    df["winners"] = df.apply(_extract_winners, axis=1)
    df = df.dropna(subset=["winners"]).reset_index(drop=True)
    return df[["抽せん日", "winners"]]


# ---------------------------- Trial scoring ----------------------------------
def score_trial(bf_df: pd.DataFrame, truth_df: pd.DataFrame, topk: int, max_sim: float) -> float:
    merged = pd.merge(bf_df, truth_df, on="抽せん日", how="inner")
    if merged.empty:
        return 0.0

    def best_hits_for_row(row) -> int:
        cands = _enforce_final_diversity(row["cand_list"], max_sim=max_sim)
        cands = cands[: max(1, int(topk))]
        w = set(row["winners"])
        best = 0
        for c in cands:
            h = len(set(c) & w)
            if h > best:
                best = h
        return best

    hits = merged.apply(best_hits_for_row, axis=1)
    # objective: average(best_hits) + tiny regularization to avoid too small topk
    return float(hits.mean() + 1e-6 * (topk / 40.0))


# ------------------------------- CLI -----------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", required=True, help="backfill_predictions.csv (from backfill_forecasts.py)")
    ap.add_argument("--csv", required=True, help="ground-truth CSV (e.g., loto7_prediction_evaluation_with_bonus.csv)")
    ap.add_argument("--n_trials", type=int, default=80)
    ap.add_argument("--out", default="optuna_results/best_params.json")
    ap.add_argument("--storage", default="sqlite:///optuna_results/optuna_study.db")
    ap.add_argument("--study_name", default="loto7_optuna")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--checkpoint_interval", type=int, default=10)
    # search spaces
    ap.add_argument("--topk_min", type=int, default=5)
    ap.add_argument("--topk_max", type=int, default=40)
    ap.add_argument("--max_sim_min", type=float, default=0.4)
    ap.add_argument("--max_sim_max", type=float, default=0.9)
    return ap.parse_args()


# ------------------------------ main -----------------------------------------
def main():
    args = parse_args()

    # Ensure out dir exists
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    bf_df = load_backfill(args.backfill)
    truth_df = load_truth(args.csv)

    # Optuna (resume-friendly)
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner  = optuna.pruners.MedianPruner(n_warmup_steps=10)

    study = optuna.create_study(direction="maximize",
                                study_name=args.study_name,
                                storage=args.storage,
                                load_if_exists=True,
                                sampler=sampler)

    def objective(trial: optuna.Trial):
        topk    = trial.suggest_int("topk", args.topk_min, args.topk_max)
        max_sim = trial.suggest_float("max_sim", args.max_sim_min, args.max_sim_max)
        value   = score_trial(bf_df, truth_df, topk=topk, max_sim=max_sim)
        trial.report(value, step=1)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return value

    def checkpoint_cb(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        """Periodically copy sqlite DB as a checkpoint file."""
        if not args.checkpoint_interval:
            return
        if trial.number % args.checkpoint_interval != 0:
            return
        if args.storage.startswith("sqlite:///"):
            db_path = args.storage.replace("sqlite:///", "", 1)
            src = Path(db_path)
            if src.exists():
                dst = src.with_name(f"{src.stem}_ckpt_{trial.number}{src.suffix}")
                try:
                    shutil.copy2(src, dst)
                    print(f"[CKPT] Saved checkpoint DB: {dst}", flush=True)
                except Exception as e:
                    print(f"[CKPT] Failed to save checkpoint: {e}", flush=True)

    study.optimize(objective, n_trials=args.n_trials, callbacks=[checkpoint_cb], gc_after_trial=True)

    # Best result
    best = study.best_trial
    result = {
        "topk": int(best.params["topk"]),
        "max_sim": float(best.params["max_sim"]),
        # Post-diversity only tuning in this script; keep defaults for below (editable later)
        "lambda_div": 0.6,
        "temperature": 0.35,
        "best_value": float(best.value),
        "study_name": args.study_name,
        "storage": args.storage
    }

    # Overwrite if exists; create if not
    try:
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[BEST] saved to {out_path}", flush=True)
    except Exception as e:
        print(f"[ERROR] failed to write {out_path}: {e}", file=sys.stderr)
        sys.exit(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())
