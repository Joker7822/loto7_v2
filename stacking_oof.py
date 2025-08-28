
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge

def fit_oof_stacking(base_preds, y, n_splits=5, random_state=42):
    """
    Fit OOF stacking meta-models (Ridge) per position.
    Parameters
    ----------
    base_preds : dict
        Mapping from model name to ndarray of shape (N, 7) with base predictions.
    y : ndarray (N, 7)
        Ground-truth targets per position.
    n_splits : int
        Number of KFold splits.
    random_state : int
        Random seed.

    Returns
    -------
    meta_models : list of 7 trained Ridge models
    oof_pred : ndarray (N, 7) with out-of-fold meta predictions
    model_names : list of str
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    N = y.shape[0]
    model_names = sorted(base_preds.keys())
    X_stack = np.hstack([base_preds[k] for k in model_names])
    oof_pred = np.zeros_like(y, dtype=float)
    meta_models = []
    for pos in range(7):
        Xp = X_stack[:, pos::7]
        yp = y[:, pos]
        fold_pred = np.zeros(N, dtype=float)
        for train_idx, valid_idx in kf.split(Xp):
            model = Ridge(alpha=1.0, random_state=random_state)
            model.fit(Xp[train_idx], yp[train_idx])
            fold_pred[valid_idx] = model.predict(Xp[valid_idx])
        meta = Ridge(alpha=1.0, random_state=random_state).fit(Xp, yp)
        meta_models.append(meta)
        oof_pred[:, pos] = fold_pred
    return meta_models, oof_pred, model_names

def predict_with_meta(meta_models, base_preds_row):
    """
    Predict with trained meta models for a single row of base predictions.
    base_preds_row: dict[name] -> np.array(7,)
    """
    model_names = sorted(base_preds_row.keys())
    row_vec = np.hstack([base_preds_row[k] for k in model_names])
    out = []
    for pos in range(7):
        seg = row_vec[pos::7].reshape(1, -1)
        out.append(float(meta_models[pos].predict(seg)[0]))
    return np.array(out)
