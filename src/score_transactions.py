"""Transaction risk scoring: IsolationForest + XGBoost blend (0-100).

- IsolationForest on behavior-relative features = unsupervised normal-behavior signal.
- XGBoost trained on the EARLIER time window (first ~60% of days) with labels;
  later windows scored with the fitted model; earlier-window rows use
  out-of-fold predictions so no row is scored by a model that saw its label.
- Blend: final = w_if * scaled_IF + w_sup * scaled_sup (weights from config).
- Suppression rule caps legit-looking high-value txns below medium (logged).
- Ring bump (applied AFTER detect_rings): small uplift for ring members.

Writes data/outputs/transactions_scored.csv columns + txn_risk_score,
txn_risk_band, txn_reasons, txn_action.
"""
from __future__ import annotations

import random

import numpy as np
import pandas as pd

from explain import build_txn_reasons, shap_top_features
from features import FEATURE_COLUMNS, set_seed


def score_band(score: float, thresholds: dict) -> str:
    """Map a 0-100 score to low/medium/high/critical."""
    if score >= thresholds.get("critical", 85):
        return "critical"
    if score >= thresholds.get("high", 65):
        return "high"
    if score >= thresholds.get("medium", 40):
        return "medium"
    return "low"


def default_action(band: str) -> str:
    """Preliminary action per band (refined later by actions.py)."""
    return {"low": "allow", "medium": "review",
            "high": "step_up_auth", "critical": "block"}[band]


def minmax_to_100(values: np.ndarray) -> np.ndarray:
    """Min-max scale an array to 0-100 (flat input -> all 50)."""
    v = np.asarray(values, dtype=float)
    lo, hi = float(np.min(v)), float(np.max(v))
    if hi - lo < 1e-9:
        return np.full_like(v, 50.0)
    return 100.0 * (v - lo) / (hi - lo)


def isolation_scores(X: pd.DataFrame, seed: int,
                     contamination: float = 0.05) -> np.ndarray:
    """Unsupervised risk 0-100 (higher = more anomalous)."""
    from sklearn.ensemble import IsolationForest
    clf = IsolationForest(n_estimators=200, contamination=contamination,
                          random_state=seed, n_jobs=-1)
    clf.fit(X)
    # decision_function: higher = more normal -> negate to get risk.
    risk_raw = -clf.decision_function(X)
    return minmax_to_100(risk_raw)


def train_xgb(X_train: pd.DataFrame, y_train: pd.Series, seed: int,
              config: dict):
    """Train a class-weighted XGBoost classifier."""
    from xgboost import XGBClassifier
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    spw = (n_neg / max(n_pos, 1)) if n_pos else 1.0
    m = config.get("model", {})
    clf = XGBClassifier(
        n_estimators=int(m.get("n_estimators", 200)),
        max_depth=int(m.get("max_depth", 5)),
        learning_rate=float(m.get("learning_rate", 0.05)),
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        scale_pos_weight=spw, random_state=seed, n_jobs=-1,
        eval_metric="logloss", tree_method="hist",
    )
    clf.fit(X_train, y_train)
    return clf


def supervised_oof_scores(X: pd.DataFrame, y: pd.Series,
                          timestamps: pd.Series, seed: int,
                          config: dict) -> tuple[np.ndarray, object]:
    """Supervised 0-100 scores with temporal split + OOF for the train window.

    Train window = first train_window_frac of days. Rows inside it get
    out-of-fold predictions (StratifiedKFold); rows after it get predictions
    from the model fit on the full train window. Returns (scores, model).
    """
    m = config.get("model", {})
    frac = float(m.get("train_window_frac", 0.6))
    n_splits = int(m.get("oof_splits", 3))
    tmin, tmax = timestamps.min(), timestamps.max()
    cutoff = tmin + (tmax - tmin) * frac
    is_train = timestamps <= cutoff

    from sklearn.model_selection import StratifiedKFold
    oof = np.full(len(X), np.nan)
    model = None
    train_idx = np.where(is_train.values)[0]
    minority = int(min(y[is_train].sum(), (y[is_train] == 0).sum()))
    if is_train.sum() > 0 and minority >= 2:
        Xtr, ytr = X.iloc[train_idx], y.iloc[train_idx]
        n_use = max(2, min(n_splits, minority))
        skf = StratifiedKFold(n_splits=n_use, shuffle=True, random_state=seed)
        for tri, vai in skf.split(Xtr, ytr):
            fold_model = train_xgb(Xtr.iloc[tri], ytr.iloc[tri], seed, config)
            oof[train_idx[vai]] = fold_model.predict_proba(
                Xtr.iloc[vai])[:, 1] * 100.0
        model = train_xgb(Xtr, ytr, seed, config)  # final model for later rows
        if (~is_train).sum() > 0:
            oof[~is_train.values] = model.predict_proba(
                X[~is_train])[:, 1] * 100.0
    elif y.nunique() >= 2:  # degenerate window: plain KFold OOF on all rows
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        for tri, vai in skf.split(X, y):
            fold_model = train_xgb(X.iloc[tri], y.iloc[tri], seed, config)
            oof[vai] = fold_model.predict_proba(X.iloc[vai])[:, 1] * 100.0
        model = train_xgb(X, y, seed, config)
    return oof, model


def apply_suppression(df: pd.DataFrame, scores: np.ndarray,
                      config: dict) -> tuple[np.ndarray, list[str]]:
    """Cap legit-looking high-value txns below the medium threshold.

    Fires when: amount is high BUT z-score is normal AND device+city are known
    AND destination fan-in is low. Returns (capped_scores, fired_txn_ids).
    Every firing is logged by the caller.
    """
    sup = config.get("suppression", {})
    amt_hi = float(sup.get("amount_high_inr", 50000))
    zmax = float(sup.get("zscore_max", 2.0))
    fanin_max = int(sup.get("fanin_max", 1))
    cap = float(sup.get("cap_score", 39))
    out = scores.copy()
    fired: list[str] = []
    for i, (_, r) in enumerate(df.iterrows()):
        if (float(r.get("amount", 0)) >= amt_hi
                and abs(float(r.get("amount_zscore", 0))) <= zmax
                and int(r.get("is_new_device", 0)) == 0
                and int(r.get("is_new_city", 0)) == 0
                and int(r.get("dest_fanin", 0)) <= fanin_max
                and out[i] >= cap):
            out[i] = min(out[i], cap)
            fired.append(str(r["txn_id"]))
    return out, fired


def score_transactions(features: pd.DataFrame, transactions: pd.DataFrame,
                       labels: pd.DataFrame | None,
                       config: dict) -> pd.DataFrame:
    """Score every transaction 0-100 with band, reasons, and action."""
    seed = int(config.get("seed", 42))
    set_seed(seed)
    random.seed(seed)
    th = config.get("risk_thresholds", {})
    blend = config.get("blend", {})
    w_if = float(blend.get("isolation_forest", 0.5))
    w_sup = float(blend.get("supervised", 0.5))

    feats = features.sort_values(["timestamp", "txn_id"]).reset_index(drop=True)
    X = feats[FEATURE_COLUMNS].fillna(0).astype(float)
    txns = transactions.set_index("txn_id")

    # 1. Unsupervised signal on all rows.
    if_score = isolation_scores(X, seed,
                                float(config.get("model", {}).get("contamination", 0.05)))

    # 2. Supervised signal (temporal split + OOF); isolation-only if no labels.
    model = None
    if labels is not None and len(labels):
        lab = labels.set_index("txn_id")["is_fraud"].reindex(feats["txn_id"])
        # Reset to a RangeIndex so boolean masks align positionally with X.
        y = lab.fillna(0).astype(int).reset_index(drop=True)
        if y.sum() > 0:
            sup_score, model = supervised_oof_scores(
                X, y, pd.to_datetime(feats["timestamp"]), seed, config)
            sup_score = np.where(np.isnan(sup_score), if_score, sup_score)
        else:
            sup_score = if_score.copy()
    else:
        sup_score = if_score.copy()

    # 3. Blend to 0-100.
    # NOTE: ring-A txns look ordinary individually BY DESIGN; do not overfit
    # the transaction model to catch them here -- detect_rings adds ring risk
    # back onto members afterwards.
    final = w_if * if_score + w_sup * sup_score

    # 4. Suppression rule for legit high-value transactions.
    joined = feats.copy()
    joined["amount"] = joined["txn_id"].map(
        lambda t: float(txns.loc[t, "amount"]) if t in txns.index else 0.0)
    final, fired = apply_suppression(joined, np.asarray(final), config)
    print(f"[score_transactions] suppression fired on {len(fired)} txn(s).")
    for tid in fired[:20]:
        print(f"  suppressed: {tid}")
    if len(fired) > 20:
        print(f"  ... and {len(fired) - 20} more")

    # 5. Explanations: SHAP top-3 (if a supervised model exists) + rules.
    shap_tops = None
    if model is not None:
        try:
            shap_tops = shap_top_features(model, X, top_k=3, seed=seed)
        except Exception as e:
            print(f"[score_transactions] SHAP failed ({e}); rule reasons only.")
            shap_tops = None

    scored = feats[["txn_id"]].copy()
    scored["txn_risk_score"] = np.clip(np.round(final, 1), 0, 100)
    scored["txn_risk_band"] = scored["txn_risk_score"].map(
        lambda s: score_band(float(s), th))
    scored["txn_reasons"] = build_txn_reasons(scored, feats, transactions,
                                              shap_tops)
    scored["txn_action"] = scored["txn_risk_band"].map(default_action)

    # 6. Merge back ALL original transaction columns (fixed schema).
    out = transactions.merge(scored, on="txn_id", how="left")
    out["txn_risk_score"] = out["txn_risk_score"].fillna(0).clip(0, 100)
    out["txn_risk_band"] = out["txn_risk_band"].fillna("low")
    out["txn_reasons"] = out["txn_reasons"].fillna(
        "Consistent with this account's normal spending pattern")
    out["txn_action"] = out["txn_action"].fillna("allow")
    cols = (list(transactions.columns) + ["txn_risk_score", "txn_risk_band",
                                          "txn_reasons", "txn_action"])
    return out[cols].sort_values("timestamp").reset_index(drop=True)


def apply_ring_bump_to_transactions(scored_txns: pd.DataFrame,
                                    rings: list[dict],
                                    config: dict) -> pd.DataFrame:
    """Uplift transaction scores for ring members (called after detect_rings)."""
    bump = float(config.get("account_scoring", {}).get("ring_txn_bump", 10))
    gate = float(config.get("risk_thresholds", {}).get("high", 65))
    member_to_ring: dict[str, str] = {}
    for r in rings:
        # Gate on ring risk, same as the account bump: low-evidence groups
        # must not inflate innocent transactions.
        if float(r.get("ring_risk_score", 0)) < gate:
            continue
        for a in r.get("accounts", []):
            member_to_ring[a] = r["ring_id"]
    out = scored_txns.copy()
    for i, (_, row) in enumerate(out.iterrows()):
        ring = member_to_ring.get(row["account_id"])
        if ring:
            out.loc[out.index[i], "txn_risk_score"] = min(
                99.0, float(row["txn_risk_score"]) + bump)
            prev = str(row.get("txn_reasons", ""))
            extra = f"Account is a member of {ring} (coordinated-ring link)"
            if ring not in prev:
                out.loc[out.index[i], "txn_reasons"] = (
                    f"{prev}; {extra}" if prev else extra)
    th = config.get("risk_thresholds", {})
    out["txn_risk_band"] = out["txn_risk_score"].map(
        lambda s: score_band(float(s), th))
    return out
