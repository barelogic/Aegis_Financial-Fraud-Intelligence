"""Live scoring: score raw transactions as they arrive (no batch rerun).

Same source of truth as the batch pipeline, reused directly:
- causal features via `features.features_for_row` / `update_state`
- IsolationForest + XGBoost blend, suppression rule, bands, reason
  builders from `score_transactions`, `score_accounts`, `explain`
- ring detection via `detect_rings.detect_rings` on the growing frame

Batch models are NOT reused: at startup a LiveScorer refits on history
(IF + full-window XGB; live rows are future by definition, so no OOF is
needed) and freezes the IF min-max range for single-row scaling.

Send rows in time order; per-account histories assume chronological
arrival (out-of-order rows still score, but velocity windows degrade).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from detect_rings import detect_rings
from explain import build_txn_reasons, shap_top_features
from features import (FEATURE_COLUMNS, build_account_index, features_for_row,
                      init_state, register_account, set_seed, update_state)
from score_accounts import apply_ring_bump, score_accounts
from score_transactions import (apply_suppression, default_action,
                                minmax_to_100, score_band, train_xgb)

REQUIRED_FIELDS = ("txn_id", "timestamp", "account_id", "amount")


class LiveFeatureStore:
    """Incremental causal-history state + single-row feature computation."""

    def __init__(self, history: pd.DataFrame, accounts: pd.DataFrame,
                 config: dict):
        """Warm up by replaying history (computes + keeps training feats)."""
        set_seed(int(config.get("seed", 42)))
        self.acct_info = build_account_index(accounts)
        self.state = init_state()
        self.seen_ids: set[str] = set()
        hist = history.copy()
        hist["timestamp"] = pd.to_datetime(hist["timestamp"])
        hist = hist.sort_values(["timestamp", "txn_id"]).reset_index(drop=True)
        feat_rows: list[dict] = []
        for _, r in hist.iterrows():
            tid = str(r["txn_id"])
            self.seen_ids.add(tid)
            if str(r["account_id"]) not in self.acct_info:
                register_account(self.acct_info, str(r["account_id"]))
            feat_rows.append(features_for_row(r, self.state, self.acct_info))
            seg = self.acct_info[str(r["account_id"])].get("segment",
                                                           "default")
            update_state(self.state, r, seg)
        self.train_feats = pd.DataFrame(feat_rows)

    def ingest(self, raw: dict) -> tuple[pd.Series, dict]:
        """Validate + featurize one raw transaction dict.

        Returns (row Series, feature dict). Raises ValueError on missing
        fields or duplicate txn_id. Folds the row into history state.
        """
        for field in REQUIRED_FIELDS:
            if field not in raw or raw[field] in (None, ""):
                raise ValueError(f"ingest: missing required field {field!r}")
        tid = str(raw["txn_id"])
        if tid in self.seen_ids:
            raise ValueError(f"ingest: duplicate txn_id {tid!r}")
        try:
            ts = pd.to_datetime(raw["timestamp"])
            amt = float(raw["amount"])
        except (TypeError, ValueError) as e:
            raise ValueError(f"ingest: bad timestamp/amount: {e}")
        acct = str(raw["account_id"])
        if acct not in self.acct_info:
            created = None
            if raw.get("created_at"):
                try:
                    created = pd.to_datetime(raw["created_at"])
                except (TypeError, ValueError):
                    created = None
            register_account(self.acct_info, acct, created_at=created,
                             segment=str(raw.get("segment", "default")))
        r = pd.Series({**raw, "txn_id": tid, "timestamp": ts,
                       "account_id": acct, "amount": amt})
        feat = features_for_row(r, self.state, self.acct_info)
        seg = self.acct_info[acct].get("segment", "default")
        update_state(self.state, r, seg)
        self.seen_ids.add(tid)
        return r, feat


class LiveScorer:
    """Fitted batch models + single-row scoring with frozen scaling."""

    def __init__(self, config: dict):
        self.config = config
        self.seed = int(config.get("seed", 42))
        blend = config.get("blend", {})
        self.w_if = float(blend.get("isolation_forest", 0.5))
        self.w_sup = float(blend.get("supervised", 0.5))
        self.store: LiveFeatureStore | None = None
        self.if_model = None
        self.if_lo = 0.0
        self.if_hi = 1.0
        self.sup_model = None
        self.bg = None

    def fit(self, transactions: pd.DataFrame, accounts: pd.DataFrame,
            labels: pd.DataFrame | None) -> "LiveScorer":
        """Warm history state and fit IF + full-window XGB."""
        from sklearn.ensemble import IsolationForest
        set_seed(self.seed)
        self.store = LiveFeatureStore(transactions, accounts, self.config)
        X = self.train_X
        self.if_model = IsolationForest(
            n_estimators=200,
            contamination=float(self.config.get("model", {}).get(
                "contamination", 0.05)),
            random_state=self.seed, n_jobs=-1)
        self.if_model.fit(X)
        raw = -self.if_model.decision_function(X)
        self.if_lo, self.if_hi = float(np.min(raw)), float(np.max(raw))
        if labels is not None and len(labels):
            tids = self.store.train_feats["txn_id"]
            y = (labels.set_index("txn_id")["is_fraud"].reindex(tids)
                 .fillna(0).astype(int).reset_index(drop=True))
            if int(y.sum()) > 0:
                self.sup_model = train_xgb(X, y, self.seed, self.config)
                self.bg = X.sample(n=min(200, len(X)), random_state=self.seed)
        return self

    @property
    def train_X(self) -> pd.DataFrame:
        """Training feature matrix (warmed history feats)."""
        assert self.store is not None
        return self.store.train_feats[FEATURE_COLUMNS].fillna(0).astype(float)

    def _if_part(self, X1: pd.DataFrame) -> float:
        raw = float(-self.if_model.decision_function(X1)[0])
        span = self.if_hi - self.if_lo
        if span < 1e-9:
            return 50.0
        return float(np.clip(100.0 * (raw - self.if_lo) / span, 0, 100))

    def score_one(self, raw: dict) -> tuple[dict, pd.Series]:
        """Ingest + score one raw transaction dict.

        Returns (scored event dict, raw row Series). The event is JSON-safe
        (timestamps as ISO strings, NaN as None) and carries live=True.
        """
        assert self.store is not None, "call fit() first"
        r, feat = self.store.ingest(raw)
        X1 = pd.DataFrame([feat])[FEATURE_COLUMNS].fillna(0).astype(float)
        if_s = self._if_part(X1)
        if self.sup_model is not None:
            sup_s = float(self.sup_model.predict_proba(X1)[0, 1] * 100.0)
        else:
            sup_s = if_s
        final = self.w_if * if_s + self.w_sup * sup_s
        joined = pd.DataFrame([{**feat, "amount": float(r["amount"])}])
        capped, _ = apply_suppression(joined, np.array([final]), self.config)
        score = round(float(np.clip(capped[0], 0, 100)), 1)
        th = self.config.get("risk_thresholds", {})
        band = score_band(score, th)
        shap_tops = None
        if self.sup_model is not None:
            try:
                shap_tops = shap_top_features(self.sup_model, X1, top_k=3,
                                              seed=self.seed)
            except Exception:
                shap_tops = None  # rules-only reasons; still never empty
        scored1 = pd.DataFrame([{"txn_id": feat["txn_id"],
                                 "txn_risk_score": score}])
        feats1 = pd.DataFrame([feat])
        txns1 = pd.DataFrame([r.to_dict()])
        reasons = build_txn_reasons(scored1, feats1, txns1, shap_tops)[0]
        event = {}
        for k, v in r.to_dict().items():
            if isinstance(v, pd.Timestamp):
                v = v.isoformat()
            elif isinstance(v, float) and np.isnan(v):
                v = None
            event[k] = v
        event.update({"txn_risk_score": score, "txn_risk_band": band,
                      "txn_reasons": reasons,
                      "txn_action": default_action(band), "live": True})
        return event, r


class LiveRings:
    """Incremental ring detection over batch + live scored rows."""

    def __init__(self, config: dict):
        self.config = config
        self.raw_rows: list[dict] = []
        self.scored_rows: list[dict] = []

    def add(self, raw: pd.Series, event: dict) -> None:
        """Record one scored live transaction (raw Series + event dict)."""
        self.raw_rows.append(raw.to_dict())
        self.scored_rows.append(event)

    def refresh(self, batch_scored: pd.DataFrame,
                batch_raw: pd.DataFrame) -> list[dict]:
        """Re-run ring detection over batch + live rows. Returns rings."""
        if not self.scored_rows:
            base_scored, base_raw = batch_scored, batch_raw
        else:
            live_scored = pd.DataFrame(self.scored_rows)
            live_raw = pd.DataFrame(self.raw_rows)
            base_scored = pd.concat([batch_scored, live_scored],
                                    ignore_index=True)
            base_raw = pd.concat([batch_raw, live_raw], ignore_index=True)
        return detect_rings(base_scored, pd.DataFrame(), base_raw, self.config)

    def live_accounts_frame(self, store_acct_info: dict) -> pd.DataFrame:
        """Accounts table extended with live-seen accounts."""
        rows = [{"account_id": a,
                 "created_at": (info.get("created_at")
                                if info.get("created_at") is not None else ""),
                 "home_city": "", "home_country": "",
                 "segment": info.get("segment", "default")}
                for a, info in store_acct_info.items()]
        return pd.DataFrame(rows)


def rescore_accounts(batch_scored: pd.DataFrame, live_scored_rows: list[dict],
                     accounts_df: pd.DataFrame, rings: list[dict],
                     config: dict) -> pd.DataFrame:
    """Recompute the account view over batch + live rows with ring bumps."""
    from score_accounts import apply_ring_bump
    if live_scored_rows:
        combined = pd.concat([batch_scored,
                              pd.DataFrame(live_scored_rows)],
                             ignore_index=True)
    else:
        combined = batch_scored
    accs = score_accounts(combined, None, accounts_df, None, config)
    if rings:
        # Pass combined txns so sink/collector attribution still works.
        accs = apply_ring_bump(accs, rings, config, combined)
    return accs
