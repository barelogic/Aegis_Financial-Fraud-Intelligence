"""No feature may use future data.

Compute features on the full frame, then recompute with all future rows
(shuffled) appended/perturbed: features for the earlier rows must be
unchanged.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features import FEATURE_COLUMNS, compute_features
from tests.test_scores import CFG, _toy


def test_no_future_leakage():
    txns, accs = _toy(n_accts=6, per=10)
    cutoff = txns["timestamp"].quantile(0.5)
    early = txns[txns["timestamp"] <= cutoff].copy()
    late = txns[txns["timestamp"] > cutoff].copy()

    f_early = compute_features(early, accs, CFG).set_index("txn_id").sort_index()
    # Shuffle the future rows and shift them later: earlier rows must not change.
    shuffled_future = late.sample(frac=1.0, random_state=999)
    shuffled_future["timestamp"] += pd.to_timedelta(30, unit="D")
    f_full = compute_features(
        pd.concat([early, shuffled_future]), accs, CFG
    ).set_index("txn_id").sort_index().loc[f_early.index]

    pd.testing.assert_frame_equal(f_early[FEATURE_COLUMNS],
                                  f_full[FEATURE_COLUMNS],
                                  check_exact=False, rtol=1e-9)
