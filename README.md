# Real-Time Financial Fraud Intelligence

7-hour hackathon project. Laptop-only, no GPU, no paid APIs, no internet at runtime.

## Quickstart

```bash
pip install -r requirements.txt
python run_all.py --sample        # fast test on 2,000 rows
python run_all.py                 # full run
python run_all.py --skip-generate # reuse existing data/raw
python run_all.py --seed 123
```

## Pipeline

`run_all.py` runs, in order:

1. `generate_data` — synthetic transactions/accounts + ground truth
2. `features` — behavior-relative, strictly causal features (only data
   before each transaction's timestamp)
3. `score_transactions` — IsolationForest + XGBoost blend (0-100) with
   suppression rule + SHAP/rule explanations
4. `score_accounts` — blend of max txn score + top-3 mean (+ ring bump later)
5. `detect_rings` — shared-device/IP/destination graph + community detection;
   ring risk is pushed back onto member accounts/transactions
6. `actions` — recommended action per flagged item
7. `report` — `data/outputs/report.md`
8. `evaluate` — `data/outputs/metrics.json`

## Key design notes

- Ring A transactions look ordinary individually on purpose; transaction
  scores alone will NOT catch ring A. Ring detection pushes risk back down
  onto member accounts/transactions afterwards.
- Suppression rule protects legit high-value transactions: high amount but
  normal z-score + known device/city + no destination fan-in caps the score
  below the medium threshold (every firing is logged).
- Every score ships with plain-English reasons (SHAP top-3 + rules).
  Low-risk rows get a reason too (e.g. "Consistent with ... normal spending").
- `config.yaml:column_mapping` maps logical schema names to dataset columns
  so a different dataset can be plugged in without code changes.

## Layout

See repo tree. Outputs land in `data/outputs/` with the fixed schemas.
