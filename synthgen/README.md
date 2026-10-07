# Synthetic Fraud Dataset — Agent Handover Notes

Isolated notes dir. The implementation itself lives in `src/generate_data.py`
and `tests/test_generate.py` (repo root relative). This directory contains
only documentation so it cannot interfere with the pipeline.

## What was done

Rewrote `src/generate_data.py` (`generate_all(raw_dir, seed=42, config=None,
n_accounts=560, n_days=30)`, same signature as before so `run_all.py` is
unaffected) plus new `tests/test_generate.py` (9 tests, all passing; full
suite 12 passed).

Reference run (seed 42): **607 accounts, 19,998 txns (191 fraud)**, INR,
30 days from 2025-01-01. Verified deterministic (two runs byte-identical).

## Output schemas (exact column order, must not change)

- `transactions.csv`: txn_id,timestamp,account_id,amount,currency,
  merchant_id,merchant_category,item_id,item_category,channel,device_id,
  ip_address,city,country,txn_type,dest_account_id
- `accounts.csv`: account_id,created_at,home_city,home_country,segment
- `ground_truth_txn.csv`: txn_id,is_fraud,ring_id,scenario
- `ground_truth_accounts.csv`: account_id,is_fraud_account,ring_id,scenario

Every txn has a ground-truth row. Hard negatives use their `hard_neg_*`
scenario with `is_fraud=0`.

## Scenario inventory (seed-42 counts)

| scenario | accounts | fraud txns | notes |
|---|---|---|---|
| legit | 532 | 0 | base normals (segments below) |
| ring_a | 10 | 40 | core stealth; ring_id RING_A |
| ring_a_sink | 2 | 0 | sinks acc_sink_A1/A2 (NOT counted as members) |
| ring_s | 6 | 24 | stretch; shared dev + 192.168.77/24; ring_id RING_S |
| ring_s_sink | 2 | 0 | sinks acc_sink_S1/S2 |
| ring_b_ato | 5 | 12 | takeover of long-standing accounts; sink acc_sink_B |
| lone_fraud | 4 | 40 | bursts + 5–12x amount spikes |
| behavior_change | 3 | 75 | normal 20d, switched pattern after |
| hard_neg_laptop | 4 | 0 | one 85,000 INR LAPTOP_85000 each |
| hard_neg_supplier | 4 | 0 | weekly 60k–150k supplier transfers |
| hard_neg_travel | 4 | 0 | foreign city + new device, normal amounts |
| hard_neg_family_device | 9 | 0 | 3 families × 3, one device each |
| hard_neg_shared_ip | 8 | 0 | 2 groups × 4, one shared IP each |
| hard_neg_festival | 8 | 0 | days 13–19 spike (plus mild global boost, still legit) |
| hard_neg_new_legit | 5 | 0 | created inside window, 3–8 normal txns |

Segments: salaried (med 2500, hrs 8–22), student (700, 10–23),
small_business (18000, 9–20), retiree (1500, 7–20); lognormal sigma
0.6/0.7/0.8/0.6. Recurrents: SALARY_CREDIT, RENT_MONTHLY, SUBSCRIPTION_*.
Ages: created_at = start − rand(60, 1825)d.

## Ring A guarantees (the tricky part)

- 10 distinct cities, mixed segments, unique device + unique 172.16.(11–20)/24
  each — zero shared device or /24 by construction.
- Distinct merchant (`m_A_*`) and shuffled channel per member.
- Only links: `SKU_GIFTCARD_RARE_7714` then `SKU_ELEC_RARE_3391` (never used
  by normal traffic), same order, inside one 48h window (day 22 + 0–36h
  offset), then 2 transfers (one to EACH sink, so both sinks shared by all 10).
- Ordinary-looking: purchase/transfer amounts clipped to [0.3×, 3.5×] segment
  median (inside legit p0.5–p99.5), segment-normal hours, fraud window
  (±6h) excluded from background txns, ≥1h spacing between fraud txns.

## Tests → requirement map

`tests/test_generate.py` (all generate into pytest `tmp_path`, never
`data/raw/`):

1. schema columns present (+ all INR)
2. no duplicate txn_id; gt covers exactly all txns
3. dataset scale 550–700 accounts / 15k–25k txns
4. ring_a == exactly 10 accounts (filter `scenario == "ring_a"`, NOT ring_id,
   which also covers the 2 sinks)
5. ring_a shares no device or /24 (subnet = first 3 octets)
6. ring_a purchase amounts within per-segment legit p0.5–p99.5
7. ≥2 rare items (≤2 legit buyers, ≥8 ring buyers) + ≥1 dest shared by ≥8 + same order
8. family accounts `is_fraud_account == 0`, all their txns `is_fraud == 0`,
   and they genuinely share devices
9. ≥25 hard negatives, all 7 `hard_neg_*` scenarios present, all clean

## How to run

```bash
.venv/bin/python -m pytest tests/test_generate.py -q   # isolated, safe
.venv/bin/python -m pytest tests/ -q                   # full suite
.venv/bin/python src/generate_data.py --raw-dir /tmp/opencode/out --seed 42 --plot out.png
.venv/bin/python -c "import sys; sys.path.insert(0,'src'); from generate_data import save_amount_scatter; save_amount_scatter('txns.csv', out_path='s.png')"
```

Do NOT point `--raw-dir` at `data/raw/` unless you intend to replace the
pipeline's current input. Other working-tree modifications
(`detect_rings.py`, `evaluate.py`, …) predate this work — only
`src/generate_data.py` + `tests/test_generate.py` were touched here.
