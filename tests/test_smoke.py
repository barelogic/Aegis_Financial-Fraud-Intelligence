"""Smoke test: every pipeline module must import."""
import importlib


def test_all_modules_import():
    for mod in ["generate_data", "features", "score_transactions",
                "score_accounts", "detect_rings", "explain", "actions",
                "report", "evaluate", "simulate_stream"]:
        importlib.import_module(mod)
