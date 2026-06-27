"""
Baseline: Majority-Class Predictor
----------------------------------
Berechnet die Baseline Accuracy (= immer die häufigere Klasse vorhersagen)
aus der aggregierten Klassenverteilung in class_balance.json.

Die JSON wird von 05_post_split_prep/class_balance.py erzeugt und enthält
die Breakout-Rate über ALLE 100 NASDAQ-100 Symbole und ALLE Trainings-Shards.

Ergebnis: ~50% Baseline Accuracy (49.78% Breakout-Rate).
Das Modell muss diese Marke schlagen, um Mehrwert zu zeigen.
"""

import json
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
params = yaml.safe_load(open(PROJECT_ROOT / "conf" / "params.yaml"))

balance_path = PROJECT_ROOT / params["POST_SPLIT"]["BALANCE_PATH"]

with open(balance_path, encoding="utf-8") as f:
    balance = json.load(f)

pos_ratio = balance["positive_ratio"]
baseline_acc = max(pos_ratio, 1 - pos_ratio)

print("=== Baseline (Majority Class Predictor) ===")
print(f"  Datenbasis:         ALLE 100 NASDAQ-100 Symbole (Training-Shards)")
print(f"  Training-Samples:   {balance['total_samples']:,}")
print(f"  Breakout (1):       {balance['positive_samples']:,}  ({pos_ratio:.2%})")
print(f"  Kein Breakout (0):  {balance['negative_samples']:,}  ({1 - pos_ratio:.2%})")
print(f"  Baseline Accuracy:  {baseline_acc:.2%}")
print(f"  Strategie:          Immer die häufigere Klasse vorhersagen")
print(f"  => Modell muss > {baseline_acc:.2%} erreichen, um Mehrwert zu zeigen")