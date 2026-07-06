"""
Data-Leakage-Check: Vergleicht Paper-Trading-Performance mit historischen Metriken.

Paper-Trading-Daten sind die ultimative Out-of-Sample-Validierung:
- Kein Shuffle, keine geteilten Zeitraeume
- Echte Market-Bedingungen (Slippage, Fuellzeit, etc.)
- Wenn Paper-Performance >20% schlechter als historisch: Leakage-Verdacht

Aufruf:
  python scripts/08_deployment/leakage_check.py --days 3
  python scripts/08_deployment/leakage_check.py --date 2026-07-04
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PAPER_DIR = _PROJECT_ROOT / "data" / "paper_trading"
_EVAL_DIR = _PROJECT_ROOT / "artifacts" / "evaluation"
_IMG_DIR = _PROJECT_ROOT / "artifacts" / "images" / "05_paper_trading"

BG, CARD, TEXT = "#0a0e17", "#111827", "#e2e8f0"
GREEN, RED, AMBER, BLUE = "#16a34a", "#dc2626", "#ea580c", "#3b82f6"


def load_historical_metrics() -> dict:
    """Laedt die historischen Ensemble-Metriken."""
    path = _EVAL_DIR / "ensemble_evaluation.json"
    if not path.exists():
        print(f"[WARN] ensemble_evaluation.json nicht gefunden: {path}")
        return {}

    with open(path) as f:
        data = json.load(f)

    # Extrahiere Winner-Strategie
    winner = data.get("winner", "finder_majority")
    strat = data.get("strategies", {}).get(winner, {})

    return {
        "winner": winner,
        "n_test_samples": data.get("metadata", {}).get("n_test_samples", 0),
        "accuracy": strat.get("accuracy", 0),
        "classification_report": strat.get("classification_report", {}),
        "trading_metrics": strat.get("trading_metrics", {}),
        "signal_rate": strat.get("signal_rate", 0),
        "confusion_matrix": strat.get("confusion_matrix", {}),
    }


def load_paper_signals() -> pd.DataFrame:
    """Laedt alle Paper-Trading-Signale."""
    files = sorted(_PAPER_DIR.glob("signals/signals_*.parquet"))
    if not files:
        return pd.DataFrame()

    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


def load_paper_positions() -> pd.DataFrame | None:
    """Laedt alle Paper-Trading-Positions."""
    path = _PAPER_DIR / "positions" / "positions.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def compute_paper_metrics(signals: pd.DataFrame, positions: pd.DataFrame | None) -> dict:
    """Berechnet Metriken aus Paper-Trading-Daten."""
    n = len(signals)
    if n == 0:
        return {"n_samples": 0, "warning": "Keine Paper-Trading-Daten vorhanden"}

    # Signal-Metriken (nur wo target bekannt ist - d.h. 30 Min nach Signal)
    # Da wir im Live-Trading target=-1 loggen (noch nicht bekannt),
    # koennen wir Precision/Recall erst berechnen wenn wir das tatsaechliche
    # Breakout-Label nach 30 Minuten wissen.
    #
    # Fuer den Leakage-Check vergleichen wir:
    # 1. Signal-Rate (Paper vs. Historisch)
    # 2. Durchschnittliche Modell-Probabilities
    # 3. P&L aus tatsaechlichen Trades (beste Metrik!)

    signal_rate = float(signals["ensemble_signal"].mean()) if "ensemble_signal" in signals.columns else 0.0

    prob_means = {}
    for col in ["p_mlp", "p_lstm", "p_gru", "p_cnn", "p_lgb"]:
        if col in signals.columns:
            prob_means[col] = float(signals[col].mean())

    # Trading-Metriken aus Positionen
    trading = {}
    if positions is not None and len(positions) > 0:
        pnl_col = "pnl" if "pnl" in positions.columns else "pnl_pct"
        trading["n_trades"] = len(positions)
        trading["win_rate"] = float((positions[pnl_col] > 0).mean())
        trading["total_pnl"] = float(positions[pnl_col].sum())
        trading["avg_pnl"] = float(positions[pnl_col].mean())

        winners = positions[positions[pnl_col] > 0]
        losers = positions[positions[pnl_col] <= 0]
        total_gain = float(winners[pnl_col].sum()) if len(winners) > 0 else 0.0
        total_loss = abs(float(losers[pnl_col].sum())) if len(losers) > 0 else 0.0
        trading["profit_factor"] = float(total_gain / total_loss) if total_loss > 0 else float("inf")

    return {
        "n_samples": int(n),
        "n_trading_days": int(signals["timestamp"].nunique() if "timestamp" in signals.columns else 0),
        "signal_rate": signal_rate,
        "prob_means": prob_means,
        "trading": trading,
    }


def compare_and_report(historical: dict, paper: dict) -> dict:
    """Vergleicht historische mit Paper-Metriken."""
    if "warning" in paper:
        return {"status": "no_data", "warning": paper["warning"]}

    hist_signal_rate = historical.get("signal_rate", 0)
    paper_signal_rate = paper.get("signal_rate", 0)

    hist_trades = historical.get("trading_metrics", {})
    paper_trades = paper.get("trading", {})

    hist_win_rate = hist_trades.get("win_rate", 0)
    paper_win_rate = paper_trades.get("win_rate", 0)
    hist_pf = hist_trades.get("profit_factor", 0)
    paper_pf = paper_trades.get("profit_factor", 0)

    # Deltas berechnen
    signal_delta = paper_signal_rate - hist_signal_rate
    wr_delta = paper_win_rate - hist_win_rate
    pf_delta = paper_pf - hist_pf if hist_pf < float("inf") else 0.0

    # Warnung wenn >20% Abweichung (relativ)
    wr_warning = abs(wr_delta) > 0.20 * hist_win_rate if hist_win_rate > 0 else False
    pf_warning = abs(pf_delta) > 0.30 * hist_pf if hist_pf > 0 else False

    status = "ok"
    warnings = []
    if wr_warning:
        status = "warning"
        warnings.append(f"Win-Rate Abweichung: hist={hist_win_rate:.1%} paper={paper_win_rate:.1%} (delta={wr_delta:+.1%})")
    if pf_warning:
        status = "warning"
        warnings.append(f"Profit-Faktor Abweichung: hist={hist_pf:.2f} paper={paper_pf:.2f}")

    if status == "warning":
        status = "LEAKAGE_VERDACHT" if pf_delta < -0.5 else "WARNING"

    result = {
        "status": status,
        "warnings": warnings,
        "comparison": {
            "historical": {
                "signal_rate": hist_signal_rate,
                "win_rate": hist_win_rate,
                "profit_factor": hist_pf,
                "n_samples": historical.get("n_test_samples", 0),
            },
            "paper": {
                "signal_rate": paper_signal_rate,
                "win_rate": paper_win_rate,
                "profit_factor": paper_pf,
                "n_samples": paper.get("n_samples", 0),
                "n_trades": paper_trades.get("n_trades", 0),
            },
            "deltas": {
                "signal_rate_pp": float(signal_delta),
                "win_rate_pp": float(wr_delta),
                "profit_factor": float(pf_delta),
            },
        },
    }

    return result


def plot_comparison(historical: dict, paper: dict, result: dict):
    """Erstellt Vergleichs-Chart: Historisch vs. Paper."""
    _IMG_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor(BG)

    metrics = ["Signal-Rate", "Win-Rate", "Profit-Faktor"]
    hist_vals = [
        historical.get("signal_rate", 0) * 100,
        historical.get("trading_metrics", {}).get("win_rate", 0) * 100,
        min(historical.get("trading_metrics", {}).get("profit_factor", 0), 10),
    ]
    paper_vals = [
        paper.get("signal_rate", 0) * 100,
        paper.get("trading", {}).get("win_rate", 0) * 100,
        min(paper.get("trading", {}).get("profit_factor", 0), 10),
    ]

    x = np.arange(len(metrics))
    width = 0.35

    for i, ax in enumerate(axes):
        ax.set_facecolor(CARD)
        vals = [hist_vals[i], paper_vals[i]]
        colors = [BLUE, GREEN]
        ax.bar(["Historisch", "Paper"], vals, color=colors, width=0.5)
        ax.set_title(metrics[i], color=TEXT, fontsize=13, fontweight="bold")
        ax.tick_params(colors=TEXT)
        for spine in ax.spines.values(): spine.set_color("#1e293b")
        # Annotate values
        for j, v in enumerate(vals):
            ax.text(j, v + max(vals)*0.02, f"{v:.1f}{'%' if i<2 else ''}",
                    ha="center", color=TEXT, fontsize=11)

    status = result.get("status", "ok")
    status_color = GREEN if status == "ok" else RED if "LEAKAGE" in status else AMBER
    fig.suptitle(f"Leakage-Check: {status}", color=status_color, fontsize=16, fontweight="bold", y=1.02)

    plt.tight_layout()
    out = _IMG_DIR / "leakage_comparison.png"
    fig.savefig(out, dpi=120, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart: {out}")


def main():
    p = argparse.ArgumentParser(description="Data-Leakage-Check: Paper vs. Historisch")
    p.add_argument("--days", type=int, help="Letzte N Paper-Trading-Tage vergleichen")
    p.add_argument("--date", type=str, help="Einzelnes Paper-Trading-Datum")
    args = p.parse_args()

    print(f"\n{'=' * 55}")
    print("DATA-LEAKAGE-CHECK: Paper vs. Historisch")
    print(f"{'=' * 55}")

    # Historische Metriken laden
    historical = load_historical_metrics()
    if not historical:
        print("[ERROR] Keine historischen Metriken gefunden.")
        return

    hist_winner = historical.get("winner", "finder_majority")
    print(f"\nHistorisch ({hist_winner}):")
    print(f"  Samples:    {historical.get('n_test_samples', 0):,}")
    print(f"  Signal-Rate: {historical.get('signal_rate', 0):.1%}")
    tm = historical.get("trading_metrics", {})
    print(f"  Win-Rate:   {tm.get('win_rate', 0):.1%}")
    print(f"  P-Faktor:   {tm.get('profit_factor', 0):.2f}")

    # Paper-Daten laden
    signals = load_paper_signals()
    positions = load_paper_positions()

    # Auf Datum filtern
    if args.date and "timestamp" in signals.columns:
        signals = signals[signals["timestamp"].astype(str).str.startswith(args.date)]

    paper_metrics = compute_paper_metrics(signals, positions)
    print(f"\nPaper Trading:")
    if "warning" in paper_metrics:
        print(f"  {paper_metrics['warning']}")
    else:
        print(f"  Samples:    {paper_metrics.get('n_samples', 0):,}")
        print(f"  Trading-Tage: {paper_metrics.get('n_trading_days', 0)}")
        print(f"  Signal-Rate: {paper_metrics.get('signal_rate', 0):.1%}")
        pt = paper_metrics.get("trading", {})
        print(f"  Trades:     {pt.get('n_trades', 0)}")
        print(f"  Win-Rate:   {pt.get('win_rate', 0):.1%}")
        print(f"  P-Faktor:   {pt.get('profit_factor', 0):.2f}")
        print(f"  Total P&L:  ${pt.get('total_pnl', 0):,.2f}")

    # Vergleich
    result = compare_and_report(historical, paper_metrics)
    print(f"\n{'=' * 55}")
    status_emoji = "PASS" if result["status"] == "ok" else "WARN" if result["status"] == "warning" else "FAIL"
    print(f"ERGEBNIS: {status_emoji} ({result['status']})")
    print(f"{'=' * 55}")

    comp = result.get("comparison", {}).get("deltas", {})
    print(f"  Signal-Rate Delta:  {comp.get('signal_rate_pp', 0):+.1%}")
    print(f"  Win-Rate Delta:     {comp.get('win_rate_pp', 0):+.1%}")
    print(f"  Profit-Faktor Delta: {comp.get('profit_factor', 0):+.2f}")

    for w in result.get("warnings", []):
        print(f"  [!] {w}")

    # JSON speichern
    out_path = _EVAL_DIR / "leakage_check.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "historical": historical,
            "paper": paper_metrics,
            "result": result,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  JSON: {out_path}")

    # Chart
    try:
        plot_comparison(historical, paper_metrics, result)
    except Exception as e:
        print(f"  [WARN] Chart fehlgeschlagen: {e}")


if __name__ == "__main__":
    main()
