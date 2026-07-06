"""
Taegliche Paper-Trading-Analyse.

Liest Paper-Trading-Logs und erstellt:
  - Tages-Metriken (Signale, Trades, Win-Rate, P&L, Profit-Faktor)
  - Signal-Verteilung ueber den Tag
  - Top/Bottom-Symbole
  - JSON-Output + optional Chart-PNG

Aufruf:
  python scripts/08_deployment/analyze_day.py --date 2026-07-04
  python scripts/08_deployment/analyze_day.py --all   # Alle Tage auf einmal
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
import matplotlib.ticker as mticker

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PAPER_DIR = _PROJECT_ROOT / "data" / "paper_trading"
_EVAL_DIR = _PROJECT_ROOT / "artifacts" / "evaluation"
_IMG_DIR = _PROJECT_ROOT / "artifacts" / "images" / "05_paper_trading"

# Dark theme
BG, CARD, TEXT, ACCENT = "#0a0e17", "#111827", "#e2e8f0", "#3b82f6"
GREEN, RED, AMBER = "#16a34a", "#dc2626", "#ea580c"


def parse_date(s: str) -> str:
    return datetime.strptime(s, "%Y-%m-%d").strftime("%Y-%m-%d")


def load_signals(date_str: str) -> pd.DataFrame | None:
    path = _PAPER_DIR / "signals" / f"signals_{date_str}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def load_orders(date_str: str) -> pd.DataFrame | None:
    path = _PAPER_DIR / "orders" / f"orders_{date_str}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_positions() -> pd.DataFrame | None:
    path = _PAPER_DIR / "positions" / "positions.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def analyze_signals(df: pd.DataFrame) -> dict:
    """Analysiert Signal-Parquet."""
    n = len(df)
    if n == 0:
        return {"n_signals": 0}

    n_positive = int(df["ensemble_signal"].sum())

    return {
        "n_signals": n,
        "n_positive_signals": n_positive,
        "signal_rate": float(n_positive / n) if n > 0 else 0.0,
        "p_mlp_mean": float(df["p_mlp"].mean()),
        "p_lstm_mean": float(df["p_lstm"].mean()),
        "p_gru_mean": float(df["p_gru"].mean()),
        "p_cnn_mean": float(df["p_cnn"].mean()),
        "p_lgb_mean": float(df["p_lgb"].mean()),
        "ensemble_score_mean": float(df["ensemble_score"].mean()) if "ensemble_score" in df.columns else 0.0,
    }


def analyze_orders(df: pd.DataFrame) -> dict:
    """Analysiert Order-CSV."""
    if df is None or len(df) == 0:
        return {"n_orders": 0, "n_buys": 0, "n_sells": 0, "n_filled": 0}

    buys = df[df["side"] == "BUY"]
    sells = df[df["side"] == "SELL"]
    filled = df[df["status"] == "FILLED"]

    return {
        "n_orders": len(df),
        "n_buys": len(buys),
        "n_sells": len(sells),
        "n_filled": len(filled),
    }


def analyze_positions(df: pd.DataFrame, date_str: str) -> dict:
    """Analysiert abgeschlossene Positionen."""
    if df is None or len(df) == 0:
        return {"n_positions": 0, "win_rate": 0.0, "total_pnl": 0.0,
                "avg_pnl_pct": 0.0, "profit_factor": 0.0, "best_symbol": None,
                "worst_symbol": None, "symbol_pnl": {}}

    # Filtere auf Tages-Eintraege
    if "entry_time" in df.columns:
        df_day = df[df["entry_time"].str.startswith(date_str)]
    else:
        df_day = df

    if len(df_day) == 0:
        return {"n_positions": 0, "win_rate": 0.0, "total_pnl": 0.0,
                "avg_pnl_pct": 0.0, "profit_factor": 0.0, "best_symbol": None,
                "worst_symbol": None, "symbol_pnl": {}}

    n = len(df_day)
    pnl_col = "pnl" if "pnl" in df_day.columns else "pnl_pct"
    winners = df_day[df_day[pnl_col] > 0]
    losers = df_day[df_day[pnl_col] <= 0]
    win_rate = len(winners) / n if n > 0 else 0.0

    total_gain = float(winners[pnl_col].sum()) if len(winners) > 0 else 0.0
    total_loss = abs(float(losers[pnl_col].sum())) if len(losers) > 0 else 0.0
    profit_factor = total_gain / total_loss if total_loss > 0 else float("inf")

    # Per-Symbol P&L
    symbol_pnl = {}
    for sym in df_day["symbol"].unique():
        sdf = df_day[df_day["symbol"] == sym]
        symbol_pnl[sym] = float(sdf[pnl_col].sum())

    best = max(symbol_pnl, key=symbol_pnl.get) if symbol_pnl else None
    worst = min(symbol_pnl, key=symbol_pnl.get) if symbol_pnl else None

    return {
        "n_positions": n,
        "win_rate": float(win_rate),
        "total_pnl": float(df_day[pnl_col].sum()),
        "avg_pnl_pct": float(df_day[pnl_col].mean()) if pnl_col == "pnl_pct" else 0.0,
        "profit_factor": float(profit_factor),
        "best_symbol": best,
        "worst_symbol": worst,
        "symbol_pnl": {k: float(v) for k, v in sorted(symbol_pnl.items(), key=lambda x: -x[1])[:10]},
    }


def plot_daily_summary(metrics: dict, date_str: str):
    """Erstellt ein 2x2 Tages-Chart."""
    _IMG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(BG)

    sig = metrics.get("signals", {})
    pos = metrics.get("positions", {})
    ord_ = metrics.get("orders", {})

    # 1) KPI-Summary (Text)
    ax = axes[0, 0]
    ax.set_facecolor(CARD)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    lines = [
        f"Signale: {sig.get('n_signals', 0):,} ({sig.get('signal_rate', 0):.1%} positiv)",
        f"Trades: {pos.get('n_positions', 0)}",
        f"Win-Rate: {pos.get('win_rate', 0):.1%}",
        f"Total P&L: ${pos.get('total_pnl', 0):,.2f}",
        f"Profit-Faktor: {pos.get('profit_factor', 0):.2f}",
        f"Orders: {ord_.get('n_orders', 0)} (Buy: {ord_.get('n_buys', 0)}, Sell: {ord_.get('n_sells', 0)})",
    ]
    for i, line in enumerate(lines):
        ax.text(1, 9 - i * 1.4, line, fontsize=13, color=TEXT, fontfamily="monospace")
    ax.set_title(f"Paper Trading - {date_str}", color=TEXT, fontsize=15, fontweight="bold", pad=12)

    # 2) Signal-Rate ueber Symbole (falls vorhanden)
    ax = axes[0, 1]
    ax.set_facecolor(CARD)
    sym_pnl = pos.get("symbol_pnl", {})
    if sym_pnl:
        symbols = list(sym_pnl.keys())[:10]
        values = list(sym_pnl.values())[:10]
        colors = [GREEN if v > 0 else RED for v in values]
        ax.barh(range(len(symbols)), values, color=colors, height=0.6)
        ax.set_yticks(range(len(symbols)))
        ax.set_yticklabels(symbols, color=TEXT)
        ax.axvline(0, color="white", linewidth=0.5)
        ax.set_title("Top/Bottom Symbole (P&L)", color=TEXT, fontsize=12)
        ax.tick_params(colors=TEXT)
        ax.set_facecolor(CARD)
        ax.xaxis.label.set_color(TEXT)

    # 3) Modell-Probabilities
    ax = axes[1, 0]
    ax.set_facecolor(CARD)
    models = ["MLP", "LSTM", "GRU", "CNN", "LGB"]
    means = [sig.get(f"p_{m.lower()}_mean", 0) for m in ["mlp", "lstm", "gru", "cnn", "lgb"]]
    colors_m = ["#2563eb", "#dc2626", "#ea580c", "#7c3aed", "#16a34a"]
    ax.bar(models, means, color=colors_m)
    ax.set_title("Durchschn. Modell-Probabilities", color=TEXT, fontsize=12)
    ax.tick_params(colors=TEXT)
    ax.set_facecolor(CARD)
    for spine in ax.spines.values(): spine.set_color("#1e293b")

    # 4) Trading-Metriken
    ax = axes[1, 1]
    ax.set_facecolor(CARD)
    metrics_trade = ["Win-Rate", "Profit-Faktor", "Avg P&L%"]
    values_t = [
        pos.get("win_rate", 0) * 100,
        min(pos.get("profit_factor", 0), 10),  # Cap at 10
        pos.get("avg_pnl_pct", 0) * 100 if pos.get("avg_pnl_pct", 0) else 0,
    ]
    colors_t = [GREEN, ACCENT, AMBER]
    ax.bar(metrics_trade, values_t, color=colors_t)
    ax.set_title("Trading-KPIs", color=TEXT, fontsize=12)
    ax.tick_params(colors=TEXT)
    ax.set_facecolor(CARD)
    for spine in ax.spines.values(): spine.set_color("#1e293b")

    plt.tight_layout()
    out_path = _IMG_DIR / f"daily_{date_str}.png"
    fig.savefig(out_path, dpi=120, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart gespeichert: {out_path}")


def run_day(date_str: str) -> dict:
    """Analysiert einen einzelnen Handelstag."""
    print(f"\n{'=' * 50}")
    print(f"ANALYSE: {date_str}")
    print(f"{'=' * 50}")

    signals = load_signals(date_str)
    orders = load_orders(date_str)
    positions = load_positions()

    sig_metrics = analyze_signals(signals) if signals is not None else {"n_signals": 0}
    ord_metrics = analyze_orders(orders)
    pos_metrics = analyze_positions(positions, date_str)

    print(f"  Signale:   {sig_metrics.get('n_signals', 0):,} "
          f"({sig_metrics.get('signal_rate', 0):.1%} positiv)")
    print(f"  Orders:    {ord_metrics.get('n_orders', 0)} "
          f"(Buy: {ord_metrics.get('n_buys', 0)}, Sell: {ord_metrics.get('n_sells', 0)})")
    print(f"  Positions: {pos_metrics.get('n_positions', 0)}")
    print(f"  Win-Rate:  {pos_metrics.get('win_rate', 0):.1%}")
    print(f"  Total P&L: ${pos_metrics.get('total_pnl', 0):,.2f}")
    print(f"  P-Faktor:  {pos_metrics.get('profit_factor', 0):.2f}")
    if pos_metrics.get("best_symbol"):
        print(f"  Best:      {pos_metrics['best_symbol']}")
    if pos_metrics.get("worst_symbol"):
        print(f"  Worst:     {pos_metrics['worst_symbol']}")

    result = {
        "date": date_str,
        "signals": sig_metrics,
        "orders": ord_metrics,
        "positions": pos_metrics,
    }

    # Save JSON
    out_dir = _EVAL_DIR / "daily"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"daily_{date_str}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"  JSON: {json_path}")

    # Chart
    try:
        plot_daily_summary(result, date_str)
    except Exception as e:
        print(f"  [WARN] Chart fehlgeschlagen: {e}")

    return result


def run_all() -> list[dict]:
    """Analysiert alle verfuegbaren Tage."""
    results = []
    signal_files = sorted(_PAPER_DIR.glob("signals/signals_*.parquet"))
    for f in signal_files:
        date_str = f.stem.replace("signals_", "")
        results.append(run_day(date_str))

    # Cumulative summary
    all_positions = load_positions()
    if all_positions is not None and len(all_positions) > 0:
        pnl_col = "pnl" if "pnl" in all_positions.columns else "pnl_pct"
        total_pnl = float(all_positions[pnl_col].sum())
        n_trades = len(all_positions)
        win_rate = float((all_positions[pnl_col] > 0).mean())

        print(f"\n{'=' * 50}")
        print(f"GESAMT ({len(results)} Handelstage)")
        print(f"{'=' * 50}")
        print(f"  Trades:    {n_trades}")
        print(f"  Win-Rate:  {win_rate:.1%}")
        print(f"  Total P&L: ${total_pnl:,.2f}")

    return results


def main():
    p = argparse.ArgumentParser(description="Paper-Trading-Tagesanalyse")
    p.add_argument("--date", type=str, help="Datum (YYYY-MM-DD)")
    p.add_argument("--all", action="store_true", help="Alle verfuegbaren Tage analysieren")
    p.add_argument("--days", type=int, default=None, help="Letzte N Tage analysieren")
    args = p.parse_args()

    _EVAL_DIR.mkdir(parents=True, exist_ok=True)

    if args.all:
        run_all()
    elif args.days:
        signal_files = sorted(_PAPER_DIR.glob("signals/signals_*.parquet"), reverse=True)
        for f in signal_files[:args.days]:
            date_str = f.stem.replace("signals_", "")
            run_day(date_str)
    elif args.date:
        run_day(parse_date(args.date))
    else:
        # Default: heute
        today = datetime.now().strftime("%Y-%m-%d")
        run_day(today)


if __name__ == "__main__":
    main()
