"""
Data-Logger: Persistiert alle Paper-Trading-Rohdaten fuer spaetere Analyse.

Run-basiertes Logging:
  - Jeder Start erzeugt ein eigenes Verzeichnis: data/paper_trading/YYYY-MM-DD_run-NNN/
  - Kein Ueberschreiben bei Restart oder Crash
  - run_info.json enthaelt ALLE Parameter → vollstaendig reproduzierbar
  - run_exit.json dokumentiert, wie/warum der Run endete

Verzeichnisstruktur:
  data/paper_trading/
  ├── 2026-07-06_run-001/
  │   ├── run_info.json       ← Parameter + Startzeit
  │   ├── run_exit.json       ← Endzeit + Exit-Grund (wird beim Shutdown geschrieben)
  │   ├── bars.parquet
  │   ├── signals.parquet
  │   ├── orders.csv
  │   └── positions.parquet
  ├── 2026-07-06_run-002/     ← Neustart am gleichen Tag
  └── ...
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


class DataLogger:
    """Run-basiertes Logging aller Trading-Daten."""

    def __init__(self, base_dir: Path = None, params: dict = None):
        if base_dir is None:
            base_dir = Path(__file__).resolve().parents[2] / "data" / "paper_trading"
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Naechste Run-Nummer fuer heute finden
        today = datetime.now().strftime("%Y-%m-%d")
        existing = sorted(self.base_dir.glob(f"{today}_run-*"))
        run_num = len(existing) + 1
        self.run_name = f"{today}_run-{run_num:03d}"
        self.run_dir = self.base_dir / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Parameter speichern
        self.params = params or {}
        self._start_time = datetime.now()

        # Puffer
        self._bar_buffer: list[dict] = []
        self._signal_rows: list[dict] = []
        self._order_rows: list[dict] = []

        # Run-Info sofort schreiben
        self._write_run_info()

        print(f"[DataLogger] Run: {self.run_name}")

    # ---- Run Metadata --------------------------------------------------

    def _write_run_info(self):
        """Schreibt run_info.json mit allen Parametern und Startzeit."""
        info = {
            "run_name": self.run_name,
            "started_at": self._start_time.isoformat(),
            "started_at_unix": self._start_time.timestamp(),
            "params": self.params,
        }
        path = self.run_dir / "run_info.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False, default=str)

    def write_run_exit(self, reason: str = "normal"):
        """Schreibt run_exit.json beim Beenden des Runs.

        Args:
            reason: "normal" | "crash" | "manual_stop" | "market_close"
        """
        exit_info = {
            "run_name": self.run_name,
            "started_at": self._start_time.isoformat(),
            "ended_at": datetime.now().isoformat(),
            "duration_minutes": round((datetime.now() - self._start_time).total_seconds() / 60, 1),
            "exit_reason": reason,
        }
        path = self.run_dir / "run_exit.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(exit_info, f, indent=2, ensure_ascii=False, default=str)

    # ---- Bars ----------------------------------------------------------

    def log_bar(self, symbol: str, bar: dict):
        """Puffert eine empfangene Bar."""
        ts = bar.get("timestamp")
        self._bar_buffer.append({
            "symbol": symbol,
            "timestamp": ts,
            "open": bar.get("open"),
            "high": bar.get("high"),
            "low": bar.get("low"),
            "close": bar.get("close"),
            "volume": bar.get("volume"),
            "vwap": bar.get("vwap"),
        })

    def flush_bars(self):
        """Schreibt gepufferte Bars als Parquet."""
        if not self._bar_buffer:
            return
        df = pd.DataFrame(self._bar_buffer)
        path = self.run_dir / "bars.parquet"
        if path.exists():
            existing = pd.read_parquet(path)
            df = pd.concat([existing, df], ignore_index=True)
        df.to_parquet(path, index=False)
        self._bar_buffer.clear()

    # ---- Signals -------------------------------------------------------

    def log_signal(
        self,
        timestamp,
        symbol: str,
        target: int,
        p_mlp: float,
        p_lstm: float,
        p_gru: float,
        p_cnn: float,
        p_lgb: float,
        ensemble_score: float,
        ensemble_signal: int,
    ):
        """Loggt ein Ensemble-Signal mit allen 5 Modell-Probabilities."""
        self._signal_rows.append({
            "timestamp": timestamp,
            "symbol": symbol,
            "target": target,
            "p_mlp": p_mlp,
            "p_lstm": p_lstm,
            "p_gru": p_gru,
            "p_cnn": p_cnn,
            "p_lgb": p_lgb,
            "ensemble_score": ensemble_score,
            "ensemble_signal": ensemble_signal,
        })

    def flush_signals(self):
        """Schreibt alle Signal-Rows als Parquet."""
        if not self._signal_rows:
            return
        df = pd.DataFrame(self._signal_rows)
        path = self.run_dir / "signals.parquet"
        if path.exists():
            existing = pd.read_parquet(path)
            df = pd.concat([existing, df], ignore_index=True)
        df.to_parquet(path, index=False)
        self._signal_rows.clear()

    # ---- Orders --------------------------------------------------------

    def log_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        qty: float,
        order_type: str,
        status: str,
        filled_price: float = None,
        filled_at: str = None,
    ):
        """Loggt ein Order-Event."""
        self._order_rows.append({
            "timestamp": datetime.now().isoformat(),
            "order_id": str(order_id),
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "type": order_type,
            "status": status,
            "filled_price": filled_price,
            "filled_at": filled_at,
        })

    def flush_orders(self):
        """Schreibt Order-Logs als CSV."""
        if not self._order_rows:
            return
        path = self.run_dir / "orders.csv"
        file_exists = path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp", "order_id", "symbol", "side", "qty",
                "type", "status", "filled_price", "filled_at",
            ])
            if not file_exists:
                writer.writeheader()
            writer.writerows(self._order_rows)
        self._order_rows.clear()

    # ---- Positions -----------------------------------------------------

    def log_position(self, symbol: str, entry_time, entry_price: float,
                     exit_time=None, exit_price: float = None,
                     pnl: float = None, pnl_pct: float = None,
                     exit_reason: str = ""):
        """Loggt eine abgeschlossene Position.

        Args:
            exit_reason: \"take_profit\", \"trailing_stop\", \"stop_loss\",
                         \"time_stop\", \"ratchet_exit\", \"signal_collapse\"
        """
        row = {
            "symbol": symbol,
            "entry_time": entry_time,
            "entry_price": entry_price,
            "exit_time": exit_time,
            "exit_price": exit_price,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "exit_reason": exit_reason,
        }
        path = self.run_dir / "positions.parquet"
        df = pd.DataFrame([row])
        if path.exists():
            existing = pd.read_parquet(path)
            df = pd.concat([existing, df], ignore_index=True)
        df.to_parquet(path, index=False)

    # ---- End-of-Day / Flush --------------------------------------------

    def end_of_day(self):
        """Flusht alle Puffer."""
        self.flush_bars()
        self.flush_signals()
        self.flush_orders()
        print(f"[DataLogger] Puffer geleert → {self.run_dir}")

    # ---- Summary -------------------------------------------------------

    def get_summary(self) -> dict:
        """Liest die Daten dieses Runs und erstellt eine Zusammenfassung."""
        summary = {
            "run_name": self.run_name,
            "n_signals": 0,
            "n_orders": 0,
            "n_positions": 0,
            "total_pnl": 0.0,
        }

        sig_path = self.run_dir / "signals.parquet"
        if sig_path.exists():
            summary["n_signals"] = len(pd.read_parquet(sig_path))

        ord_path = self.run_dir / "orders.csv"
        if ord_path.exists():
            summary["n_orders"] = len(pd.read_csv(ord_path))

        pos_path = self.run_dir / "positions.parquet"
        if pos_path.exists():
            df = pd.read_parquet(pos_path)
            summary["n_positions"] = len(df)
            if "pnl" in df.columns:
                summary["total_pnl"] = float(df["pnl"].sum())
                summary["win_rate"] = float((df["pnl"] > 0).mean())
            if "pnl_pct" in df.columns:
                summary["avg_pnl_pct"] = float(df["pnl_pct"].mean())
                summary["total_pnl_pct"] = float(df["pnl_pct"].sum())

        return summary

    @staticmethod
    def list_runs(base_dir: Path = None) -> list[dict]:
        """Listet alle gespeicherten Runs mit ihren Metadaten auf.

        Nutzung:
          for run in DataLogger.list_runs():
              print(f"{run['run_name']}: {run['params'].get('tp_pct')}")
        """
        if base_dir is None:
            base_dir = Path(__file__).resolve().parents[2] / "data" / "paper_trading"
        base_dir = Path(base_dir)

        runs = []
        for run_dir in sorted(base_dir.glob("*_run-*")):
            info_path = run_dir / "run_info.json"
            exit_path = run_dir / "run_exit.json"
            if not info_path.exists():
                continue

            with open(info_path, encoding="utf-8") as f:
                info = json.load(f)

            exit_info = {}
            if exit_path.exists():
                with open(exit_path, encoding="utf-8") as f:
                    exit_info = json.load(f)

            runs.append({
                "run_name": info.get("run_name", run_dir.name),
                "started_at": info.get("started_at"),
                "duration_minutes": exit_info.get("duration_minutes"),
                "exit_reason": exit_info.get("exit_reason", "unknown"),
                "params": info.get("params", {}),
                "dir": str(run_dir),
            })

        return runs
