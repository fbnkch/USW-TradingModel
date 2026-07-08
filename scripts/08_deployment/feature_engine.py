"""
Feature-Engine: Inkrementelle 82-Feature-Berechnung aus 1-Minuten-Bars.

Verwendet einen Rolling-Buffer-Ansatz:
- Haelt ~1500 Bars im Speicher pro Symbol
- Fuehrt die EXAKT gleiche generate_features() Pipeline aus wie das Training
- Extrahiert nur die Features der neuesten Bar
- Garantiert identische Features zum Training (kein Bug-Risiko durch Neuimplementierung)

Memory: ~1500 Bars * 82 Features * 4 Bytes * 100 Symbole ~= 50 MB
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Import generate_features from the training pipeline
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent / "03_pre_split_prep"))
from features import generate_features


def _to_utc_naive(ts):
    """Konvertiert Timestamps zu UTC-naive (Training-Format). Funktioniert mit tz-aware und tz-naive."""
    t = pd.to_datetime(ts)
    if t.tz is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t


class FeatureEngine:
    """Haelt einen Rolling-Buffer pro Symbol und berechnet Features via generate_features()."""

    def __init__(
        self,
        symbol: str,
        warmup_df: pd.DataFrame,
        scaler,
        features_list: list[str],
        ema_periods: list = None,
        slope_periods: list = None,
        z_norm_window: int = 1200,
        buffer_size: int = 1500,
    ):
        self.symbol = symbol
        self.scaler = scaler
        self.features_list = features_list
        self.n_features = len(features_list)
        self.ema_periods = ema_periods or [5, 10, 20, 60, 120, 240]
        self.slope_periods = slope_periods or [1, 3, 5]
        self.z_norm_window = z_norm_window
        self.buffer_size = buffer_size

        # Rolling-Buffer: OHLCV + timestamp
        required_cols = ["timestamp", "open", "high", "low", "close", "volume", "vwap"]
        self._buffer = warmup_df[required_cols].copy()
        self._buffer["timestamp"] = self._buffer["timestamp"].apply(_to_utc_naive)
        self._buffer = self._buffer.sort_values("timestamp").reset_index(drop=True)

        # Warmup-Check
        self._ready = len(self._buffer) >= (z_norm_window // 2)  # min_periods=600

        # Sequence-Puffer: letzte 30 Feature-Vektoren fuer sequenzielle Modelle
        self._seq_buffer: list = []  # list of (82,) numpy arrays
        self._seq_ready = False

        # Sequence-Buffer aus Warmup-Daten vorfuellen (Bug-Fix: vorher blieb er leer!)
        self._warmup_sequence_buffer()

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def seq_ready(self) -> bool:
        """True wenn genug Feature-Vektoren fuer ein 30er-Sequenz-Fenster vorhanden."""
        return self._seq_ready

    @property
    def buffer_len(self) -> int:
        return len(self._buffer)

    def _warmup_sequence_buffer(self):
        """Fuellt den Sequence-Buffer aus den Warmup-Daten vor (EINZIGER generate_features-Aufruf).

        WICHTIG: Ohne diesen Schritt waere get_sequence() die ersten 30 Minuten
        des Trading-Tages None, und LSTM/GRU/CNN wuerden auf LightGBM-Werte
        zurueckfallen.

        Strategie: Einmal generate_features() auf dem kompletten Warmup-Buffer,
        dann die letzten 30 Feature-Vektoren skalieren und direkt in den
        Sequence-Buffer legen. O(1) pro Symbol statt O(60) process_bar-Aufrufe.
        """
        min_bars = max(self.z_norm_window // 2, 60)
        if len(self._buffer) < min_bars:
            return

        # Einmal Features berechnen — nur die letzten 200 Bars (z-Norm braucht
        # 1200, aber 200 reicht fuer initialen Sequence-Buffer; wird innerhalb
        # von 30 Minuten durch echte Process-Bar-Durchlaeufe ueberschrieben)
        # 200 Bars × 95 Symbole ~ 3s statt 1500 Bars × 95 Symbole ~ 60s
        n_warmup_bars = min(200, len(self._buffer))
        df_full, _ = generate_features(
            self._buffer.iloc[-n_warmup_bars:].copy(),
            ema_periods=self.ema_periods,
            slope_periods=self.slope_periods,
            z_norm_window=min(self.z_norm_window, n_warmup_bars),
        )

        # Letzte 30 Zeilen extrahieren (genug fuer 30er-Sequenz-Fenster)
        recent = df_full.iloc[-30:]

        for i in range(len(recent)):
            row = recent.iloc[i]
            feature_vec = np.array(
                [row.get(f, 0.0) for f in self.features_list], dtype=np.float64
            )

            # NaN mit 0.0 fuellen (neutral im Z-Score-Raum)
            nan_mask = np.isnan(feature_vec)
            if nan_mask.any():
                feature_vec[nan_mask] = 0.0

            # GlobalScaler anwenden
            scaled = self.scaler.transform(feature_vec.reshape(1, -1)).astype(np.float32)
            self._seq_buffer.append(scaled[0].copy())

        self._seq_ready = len(self._seq_buffer) >= 30

    def get_sequence(self) -> Optional[np.ndarray]:
        """Gibt die letzten 30 Feature-Vektoren als (30, 82) Array zurueck."""
        if not self._seq_ready:
            return None
        return np.stack(self._seq_buffer[-30:], axis=0)  # (30, 82)

    def process_bar(self, bar: dict) -> Optional[np.ndarray]:
        """Verarbeitet eine neue 1-Minuten-Bar und gibt den 82-Dim Feature-Vektor zurueck.

        Args:
            bar: Dict mit keys: timestamp, open, high, low, close, volume, vwap

        Returns:
            (82,) float32 numpy array, oder None wenn Z-Norm noch nicht warm.
        """
        # Neue Bar anhaengen
        new_row = pd.DataFrame([{
            "timestamp": _to_utc_naive(bar["timestamp"]),
            "open": float(bar["open"]),
            "high": float(bar["high"]),
            "low": float(bar["low"]),
            "close": float(bar["close"]),
            "volume": float(bar["volume"]),
            "vwap": float(bar["vwap"]),
        }])

        self._buffer = pd.concat(
            [self._buffer, new_row], ignore_index=True
        )

        # Buffer trimmen (letzte buffer_size Bars behalten)
        if len(self._buffer) > self.buffer_size:
            self._buffer = self._buffer.iloc[-self.buffer_size:].reset_index(drop=True)

        # Features berechnen (generate_features braucht timestamp-Spalte)
        df_full, _ = generate_features(
            self._buffer.copy(),
            ema_periods=self.ema_periods,
            slope_periods=self.slope_periods,
            z_norm_window=self.z_norm_window,
        )

        # Feature-Vektor der LETZTEN Zeile extrahieren
        latest = df_full.iloc[-1]
        feature_vec = np.array([latest.get(f, 0.0) for f in self.features_list], dtype=np.float64)

        # NaN-Features mit 0.0 fuellen (neutral im Z-Score-Raum).
        # opening_range_position ist z.B. die ersten 30 Bars eines Tages NaN
        # (Data-Leakage-Schutz: Hoch/Tief der ersten 30 Min noch nicht bekannt).
        # Ein einzelnes NaN soll nicht den ganzen Feature-Vektor verwerfen.
        nan_mask = np.isnan(feature_vec)
        if nan_mask.any():
            feature_vec[nan_mask] = 0.0

        # Pruefen ob Z-Norm schon warm ist (>50% NaN = Buffer noch zu klein)
        if nan_mask.mean() > 0.5:
            self._ready = False
            return None

        self._ready = True

        # GlobalScaler anwenden
        feature_vec_scaled = self.scaler.transform(feature_vec.reshape(1, -1)).astype(np.float32)
        result = feature_vec_scaled[0]  # (82,)

        # Sequence-Puffer updaten (letzte 30 Feature-Vektoren)
        self._seq_buffer.append(result.copy())
        if len(self._seq_buffer) > 30:
            self._seq_buffer.pop(0)
        self._seq_ready = len(self._seq_buffer) >= 30

        return result


class MultiSymbolEngine:
    """Verwaltet Feature-Engines fuer alle Symbole."""

    def __init__(self, symbols, warmup_bars, scaler, features_list):
        self.engines = {}
        for sym in symbols:
            if sym in warmup_bars and len(warmup_bars[sym]) > 0:
                self.engines[sym] = FeatureEngine(
                    sym, warmup_bars[sym], scaler, features_list
                )

    def process_bars(self, latest_bars: dict) -> dict[str, np.ndarray]:
        """Verarbeitet neue Bars fuer alle Symbole.

        Returns:
            {symbol: feature_vector(82,)} fuer Symbole mit warmem Z-Norm.
        """
        results = {}
        for sym, bar in latest_bars.items():
            if sym not in self.engines:
                continue
            fv = self.engines[sym].process_bar(bar)
            if fv is not None:
                results[sym] = fv
        return results

    def ready_symbols(self) -> list[str]:
        return [s for s, e in self.engines.items() if e.is_ready]
