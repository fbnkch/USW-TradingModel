"""

Dieses Modul berechnet die binäre Zielvariable für jede 1-Minuten-Bar.
Sie gibt an, ob der Kurs innerhalb der nächsten `horizon` Bars um mindestens
`theta` Prozent gestiegen ist (gemessen am Maximum der High-Preise).

Zielvariablen

  breakout_{horizon}m  : int (0 oder 1)
      1 = max(high[t+1 : t+horizon]) / close[t] - 1  >=  theta
      0 = sonst
  high_max_{horizon}m  : float
      Tatsächliches High-Maximum im Fenster (für Analyse / Debugging).

Eingabe-Annahmen
-
  - DataFrame enthält mindestens 'close' und 'high'.
  - Daten sind chronologisch sortiert (RTH-gefiltert, keine Lücken).
  - horizon und theta kommen aus params.yaml

Ausgabe
-
  Kopie des Input-DataFrame mit zwei neuen Spalten:
    high_max_{horizon}m
    breakout_{horizon}m
  Die letzten horizon Zeilen erhalten NaN (kein vollständiges Fenster).
  main.py droppt alle NaNs nach vollständiger Assembly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_breakout_target(
    df: pd.DataFrame,
    horizon: int = 30,
    theta: float = 0.003,
    close_col: str = "close",
    high_col: str = "high",
) -> pd.DataFrame:
    """Berechnet die binäre Breakout-Zielvariable über ein Forward-Fenster.

    Für jede Bar zum Zeitpunkt t:
      1. Bestimme max(high) in den nächsten horizon Bars (t+1 bis t+horizon).
      2. Berechne potenzielle Rendite: (high_max - close_t) / close_t
      3. Label = 1 wenn Rendite >= theta, sonst 0.

    Parameters
    -
    df : pd.DataFrame
        Input-DataFrame mit mindestens close_col und high_col.
    horizon : int, default 30
        Anzahl vorwärtsschauender Bars (Minuten).
    theta : float, default 0.003
        Mindeststeigerung für Label=1 (0.003 = 0.3%).
    close_col : str, default 'close'
        Spaltenname des Close-Preises.
    high_col : str, default 'high'
        Spaltenname des High-Preises.

    Returns
    -
    pd.DataFrame
        Kopie des Input-DataFrame mit neuen Spalten
        high_max_{horizon}m und breakout_{horizon}m.

    """
    df = df.copy()

    close = df[close_col].to_numpy(dtype=float)
    high  = df[high_col].to_numpy(dtype=float)
    n     = len(df)

    high_max_arr = np.full(n, np.nan)
    label_arr    = np.full(n, np.nan)

    for i in range(n - horizon):
        hmax = high[i + 1 : i + horizon + 1].max()
        high_max_arr[i] = hmax
        potential_return = (hmax - close[i]) / close[i]
        label_arr[i] = 1.0 if potential_return >= theta else 0.0

    df[f"high_max_{horizon}m"] = high_max_arr
    df[f"breakout_{horizon}m"] = label_arr

    return df
