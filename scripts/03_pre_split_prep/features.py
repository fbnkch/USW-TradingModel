"""
leitet normierte technische Indikatoren aus 1-Minuten-Bars ab

Feature-Hierarchie:
  Level 0 - Basis:
    close_norm          : Rolling-Z-Score des Close-Preises
                          Wie weit weicht der aktuelle Preis vom Durchschnitt der
                          letzten ~3 Handelstage ab? Positiv = überdurchschnittlich hoch.

    volume_norm         : Rolling-Z-Score des Volumens
                          Wie ungewöhnlich hoch/niedrig ist das aktuelle Handelsvolumen
                          im Vergleich zu den letzten ~3 Handelstagen?

    return_1m           : Der aktuelle Close-Preis im Verhältnis zur Vorminute als
                          prozentuale Änderung. Positiv = Preis ist gestiegen,
                          negativ = gefallen. Dient zur Erkennung kurzfristiger Impulse.

    vwap_distance       : Wie weit liegt der aktuelle Preis vom volumengewichteten
                          Tagesdurchschnitt (VWAP) entfernt, in Prozent?
                          Positiv = Preis über VWAP (bullish), negativ = darunter.

    rolling_high_distance : Wie weit ist der aktuelle Preis noch vom bisherigen
                          Tageshoch entfernt? Nur der heutige Tag zählt.
                          Nähert sich der Wert 0, ist ein Ausbruch auf ein neues
                          Tageshoch möglich.

    cumulative_delta    : Netto-Kaufdruck seit Marktöffnung, aufaddiert.
                          Jede Minute wird das Volumen mit der Preisrichtung
                          gewichtet (return_1m * volume) und aufsummiert.
                          Steigt der Wert -> institutionelle Käufer aktiv.

    opening_range_position : Wo steht der aktuelle Preis relativ zum Hoch und Tief
                          der ersten 30 Minuten des Handelstages?
                          Positiv = Preis über dem Morgenhoch -> Breakout-Signal.
                          Für die ersten 30 Bars NaN (wir kennen das Hoch/Tief
                          dieser Phase noch nicht vollständig -> Data-Leakage-Schutz).

    RSI_14              : Relative Strength Index der letzten 14 Minuten (Skala 0–100).
                          Misst ob der Preis zuletzt stärker gestiegen oder gefallen ist.
                          > 70 = überkauft (möglicher Rücksetzer),
                          < 30 = überverkauft (mögliche Erholung).

    BB_position         : Wo liegt der aktuelle Preis innerhalb der Bollinger Bänder
                          (20-Bar-Mittelwert ± 2 Standardabweichungen)?
                          0 = unteres Band, 0.5 = Mittelwert, > 1 = Preis bricht
                          das obere Band nach oben -> klassisches Breakout-Signal.

    minutes_since_open  : Wie viele Minuten seit Marktöffnung (09:30) sind vergangen?
                          0 = Eröffnung, 390 = Marktschluss. Gibt dem Netz ein
                          Gefühl für die Tageszeit, da Volumen und Volatilität
                          klare Tagesmuster zeigen (Spikes zur Eröffnung und zum Schluss).

  Level 1 - EMAs & direkte Ableitungen:
    EMA_{h}             : Exponentiell gewichteter Durchschnitt des Close über
                          die letzten h Bars. Neuere Bars zählen stärker.
                          Kurze EMAs (z.B. 5) reagieren schnell auf Bewegungen,
                          lange EMAs (z.B. 60) zeigen den übergeordneten Trend.

    volume_spike_ratio  : Aktuelles Volumen geteilt durch den bisherigen
                          Tagesdurchschnitt. Wert = 1.0 ist normal,
                          Wert > 2 bedeutet doppeltes Durchschnittsvolumen ->
                          starkes Breakout-Signal. Die erste Bar eines Tages
                          bekommt 1.0 (neutral), da noch kein Vergleichswert existiert.

  Level 2 - EMA-Kombinationen & Momentum:
    MACD_line           : Differenz zwischen kurzem (EMA_12) und langem (EMA_26) EMA.
                          Positiv = kurzfristiger Trend stärker als langfristiger -> Aufwärtsdruck.

    MACD_signal         : Geglättete Version der MACD_line (EMA_9 darüber).
                          Reduziert Rauschen für klarere Signale.

    MACD_histogram      : MACD_line minus MACD_signal. Der wichtigste der drei MACD-Werte:
                          Ein Vorzeichenwechsel von negativ auf positiv kündigt
                          Trendwechsel und mögliche Breakouts an.

    EMA_cross_{f}_{s}   : Abstand zwischen kurzem EMA (f) und langem EMA (s),
                          normiert durch den aktuellen Preis.
                          Positiv = kurzer EMA über langem = Aufwärtstrend aktiv.
                          Kreuzung durch 0 = Trendwechselsignal.
                          Berechnet für die Paare (5,20) und (10,60).

  Level 2 - Slopes 1. Ordnung (diskrete Steigungen):
    Slope_{col}_{t}     : Wie stark hat sich ein Wert in den letzten t Bars verändert?
                          Berechnet als (aktueller Wert - Wert vor t Bars) / t.
                          Positiv = Aufwärtsbewegung, negativ = Abwärtsbewegung.
                          Berechnet für Close, Volume und alle EMAs.

    volume_price_divergence_{t} : Steigung des Preises multipliziert mit der Steigung
                          des Volumens über t Bars. Wenn der Preis steigt aber das
                          Volumen fällt (negativer Wert), fehlt dem Breakout der
                          "Treibstoff" -> Warnsignal für eine Falle.

  Level 3 - Slopes 2. Ordnung (Beschleunigung / Momentum):
    Slope_Slope_{...}_1 : Steigung der Steigung — also ob eine Bewegung gerade
                          schneller oder langsamer wird (Beschleunigung).
                          Positiv = Bewegung nimmt zu, negativ = flacht ab.

  Level 4 - Lagged Features (zeitverzögertes Gedächtnis):
    {col}_lag5/10/15    : Der Wert eines Features von vor 5, 10 oder 15 Minuten.
                          Berechnet für close_norm, volume_norm und Slope_close_1.
                          Gibt dem Netz explizit die "Geschichte" der letzten
                          Viertelstunde mit — Breakouts kündigen sich oft nicht
                          sofort an, sondern bauen sich langsam auf.

Normierung:
  Alle Features werden per Rolling-Z-Score normiert
  (Fensterlänge z_norm_window, default 1200 Bars ~ 3 Handelstage)
  Ausnahmen (bereits interpretierbar skaliert, kein z_norm):
    close_norm, volume_norm, return_1m, vwap_distance
  NaN-Verhalten:
    Die ersten z_norm_window Bars eines Symbols sind NaN (Warmup der Rolling-Statistik)
    -> werden durch dropna() in main.py entfernt, bevor der Train/Val/Test-Split erfolgt

Eingabe-Annahmen:
  - DataFrame enthält mindestens 'close', 'high', 'low', 'open', 'volume', 'vwap', 'timestamp'
  - EMA-Perioden und Slope-Lags kommen aus params.yaml
  - Daten sind RTH-gefiltert und chronologisch sortiert (keine Lücken innerhalb eines Tages)

Ausgabe:
  - Erweiterter DataFrame mit allen Feature-Spalten angehängt
  - Liste der Feature-Spaltennamen für Logging und Training-Loader
"""

from __future__ import annotations

import pandas as pd
import numpy as np

_EPS = 1e-12  # Schutz vor Division durch Null bei rollender Standardabweichung


def generate_features(
    df: pd.DataFrame,
    ema_periods: list,
    slope_periods: list,
    z_norm_window: int = 1200,
) -> tuple:
    """Erstellt normierte technische Features aus 1-Minuten-Bar-Daten.

    Parameters
    df : pd.DataFrame
        Input mit mindestens 'close' und 'volume'.
    ema_periods : list[int]
        Bars für EMAs auf dem Close-Preis, z.B. [5, 10, 20, 60].
    slope_periods : list[int]
        Lags (Bars) für diskrete Steigungsberechnungen, z.B. [1, 3, 5].
    z_norm_window : int, default 1200
        Rolling-Fensterlänge für Z-Normierung.
        1200 Bars = ca. 3 Handelstage a 390 Bars -> stabile Statistiken.

    Returns
    tuple : (df_with_features, features_added)
        Erweiterter DataFrame + Liste der hinzugefügten Feature-Namen.

    """
    df = df.copy()

    #  Lokale Hilfsfunktionen
    def ema(series, span):
        """Exponentiell gewichteter Mittelwert."""
        return series.ewm(span=span, adjust=False).mean()

    def slope(series, t=1):
        """Diskrete Steigung: (x_t - x_{t-lag}) / lag."""
        return (series - series.shift(t)) / t

    def z_norm(series, window):
        """Rolling Z-Score: (x - rolling_mean) / (rolling_std + eps).

        min_periods = window // 2: Features wie opening_range_position haben
        konstruktionsbedingt taegliche NaN-Luecken (erste 30 Bars/Tag). Ohne
        min_periods wuerde jedes Rolling-Fenster, das so eine Luecke enthaelt,
        komplett NaN -> die ganze Spalte waere NaN und dropna() in main.py
        wuerde JEDE Zeile loeschen. min_periods erlaubt die Berechnung sobald
        genug (nicht alle) Werte im Fenster vorhanden sind.
        """
        mean = series.rolling(window, min_periods=window // 2).mean()
        std  = series.rolling(window, min_periods=window // 2).std(ddof=0)
        return (series - mean) / (std + _EPS)

    #  Feature-Container (separater DataFrame, erst am Ende concat-en)
    feats = pd.DataFrame(index=df.index)

    # Datum der Bar extrahieren -> für rolling_high_distance wichtig
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date

    #  Level 0: Basis-Features
    feats["close_norm"]  = z_norm(df["close"], z_norm_window)
    feats["volume_norm"] = z_norm(df["volume"], z_norm_window)
    feats["return_1m"]   = df["close"].pct_change()
    # VWAP - Volume Weighted Average Price - durchschnittliche Preis des Tages gewichtet nach Handelsvolumen (besteht bereits in den Rohdaten)
    # Berechnung wie weit der aktuelle Close Wert zum VWAP entfernt ist in Prozent , also (close-vwap)/vwap
    # Muss nich nochmal normiert werden, da es schon Prozent sind
    feats["vwap_distance"] = (df["close"]-df["vwap"])/df["vwap"]
    # Preis bricht das Tageshoch = starkes Breakout-Signal
    # Wir messen den Abstand zum bisher höchsten Preis des aktuellen Handelstages
    # Wichtig: nur der heutige Tag zählt, nicht der Vortag (deshalb groupby date)
    feats["rolling_high_distance"] = (df.groupby("date")["high"].transform(lambda x: x.expanding().max())- df["close"]) / df["close"]

    # Cumulative Delta: Netto-Kaufvolumen seit Marktöffnung
    # Wenn der letzte Preis (close) gestiegen ist -> war die Minute ein Kauf (positives Volumen)
    # Wenn der letzte Preis (close) gefallen ist  -> war die Minute ein Verkauf (negatives Volumen)
    # Wir multiplizieren die prozentuale Preisänderung (return_1m) mit dem Handelsvolumen
    # und summieren das seit Marktöffnung auf -> zeigt ob institutionelle Käufer aktiv sind
    # Wichtig: nur der heutige Tag zählt, nicht der Vortag (deshalb groupby date)
    directed_volume = feats["return_1m"] * df["volume"].values
    feats["cumulative_delta"] = directed_volume.groupby(df["date"]).transform("cumsum")

    # Opening Range Breakout
    # Hoch und Tief der ersten 30 Minuten pro Tag berechnen
    opening_high = df.groupby("date")["high"].transform(lambda x: x.iloc[:30].max())
    opening_low = df.groupby("date")["low"].transform(lambda x: x.iloc[:30].min())
    opening_range = opening_high - opening_low + _EPS

    # WICHTIG gegen Data Leakage: Feature erst AB Minute 30 gültig.
    # Vorher würde es das Hoch/Tief aus der Zukunft kennen -> auf NaN setzen.
    # Wir zählen die Position innerhalb jedes Tages (0, 1, 2, ...)
    bar_of_day = df.groupby("date").cumcount()
    opening_range_position = (df["close"] - opening_high) / opening_range
    feats["opening_range_position"] = opening_range_position.where(bar_of_day >= 30)

    # RSI (Relative Strength Index): Ist die Aktie überkauft oder überverkauft?
    # Wir berechnen wie stark der Preis (close) in den letzten 14 Minuten gestiegen/gefallen ist
    # RSI > 70 = überkauft (Preis stark gestiegen, evtl. Rücksetzer)
    # RSI < 30 = überverkauft (Preis stark gefallen, evtl. Erholung)
    # Starker RSI-Anstieg = Momentum -> möglicher Breakout
    rsi_window = 14
    delta  = df["close"].diff()                          # Veränderung des letzten Preises zur Vorminute
    gains  = delta.clip(lower=0)                         # nur positive Bewegungen (Käufe)
    losses = -delta.clip(upper=0)                        # nur negative Bewegungen (Verkäufe), als positive Zahl
    avg_gain = gains.ewm(span=rsi_window, adjust=False).mean()
    avg_loss = losses.ewm(span=rsi_window, adjust=False).mean()
    rs = avg_gain / (avg_loss + _EPS)                   # Verhältnis Gewinne zu Verlusten
    feats["RSI_14"] = 100 - (100 / (1 + rs))            # RSI-Formel: 0-100

    # Bollinger Band Position: Wo liegt der aktuelle Preis relativ zu seinen Bändern?
    # Oberes Band = rollender Durchschnitt + 2 x Standardabweichung (normaler Schwankungsbereich)
    # Unteres Band = rollender Durchschnitt - 2 x Standardabweichung
    # Position > 1 = Preis bricht oberes Band nach oben -> klassisches Breakout-Signal
    # Position = 0.5 = Preis in der Mitte (= rollender Durchschnitt)
    bb_window = 20
    bb_mean  = df["close"].rolling(bb_window).mean()     # rollender Durchschnitt des letzten Preises
    bb_std   = df["close"].rolling(bb_window).std(ddof=0)
    bb_upper = bb_mean + 2 * bb_std                      # oberes Band
    bb_lower = bb_mean - 2 * bb_std                      # unteres Band
    feats["BB_position"] = (df["close"] - bb_lower) / (bb_upper - bb_lower + _EPS)

    # Minuten seit Marktöffnung (09:30 ET = 9*60+30 = 570 Minuten seit Mitternacht)
    # Unsere eigenen Data-Understanding Plots zeigen: Volumen und Volatilität haben
    # klare Tagesmuster -> das Netz soll wissen wo im Handelstag wir gerade sind
    # 0 = Marktöffnung, 390 = Marktschluss
    ts = pd.to_datetime(df["timestamp"])
    feats["minutes_since_open"] = ts.dt.hour * 60 + ts.dt.minute - 570

    # Opening Range Position: Wo liegt der aktuelle Preis relativ zur ersten 30 Minuten des Handelstages?
    # Die ersten 30 Minuten (09:30 - 10:00 Uhr) setzen oft den Ton für den ganzen Tag
    # Wir berechnen das höchste Hoch und niedrigste Tief dieser ersten 30 Minuten
    # und messen wie weit der aktuelle Preis (close) davon entfernt ist

    # Höchster Preis (high) der ersten 30 Minuten des Tages
    opening_high = df.groupby("date")["high"].transform(lambda x: x.iloc[:30].max())

    # Niedrigster Preis (low) der ersten 30 Minuten des Tages
    opening_low = df.groupby("date")["low"].transform(lambda x: x.iloc[:30].min())

    # Breite der Opening Range (wie groß war die Spanne in den ersten 30 Minuten?)
    # + _EPS schützt vor Division durch Null falls high == low
    opening_range = opening_high - opening_low + _EPS

    # Position des aktuellen Preises (close) relativ zur Opening Range
    # Positiv = Preis über dem Tageshoch der ersten 30 Min -> Breakout Signal
    # 0 bis 1  = Preis innerhalb der Range
    # Negativ  = Preis unter dem Tagestief der ersten 30 Min
    opening_range_position = (df["close"] - opening_high) / opening_range

    # WICHTIG - Data Leakage Fix:
    # Die ersten 30 Minuten kennen das opening_high/low noch nicht vollständig
    # z.B. weiß Minute 5 noch nicht was Minute 29 als Hoch haben wird
    # Lösung: Feature erst ab Minute 30 gültig setzen, vorher NaN
    # cumcount() zählt durch welche Minute des Tages wir gerade sind (0, 1, 2, ...)
    bar_of_day = df.groupby("date").cumcount()
    feats["opening_range_position"] = opening_range_position.where(bar_of_day >= 30)

    #  Level 1: EMAs des Close
    for h in ema_periods:
        feats[f"EMA_{h}"] = ema(df["close"], h)

    # Level 1: Volume Spike Ratio
    # Wie stark weicht das aktuelle Volumen vom bisherigen Tagesdurchschnitt ab?
    # expanding().mean() berechnet den Durchschnitt aller Bars seit Marktöffnung
    # shift(1) ist ZWINGEND: wir dürfen die aktuelle Bar nicht in ihren eigenen
    # Durchschnitt einrechnen -> sonst Data Leakage (Bar kennt ihren eigenen Wert)
    # Wert = 1.0  -> normales Volumen
    # Wert = 3.0  -> dreifaches Durchschnittsvolumen = signifikanter Volumen-Spike
    daily_avg_volume = df.groupby("date")["volume"].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    # fillna(1.0): Die erste Bar jedes Tages hat kein Vortags-Mittel -> 1.0 = neutral.
    # Ohne fillna würde z_norm() jedes Rolling-Fenster das diese NaN-Bar enthält
    # ebenfalls auf NaN setzen -> alle Werte des Features wären NaN (Rolling-NaN-Propagation).
    feats["volume_spike_ratio"] = (df["volume"] / (daily_avg_volume + _EPS)).fillna(1.0)

    # Level 2: MACD (Moving Average Convergence Divergence)
    # Standardindikator für Trendstärke und Trendwechsel, basiert auf EMA-Differenz
    # MACD_line     = EMA_12 - EMA_26: positiv = kurzfristiger Aufwärtstrend
    # MACD_signal   = EMA_9 der MACD_line: geglättetes Signal zum Rauschen reduzieren
    # MACD_histogram = Differenz beider: Vorzeichenwechsel = Trendwechselsignal
    # Warum wichtig: MACD-Histogramm-Kreuzungen kündigen Breakouts an bevor sie passieren
    ema_fast   = df["close"].ewm(span=12, adjust=False).mean()
    ema_slow   = df["close"].ewm(span=26, adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    macd_sig   = macd_line.ewm(span=9, adjust=False).mean()
    feats["MACD_line"]      = macd_line
    feats["MACD_signal"]    = macd_sig
    feats["MACD_histogram"] = macd_line - macd_sig   # das stärkste der drei

    # Level 2: EMA-Cross-Features
    # Misst den relativen Abstand zwischen kurzem und langem EMA
    # Normiert durch Close-Preis -> vergleichbar über verschiedene Preisniveaus
    # Positiv = kurzer EMA über langem EMA = Aufwärtstrend aktiv
    # Negativ = kurzer EMA unter langem EMA = Abwärtstrend
    # Kreuzung durch 0 = klassisches Trendwechselsignal
    # Wir prüfen ob die benötigten EMAs überhaupt berechnet wurden
    for h_fast, h_slow in [(5, 20), (10, 60)]:
        if f"EMA_{h_fast}" in feats.columns and f"EMA_{h_slow}" in feats.columns:
            feats[f"EMA_cross_{h_fast}_{h_slow}"] = (
                feats[f"EMA_{h_fast}"] - feats[f"EMA_{h_slow}"]
            ) / df["close"]

    #  Level 2: Slopes 1. Ordnung - Close
    base_cols = ["close", "volume"] + [f"EMA_{h}" for h in ema_periods]
    for col in base_cols:
        source = df[col] if col in df.columns else feats[col]
        for t in slope_periods:
            feats[f"Slope_{col}_{t}"] = slope(source, t)

    #Volumen-Price Divergenz: Wenn Preis steigt aber Volumen fällt ist das oft eine Falle, das müssen wir berechnen um es zu sehen
    for t in slope_periods:
        feats[f"volume_price_divergence_{t}"] = (feats[f"Slope_close_{t}"] * feats[f"Slope_volume_{t}"])

    #  Level 3: Slopes 2. Ordnung (Beschleunigung) - Volumen
    slope_cols = [c for c in feats.columns if c.startswith("Slope_")]
    slope_of_slope = pd.DataFrame(
        {f"Slope_{col}_1": slope(feats[col], 1) for col in slope_cols},
        index=df.index,
    )
    feats = pd.concat([feats, slope_of_slope], axis=1)

    # Level 4: Lagged Features (zeitverzögert) - vom Prof explizit gewünscht
    # Breakouts kündigen sich oft nicht sofort an, sondern mit Verzögerung
    # Wir schauen: wie war close_norm/volume_norm/Slope vor 5, 10, 15 Minuten?
    # Das Netz bekommt so die "Geschichte" der Bewegung als expliziten Input
    # lag5  = Wert von vor 5 Minuten  (kurzfristiges Gedächtnis)
    # lag10 = Wert von vor 10 Minuten (mittelfristiges Gedächtnis)
    # lag15 = Wert von vor 15 Minuten (halbes Breakout-Fenster zurück)
    cols_to_lag = ["close_norm", "volume_norm", "Slope_close_1"]
    for col in cols_to_lag:
        if col in feats.columns:
            for lag in [5, 10, 15]:
                # .shift(lag) verschiebt die Spalte um lag Zeilen nach unten
                # -> Minute 20 bekommt den Wert von Minute 15 (bei lag=5)
                feats[f"{col}_lag{lag}"] = feats[col].shift(lag)

    #  Z-Normierung aller noch nicht normierten Features
    already_normed = {"close_norm", "volume_norm", "return_1m"}
    to_norm = [c for c in feats.columns if c not in already_normed]
    feats[to_norm] = feats[to_norm].apply(lambda s: z_norm(s, z_norm_window))

    #  Zusammenfuehren
    df = pd.concat([df, feats], axis=1)
    features_added = feats.columns.tolist()

    return df, features_added