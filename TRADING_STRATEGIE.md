# Beta-Trading-Strategie
### Erster Ansatz für Strategie anhand unserer ersten Ergebnisse ohne Backpropagation

---

## 1. Executive Summary

### 1.1 Modellergebnisse (Test-Set)

| Modell | Accuracy | F1 | Precision | Recall | Threshold | Rolle |
|--------|----------|-----|-----------|--------|-----------|-------|
| Baseline  | 50.22% | – | – | – | – | – |
| **MLP V2** | **64.04%** | 0.563 | **0.600** | 0.531 | 0.50 |  **Filter** |
| GRU | 58.42% | **0.647** | 0.514 | 0.871 | 0.33 |  Finder |
| CNN | 57.93% | 0.645 | 0.511 | 0.873 | 0.31 | Finder |
| LSTM | 57.09% | 0.642 | 0.505 | **0.878** | 0.32 |  Finder |
| LightGBM | 56.47% | 0.623 | 0.485 | 0.861 | 0.36 |  Finder |

### 1.2 Die zwei Modell-Familien

Unsere 5 Modelle fallen in zwei klare Kategorien:

| | Filter (MLP V2) | Finder (LSTM/GRU/CNN/LightGBM) |
|---|---|---|
| **Stärke** | Hohe Precision (60%) | Hoher Recall (87-88%) |
| **Signal** | „Wenn ich Alarm schlage, stimmt es meistens" | „Ich finde fast jeden Breakout" |
| **Schwäche** | Verpasst ~47% der Breakouts | ~49% Fehlalarm-Rate |
| **Einsatz** | Bestätigung | Kandidaten-Screening |

---

## 2. Feature-Auswahl: 5 Schlüssel-Features

Basierend auf LightGBM Feature Importance (Gain) konzentrieren wir uns auf die dominanten Features:

| # | Feature | Gain | Typ | Trading-Bedeutung |
|---|---------|------|-----|-------------------|
| 1 | `return_1m` | 3.78M | Momentum | **Der stärkste Prädiktor.** 1-Minuten-Return. Positiv = bereits in Bewegung. |
| 2 | `Slope_close_1` | 2.40M | Trend | Kurzfristige Preisrichtung. Steigend = Momentum baut auf. |
| 3 | `minutes_since_open` | 2.32M | Zeit | **Tageszeit-Muster.** Breakouts häufen sich zu bestimmten Zeiten. |
| 4 | `Slope_Slope_EMA_240_1_1` | 0.26M | Beschleunigung | Zweite Ableitung des Langzeit-Trends (4h-EMA). |
| 5 | `volume_norm` | 0.17M | Volumen | Relatives Volumen. Ungewöhnlich hohes Volumen = Treibstoff. |

**Erkenntnis:** Die Top-3 Features machen **>90%** der gesamten Vorhersagekraft aus. Das Modell lernt im Kern: *„Wenn der Preis bereits steigt (return_1m, Slope_close_1) und die Tageszeit günstig ist (minutes_since_open), ist ein Breakout wahrscheinlich."*

Die restlichen 77 Features (EMAs, MACD, RSI, Lagged-Features etc.) tragen nur marginal bei – sie machen das Modell komplexer, ohne nennenswert bessere Vorhersagen zu liefern.

### 2.1 Feature-Gruppen nach Wichtigkeit

```
5/5 KRITISCH (Top-3, >90% Gain):
    return_1m, Slope_close_1, minutes_since_open

4/5 HILFREICH (0.1-0.3M Gain):
    Slope_Slope_EMA_240_1_1, volume_norm, EMA_240

3/5 MODERAT (0.05-0.12M Gain):
    cumulative_delta, MACD_signal, MACD_histogram,
    opening_range_position, close_norm, RSI_14

2/5 MARGINAL (<0.08M Gain):
    Alle 15 EMA-Varianten, 5 BB/RSI-Derivate
    → Zusammen <5% des Gesamt-Gains

1/5 VERNACHLÄSSIGBAR (<0.01M Gain):
    Lagged-Features (close_norm_lag5/10/15, etc.)
    → <1% des Gesamt-Gains
```

---

## 3. Ensemble-Trading-Strategie

### 3.1 Signal-Pipeline

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ 1. FINDER    │───▶│ 2. FILTER    │───▶│ 3. MARKT-    │───▶│ 4. EXECUTION │
│  (GRU/LSTM)  │    │  (MLP V2)    │    │  KONTEXT     │    │              │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
 P_breakout > 0.33   P_breakout > 0.50   Time + Volume       Entry + Risk
 ~88% aller BO       60% Precision       + Confirmation      Mgmt
 gefunden            der Bestätigungen
```

### 3.2 Entry-Regeln

Ein Trade wird NUR eröffnet, wenn **ALLE** Bedingungen erfüllt sind:

| # | Bedingung | Wert | Quelle |
|---|-----------|------|--------|
| E1 | **Finder-Signal** | GRU oder LSTM: P(breakout) > 0.33 | Modell-Output |
| E2 | **Filter-Bestätigung** | MLP V2: P(breakout) > 0.50 | Modell-Output |
| E3 | **Momentum-Richtung** | `return_1m` > 0 | Feature #1 |
| E4 | **Kurz-Trend positiv** | `Slope_close_1` > 0 | Feature #2 |
| E5 | **Keine Mittagsflaute** | 10:00–12:00 oder 14:00–15:30 ET | Feature #3 |

**Warum so streng?** Jeder weitere Filter reduziert False Positives. Mit E1–E5 erwarten wir Precision ≥ 62% (von 50% ohne Filter).

### 3.3 Exit-Regeln

| Exit-Typ | Bedingung | Begründung |
|----------|-----------|------------|
| **Take Profit** | Preis ≥ Entry × 1.0025 (+0.25%) | Unter Modell-Theta (0.30%) für erreichbaren TP |
| **Stop Loss** | Preis ≤ Entry × 0.99 (−1.0%) | Weiter als Markt-Rauschen (~0.10%) – lässt Trade atmen |
| **Trailing Stop** | Dynamisch: SL wandert mit steigendem Profit nach oben | Sichert Gewinne, ohne Trades früh auszustoppen |
| **Ratchet-Exit** | Kurs fällt unter TP-Level (nachdem TP einmal erreicht wurde) | Gewinn gesperrt, Trade läuft weiter |
| **Time Stop** | 30 Minuten ohne Exit | Vorhersagehorizont abgelaufen |
| **Signal-Kollaps** | Finder P(breakout) < 0.20 | Modell-Konfidenz eingebrochen |

#### 3.3.1 Trailing Stop Loss (graduell)

Der Trailing Stop ist ein **dynamischer Stop Loss**, der mit steigendem Kurs nach oben wandert, aber niemals fällt.

**Kernidee:** Je weiter der Kurs ins Plus läuft, desto enger wird der Abstand des SL zum aktuellen Höchstpreis – Gewinne werden schrittweise eingesackt.

```
Trail-Abstand (Abstand SL zu highest_price)

  0.50% ┤╲
        │  ╲
  0.35% ┤    ╲         ← Je mehr Profit, desto enger
        │      ╲
  0.20% ┤        ╲_____
        │
        └─────┼─────────┼────── Profit über Entry
            0.25%      0.50%
```

| Parameter | Default | Bedeutung |
|-----------|---------|-----------|
| `trailing_sl_pct` | 0.50% | Start-Abstand bei Entry |
| `trailing_min_pct` | 0.20% | Minimal-Abstand bei vollem Profit |
| `trailing_ramp_pct` | 0.50% | Profit-Level, bei dem die volle Straffung greift |

**Wichtig:** Der Trailing Stop **schläft bei Entry** – solange der Kurs nicht über den Einstiegspreis steigt, bleibt der fixe SL (−1.0%) unangetastet. Erst wenn `highest_price > entry_price`, wacht der Trailing auf.

#### 3.3.2 Ratchet-Mode (Gewinnsperre)

Der Ratchet-Mode ersetzt den klassischen TP-Exit: Bei Erreichen des TP-Levels wird **nicht verkauft**, sondern der SL auf das TP-Niveau angehoben. Der Trade läuft weiter, kann aber nie mehr unter den ursprünglichen TP-Gewinn fallen.

```
$85.50 ┤              ╱╲
       │             ╱  ╲   ← Trailing zieht SL weiter hoch
$85.00 ┤        ╱──╱    ╲
       │   ╱──╱           ╲___  Exit via Trailing Stop
$84.70 ┤══╱════════════════════  ← TP erreicht → SL springt auf $84.70
       │╱                        KEIN Exit! Trade läuft weiter.
$84.49 ┼────────────────────────  ← Entry
       │
$83.65 ┤- - - - - - - - - - - -  ← Fixer SL (−1%)
```

**Kombination Ratchet + Trailing = „Gewinne laufen lassen, Verluste begrenzen"**

---

## 4. Risikomanagement

### 4.1 Positionsgrößen

| Regel | Wert |
|-------|------|
| Max. Risiko pro Trade | 0.5% des Portfolios |
| Max. Risiko pro Tag | 2.0% des Portfolios |
| Max. gleichzeitige Positionen | 10 |
| Max. Position pro Symbol | 5% des Portfolios |

### 4.2 Aktuelle Standard-Parameter

| Parameter | Wert | Begründung |
|-----------|------|------------|
| `max_positions` | 10 | Diversifikation über mehr Signale |
| `tp_pct` | 0.25% | Unter Modell-Theta (0.30%), erreichbar |
| `sl_pct` | 1.00% | Weiter als 1-Min-Rauschen (~0.10%) |
| `position_size_pct` | 5% | Kapitalerhalt – max 50% Exposure |
| Trailing SL | AN (default) | Graduell, schläft bei Entry |
| Ratchet-Mode | AN (default) | TP-Level wird SL-Boden statt Exit |

### 4.3 Kelly Criterion (Half-Kelly)

```
f* = (p × b - q) / b
   = (0.60 × 2.4 - 0.40) / 2.4
   = 0.433  (Full Kelly)

Half-Kelly: 21.7% des Portfolios pro Trade
→ In der Praxis: 0.5% Risk pro Trade (konservativer)
```

---

## 5. Backtesting-Plan

### 5.1 Setup

- **Walk-Forward:** Train bis 2023, Val H1 2024, **Test H2 2024**
- **Transaktionskosten:** 0.01% pro Trade (Spread + Slippage)
- **Startkapital:** $100.000

### 5.2 Vergleichs-Benchmarks

| Strategie | Erwartete Annual Return | Max DD | Sharpe |
|-----------|------------------------|--------|--------|
| Buy & Hold NASDAQ-100 | ~15-20% | ~20-30% | ~0.8 |
| Unsere Breakout-Strategie | TBD | TBD | TBD |
| 60/40 Portfolio | ~8-10% | ~10-15% | ~1.0 |

### 5.3 KPIs

| KPI | Ziel |
|-----|------|
| Win Rate | ≥ 60% |
| Profit Factor | ≥ 1.3 |
| Sharpe Ratio | ≥ 1.2 |
| Max Drawdown | ≤ 10% |
| Trades / Tag (ø) | 5–20 |

---

## 6. Paper Trading (Alpaca)

- **Plattform:** Alpaca Markets Paper API
- **Startkapital:** $100.000 (Papiergeld)
- **Zeitraum:** 2–4 Wochen vor Live-Einsatz
- **Fokus:** Slippage-Realität, Order-Fill-Rate, Strategie-Stabilität
- **Startbefehl:** `python scripts/08_deployment/trading_loop.py --paper`
- **Deaktivieren von Features:** `--no_trailing_sl` / `--no_ratchet`

---

## 7. Nächste Schritte

1. **Ensemble-Predictor** (`ensemble.py`) --> Finder + Filter kombinieren
2. **Backtest** (`backtest.py`) --> Walk-Forward mit Transaktionskosten
3. **Paper Trading Setup** --> Alpaca API Integration
4. **Live-Monitoring** --> Real-Time Dashboard für Paper-Trading-Phase
