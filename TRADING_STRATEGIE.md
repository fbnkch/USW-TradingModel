# 📈 Trading-Strategie: Intraday Breakout NASDAQ-100

> **Basiert auf:** Datenanalyse von 101 NASDAQ-100 Aktien, 1-Minuten-Bars (2022–2025)
> **Zielvariable:** `breakout_30m` – Kurs steigt ≥0.3% innerhalb der nächsten 30 Minuten
> **Stand:** 27.06.2026

---

## 1. Executive Summary

### 1.1 Das Problem

Wir wollen kurzfristige intraday Kursausbrüche (Breakouts) bei NASDAQ-100 Aktien vorhersagen. Konkret: Steigt der Kurs einer Aktie in den nächsten 30 Minuten um mindestens 0.3%?

### 1.2 Die Datenlage

| Metrik | Wert |
|--------|------|
| Trainings-Samples | 10.394.874 (1-Minuten-Bars) |
| Symbole | 100 NASDAQ-100 Aktien |
| Zeitraum | 2022-01-01 bis 2025-01-01 |
| Features | 82 technische Indikatoren (Momentum, EMA, Slope, Volumen, Lagged) |
| Klassen-Balance | 49.78% Breakout / 50.22% Kein Breakout (nahezu perfekt balanced) |
| Breakout-Rate variiert pro Symbol | 32% (AAPL) bis 55% (ABNB) |

### 1.3 Das aktuelle Modell

| Modell | Accuracy | Precision (Breakout) | Recall (Breakout) | F1 |
|--------|----------|---------------------|-------------------|-----|
| Baseline (Majority) | 50.22% | – | – | – |
| MLP V1 (82→64→32→1) | 59.6% | 52% | 53% | 0.52 |
| MLP V2 (verbessert) | *in Training* | – | – | – |
| LSTM (sequentiell) | *in Training* | – | – | – |
| GRU (sequentiell) | *in Training* | – | – | – |
| LightGBM | *in Training* | – | – | – |

**Kernproblem:** Precision von 52% bedeutet: **Jeder zweite Trade ist ein Fehlalarm.** Recall von 53% bedeutet: **Fast die Hälfte aller Breakouts wird verpasst.**

---

## 2. Strategie-Design

### 2.1 Grundprinzip

Die Strategie nutzt ein **Multi-Modell-Ensemble** mit strengen Entry-Filtern, um die Precision zu erhöhen – auch auf Kosten von Recall. Im Trading ist es besser, wenige gute Trades zu machen als viele schlechte.

```
HOHE PRECISION > HOHER RECALL (für Trading)
```

**Faustregel:** Ein Fehlalarm (False Positive) kostet Geld (Spread + Slippage + Stop-Loss). Ein verpasster Breakout (False Negative) kostet nur Opportunität. Daher liegt der Fokus auf Precision.

### 2.2 Signal-Pipeline

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ 1. MODELL-   │───▶│ 2. SIGNAL-   │───▶│ 3. MARKT-    │───▶│ 4. EXECUTION │
│    ENSEMBLE  │    │    FILTER    │    │    KONTEXT   │    │              │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
    5 Modelle         Threshold +        Time-of-Day +       Position Size +
    Weighted Avg      Confirmation        Volatility          Risk Limits
```

### 2.3 Modell-Ensemble

Vier Modell-Familien decken unterschiedliche Signalmuster ab:

| # | Modell | Typ | Stärke |
|---|--------|-----|--------|
| 1 | MLP V2 | Feedforward | Allgemeine Feature-Interaktionen |
| 2 | LSTM | Sequentiell (RNN) | Langfristige zeitliche Abhängigkeiten |
| 3 | GRU | Sequentiell (RNN) | Effiziente Zeitmuster-Erkennung |
| 4 | CNN-1D | Sequentiell (Conv) | Lokale Pattern-Erkennung (3–10 Min) |
| 5 | LightGBM | Gradient Boosting | Nicht-lineare Interaktionen, Feature-Importance |

**Ensemble-Regel:**
```
P_ensemble = 0.25 × P_mlp + 0.20 × P_lstm + 0.15 × P_gru + 0.15 × P_cnn + 0.25 × P_lgb
```
Gewichte basieren auf Validierungs-Performance (können nach Training angepasst werden).

---

## 3. Entry-Regeln

### 3.1 Primärer Entry

Ein Trade wird eröffnet, wenn ALLE folgenden Bedingungen erfüllt sind:

| # | Bedingung | Wert | Begründung |
|---|-----------|------|------------|
| E1 | **Ensemble-Wahrscheinlichkeit** | P_ensemble > θ_symbol | Kalibrierter Threshold pro Symbol |
| E2 | **Modell-Übereinstimmung** | ≥ 3 von 5 Modellen signalisieren Breakout | Reduziert False Positives durch Konsens |
| E3 | **Volumen-Bestätigung** | volume_spike_ratio > 1.5 | Breakout ohne Volumen = Fake |
| E4 | **Momentum-Richtung** | Slope_close_5 > 0 | Preis muss bereits steigen |
| E5 | **Keine Überkauft-Situation** | RSI_14 < 75 | Vermeidet Einstieg am Top |
| E6 | **Time-of-Day-Filter** | 09:45–15:45 ET | Erste/letzte 15 Min zu volatil |

### 3.2 Symbol-Kalibrierung

Jedes Symbol hat eine eigene Breakout-Rate. Der Threshold θ_symbol wird pro Symbol auf dem Validation-Set so kalibriert, dass Precision ≥ 60% erreicht wird:

```
θ_symbol = Percentile_{90}(P_ensemble | kein Breakout)  # 90% der Non-Breakouts liegen darunter
```

Das bedeutet: Nur die stärksten 10% der Ensemble-Signale werden gehandelt.

### 3.3 Time-of-Day-Gewichtung

Breakouts sind nicht gleichmäßig über den Tag verteilt (laut unserer EDA):

| Phase | Zeit (ET) | Breakout-Häufigkeit | Strategie |
|-------|-----------|---------------------|-----------|
| Eröffnung | 09:30–10:00 | Sehr hoch (volatil) | KEIN Handel |
| Morning | 10:00–12:00 | Hoch | Normal |
| Mittagsflaute | 12:00–14:00 | Niedrig | Reduzierte Positionsgröße (×0.5) |
| Nachmittag | 14:00–15:30 | Hoch | Normal |
| Schluss | 15:30–16:00 | Sehr hoch (volatil) | KEIN Handel |

---

## 4. Exit-Regeln

### 4.1 Take Profit

```
TP = Einstiegspreis × (1 + THETA × 1.2) = Einstiegspreis × 1.0036
```

Der Faktor 1.2 gibt einen kleinen Puffer über den 0.3% Breakout hinaus – viele Breakouts laufen weiter, und wir wollen nicht zu früh aussteigen.

### 4.2 Stop Loss

```
SL = Einstiegspreis × (1 - THETA × 0.5) = Einstiegspreis × 0.9985
```

Der Stop Loss ist bei der Hälfte des Breakout-Thresholds. Wenn der Preis um 0.15% fällt, war das Signal wahrscheinlich falsch.

### 4.3 Time Stop

```
Wenn nach 30 Minuten weder TP noch SL erreicht → Markt-Exit
```

Nach 30 Minuten ist der Vorhersagehorizont abgelaufen. Länger zu halten ist Spekulation, nicht Modell-basiert.

### 4.4 Signal-Reversal

```
Wenn P_ensemble während des Trades unter 0.35 fällt → sofortiger Exit
```

Ein drastischer Rückgang der Modell-Konfidenz deutet auf eine veränderte Marktlage hin.

---

## 5. Positionsgrößen & Risikomanagement

### 5.1 Kelly Criterion (modifiziert)

Die optimale Positionsgröße nach dem Kelly-Kriterium:

```
f* = (p × b - (1-p)) / b

wobei:
  p  = Precision des Modells (≈ 0.60 nach Kalibrierung)
  b  = Gewinn/Verlust-Ratio = 0.36% / 0.15% = 2.4
  f* = (0.60 × 2.4 - 0.40) / 2.4 = 0.433
```

**Empfehlung:** Verwende **Half-Kelly (f*/2 = 21.7%)**, um die Volatilität zu reduzieren.

### 5.2 Praktische Positionsgrößen-Regeln

| Regel | Wert | Begründung |
|-------|------|------------|
| Max. Risiko pro Trade | 0.5% des Portfolios | Konservativ |
| Max. Risiko pro Tag | 2.0% des Portfolios | Max. 4 Verlust-Trades |
| Max. gleichzeitige Positionen | 3 | Diversifikation |
| Max. Position pro Symbol | 5% des Portfolios | Konzentrationsrisiko |
| Min. Handelsvolumen | $1M Tagesumsatz | Liquidität sicherstellen |

### 5.3 Konkrete Berechnung

```
Positionsgröße ($) = Portfolio × 0.5% / (Einstiegspreis × 0.15%)

Beispiel:
  Portfolio = $100,000
  Max Risk  = $500 pro Trade
  Einstieg  = $150.00
  SL-Distanz = $150.00 × 0.15% = $0.225
  
  Positionsgröße = $500 / $0.225 = 2,222 Shares
  Positionswert   = 2,222 × $150 = $333,300  ← Hebel nötig!
```

**Wichtig:** Ohne Hebel ist die Strategie für kleine Portfolios schwierig umzusetzen. Alternativ: Breiteren Stop-Loss (0.25%) oder Micro-Futures (MNQ) nutzen.

---

## 6. Backtesting-Framework

### 6.1 Walk-Forward-Test

```
┌──────────┐    ┌──────────┐    ┌──────────┐
│ TRAIN    │    │ VAL      │    │ TEST     │
│ bis 2023 │    │ H1 2024  │    │ H2 2024  │
└──────────┘    └──────────┘    └──────────┘
      │               │               │
   Training     Kalibrierung    Out-of-Sample
   der Modelle  der Thresholds  Performance
```

### 6.2 Key Performance Indicators (KPIs)

| KPI | Ziel | Aktuell (MLP V1) |
|-----|------|------------------|
| Win Rate (Precision) | ≥ 60% | 52% ❌ |
| Profit Factor | ≥ 1.3 | Noch nicht berechnet |
| Sharpe Ratio | ≥ 1.0 | Noch nicht berechnet |
| Max Drawdown | ≤ 10% | Noch nicht berechnet |
| Trades pro Tag (ø) | 5–20 | Noch nicht berechnet |
| Avg Trade Duration | 5–25 Min | Noch nicht berechnet |

### 6.3 Backtest-Skript (geplant)

```python
# pseudocode für backtest.py
for symbol in nasdaq100:
    for timestamp in test_data:
        signal = ensemble.predict(features_at(timestamp))
        if entry_conditions_met(signal):
            open_position(symbol, timestamp, size=calculate_size())
        for open_pos in active_positions:
            if exit_condition(open_pos, timestamp):
                close_position(open_pos)
```

---

## 7. Risiken & Fallstricke

### 7.1 Modell-Risiken

| Risiko | Beschreibung | Mitigation |
|--------|-------------|------------|
| **Overfitting** | Modell lernt Rauschen statt Signal | Walk-Forward-Test, Early Stopping |
| **Regime-Change** | 2022–2024 Muster gelten 2025 nicht mehr | Regelmäßiges Retraining (monatlich) |
| **Look-Ahead-Bias** | Features enthalten zukünftige Information | Data-Leakage-Checks (Bereits implementiert ✓) |
| **Survivorship-Bias** | Nur heute existierende NASDAQ-100 Symbole | Historische Index-Zusammensetzung prüfen |

### 7.2 Trading-Risiken

| Risiko | Beschreibung | Mitigation |
|--------|-------------|------------|
| **Slippage** | Ausführungspreis weicht vom Signalpreis ab | Limit-Orders, liquide Symbole |
| **Gap-Risk** | Preis springt über SL hinweg | Keine Trades über Earnings, FOMC |
| **Korrelation** | NASDAQ-100 Aktien sind stark korreliert | Max. 3 Positionen gleichzeitig |
| **Transaction Costs** | Spread + Commission fressen Gewinne | Mindest-Breakout ≥ 2× Spread |
| **Black Swan** | Flash Crash, 9/11-artige Events | Max Daily Loss = 2%, dann Stop |

---

## 8. Nächste Schritte & Roadmap

### 8.1 Phase 1: Modell-Training (aktuell) ⏳

- [x] MLP V1 trainiert (Baseline: 59.6% Acc, F1=0.52)
- [ ] MLP V2 mit GPU + Scaler trainieren
- [ ] LSTM trainieren (sequentiell)
- [ ] GRU trainieren (sequentiell)
- [ ] CNN-1D trainieren (sequentiell)
- [ ] LightGBM trainieren
- [ ] Modell-Vergleich & Ensemble-Gewichte bestimmen

### 8.2 Phase 2: Strategie-Implementierung

- [ ] `backtest.py` – Walk-Forward-Backtest mit Transaktionskosten
- [ ] `calibrate_thresholds.py` – Pro-Symbol-Threshold-Kalibrierung
- [ ] `ensemble.py` – Ensemble-Predictor mit Gewichtung
- [ ] `risk_manager.py` – Positionsgrößen & Risiko-Limits

### 8.3 Phase 3: Live-Trading (Alpha)

- [ ] `live_signal.py` – Real-Time-Signal-Generator (Alpaca Stream)
- [ ] `order_manager.py` – Order-Execution (Alpaca Trading API)
- [ ] `monitor.py` – Live-Monitoring & Alerting
- [ ] Paper-Trading für 4 Wochen vor Live-Einsatz

---

## 9. Zusammenfassung

Diese Trading-Strategie basiert auf einer soliden Datenbasis (10M+ Samples, 82 Features, 100 Symbole) und nutzt ein Multi-Modell-Ensemble, um die Schwächen einzelner Modelle auszugleichen.

**Der Schlüssel zum Erfolg liegt in der Signal-Filterung:**
- Nur die besten 10% der Signale handeln (kalibrierter Threshold)
- Konsens von mindestens 3 Modellen verlangen
- Volumen-Bestätigung erzwingen
- Klare Exit-Regeln mit Take Profit & Stop Loss

Das Modell-Ensemble wird die Precision von 52% (MLP V1) voraussichtlich auf 58–62% verbessern – genug für eine profitable Strategie nach Transaktionskosten.

