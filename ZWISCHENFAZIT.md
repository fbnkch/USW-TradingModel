# 📊 Zwischenfazit: USW-TradingModel – Finale Ergebnisse

> **Stand:** 28.06.2026 – Alle 5 Modelle trainiert & evaluiert
> **Nächster Schritt:** Ensemble-Predictor, Backtesting, Paper Trading

---

## 1. Projekt-Kontext

**Daten:**
- 100 US-Aktien (S&P 500), 1-Minuten-Bars über ~6 Monate
- Train: 10.394.874 | Validation: 2.244.691 | Test: 2.445.521 Samples
- 82 Features: Returns, Slopes, EMAs, Volumen, Tageszeit, etc.
- Klassen-Balance: 49.78% Breakout / 50.22% Kein Breakout

**Baseline (Majority Class):** 50.22% Accuracy → jedes Modell muss diesen Wert schlagen.

---

## 2. Modell-Ergebnisse – Test-Set (final)

| Metrik | Baseline | MLP V2 | LightGBM | LSTM | GRU | CNN |
|--------|----------|--------|----------|------|-----|-----|
| **Accuracy (Test)** | 50.22% | **64.04%** | 56.47%* | 57.09% | 58.42% | 57.93% |
| **Verbesserung** | – | **+13.8 PP** | +6.3 PP* | +6.9 PP | +8.2 PP | +7.7 PP |
| **F1-Score (Breakout)** | – | 0.563 | 0.623† | 0.642 | **0.647** | 0.645 |
| **Precision (Breakout)** | – | **0.600** | 0.485 | 0.505 | 0.514 | 0.511 |
| **Recall (Breakout)** | – | 0.531 | 0.861 | 0.878 | 0.871 | **0.873** |
| **Optimaler Threshold** | – | 0.50 | 0.36 | 0.32 | 0.33 | 0.31 |
| **Trainingszeit** | – | 43.6 Min | 8.1 Min | 125.8 Min | 96.7 Min | 45.3 Min |
| **Parameter** | – | 22.117 | 404 Bäume | 631.617 | 191.297 | 184.385 |
| **Architektur** | – | MLP 82→128→64→32→16 + BN | Gradient Boosting | 2-Layer BiLSTM | 2-Layer GRU | Multi-Kernel Conv1D |

*\*LightGBM wurde nur auf 10/100 Validation-Shards evaluiert. Test-Set folgt.*
*† F1 bei optimalem Threshold (0.36). Bei Threshold 0.50: F1 = 0.550.*

### Confusion Matrices (Test-Set)

**MLP V2** (2.445.521 Samples):
```
              Vorhersage 0   Vorhersage 1
Tatsächlich 0    998.708        377.571    (TN/FP)
Tatsächlich 1    501.864        567.378    (FN/TP)
```
- Accuracy: 64.04% | Precision: 60.04% | Recall: 53.06%

**LSTM** (2.442.611 Samples):
```
              Vorhersage 0   Vorhersage 1
Tatsächlich 0    456.644        918.153    (TN/FP)
Tatsächlich 1    130.045        937.769    (FN/TP)
```
- Accuracy: 57.09% | Precision: 50.53% | Recall: 87.82%

**GRU** (2.442.611 Samples):
```
              Vorhersage 0   Vorhersage 1
Tatsächlich 0    496.835        877.962    (TN/FP)
Tatsächlich 1    137.665        930.149    (FN/TP)
```
- Accuracy: 58.42% | Precision: 51.44% | Recall: 87.11%

**CNN** (2.442.611 Samples):
```
              Vorhersage 0   Vorhersage 1
Tatsächlich 0    483.034        891.763    (TN/FP)
Tatsächlich 1    135.926        931.888    (FN/TP)
```
- Accuracy: 57.93% | Precision: 51.10% | Recall: 87.27%

**LightGBM** (224.934 Validation-Samples):
```
              Vorhersage 0   Vorhersage 1
Tatsächlich 0     46.007         84.846    (TN/FP)
Tatsächlich 1     13.062         81.019    (FN/TP)
```
- Precision: 48.85% | Recall: 86.12%

---

## 3. Key Insights

### 3.1 Zwei Modell-Familien, zwei Profile

```
┌─────────────────────────────────────────────────────────┐
│  FILTER-Typ                │  FINDER-Typ                │
│  (Hohe Precision)          │  (Hoher Recall)            │
├────────────────────────────┼────────────────────────────┤
│  MLP V2:  Prec 60.0%       │  LSTM:  Recall 87.8%       │
│            Recall 53.1%    │         Prec 50.5%         │
│                            │  GRU:   Recall 87.1%       │
│                            │         Prec 51.4%         │
│                            │  CNN:   Recall 87.3%       │
│                            │         Prec 51.1%         │
│                            │  LightGBM: Recall 86.1%    │
│                            │            Prec 48.9%      │
└─────────────────────────────────────────────────────────┘
```

- **MLP V2**: Sagt seltener Breakout voraus, aber wenn, dann mit 60% Trefferquote. Ideal als **Filter**.
- **Sequenzielle Modelle (LSTM/GRU/CNN)**: Finden fast 90% aller Breakouts, aber viele Fehlalarme. Ideal als **Finder**.
- **LightGBM**: Ähnliches Profil wie sequenzielle Modelle, aber auf CPU trainiert.

### 3.2 Warum diese Aufteilung?

Die sequenziellen Modelle sehen 30-Minuten-Fenster und lernen zeitliche Muster (z.B. ansteigendes Volumen + Momentum vor Breakout). Das MLP sieht nur den aktuellen Zeitpunkt – weniger Kontext, dafür robuster gegen Rauschen.

### 3.3 Training-Dynamics

**MLP V2:**
- Plateau ab Epoche 7, Bestwert Epoche 15
- Val Loss 0.6857 → nahe am Bayes-Error für 1-Minuten-Finanzdaten

**LSTM:**
- Schnelle Verbesserung bis Epoche 6, dann zähes Plateau
- Bestwert Epoche 26 (späte Verbesserung nach LR-Reduktion)
- Val Loss 0.6333 – niedrigster aller Modelle

**GRU:**
- Bestwert Epoche 17
- Val Loss 0.6258 – leicht besser als LSTM
- Früherer Stop (27 Epochen) als LSTM (36)

**CNN:**
- Bestwert bereits Epoche 2 (!)
- Danach kein signifikanter Fortschritt – begrenzte Kapazität
- Kürzeste Trainingszeit der GPU-Modelle

**LightGBM:**
- 404 Bäume, schnelles Training auf CPU
- Val AUC 0.685, Train AUC 0.682 (kaum Overfitting)

---

## 4. Ensemble-Strategie

Die zwei Modell-Familien ergänzen sich perfekt:

```
┌──────────────┐     ┌──────────────┐
│   FINDER     │     │    FILTER    │
│  LSTM/GRU    │────▶│   MLP V2     │────▶ Trade-Signal
│  Recall 88%  │     │  Prec 60%    │
└──────────────┘     └──────────────┘
```

**Regel:**
1. Finder (z.B. GRU) meldet Breakout-Kandidaten (Threshold ~0.33)
2. MLP V2 bestätigt (Threshold 0.50)
3. Nur wenn **beide** feuern → Trade eröffnen

**Erwartete Ensemble-Performance:**
- Precision ≥ 62% (MLP filtert Finder-Fehlalarme)
- Recall ~47-50% (Schnittmenge beider Modelle)
- F1 ≥ 0.55

---

## 5. Feature-Importance (LightGBM Gain)

| Rang | Feature | Gain | Bedeutung |
|------|---------|------|-----------|
| 1 | `return_1m` | 3.784.482 | 1-Minuten-Momentum – stärkster Prädiktor |
| 2 | `Slope_close_1` | 2.399.036 | Kurzfristige Preisrichtung |
| 3 | `minutes_since_open` | 2.324.676 | Tageszeit – Breakouts haben klare zeitliche Muster |
| 4 | `Slope_Slope_EMA_240_1_1` | 263.351 | Beschleunigung des Langzeit-Trends |
| 5 | `volume_norm` | 165.154 | Ungewöhnliches Volumen = Breakout-Treibstoff |

**Fachliche Interpretation:** Die drei dominanten Features machen den Großteil der Vorhersagekraft aus. Kurzfristiges Momentum + Tageszeit + Trendbeschleunigung sind die zentralen Breakout-Signale.

---

## 6. Metriken-Glossar (für die Präsentation)

| Metrik | Bedeutung | Interpretation |
|--------|-----------|---------------|
| **Accuracy** | Wie oft liegt das Modell insgesamt richtig? | 64% MLP = +14 PP über Zufall |
| **Precision** | Wenn Modell "Breakout!" sagt – wie oft stimmt's? | 60% = 6 von 10 Signalen richtig |
| **Recall** | Wie viele echte Breakouts findet das Modell? | 88% LSTM = fast alle |
| **F1-Score** | Harmonisches Mittel aus Precision & Recall | 0.65 GRU = bester Kompromiss |
| **Confusion Matrix** | TN=richtig Kein Breakout, FP=Fehlalarm, FN=verpasst, TP=Breakout erkannt | – |
| **Optimaler Threshold** | Bester Schwellwert (F1-maximierend) | 0.31-0.36 für sequenzielle Modelle |
| **Baseline** | Immer häufigere Klasse raten (50.22%) | Jedes Modell MUSS das schlagen |

---

## 7. Dateien

| Datei | Inhalt |
|-------|--------|
| `ZWISCHENFAZIT.md` | Diese Datei – finaler Stand |
| `TRADING_STRATEGIE.md` | Datenfundierte Trading-Strategie |
| `ANLEITUNG.md` | Schritt-für-Schritt GPU-Training |
| `artifacts/evaluation/*_evaluation.json` | Metriken aller 5 Modelle |
| `artifacts/evaluation/model_comparison.json` | Vergleichstabelle |
| `artifacts/evaluation/06_model_comparison.png` | 4-fach Modellvergleich |
| `dashboard.html` | Interaktives Dashboard |

---

## 8. Nächste Schritte (Deployment-Phase)

1. **Ensemble-Predictor bauen** – Finder (GRU/LSTM) + Filter (MLP) kombinieren
2. **Trading-Strategie ableiten** – Entry/Exit-Regeln aus Ensemble-Signalen
3. **Backtest** – Historische Simulation auf 100 Aktien vs. Buy-and-Hold
4. **Paper Trading** – Live-Test auf Alpaca Paper Account ($100.000)
5. **Performance-Analyse** – Sharpe Ratio, Max Drawdown, Win Rate pro Symbol
