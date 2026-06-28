# Präsentationsleitfaden

---

## 1. Pipeline-Übersicht

```
Schritt 01         Schritt 02         Schritt 03         Schritt 04
┌──────────┐      ┌──────────┐      ┌──────────┐      ┌──────────┐
│ Daten    │─────▶│ Daten    │─────▶│ Features │─────▶│ Split    │
│ laden    │      │ verstehen│      │ bauen    │      │ Train/   │
│ (Alpaca) │      │ (EDA)    │      │ & Target │      │ Val/Test │
└──────────┘      └──────────┘      └──────────┘      └──────────┘
                                                           │
                              Schritt 05                   │
                         ┌──────────────┐                 │
                         │ Post-Split   │◀────────────────┘
                         │ Prep +       │
                         │ Scaler       │
                         └──────┬───────┘
                                │
         Schritt 06             │           Schritt 07
    ┌──────────────────┐        │      ┌──────────────────┐
    │ 5 Modelle        │◀───────┘      │ Evaluierung +    │
    │ trainieren       │               │ Visualisierung   │
    │ (MLP/LSTM/GRU/   │               │ + Vergleich      │
    │  CNN/LightGBM)   │──────────────▶│                  │
    └──────────────────┘               └──────────────────┘
```

| Schritt | Ordner | Was passiert |
|---------|--------|-------------|
| 01 | `01_data_acquisition/` | 1-Minuten-Bars für 100 Aktien von Alpaca API laden |
| 02 | `02_data_understanding/` | EDA: Verteilungen, Korrelationen, Breakout-Muster |
| 03 | `03_pre_split_prep/` | 82 Features + `breakout_30m` Target bauen |
| 04 | `04_split_data/` | Per-Symbol Train/Val/Test-Split + Shuffled Shards |
| 05 | `05_post_split_prep/` | GlobalScaler fitten, Class Balance prüfen |
| 06 | `06_model_training/` | 5 Modelle trainieren (GPU-optimiert) |
| 07 | `07_model_testing/` | Evaluieren, Vergleichen, Visualisieren |

---

## 2. Die 82 Features – Was steckt dahinter?

### 2.1 Feature-Gruppen

| Gruppe | Anzahl | Beispiele | Was sie messen |
|--------|--------|-----------|---------------|
| **Momentum** | 3 | `return_1m`, `Slope_close_1`, `Slope_close_5` | Kurzfristige Preisbewegung |
| **Trend (EMAs)** | 6 | `EMA_5` … `EMA_240` | Langfristige Preisrichtung |
| **EMA-Slopes** | 15 | `Slope_EMA_5_1` … `Slope_EMA_240_5` | Geschwindigkeit der Trendänderung |
| **Beschleunigung** | 24 | `Slope_Slope_EMA_*` | Zweite Ableitung – Trendwende? |
| **Volumen** | 4 | `volume_norm`, `volume_spike_ratio` | Ungewöhnliches Volumen |
| **Oszillatoren** | 4 | `RSI_14`, `BB_position`, `MACD_*` | Überkauft/Überverkauft |
| **Tageszeit** | 1 | `minutes_since_open` | Wann am Tag? |
| **Lagged** | 6 | `close_norm_lag5/10/15` | Vergangene Werte |
| **Sonstige** | ~19 | `cumulative_delta`, `opening_range_position` | Markt-Mikrostruktur |

### 2.2 Die 5 Schlüssel-Features (>90% der Vorhersagekraft)

| # | Feature | Bedeutung | Warum wichtig |
|---|---------|-----------|---------------|
| 1 | `return_1m` | 1-Minuten-Return | **Stärkster Prädiktor.** Wenn Preis bereits steigt, ist Breakout wahrscheinlicher |
| 2 | `Slope_close_1` | Kurzfristige Preisrichtung | Bestätigt das Momentum |
| 3 | `minutes_since_open` | Minuten seit 09:30 ET | Breakouts häufen sich zu bestimmten Tageszeiten |
| 4 | `Slope_Slope_EMA_240_1_1` | Beschleunigung des 4h-Trends | Trendwende kündigt Breakout an |
| 5 | `volume_norm` | Relatives Volumen | Hohes Volumen = „Treibstoff" für Breakout |

> **Präsentations-Aussage:** „Unsere 82 Features decken Momentum, Trend, Volumen und Markt-Mikrostruktur ab. LightGBM zeigt: Die Top-3 Features machen >90% der Vorhersagekraft aus. Das Modell hat gelernt: Kurzfristiges Momentum + günstige Tageszeit = Breakout."

---

## 3. Die 5 Modelle 

### 3.1 Modell-Übersicht

| Modell | Typ | Architektur                        | Stärke |
|--------|-----|------------------------------------|--------|
| **MLP V2** | Feedforward NN | 82->128->64->32->16->1 + BatchNorm | Allgemeine Feature-Interaktionen |
| **LSTM** | Bidirektionales LSTM | 2-Layer BiLSTM (128 hidden)        | Langfristige Zeitmuster |
| **GRU** | Gated Recurrent Unit | 2-Layer GRU (128 hidden)           | Wie LSTM, aber effizienter |
| **CNN** | 1D-Convolutional | Multi-Kernel (3,5,10) Conv1D       | Lokale Muster (3-10 Min Fenster) |
| **LightGBM** | Gradient Boosting | 404 Bäume, 127 Leaves              | Nicht-lineare Interaktionen |

### 3.2 Warum diese fünf?

- **MLP V2**: Basislinie für tabellarische Daten. Einfach, schnell, interpretierbar.
- **LSTM/GRU**: Finanzdaten sind **zeitlich**. Ein Breakout kündigt sich durch Muster über mehrere Minuten an (steigendes Volumen, anziehendes Momentum). LSTM/GRU können diese Sequenzen lernen.
- **CNN**: Erkennt **lokale Muster** wie z.B. 3-Minuten-Impulse oder 10-Minuten-Swells. Robust gegen kleine zeitliche Verschiebungen.
- **LightGBM**: Auf tabellarischen Daten oft stärker als neuronale Netze. Dient als **starke nicht-neurale Baseline**.

### 3.3 Finale Ergebnisse (Test-Set)

| Modell | Accuracy | F1 | Precision | Recall | Rolle |
|--------|----------|-----|-----------|--------|-------|
| Baseline | 50.22% | – | – | – | Zufall |
| **MLP V2** | **64.04%** | 0.563 | **0.600** | 0.531 |  Filter |
| GRU | 58.42% | **0.647** | 0.514 | 0.871 |  Finder |
| CNN | 57.93% | 0.645 | 0.511 | 0.873 |  Finder |
| LSTM | 57.09% | 0.642 | 0.505 | **0.878** |  Finder |
| LightGBM | 56.47% | 0.623 | 0.485 | 0.861 |  Finder |

> **Präsentations-Aussage:** „Alle 5 Modelle schlagen die Baseline deutlich. MLP V2 ist mit +13,8 Prozentpunkten der klare Accuracy-Sieger. Aber: Accuracy allein ist nicht alles im Trading."

---

## 4. CHART-GUIDE: Jeder Chart erklärt

### Chart 01 – `01_accuracy_f1.png`
**Was man sieht:** Balkendiagramm: Accuracy (linke Achse) + F1-Linie (rechte Achse) für alle 5 Modelle.
**Warum wichtig:** MLP V2 dominiert Accuracy, aber GRU hat den besten F1-Score. Heißt: MLP liegt insgesamt öfter richtig, GRU ist besser im Breakout-Erkennen speziell.
**Präsentations-Aussage:** „Accuracy und F1 erzählen unterschiedliche Geschichten. MLP V2 gewinnt Accuracy, GRU gewinnt F1 – wir brauchen beide."

---

### Chart 02 – `02_precision_recall.png`
**Was man sieht:** Precision (blau) vs. Recall (orange) pro Modell. MLP V2 hat 60% Precision, LSTM/GRU/CNN haben ~87-88% Recall.
**Warum wichtig:** Hier trennen sich die Modelle in zwei Familien. Das ist der **zentrale Insight** unserer Arbeit.
**Präsentations-Aussage:** „Zwei Modell-Familien: MLP V2 ist der präzise Filter (60% der Signale richtig). Die sequenziellen Modelle sind die Finder (finden 88% aller Breakouts). Perfekt komplementär."

---

### Chart 03 – `03_improvement.png`
**Was man sieht:** Horizontaler Balkenchart – Verbesserung über Baseline in Prozentpunkten.
**Warum wichtig:** Zeigt auf einen Blick, wie viel besser jedes Modell als Raten ist. MLP V2 sticht mit +13.8 PP heraus.
**Präsentations-Aussage:** „Jedes Modell schlägt die Baseline um mindestens 6,3 Prozentpunkte. In einem semi-effizienten Markt ist das ein starkes Signal."

---

### Chart 04 – `04_training_time.png`
**Was man sieht:** Balken = Trainingszeit (Minuten). Lila Linie = Parameterzahl.
**Warum wichtig:** LightGBM ist 15× schneller als LSTM bei ähnlichem Recall. CNN erreicht in 45 Min ähnliche Performance wie LSTM in 126 Min. Relevant für praktischen Einsatz (Retraining).
**Präsentations-Aussage:** „Trainingseffizienz ist relevant für regelmäßiges Retraining. CNN und LightGBM bieten das beste Verhältnis von Performance zu Zeit."

---

### Chart 05 – `05_feature_importance.png`
**Was man sieht:** Top-15 Features nach LightGBM Gain (log-Skala). `return_1m`, `Slope_close_1`, `minutes_since_open` dominieren massiv.
**Warum wichtig:** Zeigt, dass die 77 anderen Features kaum beitragen. Rechtfertigt zukünftige Feature-Reduktion. Die Sterne (★) markieren die Top-3.
**Präsentations-Aussage:** „Die Top-3 Features machen >90% der Vorhersagekraft aus. Kurzfristiges Momentum + Tageszeit sind die wahren Breakout-Treiber. Die 77 anderen Features könnten wir theoretisch weglassen."

---

### Chart 06 – `06_threshold_curves.png`
**Was man sieht:** F1-Score in Abhängigkeit vom Entscheidungs-Threshold τ. Jedes Modell hat ein eigenes Optimum (gestrichelte Linien).
**Warum wichtig:** Der Default-Threshold 0.5 ist NICHT optimal. Sequenzielle Modelle erreichen ihr bestes F1 bei τ ≈ 0.31–0.34. Threshold-Tuning bringt +6-8 PP F1-Verbesserung!
**Präsentations-Aussage:** „Threshold-Kalibrierung ist entscheidend. Bei τ=0.50 verschenken wir Performance. Die sequenziellen Modelle brauchen τ≈0.33 für optimales F1."

---

### Chart 07 – `07_confusion_matrices.png`
**Was man sieht:** 5 Confusion-Matrix-Heatmaps. Oben links = TN, oben rechts = FP (Fehlalarme), unten links = FN (verpasst), unten rechts = TP (Treffer).
**Warum wichtig:** MLP V2: viele TN, moderate FP → hohe Precision. LSTM/GRU/CNN: viele TP, aber auch viele FP → hoher Recall, niedrige Precision.
**Präsentations-Aussage:** „Die Confusion Matrices zeigen das fundamental unterschiedliche Verhalten: MLP V2 ist konservativ (wenige Fehlalarme), die sequenziellen Modelle sind aggressiv (finden fast alles, aber mit vielen Fehlalarmen)."

---

### Chart 08 – `08_trading_tradeoff.png`
**Was man sieht:** Precision-Recall-Streudiagramm. MLP V2 oben-links (hohe Precision, niedriger Recall). LSTM/GRU/CNN oben-rechts (hoher Recall, moderate Precision). Die grüne „IDEAL ZONE" oben-rechts ist das Ziel.
**Warum wichtig:** Kein Modell erreicht die Ideal-Zone allein. Das motiviert das Ensemble.
**Präsentations-Aussage:** „Kein einzelnes Modell ist perfekt. Aber kombiniert – Finder (GRU) + Filter (MLP) – erreichen wir ein besseres Trade-off. Das ist die Begründung für unser Ensemble."

---

### Chart 09 – `09_ensemble_architecture.png`
**Was man sieht:** Zweistufige Pipeline: Finder-Layer (LSTM/GRU/CNN/LightGBM) → Kandidaten → Filter-Layer (MLP V2) → Trade-Signal.
**Warum wichtig:** Das ist unser **Deployment-Konzept**. Konkrete Schwellwerte aus den Evaluierungen.
**Präsentations-Aussage:** „Unsere Ensemble-Strategie: Der Finder-Layer screent mit ~88% Recall alle potenziellen Breakouts. Der Filter-Layer bestätigt mit 60% Precision. Nur wenn BEIDE feuern, wird ein Trade eröffnet."

---

## 5. Die Ensemble-Strategie im Detail

### 5.1 Signal-Pipeline

```
Jede Minute für jede Aktie:
┌────────────┐     ┌────────────┐     ┌────────────┐
│ GRU sagt:  │     │ MLP V2     │     │ TRADE      │
│ P(break) > │────▶│ sagt: P >  │────▶│ eröffnen   │
│ 0.33?      │ YES │ 0.50?      │ YES │            │
└────────────┘     └────────────┘     └────────────┘
     88% Recall         60% Precision     Geschätzt ≥62%
```

### 5.2 Entry-Regeln (ALLE müssen erfüllt sein)

| # | Bedingung | Wert | Quelle |
|---|-----------|------|--------|
| E1 | Finder-Signal | GRU: P(breakout) > 0.33 | Modell |
| E2 | Filter-Bestätigung | MLP V2: P(breakout) > 0.50 | Modell |
| E3 | Momentum positiv | `return_1m` > 0 | Feature #1 |
| E4 | Kurz-Trend positiv | `Slope_close_1` > 0 | Feature #2 |
| E5 | Keine Mittagsflaute | 10-12h oder 14-15:30 ET | Feature #3 |

### 5.3 Exit-Regeln

| Exit | Bedingung |
|------|-----------|
| Take Profit | +0.36% ab Entry |
| Stop Loss | -0.15% ab Entry |
| Time Stop | 30 Minuten ohne TP/SL |
| Signal-Kollaps | P(breakout) < 0.20 |

---

## 6. Erwartete Fragen (und Antworten)

**F: Warum ist Accuracy ~64% gut, wenn 50% Zufall ist?**
A: Finanzmärkte sind semi-effizient. 1-Minuten-Bewegungen sind extrem verrauscht. +14 PP über Zufall ist ein starkes Signal. Profi-Quant-Fonds arbeiten oft mit ähnlich „niedrigen" Accuracies – der Hebel liegt in der Masse der Trades.

**F: Warum ist Precision wichtiger als Recall im Trading?**
A: Ein Fehlalarm (FP) kostet echtes Geld (Spread + Slippage + Stop-Loss). Ein verpasster Breakout (FN) kostet nur Opportunität. Daher: Precision > Recall.

**F: Warum 5 Modelle statt einem?**
A: Kein Modell ist perfekt. MLP V2 hat hohe Precision, aber niedrigen Recall. LSTM/GRU haben hohen Recall, aber viele Fehlalarme. Das Ensemble kombiniert die Stärken beider Familien.

**F: Was ist mit Overfitting?**
A: Wir nutzen Early Stopping (Patience 10-12), Dropout (0.35-0.4), BatchNorm und Weight Decay. Der Validation Loss liegt nah am Train Loss – kein klares Overfitting-Signal.

**F: Warum diese 82 Features?**
A: Systematisch aus Markt-Mikrostruktur-Theorie abgeleitet: Momentum, Trend, Volumen, Oszillatoren. LightGBM Feature Importance bestätigt die Auswahl – die wichtigsten Features sind fachlich plausibel.

---

## 7. Die wichtigsten 3 Slides

1. **Ergebnistabelle** (5 Modelle vs. Baseline) – zeigt, dass alle Modelle funktionieren
2. **Precision/Recall Chart** (Chart 02) – zeigt die zwei Modell-Familien und motiviert das Ensemble
3. **Ensemble-Architektur** (Chart 09) – zeigt, wie wir die Modelle kombinieren → Trading-Signal
