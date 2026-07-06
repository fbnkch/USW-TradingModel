# USW-TradingModel — Modellarchitekturen im Vergleich

## 1. Warum genau diese fünf Modelle?

Die Modellauswahl folgt dem Prinzip der **architektonischen Diversifikation**: Jedes Modell verarbeitet dieselben 82 Features auf eine grundlegend andere Weise. Wo ein Modell blinde Flecken hat, kann ein anderes kompensieren.

| Modell | Architekturfamilie | Kernidee | Parameter | Accuracy | Δ zur Baseline |
|--------|-------------------|----------|-----------|----------|----------------|
| **MLP V2** | Feed-Forward | Reine Feature-Interaktion, keine Sequenz | 22.117 | 64,04% | **+13,82 PP** |
| **GRU** | Rekurrent (GRU) | Sequenzmuster, ~20% schneller als LSTM | 191.297 | 58,42% | +8,20 PP |
| **CNN-1D** | Konvolutional | Lokale Muster in 3 Zeitskalen (3/5/10 min) | 184.385 | 57,93% | +7,71 PP |
| **LSTM** | Rekurrent (BiLSTM) | Sequenzmuster, Long-Range-Dependencies | 631.617 | 57,09% | +6,87 PP |
| **LightGBM** | Tree Ensemble | Nichtlineare Splits, Feature-Importance, schnell | ~2,1 Mio. | 56,47% | +6,25 PP |

Die fünf Architekturen decken **vier fundamental verschiedene Lernparadigmen** ab:

1. **Feed-Forward (MLP)** — Lernt Feature-Interaktionen ohne Zeitbezug. Sieht einen 82-dimensionalen Vektor und kombiniert ihn durch gewichtete Summen. Kein Gedächtnis, keine Reihenfolge.

2. **Rekurrent (LSTM + GRU)** — Lernt zeitliche Abhängigkeiten über eine Sequenz von 30 Minuten. Beide haben ein "Gedächtnis" über die Sequenzlänge. LSTM mit separatem Cell-State (langfristig), GRU mit fusioniertem Gate-Mechanismus (effizienter).

3. **Konvolutional (CNN-1D)** — Lernt lokale Muster durch gleitende Filter über die Zeitachse. Drei verschiedene Kernelgrößen (k=3,5,10) erkennen Impulse (3 min), Trends (5 min) und Schwellbewegungen (10 min) parallel.

4. **Tree Ensemble (LightGBM)** — Lernt durch rekursive, datengetriebene Splits entlang einzelner Features. Nichtlinear, nicht parametrisch im klassischen Sinne, und liefert als einziges Modell native Feature-Importance. Kein Gradient Descent — stattdessen sequentielles Boosting auf Residuen.

---

## 2. Detaillierte Architekturbeschreibungen

### 2.1 MLP V2 — Feed-Forward Breakout Classifier

```
Input(82) → Hidden(128) → Hidden(64) → Hidden(32) → Hidden(16) → Output(1)
             BatchNorm       BatchNorm       BatchNorm       Dropout 0.22
             ReLU            ReLU            ReLU
             Dropout 0.40    Dropout 0.34    Dropout 0.28
```

**Warum so aufgebaut?**

- **Trichterarchitektur (128→64→32→16):** Erzwingt eine hierarchische Kompression der 82 Features. Jede Schicht muss eine zunehmend abstraktere Repräsentation lernen. Der Trichter wirkt als impliziter Information-Bottleneck — irrelevante Feature-Interaktionen werden herausgefiltert.

- **Aggressives Dropout (0.40 startend):** MLPs auf tabularen Finanzdaten neigen zu Overfitting, weil sie keine strukturellen Induktions-Biases haben (kein Sequenzmodell, keine Translation-Invarianz). Das hohe initiale Dropout zwingt das Netzwerk, redundante Repräsentationen zu lernen. Das absteigende Dropout (0.40→0.22) gibt den tieferen, kompakteren Schichten mehr Kapazität — dort sind weniger Neuronen, also weniger Redundanz nötig.

- **BatchNorm vor jeder Aktivierung:** Bei 10 Mio. Samples und Batch-Größe 1024 sind ~10.000 Batches pro Epoch verfügbar. BatchNorm stabilisiert das Training über diese vielen Mini-Batches und erlaubt höhere Lernraten.

- **AdamW mit weight_decay=1e-5:** Weight Decay (entkoppelt von Adam's adaptiven Learning Rates) regularisiert sanft, ohne die adaptiven Schrittweiten zu verzerren.

**Stärken:**
- Beste Accuracy aller Modelle (64,04%)
- Beste Precision auf "Kein Breakout" (66,6%) — vermeidet False Positives
- Kein Threshold-Tuning nötig (optimaler Threshold = 0.5)
- Schnellste Inferenz (einfache Matrix-Multiplikationen)
- Kein Overfitting (Train/Val-Loss nahezu identisch)

**Schwächen:**
- Kein Sequenzverständnis — die Reihenfolge der 30 Minuten geht verloren (Input ist flach: batch×82)
- Recall auf Breakouts nur 53,1% — übersieht fast die Hälfte der tatsächlichen Breakouts
- Kann keine zeitlichen Muster wie "Beschleunigung" oder "Umkehr" direkt modellieren

---

### 2.2 LSTM — Bidirectional Sequence Model

```
Input(30×82) → BiLSTM(2 Layers, hidden=128) → Concat(fwd+bwd, 256) → LayerNorm
             → Linear(256→64) → Linear(64→32) → Linear(32→1) → Sigmoid
```

**Warum so aufgebaut?**

- **Bidirektional (BiLSTM):** In Finanzzeitreihen ist der Kontext in beide Richtungen informativ. Ein Kursanstieg in Minute 25 wird anders interpretiert, wenn Minute 26–30 fallen (der Forward-LSTM sieht das nicht, der Backward-LSTM schon). Bidirektionalität verdoppelt den Hidden-State (128+128=256), gibt aber Zugriff auf den vollen zeitlichen Kontext.

- **2 Layer, hidden=128:** Zwei LSTM-Layer erlauben hierarchische zeitliche Abstraktion — Layer 1 lernt kurzfristige Dynamiken (2–5 Minuten), Layer 2 kombiniert diese zu längeren Mustern (10–30 Minuten). 128 Hidden-Units geben genug Kapazität für 82-dimensionale Inputs pro Zeitschritt.

- **LayerNorm statt BatchNorm:** LayerNorm operiert über die Feature-Dimension pro Sample — unabhängig von der Batch-Größe. Bei rekurrenten Netzen ist das kritisch, weil die effektive "Batch-Größe" über die Zeitachse variiert und BatchNorm's Statistiken instabil werden.

- **Classifier-Kopf (256→64→32→1):** Nach der Sequenzverarbeitung komprimiert ein kleiner MLP-Kopf den Concatenated-State auf die finale Logit. Gleiche Trichterlogik wie beim MLP, aber mit sanfterem Dropout (0.245→0.175), da die rekurrente Architektur bereits strukturelle Regularisierung bietet.

- **Cell-State + Hidden-State:** LSTM's dualer Speicher (Cell-State für Langzeitinformation, Hidden-State für Kurzzeit-Output) ist spezifisch nützlich für Finanzdaten: Der Cell-State kann eine über 30 Minuten aufgebaute Trendrichtung speichern, während der Hidden-State die aktuelle Mikrostruktur verarbeitet.

**Stärken:**
- Höchster Recall auf Breakouts (87,8%) — findet fast 9 von 10 Breakouts
- Bidirektionalität erfasst Vor- und Nachlauf-Muster
- 631K Parameter geben hohe Modellkapazität
- Long-Range-Dependencies über 30 Minuten durch Cell-State

**Schwächen:**
- Deutliches Overfitting-Tendenz (braucht Threshold-Optimierung: 0.32 statt 0.5)
- Geringste Precision (50,5%) — viele False Positives (918K FP)
- Teuerste Architektur in Training und Inferenz (631K Parameter, sequentielle Verarbeitung)
- Accuracy nur +6,87 PP — drittschlechtestes Modell

---

### 2.3 GRU — Recurrent Sequence Model

```
Input(30×82) → GRU(2 Layers, hidden=128) → h_last(128) → LayerNorm
             → Linear(128→64) → Linear(64→32) → Linear(32→1) → Sigmoid
```

**Warum so aufgebaut?**

- **GRU statt LSTM:** GRU fusioniert Forget- und Input-Gate in ein einziges "Update-Gate" und verzichtet auf den separaten Cell-State. Das reduziert die Parameter pro rekurrenter Einheit um ~25% und beschleunigt das Training um 15–20%. Bei einem 30-Minuten-Fenster ist die vereinfachte Gating-Struktur oft ausreichend — es gibt keine sehr langfristigen Abhängigkeiten wie bei Text oder Sprache.

- **Nur letzter Hidden-State (h_last):** Anders als der BiLSTM, der forward und backward concateniert, nimmt die GRU nur den letzten Hidden-State der Sequenz. Das reduziert die Dimensionalität von 256 auf 128 und zwingt das Netzwerk, die gesamte Sequenzinformation in diesem einen Vektor zu komprimieren. Bei unidirektionaler Verarbeitung ist das der informativste Zeitpunkt (gesamte Sequenz gesehen).

- **Unidirektional:** Die GRU läuft nur vorwärts — im Trading-Kontext eine realistischere Annahme (in Produktion sieht man auch nur Vergangenheitsdaten). Der Verzicht auf Bidirektionalität halbiert die Parameter im rekurrenten Teil (191K vs. 632K).

- **Gleicher Classifier-Kopf wie LSTM:** 128→64→32→1 mit gleichem Dropout-Regime. Die Konsistenz im Classifier-Design über alle neuronalen Modelle hinweg eliminiert den Classifier als Confounder beim Architekturvergleich.

**Stärken:**
- Bester F1-Score aller Modelle (0,647) — beste Balance aus Precision und Recall
- Beste Accuracy unter den Sequenzmodellen (+8,20 PP)
- 191K Parameter — 3× weniger als LSTM bei besserer Performance
- ~15–20% schnelleres Training als LSTM

**Schwächen:**
- Kein Cell-State für sehr langfristige Muster (bei 30 Minuten meist irrelevant)
- Nur vorwärtsgerichtet — verpasst "nachträgliche" Kontextmuster
- Braucht Threshold-Tuning (optimal: 0.334, nicht 0.5)
- 514K False Positives (ähnlich hohe FP-Rate wie LSTM)

---

### 2.4 CNN-1D — Multi-Kernel Convolutional Classifier

```
Input(batch, 30, 82)
  ├─ Conv1D(k=3, 82→64) → BN+ReLU → Conv1D(k=3, 64→64) → BN+ReLU → GlobalMaxPool
  ├─ Conv1D(k=5, 82→64) → BN+ReLU → Conv1D(k=5, 64→64) → BN+ReLU → GlobalMaxPool
  └─ Conv1D(k=10, 82→64) → BN+ReLU → Conv1D(k=10, 64→64) → BN+ReLU → GlobalMaxPool
       ↓ Concat(64+64+64=192)
  BatchNorm → Linear(192→64) → Linear(64→32) → Linear(32→1) → Sigmoid
```

**Warum so aufgebaut?**

- **Drei parallele Kernelgrößen (k=3, 5, 10):** Dies ist der zentrale Architekturentscheid. Finanzzeitreihen enthalten Muster auf verschiedenen Zeitskalen:
  - **k=3 (Cyan):** 3-Minuten-Impulse — schnelle, kurze Preisbewegungen, Micro-Breakouts
  - **k=5 (Purple):** 5-Minuten-Trends — mittelfristige Richtungsbewegungen
  - **k=10 (Pink):** 10-Minuten-Schwellbewegungen — langsame Akkumulation vor dem Breakout

  Jeder Kernel erfasst eine andere zeitliche Auflösung — die Multi-Kernel-Architektur ist eine explizite "Multi-Resolution"-Repräsentation.

- **Zwei Conv1D-Layer pro Branch (82→64→64):** Der erste Layer projiziert die 82 Features auf 64 Channel, der zweite verarbeitet die gelernten Feature-Maps weiter. Zwei Layer geben dem Netzwerk ein rezeptives Feld, das gröber ist als die Kernel-Größe allein — bei k=3 mit zwei Layern effektiv ~5 Minuten Kontext.

- **GlobalMaxPool statt GlobalAvgPool:** Max-Pooling selektiert den stärksten Ausschlag über die Zeitachse. Für Breakout-Detektion ist das sinnvoll: Ein einziger starker Impuls (z.B. plötzlicher Volumensprung) ist oft aussagekräftiger als der Durchschnitt über 30 Minuten. MaxPool ist ein "Trigger-Detektor", AvgPool ein "Zustands-Detektor".

- **Concat aller Branches:** Die drei Zeitskalen werden nicht gemischt, sondern konkateniert. Der Classifier-Kopf kann dann lernen, welche Kombination von Zeitskalen einen Breakout signalisiert (z.B. "k=3 Impuls + k=10 Schwellbewegung = starker Breakout").

- **Nur 12 Epochen:** CNNs konvergieren auf dieser Datenmenge schneller als rekurrente Modelle, weil die konvolutionalen Filter lokale Muster direkt lernen (kein BPTT durch 30 Zeitschritte).

**Stärken:**
- Explizite Multi-Zeitskalen-Repräsentation (3/5/10 min)
- Sehr gute Precision-Recall-Balance (F1=0,645)
- Nur 184K Parameter bei wettbewerbsfähiger Performance
- Schnelles Training (12 Epochen)
- GlobalMaxPool als expliziter "Trigger-Detektor"

**Schwächen:**
- Kernel-Größen sind fest — kann keine dynamischen Zeitskalen lernen
- Max-Pooling verwirft die zeitliche Position des stärksten Signals
- Keine Interaktion zwischen den Zeitskalen vor dem Concat
- 512K False Positives

---

### 2.5 LightGBM — Leaf-wise Gradient Boosting Ensemble

```
Input(82 Features)
  → Tree₀ (fittet Initial-Logits)
  → Tree₁ (fittet Residuen von Tree₀, shrinkage=0.05)
  → Tree₂ (fittet Residuen von Tree₀₊₁, shrinkage=0.05)
  → ...
  → Tree₄₀₃ (fittet Residuen der ersten 403 Bäume)
  → Σ treeᵢ(x) → Sigmoid → Probability [0,1]

Jeder Baum:
  Root → Split S₁ → ... → 127 Blätter (leaf-wise, depth variabel 2–8)
  Split-Kriterien: Feature ≤ Threshold (z.B. RSI_14 ≤ 32.5)
  Blatt-Wert: λ ∈ [-0.09, +0.09] (Residual-Fit)
```

**Warum so aufgebaut?**

- **GBDT (nicht GOSS/DART/RF):** Standard-Gradient-Boosting wurde gewählt, weil bei 10 Mio. Samples und 82 Features die Datenmenge ausreicht, um ohne aggressives Sampling zu trainieren. GOSS wäre bei >100 Mio. Samples oder strikter Latenzanforderung sinnvoll gewesen. DART hätte Dropout auf Baum-Ebene eingeführt, was bei 404 Bäumen Overfitting-Risiko birgt.

- **404 Bäume (Early Stopping bei 5.000 max):** Early Stopping mit Patience=50 terminierte das Training nach 404 Runden — das Modell hatte bei Runde 354 sein Bestes erreicht und 50 Runden lang keine Verbesserung. 404 Bäume bedeuten: Das Residuum nach 403 Bäumen enthält kein konsistentes Signal mehr, das ein weiterer Baum extrahieren könnte.

- **num_leaves=127, max_depth=8:** 127 Blätter bei max. Tiefe 8 erlaubt sehr buschige, asymmetrische Bäume. Im Leaf-wise-Modus wächst der Baum dort, wo der Loss am stärksten fällt — nicht gleichmäßig über alle Ebenen. Ein Ast kann Tiefe 8 erreichen (256 mögliche Blätter auf diesem Pfad), während ein anderer bei Tiefe 3 terminiert. Das ist effizienter als Level-wise-Growth (XGBoost Default), weil es keine "unnötigen" Splits in informationsarmen Regionen gibt.

- **learning_rate=0.05:** Relativ moderates Shrinkage. Bei nur 404 Bäumen (Early Stopping) ist ein höheres lr als 0.01–0.02 angemessen — mit lr=0.01 wären ~2.000 Bäume nötig gewesen für die gleiche Fit-Qualität.

- **max_bin=255:** Die diskreten Histogram-Bins (255 pro Feature) erlauben Split-Suche in O(#bins) statt O(#data). Bei 10 Mio. Samples ist das der entscheidende Geschwindigkeitsvorteil: Training in 8 Minuten auf CPU.

- **subsample=0.8, colsample_bytree=0.8:** Row-Sampling (Bagging) und Column-Sampling dekorrelieren die Bäume und wirken als Regularisierung. 80% ist ein konservativer Wert — hoch genug, um wenig Signal zu verlieren, niedrig genug für effektive Diversifikation.

- **EFB (Exclusive Feature Bundling):** Automatisch aktiv. Viele der 82 Features sind sparse oder schließen sich gegenseitig aus — EFB bündelt sie zu kompakten Feature-Gruppen. Reduziert die effektive Dimensionalität um schätzungsweise 30–50%.

**Stärken:**
- Kein Overfitting (Val-AUC 0.685 > Train-AUC 0.682)
- Training in nur 8 Minuten (vs. 44 Minuten MLP)
- Native Feature-Importance (Top 3: return_1m, Slope_close_1, minutes_since_open)
- Interpretierbare Split-Entscheidungen (Feature ≤ Threshold)
- Automatische Behandlung fehlender Werte
- Geringster False-Negative-Anteil (13K FN) — verpasst wenige Breakouts

**Schwächen:**
- Niedrigste Accuracy (+6,25 PP) — schwächstes Modell insgesamt
- Keine Feature-Interaktionen über Baumgrenzen hinweg (jeder Split ist univariat)
- Kann keine glatten Funktionsverläufe lernen (treppenförmige Entscheidungsgrenzen)
- Hohe False-Positive-Rate (84K FP bei nur 225K Val-Samples)
- 5.6 MB Modellgröße auf Disk (vs. <1 MB für neuronale Modelle)

---

## 3. Komplexitätsvergleich

| Aspekt | MLP | LSTM | GRU | CNN | LightGBM |
|--------|-----|------|-----|-----|----------|
| **Parameter** | 22.117 | 631.617 | 191.297 | 184.385 | ~2.134.217 |
| **Trainingszeit** | 43,6 min | ? | ? | ? | 8,1 min |
| **Modellgröße (Disk)** | ~90 KB | ~2.5 MB | ~770 KB | ~740 KB | 5.6 MB |
| **Inferenz (relativ)** | 1× (Referenz) | ~30× (sequentiell) | ~20× (sequentiell) | ~8× (Convolutions) | ~0.5× (Tree Traversal) |
| **Epochen / Runden** | 27 (best: 15) | 36 | 27 | 12 | 404 Trees |
| **Hidden-Dims** | [128,64,32,16] | [256,64,32] | [128,64,32] | [192,64,32] | 127 leaves×404 |
| **Regularisierung** | Drop 0.40–0.22 | Drop 0.35–0.175 | Drop 0.35–0.175 | Drop 0.245–0.175 | L1=0.1, L2=0.1, Subsampling |

**Komplexitäts-Kontext:**

- LightGBM hat die meisten "Parameter" (2.1M), aber jeder Parameter ist ein simpler Skalar (Split-Threshold oder Leaf-Value). Die neuronale Netze haben weniger Parameter, aber jeder wird durch Gradientenfluss über die gesamte Architektur gelernt — eine komplexere Optimierungsaufgabe.

- Die Sequenzmodelle (LSTM, GRU) sind komplexer in der Inferenz als MLP/CNN, weil sie 30 Zeitschritte sequentiell verarbeiten müssen — nicht parallelisierbar.

- CNN ist das effizienteste neuronale Modell: 184K Parameter, nur 12 Epochen, wettbewerbsfähige Performance. Die induktiven Biases (Translationsinvarianz, lokale Konnektivität) passen gut zum Problem.

---

## 4. Komplementarität und Blinden-Fleck-Abdeckung

### 4.1 Was jedes Modell sieht — und was es nicht sieht

| Modell | Sieht | Sieht nicht |
|--------|-------|-------------|
| **MLP** | Globale Feature-Interaktionen (alle 82 Features gleichzeitig) | Zeitliche Reihenfolge, lokale Muster, Sequenzabhängigkeiten |
| **LSTM** | Bidirektionale Sequenzmuster, Long-Range-Dependencies | Nicht-sequenzielle Feature-Interaktionen, lokale Impulse |
| **GRU** | Vorwärts-Sequenzmuster, effiziente Langzeitabhängigkeit | Rückwärts-Kontext, Multi-Zeitskalen parallel |
| **CNN** | Lokale Muster in 3 festen Zeitskalen (3/5/10 min) | Globale Feature-Interaktionen, variable Zeitskalen, Langzeitdependenzen |
| **LightGBM** | Univariate Splits, nichtlineare Schwellwerte, Feature-Wichtigkeit | Feature-Interaktionen (jeder Split isoliert), glatte Funktionen |

### 4.2 Komplementäre Stärken

**MLP ↔ Sequenzmodelle (LSTM/GRU):**
- MLP gewinnt, wenn die besten 82 Features für sich sprechen (64% Acc). Es lernt nichtlineare Kombinationen aller Features — etwas, das LSTM/GRU durch die sequentielle Verarbeitung nur indirekt können.
- LSTM/GRU gewinnen, wenn die zeitliche Abfolge zählt: Beschleunigung, Umkehr, Divergenz.
- **Ensemble-Implikation:** MLP und GRU zusammen sollten besser sein als jedes allein — sie sehen fundamental verschiedene Aspekte der Daten.

**GRU ↔ LSTM:**
- GRU schlägt LSTM in Accuracy (+8.2 vs +6.9 PP) und F1 (0.647 vs 0.641) — bei einem Drittel der Parameter. Der einfachere Gate-Mechanismus regularisiert implizit.
- LSTM schlägt GRU im Recall (87.8% vs 87.1%) — der Cell-State findet marginal mehr Breakouts.
- **Ensemble-Implikation:** Beide sind stark korreliert (ähnliche Architektur). Diversifikationsgewinn durch Hinzunahme beider ist gering.

**CNN ↔ LSTM/GRU:**
- CNN erfasst lokale Muster explizit und parallel (3 Zeitskalen gleichzeitig). LSTM/GRU erfassen Muster implizit und sequentiell.
- CNN's GlobalMaxPool ist ein Trigger-Detektor — LSTM's Hidden-State ist ein Zustandsspeicher. Fundamental verschiedene Repräsentationen.
- **Ensemble-Implikation:** CNN + GRU ist das vielversprechendste Paar — ähnlich gute Performance, aber unterschiedliche Architekturbias.

**LightGBM ↔ Neuronale Modelle:**
- LightGBM's Stärke ist die Interpretierbarkeit und Geschwindigkeit, nicht die reine Performance. Es ist das einzige Modell, das direkte Feature-Importance liefert.
- Die univariaten Splits komplementieren die dichten Feature-Interaktionen der neuronalen Modelle.
- LightGBM ist robuster gegen Feature-Skalen und Ausreißer als neuronale Modelle.
- **Ensemble-Implikation:** LightGBM im Stacking mit MLP/GRU als Meta-Learner — LightGBM's Vorhersagen als zusätzliches Feature für die neuronalen Modelle.

### 4.3 Blinden-Fleck-Matrix

| Blinder Fleck | Betroffene Modelle | Abgedeckt durch |
|---------------|-------------------|-----------------|
| Kein Sequenzkontext | MLP, LightGBM | LSTM, GRU, CNN |
| Keine globalen Feature-Interaktionen | LightGBM (univariate Splits) | MLP |
| Nur feste Zeitskalen | CNN (k=3,5,10) | LSTM (dynamisch), GRU |
| Overfitting bei Threshold 0.5 | LSTM, GRU, CNN, LightGBM | MLP (Threshold=0.5 optimal) |
| Keine Interpretierbarkeit | MLP, LSTM, GRU, CNN | LightGBM (Split-Entscheidungen) |
| Langsame Inferenz | LSTM, GRU | MLP, CNN, LightGBM |
| Kein Bidirektionaler Kontext | MLP, GRU, CNN, LightGBM | LSTM |
| Treppenförmige Entscheidungsgrenzen | LightGBM | Alle neuronalen Modelle |

---

## 5. Performance-Rangfolge und praktische Bewertung

### 5.1 Nach Accuracy (Test-Set, 2.445.521 Samples)

| Rang | Modell | Accuracy | Verbesserung |
|------|--------|----------|-------------|
| 1 | **MLP V2** | 64,04% | +13,82 PP |
| 2 | **GRU** | 58,42% | +8,20 PP |
| 3 | **CNN-1D** | 57,93% | +7,71 PP |
| 4 | **LSTM** | 57,09% | +6,87 PP |
| 5 | **LightGBM** | 56,47% | +6,25 PP |

### 5.2 Nach F1-Score (Breakout-Klasse, optimierter Threshold)

| Rang | Modell | Best F1 | Best Threshold | F1 bei 0.5 |
|------|--------|---------|---------------|------------|
| 1 | **GRU** | 0,647 | 0,334 | 0,563 |
| 2 | **CNN-1D** | 0,645 | 0,314 | 0,579 |
| 3 | **LSTM** | 0,642 | 0,320 | 0,576 |
| 4 | **MLP V2** | 0,608 | 0,500 | 0,563 |
| 5 | **LightGBM** | 0,623 | 0,355 | 0,550 |

### 5.3 Praktische Bewertung für den Produktiveinsatz

**Beste reine Performance:** MLP V2 (+13.82 PP) — aber nur wenn Accuracy die relevante Metrik ist.

**Beste Breakout-Erkennung:** GRU (F1=0,647) — beste Balance aus "Finden" und "nicht falsch Alarm schlagen".

**Beste Interpretierbarkeit:** LightGBM — erklärt *warum* ein Breakout vorhergesagt wird (welches Feature, welcher Schwellwert).

**Beste Trainingseffizienz:** LightGBM (8 min) und CNN (12 Epochen).

**Beste Inference-Effizienz:** MLP und LightGBM — beide in Microsekunden pro Sample.

---

## 6. Empfehlungen

### 6.1 Ensemble-Strategie

Die fünf Modelle sind architektonisch divers genug, dass ein Ensemble signifikant besser sein sollte als jedes Einzelmodell. Empfohlene Strategie:

1. **Simple Average:** Mean(MLP, GRU, CNN) — die drei besten Modelle mit unterschiedlichen Architekturen. Erwarteter Gain: +2–4 PP über das beste Einzelmodell.

2. **Weighted Ensemble:** Gewichtet nach Val-F1 (GRU=0.30, CNN=0.28, MLP=0.24, LSTM=0.10, LightGBM=0.08). Erwarteter Gain: +3–5 PP.

3. **Stacking mit LightGBM als Meta-Learner:** Die 5 Modell-Outputs als 5 Features → LightGBM lernt die optimale Kombination. Vorteil: Nichtlinear, interpretierbar (welches Modell wird wann vertraut?).

### 4.2 Nächste Schritte

- **Threshold-Kalibrierung:** LSTM, GRU, CNN operieren aktuell mit optimierten Thresholds (0.31–0.33) — das deutet auf mangelhafte Probability-Kalibrierung hin. Isotonic Regression oder Platt Scaling auf den Outputs würde die Modelle auf Threshold=0.5 bringen.
- **CNN-Kernel erweitern:** k=2 (2-Minuten-Impulse) und k=20 (20-Minuten-Makrotrends) als zusätzliche Branches testen.
- **Feature-Selektion:** LightGBM zeigt, dass Top-3 Features ~90% des Gains liefern. Ein reduziertes Feature-Set (Top-20) könnte Overfitting in LSTM/GRU reduzieren.
- **LightGBM mit GOSS testen:** Bei 10 Mio. Samples könnte GOSS (top_rate=0.2, other_rate=0.1) das Training weiter beschleunigen und Overfitting reduzieren — und möglicherweise die Accuracy verbessern.

---

## A. Anhang — Hyperparameter-Referenz

### MLP V2
```
epochs=50 (best@15), batch_size=1024, lr=0.001, weight_decay=1e-5
hidden_sizes=[128, 64, 32, 16], dropout=[0.40, 0.34, 0.28, 0.22]
optimizer=AdamW, loss=BCEWithLogitsLoss
```

### LSTM
```
input_shape=(30, 82), hidden_size=128, num_layers=2, bidirectional=True
dropout=0.35, classifier_dropout=[0.245, 0.175]
classifier=[256→64, 64→32, 32→1]
```

### GRU
```
input_shape=(30, 82), hidden_size=128, num_layers=2, bidirectional=False
dropout=0.35, classifier_dropout=[0.245, 0.175]
classifier=[128→64, 64→32, 32→1]
```

### CNN-1D
```
input_shape=(batch, 30, 82) → (batch, 82, 30)
kernels=[3, 5, 10], channels=[82→64, 64→64] per branch
GlobalMaxPool → Concat(192) → classifier=[192→64, 64→32, 32→1]
dropout=[0.245, 0.175]
```

### LightGBM
```
objective=binary, metric=[binary_logloss, auc], boosting_type=gbdt
num_leaves=127, max_depth=8, learning_rate=0.05, n_estimators=5000 (best@404)
max_bin=255, min_child_samples=100, subsample=0.8, colsample_bytree=0.8
reg_alpha=0.1, reg_lambda=0.1, scale_pos_weight=1.0088
```

---

*Analyse erstellt am 28.06.2026 auf Basis der trainierten Modelle und Evaluierungs-Metriken in `artifacts/evaluation/`.*
