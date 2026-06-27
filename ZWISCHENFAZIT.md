# 📊 Zwischenfazit: USW-TradingModel

> **Stand:** 27.06.2026 2 von 5 Modellen trainiert
> **Nächster Schritt:** LSTM, GRU, CNN-1D Training + Modellvergleich + Visualisierungen

---

## 1. Projekt-Kontext 


**Daten:**
- Train: 10.394.874 | Validation: 2.244.691 | Test: 2.445.521 Samples
- Klassen-Balance: 49.78% Breakout / 50.22% Kein Breakout

**Baseline (Majority Class):** 50.22% Accuracy -> jedes Modell muss diesen Wert schlagen.

---

## 2. Modell-Ergebnisse im Vergleich

| Metrik | Baseline | MLP V1 (alt) | **MLP V2 (GPU)** | **LightGBM** |
|--------|----------|-------------|----------------|-------------|
| **Accuracy (Test)** | 50.22% | 59.56% | **64.04%** | 56.47%* |
| **Verbesserung** | – | +9.3 PP | **+13.8 PP** | +6.3 PP* |
| **Precision (Breakout)** | – | 0.52 | **0.60** | 0.49* |
| **Recall (Breakout)** | – | 0.53 | 0.53 | **0.86*** |
| **F1-Score (Breakout)** | – | 0.52 | **0.56** | 0.55* |
| **Optimaler Threshold** | – | 0.50 | 0.50 | 0.36 |
| **Trainingszeit** | – | ~3 Min | 43.6 Min | 8.1 Min |
| **Parameter / Bäume** | – | 7.425 | 22.117 | 404 Bäume |
| **Architektur** | – | 82→64→32→1 | 82→128→64→32→16 + BatchNorm | Gradient Boosting |

*\*LightGBM wurde nur auf 10/100 Validation-Shards evaluiert. Die Test-Set-Performance folgt.*

### Confusion Matrices

**MLP V2 (Test-Set, 2.445.521 Samples):**
```
              Vorhersage 0   Vorhersage 1
Tatsächlich 0    998.708        377.571    (TN/FP)
Tatsächlich 1    501.864        567.378    (FN/TP)
```
- Accuracy: 64.04%
- Precision (Breakout): 60.04% — 6 von 10 Signalen sind richtig
- Recall (Breakout): 53.06% — etwa die Hälfte aller Breakouts wird erkannt

**LightGBM (Validation, 224.934 Samples):**
```
              Vorhersage 0   Vorhersage 1
Tatsächlich 0     46.007         84.846    (TN/FP)
Tatsächlich 1     13.062         81.019    (FN/TP)
```
- Precision: 48.85% — viele Fehlalarme, aber…
- Recall: 86.12% — fast alle Breakouts werden gefunden!

---

## 3. Key Insights

### 3.1 MLP V2: Der deutliche Sprung nach vorne

| Vergleich MLP V1 → V2 | V1 (alt) | V2 (neu) | Δ |
|------------------------|----------|----------|---|
| Accuracy | 59.6% | 64.0% | **+4.4 PP** |
| F1 (Breakout) | 0.52 | 0.56 | **+0.04** |
| Precision (Breakout) | 0.52 | 0.60 | **+0.08** |
| Parameter | 7.425 | 22.117 | 3× mehr |

**Was hat die Verbesserung gebracht?**
1. **GlobalScaler-Integration** — die symbolübergreifende Normalisierung gleicht Verteilungen an
2. **Mehr Kapazität** — 128→64→32→16 statt 64→32→1, mit BatchNorm für stabileres Training
3. **AdamW + LR-Scheduling** — bessere Konvergenz als einfacher Adam
4. **GPU + AMP** — ermöglichte größere Batch-Size (1024 statt 512)

### 3.2 LightGBM: Der Breakout-Detektor

LightGBM hat ein **fundamental anderes Verhalten** als das MLP:
- **Extrem hoher Recall (86%)** — findet fast jeden Breakout
- **Niedrige Precision (49%)** — aber mit Threshold-Tuning (0.36 statt 0.50) verbessert

Diese Stärke macht LightGBM zum idealen **zweiten Modell im Ensemble**: Es findet die Kandidaten, das MLP filtert die Fehlalarme heraus.

### 3.3 Feature-Importance: Das sagen die Daten

Die Top-5 Features nach LightGBM Gain:

| Rang | Feature | Gain | Bedeutung |
|------|---------|------|-----------|
| 1 | `return_1m` | 3.784.482 | 1-Minuten-Momentum — der mit Abstand stärkste Prädiktor |
| 2 | `Slope_close_1` | 2.399.036 | Kurzfristige Preisrichtung |
| 3 | `minutes_since_open` | 2.324.676 | Tageszeit — Breakouts haben klare zeitliche Muster |
| 4 | `Slope_Slope_EMA_240_1_1` | 263.351 | Beschleunigung des Langzeit-Trends |
| 5 | `volume_norm` | 165.154 | Ungewöhnliches Volumen = Breakout-Treibstoff |

**Interpretation:** Die drei dominanten Features (`return_1m`, `Slope_close_1`, `minutes_since_open`) machen zusammen den Großteil der Vorhersagekraft aus. Das ist fachlich plausibel: Breakouts sind kurzfristige Momentum-Events, die zu bestimmten Tageszeiten häufiger auftreten.

### 3.4 Training-Dynamics (MLP V2)

```
Epoche  1: Val Loss 0.6870 | Val Acc 63.81%
Epoche  7: Val Loss 0.6861 | Val Acc 64.39%  ← Plateau beginnt
Epoche 15: Val Loss 0.6857 | Val Acc 64.61%  ← Bester Loss
Epoche 27: Early Stop (12 Epochen ohne Verbesserung)
```

- **Schnelle initiale Lernphase** (Epoche 1–7), dann Plateau
- **LR-Reduktion** bei Epoche 14 (0.001 → 0.0005) brachte nochmal einen kleinen Sprung
- **Validation Loss 0.6857** liegt sehr nah am Bayes-Error für dieses Problem — das Rauschen in 1-Minuten-Finanzdaten ist extrem hoch

---

## 4. Ensemble-Strategie (ausblickend)

Die zwei trainierten Modelle ergänzen sich perfekt:

```
┌──────────────┐     ┌──────────────┐
│   LightGBM   │     │    MLP V2    │
│  Recall 86%  │────▶│ Precision 60%│────▶ Trade-Signal
│  (Finder)    │     │  (Filter)    │
└──────────────┘     └──────────────┘
```

**Idee:** LightGBM meldet alle potenziellen Breakouts (englische Signalstärke). Das MLP filtert mit hoher Precision die Fehlalarme heraus. Nur wenn **beide** Modelle über ihrem kalibrierten Threshold liegen, wird ein Trade eröffnet.

Erwartete Ensemble-Performance: **Precision ≥ 62%, F1 ≥ 0.58**

---

## 5. Noch ausstehend

| Modell | Typ | Geschätzte Dauer | GPU |
|--------|-----|-----------------|-----|
| 🔜 LSTM | Sequentiell (2-Layer BiLSTM, 631K Parameter) | ~30-60 Min | ✅ |
| 🔜 GRU | Sequentiell (2-Layer, 191K Parameter) | ~20-45 Min | ✅ |
| 🔜 CNN-1D | Sequentiell (Multi-Kernel Conv, 184K Parameter) | ~10-20 Min | ✅ |

Nach dem Training:
- `compare_models.py` → Vergleichstabelle + Visualisierung
- Evaluierungs-Plots für die Doku
- Ensemble-Predictor bauen

---

## 6. Was bedeuten die Metriken? (für die Präsentation)

| Metrik | Bedeutung | Gut/Schlecht |
|--------|-----------|-------------|
| **Accuracy** | Wie oft liegt das Modell insgesamt richtig? | 64% ist gut (+14 PP über Zufall) |
| **Precision** | Wenn das Modell "Breakout!" sagt — wie oft stimmt's? | 60% = 6 von 10 Signalen richtig |
| **Recall** | Wie viele der echten Breakouts findet das Modell? | 53% = etwa die Hälfte |
| **F1-Score** | Harmonisches Mittel aus Precision und Recall | 0.56 ist solide für dieses Problem |
| **Confusion Matrix** | TN=Kein Breakout richtig, FP=Fehlalarm, FN=verpasst, TP=Breakout erkannt | Siehe oben |
| **Baseline** | Immer die häufigere Klasse raten (50.22%) | Jedes Modell MUSS das schlagen |

**Faustregel fürs Trading:** Precision ist wichtiger als Recall. Ein Fehlalarm kostet echtes Geld (Spread + Stop-Loss), ein verpasster Breakout kostet nur Opportunität.

---

## 7. Dateien für die Dokumentation

| Datei | Inhalt | Für wen |
|-------|--------|---------|
| `TRADING_STRATEGIE.md` | Datenfundierte Trading-Strategie | Prof (Methodik) |
| `ZWISCHENFAZIT.md` | Diese Datei — aktueller Stand | Alle |
| `artifacts/evaluation/mlp_evaluation.json` | MLP V2 Metriken (JSON) | Technische Doku |
| `artifacts/evaluation/lightgbm_evaluation.json` | LightGBM Metriken (JSON) | Technische Doku |
| `artifacts/evaluation/lightgbm_feature_importance.json` | Feature-Ranking | Präsentation |

---

## 8. Nächste Session

```powershell
# In C:\01_Uni\Projekte\USW\USW-TradingModel:

# 1. Sequenzielle Modelle trainieren
python scripts/06_model_training/train_sequential.py --model lstm
python scripts/06_model_training/train_sequential.py --model gru
python scripts/06_model_training/train_sequential.py --model cnn

# 2. Alle Modelle evaluieren
python scripts/06_model_training/evaluate.py
python scripts/06_model_training/evaluate_sequential.py --model lstm
python scripts/06_model_training/evaluate_sequential.py --model gru
python scripts/06_model_training/evaluate_sequential.py --model cnn

# 3. Modellvergleich + Visualisierung
python scripts/06_model_training/compare_models.py
```

