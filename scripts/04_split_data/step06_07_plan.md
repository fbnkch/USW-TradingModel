# Plan für Step 06 und 07 – Feature Selection und Model Training

**Projekt:** USW-TradingModel (Intraday Breakout-Vorhersage, NASDAQ-100)
**Datum:** Juni 2026

## Inhalt

1. Was haben wir bisher, was fehlt noch
2. Step 06 – Feature Selection (wie gehen wir die 82 Features an)
3. Step 07 – Model Training (welche Modelle, wie trainieren, wie evaluieren)
4. Warum diese Entscheidungen so getroffen wurden
5. Welche neün Scripts und Configs brauchen wir
6. Reihenfolge der Ausführung


## 1. Ausgangslage

Nach den Steps 01 bis 05 stehen folgende Daten und Artefakte bereit:

- **100 Trainings-Shards** mit insgesamt rund 10,4 Millionen Zeilen, jede etwa 104k Zeilen gross. Enthalten alle 82 Features plus das Target breakout_30m, Timestamp und Symbol. Die Features sind Rolling-Z-normalisiert (Step 03) und die Shards sind symbol-durchmischt und geshuffelt (Step 04).
- **100 Validation-Shards** mit rund 2,5 Millionen Zeilen (je 25k).
- **100 Test-Shards** mit rund 2,6 Millionen Zeilen (je 26k).
- **class_balance.json**: Die Klassen sind fast perfekt 50/50 balanciert. scale_pos_weight liegt bei etwa 1,0.
- **global_scaler.pkl**: Ein auf den Trainingsdaten gefitteter StandardScaler, der die symbol-übergreifenden Unterschiede ausgleicht.

**Konflünz-These (aus der vorherigen Analyse):**

Die Rechtfertigungsplots aus Step 02 haben gezeigt, dass Breakouts nicht durch ein einzelnes Signal entstehen, sondern durch das Zusammenspiel mehrerer Faktoren. Zum Beispiel braucht ein echter Breakout einen Volumen-Spike, einen Preis über VWAP, einen positiven EMA-Cross und der RSI darf nicht überkauft sein. Fehlt einer dieser Faktoren – etwa das Volumen – dann ist es wahrscheinlich ein Fake-Out.

Das heisst für die Modellierung: Ein Modell, das Feature-Interaktionen nicht lernen kann (wie ein sehr flacher Decision Tree), wird diese Muster nicht erkennen. Wir brauchen Architekturen, die Zusammenhänge zwischen mehreren Features erfassen können.

**Prof-Vorgaben:**

Aus dem PDF "ProjektRequirementsDozent" wissen wir, was erwartet wird. Der Prof listet explizit Decision Trees, MLP, LSTM, RNN und Attention als mögliche Architekturen. Metriken sollen Returns, Accuracy und Sharpe Ratio sein. Es soll eine Baseline geben (Markt-Performance oder Dummy-Strategie). Und der Code soll einfach und verständlich bleiben ("vibe coding").


## 2. Step 06 – Feature Selection

**Ziel:** Von den 82 Features diejenigen identifizieren, die tatsächlich Vorhersagekraft haben, und redundante oder irrelevante Features entfernen. Zielgrösse: etwa 40 bis 55 Features.

**Output:**

- selected_features.txt – die finale Feature-Liste
- feature_analysis.csv – Tabelle mit allen Metriken pro Feature
- feature_importance.png – Visualisierung der Feature-Wichtigkeit

**Neüs Script:** scripts/06_feature_selection/feature_selection.py

### Vier Stufen der Analyse

**Stufe 1 – Lineare Korrelation mit dem Target**

Wir laden ein Sample von 5 Trainings-Shards (etwa 500.000 Zeilen). Für jedes Feature berechnen wir die Pearson-Korrelation mit breakout_30m und sortieren absteigend nach dem Betrag der Korrelation.

Erwartung: Die Einzelkorrelationen werden sehr niedrig sein, vermutlich |r| < 0,05. Das ist genau die Bestätigung der Konflünz-These – kein Feature kann alleine den Breakout vorhersagen. Wir werden KEIN Feature allein wegen einer niedrigen Pearson-Korrelation entfernen. Die Stufe dient als Baseline, um die Ergebnisse der nächsten Stufen einordnen zu können.

**Stufe 2 – Mutual Information**

Wir nutzen sklearns mutual_info_classif auf demselben Sample. Mutual Information erfasst im Gegensatz zur Pearson-Korrelation auch nichtlineare Zusammenhänge zwischen Feature und Target. Die Ergebnisse werden mit dem Pearson-Ranking verglichen.

Erwartung: Die Mutual Information wird höher sein als die Pearson-Korrelation, aber immer noch moderate Werte zeigen. Features, die bei BEIDEN Metriken ganz unten rangieren, werden als schwache Kandidaten markiert.

**Stufe 3 – Permutation Importance mit einem LightGBM-Baseline-Modell**

Das ist die zentrale Stufe und das wichtigste Kriterium für die Feature-Auswahl.

Wir trainieren ein schnelles LightGBM-Modell (100 Bäume, max_depth=6) auf den 5 Trainings-Shards und evaluieren die Permutation Importance auf 2 Validation-Shards. Permutation Importance misst, wie stark die Performance des Modells sinkt, wenn man ein einzelnes Feature zufällig permutiert, während alle anderen Features unverändert bleiben.

Der grosse Vorteil: Ein Feature, das alleine schwach mit dem Target korreliert, kann trotzdem eine hohe Permutation Importance haben, weil es im Zusammenspiel mit anderen Features eine wichtige Rolle spielt. Genau das ist unsere Konflünz-These. Die Permutation Importance erfasst das indirekt, weil das LightGBM-Modell die Interaktionen während des Trainings gelernt hat.

**Stufe 4 – Kollinearitäts-Filter und finale Auswahl**

Features, die untereinander stark korrelieren (|r| > 0,95), liefern dem Modell kaum zusätzliche Information. Wir berechnen die Pearson-Korrelationsmatrix aller Features untereinander, identifizieren Feature-Gruppen mit extrem hoher Korrelation und behalten pro Gruppe nur das Feature mit der höchsten Permutation Importance aus Stufe 3.

Zusätzlich entfernen wir Features mit nahezu konstanter Varianz, erkennbar an der GlobalScaler-Statistik (scale < 0,01). Das betrifft nach unserer Analyse aus Step 05 insbesondere feature_2 (return_1m) mit scale=0,0012.

### Sampling-Strategie

Für Stufe 1 und 2 reichen 5 Trainings-Shards (500k Zeilen). Für das LightGBM-Baseline-Modell in Stufe 3 nutzen wir dieselben 5 Trainings-Shards zum Training und 2 Validation-Shards für die Permutation Importance. So vermeiden wir, alle 10,4 Millionen Zeilen verarbeiten zu müssen, und bekommen trotzdem statistisch belastbare Ergebnisse.

### Neü Config-Einträge in params.yaml

Geplant ist eine neü Sektion FEATURE_SELECTION:

- SAMPLE_SHARDS: 5 (wie viele Trainings-Shards für die Analyse)
- CORRELATION_THRESHOLD: 0.01 (Mindestkorrelation, unterhalb derer ein Feature als irrelevant markiert wird – dient nur der Orientierung)
- COLLINEARITY_THRESHOLD: 0.95 (Schwelle für Kollinearitätsgruppen)
- N_PERMUTATIONS: 5 (Wiederholungen für Permutation Importance)


## 3. Step 07 – Model Training

**Ziel:** Modelle trainieren, die breakout_30m vorhersagen, und auf dem Validation-Set evaluieren. Am Ende vergleichen wir die Modelle untereinander und mit Dummy-Baselines.

**Reihenfolge der Modelle:**

1. Decision Tree – einfachste Baseline, maximal interpretierbar.
2. LightGBM – das Hauptmodell, das die beste Performance auf tabellarischen Daten erwarten lässt.
3. MLP (PyTorch) – ein neuronales Netz zum Vergleich, weil der Prof explizit verschiedene Netzwerkarchitekturen sehen will.

Alle drei Modelle nutzen dieselbe Datenpipeline: parquetShardDataset lädt die Shards, wendet den GlobalScaler an und gibt Batches zurück.

### Neü Scripts

- scripts/07_model_training/dataset.py – parquetShardDataset (gemeinsamer DataLoader)
- scripts/07_model_training/train_decision_tree.py
- scripts/07_model_training/train_lightgbm.py
- scripts/07_model_training/train_mlp.py

### 3.1 Daten-Pipeline (dataset.py)

parquetShardDataset ist ein IterableDataset, der Shard für Shard streamt. Er liest selected_features.txt, um zu wissen, welche Spalten geladen werden müssen. Beim Laden eines Batches wendet er den GlobalScaler an (X_scaled = scaler.transform(X)). Die Batch-Grösse orientiert sich am 09-trading Projekt und liegt bei 2048.

### 3.2 Baseline: Decision Tree

Sklearns DecisionTreeClassifier mit max_depth=8 (interpretierbar, analog zum 09-trading), min_samples_leaf=500 (gegen Overfitting auf einzelnen Bars), kein class_weight (da balanciert). Kriterium: Gini.

Das Training lädt alle Trainings-Shards und sammelt die Daten, solange der RAM es zulässt (gedeckelt auf etwa 4 GB). Der Baum wird einmalig auf den gesammelten Daten trainiert.

Bei der Evaluation schaün wir auf Accuracy, F1 und die Confusion Matrix auf dem Validation-Set. Optional können wir den Baum visualisieren und die Entscheidungspfade extrahieren, um zu verstehen, welche Feature-Kombinationen zu Breakout-Vorhersagen führen.

Der Decision Tree dient als Baseline. Jedes komplexere Modell sollte besser abschneiden. Ausserdem erfüllt er die Prof-Vorgabe, Decision Trees als Architektur zu zeigen.

### 3.3 Hauptmodell: LightGBM

LightGBMs LGBMClassifier mit 500 Bäumen, max_depth=8, learning_rate=0.05, num_leaves=31. Gegen Overfitting setzen wir feature_fraction=0.8, bagging_fraction=0.8 und bagging_freq=5. Das scale_pos_weight kommt aus der class_balance.json (wird bei etwa 1,0 liegen). Early Stopping bricht nach 50 Runden ohne Verbesserung auf dem Validation-Set ab.

Der Trainingsprozess ist einfach: Alle Trainings-Shards werden einmal durchlaufen, LightGBM boostet die Bäume, und nach jedem Shard oder nach N Boosting-Runden wird auf dem Validation-Set evaluiert. Early Stopping greift abhängig vom Binary Logloss.

Nach dem Training speichern wir das Modell, plotten die Feature Importance (LightGBM hat das eingebaut) und berechnen Accuracy, F1, Precision, Recall, ROC-AUC und die Confusion Matrix.

LightGBM ist unser Hauptmodell, weil es der aktülle Stand der Technik für tabellarische Daten ist. Es lernt Feature-Interaktionen über die Baumtiefe (mehrere Splits hintereinander kombinieren verschiedene Features), skaliert gut auf 10 Millionen Zeilen und ist mit wenigen Zeilen Code nutzbar (vibe-coding-konform).

### 3.4 Vergleich: Feed-Forward Neural Network (MLP)

Ein MLP mit PyTorch, angelehnt an die Architektur aus dem 09-trading Projekt, aber für binäre Klassifikation:

- Input: n_selected_features (40–55 Features)
- Hidden Layer 1: 128 Neuronen, ReLU, Dropout(0.2)
- Hidden Layer 2: 64 Neuronen, ReLU, Dropout(0.2)
- Output: 1 Neuron, Sigmoid
- Loss: BCEWithLogitsLoss
- Optimizer: AdamW (lr=0.0005, weight_decay=1e-4)
- Batch-Grösse: 2048
- Maximal 50 Epochen, Early Stopping mit patience=10 auf Validation F1

Der Training-Loop läuft epoch-basiert: Pro Epoche werden alle Trainings-Shards einmal durchgegangen, danach wird auf allen Validation-Shards evaluiert. Early Stopping bricht ab, wenn der F1-Score auf dem Validation-Set 10 Epochen lang nicht besser wird.

Das MLP lernt Feature-Interaktionen über die Gewichtsmatrizen der Hidden Layer – jede Linearkombination aus Features kann durch die Nichtlinearität (ReLU) komplexe Muster abbilden. Wir nehmen das MLP als Vergleich auf, weil der Prof in den Folien explizit Feed-Forward-Netze, RNNs, LSTMs und Transformer auflistet. Wir zeigen damit, dass wir mehrere Architekturen evaluiert haben.

### 3.5 Baseline-Vergleiche

Um unsere Modelle einordnen zu können, berechnen wir simple Dummy-Baselines mit sklearns DummyClassifier:

- Stratified: Zufällige Vorhersagen mit gleicher Klassenverteilung wie das Training (~50/50).
- Most Freqünt: Immer die häufigste Klasse vorhersagen (0, also "kein Breakout" – würde 50,2% Accuracy liefern).
- Uniform: Rein zufällig 50/50.

Ein Markt-Benchmark (Buy-and-Hold-Return des NASDAQ-100 über den Test-Zeitraum, Sharpe Ratio) ist optional und eher für Step 09 (Deployment) relevant.

### 3.6 Metriken und Logging

Da die Klassen balanciert sind (~50/50), sind mehrere Metriken gleichermassen valide:

Primäre Metriken (Validation-basiert):

- Accuracy (50% ist Zufall, alles darüber bedeutet das Modell lernt etwas)
- F1-Score (Harmonisches Mittel aus Precision und Recall)
- ROC-AUC (Schwellwert-unabhängig, sollte deutlich über 0,5 liegen)

Sekundäre Metriken:

- Precision und Recall einzeln (wie viele vorhergesagte Breakouts sind echt, wie viele echte Breakouts wurden erkannt)
- Confusion Matrix

Logging:

- training_log.csv mit allen Metriken pro Epoche bzw. pro N Boosting-Runden
- training_metrics.png als Verlaufsdiagramm (Loss und Metriken über die Zeit)


## 4. Warum diese Entscheidungen

### LightGBM statt XGBoost

Beide sind äquivalent in der Vorhersagequalität. LightGBM ist bei grossen Datensätzen schneller, braucht weniger RAM und hat eine native kategoriale Feature-Unterstützung (auch wenn wir die hier nicht brauchen). Für unsere 10,4 Millionen Zeilen macht das einen spürbaren Unterschied in der Trainingszeit.

### Kein LSTM jetzt – warum

LSTM ist in den Prof-Folien explizit als valide Architektur genannt. Warum gehen wir es nicht direkt an?

Der Hauptgrund ist die Datenaufbereitung. Ein LSTM erwartet Seqünzen als Input, also Samples der Form (n_samples, timesteps, features). Unsere Daten liegen als einzelne Zeilen pro 1-Minuten-Bar vor. Um sie als Seqünzen zu nutzen, müssten wir z.B. immer 60 aufeinanderfolgende Bars zu einem Sample zusammenfassen. Das erfordert einen komplett anderen DataLoader und eine andere Split-Logik (Seqünzen dürfen sich nicht zwischen Train/Val/Test überlappen).

Zweitens ist die Rechenzeit ein Problem. 10,4 Millionen Zeilen als Seqünzen auf CPU zu trainieren würde Tage daürn.

Drittens die vibe-coding-Vorgabe: Ein LSTM mit Seqünce-Dataset ist deutlich mehr und komplexerer Code als ein sklearn-ähnliches LightGBM.

Wir werden LSTM als Ausblick präsentieren – als nächsten Schritt, um die Architekturliste des Profs zu vervollständigen.

### Drei Modelle statt einem

Wir wollen zeigen, dass wir verschiedene Ansätze evaluiert haben, wie es die Projektanforderungen vorsehen. Der Decision Tree zeigt maximale Interpretierbarkeit (Entscheidungspfade, Schwellwerte). LightGBM zeigt, was mit modernen Gradient-Boosting-Verfahren möglich ist. Das MLP zeigt die Evaluierung einer neuronalen Netzwerkarchitektur. Zusammen decken sie einen grossen Teil der vom Prof genannten Architekturen ab.

### Kein Random Forest

LightGBM übertrifft Random Forest in praktisch jeder Hinsicht – schnellere Trainingszeit, weniger RAM, bessere Generalisierung. Ein zusätzlicher Random Forest würde keine neün Erkenntnisse bringen.


## 5. Neü Dateien und Config-Einträge

### Neü Verzeichnisse und Scripts

```
scripts/06_feature_selection/
  __init__.py
  feature_selection.py

scripts/07_model_training/
  __init__.py
  dataset.py
  train_decision_tree.py
  train_lightgbm.py
  train_mlp.py
```

### Neü Config-Einträge in params.yaml

```
FEATURE_SELECTION:
  SAMPLE_SHARDS: 5
  CORRELATION_THRESHOLD: 0.01
  COLLINEARITY_THRESHOLD: 0.95
  N_PERMUTATIONS: 5

MODEL_TRAINING:
  TARGET: "breakout_30m"
  BATCH_SIZE: 2048
  SELECTED_FEATURES_PATH: "data/processed/selected_features.txt"
  MODELS_PATH: "data/processed/models/"

  DECISION_TREE:
    MAX_DEPTH: 8
    MIN_SAMPLES_LEAF: 500

  LIGHTGBM:
    N_ESTIMATORS: 500
    MAX_DEPTH: 8
    LEARNING_RATE: 0.05
    EARLY_STOPPING: 50

  MLP:
    HIDDEN_LAYERS: [128, 64]
    DROPOUT: 0.2
    EPOCHS: 50
    LEARNING_RATE: 0.0005
    WEIGHT_DECAY: 0.0001
    EARLY_STOPPING_PATIENCE: 10
```

### Neü Abhängigkeiten in requirements.txt

```
lightgbm
torch
```


## 6. Reihenfolge der Ausführung

**Schritt 1:** python scripts/06_feature_selection/feature_selection.py

Analysiert die Features mit den vier Stufen und schreibt selected_features.txt und die Analyse-Plots. Daürt etwa 2 bis 3 Minuten, weil nur ein Sample von 5 Shards verarbeitet wird.

**Schritt 2:** python scripts/07_model_training/train_decision_tree.py

Trainiert die Decision-Tree-Baseline auf allen Trainings-Shards. Schreibt das Modell und die Metriken. Etwa 1 bis 2 Minuten.

**Schritt 3:** python scripts/07_model_training/train_lightgbm.py

Trainiert das LightGBM-Hauptmodell mit Early Stopping. Daürt etwa 5 bis 10 Minuten (500 Bäume auf 10,4 Millionen Zeilen).

**Schritt 4:** python scripts/07_model_training/train_mlp.py

Trainiert das MLP mit bis zu 50 Epochen auf allen Trainings-Shards. Mit Early Stopping bricht es vorher ab, wenn sich der Validation F1 nicht mehr verbessert. Auf CPU etwa 10 bis 20 Minuten.

Alle vier Schritte nutzen selected_features.txt aus Schritt 1, sodass nur die relevanten Features geladen werden.
