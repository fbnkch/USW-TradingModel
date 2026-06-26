# Bericht zu Step 04 und 05 – Split Data und Post-Split Preparation

**Projekt:** USW-TradingModel (Intraday Breakout-Vorhersage, NASDAQ-100)
**Datum:** Juni 2026

## Inhalt

1. Was wurde in Step 04 und 05 gemacht
2. Wie sieht die Datenstruktur jetzt aus
3. Konkrete Ergebnisse und Kennzahlen
4. Was bedeutet das für die nächsten Schritte (06 und 07)


## 1. Was wurde gemacht

### Step 04 – Shuffeln und globales Sharding (shuffle_shard.py)

Ausgangspunkt waren die 300 pro-Symbol parquet-Dateien aus Step 03, die im Verzeichnis data/processed/pre_split/ liegen. Jeweils 100 Dateien für Train, Validation und Test, eine pro NASDAQ-100 Symbol. Jede Datei enthält die fertigen Features und das Target, aber noch in zeitlich sortierter Reihenfolge und nach Symbolen getrennt – ungeeignet für Batch-Training über mehrere Symbole hinweg.

Pro Split (train, validation, test) läuft folgender Prozess ab:

**a) In-File-Shuffle**

Jede einzelne Symbol-Datei wird geladen und mit pandas.sample(frac=1, random_state=42) vollständig durchmischt. Die zeitliche Reihenfolge innerhalb des Splits wird damit aufgehoben. Das ist legitim, weil jedes Feature in Step 03 bereits ausschliesslich aus Vergangenheitsdaten berechnet wurde (Rolling-Fenster, Slopes, Lags). Keine Zeile benötigt ihre Nachbarzeile, um interpretierbar zu sein.

**b) Round-Robin-Verteilung auf Shards**

Die durchmischten Zeilen werden gleichmässig auf 100 Shards verteilt, per shard_id = Zeilennummer % 100. Dadurch bekommt jeder Shard etwa 1% der Zeilen aus jeder Symbol-Datei. Nach diesem Schritt hat jeder Shard also einen kleinen, aber repräsentativen Ausschnitt jedes Symbols.

**c) Finaler Shuffle pro Shard**

Sobald alle Symbole verarbeitet sind, werden die gesammelten Zeilen eines Shards noch einmal final durchmischt (sample(frac=1, random_state=42+k) für Shard k). Das stellt sicher, dass innerhalb eines Shards die verschiedenen Symbole gut gemischt sind.

**d) Speichern**

Die fertigen Shards werden als {split}_shard_{k}.parquet unter data/processed/shuffled/ abgelegt. Insgesamt entstehen 300 Dateien (100 pro Split).

### Step 05a – Klassenverteilung des Targets berechnen (class_balance.py)

Wir zählen, wie häufig das Label breakout_30m in den Trainings-Shards den Wert 0 oder 1 hat. Insgesamt laufen dafür alle 100 Trainings-Shards durch (rund 10.4 Millionen Zeilen), es wird aber nur die Target-Spalte geladen – der Vorgang daürt wenige Sekunden.

Aus den Zählwerten berechnen wir scale_pos_weight (negativ/positiv, für LightGBM), class_weight_0 und class_weight_1 (für sklearns class_weight='balanced'). Die Werte landen in data/processed/class_balance.json.

Die Berechnung erfolgt ausschliesslich auf den Trainings-Shards. Validation- und Test-Daten werden nicht angefasst – sonst würden Informationen aus der Zukunft in die Gewichtung einfliessen.

### Step 05b – Globalen StandardScaler fitten (global_scaler.py)

Der GlobalScaler ist die zweite Normalisierungsstufe nach der Rolling-Z-Norm aus Step 03. Während die Rolling-Z-Norm jedes Symbol für sich normalisiert, gleicht der GlobalScaler die Verteilungen über alle Symbole hinweg an. Das entspricht der Vorgabe aus den Projektanforderungen: "Across-sample statistics are allowed, computed based on training set and applied to validation and test set."

Technisch nutzen wir sklearn.preprocessing.StandardScaler mit partial_fit(). Der Scaler iteriert über alle 100 Trainings-Shards, lädt jeweils nur die 82 Feature-Spalten (nicht Target, nicht Metadaten) und aktualisiert inkrementell die laufenden Mittelwerte und Standardabweichungen. Dadurch bleibt der RAM-Verbrauch konstant, egal wie viele Trainingsdaten wir haben.

Der fertige Scaler wird per joblib als data/processed/global_scaler.pkl gespeichert. Die Shard-Dateien selbst werden nicht überschrieben – die Transformation erfolgt später im DataLoader on-the-fly.


## 2. Datenstruktur nach der Ausführung

```
data/processed/
  |
  +-- pre_split/                     (aus Step 03, unverändert)
  |     features.txt                 (82 Feature-Namen, eine pro Zeile)
  |     AAPL_train.parquet           (100 Dateien)
  |     AAPL_validation.parquet      (100 Dateien)
  |     AAPL_test.parquet            (100 Dateien)
  |     ...
  |
  +-- shuffled/                      (NEU aus Step 04)
  |     train_shard_0.parquet        (100 Dateien, je etwa 104k Zeilen)
  |     train_shard_1.parquet
  |     ...
  |     train_shard_99.parquet
  |     validation_shard_0.parquet   (100 Dateien, je etwa 25k Zeilen)
  |     ...
  |     test_shard_0.parquet         (100 Dateien, je etwa 26k Zeilen)
  |     ...
  |
  +-- class_balance.json             (NEU aus Step 05a)
  +-- global_scaler.pkl              (NEU aus Step 05b)
```


## 3. Konkrete Ergebnisse und Kennzahlen

### 3.1 Grösse der Shards

| Split | Symbole | Zeilen gesamt | Shards | Zeilen pro Shard (Durchschnitt) |
|-------|---------|---------------|--------|----------------------------------|
| Train | 100 | 10.394.874 | 100 | 103.949 |
| Validation | 100 | 2.525.772 | 100 | 25.258 |
| Test | 100 | 2.603.398 | 100 | 26.034 |

Jeder Trainings-Shard enthält etwa 94 Spalten: die 82 Features, die OHLCV-Bars, das Target breakout_30m, den Timestamp und das Symbol-Kürzel.

### 3.2 Symbol-Durchmischung in den Shards

Stichprobe aus train_shard_0.parquet – Verteilung der Symbole:

| Symbol | Zeilen im Shard |
|--------|-----------------|
| AAPL | 1.772 |
| AMD | 1.752 |
| INTC | 1.747 |
| NVDA | 1.721 |
| TSLA | 1.719 |
| MSFT | 1.689 |
| CSCO | 1.687 |
| META | 1.672 |
| MU | 1.656 |
| PYPL | 1.625 |
| ... (alle 100 Symbole) | jeweils etwa 1.600–1.800 |

Alle 100 NASDAQ-100 Symbole sind in jedem Shard vertreten, mit relativ gleichmässiger Verteilung. Die Unterschiede in der Zeilenzahl kommen daher, dass manche Symbole leicht mehr oder weniger Daten im Trainingszeitraum haben (z.B. weil sie erst später in den NASDAQ-100 aufgenommen wurden).

### 3.3 Klassenverteilung des Targets

Ergebnis aus class_balance.json:

- Trainings-Bars gesamt: **10.394.874**
- Label 1 (Breakout): **5.174.515** (49,78 %)
- Label 0 (kein Breakout): **5.220.359** (50,22 %)
- scale_pos_weight: 1,0089
- class_weight_0 / class_weight_1: 0,9956 / 1,0044

Die Klassen sind fast perfekt 50/50 balanciert. Das hatten wir so nicht erwartet – mit einem theta von 0,3% und einem 30-Minuten-Horizont tritt etwa in der Hälfte aller 1-Minuten-Bars irgendwann in den nächsten 30 Minuten ein Anstieg von mindestens 0,3% auf. Das bedeutet konkret:

- Keine Class-Imbalance-Problematik, die wir mit speziellen Techniken behandeln müssen.
- Accuracy ist als Metrik valide, weil 50% die Zufallsrate ist.
- scale_pos_weight und class_weight liegen bei etwa 1,0 – kaum nötig, aber schadet auch nicht.

### 3.4 Analyse des GlobalScalers

Wir haben uns die vom Scaler gelernten Statistiken genaür angeschaut. Interessant ist vor allem, wie stark die einzelnen Features vom Idealwert mean=0, scale=1 abweichen – denn sie wurden ja bereits in Step 03 per Rolling-Z-Norm normalisiert. Ware die Rolling-Z-Norm perfekt, müsste der GlobalScaler kaum etwas tun.

**Mittelwerte (82 Features):**

- Median der Absolutwerte: 0,0018 – die meisten Features liegen also sehr nah an 0.
- Maximaler Absolutwert: 0,2179 bei feature_9 (minutes_since_open). Das Feature ist offenbar nicht nullzentriert – plausibel, denn die Minuten seit Marktöffnung sind immer positiv und folgen einem Sägezahnmuster über den Tag.
- Nur 4 von 82 Features haben einen Mittelwert, der betragsmässig über 0,1 liegt.

**Standardabweichungen (82 Features):**

- Median der Abweichung von 1,0: 0,257 – moderate, aber merkbare Abweichungen.
- feature_2 (return_1m): scale = 0,0012 – die Varianz ist extrem gering. Das Feature ist fast konstant nahe Null. Ein starker Kandidat um es in Step 06 zu entfernen.
- feature_15 (EMA_120): scale = 1,574 – hier ist die Streuung deutlich höher als bei den anderen Features.

**Fazit:** Der GlobalScaler leistet messbare Arbeit. Die Rolling-Z-Norm hat die Features pro Symbol gut, aber nicht perfekt symbolübergreifend normalisiert. Einige Features zeigen deutliche Abweichungen von der idealen Standardnormalverteilung. Der Scaler gleicht das aus und macht die Features über alle Symbole hinweg vergleichbar.


## 4. Erkenntnisse für die nächsten Schritte

### Keine Class-Imbalance

Da die Klassen etwa 50/50 verteilt sind, brauchen wir uns um class_weight, Oversampling oder spezielle Loss-Funktionen keine Gedanken zu machen. BCE bzw. Cross-Entropy funktionieren direkt. Accuracy, F1 und ROC-AUC sind alle gleichermassen aussagekräftig.

### Feature-Kandidat für Entfernung: return_1m

Der GlobalScaler zeigt, dass return_1m (1-Minuten-Return) eine verschwindend geringe Varianz hat (scale=0,0012). Das Feature enthält kaum Information und sollte in der Feature-Selection (Step 06) als erstes auf den Prüfstand.

### Batch-Training ist jetzt möglich

Jeder Shard enthält alle Symbole gut gemischt. Ein DataLoader kann einfach Shard für Shard durchgehen und Batches daraus ziehen, ohne dass die Symbole nachträglich gruppiert werden müssen.

### GlobalScaler muss im DataLoader sitzen

Die Transformation mit dem GlobalScaler sollte on-the-fly im DataLoader passieren. Das ist I/O-effizienter, als alle Shards vorab zu transformieren. Ausserdem erfüllt es die Prof-Vorgabe: Statistiken auf dem Trainings-Set berechnen, auf alle Splits anwenden.

### Sampling für Feature-Selection reicht aus

Für Korrelationen und Mutual Information in Step 06 müssen wir nicht alle 10.4 Millionen Trainingszeilen durchrechnen. 5 Shards mit je etwa 100k Zeilen (insgesamt 500k) liefern statistisch belastbare Ergebnisse in wenigen Sekunden statt Minuten.
