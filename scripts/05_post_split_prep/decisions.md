# Entscheidungen zu Step 05 – Post-Split Data Preparation

## 1. Class-Balance-Statistik nur auf dem Trainings-Set

Wir zählen die Klassenverteilung des Labels breakout_30m über alle Trainings-Shards und speichern sie als JSON. Das brauchen wir später, um scale_pos_weight (für LightGBM) und class_weight (für sklearn) zu setzen. Wichtig: Wir rechnen das ausschliesslich auf den Trainingsdaten – Validation und Test bleiben unangetastet, sonst würden Informationen aus der Zukunft ins Training sickern.

## 2. Globaler StandardScaler als zweite Normalisierungsstufe

In Step 03 haben wir jedes Symbol einzeln per Rolling-Z-Norm normalisiert. Unterschiedliche Symbole können danach aber immer noch leicht abweichende Verteilungen haben. Der GlobalScaler gleicht das symbolübergreifend aus, indem er einen Mittelwert und eine Standardabweichung über alle Trainingsdaten hinweg lernt.

## 3. Inkrementelles Fitting mit partial_fit

Wir nutzen sklearns StandardScaler.partial_fit(). Der verarbeitet die Trainings-Shards nacheinander, ohne alle auf einmal in den RAM laden zu müssen. Der Speicherverbrauch bleibt konstant, egal wie viele Trainingsdaten wir haben. Bei unseren 10.4 Millionen Zeilen wäre ein normaler fit() kaum machbar.

## 4. Kein Rewrite der Shard-Dateien

Der fertig gefittete Scaler wird per joblib als .pkl gespeichert. Die Transformation der Daten passiert erst später im DataLoader on-the-fly während des Trainings. So sparen wir uns massiven I/O und die Shards bleiben im Originalzustand erhalten.

## 5. Reihenfolge im Training

Die Features liegen in den Shards bereits Rolling-Z-normalisiert vor (aus Step 03). Der DataLoader lädt dann den GlobalScaler und wendet ihn beim Laden eines Batches an: X_scaled = global_scaler.transform(X_z_normed). Das ist exakt die vom Prof geforderte Reihenfolge: erst within-sample preparation (Step 03), dann across-sample statistics (Step 05), angewandt auf alle drei Splits.

## 6. Anwendung auf Validation und Test

Der auf dem Trainings-Set gefittete Scaler wird unverändert für Validation und Test verwendet. Kein Refitting – das würde Information aus Val/Test in die Normalisierung einfliessen lassen und wäre Data Leakage.
