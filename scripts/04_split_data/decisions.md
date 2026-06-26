# Entscheidungen zu Step 04 – Split Data

## 1. Shuffling mit pandas+numpy statt DuckDB

Keine externe Datenbank-Abhängigkeit. pandas.sample(frac=1) liefert eine vollständige Permutation. numpy.arange(n_rows) % n_shards wird für die Round-Robin-Zuweisung der Zeilen auf die Shards genutzt. Der ganze Ablauf ist durch feste random_state-Werte reproduzierbar.

## 2. Anzahl Shards: n_shards = 100

Das 09-trading Projekt verwendet 1 Shard pro Symbol, bei rund 500 Symbolen also etwa 500 Shards. Wir haben in etwa 100 NASDAQ-100 Symbole, daher 100 Shards. Im Training enthält jeder Shard dann um die 100k bis 200k Zeilen – gross genug für stabile Batches, klein genug um problemlos in den RAM zu passen.

## 3. Dreistufiger Shuffle

Erst ein In-File-Shuffle pro Symbol-Datei über pandas.sample(frac=1). Damit zerstören wir innerhalb eines Splits die zeitliche Reihenfolge. Danach verteilen wir jede Datei per Round-Robin gleichmässig auf die 100 Shards, sodass jedes Symbol mit etwa gleichem Anteil in jedem Shard landet. Zuletzt noch ein finaler Shuffle jedes einzelnen Shards, der die verschiedenen Symbole innerhalb des Shards durchmischt (pro Shard ein eigener Seed, also 42+k für Shard k).

## 4. Reproduzierbarkeit

Der In-File-Shuffle nutzt random_state=42. Der finale Shuffle pro Shard nutzt 42+k. Bei einem erneuten Durchlauf mit denselben Eingangsdaten entstehen exakt dieselben Shard-Dateien.

## 5. Keine zeitliche Ordnung nötig

In Step 03 wurden alle Features so berechnet, dass jede Zeile für sich alle nötigen Informationen aus der Vergangenheit enthält (Rolling-Fenster, Slopes, Lags). Keine Zeile ist von einer benachbarten Zeile abhängig. Shuffeln zerstört daher keine relevante Information.

## 6. Subset-Option für Testläufe

In der params.yaml kann SUBSET_SYMBOLS gesetzt werden, z.B. auf 5. Dann werden nur die ersten N Symbole verarbeitet, was für schnelle Tests während der Entwicklung nützlich ist, ohne jedes Mal alle 10 Millionen Trainingszeilen verarbeiten zu müssen.
