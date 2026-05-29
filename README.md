# Intraday Breakout-Vorhersage | NASDAQ-100 
### Milestone 1: Problemdefinition & Datenbeschaffung

## Problemdefinition

### Was wollen wir vorhersagen?

Aktienpreise schwanken ständig über den Handelstag. Manchmal steigt eine Aktie innerhalb kurzer Zeit deutlich an, auch Breakout genannt. Unser Ziel ist es, für jeden Moment des Handelstages vorherzusagen, ob eine Aktie in den nächsten 30 Minuten einen solchen Preisanstieg erleben wird.

Hierbei handelt es sich also um ein binäres Klassifikationsproblem: Für jede 1-Minuten-Bar und jede Aktie im NASDAQ-100-Index beantwortet unser Modell eine einzige Frage:

> Wird der Preis dieser Aktie in den nächsten 30 Minuten um mindestens X% steigen? Ja (1) oder Nein (0)?

Dieses Signal ist direkt umsetzbar: Bei einer Vorhersage von "Ja" betrachten wir einen Kauf der Aktie (Long-Trade). Bei "Nein" unternehmen wir nichts.

### Zielvariable: `breakout_30m`

Für jede 5-Minuten-Bar zum Zeitpunkt `t` und jede NASDAQ-100-Aktie definieren wir das Target folgendermaßen:

1. Betrachtung der nächsten 30 Bars (= 30 Minuten).
2. Ermittlung des höchsten erreichten Preises in diesen 30 Bars: `High_max`.
3. Vergleich dieses Maximums mit dem aktuellen Schlusskurs `Close_t`.
4. Berechnung des potenziellen Gewinns: `(High_max - Close_t) / Close_t`
5. Wenn dieser Wert >= Schwellenwert `theta` (z.B. 0.3% bis 0.5%), wird diese Bar als 1 (Breakout) gelabelt. Andernfalls als 0 (Kein Breakout).

Der Schwellenwert `theta` ist ein Parameter, den wir später anhand der Validierungsdaten einstellen. Ein kleineres `theta` bedeutet häufigere Breakout-Ereignisse (aber kleinere Preisbewegungen), ein größeres `theta` bedeutet seltenere, aber stärkere Ausbrüche.

**Beispiel:**
- Aktueller Schlusskurs: 100,00 USD
- Höchster Preis in den nächsten 30 Minuten: 100,40 USD
- Potenzielle Rendite: (100,40 - 100,00) / 100,00 = 0,4%
- Bei theta = 0,3%: diese Kerze wird als **Breakout = 1** gelabelt
- Bei theta = 0,5%: diese Kerze wird als **Breakout = 0** gelabelt

### Warum NASDAQ-100?

Der NASDAQ-100-Index enthält rund 100 der größten nicht-finanziellen Unternehmen an der NASDAQ-Börse, darunter bekannte Namen wie Apple, Microsoft, NVIDIA, Google und Tesla. Diese Aktien werden sehr aktiv gehandelt (hohe Liquidität), dadurch kriegen wir qualitativ hochwertige Daten.


### Geplante Eingabe-Features (Überblick, Details später)

Wir planen folgende Signale aus den Rohdaten abzuleiten:

- **Normalisierter Preis und Volumen**: Z-skalierter VWAP (volumengewichteter Durchschnittspreis) und gehandeltes Volumen.
- **Exponentielle gleitende Durchschnitte (EMAs)**: Geglättete Preisdurchschnitte über verschiedene Zeitfenster (z.B. 15, 30, 60, 120 Bars = 15 Min, 30 Min, 1 Std, 2 Std).
- **Steigungen und Beschleunigungen**: Wie schnell der EMA gerade ansteigt oder fällt und ob diese Geschwindigkeit zunimmt oder abnimmt.
- **Volumen-Spikes**: Ob die aktuelle Bar ein ungewöhnlich hohes Handelsvolumen im Vergleich zum jüngsten Verlauf zeigt.
- **Intraday-Preisposition**: Wo der aktuelle Preis relativ zum bisherigen Tageshoch und Tagestief liegt.

Dadurch erhoffen wir Muster zu identifizieren wie etwa EMA-Kreuzungen in Kombination mit steigendem Volumen, die zuverlässig auf bevorstehende Breakouts hinweisen.

## Datenbeschaffung

Wir sammeln historische 1-Minuten-Bars für alle NASDAQ-100-Aktien und speichern eine Datei pro Aktie. 

### Datenquelle und API

Wir verwenden die **Alpaca Market Data API**, um historische Aktiendaten herunterzuladen.

Die API liefert:

- **Adjustierte Bars**: Open, High, Low, Close, Volume und VWAP, bereinigt um Aktiensplits und Dividendenzahlungen. Dadruch werden Preissprünge durch Unternehmensereignisse die Trainingsdaten nicht verzerrt.
- **1-Minuten-Auflösung**: Die feinste verfügbare Auflösung, die maximale Flexibilität für spätere Aggregationen bietet.
- **Nur reguläre Handelszeiten**: Wir filtern alle Bars auf die offizielle US-Börsensitzung (9:30 – 16:00 Uhr Eastern Time) mithilfe des offiziellen Alpaca-Handelskalenders.

### Symbol-Universum

Wir verwenden eine CSV-Datei mit den NASDAQ-100-Konstituenten:

```
data/nasdaq100_symbols.csv
```

Die Datei enthält mindestens eine Spalte `Symbol` mit den Ticker-Namen (z.B. AAPL, MSFT, NVDA). Für erste Tests in Milestone 1 arbeiten wir mit einem kleinen Subset von 5–10 Symbolen, um die Pipeline zu verifizieren. Das vollständige Universum wird heruntergeladen, sobald die Pipeline korrekt funktioniert.

### Script

[scripts/01_data_acquisition/bar_retriever.py](scripts/01_data_acquisition/bar_retriever.py)

Das Script führt folgende Schritte durch:

- Liest API-Zugangsdaten aus `conf/keys.yaml`
- Liest Konfiguration (Datenpfad, Zeitraum) aus `conf/params.yaml`
- Lädt die NASDAQ-100-Symbolliste aus `data/nasdaq100_symbols.csv`
- Für jedes Symbol:
  - Sendet eine Anfrage an die Alpaca API für 1-Minuten-Bars im konfigurierten Zeitraum
  - Nutzt den Alpaca-Handelskalender, um offizielle Handelszeiten pro Tag zu ermitteln
  - Filtert alle Bars außerhalb der regulären Handelszeiten heraus
  - Speichert den resultierenden DataFrame als Parquet-Datei:
    `{DATA_PATH}/Bars_1m_adj/{SYMBOL}.parquet`

Wir speichern die Daten im **Parquet-Format**, weil es deutlich schneller zu lesen und zu schreiben ist als CSV bei großen Datensätzen und Datentypen automatisch beibehält.

### Konfiguration

Alle Parameter werden in `conf/params.yaml` gespeichert, damit sie an einer einzigen Stelle geändert werden können:

```yaml
DATA_ACQUISITON:
  DATA_PATH: "data/raw"
  START_DATE: "2022-01-01"
  END_DATE: "2025-01-01"
```

### Beispiel: Rohdaten-Struktur (AAPL)

Jede Parquet-Datei enthält eine Zeile pro 1-Minuten-Bar während der regulären Handelszeiten. Nachfolgend ein Ausschnitt der ersten Zeilen für AAPL:

| timestamp           | open   | high   | low    | close  | volume | vwap   |
|---------------------|--------|--------|--------|--------|--------|--------|
| 2022-01-03 09:30:00 | 177.83 | 178.31 | 177.71 | 178.01 | 126094 | 177.97 |
| 2022-01-03 09:31:00 | 178.05 | 178.22 | 177.97 | 178.12 | 60823  | 178.10 |
| 2022-01-03 09:32:00 | 178.12 | 178.35 | 178.08 | 178.22 | 44510  | 178.21 |

*Beispielwerte; echte Daten werden über die Alpaca API bezogen.*

**Spalten-Erklärung:**

| Spalte    | Beschreibung |
|-----------|--------------|
| timestamp | Datum und Uhrzeit der Bar (US Eastern Time) |
| open      | Erster gehandelter Preis im 1-Minuten-Fenster |
| high      | Höchster gehandelter Preis in diesem Fenster |
| low       | Niedrigster gehandelter Preis in diesem Fenster |
| close     | Letzter gehandelter Preis in diesem Fenster |
| volume    | Gesamte Anzahl gehandelter Aktien |
| vwap      | Volumengewichteter Durchschnittspreis (Durchschnittspreis gewichtet nach Handelsgröße) |


## Step 2 – Data Understanding

Bevor wir Features und ein Modell aufbauen, untersuchen wir die Rohdaten, um ein Gefühl dafür zu bekommen, womit wir arbeiten und wie der Datensatz aufgebaut ist.

### Script

[scripts/02_data_understanding/plotter.py](scripts/02_data_understanding/plotter.py)

Das Script lädt die 1-Minuten-Bars für ein oder mehrere Symbole und erzeugt Plots der Intraday-Preisreihen. Für jedes Symbol wird ein Fenster von mehreren hundert Kerzen um einen gewählten Referenzpunkt angezeigt. Tagesgrenzen werden mit grünen gestrichelten Linien markiert, der Referenzbalken mit einer roten gestrichelten Linie.

### Plots (kommen dann wenn erstellt)




