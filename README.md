# Intraday Breakout-Vorhersage | NASDAQ-100

## Problemdefinition:

**Target**  
Wir wollen für jede 1-Minuten-Bar und jede NASDAQ-100-Aktie vorhersagen, ob der Preis in den nächsten 30 Minuten um mindestens X% ansteigt.  
Wir haben also ein binäres Klassifikationsproblem:

> Steigt der Preis in den nächsten 30 Minuten um mindestens X%? Ja (1) oder Nein (0)

**Definition der Zielvariable `breakout_30m`**  
Für jede 1-Minuten-Bar zum Zeitpunkt `t`:

1. Betrachte die nächsten 30 Bars (30 Minuten)
2. Bestimme das Maximum der High-Preise in diesem Fenster (`High_max`)
3. Berechne die potenzielle Rendite: `(High_max - Close_t) / Close_t`
4. Wenn Rendite >= `theta` (z.B. 0.3% bis 0.5%), dann Label = 1, sonst 0


## Data Acquisition

Wir laden historische 1-Minuten-Bars für alle NASDAQ-100-Symbole und speichern eine Datei pro Aktie.

**Script**  
[scripts/01_data_acquisition/bar_retriever.py](scripts/01_data_acquisition/bar_retriever.py)

**Datenquelle**  
Alpaca Market Data API

**Was genau passiert?**  
- Lade API-Keys und Parameter aus YAML  
- Lese die NASDAQ-100-Symbolliste  
- Lade 1-Minuten-Bars (adjusted)  
- Filtere strikt auf Regular Trading Hours (RTH) via Alpaca-Handelskalender  
- Speichere je Symbol eine Parquet-Datei unter:  
  `data/raw/Bars_1m_adj/{SYMBOL}.parquet`

## Step 2 – Data Understanding

Wir analysieren die Rohdaten, um Struktur, Verteilungen, Tagesmuster und Korrelationen zu verstehen.

**Hauptscript**  
[scripts/02_data_understanding/data_understanding.py](scripts/02_data_understanding/data_understanding.py)

### 1) Basis-Checks (Qualität und Struktur)
Wir prüfen je Symbol:
- Anzahl Zeilen  
- Zeitspanne  
- Missing Values in close/volume  
- Zeitzone des Index  

Diese Checks stellen sicher, dass alle Symbole vergleichbar und sauber gespeichert sind.

### 2) Deskriptive Statistiken
Für Beispiel-Symbole (AAPL, MSFT, NVDA):
- Mittelwert, Median, Std, Min, Max für close und volume  
- Mittelwert und Std der 1-Minuten-Returns  

Damit zeigen wir, dass typische Preisbewegungen sehr klein sind und die Verteilung stark konzentriert ist.

### 3) Intraday-Preisplots (AAPL, MSFT)
plotted 1-Minuten-Close-Preise über mehrere Handelstage:
- Tagesgrenzen als grün gestrichelte Linien  
- Referenzbalken (Mitte) als rote Linie  

**Plots:**  
- `artifacts/images/AAPL_intraday_close_sample.png`  
- `artifacts/images/MSFT_intraday_close_sample.png`

Diese Plots machen sichtbar, wie stark Preise intraday schwanken und wie Tagesstrukturen aussehen.

### 4) Verteilungen der Returns (AAPL)
zeigt die Verteilung der 1-Minuten-Returns:
- Full Range  
- Zoom (-1% bis +1%)  

**Plots:**  
- `artifacts/images/AAPL_returns_hist.png`  
- `artifacts/images/AAPL_returns_hist_zoom.png`

Die Zoom-Version zeigt klar die Glockenform und macht deutlich, dass grosse Sprünge selten sind.

### 5) Verteilung des Volumens (AAPL)
zeigt die Volumenverteilung:
- Linear  
- Log-Skala (log10)  

**Plots:**  
- `artifacts/images/AAPL_volume_hist.png`  
- `artifacts/images/AAPL_volume_hist_log.png`

Die Log-Version macht den Long-Tail sichtbar (wenige Minuten mit extrem hohem Volumen).

### 6) Intraday-Muster über den Tag (AAPL)
berechnen von Durchschnittswerten pro Uhrzeit:
- Durchschnittliches Volumen pro Uhrzeit  
- Durchschnittliche Intraday-Volatilitaet pro Uhrzeit  

**Plots:**  
- `artifacts/images/AAPL_avg_volume_by_time.png`  
- `artifacts/images/AAPL_avg_volatility_by_time.png`

Diese Plots zeigen typischerweise Aktivitätsspitzen am Morgen und kurz vor Handelsschluss.

### 7) Korrelationen zwischen Aktien
Wir berechnen die Korrelation der 1-Minuten-Returns für ein kleines NASDAQ-100-Subset (z.B. AAPL, MSFT, NVDA, AMZN, META) und stellen sie als Heatmap dar.

**Plot:**  
- `artifacts/images/corr_returns_heatmap.png`

Ziel: Demonstrieren, dass viele Aktien gemeinsam laufen, was für spätere Modellvalidierung wichtig ist (Overfitting auf Marktbewegung vermeiden).


## Fazit Data Understanding

Durch die Plots und Statistiken zeigen wir:

- Intraday-Daten sind sehr fein, aber Bewegungen pro Minute sind klein und stark konzentriert  
- Volumen und Volatilität haben klare Tagesmuster  
- Extremwerte sind selten (Long-Tail bei Volumen, seltene grosse Returns)  
- Aktien bewegen sich nicht unabhängig voneinander (hohe Korrelation)