# Backtesting: Modell-Signale in der historischen Simulation

---

## 1. Vom Modell zur Trading-Strategie

### 1.1 Modell-Ensemble

Nach dem Training von 5 Modellen (MLP V2, LSTM, GRU, CNN, LightGBM) auf 1-Min-Bars
(2022–2023) wurde ein Ensemble gebildet. Jedes Modell gibt eine binäre Vorhersage
ab: Wird der High-Preis in den nächsten 30 Minuten mindestens 0.3% über dem
aktuellen Close liegen?

Aus den 5 Einzelsignalen wurden 5 Aggregations-Strategien abgeleitet:

| Strategie | Aggregations-Regel |
|-----------|-------------------|
| `finder_majority` | Filter-MLP + 3-von-5-Mehrheit |
| `two_stage` | MLP-Filter → zweistufige Modell-Gewichtung |
| `simple_avg` | Arithmetisches Mittel aller 5 Wahrscheinlichkeiten |
| `weighted_f1` | F1-gewichtetes Mittel |
| `stacking` | Stacking-Ensemble mit LogisticRegression |

### 1.2 Entry- und Exit-Regeln (wie im Backtest getestet)

**Entry:**
- Ein Symbol wird nur gehandelt, wenn das aggregierte Ensemble-Signal einen
  konfigurierten Schwellwert überschreitet (`entry_rules: true`).
- Maximal 3 gleichzeitige Positionen (`max_positions: 3`).

**Exit (drei Mechanismen, der erste zählt):**
- **Take Profit (TP):** +0.36% über Entry-Preis
- **Stop Loss (SL):** −0.15% unter Entry-Preis
- **Time Stop:** automatischer Exit nach 30 Minuten

**Positionsgröße:**
- Risikobasierte Berechnung: 0.5% des Equity-Risikos pro Trade, Positionsgröße
  abgeleitet aus SL-Distanz. Faktisch führte das zu ~100% Kapital pro Position.

### 1.3 Backtest-Zeitraum und Setup

| Parameter | Wert |
|-----------|------|
| Test-Zeitraum | Juli – Dezember 2024 (127 Handelstage) |
| Symbole | 97 NASDAQ-100 |
| Bar-Auflösung | 1 Minute |
| Startkapital | $100.000 |
| Transaktionskosten | 0.02% des Notionals (Spread + Slippage) |
| Benchmark | NASDAQ-100 Buy & Hold |

Modelle auf 2022–2023 trainiert, auf H2 2024 getestet — ein einfacher
Walk-Forward-Ansatz (Train/Test-Split, kein rollierendes Fenster).

---

## 2. Backtest-Ergebnisse

### 2.1 Performance aller Strategien

| Strategie | Return | CAGR | Sharpe | Max DD | Trades | Win Rate | PF |
|-----------|--------|------|--------|--------|--------|----------|-----|
| finder_majority | **−18.9%** | −34.1% | −2.51 | −26.7% | 6.224 | 32.7% | 1.02 |
| two_stage | **−18.4%** | −33.2% | −2.41 | −26.2% | 6.208 | 33.0% | 1.02 |
| simple_avg | **−29.6%** | −50.2% | −5.81 | −30.5% | 4.486 | 33.6% | 0.85 |
| weighted_f1 | **−28.2%** | −48.2% | −5.15 | −29.5% | 4.546 | 33.8% | 0.86 |
| stacking | **−27.8%** | −47.6% | −5.36 | −29.7% | 4.476 | 33.1% | 0.87 |

**Benchmark (NASDAQ-100):** +78.9% im gleichen Zeitraum (Sharpe 0.32).

**Alle Strategien verlieren massiv Geld** — auf demselben Datensatz, für den die
statische Evaluation 794% Gewinn prognostiziert hatte. Die Diskrepanz zwischen
Evaluations-Metriken und Trading-Ergebnis ist der zentrale Backtesting-Befund.

### 2.2 Wie wurden die Trades beendet?

Die Exit-Analyse für die beste Strategie (`two_stage`, 6.208 Trades):

| Exit-Typ | Anzahl | Anteil |
|----------|--------|--------|
| Take Profit (+0.36%) | 2.016 | 32.5% |
| Stop Loss (−0.15%) | 4.145 | **66.8%** |
| Time Stop | 43 | 0.7% |

**Zwei von drei Trades wurden durch den engen Stop Loss beendet.** Der SL von
0.15% liegt innerhalb der normalen 1-Min-Volatilität von ca. 0.05–0.10%.

### 2.3 Monatliche Returns — wie entwickelte sich der Markt?

| Monat | finder_majority | two_stage | NASDAQ-100 (approx.) |
|-------|----------------|-----------|----------------------|
| Jul 2024 | +4.6% | +4.6% | positiv |
| Aug 2024 | −4.0% | −4.0% | schwankend |
| Sep 2024 | +2.1% | +1.4% | positiv |
| Okt 2024 | −2.3% | +0.01% | schwankend |
| Nov 2024 | −6.6% | −4.5% | stark positiv (Post-Election Rally) |
| Dez 2024 | −9.5% | −12.0% | negativ |

Die Strategie performte in Trend-Märkten (Juli, September) positiv, verlor aber
in volatilen Seitwärtsmärkten (November–Dezember) überproportional. Der NASDAQ-100
erzielte im gleichen Zeitraum +78.9% — die Strategie nahm an der Rally nicht teil,
weil der enge SL systematisch Trades vorzeitig beendete.

### 2.4 Plots

Die folgenden Visualisierungen liegen unter `artifacts/images/06_backtesting/`:

- `backtest_equity_curve.png` — Equity-Verlauf aller Strategien vs. Benchmark
- `backtest_drawdown.png` — Drawdown-Chart (max. Drawdown: −26.7%)
- `backtest_monthly_returns.png` — Monatliche Returns im Zeitverlauf

---

## 3. Was der Backtest aufgedeckt hat

### 3.1 Position Sizing: 100% Kapital pro Trade

Der Positionsgrößen-Algorithmus leitete aus 0.5% Risiko und 0.08% SL-Distanz
eine Notional von $625.000 ab (6.25× Hebel). Gedeckelt durch das verfügbare
Kapital wurden $100.000 pro Position allokiert — das gesamte Startkapital.

Konsequenz: Position #1 verbrauchte 100% des Kapitals, die restlichen
`max_positions`-Slots liefen leer. Gleichzeitig fielen auf 6.224 Trades
$124.480 Transaktionskosten an — 124% des Startkapitals allein durch Gebühren.

### 3.2 Stop Loss im 1-Min-Rauschen

0.15% SL auf 1-Minuten-Bars bedeutet: Der Preis muss nur um 0.15% vom Entry
abweichen, um den Trade zu beenden. Die typische 1-Min-Volatilität liegt bei
0.05–0.10%. Der SL wird daher regelmäßig durch reines Markt-Rauschen
ausgelöst — nicht durch tatsächliche Fehlsignale.

Der Backtest zeigt: 66.8% aller Exits sind Stop Losses, die Avg Loss liegt bei
exakt −0.15% (dem SL-Level). Die Trades hatten keine Chance, den 30-Min-Zyklus
zu durchlaufen.

### 3.3 TP über dem Modell-Target

Das Modell wurde auf ein Theta von 0.3% trainiert (Preissteigerung in 30 Min.).
Der TP war auf 0.36% gesetzt — 20% über dem Modell-Target. Selbst korrekte
Modell-Vorhersagen (Breakout erreicht +0.32%) verfehlen den TP und enden im
Time Stop.

---

## 4. Methodische Erkenntnisse aus dem Backtesting

### 4.1 Evaluation-Metriken ≠ Trading-Ergebnis

Die statische Profit-Schätzung (`TP_COUNT × 0.0036`) ignorierte:

| Faktor | Auswirkung im Backtest |
|--------|----------------------|
| **Pfadabhängigkeit** | Trade korrekt bei t+25, aber vorher durch SL beendet |
| **Intra-Bar-Volatilität** | SL wird durch 1-Min-Rauschen getriggert |
| **Kapitalbindung** | 100% in einer Position → keine Diversifikation |
| **Transaktionskosten** | $124K Gebühren auf $100K Startkapital |
| **Time-Stop-Realität** | Korrekte Signale enden im Time-Stop weil TP zu hoch |

### 4.2 Kern-Learnings

1. **Ein Backtest muss die echte Handelslogik simulieren.** Tick-für-Tick durch
   historische Bars, mit TP/SL/Time-Stop an echten High/Low-Preisen. Statische
   Formeln reichen nicht.

2. **Position Sizing bestimmt Überleben.** 100% Kapital pro Trade + enger SL =
   garantierter Kapitalverzehr durch Gebühren. Die Positionsgröße muss zur
   SL-Weite und zum Kapital passen.

3. **Stop Loss muss breiter sein als das Markt-Rauschen.** Auf 1-Min-Bars
   bedeutet das: SL ≥ 0.20%, sonst wird er zum Zufallsgenerator.

4. **TP muss unter dem Modell-Theta liegen.** Sonst erreichen selbst perfekte
   Vorhersagen nie den TP. Bei Theta = 0.3% sollte TP ≤ 0.25% sein.

5. **Transaktionskosten sind der stille Killer.** Bei 6.000+ Trades auf 127 Tagen
   summieren sich 0.02% Spread auf fatale Beträge.

---

## 5. Verteilung der Trading-Punkte über die Zeit

Die Signal-Rate lag bei 10–14% der Bars (je Strategie). Das bedeutet: In jeder
Minute gaben 10–14% der 97 Symbole ein Entry-Signal — etwa 10–14 gleichzeitige
Kaufsignale. Mit `max_positions=3` wurden über 70% aller Signale ignoriert.

Die Trades-per-Day lagen zwischen 35 (simple_avg) und 49 (finder_majority).
Die durchschnittliche Haltedauer betrug 4.6–10.5 Minuten — weit unter den
möglichen 30 Minuten, was den dominanten Einfluss des engen SL bestätigt.

---

## 6. Fazit und nächste Schritte aus dem Backtesting

Der Backtest mit statischen historischen Daten (H2 2024) hat drei kritische
Parameter-Probleme identifiziert, bevor ein einziger Live-Trade ausgeführt wurde:

- **Position Sizing** muss von 100% auf einen Kapital-erhaltenden Anteil
  (z.B. 5% pro Position) gesenkt werden.
- **Stop Loss** muss von 0.15% auf mindestens 0.20% verbreitert werden, um
  außerhalb des 1-Min-Rauschens zu liegen.
- **Take Profit** muss von 0.36% auf maximal 0.25% gesenkt werden, um unter
  dem Modell-Theta von 0.30% zu liegen.

Die Modell-Signale haben prädiktiven Wert (Precision 67.8% in der Evaluation),
aber die Execution-Parameter müssen durch einen Walk-Forward-Parameter-Sweep
auf historischen Daten optimiert werden, bevor die Strategie für Paper Trading
freigegeben werden kann.

Die Lücke zwischen Prediction-Qualität und Trading-Ergebnis wurde durch den
Backtest aufgedeckt — nicht durch Intuition, nicht durch Live-Trading, sondern
durch systematische Simulation an statischen historischen Daten.
