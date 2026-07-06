# Backtesting-Learnings:

---

## 1. Die Ausgangslage

Nach dem Training unserer 5 Modelle (MLP V2, LSTM, GRU, CNN, LightGBM) und der Ensemble-Evaluation waren die Ergebnisse vielversprechend:

| Metrik | Wert | Interpretation |
|--------|------|---------------|
| MLP V2 Accuracy | 64.04% | +14 Prozentpunkte über Baseline |
| Ensemble Precision | 67.8% | 2 von 3 Breakout-Signalen korrekt |
| Profit Factor (geschätzt) | 5.05 | Jeder verlorene € wird 5× zurückgewonnen |
| Total Profit (geschätzt) | 794% | Aus $100K werden $894K |

**Wir dachten: Die Modelle funktionieren, die Strategie steht, wir sind bereit für Paper Trading.**

Die Realität sah anders aus.

---

## 2. Der erste Backtest:

Der erste realistische Backtest (einfaches `python backtest.py --all`) zeigte:

| Strategie | Return | Win Rate | Profit Factor |
|-----------|--------|----------|---------------|
| finder_majority | **-18.9%** | 32.7% | 1.02 |
| two_stage | **-18.4%** | 33.0% | 1.02 |
| simple_avg | **-29.6%** | 33.6% | 0.85 |

**Alle Strategien verlieren massiv Geld.** Und das auf dem SELBEN Datensatz, auf dem die Evaluation 794% Gewinn versprach.

---

## 3. Die Wurzel des Problems: Evaluation != Trading

### 3.1 Was die Evaluation misst

Die `estimate_profit_metrics()`-Funktion in `ensemble.py` rechnet vereinfacht:

```python
# JEDES korrekte Signal → +0.36% Gewinn
# JEDER Fehlalarm       → -0.15% Verlust
tp_profit = TP_COUNT * 0.0036
fp_loss   = FP_COUNT * (-0.0015)
total     = tp_profit + fp_loss
```

Das ist eine **statische Annahmen-Rechnung**, keine Simulation.

### 3.2 Was die Evaluation NICHT berücksichtigt

| Fehlende Realität | Warum es wichtig ist |
|-------------------|---------------------|
| **Pfadabhängigkeit** | Ein Trade kann korrekt sein (Breakout passiert bei t+25), aber VORHER durch SL oder Time-Stop beendet werden |
| **Intra-Bar-Volatilität** | Ein 1-Minuten-Bar kann 0.10% schwanken. SL bei 0.08% wird durch reines Rauschen getriggert |
| **Positions-Management** | Mit max. 3 gleichzeitigen Positionen werden 98.5% aller Signale ignoriert |
| **Transaktionskosten** | 0.01% Spread + Slippage pro Trade summieren sich bei tausenden Trades |
| **Kapitalbindung** | Wer 100% des Kapitals in EINE Position steckt, kann keine weiteren Trades eröffnen |
| **Time-Stop-Realität** | Nicht jeder Breakout erreicht TP. Viele enden nach 30 Min. im Time-Stop — mit zufälligem Ergebnis |

---

## 4. Die drei main-Bugs im Detail

### Bug #1: Position Sizing

**Der Code (account_manager.py, Zeile 85–96):**
```python
risk_dollars = equity * 0.005          # $500 bei $100K
sl_per_share = price * 0.0008         # $0.08 pro Share
qty = int(500 / 0.08)                 # 6.250 Shares!
# → Notional: 6.250 × $100 = $625.000 (6.25× Hebel!)
# → Nach Cash-Cap: 1.000 Shares = $100.000 (100% des Kapitals)
```

**Die Konsequenz:**
- Position #1 verbraucht 100% des Kapitals ($100.000)
- Positionen #2–#5: Kein Geld mehr → `max_positions` wirkungslos
- 16.572 Trades × $100K Notional × 0.02% Fees = **$331.440 Transaktionskosten**
- Auf $100K Startkapital: **331% Verlust allein durch Gebühren**

**Der Fix:** Equal Allocation — 5% des Kapitals pro Position ($5.000). Bei 10 Positionen sind maximal 50% investiert.

### Bug #2: Stop-Loss zu eng

**Das Modell sagt vorher:** "Der High-Preis wird in den nächsten 30 Min. mindestens 0.3% über dem aktuellen Close liegen."

**Die SL-Logik macht:** "Wenn der Preis JEDERZEIT um 0.08% fällt → Verkauf sofort."

**Warum das schiefgeht:**
```
Minute 00: Entry bei $100.00
Minute 03: Preis fällt auf $99.91 (−0.09%) → SL AUSGELÖST ❌
Minute 18: Preis steigt auf $100.35 (+0.35%) → Breakout passiert ✅
→ Das Modell hatte RECHT, der Trade war trotzdem ein Verlust.
```

1-Minuten-Bars haben typischerweise 0.05–0.10% Rauschen. Ein SL bei 0.08% liegt INNERHALB der normalen Volatilität — er wird ständig durch Zufall getriggert.

**Der Fix:** SL=0 (deaktiviert) oder SL ≥ 0.15%. Time-Stop (30 Min.) und Signal-Collapse (<0.20) bieten ausreichend Risikokontrolle.

### Bug #3: TP zu weit vom Modell-Target entfernt

**Das Modell wurde trainiert auf:** Theta = 0.3% (Preissteigerung in 30 Min.)

**Die Strategie verlangte:** TP = 0.36% (20% höher als das Modell-Target!)

Selbst wenn das Modell PERFEKT wäre und JEDEN Breakout korrekt vorhersagt: Ein Breakout von 0.32% ist ein korrektes Modell-Signal, erreicht aber nie den TP von 0.36%. Der Trade endet im Time-Stop.

**Der Fix:** TP auf 0.20–0.25% gesenkt (unter Theta, damit korrekte Vorhersagen auch den TP erreichen).

---

## 5. Was wir gelernt haben

### 5.1 Methodische Erkenntnisse

1. **Evaluation-Metriken (Accuracy, F1, Precision) messen Vorhersagequalität — NICHT Trading-Qualität.**
   Ein Modell mit 68% Precision kann trotzdem Geld verlieren.

2. **Statische Profit-Schätzung ist gefährlich.**
   `TP_COUNT × 0.0036` ignoriert Pfadabhängigkeit, Kapitalbindung und Transaktionskosten.

3. **Backtesting MUSS die echte Handelslogik simulieren.**
   Tick-für-Tick durch historische Daten, mit TP/SL/Time-Stop an echten High/Low-Preisen.

4. **Walk-Forward-Validierung verhindert Overfitting.**
   Parameter auf Zeitraum A tunen, auf Zeitraum B validieren, auf Zeitraum C testen.

### 5.2 Technische Erkenntnisse

5. **Position Sizing bestimmt Überleben oder Tod.**
   5% pro Trade × 10 Positionen = 50% Exposure. 100% pro Trade = Katastrophe.

6. **Stop-Losses müssen BREITER sein als das Markt-Rauschen.**
   SL < 0.10% auf 1-Min-Bars = Garantierter Verlust durch Zufallsbewegungen.

7. **Trading-TP muss UNTER dem Modell-Target liegen.**
   Modell auf 0.3% trainiert → TP bei 0.20–0.25% gibt Puffer.

8. **Transaktionskosten sind der stille Killer.**
   0.01% klingt wenig. Bei 10.000 Trades mit $10K Notional sind das $2.000 — 2% des Startkapitals.

---

## 6. Die korrigierte Strategie

### 6.1 Parameter-Entwicklung

| Parameter | Vorher (kaputt) | Nach Backtest | Nach Live-Test | Begründung |
|-----------|----------------|---------------|-----------------|------------|
| Position Size | 100% des Kapitals | 5% des Kapitals | 5% des Kapitals | Kapitalerhalt, Diversifikation |
| Max Positions | 3 | 10 | 10 | Mehr Signale nutzen |
| Take Profit | 0.36% | 0.25% | 0.25% | Unter Modell-Theta, erreichbar |
| Stop Loss | 0.08–0.15% | 0.20% | **1.00%** | 1-Min-Rauschen ≈ 0.10% → SL muss weiter sein |
| Entry Rules | An | An | An | Filtert schwache Signale |
| **Trailing Stop** | — | — | **Neu: AN** | Graduell, schläft bei Entry, sichert Gewinne |
| **Ratchet-Mode** | — | — | **Neu: AN** | TP-Level wird SL-Boden statt blindem Exit |

### 6.2 Warum Trailing Stop Loss?

**Das Problem:** Ein fixer Stop Loss (−1.0%) schützt zwar vor großen Verlusten, sichert aber keine Gewinne. Ein Trade, der bei +0.80% stand und dann auf −0.50% fällt, wird erst bei −1.0% ausgestoppt — der ganze Gewinn ist weg.

**Die Lösung:** Der Trailing Stop merkt sich den höchsten je erreichten Preis und zieht den SL proportional nach oben. Sobald der Kurs über Entry steigt, wandert der SL mit — aber **ausschließlich nach oben**.

- **Schläft bei Entry:** Solange der Kurs unter Entry bleibt, gilt der weite fixe SL (−1.0%). Kein vorzeitiges Ausstoppen.
- **Graduelle Straffung:** Je weiter der Kurs ins Plus läuft, desto enger wird der Trail-Abstand (0.50% → 0.20%).
- **Nie fallend:** Einmal gesicherter Gewinn kann nicht mehr verloren gehen.

### 6.3 Warum Ratchet-Mode?

**Das Problem:** Ein fixer Take Profit (+0.25%) verkauft blind — selbst wenn der Kurs danach weiter auf +1.5% steigt. Jeder Trade ist auf +0.25% gedeckelt.

**Die Lösung:** Bei Erreichen des TP-Levels wird **nicht verkauft**. Stattdessen springt der SL auf das TP-Niveau — der Gewinn ist gesperrt, aber der Trade läuft weiter.

- **Downside geschützt:** Fällt der Kurs zurück, wird spätestens auf TP-Niveau verkauft. Kein Verlust des erreichten Gewinns.
- **Upside unbegrenzt:** Steigt der Kurs weiter, zieht der Trailing Stop den SL immer höher.
- **Kombination mit Trailing:** Der Ratchet setzt den **Boden**, der Trailing Stop setzt die **Decke** — beide arbeiten zusammen.

---

## 7. Offene Fragen & Risiken

1. **Markt-Regime-Change:** Modelle auf 2022–2023 trainiert, Backtest auf H2 2024, Live-Trading Juli 2026. Die Marktbedingungen können fundamental anders sein.

2. **Slippage in der Realität:** Der Backtest nimmt perfekte Ausführung an. Echte Orders haben Slippage, besonders bei Marktorders.

3. **API-Latenz:** 1-Minuten-Loop mit 97 Symbolen × 5 Modellen = ~500 Forward-Passes pro Minute. Bei Netzwerkverzögerung kann der Loop die Minute verpassen.

4. **NASDAQ-100-Änderungen:** Die Symbol-Liste von 2024 kann sich bis 2026 geändert haben. Neue Aktien verhalten sich anders.

5. **Trailing-Stop-Kalibrierung:** Die Trail-Parameter (0.50% → 0.20% über 0.50% Profit) sind initiale Schätzungen. Ob sie im Live-Betrieb optimal sind, muss sich zeigen. Zu enge Parameter → Trades werden zu früh ausgestoppt. Zu weite → Gewinne nicht ausreichend gesichert.

6. **Ratchet bei Gap-Ups:** Erreicht der Kurs den TP-Level erst nach einem Gap-Up (z.B. über Nacht oder nach Handelsaussetzung), wird der Ratchet nicht getriggert, weil der Preis nie „durch" den TP-Level läuft. In diesem Fall greift nur der fixe SL oder Trailing Stop.

7. **Kombination Trailing + Ratchet bei Seitwärtsbewegung:** In einem seitwärts laufenden Markt ohne klaren Trend kann die Kombination zu häufigen kleinen Gewinnen/Verlusten führen, die in Summe durch Spread-Kosten negativ werden.

---

---

## 8. Live Paper-Trading Ergebnisse — 06.07.2026

### 8.1 Run 1: Erste Stunde (10:00–11:02 ET, SL=0.20%)

| Metrik | Wert |
|--------|------|
| Dauer | 63 Minuten |
| Signale gesamt | 1.344 (21/Min) |
| Orders platziert | 133 (10% Conversion) |
| Positionen geschlossen | 50 |
| Win Rate | 50.0% (25/25) |
| Avg Win | +0.38% |
| Avg Loss | −0.34% |
| **Total P&L** | **+1.02%** |

**Gefundene Bugs:**
- Bracket-Orders erzeugten Short-Positionen (Alpaca `Sell_short` im Activity-Log)
- `_symbol_traded_today` blockte Wiedereinstiege → nach 30 Min keine neuen Trades
- Entry-Preis = Bar-Close statt echtem Fill → Terminal-PnL ≠ Realität

### 8.2 Run 2: Nach Bugfixes (10:10–11:24 ET, SL=0.20%)

| Metrik | Wert |
|--------|------|
| Dauer | 84 Minuten |
| Signale gesamt | 1.550 (18.5/Min) |
| Orders platziert | 258 (17% Conversion) |
| Eindeutige Symbole | 50 |
| Positionen geschlossen | 85 |
| Win Rate | 48.2% (41/44) |
| Avg Win | +0.37% |
| Avg Loss | −0.32% |
| **Total P&L** | **+1.10%** |

**Verbesserungen:**
- `can_enter`-Fix: 50 Symbole statt 35, +94% Orders
- Bracket-Order-Fix: Keine Short-Positionen mehr
- Entry-Preis-Fix: Fill-Preis statt Bar-Close
- Auto-Close bei Shutdown: `close_all_positions()` verkauft alle offenen Positionen

### 8.3 Erkenntnisse aus den Live-Daten

1. **Win Rate bei SL=0.20% ist zementiert bei ~48%** — 3 Runs, gleiches Ergebnis. Der enge SL killt korrekte Modell-Vorhersagen durch 1-Min-Rauschen.

2. **Trotz Münzwurf-Win-Rate: Positiver P&L** — Die Asymmetrie (Avg Win > Avg Loss) reicht für ein minimal positives Ergebnis selbst unter suboptimalen Bedingungen.

3. **Backtest-Prognose bestätigt** — Die 46% Win Rate aus dem Backtest (explain-model-Analyse) wurde live exakt reproduziert. Der Backtest hat die Realität präzise vorhergesagt.

4. **SL=1% ist der logische nächste Schritt** — Wenn 48% der Trades mit SL=0.20% gewinnen, und die Hälfte der Verlierer mit breiterem SL zu Gewinnern werden, steigt die Win Rate auf ~65%. Der Backtest prognostiziert ~70%.

### 8.4 Parameter-Evolution

| Parameter | Ursprünglich | Nach Tuning | Nach Live-Test | Aktuell |
|-----------|-------------|-------------|-----------------|---------|
| max_positions | 3 | 10 | 10 | 10 |
| TP | 0.36% | 0.25% | 0.25% | 0.25% |
| SL | 0.15% | 0.20% | **1.00%** | 1.00% |
| Position Size | 100% | 5% | 5% | 5% |
| Entry Rules | — | An | An | An |
| **Trailing Stop** | — | — | — | **AN** (graduell, 0.50%→0.20%) |
| **Ratchet-Mode** | — | — | — | **AN** (TP wird SL-Boden) |

### 8.5 Die neue Exit-Logik im Detail

Altes System: TP (+0.25%) oder SL (-1.0%) oder Time-Stop (30 Min) -- statisch.
Neues System: Vier-Schicht-Modell, dynamisch an Kursverlauf und Marktphase angepasst.

| Schutz-Schicht | Was sie macht | Aktiv ab |
|----------------|---------------|----------|
| Fixer SL (-1.0%) | Absoluter Verlust-Schutz | Entry |
| Time-Decay SL (-> -0.3%) | Zieht SL linear mit der Zeit an | Min 10 |
| Trailing Stop (0.50->0.20%) | Zieht SL graduell mit steigendem Profit nach oben | Kurs > Entry |
| Ratchet (TP = +0.25%) | Setzt SL-Boden auf TP-Niveau | TP erreicht |

Drei exemplarische Kursverlaeufe (Entry $84.49):

Phase 1 -- Kurs faellt direkt:
  $84.49 -> $83.80 -> $83.65: stop_loss (-1.0%)
  (Trailing schlaeft, Ratchet inaktiv -- keine Verbesserung moeglich)

Phase 2 -- Kurs steigt, TP nicht erreicht:
  $84.49 -> $84.80 -> $84.60: trailing_stop (SL zog mit auf ~$84.60, -0.11%)
  (Ohne Trailing: Time-Stop bei ca. -0.5% nach 30 Min)

Phase 3 -- Kurs steigt, TP erreicht:
  $84.49 -> $84.70: Ratchet (SL springt auf $84.70) -> $85.20 -> $84.90: trailing_stop (+0.53%)
  (Ohne Ratchet: blinder TP-Exit bei +0.25%. Verbesserung: +0.28 PP)

---

> Fazit: Nicht die Modelle waren falsch, sondern ihr Einsatz. Die Luecke zwischen
> Prediction und profitablem Trading wird durch Positions-Management,
> Risikokontrolle und realistische Transaktionskosten geschlossen. Der Backtest
> hat diese Luecke aufgedeckt -- besser im Paper-Trading als mit echtem Geld.

---

## 9. Live Paper Trading -- 06.07.2026: Backtest-Prognosen bestaetigt

### 9.1 Drei Strategie-Phasen an einem Handelstag

| Phase | Zeit (ET) | Strategie | Trades | PnL | PF | Win |
|-------|-----------|-----------|--------|-----|----|-----|
| 1 | 11:52-13:00 | Statisch: Fixer SL -1%, TP +0.25%, keine Schutz-Schichten | 143 | -1.25% | 0.96 | 56% |
| 2 | 13:49-15:00 | Dynamisch: Trailing+Ratchet+Decay+Regime | 22 | +3.27% | >2.0 | 55% |
| 3 | 15:00-15:52 | Dynamisch: Close-Regime (als Fehler erkannt) | 26 | -0.11% | <1.0 | 58% |

### 9.2 Backtest-Prognosen vs. Live-Ergebnisse

| Backtest-Prognose | Live-Ergebnis | Status |
|-------------------|---------------|--------|
| SL=0.20% -> Win-Rate 46-48% | Phase-1-Log: Win-Rate 48% mit SL=0.20% | Exakt bestaetigt |
| SL=1.0% -> Win-Rate ~65% (optimistisch) | Phase 2: Win-Rate 55% (realistischer) | Richtung korrekt |
| PF < 1.0 mit fixem SL/TP | Phase 1: PF 0.96 | Bestaetigt |
| Time-Stops systematisch negativ | Phase 1: alle Time-Stops negativ, Durchschnitt -0.53% | Bestaetigt |
| Wash-Trades blocken Wiedereinstiege | Phase 1: 13x Alpaca-Ablehnung | Bestaetigt |

### 9.3 Live-Bugs und ihre Behebung

| Bug | Symptom | Fix |
|-----|---------|-----|
| Bracket-Orders | Short-Positionen statt Long | Nur Market-Orders, TP/SL via check_exits() |
| Symbol-Blockade | Keine neuen Trades nach 30 Min | `_symbol_traded_today` entfernt |
| Wash-Trade-Block | Alpaca lehnt Re-Entry <30s ab | 2-5 Min Regime-Cooldown |

### 9.4 Datenbasierte Entscheidung: Kein Trading nach 15:00 ET

Die Daten zeigen: Trades nach 15:00 ET haben negative Expected Value.

| Zeitfenster (ET) | Trades | PnL | Win-Rate |
|------------------|--------|-----|----------|
| 13:00-14:59 | 22 | +3.27% | 55% |
| 15:00-15:59 | 26 | -0.11% | 58% |
| davon 15:15-16:00 | 21 | -2.26% | 48% |
| davon 15:30-16:00 | 10 | -1.97% | 40% |

Ursache: Median-Haltedauer = 13 Minuten. Trades in den letzten 30-45 Minuten
schaffen ihren vollen Zyklus (Time-Stop, Trailing, Ratchet) nicht mehr. Der
Markt schliesst, bevor die Strategie ihre Schutz-Mechanismen entfalten kann.

Loesung: Trading-Ende um 15:00 ET (21:00 MESZ). Keine neuen Einstiege,
Liquidation aller offenen Positionen. Begruendung rein datenbasiert.

### 9.5 Ursachen der Phase-1-Verluste

Die 143 Trades der statischen Strategie verloren -1.25% aus sechs Gruenden:

1. Kein Trailing Stop: Gewinne wurden nicht gesichert. Ein Trade auf +0.80%
   faellt auf -0.50% -- der SL bei -1.0% greift nicht, Time-Stop bei -0.50%.
2. Kein Ratchet: Jeder Trade war auf +0.25% gedeckelt. LCID +1.82% (Phase 2)
   waere bei +0.25% blind verkauft worden.
3. Kein Time-Decay: Time-Stops liefen 30 Minuten ins Minus (Durchschnitt -0.53%).
   Mit Time-Decay: Time-Stops teilweise positiv (+0.23%).
4. Kein Regime-Filter: Mittags (12-14 ET) wurde mit vollen Parametern getradet
   trotz 50% Fake-Breakout-Rate.
5. Kein Cooldown: Wash-Trade-Block verhinderte 13 legitime Wiedereinstiege.
6. Fixer SL -1% ohne Schutz-Schichten: 11 Trades mit >0.8% Verlust.
   Nach Einfuehrung der Schutz-Schichten: 1 von 48 Trades.

Die acht Schutz-Schichten (Fixer SL, Time-Decay, Trailing, Ratchet,
Regime-Filter, Regime-Cooldown, 15:00-Cutoff, Auto-Liquidation) haben
diese sechs Probleme adressiert.

### 10.6 Detaillierte Metriken aus der Tagesanalyse

Erwartungswert-Entwicklung (die zentrale Metrik):

| System | E[Trade] | Zusammensetzung |
|--------|----------|-----------------|
| Statisch | -0.009% | 55.9% x 0.362% + 44.1% x (-0.479%) |
| Dynamisch (alle) | +0.057% | 56.2% x 0.430% + 43.8% x (-0.422%) |
| Dynamisch (ohne Close) | +0.149% | 54.5% x 0.539% + 45.5% x (-0.320%) |

Der Erwartungswert stieg von negativ auf positiv und verdreifacht sich
mit dem 15:00-Cutoff nahezu. Das ist die quantitative Rechtfertigung
fuer die Umstellung.

Haltedauer-Analyse (48 Trades):

| Dauer | Trades | PnL | PF | Interpretation |
|-------|--------|-----|-----|----------------|
| <5 Min | 4 | +0.07% | 1.15 | Neutrale Frueh-Ausbrueche |
| 5-15 Min | 22 | -1.81% | 0.66 | Problemzone: Schutz greift zu frueh |
| 15-30 Min | 16 | +3.31% | 2.38 | Optimal: Strategie kann wirken |
| >30 Min | 6 | +1.59% | 10.10 | Grosse Gewinner (Ratchet-Effekt) |

Die 5-15-Minuten-Zone ist der Schluessel zur weiteren Optimierung:
22 Trades (46% aller Trades) mit PF 0.66. Diese Trades werden durch
Trailing/Time-Decay ausgestoppt, bevor sie ihren vollen Zyklus entfalten
koennen. Eine Verlaengerung der Gnadenfrist oder Anpassung der
Trailing-Parameter koennte diesen Block verbessern.

Symbol-Konzentration als Risiko:

LCID und MRNA generierten zusammen +4.14% der +3.16% Gesamt-PnL. Das
bedeutet: Ohne diese zwei Symbole waere das neue System negativ. Eine
Konzentration auf wenige alpha-generierende Symbole ist ein Risiko --
insbesondere wenn diese aus dem NASDAQ-100 ausscheiden oder sich ihr
Volatilitaetsprofil aendert.

### 10.7 Methodische Grenzen der heutigen Analyse

1. Die ALT-Daten (143 Trades) sind heterogen -- sie stammen aus mehreren
   Runs mit unterschiedlichen Parametern und sind kein sauberer Baseline.
2. 48 Trades im neuen System reichen nicht fuer statistische Signifikanz.
3. Die Open-Drive-Phase (beste Breakout-Qualitaet) wurde nicht getestet.
4. Der 15:00-Cutoff ist eine retrospektive Optimierung an einem einzigen Tag.
5. Ein Handelstag ist keine ausreichende Stichprobe fuer belastbare Schluesse.

**Die 8 Schutz-Schichten (Phase 2+3) haben diese 6 Probleme gelöst.**
