# Strategie-Entscheidung: Warum wir NICHT auf TP/SL setzen

> **Datum:** 06.07.2026, 14:30 MESZ — 1 Stunde vor Marktöffnung
> **Entscheidungsträger:** Quant-Analyse (explain-model, statistical-analysis, validate-data)
> **Status:** FINAL — Launch-Parameter festgelegt

---

## 1. Executive Summary

Nach 4 Tuning-Durchläufen mit 150+ Parameter-Kombinationen, Walk-Forward-Validierung auf 3 Zeitfenstern, und detaillierter Feature-Level-Analyse der Gewinner/Verlierer-Unterscheidung kommen wir zu einem klaren Ergebnis:

**Unser Modell ist ein exzellenter Breakout-Prädiktor. Es ist KEIN Trading-Signal-Generator.**

Der Unterschied: Ein Prädiktor sagt "mit 64% Wahrscheinlichkeit passiert X in 30 Minuten". Ein Trading-Signal-Generator sagt "kaufe JETZT zu DIESEM Preis, verkaufe zu JENEM Preis, und du machst Gewinn". Das Erste haben wir. Das Zweite nicht.

---

## 2. Die entscheidende Analyse

### 2.1 Kein Feature unterscheidet Gewinner von Verlierern

Mit TP=0.25%, SL=0.20%, 41.096 analysierte Signale:

| Feature | Gewinner Ø | Verlierer Ø | Differenz |
|---------|-----------|------------|-----------|
| `p_mlp` (Model Confidence) | 0.7202 | 0.7188 | **+0.0014** (NULL) |
| `p_lstm` | 0.6379 | 0.6367 | **+0.0012** (NULL) |
| `p_gru` | 0.6487 | 0.6466 | **+0.0022** (NULL) |
| `ensemble_score` | 0.9987 | 0.9987 | **0.0000** (IDENTISCH) |
| `return_1m` | 0.0017 | 0.0016 | **+0.0001** (NULL) |

**Selbst die höchste Model-Confidence-Kategorie (p_mlp > 0.8) hat nur 45.8% Win Rate** — exakt gleich wie die niedrigste (p_mlp 0.5-0.6: 45.1%). Die Modelle wissen nicht, welche ihrer eigenen Vorhersagen im Trading gut enden.

### 2.2 Alle Strategie-Varianten sind unprofitabel

| Strategie | Win Rate | Erwartungswert/Trade |
|-----------|----------|---------------------|
| Time-Stop-Only (30 Min halten) | 47.5% | **−0.040%** |
| TP=0.25% / SL=0.20% | 45.8% | **+0.006%** (≈ breakeven) |
| TP=0.40% / SL=0.20% | ~35% (geschätzt) | **+0.040%** (geschätzt) |

Keine Konfiguration erreicht die kritische Schwelle von >0.05% EV/Trade, die für Netto-Profit nach 0.02% Transaktionskosten nötig wäre.

### 2.3 Das fundamentale Problem: Pfadabhängigkeit

```
Das Modell sagt: P(High[t+30] > Close[t] × 1.003) = 64%

Das Trading fragt: Wird TP=0.25% VOR SL=0.20% erreicht?

Das sind ZWEI VERSCHIEDENE FRAGEN:
┌─────────────────────────────────────────────────────┐
│ Modell-Frage (Endpunkt):                            │
│   Erreicht der Preis jemals +0.3% in 30 Min?       │
│   → Antwort: Ja (64% Accuracy)                     │
│                                                     │
│ Trading-Frage (Pfad):                               │
│   Erreicht der Preis +0.25% BEVOR er -0.20% trifft? │
│   → Antwort: Münzwurf (46% Accuracy)               │
└─────────────────────────────────────────────────────┘
```

Das Modell wurde NIE darauf trainiert, Pfade vorherzusagen. Es wurde darauf trainiert, Endpunkte vorherzusagen. Wir haben 3 Wochen lang versucht, eine Endpunkt-Vorhersage als Pfad-Vorhersage zu nutzen. Das funktioniert nicht.

---

## 3. Warum das nicht früher aufgefallen ist

| Phase | Was wir dachten | Was tatsächlich passierte |
|-------|----------------|--------------------------|
| **Evaluation** | "68% Precision = 68% profitable Trades" | Die Evaluation ignoriert Pfadabhängigkeit komplett |
| **Erster Backtest** | "−374% Return = Bugs im Code" | Bugs waren real (Position Sizing, SL zu eng), aber... |
| **Zweiter Backtest** | "Mit Fixes wird's profitabel" | Win Rate stieg von 33% auf 46% — aber EV blieb negativ |
| **Feature-Analyse** | "Irgendein Feature muss Gewinner erkennen" | **Kein einziges Feature tut das** |

Erst die Feature-Level-Analyse mit `explain-model` hat die Wahrheit aufgedeckt: Das Modell ist blind für den Unterschied zwischen "Breakout erreicht TP vor SL" und "Breakout kommt, aber zu spät".

---

## 4. Die Entscheidung für heute

### Launch-Parameter (14.07.2026)

```powershell
python scripts/08_deployment/trading_loop.py --paper \
    --max_positions 10 \
    --tp_pct 0.0025 \
    --sl_pct 0.0020 \
    --position_size_pct 0.05
```

### Warum NICHT auf Time-Stop-Only gewechselt?

Obwohl die Modell-Logik ("Endpunkt vorhersagen → 30 Min halten") intuitiv besser passt, zeigen die Daten: Time-Stop-Only hat mit −0.04% EV/Trade den SCHLECHTESTEN Erwartungswert. Die Verlierer-Trades (−40% Worst Case!) zerstören jeden Vorteil.

TP/SL bei 0.25/0.20 ist der am WENIGSTEN schlechte Kompromiss:
- EV ≈ breakeven (+0.006%/Trade vor Fees)
- SL bei 0.20% begrenzt Katastrophen
- TP bei 0.25% ist unter Theta (erreichbar)

### Erwartung für heute

- **Realistisch:** Leichter Verlust (−2% bis −5%), 50–100 Trades
- **Best Case:** Breakeven bis +2%
- **Worst Case:** −10% (unwahrscheinlich mit 5% Allocation und SL-Schutz)

---

## 5. Was wir für die Präsentation gelernt haben

### Das ist KEIN Scheitern. Das ist die wertvollste Erkenntnis des Projekts.

1. **"Good ML" ≠ "Good Trading"** — Das ist die Lektion, die jeder Quant irgendwann lernt. Wir haben sie in 3 Wochen gelernt.

2. **Die Pipeline funktioniert** — Data Acquisition → Features → 5 Modelle → Ensemble → Backtest. Technisch einwandfrei. Dass die Strategie nicht profitabel ist, ist eine ERKENNTNIS, kein Fehler.

3. **Wir haben die richtigen Fragen gestellt** — Pfadabhängigkeit, Positions-Management, Transaktionskosten, Walk-Forward-Validierung. Das sind EXAKT die Themen, die ein Uni-Projekt behandeln soll.

4. **Der Prozess war korrekt** — Wir haben nicht blind optimiert, bis es "gut aussah". Wir haben Bugs gefunden, gefixt, neu getestet, und die unbequeme Wahrheit akzeptiert.

### Fürs Endfazit

> "Unser Modell erreicht 64% Accuracy auf einem hochverrauschten Problem und schlägt die Baseline um 14 Prozentpunkte. Der Backtest mit realistischen Trading-Bedingungen hat jedoch gezeigt, dass die Vorhersagegüte nicht ausreicht, um nach Transaktionskosten profitabel zu traden. Die zentrale Erkenntnis: Breakout-Prädiktion und profitables Trading sind fundamentale verschiedene Probleme, getrennt durch Pfadabhängigkeit. Diese Lücke zu schließen — etwa durch Richtungsmodelle statt Endpunkt-Modelle — ist der logische nächste Schritt."

---

## 6. Nächste Schritte

1. **[heute]** Paper-Trading-Daten sammeln (76±30 Trades erwartet)
2. **[heute Abend]** `analyze_day.py` → Live vs Backtest vergleichen
3. **[morgen]** Wenn Live-Daten Backtest bestätigen → Strategie als "funktionierender Prototyp mit dokumentierten Limitationen" präsentieren
4. **[Präsentation 14.07.]** Fokus auf Lernprozess, nicht auf Profit
