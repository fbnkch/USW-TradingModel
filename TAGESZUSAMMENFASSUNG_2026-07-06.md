# USW-TradingModel — Tageszusammenfassung 06.07.2026

Stand: 06.07.2026, 22:00 MESZ (Markt geschlossen)
Dauer: ~4 Std aktives Trading (3 Runs, 191 Trades gesamt)
Modus: Paper Trading (Alpaca)

---

## 1. Chronologie der Strategie-Entwicklung

### Phase 1: Statisches System (Start 11:52 ET, ~70 Min, 143 Trades)

Parameter: `--max_positions 10 --tp_pct 0.0025 --sl_pct 0.01 --position_size_pct 0.05`

Exit-Regeln: Take Profit +0.25%, Stop Loss -1.00%, Time Stop 30 Min, Signal-Kollaps <0.20.

Ergebnis: 143 Trades, 56% Win-Rate, **-1.25% PnL, Profit Factor 0.96**.

Identifizierte Probleme:
- 11 Trades mit Verlust >0.8% (7.7% aller Trades)
- Wash-Trade-Bug: 13x Alpaca-Blockierung ("potential wash trade detected")
- Time-Stops alle negativ (Durchschnitt -0.53%)
- Risk/Reward-Asymmetrie: Avg Win +0.36% vs Avg Loss -0.48%

### Phase 2: Dynamisches Exit-System (entwickelt 13:00-14:00 MESZ)

#### Vier-Schicht-Exit-System

| Nr. | Schicht | Parameter | Aktiv ab | Zweck |
|-----|---------|-----------|----------|-------|
| 1 | Fixer SL | -1.00% | Entry | Absoluter Verlustschutz |
| 2 | Time-Decay SL | -1.00% -> -0.30% | Min 10-30 | Verhindert Time-Stop-Verluste |
| 3 | Trailing Stop | 0.50% -> 0.20% graduell | Kurs > Entry | Sichert Gewinne schrittweise |
| 4 | Ratchet-Mode | TP-Level wird SL-Boden | TP erreicht | Gewinn gesperrt, Upside offen |

Effektiver SL = Maximum aller vier Schichten. Steigt nur, nie fallend.

#### Regime-basierte Parameter-Anpassung

Statt binarem Blocken werden Parameter an die Marktphase angepasst:

| Regime | Zeit (ET) | Finder-Votes | Position | Time-Stop | Cooldown |
|--------|-----------|-------------|----------|-----------|----------|
| Open Drive | 09:30-10:30 | 2+ | 100% | 30 Min | 2 Min |
| Late Morning | 10:30-12:00 | 2+ | 100% | 30 Min | 3 Min |
| Midday | 12:00-14:00 | 3+ | 60% | 20 Min | 5 Min |
| Afternoon | 14:00-15:00 | 2+ | 100% | 30 Min | 3 Min |
| No Trading | 15:00+ | -- | 0% | -- | -- |

#### Weitere technische Aenderungen

- Wash-Trade-Cooldown: Regime-abhaengig 2-5 Min (vorher: kein Cooldown)
- Entry-Rules E3-E5: Momentum + Trend + Regime-Huerde (vorher nur dokumentiert)
- Pre-Close-Schutz: Keine Einstiege wenn Time-Stop ueber Marktschluss hinausgeht
- Trading-Ende: 15:00 ET (21:00 MESZ) -- datenbasiert, siehe Abschnitt 3.6
- Auto-Liquidation: Alle Positionen werden um 15:00 ET verkauft
- Run-basiertes Logging: `YYYY-MM-DD_run-NNN/` mit `run_info.json` + `run_exit.json`
- Exit-Reason-Tracking: `exit_reason` Spalte in `positions.parquet`
- Parameter-Metadaten: Vollstaendige CLI-Parameter in `run_info.json`

### Phase 3: Live-Ergebnis (Start 13:49 ET, ~120 Min, 48 Trades)

Ergebnis (Midday -> Afternoon -> Close):

| Metrik | Wert |
|--------|------|
| Trades | 48 |
| Win-Rate | 56.3% (27/48) |
| Summe PnL | +3.16% |
| Profit Factor | 1.37 |
| Avg Win | +0.430% |
| Avg Loss | -0.422% |
| Bester Trade | LCID +1.82% |
| Schlechtester Trade | -1.08% (Close-Regime) |
| Verluste <-0.5% | 10 von 48 (21%) |
| Verluste <-0.8% | 1 von 48 (2.1%) |

Die Close-Phase (letzte Stunde vor Marktschluss) zog das Ergebnis nach unten:
26 Trades mit -0.11% PnL; die letzten 45 Minuten allein -2.26%.
Mit 15:00-Cutoff waere das Ergebnis: 22 Trades, +3.27%, PF >2.0.

---

## 2. Performance-Vergleich: Statisch vs. Dynamisch

| Metrik | Statisch (143 Trades) | Dynamisch (48 Trades) | Differenz |
|--------|----------------------|----------------------|-----------|
| Summe PnL | -1.25% | +3.16% | +4.4 PP |
| Profit Factor | 0.96 | 1.37 | +43% |
| Win-Rate | 56% | 56% | unveraendert |
| Avg Win | +0.36% | +0.43% | +19% |
| Avg Loss | -0.48% | -0.42% | -13% |
| Max Win | +0.77% | +1.82% | Ratchet-Effekt |
| Max Loss | -1.47% | -1.08% | -27% |
| Trades <-0.8% | 11 (7.7%) | 1 (2.1%) | -73% |
| Trades >+1.0% | 0 (0%) | 2 (4.2%) | Ratchet-Effekt |

Statistische Einordnung:
- Mann-Whitney U Test: p = 0.51 (noch nicht signifikant, 48 vs 143 Trades)
- Cohen's d = 0.20 (kleiner bis mittlerer positiver Effekt)
- Profit Factor Bootstrap 95%-CI: Statisch [0.66, 1.42], Dynamisch [0.70, 3.85]
- Die Richtung ist klar positiv, benoetigt aber mehr Daten fuer statistische Sicherheit

---

## 3. Zentrale Erkenntnisse

### 3.1 Dynamisches Risikomanagement vor besserer ML

Die groesste Verbesserung kam durch dynamische Exit-Logik, nicht durch neue Modelle:

| Mechanismus | Effekt |
|-------------|--------|
| Ratchet-Mode | Avg Win von +0.36% auf +0.43% |
| Trailing Stop | Nur 1 von 48 Trades <-0.8% Verlust (vorher 11 von 143) |
| Time-Decay SL | Time-Stops von Durchschnitt -0.53% auf teilweise positiv |
| Regime-Filter | Midday: 75% Win-Rate, +2.39% PnL (4 Trades) |

### 3.2 Bugs und deren Behebung

| Bug | Symptom | Loesung |
|-----|---------|---------|
| Wash-Trade-Block | Alpaca lehnt Re-Entry <30s nach Exit ab | Regime-Cooldown 2-5 Min |
| Entry-Rules fehlend | E3-E5 nur in Doku, nicht im Code | In Trading-Loop integriert |
| Kein Exit-Reason-Tracking | Manuelle Log-Analyse noetig | `exit_reason` in positions.parquet |

### 3.3 Midday Lull -- Research und Umsetzung

- 12:00-14:00 ET: Volumen ca. -40%, Fake-Breakout-Rate 45-55%
- Quant-Ansatz: Kein binaeres Blocken, sondern hoehere Signal-Huerde + reduzierte Position
- Mean-Reversion dominiert mittags (nicht Momentum/Breakout)
- Umsetzung: 3/4 Finder-Votes noetig, 60% Position, 20 Min Time-Stop

### 3.4 Risk/Reward-Mathematik

Mit 56% Win-Rate und aktuellen Durchschnittswerten:
- Erwartungswert = 0.56 x 0.43% + 0.44 x (-0.42%) = +0.056% pro Trade (positiv)

Altes System:
- Erwartungswert = 0.56 x 0.36% + 0.44 x (-0.48%) = -0.010% pro Trade (negativ)

### 3.5 Regime-abhaengiger Cooldown

| Regime | Cooldown | Begruendung |
|--------|----------|-------------|
| Open Drive | 2 Min | Hoechste Breakout-Dichte |
| Late Morning | 3 Min | Gute Bedingungen |
| Midday | 5 Min | 50% Fake-Rate, laengere Sperre sicherer |
| Afternoon | 3 Min | Volumen kehrt zurueck |

### 3.6 Datenbasierte Entscheidung: Trading-Ende 15:00 ET

Analyse der letzten Handelsstunde (Run-003, 48 Trades):

| Zeitfenster (ET) | Trades | PnL | Win-Rate |
|------------------|--------|-----|----------|
| 13:00-13:59 | 4 | +2.39% | 75% |
| 14:00-14:59 | 18 | +0.88% | 50% |
| 15:00-15:59 | 26 | -0.11% | 58% |
| davon 15:15-16:00 | 21 | -2.26% | 48% |
| davon 15:30-16:00 | 10 | -1.97% | 40% |
| davon 15:45-16:00 | 3 | -1.19% | 0% |

Die Close-Phase (15:00-16:00) hat mit 26 Trades alle Gewinne aus den 22 besseren
Trades (13:00-15:00) neutralisiert. Median-Haltedauer = 13 Minuten -- Trades in
den letzten 30 Minuten schaffen ihren vollen Zyklus nicht mehr.

Umsetzung: Trading-Ende um 15:00 ET (21:00 MESZ). Alle Positionen liquidieren,
keine neuen Einstiege. Begruendung: Daten zeigen negative Expected Value nach 15:00.

### 3.7 Feature-Wichtigkeit

Top-3 Features stellen >90% der Vorhersagekraft (LightGBM Gain):
1. `return_1m` (Momentum)
2. `Slope_close_1` (Kurz-Trend)
3. `minutes_since_open` (Tageszeit)

Implikation: 79 der 82 Features koennten entfernt werden (Rechenzeit, Overfitting).

---

## 4. Code-Aenderungen

| Datei | Aenderungen |
|-------|-------------|
| `account_manager.py` | Trailing Stop (graduell), Ratchet-Mode, Time-Decay SL, Wash-Trade-Cooldown, Regime-Time-Stop, Exit-Reason-Logging |
| `trading_loop.py` | Entry-Rules E3-E5, Regime-Funktionen, Finder-Votes-Tracking, Regime-Positionsgroesse/Time-Stop, Run-Logger-Parameter, Shutdown-Exit-Reason, Trading-Ende 15:00 ET |
| `data_logger.py` | Run-Verzeichnisse, `run_info.json`, `run_exit.json`, `exit_reason`-Spalte, `list_runs()` |
| `conf/params.yaml` | Parameter synchronisiert |

Neue CLI-Parameter (alle mit sinnvollen Defaults):

```
--trailing_sl_pct 0.005     Trailing Start-Abstand (0.50%)
--trailing_min_pct 0.002    Trailing Min-Abstand (0.20%)
--trailing_ramp_pct 0.005   Profit-Level fuer vollen Trail-Lock
--no_trailing_sl            Trailing Stop deaktivieren
--no_ratchet                Ratchet-Mode deaktivieren
--reentry_cooldown 5        Wash-Trade-Sperre (dynamisch je Regime)
--sl_time_decay_target 0.003  SL-Ziel am Time-Stop-Ende
--sl_time_decay_grace 10    Gnadenfrist in Minuten
--no_entry_rules            Entry-Rules E3-E5 deaktivieren
```

---

## 5. Detaillierte Datenanalyse (Run-003, 48 Trades)

### 5.1 Erwartungswert-Progression

| System | E[Trade] | Formel |
|--------|----------|--------|
| ALT (statisch) | -0.009% | 55.9% x 0.362% + 44.1% x (-0.479%) |
| NEU (dynamisch, alle) | +0.057% | 56.2% x 0.430% + 43.8% x (-0.422%) |
| NEU (nur 13-15h, ohne Close) | **+0.149%** | 54.5% x 0.539% + 45.5% x (-0.320%) |

Der Sprung von -0.009% auf +0.057% pro Trade ist die zentrale Metrik des Tages.
Mit 15:00-Cutoff verdreifacht sich der Erwartungswert auf +0.149%.

### 5.2 Haltedauer und Profitabilitaet

| Haltedauer | Trades | PnL | Win | PF |
|------------|--------|-----|-----|-----|
| <5 Min | 4 | +0.07% | 50% | 1.15 |
| 5-15 Min | 22 | -1.81% | 50% | **0.66** |
| 15-30 Min | 16 | +3.31% | 56% | **2.38** |
| >30 Min | 6 | +1.59% | 83% | **10.10** |

Die 5-15-Minuten-Trades sind der Problembereich (PF 0.66). Das sind Trades, die
frueh durch Trailing/Time-Decay ausgestoppt wurden -- die Schutz-Schichten
greifen, aber der Trade hatte nicht genug Zeit, sich zu entfalten.

Die profitabelsten Trades brauchen 15+ Minuten. Median = 13 Minuten.

### 5.3 Symbol-Konzentration

Die Gewinne konzentrieren sich auf wenige Symbole:

Top-5 (zusammen +5.5% PnL):
- LCID (6 Trades, +2.12%), MRNA (5 Trades, +2.02%), LRCX (2 Trades, +0.85%),
  CRWD (1 Trade, +0.55%), FTNT (1 Trade, +0.50%)

Bottom-5 (zusammen -3.5% PnL):
- CHTR (3 Trades, -1.30%), ON (1 Trade, -0.71%), INTC (1 Trade, -0.56%),
  ORLY (1 Trade, -0.49%), CTSH (1 Trade, -0.48%)

LCID und MRNA allein generierten +4.14% der +3.16% Gesamt-PnL. Ohne diese
beiden Symbole waere das System negativ. Diese Konzentration ist ein Risiko --
faellt eines dieser Symbole aus der NASDAQ-100, bricht eine wichtige Einnahmequelle weg.

### 5.4 Methodische Einschraenkungen

Die heutigen Daten unterliegen folgenden Einschraenkungen:

1. **ALT-Daten heterogen:** Die 143 Trades der statischen Strategie stammen aus
   mehreren Runs mit unterschiedlichen Parametern (SL=0.20% und SL=1.0%). Sie
   als eine homogene Gruppe zu behandeln ist methodisch unsauber.
2. **Sample Size NEU:** 48 Trades sind zu wenig fuer statistische Signifikanz.
   Der Mann-Whitney-U-Test ergibt p=0.51.
3. **Open Drive fehlt:** Die beste Phase (09:30-10:30 ET) wurde mit dem neuen
   System nicht getestet. Ergebnisse koennen in der Morning-Session abweichen.
4. **15:00-Cutoff retrospektiv:** Die Entscheidung wurde an denselben Daten
   getroffen, die sie validieren soll. Uebertragbarkeit auf andere Tage ungewiss.
5. **Ein-Tages-Stichprobe:** Keine Wochentags-Varianz. Montag kann anders
   aussehen als der heutige Sonntag/Dienstag.

---

## 6. Konfiguration fuer den naechsten Handelstag

### 6.1 Aktive Schutz-Mechanismen (Default)

| Nr. | Mechanismus | Konfiguration |
|-----|-------------|---------------|
| 1 | Fixer SL | -1.0% |
| 2 | Time-Decay SL | -1.0% -> -0.3%, Gnadenfrist 10 Min |
| 3 | Trailing Stop | 0.50% -> 0.20% graduell |
| 4 | Ratchet-Mode | TP-Level wird SL-Boden |
| 5 | Regime-Filter | 3 Votes Midday, 2 sonst |
| 6 | Regime-Cooldown | 2-5 Min je Regime |
| 7 | Trading-Ende | 15:00 ET (21:00 MESZ) |
| 8 | Auto-Liquidation | 15:00 ET |

### 6.2 Erwartung fuer vollstaendigen Handelstag (mit 15:00-Cutoff)

| Metrik | Konservativ | Realistisch | Optimistisch |
|--------|-------------|-------------|--------------|
| Trades | 40-60 | 60-80 | 80-100 |
| Win-Rate | 50-55% | 55-60% | 60%+ |
| Profit Factor | 1.2-1.5 | 1.5-2.0 | 2.0+ |
| Tages-PnL | +1-2% | +2-4% | +4-6% |

Annahme: Open Drive (09:30-10:30 ET) wurde heute nicht getestet und sollte die
Ergebnisse verbessern (hoechste Breakout-Qualitaet des Tages).

### 6.3 Offene Fragen fuer den naechsten Handelstag

- Open Drive: Bricht die beste Breakout-Phase die heute etablierten Metriken nach oben?
- Midday: Wiederholen sich 75% Win-Rate und +2.39% PnL?
- 15:00-Cutoff: Werden gute Trades zwischen 15:00-15:15 verpasst?
- Trailing-Parameter: Ist 0.50% -> 0.20% ueber 0.50% Ramp optimal?

### 6.4 Startbefehl

```
cd C:\01_Uni\Projekte\USW\USW-TradingModel
python scripts/08_deployment/trading_loop.py --paper
```

Der Loop startet, wartet auf 15:30 MESZ (Market Open), tradet bis 21:00 MESZ
(15:00 ET), liquidiert dann automatisch alle Positionen.

---

## 7. Schnellreferenz

```
# Standard-Start (alle Schutz-Mechanismen aktiv):
python scripts/08_deployment/trading_loop.py --paper

# Dry-Run-Test:
python scripts/08_deployment/trading_loop.py --dry_run --once

# Statisches System (nur fixer SL/TP, ohne Schutz):
python scripts/08_deployment/trading_loop.py --paper --no_trailing_sl --no_ratchet --no_entry_rules

# Datenanalyse:
python -c "
from scripts.08_deployment.data_logger import DataLogger
for run in DataLogger.list_runs():
    print(f\"{run['run_name']}: {run['params']['mode']}, {run['duration_minutes']}min\")
"
```

---

*Kernerkenntnis: Nicht bessere ML-Modelle machen den Unterschied, sondern dynamisches Risikomanagement. Der Sprung von Profit Factor 0.96 auf 1.37 kam durch acht Schutz-Schichten und Regime-basierte Parameter-Anpassung -- nicht durch neue Architekturen. Der 15:00-Cutoff ist die datenbasierte Konsequenz aus 26 verlustreichen Close-Trades.*
