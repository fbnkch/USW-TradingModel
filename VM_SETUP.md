# USW-TradingModel — VM Setup Anleitung

> **Stand:** 06.07.2026  
> **VM:** `USWS002V.f4.htw-berlin.de`  
> **User:** `wiuser`

---

## Inhaltsverzeichnis

1. [Erstverbindung zur VM](#1-erstverbindung-zur-vm)
2. [Projektdateien übertragen](#2-projektdateien-übertragen)
3. [Python installieren](#3-python-installieren)
4. [Abhängigkeiten installieren](#4-abhängigkeiten-installieren)
5. [Alpaca-Keys konfigurieren](#5-alpaca-keys-konfigurieren)
6. [Dry-Run-Test](#6-dry-run-test)
7. [Scheduled Task einrichten](#7-scheduled-task-einrichten)
8. [Manueller Start / Stop](#8-manueller-start--stop)
9. [Monitoring & Logs](#9-monitoring--logs)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Erstverbindung zur VM

### Verbindung herstellen

1. `Win + R` drücken, `mstsc` eingeben, Enter
2. Im Remotedesktop-Fenster:
   - **Computer:** `USWS002V.f4.htw-berlin.de`
   - **Benutzername:** `wiuser`
3. Auf **"Lokale Ressourcen"** klicken → Tab **"Laufwerke"** → **"Laufwerke"** aufklappen
   - ✅ Haken bei dem Laufwerk setzen, auf dem das Projekt liegt (z.B. `C:`)
   - Das ist **wichtig** für den Dateitransfer in Schritt 2!
4. Auf **"Verbinden"** klicken → Passwort eingeben

### Nach der Anmeldung

Du siehst den Windows-Desktop der VM. Öffne den **Windows Explorer** und lege folgendes Verzeichnis an:

```
C:\USW-TradingModel\
```

---

## 2. Projektdateien übertragen

### Methode A: ZIP-Datei (empfohlen)

**Auf deinem lokalen PC:**

1. Öffne `C:\01_Uni\Projekte\USW\USW-TradingModel\` im Explorer
2. Markiere ALLE Ordner und Dateien
3. Rechtsklick → **Senden an** → **ZIP-komprimierten Ordner**
4. Benenne die ZIP um in `USW-TradingModel.zip`

**Auf der VM:**

1. Öffne den Explorer auf der VM
2. Navigiere zu **"Dieser PC"** — du solltest dein lokales Laufwerk unter "Umgeleitete Laufwerke" sehen
3. Öffne das umgeleitete Laufwerk und kopiere die `USW-TradingModel.zip`
4. Füge sie in `C:\` auf der VM ein
5. Rechtsklick auf die ZIP → **Alle extrahieren...** → Ziel: `C:\USW-TradingModel\`

### Methode B: Direktes Kopieren (langsamer bei vielen Dateien)

1. Auf der VM: Explorer öffnen, zu "Dieser PC" → umgeleitetes Laufwerk → `C:\01_Uni\Projekte\USW\USW-TradingModel\`
2. Alle Ordner und Dateien markieren (Strg+A)
3. Kopieren (Strg+C)
4. In `C:\USW-TradingModel\` einfügen (Strg+V)

### Verzeichnisstruktur nach dem Kopieren

```
C:\USW-TradingModel\
├── artifacts\
│   └── models\
│       ├── mlp_model.pt
│       ├── lstm_model.pt
│       ├── gru_model.pt
│       ├── cnn_model.pt
│       └── lightgbm_model.txt
├── conf\
│   ├── keys.yaml
│   └── params.yaml
├── data\
│   ├── nasdaq100_symbols.csv
│   └── processed\
│       ├── global_scaler.pkl
│       ├── class_balance.json
│       └── pre_split\
│           └── features.txt
├── scripts\
│   ├── 08_deployment\
│   │   ├── trading_loop.py      ← Hauptskript
│   │   ├── account_manager.py
│   │   ├── feature_engine.py
│   │   └── data_logger.py
│   └── setup_vm.ps1             ← Setup-Skript
├── requirements.txt
├── README.md
└── TRADING_STRATEGIE.md
```

**Pflicht-Check:** Die folgenden Dateien MÜSSEN vorhanden sein:

| Datei | Zweck |
|-------|-------|
| `artifacts/models/mlp_model.pt` | MLP-Modell (Filter) |
| `artifacts/models/lstm_model.pt` | LSTM-Modell (Finder) |
| `artifacts/models/gru_model.pt` | GRU-Modell (Finder) |
| `artifacts/models/cnn_model.pt` | CNN-Modell (Finder) |
| `artifacts/models/lightgbm_model.txt` | LightGBM-Modell (Finder) |
| `data/nasdaq100_symbols.csv` | NASDAQ-100 Symbolliste |
| `data/processed/global_scaler.pkl` | Feature-Scaler |
| `data/processed/pre_split/features.txt` | Feature-Namen |
| `conf/keys.yaml` | Alpaca API-Keys |
| `conf/params.yaml` | Konfiguration |

---

## 3. Python installieren

### Prüfen ob Python schon installiert ist

PowerShell als Administrator öffnen (Rechtsklick auf Start → **Windows PowerShell (Administrator)**):

```powershell
python --version
```

Wenn eine Version ≥ 3.10 erscheint → weiter zu [Schritt 4](#4-abhängigkeiten-installieren).

### Falls nicht: Python 3.10 installieren

#### Option A: winget (schnell)

```powershell
winget install Python.Python.3.10 --accept-package-agreements
```

Danach PowerShell **neu starten** und prüfen:

```powershell
python --version
```

#### Option B: Manueller Download (falls winget nicht verfügbar)

1. Browser auf der VM öffnen
2. https://www.python.org/downloads/ → **Python 3.10.x** herunterladen
3. Installer ausführen
4. ⚠️ **WICHTIG:** Haken setzen bei **"Add Python to PATH"**
5. Installation abschließen
6. PowerShell neu starten und `python --version` prüfen

---

## 4. Abhängigkeiten installieren

PowerShell als Administrator:

```powershell
cd C:\USW-TradingModel

# pip upgraden
python -m pip install --upgrade pip

# CPU PyTorch (VM hat keine GPU — CPU-Variante reicht für Inference)
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Alle Pakete aus requirements.txt
python -m pip install -r requirements.txt

# Explizit alpaca-py (falls nicht in requirements)
python -m pip install alpaca-py
```

### Prüfen: Alle Pakete importierbar?

```powershell
python -c "import torch; import pandas; import numpy; import lightgbm; import joblib; import yaml; print('Alle Pakete OK'); print(f'PyTorch {torch.__version__}')"
```

Sollte ausgeben: `Alle Pakete OK` + PyTorch-Version.

---

## 5. Alpaca-Keys konfigurieren

Die API-Keys liegen in `C:\USW-TradingModel\conf\keys.yaml`. Öffne die Datei mit Notepad:

```powershell
notepad C:\USW-TradingModel\conf\keys.yaml
```

Die Datei muss so aussehen:

```yaml
ALPACA:
  API_KEY: "PK_xxxxxxxxxxxxxxxxxxxx"
  SECRET_KEY: "yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy"
```

**Wichtig:** Paper-Trading-Keys verwenden (beginnen mit `PK_`), **nicht** Live-Keys!

### Alternative: Umgebungsvariablen (sicherer für VMs)

```powershell
[System.Environment]::SetEnvironmentVariable('ALPACA_API_KEY', 'PK_xxxxxxxx', 'User')
[System.Environment]::SetEnvironmentVariable('ALPACA_SECRET_KEY', 'yyyyyyyyyyyy', 'User')
```

Der Code in `trading_loop.py` liest zuerst die Umgebungsvariablen, dann `keys.yaml`.

---

## 6. Dry-Run-Test

Bevor das Paper-Trading läuft: **Dry-Run-Test** durchführen. Dabei werden keine Orders gesendet.

```powershell
cd C:\USW-TradingModel
python scripts/08_deployment/trading_loop.py --dry_run --once
```

### Erwartete Ausgabe

```
============================================================
USW-TradingModel - Paper Trading Loop
============================================================
Mode:    DRY-RUN
Features: 82
Device:   cpu
Alpaca:   Data-Client (Dry-Run)
Symbole:  100

Lade Modelle...
  5 Modelle geladen.

[Warmup] Lade ~5 Handelstage 1-Min-Bars fuer 100 Symbole...
  [1/100] A      (xxx Bars)
  ...

Feature-Engines: xx/100 Symbole bereit

============================================================
TRADING LOOP GESTARTET
============================================================
Max Positions: 10
Trailing SL: AN (graduell, schlaeft bis Kurs > Entry) | Start=0.50% → Min=0.20% @ +0.50% Profit)
Ratchet-Mode: AN – TP-Level wird zu SL-Boden (kein fester TP-Exit)
Exit-Regeln: TP=0.25% SL=1.00% | Time-Stop=30min

[09:35:01] Iteration 1 | Positionen: 0 | Ready: xx
  Bars: 100 | Features: xx
```

### Häufige Fehler beim Dry-Run

| Fehler | Ursache | Lösung |
|--------|---------|--------|
| `ModuleNotFoundError: No module named 'alpaca'` | alpaca-py nicht installiert | `pip install alpaca-py` |
| `FileNotFoundError: artifacts/models/...` | Modelle nicht kopiert | Schritt 2 wiederholen |
| `KeyError: 'ALPACA'` | keys.yaml fehlt oder falsch | Schritt 5 prüfen |
| `ConnectionError / Timeout` | Kein Internet auf der VM | Netzwerkverbindung prüfen |

Wenn der Dry-Run ohne Fehler durchläuft → weiter zu Schritt 7.

---

## 7. Scheduled Task einrichten

Damit der Trading-Loop **jeden Handelstag automatisch startet**, ohne dass jemand eingeloggt sein muss.

### Option A: Per PowerShell (schnell)

Als Administrator in PowerShell:

```powershell
cd C:\USW-TradingModel
powershell -ExecutionPolicy Bypass -File scripts/setup_vm.ps1
```

Das Skript erledigt die Schritte 1–6 automatisch und richtet den Scheduled Task ein.

### Option B: Manuell per Aufgabenplanung

1. **Aufgabenplanung öffnen:**
   - `Win + R` → `taskschd.msc` → Enter

2. **Neue Aufgabe erstellen:**
   - Rechtsklick auf **"Aufgabenplanungsbibliothek"** → **"Aufgabe erstellen..."**

3. **Tab "Allgemein":**
   - Name: `USW-TradingLoop`
   - Beschreibung: `USW-TradingModel Paper Trading — startet taeglich 15:20 MESZ`
   - ✅ **"Unabhängig von der Benutzeranmeldung ausführen"**
   - ✅ **"Mit höchsten Privilegien ausführen"**

4. **Tab "Trigger" → Neu:**
   - Aufgabe starten: **"Bei Systemstart"**
   - ✅ Aktiviert
   > Der Loop läuft danach 24/7. Er handled Market-Open/Close selbstständig:
   > - 15:30–22:00 MESZ: aktives Trading
   > - 22:00–15:30 MESZ: Schlafmodus (alle 5 Min Check)
   > - Wochenende: Schlafmodus
   > - Montag 15:30: wacht von selbst auf und tradet weiter

5. **Tab "Aktionen" → Neu:**
   - Aktion: **"Programm starten"**
   - Programm/Skript: `C:\Program Files\Python310\python.exe`
     > ⚠️ Den tatsächlichen Pfad mit `(Get-Command python).Source` in PowerShell ermitteln!
   - Argumente: `"C:\USW-TradingModel\scripts\08_deployment\trading_loop.py" --paper`
   - Starten in: `C:\USW-TradingModel`

6. **Tab "Bedingungen":**
   - ❌ Haken bei "Nur starten, wenn Netzwerkverbindung verfügbar" — rausnehmen
   - ❌ Haken bei "Aufgabe starten, wenn Computer im Akkubetrieb" — rausnehmen

7. **Tab "Einstellungen":**
   - ✅ "Aufgabe bei Bedarf sofort starten"
   - ✅ "Aufgabe bei Scheitern neu starten nach:" → 5 Minuten, max. 3 Versuche
   - ❌ "Aufgabe anhalten nach:" → **deaktiviert** (kein Limit — Loop läuft unbegrenzt)
   - ❌ "Aufgabe beenden, wenn sie länger als... dauert" — rausnehmen
   - "Falls die Aufgabe bereits ausgeführt wird:" → **"Keine neue Instanz starten"**

8. **OK** → Windows-Passwort eingeben (wird für "Unabhängig von Benutzeranmeldung" benötigt)

### Testen

```powershell
# Manuell starten:
Start-ScheduledTask -TaskName "USW-TradingLoop"

# Status prüfen:
Get-ScheduledTask -TaskName "USW-TradingLoop" | Select-Object State

# Letztes Ergebnis:
Get-ScheduledTaskInfo -TaskName "USW-TradingLoop"
```

---

## 8. Manueller Start / Stop

Falls der Scheduled Task nicht genutzt werden soll, kann der Loop auch manuell gestartet werden.

### Starten

```powershell
cd C:\USW-TradingModel
python scripts/08_deployment/trading_loop.py --paper
```

### Stoppen

**Im Trading-Loop-Fenster:** `Ctrl + C` drücken.  
→ Das Skript schließt alle offenen Positionen sauber (Emergency Close All).

**Falls das Fenster nicht erreichbar ist:**

```powershell
# Prozess-ID finden
Get-Process python | Select-Object Id, ProcessName

# Prozess beenden (sendet Ctrl+C, sauberer Shutdown)
Stop-Process -Id <PID>
```

### Parameter-Übersicht

```bash
# Standard (alle Features an):
python scripts/08_deployment/trading_loop.py --paper

# Ohne Trailing Stop:
python scripts/08_deployment/trading_loop.py --paper --no_trailing_sl

# Ohne Ratchet-Mode:
python scripts/08_deployment/trading_loop.py --paper --no_ratchet

# Klassischer Modus (nur fixer SL + fixer TP):
python scripts/08_deployment/trading_loop.py --paper --no_trailing_sl --no_ratchet

# Nur 20 Symbole (Testmodus):
python scripts/08_deployment/trading_loop.py --paper --symbols 20

# Dry-Run (keine Orders):
python scripts/08_deployment/trading_loop.py --dry_run --once

# Einmalige Iteration:
python scripts/08_deployment/trading_loop.py --paper --once
```

---

## 9. Monitoring & Logs

### Logs einsehen

```powershell
cd C:\USW-TradingModel\data\paper_trading
dir
```

Die Logs werden im `data/paper_trading/` Verzeichnis gespeichert:
- `bars_YYYYMMDD.csv` — Jede empfangene Bar
- `signals_YYYYMMDD.csv` — Jedes generierte Signal
- `orders_YYYYMMDD.csv` — Jede platzierte Order
- `positions_YYYYMMDD.csv` — Geschlossene Positionen mit P&L

### Tägliche Analyse

```powershell
python scripts/08_deployment/analyze_day.py
```

### Während des Laufs

Das Trading-Loop-Fenster zeigt live:
```
[14:35:01] Iteration 42 | Positionen: 3 | Ready: 97
  Bars: 100 | Features: 97
  SIGNALE: 2
  [ORDER] AAPL: BUY 34 @ $195.42
  EXIT: NVDA (trailing_stop) PnL=+0.32%
```

---

## 10. Troubleshooting

### Der Loop startet nicht

```powershell
# Prüfen ob Python im PATH ist
$env:Path -split ';' | Select-String Python

# Python-Pfad ermitteln
(Get-Command python).Source
```

### "No module named 'scripts'"

```powershell
# Wichtig: Im Projekt-Root-Verzeichnis sein!
cd C:\USW-TradingModel
pwd  # Muss C:\USW-TradingModel anzeigen
```

### Scheduled Task läuft, aber kein Output

1. Prüfen ob der Task mit den richtigen Credentials läuft
2. Task-Eigenschaften → "Unabhängig von Benutzeranmeldung ausführen" muss an sein
3. Prüfen ob Python im System-PATH ist (nicht nur User-PATH)

### "API rate limit exceeded"

Alpaca Free-Tier-Limit: 200 API-Calls pro Minute. Der Loop macht ~1-2 Calls/Minute (Bars + Order). Sollte nie ein Problem sein. Falls doch: Loop pausiert automatisch.

### "ConnectionError" bei Alpaca

- Netzwerkverbindung der VM prüfen
- Firewall: Ausgehende Verbindungen auf Port 443 (HTTPS) müssen erlaubt sein
- Alpaca API-Status prüfen: https://status.alpaca.markets/

### Speicherplatz auf der VM

Die Log-Dateien und Warmup-Daten wachsen mit der Zeit. Regelmäßig prüfen:

```powershell
# Speicherplatz prüfen
Get-PSDrive C

# Alte Logs löschen (älter als 7 Tage)
Get-ChildItem C:\USW-TradingModel\data\paper_trading\*.csv |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-7) } |
    Remove-Item
```

---

## Anhang A: Schnellreferenz

| Was | Befehl |
|-----|--------|
| Dry-Run-Test | `python scripts/08_deployment/trading_loop.py --dry_run --once` |
| Paper-Trading starten | `python scripts/08_deployment/trading_loop.py --paper` |
| Paper-Trading stoppen | `Ctrl + C` im Fenster |
| Setup-Skript | `powershell -ExecutionPolicy Bypass -File scripts/setup_vm.ps1` |
| Scheduled Task starten | `Start-ScheduledTask -TaskName "USW-TradingLoop"` |
| Scheduled Task Status | `Get-ScheduledTaskInfo -TaskName "USW-TradingLoop"` |
| Logs ansehen | `ls data\paper_trading\` |

## Anhang B: 24/7 Ablauf des Trading-Loops

Der Loop läuft **einmal gestartet dauerhaft** und managed seine Zustände selbst:

```
Montag 15:30 ─┐
              │  AKTIV: Iterationen jede Minute
              │  Signale → Orders → Exit-Checks
Dienstag      │
Mittwoch      │  (15:30–22:00 MESZ jeden Handelstag)
Donnerstag    │
Freitag 22:00─┘
              │
Freitag 22:00─┐
              │  SCHLAF: end_of_day() → alle 5 Min Check
Samstag       │  "Markt offen? Nein → weiterschlafen (300s)"
Sonntag       │
Montag 15:30─┘  Markt wieder offen → TRADING STARTET
```

| Phase | Was passiert |
|-------|-------------|
| **15:30–22:00 MESZ (Mo–Fr)** | Aktives Trading. Jede Minute: Bars holen → Features → Ensemble → Signale → Entry/Exit |
| **22:00–15:30 MESZ (über Nacht)** | `end_of_day()` + `reset_daily()`. Loop schläft, prüft alle 5 Min ob Markt offen |
| **Wochenende (Fr 22:00 – Mo 15:30)** | Schlafmodus. ~65 Stunden, prüft alle 5 Min |
| **Montag 15:30** | `wait_until_market_open()` erkennt offenen Markt → Trading beginnt |

### Was der Scheduled Task macht

- **Trigger:** Bei Systemstart (einmalig)
- **Danach:** Der Loop-Prozess läuft **für immer** (kein Zeitlimit)
- **Bei Absturz:** Automatischer Restart (3 Versuche, 5 Min Abstand)
- **Bei VM-Neustart:** Scheduled Task startet Loop automatisch neu

### Wichtige Uhrzeiten

| Zeit (MESZ/MEZ) | Zeit (ET) | Bedeutung |
|-----------------|-----------|-----------|
| 15:30 | 09:30 | Market Open — Loop beginnt zu traden |
| 22:00 | 16:00 | Market Close — Loop geht in Schlafmodus |

> ⚠️ **Sommer-/Winterzeit:** US-Marktzeiten orientieren sich an Eastern Time. Deutschland und USA wechseln etwa am gleichen Wochenende (März/November). Market Open ist effektiv **immer 15:30 deutscher Zeit**.
