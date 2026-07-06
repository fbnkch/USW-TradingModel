# =============================================================================
# USW-TradingModel — VM Setup Script
# =============================================================================
# Einrichtung einer Windows-VM für den Paper-Trading-Loop.
#
# Aufruf (als Administrator!):
#   powershell -ExecutionPolicy Bypass -File setup_vm.ps1
#
# Oder im PowerShell-Fenster (Admin):
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\setup_vm.ps1
# =============================================================================

$ErrorActionPreference = "Stop"
$ProjectDir = "C:\USW-TradingModel"
$PythonVersion = "3.10"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " USW-TradingModel — VM Setup" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# ──────────────────────────────────────────────────────────────────
# 1. Admin-Check
# ──────────────────────────────────────────────────────────────────
Write-Host "[1/6] Pruefe Administrator-Rechte..." -ForegroundColor Yellow

$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")
if (-not $isAdmin) {
    Write-Host "FEHLER: Dieses Skript muss als Administrator ausgefuehrt werden!" -ForegroundColor Red
    Write-Host "Bitte PowerShell als Admin oeffnen und erneut ausfuehren." -ForegroundColor Red
    pause
    exit 1
}
Write-Host "  OK — Administrator-Rechte vorhanden" -ForegroundColor Green

# ──────────────────────────────────────────────────────────────────
# 2. Python-Check
# ──────────────────────────────────────────────────────────────────
Write-Host "`n[2/6] Pruefe Python-Installation..." -ForegroundColor Yellow

$pythonCmd = $null
try {
    $pythonVersion = & python --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        $pythonCmd = "python"
        Write-Host "  OK — $pythonVersion" -ForegroundColor Green
    }
} catch {
    # python not found
}

if (-not $pythonCmd) {
    try {
        $pythonVersion = & python3 --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            $pythonCmd = "python3"
            Write-Host "  OK — $pythonVersion" -ForegroundColor Green
        }
    } catch {}
}

if (-not $pythonCmd) {
    Write-Host "  Python nicht gefunden. Installiere Python 3.10 via winget..." -ForegroundColor Yellow
    try {
        winget install Python.Python.3.10 --accept-package-agreements --accept-source-agreements
        Write-Host "  Python 3.10 installiert. Bitte PowerShell NEU STARTEN und Skript erneut ausfuehren." -ForegroundColor Green
        Write-Host "  (winget installiert Python unter einem neuen Pfad, der erst nach Neustart verfuegbar ist)" -ForegroundColor Yellow
        pause
        exit 0
    } catch {
        Write-Host "  FEHLER: winget-Installation fehlgeschlagen." -ForegroundColor Red
        Write-Host "  Bitte Python manuell installieren: https://www.python.org/downloads/" -ForegroundColor Red
        Write-Host "  Siehe VM_SETUP.md fuer manuelle Anleitung." -ForegroundColor Red
        pause
        exit 1
    }
}

# ──────────────────────────────────────────────────────────────────
# 3. Projektdateien pruefen
# ──────────────────────────────────────────────────────────────────
Write-Host "`n[3/6] Pruefe Projekt-Dateien..." -ForegroundColor Yellow

# Pruefe ob wir im richtigen Verzeichnis sind (Sibling von scripts/)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

if (-not (Test-Path "$ProjectRoot\scripts\08_deployment\trading_loop.py")) {
    Write-Host "  WARNUNG: trading_loop.py nicht gefunden." -ForegroundColor Yellow
    Write-Host "  Dieses Skript erwartet das USW-TradingModel Projekt im gleichen Verzeichnis." -ForegroundColor Yellow
    Write-Host "  Projekt-Root: $ProjectRoot" -ForegroundColor Yellow
}

# Kritische Dateien pruefen
$criticalFiles = @(
    "artifacts\models\mlp_model.pt",
    "artifacts\models\lstm_model.pt",
    "artifacts\models\gru_model.pt",
    "artifacts\models\cnn_model.pt",
    "artifacts\models\lightgbm_model.txt",
    "data\nasdaq100_symbols.csv",
    "data\processed\global_scaler.pkl",
    "data\processed\pre_split\features.txt",
    "conf\keys.yaml",
    "conf\params.yaml",
    "scripts\08_deployment\trading_loop.py",
    "scripts\08_deployment\account_manager.py",
    "scripts\08_deployment\feature_engine.py",
    "scripts\08_deployment\data_logger.py",
    "requirements.txt"
)

$missing = @()
foreach ($file in $criticalFiles) {
    if (-not (Test-Path "$ProjectRoot\$file")) {
        $missing += $file
    }
}

if ($missing.Count -gt 0) {
    Write-Host "  FEHLER: Folgende Dateien fehlen:" -ForegroundColor Red
    foreach ($m in $missing) {
        Write-Host "    - $m" -ForegroundColor Red
    }
    Write-Host "`n  Bitte das gesamte USW-TradingModel Verzeichnis vom lokalen PC auf die VM kopieren." -ForegroundColor Yellow
    Write-Host "  (RDP-Laufwerkumleitung oder ZIP-Transfer — siehe VM_SETUP.md)" -ForegroundColor Yellow
    pause
    exit 1
}
Write-Host "  OK — Alle $($criticalFiles.Count) kritischen Dateien vorhanden" -ForegroundColor Green

# ──────────────────────────────────────────────────────────────────
# 4. Python-Abhaengigkeiten installieren
# ──────────────────────────────────────────────────────────────────
Write-Host "`n[4/6] Installiere Python-Pakete..." -ForegroundColor Yellow

# Pip upgraden
Write-Host "  Upgrade pip..."
& $pythonCmd -m pip install --upgrade pip --quiet

# CPU PyTorch (VM hat wahrscheinlich keine GPU)
Write-Host "  Installiere PyTorch (CPU)..."
& $pythonCmd -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet

# Restliche Pakete aus requirements.txt
Write-Host "  Installiere Pakete aus requirements.txt..."
& $pythonCmd -m pip install -r "$ProjectRoot\requirements.txt" --quiet

# Alpaca separat (falls nicht in requirements)
Write-Host "  Installiere alpaca-py..."
& $pythonCmd -m pip install alpaca-py --quiet

Write-Host "  OK — Alle Pakete installiert" -ForegroundColor Green

# ──────────────────────────────────────────────────────────────────
# 5. Alpaca-Keys konfigurieren
# ──────────────────────────────────────────────────────────────────
Write-Host "`n[5/6] Konfiguriere Alpaca API-Keys..." -ForegroundColor Yellow

$keysFile = "$ProjectRoot\conf\keys.yaml"
if (Test-Path $keysFile) {
    Write-Host "  keys.yaml gefunden. Inhalt:" -ForegroundColor Yellow
    Write-Host "  $(Get-Content $keysFile -Raw)" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  WICHTIG: Stelle sicher, dass die Keys in keys.yaml korrekt sind!" -ForegroundColor Yellow
    Write-Host "  Du kannst sie auch als Umgebungsvariablen setzen:" -ForegroundColor Yellow
    Write-Host "    `$env:ALPACA_API_KEY = 'DEIN_KEY'" -ForegroundColor Gray
    Write-Host "    `$env:ALPACA_SECRET_KEY = 'DEIN_SECRET'" -ForegroundColor Gray
}

# ──────────────────────────────────────────────────────────────────
# 6. Scheduled Task erstellen (Auto-Start)
# ──────────────────────────────────────────────────────────────────
Write-Host "`n[6/6] Erstelle Windows Scheduled Task..." -ForegroundColor Yellow

$taskName = "USW-TradingLoop"
$taskScript = "$ProjectRoot\scripts\08_deployment\trading_loop.py"
$taskWorkingDir = "$ProjectRoot"
$taskLogDir = "$ProjectRoot\logs"
$taskLogFile = "$taskLogDir\trading_loop_`$(Get-Date -Format 'yyyyMMdd').log"

# Log-Verzeichnis anlegen
if (-not (Test-Path $taskLogDir)) {
    New-Item -ItemType Directory -Path $taskLogDir -Force | Out-Null
}

# Alte Task loeschen falls vorhanden
try {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
} catch {}

# Task-Action: Python-Skript ausfuehren
# Der Loop laeuft 24/7: tradet waehrend Marktzeiten, schlaeft ausserhalb,
# wartet uebers Wochenende, startet Montag von selbst wieder.
$action = New-ScheduledTaskAction -Execute $pythonCmd `
    -Argument "`"$taskScript`" --paper" `
    -WorkingDirectory $taskWorkingDir

# Task-Trigger: Bei Systemstart (nicht taeglich — der Loop managed Marktzeiten selbst)
$trigger = New-ScheduledTaskTrigger -AtStartup

# Task-Einstellungen: KEIN Zeitlimit, unbegrenzt laufen lassen
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 5) `
    -ExecutionTimeLimit (New-TimeSpan -Seconds 0) `
    -MultipleInstances IgnoreNew

# Task registrieren
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType ServiceAccount `
    -RunLevel Highest

try {
    Register-ScheduledTask -TaskName $taskName `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Principal $principal `
        -Description "USW-TradingModel Paper Trading — 24/7, managed Marktzeiten automatisch" `
        -Force
    Write-Host "  OK — Scheduled Task '$taskName' erstellt" -ForegroundColor Green
    Write-Host "  Trigger: Bei Systemstart" -ForegroundColor Green
    Write-Host "  Der Loop laeuft dauerhaft und managed Marktzeiten selbst:" -ForegroundColor Green
    Write-Host "    - 15:30-22:00 MESZ → aktives Trading" -ForegroundColor Green
    Write-Host "    - 22:00-15:30 MESZ → Schlafmodus (checkt alle 5 Min)" -ForegroundColor Green
    Write-Host "    - Wochenende       → Schlafmodus" -ForegroundColor Green
} catch {
    Write-Host "  FEHLER beim Erstellen des Scheduled Tasks: $_" -ForegroundColor Red
    Write-Host "  Versuche alternative Methode (ohne ServiceAccount)..." -ForegroundColor Yellow

    try {
        $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -RunLevel Highest
        Register-ScheduledTask -TaskName $taskName `
            -Action $action `
            -Trigger $trigger `
            -Settings $settings `
            -Principal $principal `
            -Description "USW-TradingModel Paper Trading" `
            -Force
        Write-Host "  OK — Scheduled Task '$taskName' erstellt (User-Modus)" -ForegroundColor Green
    } catch {
        Write-Host "  FEHLER: Scheduled Task konnte nicht erstellt werden." -ForegroundColor Red
        Write-Host "  Bitte manuell einrichten — siehe VM_SETUP.md" -ForegroundColor Red
    }
}

# ──────────────────────────────────────────────────────────────────
# Abschluss — Dry-Run Test
# ──────────────────────────────────────────────────────────────────
Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host " SETUP ABGESCHLOSSEN" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Naechste Schritte:" -ForegroundColor Yellow
Write-Host "  1. Alpaca-Keys pruefen: $keysFile" -ForegroundColor White
Write-Host "  2. Dry-Run-Test ausfuehren:" -ForegroundColor White
Write-Host "     cd $taskWorkingDir" -ForegroundColor Gray
Write-Host "     $pythonCmd scripts/08_deployment/trading_loop.py --dry_run --once" -ForegroundColor Gray
Write-Host ""
Write-Host "  3. Paper-Trading starten:" -ForegroundColor White
Write-Host "     $pythonCmd scripts/08_deployment/trading_loop.py --paper" -ForegroundColor Gray
Write-Host ""
Write-Host "  4. Scheduled Task manuell testen:" -ForegroundColor White
Write-Host "     Start-ScheduledTask -TaskName '$taskName'" -ForegroundColor Gray
Write-Host ""

$runTest = Read-Host "Jetzt einen Dry-Run-Test ausfuehren? (j/n)"
if ($runTest -eq "j" -or $runTest -eq "J" -or $runTest -eq "y") {
    Write-Host "`nStarte Dry-Run-Test..." -ForegroundColor Yellow
    Set-Location $taskWorkingDir
    & $pythonCmd scripts/08_deployment/trading_loop.py --dry_run --once 2>&1 | Out-Host
}

Write-Host "`nFertig." -ForegroundColor Green
pause
