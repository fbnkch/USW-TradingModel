@echo off
REM USW-TradingModel - Paper Trading Starter
REM Startet den Trading-Loop und verhindert Energiesparmodus

echo ============================================
echo USW-TradingModel - Paper Trading Loop
echo ============================================
echo.
echo Starte um %date% %time%
echo.

REM Energiesparmodus deaktivieren (verhindert Standby)
powercfg -change -standby-timeout-ac 0
powercfg -change -monitor-timeout-ac 30

REM In Projektverzeichnis wechseln
cd /d "C:\01_Uni\Projekte\USW\USW-TradingModel"

REM Output-Verzeichnis
if not exist "data\paper_trading" mkdir "data\paper_trading"

REM Trading-Loop starten (Output in Log-Datei + Konsole)
python scripts\08_deployment\trading_loop.py --paper 2>&1 | tee data\paper_trading\loop_output_%date:~-4,4%%date:~-10,2%%date:~-7,2%.log

echo.
echo Trading-Loop beendet um %time%
pause
