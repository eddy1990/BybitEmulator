@echo off
echo ================================
echo Starting Bybit Emulator + Bot
echo ================================

REM ---- Emulator starten ----
echo Starting Emulator...
start "Bybit Emulator" cmd /k ^
    cd /d E:\TradingBotClean\Bybit_Emulator ^&^& ^
    python -m uvicorn Emulator_main:app --host 127.0.0.1 --port 9000
