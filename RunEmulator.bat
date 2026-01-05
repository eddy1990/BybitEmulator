@echo off
echo ================================
echo Starting Bybit Emulator + Bot
echo ================================

start "Bybit Emulator (Uvicorn)" cmd /k python -m uvicorn Emulator_main:app --app-dir "E:\Bot\Bybit_Emulator" --host 127.0.0.1 --port 9000 --reload

