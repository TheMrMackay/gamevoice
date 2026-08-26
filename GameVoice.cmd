@echo off
rem Launch the GameVoice desktop app (no console window).
setlocal
cd /d "%~dp0"
start "" ".venv\Scripts\pythonw.exe" -m gamevoice.gui.app
endlocal
