@echo off
rem GameVoice command line.  Examples:
rem   gv doctor --delay 5
rem   gv say "Hello there." --speaker "Lady Isolde"
rem   gv run
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" run.py %*
endlocal
