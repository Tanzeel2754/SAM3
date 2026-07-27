@echo off
REM Start SAM3 Studio (uses sam3 conda env if available)
set PYTHON=c:\Users\ait44\anaconda3\envs\sam3\python.exe
if not exist "%PYTHON%" set PYTHON=python
"%PYTHON%" "%~dp0run.py" --host 127.0.0.1 --port 7860 %*
