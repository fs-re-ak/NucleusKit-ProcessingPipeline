@echo off
rem Release this repository's package on TestPyPI, then PyPI. Options and details: tools\release_pip.py
setlocal
cd /d "%~dp0"
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
"%PY%" tools\release_pip.py %*
set "CODE=%ERRORLEVEL%"
rem Keep the window open when started by double-click.
if "%~1"=="" pause
exit /b %CODE%
