@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Testy Dark-Field Analyzer

where py >nul 2>nul
if %errorlevel% equ 0 (
    py -m pip install pytest >nul 2>nul
    py -m pytest -q
    goto end
)
python -m pip install pytest >nul 2>nul
python -m pytest -q

:end
echo.
pause
