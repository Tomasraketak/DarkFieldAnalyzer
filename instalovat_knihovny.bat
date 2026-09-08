@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Instalace knihoven pro Dark-Field Analyzer

echo ===================================================================
echo     Instalace / kontrola knihoven
echo ===================================================================
echo.
echo Instaluji: opencv-python, numpy, PyQt6, matplotlib
echo.

where py >nul 2>nul
if %errorlevel% equ 0 (
    py -m pip install --upgrade pip
    py -m pip install -r requirements.txt
    if errorlevel 1 goto error
    goto done
)

where python >nul 2>nul
if %errorlevel% equ 0 (
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    if errorlevel 1 goto error
    goto done
)

echo CHYBA: Python nebyl v systemu nalezen.
echo Nainstalujte Python 3.10 nebo novejsi z https://www.python.org/downloads/
echo a pri instalaci zaskrtnete "Add Python to PATH".
goto error

:done
echo.
echo ===================================================================
echo Hotovo. Aplikaci spustte souborem spustit_analyzu.bat
echo ===================================================================
echo.
pause
exit /b 0

:error
echo.
echo ===================================================================
echo Pri instalaci doslo k chybe (vypis vyse).
echo ===================================================================
pause
exit /b 1
