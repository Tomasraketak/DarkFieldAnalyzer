@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Dark-Field Contamination Analyzer

echo ===================================================================
echo     Dark-Field Slide Contamination Analyzer
echo ===================================================================
echo.
echo Spoustim aplikaci...
echo.

where py >nul 2>nul
if %errorlevel% equ 0 (
    py main.py %*
    if errorlevel 1 goto error
    goto end
)

where python >nul 2>nul
if %errorlevel% equ 0 (
    python main.py %*
    if errorlevel 1 goto error
    goto end
)

echo CHYBA: Python nebyl v systemu nalezen (PATH).
goto error

:error
echo.
echo ===================================================================
echo Program skoncil s chybou. Vypis chyby je vyse.
echo Okno zustava otevrene, abyste ho mohli zkopirovat.
echo ===================================================================
pause
exit /b 1

:end
exit /b 0
