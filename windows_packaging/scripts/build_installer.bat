@echo off
REM Build script for creating Windows installer using NSIS

echo ========================================
echo ChemGraph Windows Installer Builder
echo ========================================
echo.

REM Check if NSIS is installed
where makensis >nul 2>&1
if errorlevel 1 (
    echo ERROR: NSIS (Nullsoft Scriptable Install System) is not installed or not in PATH
    echo.
    echo Please install NSIS from: https://nsis.sourceforge.io/Download
    echo After installation, make sure makensis.exe is in your PATH
    echo.
    pause
    exit /b 1
)

echo [1/3] Checking NSIS installation...
makensis /VERSION

echo.
echo [2/3] Checking if PyInstaller build exists...
cd /d "%~dp0\.."
if not exist "dist\chemgraph\chemgraph.exe" (
    echo ERROR: PyInstaller build not found!
    echo Please run scripts\build_windows.bat first to create the executable.
    echo.
    pause
    exit /b 1
)

echo Build found: dist\chemgraph\chemgraph.exe
echo.

REM Check if EnVar plugin is available (for PATH modification)
if not exist "%NSISDIR%\Plugins\x86-unicode\EnVar.dll" (
    echo WARNING: EnVar plugin not found. PATH modification will be skipped.
    echo To enable PATH modification, install EnVar plugin:
    echo https://nsis.sourceforge.io/EnVar_plug-in
    echo.
)

echo [3/4] Getting version information...
cd /d "%~dp0\.."
for /f "delims=" %%v in ('python windows_packaging\scripts\get_version.py') do set VERSION=%%v
echo Building installer for version: !VERSION!

echo.
echo [4/4] Building installer...
cd windows_packaging
makensis /DVERSION=!VERSION! config\build_installer.nsi

if errorlevel 1 (
    echo.
    echo ERROR: Installer build failed!
    echo.
    echo Troubleshooting:
    echo 1. Verify NSIS is installed correctly
    echo 2. Check that dist\chemgraph\chemgraph.exe exists
    echo 3. Review NSIS error messages above
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Installer created successfully!
echo ========================================
echo.
echo Version: !VERSION!
echo Installer location: windows_packaging\ChemGraph-Setup-!VERSION!.exe
echo.
echo You can now distribute this installer to Windows users.
echo.
pause

