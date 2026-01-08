@echo off
REM Build script for creating Windows standalone package
REM This script builds ChemGraph as a standalone executable using PyInstaller
REM Following Windows packaging best practices

setlocal enabledelayedexpansion

echo ========================================
echo ChemGraph Windows Build Script
echo ========================================
echo.

REM Check if Python is available (try both 'python' and 'py' commands)
set PYTHON_CMD=
python --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=python
) else (
    py --version >nul 2>&1
    if not errorlevel 1 (
        set PYTHON_CMD=py
    )
)

if "%PYTHON_CMD%"=="" (
    echo ERROR: Python is not installed or not in PATH
    echo.
    echo Python was not found. Please:
    echo 1. Install Python 3.10 or higher from https://www.python.org/downloads/
    echo 2. During installation, check "Add Python to PATH"
    echo 3. Restart Command Prompt after installation
    echo.
    echo Alternatively, if Python is installed but not in PATH:
    echo - Use the full path to python.exe in the script
    echo - Or add Python to your system PATH environment variable
    echo.
    pause
    exit /b 1
)

echo [1/6] Checking Python version...
for /f "tokens=2" %%i in ('%PYTHON_CMD% --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Python version: !PYTHON_VERSION!
echo Using Python command: %PYTHON_CMD%

REM Validate Python version (must be 3.10+)
%PYTHON_CMD% -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" 2>nul
if errorlevel 1 (
    echo ERROR: Python 3.10 or higher is required
    echo Current version: !PYTHON_VERSION!
    echo.
    echo Please install Python 3.10 or higher from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo.
echo [2/6] Installing/upgrading build dependencies...
%PYTHON_CMD% -m pip install --upgrade pip --quiet
if errorlevel 1 (
    echo ERROR: Failed to upgrade pip
    pause
    exit /b 1
)

%PYTHON_CMD% -m pip install "pyinstaller>=5.13.0" --quiet
if errorlevel 1 (
    echo ERROR: Failed to install PyInstaller
    pause
    exit /b 1
)

echo PyInstaller installed successfully

echo.
echo [3/6] Getting version information...
REM Get project root (two levels up from script directory)
cd /d "%~dp0\..\.."
for /f "delims=" %%v in ('%PYTHON_CMD% windows_packaging\scripts\get_version.py') do set VERSION=%%v
echo Building version: !VERSION!

echo.
echo [4/6] Installing ChemGraph and dependencies...
%PYTHON_CMD% -m pip install -e . --quiet
if errorlevel 1 (
    echo ERROR: Failed to install ChemGraph
    pause
    exit /b 1
)

echo.
echo [5/6] Building standalone executable with PyInstaller...
REM Ensure we're in project root
cd /d "%~dp0\..\.."
if not exist "windows_packaging\dist" mkdir "windows_packaging\dist"
if not exist "windows_packaging\build" mkdir "windows_packaging\build"

cd windows_packaging
%PYTHON_CMD% -m PyInstaller --clean --noconfirm config\chemgraph.spec

if errorlevel 1 (
    echo.
    echo ERROR: PyInstaller build failed!
    echo.
    echo Troubleshooting:
    echo 1. Check that all dependencies are installed
    echo 2. Review the error messages above
    echo 3. Try running: %PYTHON_CMD% -m pip install -e .
    echo.
    pause
    exit /b 1
)

echo.
echo [6/6] Verifying build...
if not exist "dist\chemgraph\chemgraph.exe" (
    echo ERROR: Executable not found after build!
    pause
    exit /b 1
)

echo.
echo ========================================
echo Build completed successfully!
echo ========================================
echo.
echo Version: !VERSION!
echo Executable location: windows_packaging\dist\chemgraph\chemgraph.exe
echo.
echo Next steps:
echo 1. Test the executable: windows_packaging\dist\chemgraph\chemgraph.exe --help
echo 2. Create installer: Run scripts\build_installer.bat
echo.
pause

