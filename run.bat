@echo off
setlocal enabledelayedexpansion

:: --- CONFIG ---
set VENV_DIR=.venv
set SCRIPT_NAME=main.py
set WINDOW_TITLE=Chatterb0x Clipboard Reader
set REQUIREMENTS_INPUT_FILE=requirements.in
set REQUIREMENTS_LOCK_FILE=requirements.lock.txt

if not defined UV_APP_DRY set "UV_APP_DRY=0"
echo Dry Run Mode: %UV_APP_DRY%

:: --- Core Setup ---
:START
echo.
echo === Starting Chatterb0x Clipboard Reader Installer ===
echo.
where python >nul
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python is missing!
    echo Please install Python 3.11 or higher from: https://www.python.org/downloads/windows/
    echo IMPORTANT: Make sure to check "Add Python to PATH" during installation. After installing Python, re-run this script.
    pause
    exit /b 1
)
echo Checking for uv...
where uv >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo ERROR: UV is missing.
    echo Please install it first: https://github.com/astral-sh/uv#getting-started/installation
    echo If you still see "UV missing" after this, try closing and reopening this CMD window.

    echo Easiest way on Windows: pip install uv
    set /p userchoice="Do you want to auto-install uv with pip now? (Y/N): "
    if /I "%userchoice%"=="Y" (
        pip install uv
        echo.
        echo Done installing uv. Restarting the script...
        echo.
        goto START
    ) else (
        echo Please install uv manually and re-run this script.
        exit /b 1
    )
)


if not exist "%VENV_DIR%" (
    echo Creating venv...
    uv venv "%VENV_DIR%" --python python3.11 || (
        echo ERROR: Venv creation failed!
        pause
        exit /b 1
    )
)

call "%VENV_DIR%\Scripts\activate"

:: Generate lock file if it's missing or input file is newer (optional, good for dev)
:: For distribution, you'd typically commit a pre-generated lock file.
:: However, this step ensures it's up-to-date if you change requirements.txt often.
echo Ensuring lock file is up to date...
uv pip compile "%REQUIREMENTS_INPUT_FILE%" -o "%REQUIREMENTS_LOCK_FILE%" || (
    echo ERROR: Lock file generation failed!
    pause
    exit /b 1
)

:: Install from lock file
echo Installing dependencies from lock file...
uv pip sync "%REQUIREMENTS_LOCK_FILE%" || (
    echo ERROR: Dependency installation from lock file failed!
    pause
    exit /b 1
)

:: Conditional PyTorch install
if "%UV_APP_DRY%"=="0" (
    echo Installing PyTorch...
    python install_torch.py || (
        echo WARNING: PyTorch install failed. App may lack GPU support.
    )
) else (
    echo [Dry Run] Skipped PyTorch installation
)

start "%WINDOW_TITLE%" cmd /k ""%VENV_DIR%\Scripts\python.exe" "%SCRIPT_NAME%""