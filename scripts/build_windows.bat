@echo off
echo === Hue Visualizer Windows Build ===
echo.

:: ---- Check/install uv ----
where uv >nul 2>&1
if %errorlevel% neq 0 (
    echo uv not found. Installing via official installer...
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    :: Add uv to PATH for this session (installer updates registry but not current shell)
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
    where uv >nul 2>&1
    if %errorlevel% neq 0 (
        echo ERROR: uv installation failed. Please install manually:
        echo   https://docs.astral.sh/uv/getting-started/installation/
        pause
        exit /b 1
    )
    echo uv installed successfully.
) else (
    echo uv found: OK
)
echo.

:: ---- Install Python 3.12 ----
echo Installing Python 3.12...
uv python install 3.12
if %errorlevel% neq 0 (
    echo ERROR: Failed to install Python 3.12.
    pause
    exit /b 1
)

:: ---- Pin Python 3.12 and recreate venv ----
:: python-mbedtls has no wheels for 3.13+, so we MUST use 3.12
echo Pinning Python 3.12 for this project...
uv python pin 3.12
if exist .venv (
    echo Removing existing venv to ensure Python 3.12...
    rmdir /s /q .venv
)

:: ---- Install dependencies ----
echo Installing dependencies with Python 3.12...
uv sync
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)
echo.

:: ---- Build with PyInstaller ----
echo Building with PyInstaller...
uv run pyinstaller hue_visualizer.spec --clean
if %errorlevel% neq 0 (
    echo ERROR: PyInstaller build failed.
    pause
    exit /b 1
)
echo.

:: ---- Clean up pin file (don't commit it) ----
if exist .python-version del .python-version

echo === Build complete! ===
echo.
echo Output folder: dist\HueVisualizer\
echo Executable:    dist\HueVisualizer\HueVisualizer.exe
echo.
echo Next steps:
echo   1. Copy dist\HueVisualizer\ folder wherever you want
echo   2. Place your .env file inside the folder (next to HueVisualizer.exe)
echo   3. Double-click HueVisualizer.exe to run
echo   4. Right-click the tray icon (near clock) to quit
echo.
pause
