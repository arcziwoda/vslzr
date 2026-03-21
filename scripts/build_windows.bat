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

:: ---- Install Python 3.12 + dependencies ----
echo Installing Python 3.12 and project dependencies...
uv python install 3.12
if %errorlevel% neq 0 (
    echo ERROR: Failed to install Python 3.12.
    pause
    exit /b 1
)
uv sync --python 3.12
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
