@echo off
setlocal EnableDelayedExpansion
:: =============================================================================
:: Project — Windows launcher
:: Installs uv (if missing), downloads Python, syncs dependencies,
:: then starts the application.
:: Usage:  setup.bat [app arguments...]
:: =============================================================================

set UV_PYTHON=3.10
set SCRIPT_DIR=%~dp0

:: 1. Ensure uv is available
where uv >nul 2>&1
if %errorlevel% neq 0 (
    echo [project] uv not found -- installing...
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "irm https://astral.sh/uv/install.ps1 | iex"
    :: Refresh PATH from registry so we can find uv immediately
    for /f "tokens=*" %%i in ('powershell -NoProfile -Command ^
        "[System.Environment]::GetEnvironmentVariable(\"PATH\",\"User\")"') do (
        set "PATH=%%i;%PATH%"
    )
    where uv >nul 2>&1
    if !errorlevel! neq 0 (
        echo [project] ERROR: uv installation failed. Please install manually:
        echo               https://docs.astral.sh/uv/getting-started/installation/
        pause
        exit /b 1
    )
    for /f "delims=" %%v in ('uv --version') do echo [project] uv installed: %%v
) else (
    for /f "delims=" %%v in ('uv --version') do echo [project] uv found: %%v
)

:: 2. Ensure Python is available
cd /d "%SCRIPT_DIR%"
echo [project] Checking Python %UV_PYTHON%...
:: uv python install %UV_PYTHON% --quiet
uv python install %UV_PYTHON%

:: 3. Sync dependencies (create / update .venv)
echo [project] Syncing dependencies...
:: uv sync --quiet
uv sync

:: 4. Run the example
:: echo [project] Starting example...
:: uv run python .\examples\cross_framework.py %*
