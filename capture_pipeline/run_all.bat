@echo off
REM =============================================================================
REM 3DvramCap Pipeline Runner - Windows
REM =============================================================================
REM Usage: run_all.bat <capture_name>
REM Example: run_all.bat scene_01
REM
REM Prerequisites:
REM   - Python 3.8+ in PATH
REM   - Blender in PATH
REM   - Required Python packages: pip install numpy
REM =============================================================================

setlocal enabledelayedexpansion

if "%~1"=="" (
    echo Usage: run_all.bat ^<capture_name^>
    echo Example: run_all.bat scene_01
    echo.
    echo First, run RenderDoc extraction:
    echo   python pipeline.py extract-cmd captures\^<name^>.rdc
    pause
    exit /b 1
)

set CAPTURE_NAME=%~1
set SCRIPT_DIR=%~dp0
set SCENE_DIR=%~dp0export\%CAPTURE_NAME%

echo.
echo ==============================================================================
echo 3DvramCap Pipeline
echo Scene: %CAPTURE_NAME%
echo ==============================================================================

REM Check if extraction exists
if not exist "%SCENE_DIR%\Meshes" (
    echo [ERROR] Extraction not found: %SCENE_DIR%\Meshes
    echo.
    echo Run RenderDoc extraction first:
    python "%SCRIPT_DIR%pipeline.py" extract-cmd "captures\%CAPTURE_NAME%.rdc"
    pause
    exit /b 1
)

REM Run pipeline via unified entry point
python "%SCRIPT_DIR%pipeline.py" run "%SCENE_DIR%" --targets "%SCRIPT_DIR%targets"

if errorlevel 1 (
    echo [ERROR] Pipeline failed
    pause
    exit /b 1
)

echo.
echo Done!
pause
