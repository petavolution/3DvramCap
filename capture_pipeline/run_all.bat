@echo off
REM =============================================================================
REM UE4 Scene Capture Pipeline - Windows Batch Runner
REM =============================================================================
REM Usage: run_all.bat <capture_name>
REM Example: run_all.bat scene_01
REM
REM Prerequisites:
REM   - Python 3.8+ in PATH
REM   - Blender in PATH
REM   - Required Python packages installed
REM =============================================================================

setlocal enabledelayedexpansion

REM Check arguments
if "%~1"=="" (
    echo Usage: run_all.bat ^<capture_name^>
    echo Example: run_all.bat scene_01
    exit /b 1
)

set CAPTURE_NAME=%~1
set SCRIPT_DIR=%~dp0scripts
set EXPORT_DIR=%~dp0export\%CAPTURE_NAME%
set MESH_DIR=%EXPORT_DIR%\Meshes
set TEX_DIR=%EXPORT_DIR%\Textures
set LIBRARY_DIR=%~dp0library
set TARGETS_DIR=%~dp0targets

echo.
echo ==============================================================================
echo UE4 Scene Capture Pipeline
echo Capture: %CAPTURE_NAME%
echo ==============================================================================
echo.

REM Check if extraction exists
if not exist "%EXPORT_DIR%\scene.json" (
    echo [WARNING] Extraction not found for %CAPTURE_NAME%
    echo.
    echo Please run RenderDoc extraction first:
    echo   renderdoccmd python %SCRIPT_DIR%\01_extract_from_rdc.py ^
    echo     --rdc captures\%CAPTURE_NAME%.rdc ^
    echo     --out export
    echo.
    echo Or manually export meshes to: %MESH_DIR%
    echo And save final.png to: %TEX_DIR%\final.png
    echo.
    pause
    exit /b 1
)

REM Step 2: Deduplicate meshes
echo.
echo [Step 2/4] Deduplicating meshes...
python "%SCRIPT_DIR%\02_dedupe_meshes.py" "%MESH_DIR%" "%LIBRARY_DIR%\%CAPTURE_NAME%_index.json" --copy-to "%LIBRARY_DIR%\meshes"
if errorlevel 1 (
    echo [ERROR] Deduplication failed
    pause
    exit /b 1
)

REM Step 3: Blender import and bake
echo.
echo [Step 3/4] Importing to Blender and baking...
set BLEND_OUT=%TARGETS_DIR%\blender\%CAPTURE_NAME%.blend
set COLOR_IMG=%TEX_DIR%\final.png

if not exist "%COLOR_IMG%" (
    set COLOR_IMG=%TEX_DIR%\gbuffer_0.png
)

if exist "%COLOR_IMG%" (
    blender -b --python "%SCRIPT_DIR%\03_blender_import_bake.py" -- ^
        --meshes "%LIBRARY_DIR%\meshes" ^
        --color "%COLOR_IMG%" ^
        --out "%BLEND_OUT%" ^
        --bake-res 2048 ^
        --scale 0.01
) else (
    echo [WARNING] No color image found, importing without baking
    blender -b --python "%SCRIPT_DIR%\03_blender_import_bake.py" -- ^
        --meshes "%LIBRARY_DIR%\meshes" ^
        --out "%BLEND_OUT%" ^
        --scale 0.01 ^
        --no-bake
)

if errorlevel 1 (
    echo [ERROR] Blender import failed
    pause
    exit /b 1
)

REM Step 4: Export glTF and USD
echo.
echo [Step 4/4] Exporting to glTF and USD...
set GLTF_OUT=%TARGETS_DIR%\gltf\%CAPTURE_NAME%.glb
set USD_OUT=%TARGETS_DIR%\usd\%CAPTURE_NAME%.usdc

blender -b "%BLEND_OUT%" --python "%SCRIPT_DIR%\04_blender_export.py" -- ^
    --gltf "%GLTF_OUT%" ^
    --usd "%USD_OUT%"

if errorlevel 1 (
    echo [ERROR] Export failed
    pause
    exit /b 1
)

echo.
echo ==============================================================================
echo Pipeline Complete!
echo ==============================================================================
echo.
echo Outputs:
echo   Blender: %BLEND_OUT%
echo   glTF:    %GLTF_OUT%
echo   USD:     %USD_OUT%
echo.

pause
