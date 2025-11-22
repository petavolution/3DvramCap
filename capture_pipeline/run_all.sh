#!/bin/bash
# =============================================================================
# 3DvramCap Pipeline Runner - Linux/Mac
# =============================================================================
# Usage: ./run_all.sh <capture_name>
# Example: ./run_all.sh scene_01
#
# Prerequisites:
#   - Python 3.8+ in PATH
#   - Blender in PATH
#   - Required Python packages: pip install numpy
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -z "$1" ]; then
    echo "Usage: ./run_all.sh <capture_name>"
    echo "Example: ./run_all.sh scene_01"
    echo ""
    echo "First, run RenderDoc extraction:"
    echo "  python pipeline.py extract-cmd captures/<name>.rdc"
    exit 1
fi

CAPTURE_NAME="$1"
SCENE_DIR="${SCRIPT_DIR}/export/${CAPTURE_NAME}"

echo ""
echo "=============================================================================="
echo "3DvramCap Pipeline"
echo "Scene: ${CAPTURE_NAME}"
echo "=============================================================================="

# Check if extraction exists
if [ ! -d "${SCENE_DIR}/Meshes" ]; then
    echo "[ERROR] Extraction not found: ${SCENE_DIR}/Meshes"
    echo ""
    echo "Run RenderDoc extraction first:"
    python "${SCRIPT_DIR}/pipeline.py" extract-cmd "captures/${CAPTURE_NAME}.rdc"
    exit 1
fi

# Run pipeline via unified entry point
python "${SCRIPT_DIR}/pipeline.py" run "${SCENE_DIR}" --targets "${SCRIPT_DIR}/targets"

echo ""
echo "Done!"
