#!/bin/bash
# =============================================================================
# UE4 Scene Capture Pipeline - Linux/Mac Runner
# =============================================================================
# Usage: ./run_all.sh <capture_name>
# Example: ./run_all.sh scene_01
#
# Prerequisites:
#   - Python 3.8+ in PATH
#   - Blender in PATH
#   - Required Python packages installed
# =============================================================================

set -e

# Check arguments
if [ -z "$1" ]; then
    echo "Usage: ./run_all.sh <capture_name>"
    echo "Example: ./run_all.sh scene_01"
    exit 1
fi

CAPTURE_NAME="$1"
SCRIPT_DIR="$(dirname "$0")/scripts"
EXPORT_DIR="$(dirname "$0")/export/${CAPTURE_NAME}"
MESH_DIR="${EXPORT_DIR}/Meshes"
TEX_DIR="${EXPORT_DIR}/Textures"
LIBRARY_DIR="$(dirname "$0")/library"
TARGETS_DIR="$(dirname "$0")/targets"

echo ""
echo "=============================================================================="
echo "UE4 Scene Capture Pipeline"
echo "Capture: ${CAPTURE_NAME}"
echo "=============================================================================="
echo ""

# Check if extraction exists
if [ ! -f "${EXPORT_DIR}/scene.json" ]; then
    echo "[WARNING] Extraction not found for ${CAPTURE_NAME}"
    echo ""
    echo "Please run RenderDoc extraction first:"
    echo "  renderdoccmd python ${SCRIPT_DIR}/01_extract_from_rdc.py \\"
    echo "    --rdc captures/${CAPTURE_NAME}.rdc \\"
    echo "    --out export"
    echo ""
    echo "Or manually export meshes to: ${MESH_DIR}"
    echo "And save final.png to: ${TEX_DIR}/final.png"
    exit 1
fi

# Step 2: Deduplicate meshes
echo ""
echo "[Step 2/4] Deduplicating meshes..."
python "${SCRIPT_DIR}/02_dedupe_meshes.py" \
    "${MESH_DIR}" \
    "${LIBRARY_DIR}/${CAPTURE_NAME}_index.json" \
    --copy-to "${LIBRARY_DIR}/meshes"

# Step 3: Blender import and bake
echo ""
echo "[Step 3/4] Importing to Blender and baking..."
BLEND_OUT="${TARGETS_DIR}/blender/${CAPTURE_NAME}.blend"

# Find color image
COLOR_IMG="${TEX_DIR}/final.png"
if [ ! -f "${COLOR_IMG}" ]; then
    COLOR_IMG="${TEX_DIR}/gbuffer_0.png"
fi

# Create output directories
mkdir -p "${TARGETS_DIR}/blender"
mkdir -p "${TARGETS_DIR}/gltf"
mkdir -p "${TARGETS_DIR}/usd"

if [ -f "${COLOR_IMG}" ]; then
    blender -b --python "${SCRIPT_DIR}/03_blender_import_bake.py" -- \
        --meshes "${LIBRARY_DIR}/meshes" \
        --color "${COLOR_IMG}" \
        --out "${BLEND_OUT}" \
        --bake-res 2048 \
        --scale 0.01
else
    echo "[WARNING] No color image found, importing without baking"
    blender -b --python "${SCRIPT_DIR}/03_blender_import_bake.py" -- \
        --meshes "${LIBRARY_DIR}/meshes" \
        --out "${BLEND_OUT}" \
        --scale 0.01 \
        --no-bake
fi

# Step 4: Export glTF and USD
echo ""
echo "[Step 4/4] Exporting to glTF and USD..."
GLTF_OUT="${TARGETS_DIR}/gltf/${CAPTURE_NAME}.glb"
USD_OUT="${TARGETS_DIR}/usd/${CAPTURE_NAME}.usdc"

blender -b "${BLEND_OUT}" --python "${SCRIPT_DIR}/04_blender_export.py" -- \
    --gltf "${GLTF_OUT}" \
    --usd "${USD_OUT}"

echo ""
echo "=============================================================================="
echo "Pipeline Complete!"
echo "=============================================================================="
echo ""
echo "Outputs:"
echo "  Blender: ${BLEND_OUT}"
echo "  glTF:    ${GLTF_OUT}"
echo "  USD:     ${USD_OUT}"
echo ""
