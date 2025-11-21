# UE4 Scene Capture and Reconstruction Pipeline

A robust, minimal-error pipeline for capturing UE4 game scenes and reconstructing
them in Blender, with export to glTF 2.0 and USD formats.

## Quick Start (First Test)

### Prerequisites

1. **RenderDoc** (1.25+) - [Download](https://renderdoc.org/)
2. **Blender** (3.x+) - [Download](https://blender.org/)
3. **Universal Unreal Unlocker (UUU)** - For UE4 games - [Download](https://framedsc.com/GeneralGuides/universal_ue4_consoleunlocker.htm)
4. **Python** (3.8+) with packages:
   ```bash
   pip install numpy trimesh opencv-python PyYAML
   ```

### Step-by-Step First Test

#### 1. Capture a Frame

```bash
# Launch your UE4 game through RenderDoc
# OR attach RenderDoc to running game

# Once in-game:
# - Use UUU to freeze time (timestop) and enable free camera
# - Position camera at desired view
# - Press F12 to capture a frame
# - Save the .rdc file to captures/ folder
```

#### 2. Extract from RenderDoc (Manual)

Open RenderDoc with your .rdc capture:

1. Go to last event in Event Browser
2. In Texture Viewer, save final color as `captures/final.png`
3. For a few big draw calls (walls, floor):
   - Open Mesh Viewer
   - Select "Post-VS" view
   - Export Mesh → OBJ → save to `export/Meshes/`

#### 3. Run the Pipeline

```bash
# Deduplicate meshes
python scripts/02_dedupe_meshes.py export/Meshes library/index.json

# Import to Blender and bake
blender -b --python scripts/03_blender_import_bake.py -- \
  --meshes library/meshes \
  --color captures/final.png \
  --out targets/blender/scene.blend

# Export to glTF and USD
blender -b targets/blender/scene.blend --python scripts/04_blender_export.py -- \
  --gltf targets/gltf/scene.glb \
  --usd targets/usd/scene.usdc
```

#### 4. Verify Results

Open `targets/blender/scene.blend` in Blender to check:
- Meshes are visible and aligned
- Scale looks reasonable (1 Blender unit ≈ 1 meter)
- Baked colors appear on geometry

## Project Structure

```
capture_pipeline/
├── captures/              # Input: .rdc files, reference images
├── export/                # Intermediate: extracted meshes/textures
│   └── <capture_name>/
│       ├── Meshes/        # Post-VS OBJ files
│       ├── Textures/      # final.png, gbuffer_*.png
│       └── scene.json     # Extraction index
├── library/               # Deduplicated canonical meshes
│   └── meshes/
├── scene/                 # Canonical scene files
├── targets/
│   ├── blender/           # .blend files
│   ├── gltf/              # .glb/.gltf exports
│   └── usd/               # .usdc exports
├── qa/                    # Validation reports
├── scripts/               # Python scripts
│   ├── 01_extract_from_rdc.py
│   ├── 02_dedupe_meshes.py
│   ├── 03_blender_import_bake.py
│   ├── 04_blender_export.py
│   ├── 05_qa_validate.py
│   └── run_pipeline.py
├── config.yaml            # Pipeline configuration
└── README.md
```

## Pipeline Steps in Detail

### Step 1: RenderDoc Extraction

Extracts post-VS (world-space) geometry and textures from .rdc captures.

```bash
# Run inside RenderDoc's Python environment:
renderdoccmd python scripts/01_extract_from_rdc.py \
  --rdc captures/scene.rdc \
  --out export
```

**Output:**
- `Meshes/*.obj` - Post-VS geometry (already in world space)
- `Textures/final.png` - Final color buffer
- `Textures/gbuffer_*.png` - G-buffer targets (if deferred)
- `scene.json` - Index of extracted assets

### Step 2: Mesh Deduplication

Removes duplicate geometry using geometry hashing.

```bash
python scripts/02_dedupe_meshes.py \
  export/scene/Meshes \
  library/index.json \
  --copy-to library/meshes
```

**Key Features:**
- Conservative deduplication (only exact matches)
- LOD filtering (keeps highest-poly at each location)
- Preserves UV seams (face-corner UVs)

### Step 3: Blender Import and Bake

Imports meshes, creates materials, and bakes captured color.

```bash
blender -b --python scripts/03_blender_import_bake.py -- \
  --meshes library/meshes \
  --color export/scene/Textures/final.png \
  --out targets/blender/scene.blend \
  --bake-res 2048 \
  --scale 0.01
```

**Options:**
- `--meshes DIR` - Directory with OBJ files
- `--color PATH` - Color image to bake (final.png or gbuffer_0.png)
- `--camera PATH` - Camera JSON (optional)
- `--bake-res N` - Bake texture resolution (default: 2048)
- `--scale N` - Scale factor (0.01 = UE4 cm → Blender m)
- `--no-bake` - Skip baking, just import geometry

### Step 4: Export

Exports to glTF 2.0 and USD formats.

```bash
blender -b scene.blend --python scripts/04_blender_export.py -- \
  --gltf targets/gltf/scene.glb \
  --usd targets/usd/scene.usdc \
  --unlit \
  --embed
```

**Options:**
- `--gltf PATH` - Export to glTF (.glb or .gltf)
- `--usd PATH` - Export to USD (.usdc or .usda)
- `--unlit` - Use unlit materials (baked look)
- `--embed` - Embed textures in GLB

### Step 5: QA Validation

Validates reconstruction quality.

```bash
python scripts/05_qa_validate.py \
  captures/reference.png \
  qa/rendered.png \
  qa/report.json
```

**Metrics:**
- **Silhouette IoU** (≥0.92): Edge-based geometry match
- **SSIM** (≥0.80): Structural similarity

## Key Concepts

### Post-VS Mesh Export
Geometry exported **after the vertex shader** is already transformed to world
space. This eliminates transform puzzles and ensures meshes align correctly.

### Baked Color
Instead of reconstructing complex PBR materials, we bake the captured final
color onto the geometry. This gives a "what you saw" look that always imports
correctly.

### Unit Conversion
- **UE4**: 1 Unreal Unit = 1 centimeter
- **Blender**: 1 unit = 1 meter
- **Scale factor**: 0.01 (UE4 cm → Blender m)

### Coordinate Systems
- **UE4**: Left-handed, Z-up
- **Blender**: Right-handed, Z-up
- **glTF**: Right-handed, Y-up
- Exporters handle conversion automatically

## Configuration

Edit `config.yaml` to customize:

```yaml
# Units
units:
  scale_factor: 0.01  # UE4 cm -> Blender m

# Baking
bake:
  resolution: 2048
  uv_margin: 0.02

# QA thresholds
qa:
  silhouette_iou_threshold: 0.92
  ssim_threshold: 0.80
```

## Troubleshooting

### Meshes appear mirrored
- Check coordinate system conversion in import
- Try flipping scale on one axis and recalculating normals

### Scale is wrong
- Verify `--scale` argument (default 0.01)
- Check if your UE4 project uses non-standard units

### Baked colors look washed out
- Ensure you're baking from the correct image (final.png or gbuffer_0.png)
- Check if the game uses HDR/tonemapping that affects the capture

### RenderDoc won't attach
- Try launching game through RenderDoc instead of attaching
- Disable overlays (Steam, GeForce Experience)
- Use DX11 mode if available (`-d3d11` launch option)

### QA validation fails
- **Low Silhouette IoU**: Camera FOV mismatch - adjust camera settings
- **Low SSIM**: Colors differ - normal for baked vs original lighting

## License

This pipeline is provided for educational and personal use. Be mindful of
game asset usage terms when extracting content from commercial games.
