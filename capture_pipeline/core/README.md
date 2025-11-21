# Core Pipeline (Simplified)

Optimized, minimal scripts for the UE4 scene capture → glTF/USD export pipeline.

## Quick Start

```bash
# 1. Extract from RenderDoc capture (run in RenderDoc environment)
renderdoccmd python core_extract.py --rdc capture.rdc --out export/

# 2. Run full pipeline on extracted scene
python run_core.py --scene export/my_scene/

# Outputs:
#   targets/gltf/my_scene.glb
#   targets/usd/my_scene.usdc
#   targets/blender/my_scene.blend
```

## Scripts

| Script | Purpose | Run With |
|--------|---------|----------|
| `core_extract.py` | RenderDoc mesh/texture extraction | `renderdoccmd python` |
| `core_process.py` | Mesh deduplication | `python` |
| `core_blender.py` | Import, UV, bake, export | `blender -b --python` |
| `core_validate.py` | Export validation | `python` |
| `run_core.py` | Pipeline orchestrator | `python` |

## Pipeline Steps

```
[RenderDoc .rdc]
    ↓ core_extract.py (in RenderDoc)
[OBJ meshes + PNG textures]
    ↓ core_process.py
[Deduplicated meshes]
    ↓ core_blender.py
[glTF 2.0 + USD exports]
    ↓ core_validate.py
[Validated outputs]
```

## Individual Commands

### Step 1: RenderDoc Extraction
```bash
renderdoccmd python core_extract.py \
    --rdc path/to/capture.rdc \
    --out export/ \
    --max-meshes 500 \
    --min-vertices 50
```

### Step 2: Mesh Deduplication
```bash
python core_process.py \
    --input export/scene/Meshes/ \
    --output library/meshes/ \
    --min-vertices 50
```

### Step 3: Blender Processing
```bash
blender -b --python core_blender.py -- \
    --meshes library/meshes/ \
    --color export/scene/Textures/final_color.png \
    --gltf targets/scene.glb \
    --usd targets/scene.usdc \
    --blend targets/scene.blend \
    --scale 0.01 \
    --bake-res 2048
```

### Step 4: Validation
```bash
python core_validate.py --dir targets/
```

## Key Design Decisions

1. **Baked textures** - Camera-project captured color onto meshes. No PBR reconstruction.
2. **Post-VS geometry** - World-space meshes from RenderDoc. No transform puzzles.
3. **Conservative deduplication** - Geometry hash-based. Keeps highest-poly version.
4. **Scale factor 0.01** - UE4 centimeters → Blender meters.
5. **Unlit materials** - KHR_materials_unlit in glTF. No lighting artifacts.

## Dependencies

**Required:**
- Python 3.8+
- Blender 3.0+ (with USD support)
- RenderDoc (for extraction step)

**Optional:**
- numpy (mesh processing)
- trimesh (validation)
- OpenUSD/pxr (USD validation)
