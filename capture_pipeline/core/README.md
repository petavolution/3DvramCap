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

### Pipeline Scripts
| Script | Purpose | Run With |
|--------|---------|----------|
| `core_extract.py` | RenderDoc mesh/texture extraction | `renderdoccmd python` |
| `core_process.py` | Mesh deduplication | `python` |
| `core_blender.py` | Import, UV, bake, export | `blender -b --python` |
| `core_validate.py` | Export validation | `python` |
| `run_core.py` | Pipeline orchestrator | `python` |

### Utility Modules
| Module | Purpose |
|--------|---------|
| `core_vertex_decode.py` | Robust RenderDoc vertex attribute parsing |
| `core_camera.py` | Camera matrix extraction & coordinate transforms |
| `core_atlas.py` | Texture atlas packing (MaxRects/Shelf) |
| `core_scene.py` | Scene hierarchy reconstruction |
| `core_types.py` | Shared data structures & type definitions |

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

## Advanced Features

### Vertex Decoding (`core_vertex_decode.py`)
```python
from core_vertex_decode import VertexDecoder, decode_post_vs_mesh

# In RenderDoc environment
decoder = VertexDecoder(controller)
mesh = decoder.decode_draw_call(event_id)
mesh.to_obj_data()  # Ready for export
```

### Camera Projection (`core_camera.py`)
```python
from core_camera import (
    ue4_to_blender_position,
    ndc_to_world,
    project_texture_uv
)

# Convert UE4 position to Blender
blender_pos = ue4_to_blender_position([100, 200, 50])

# Project world position to texture UV
u, v = project_texture_uv(world_pos, view_projection_matrix, 1920, 1080)
```

### Texture Atlas (`core_atlas.py`)
```python
from core_atlas import AtlasPacker, pack_textures

# Pack multiple textures into atlas
packer = AtlasPacker(4096, 4096)
for tex in texture_files:
    packer.add_texture_file(tex)
result = packer.pack()
result.atlas_image.save("atlas.png")

# Remap UVs for packed textures
remapped = remap_mesh_uvs(original_uvs, result.uv_regions["texture_name"])
```

### Scene Reconstruction (`core_scene.py`)
```python
from core_scene import SceneBuilder, reconstruct_scene

# Build scene hierarchy from extracted meshes
scene = reconstruct_scene("library/mesh_index.json", "scene.json")
print(f"Clusters: {scene.stats['clusters']}")
print(f"Instances: {scene.stats['instance_groups']}")
```

## Dependencies

**Required:**
- Python 3.8+
- Blender 3.0+ (with USD support)
- RenderDoc (for extraction step)

**Optional:**
- numpy (mesh processing, camera math)
- trimesh (validation, geometry analysis)
- PIL/Pillow (texture atlas)
- OpenUSD/pxr (USD validation)
