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
| `core_materials.py` | Material/texture extraction from RenderDoc |
| `core_optimize.py` | Mesh decimation, cleanup, LOD generation |
| `core_gltf.py` | Direct glTF 2.0 export (no Blender) |
| `core_merge.py` | Mesh merging utilities |
| `core_usd.py` | Direct USD/USDA export (no Blender) |
| `core_texture.py` | DDS reading, format conversion, processing |
| `core_ninja.py` | Ninja Ripper .rip file parser (DX9 fallback) |
| `core_depth.py` | ReShade depth buffer integration |
| `core_quality.py` | Quality metrics and validation reporting |
| `core_batch.py` | Batch processing with human checkpoints |

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

### Batch Processing (`core_batch.py`)
```python
from core_batch import BatchProcessor, BatchConfig

# Configure batch processing
config = BatchConfig(
    checkpoint_after_extract=True,
    checkpoint_before_export=True,
    auto_approve_timeout=60  # Auto-approve after 60 seconds
)

# Process multiple captures
processor = BatchProcessor(config)
processor.add_captures_from_directory("captures/")
results = processor.run(auto_approve=False)  # Manual review at checkpoints

# Generate report
processor.generate_report("batch_report.html")
```

### Direct glTF Export (`core_gltf.py`)
```python
from core_gltf import GltfBuilder

# Build glTF without Blender
builder = GltfBuilder()
builder.add_mesh("mesh1", vertices, faces, normals, uvs)
builder.add_unlit_material("mat1", texture_path="diffuse.png")
builder.assign_material("mesh1", "mat1")
builder.save("output.glb")  # Binary glTF
```

### Direct USD Export (`core_usd.py`)
```python
from core_usd import UsdBuilder

# Build USD without Blender or OpenUSD
builder = UsdBuilder(up_axis="Y", scale=1.0)
builder.add_mesh("mesh1", vertices, faces, normals, uvs)
builder.add_unlit_material("mat1", texture_path="diffuse.png")
builder.save("output.usda")  # ASCII USD (no dependencies)
```

### Ninja Ripper Support (`core_ninja.py`)
```python
from core_ninja import parse_rip_directory, batch_convert_rips

# Parse Ninja Ripper output (DX9 fallback)
meshes = parse_rip_directory("ninja_output/", min_vertices=50)

# Convert to OBJ
batch_convert_rips("ninja_output/", "obj_export/", merge=False)
```

### Depth Buffer Processing (`core_depth.py`)
```python
from core_depth import DepthProcessor, CameraParams

# Load ReShade depth capture
camera = CameraParams(fov_y=90, near_plane=10, far_plane=100000)
processor = DepthProcessor("depth.png", camera)
processor.linearize()

# Generate point cloud
points = processor.to_point_cloud(subsample=4)

# Check visibility of mesh vertices
visible = processor.compute_visibility(world_positions, tolerance=10.0)
```

### Texture Processing (`core_texture.py`)
```python
from core_texture import TextureProcessor, convert_texture, batch_convert_textures

# Convert DDS to PNG
convert_texture("input.dds", "output.png", max_size=2048)

# Process texture
processor = TextureProcessor("normal.dds")
processor.flip_green_channel()  # DX -> OpenGL normals
processor.normalize_normal_map()
processor.save("normal_fixed.png")

# Batch convert
batch_convert_textures("textures/", "converted/", output_format=".png")
```

### Quality Assessment (`core_quality.py`)
```python
from core_quality import QualityChecker, assess_scene_quality

# Full scene assessment
report = assess_scene_quality("export/my_scene/")
report.print_summary()
report.to_html("quality_report.html")

# Quick sanity check
checker = QualityChecker("export/")
status = checker.quick_check()
print(f"Has meshes: {status['has_meshes']}")
print(f"glTF valid: {status['has_gltf']}")
```

## Dependencies

**Required:**
- Python 3.8+
- Blender 3.0+ (with USD support)
- RenderDoc (for extraction step)

**Optional:**
- numpy (mesh processing, camera math)
- trimesh (validation, geometry analysis)
- PIL/Pillow (texture atlas, image processing)
- OpenUSD/pxr (USD binary export, validation)
- pygltflib (direct glTF export)
- Open3D (mesh decimation)
- opencv-python (depth filtering)

## Architecture

```
                     ┌─────────────────────────────────────┐
                     │         core_batch.py               │
                     │   (Human-supervised orchestration)  │
                     └─────────────────┬───────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        │                              │                              │
        ▼                              ▼                              ▼
┌───────────────┐            ┌───────────────┐            ┌───────────────┐
│ RenderDoc     │            │ Ninja Ripper  │            │ ReShade       │
│ core_extract  │            │ core_ninja    │            │ core_depth    │
└───────┬───────┘            └───────┬───────┘            └───────┬───────┘
        │                            │                            │
        └────────────────────────────┼────────────────────────────┘
                                     │
                                     ▼
                          ┌───────────────────┐
                          │   core_process    │
                          │  (Deduplication)  │
                          └─────────┬─────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────┐          ┌───────────────┐          ┌───────────────┐
│ core_blender  │          │  core_gltf    │          │  core_usd     │
│ (Full export) │          │ (Direct glTF) │          │ (Direct USD)  │
└───────┬───────┘          └───────┬───────┘          └───────┬───────┘
        │                          │                          │
        └──────────────────────────┼──────────────────────────┘
                                   │
                                   ▼
                        ┌───────────────────┐
                        │  core_quality     │
                        │   (Validation)    │
                        └───────────────────┘
```
