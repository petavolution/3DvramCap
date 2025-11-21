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
| `core_config.py` | Unified configuration management |
| `core_logging.py` | Structured JSON logging |
| `core_database.py` | SQLite asset tracking |
| `core_library.py` | Asset library management |
| `core_cli.py` | Unified command-line interface |
| `core_presets.py` | Export preset definitions |

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

## Infrastructure Modules

### Unified CLI (`core_cli.py`)
```bash
# Initialize a new project
python -m core_cli init my_project --profile quality

# Extract from RenderDoc capture
python -m core_cli extract scene.rdc --output export/

# Batch process multiple captures with checkpoints
python -m core_cli batch captures/ --auto-approve --report batch_report.html

# Manage asset library
python -m core_cli library import --type meshes export/meshes/
python -m core_cli library search --pattern "building*" --has-uvs

# Export with direct export (no Blender)
python -m core_cli export library/ --gltf scene.glb --direct

# Show pipeline status
python -m core_cli status
```

### Configuration (`core_config.py`)
```python
from core_config import Config, get_config, load_config

# Load default config
config = get_config()

# Apply a profile
config.apply_profile("quality")  # quality, fast, minimal, debug

# Access settings
print(config.processing.scale_factor)
print(config.export.use_unlit)

# Create and save config
config = Config()
config.paths.output_dir = "my_export"
config.processing.decimate = True
config.save("pipeline.json")
```

### Structured Logging (`core_logging.py`)
```python
from core_logging import init_logging, get_logger, TaskLogger

# Initialize logging
init_logging(log_dir="logs", level="INFO", format="json")

# Basic logging with context
logger = get_logger("extraction")
logger.info("Starting extraction", capture="scene.rdc", mesh_count=150)

# Task-scoped logging with timing
with TaskLogger("process_capture", capture_id="123") as task:
    task.info("Processing started")
    task.progress(50, "Halfway done")
    task.info("Processing complete")
# Automatically logs duration on exit
```

### Asset Database (`core_database.py`)
```python
from core_database import get_database, CaptureStatus

db = get_database("pipeline.db")

# Track a capture
capture_id = db.add_capture("scene", "/path/to/scene.rdc")
db.update_capture_status(capture_id, CaptureStatus.EXTRACTING)

# Track meshes with deduplication
mesh_id = db.add_mesh("mesh_001", capture_id,
                      geometry_hash="abc123",
                      vertex_count=1500)

# Find duplicates
existing = db.find_canonical_mesh("abc123")
if existing:
    db.mark_as_duplicate(mesh_id, existing.id)

# Get statistics
stats = db.get_stats()
print(f"Unique meshes: {stats['unique_meshes']}")
```

### Asset Library (`core_library.py`)
```python
from core_library import AssetLibrary

library = AssetLibrary("library/")

# Import meshes from extraction output
results = library.import_meshes("export/meshes/",
                                source_capture="scene.rdc",
                                tags=["environment"])

# Search assets
meshes = library.search_meshes(
    name_pattern="building.*",
    vertex_count_min=1000,
    has_uvs=True
)

# Create export set for selective export
library.create_export_set("hero_props",
                         mesh_ids=[m.id for m in meshes],
                         export_config={"quality": "high"})

# Export subset
library.export_meshes("output/", mesh_ids=["mesh_abc123"])
```

### Export Presets (`core_presets.py`)
```python
from core_presets import get_preset, list_presets, create_custom_preset

# List available presets
print(list_presets())
# ['web_optimized', 'web_draft', 'unity', 'unreal', 'blender', 'threejs', 'godot', 'archive']

# Get a preset
preset = get_preset("web_optimized")
print(preset.textures.max_size)  # 1024
print(preset.formats.gltf_draco)  # True

# Create custom preset based on existing
custom = create_custom_preset(
    "my_project",
    base="web_optimized",
    textures__max_size=2048,
    meshes__decimate=False
)
custom.save("presets/my_project.json")
```

## Complete Workflow Example

```python
from core_config import Config
from core_logging import init_logging, TaskLogger
from core_database import get_database
from core_library import AssetLibrary
from core_batch import BatchProcessor, BatchConfig
from core_quality import assess_scene_quality
from core_presets import get_preset

# 1. Initialize
init_logging(format="text", level="INFO")
config = Config()
config.apply_profile("quality")
db = get_database()

# 2. Batch process captures
batch_config = BatchConfig(
    checkpoint_after_extract=True,
    checkpoint_before_export=True,
    auto_approve_timeout=60
)

processor = BatchProcessor(batch_config)
processor.add_captures_from_directory("captures/")
results = processor.run()

# 3. Import to library
library = AssetLibrary("library/")
for task_id, result in results.items():
    if result.status.value == "completed":
        library.import_meshes(f"work/{task_id}/deduplicated/",
                             source_capture=task_id)

# 4. Quality check
report = assess_scene_quality("export/")
report.to_html("quality_report.html")
if report.metrics.score < 70:
    print("Quality below threshold!")

# 5. Export with preset
preset = get_preset("web_optimized")
# Apply preset settings to export...
```
