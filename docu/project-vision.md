# 3DvramCap: Project Vision & Architecture

## Executive Summary

**3DvramCap** is a GPU VRAM capture-to-export pipeline designed to extract, reconstruct, and export 3D scene geometry from Unreal Engine 4 games. The project transforms real-time rendered frames captured via graphics debuggers into portable, standard 3D formats (glTF 2.0, USD, Blender) suitable for archival, analysis, modding, and creative reuse.

### Core Value Proposition

Transform what you see on screen into portable 3D assets—extracting post-transformed geometry with baked appearance, bypassing the complexity of reverse-engineering proprietary shaders and material systems.

---

## Project Goals

### Primary Objectives

1. **Accurate Geometry Extraction**: Capture post-vertex-shader (Post-VS) mesh data from RenderDoc captures, preserving world-space positions that eliminate transform reconstruction puzzles.

2. **Visual Fidelity Preservation**: Bake the captured final color buffer onto geometry, creating a "what you saw" representation that imports correctly everywhere without shader dependencies.

3. **Standard Format Export**: Output to industry-standard formats (glTF 2.0, USD) ensuring maximum interoperability with 3D software, game engines, and web viewers.

4. **Minimal-Error Pipeline**: Prioritize robustness and correctness over feature completeness. Every step should produce valid, verifiable output.

5. **Human-in-the-Loop Quality**: Enable batch processing with checkpoint-based review, ensuring quality control while maintaining automation benefits.

### Design Philosophy

| Principle | Implementation |
|-----------|----------------|
| **Simplicity over complexity** | Baked textures instead of PBR reconstruction |
| **Correctness over speed** | Conservative deduplication, validation at every step |
| **Portability over fidelity** | Standard formats, no proprietary dependencies |
| **Visibility over automation** | Checkpoint system, structured logging, QA reports |

---

## System Architecture

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          INPUT SOURCES                                   │
├─────────────────┬───────────────────┬──────────────────────────────────┤
│   RenderDoc     │   Ninja Ripper    │   ReShade Depth                  │
│   (.rdc)        │   (.rip)          │   (depth.png)                    │
│   [Primary]     │   [DX9 Fallback]  │   [Supplementary]                │
└────────┬────────┴─────────┬─────────┴──────────────┬───────────────────┘
         │                  │                        │
         ▼                  ▼                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       EXTRACTION LAYER                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │  core_extract   │  │   core_ninja    │  │   core_depth    │         │
│  │  (RenderDoc API)│  │   (.rip parser) │  │   (depth→3D)    │         │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘         │
└───────────┼────────────────────┼────────────────────┼───────────────────┘
            │                    │                    │
            └────────────────────┴────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       PROCESSING LAYER                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │  core_process   │  │  core_materials │  │  core_texture   │         │
│  │  (Deduplicate)  │  │  (PBR classify) │  │  (DDS→PNG)      │         │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘         │
│           │                    │                    │                   │
│           └────────────────────┴────────────────────┘                   │
│                                │                                        │
│                    ┌───────────┴───────────┐                           │
│                    │     core_scene        │                           │
│                    │  (Hierarchy rebuild)  │                           │
│                    └───────────┬───────────┘                           │
└───────────────────────────────┼─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        EXPORT LAYER                                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │  core_blender   │  │   core_gltf     │  │   core_usd      │         │
│  │  (Full export)  │  │  (Direct GLB)   │  │  (Direct USDA)  │         │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘         │
└───────────┼────────────────────┼────────────────────┼───────────────────┘
            │                    │                    │
            ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       OUTPUT FORMATS                                     │
│         .blend              .glb/.gltf            .usdc/.usda           │
│        (Native)             (Web/Game)            (VFX/Film)            │
└─────────────────────────────────────────────────────────────────────────┘
```

### Module Responsibility Matrix

| Module | Responsibility | Dependencies | Output |
|--------|---------------|--------------|--------|
| `core_extract` | RenderDoc capture → OBJ + textures | RenderDoc API | `scene.json`, `*.obj`, `*.png` |
| `core_process` | Geometry deduplication & validation | numpy | `mesh_index.json`, unique `*.obj` |
| `core_vertex_decode` | Robust vertex attribute parsing | struct | Decoded mesh data |
| `core_camera` | Coordinate transforms & projection | numpy | Camera params, UV projection |
| `core_materials` | PBR texture classification | PIL | Material definitions |
| `core_texture` | DDS decompression, format conversion | PIL | PNG textures |
| `core_atlas` | Texture atlas packing | PIL | Combined atlas images |
| `core_scene` | Hierarchy reconstruction | - | Scene graph |
| `core_blender` | Full Blender processing pipeline | bpy | `.blend`, `.glb`, `.usdc` |
| `core_gltf` | Direct glTF 2.0 export | pygltflib | `.glb`, `.gltf` |
| `core_usd` | Direct USD export | (optional pxr) | `.usda`, `.usdc` |
| `core_validate` | Export integrity verification | trimesh | Validation reports |
| `core_quality` | QA metrics (SSIM, IoU) | opencv | Quality reports |
| `core_batch` | Human-supervised batch processing | - | Batch results |

---

## Key Technical Decisions

### 1. Post-VS Geometry Extraction

**Decision**: Extract geometry after the vertex shader stage, not from raw vertex buffers.

**Rationale**:
- Vertices are already transformed to world space
- Eliminates need to reverse-engineer model/view/projection matrices
- Meshes align correctly without transform puzzles
- Handles instancing automatically (each instance is separate geometry)

**Trade-off**: Higher mesh count, potential duplicates (mitigated by deduplication).

### 2. Baked Color Over PBR Reconstruction

**Decision**: Project captured final color onto mesh UVs rather than reconstructing PBR materials.

**Rationale**:
- UE4 materials use complex shader graphs, often with custom nodes
- Texture streaming means not all textures are in VRAM
- Lighting is pre-baked in the capture—reproducing it requires scene understanding
- Baked color always imports correctly regardless of target engine

**Trade-off**: Static appearance, no relighting capability, single viewing angle per bake.

### 3. Conservative Deduplication

**Decision**: Only deduplicate exact geometric matches using normalized geometry hashing.

**Rationale**:
- Aggressive deduplication risks removing intentionally repeated geometry
- LOD meshes at same location need careful handling (keep highest-poly)
- False negatives (missing duplicates) are safer than false positives (missing geometry)

**Algorithm**:
```python
# Normalize to unit cube → Round to precision → Sort vertices → Hash
hash = MD5(sorted(round(normalize(vertices), 4)))
```

### 4. Unit and Coordinate System Handling

**Decision**: Explicit, well-documented coordinate transforms at system boundaries.

| System | Handedness | Up | Forward | Units |
|--------|------------|-----|---------|-------|
| UE4 | Left | Z | X | cm |
| Blender | Right | Z | Y | m |
| glTF | Right | Y | Z | m |
| USD | Right | Y | - | configurable |

**Scale Factor**: 0.01 (UE4 cm → Blender/glTF m)

**Y-Axis Flip**: Required for left→right handedness conversion.

### 5. Unlit Materials by Default

**Decision**: Export with `KHR_materials_unlit` extension in glTF.

**Rationale**:
- Baked colors already include lighting
- Additional lighting creates double-lit artifacts
- Unlit materials render consistently across engines
- Better performance on mobile/web

---

## Implementation Constraints

### Real-World Limitations

1. **VRAM Budget**: Only geometry/textures currently in GPU VRAM are captured
   - Solution: Multiple captures from different positions/LOD distances

2. **Texture Streaming**: High-res textures may not be loaded
   - Solution: Force LOD 0, wait for streaming before capture

3. **Culling**: Occluded/off-screen geometry isn't rendered
   - Solution: Multiple camera angles, noclip mode for interior access

4. **Anti-Piracy**: Some games block debugger attachment
   - Solution: Launch through RenderDoc, use DX11 mode

5. **Coordinate Precision**: Large scenes may have floating-point precision issues
   - Solution: Segment large scenes, use local coordinates

### Tool Dependencies

| Dependency | Required | Purpose | Fallback |
|------------|----------|---------|----------|
| RenderDoc | Yes (extraction) | GPU capture & replay | - |
| Blender 3.x | Yes (full pipeline) | UV unwrap, bake, export | `core_gltf`/`core_usd` for direct export |
| Python 3.8+ | Yes | Pipeline scripts | - |
| numpy | Recommended | Fast geometry processing | Pure Python fallback |
| trimesh | Optional | Validation, analysis | Basic file checks |
| pygltflib | Optional | Direct glTF export | Blender export |
| PIL/Pillow | Optional | Texture processing | Skip texture features |
| pxr (OpenUSD) | Optional | USD validation | ASCII USDA output |

---

## Quality Assurance

### Validation Pipeline

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Capture    │────▶│   Extract    │────▶│   Process    │
│   (.rdc)     │     │   (OBJ+PNG)  │     │   (Dedupe)   │
└──────────────┘     └──────┬───────┘     └──────┬───────┘
                           │                     │
                     ┌─────┴─────┐         ┌─────┴─────┐
                     │ Checkpoint │         │ Checkpoint │
                     │ (Human QA) │         │ (Human QA) │
                     └───────────┘         └───────────┘
                                                 │
┌──────────────┐     ┌──────────────┐     ┌──────┴───────┐
│   Validate   │◀────│   Export     │◀────│   Blender    │
│   (QA)       │     │ (glTF/USD)   │     │   (Bake)     │
└──────┬───────┘     └──────────────┘     └──────────────┘
       │
 ┌─────┴─────┐
 │  Metrics  │
 │ IoU, SSIM │
 └───────────┘
```

### Quality Metrics

| Metric | Target | Purpose |
|--------|--------|---------|
| Silhouette IoU | ≥ 0.92 | Geometry alignment accuracy |
| SSIM | ≥ 0.80 | Color/structure similarity |
| Export file size | < 100MB | Reasonable asset size |
| Vertex count | < 5M | Performance constraint |
| NaN/Inf vertices | 0 | Data integrity |

### Checkpoint System

Human review points in batch processing:

1. **Post-Extraction**: Verify mesh count, check for obvious failures
2. **Post-Deduplication**: Confirm duplicate removal didn't lose geometry
3. **Pre-Export**: Final review before committing to output format
4. **Post-Export**: Validate final files, spot-check in viewer

---

## Workflow Patterns

### Standard Single-Scene Workflow

```bash
# 1. Capture in game (RenderDoc attached)
#    - Position camera, freeze time, F12 to capture

# 2. Extract (in RenderDoc environment)
renderdoccmd python core_extract.py --rdc scene.rdc --out export/

# 3. Process and export
python run_core.py --scene export/scene/

# 4. Validate
python core_validate.py --dir targets/

# Results:
#   targets/gltf/scene.glb
#   targets/usd/scene.usdc
#   targets/blender/scene.blend
```

### Batch Processing Workflow

```python
from core_batch import BatchProcessor, BatchConfig

config = BatchConfig(
    checkpoint_after_extract=True,
    checkpoint_before_export=True,
    auto_approve_timeout=60  # Auto-approve after 1 min if no input
)

processor = BatchProcessor(config)
processor.add_captures_from_directory("captures/")
results = processor.run(auto_approve=False)  # Manual review

processor.generate_report("batch_report.html")
```

### Multi-Angle Scene Assembly

```
Capture 1 (Front)  ──┐
Capture 2 (Back)   ──┼──▶ Merge ──▶ Complete Scene
Capture 3 (Top)    ──┤
Capture 4 (Detail) ──┘
```

Use `core_merge` for combining meshes from multiple captures.

---

## Future Considerations

### Potential Enhancements

1. **Automatic LOD Selection**: Detect and prefer highest-detail LOD meshes
2. **Multi-View Baking**: Combine multiple capture angles for complete coverage
3. **Instance Detection**: Identify and optimize repeated geometry as instances
4. **Material Reconstruction**: Optional PBR estimation from G-buffer data
5. **Animation Support**: Capture and export skeletal animation data
6. **Web Viewer**: Browser-based scene preview with three.js

### Known Limitations to Address

1. **Transparency Handling**: Alpha-blended geometry often renders incorrectly
2. **Particle Systems**: Billboards and particles don't export well
3. **Foliage**: Dense vegetation creates very high mesh counts
4. **Water/Effects**: Shader-based effects can't be captured as geometry

---

## Project Structure

```
3DvramCap/
├── capture_pipeline/
│   ├── captures/              # Input: RenderDoc .rdc files
│   ├── export/                # Intermediate: extracted meshes/textures
│   │   └── <scene_name>/
│   │       ├── Meshes/        # Post-VS OBJ files
│   │       ├── Textures/      # Final color, G-buffer
│   │       └── scene.json     # Extraction manifest
│   ├── library/               # Deduplicated canonical meshes
│   │   ├── meshes/
│   │   └── mesh_index.json
│   ├── targets/               # Final exports
│   │   ├── blender/           # .blend files
│   │   ├── gltf/              # .glb/.gltf files
│   │   └── usd/               # .usdc/.usda files
│   ├── qa/                    # Validation reports
│   ├── core/                  # Core modules
│   │   ├── core_extract.py    # RenderDoc extraction
│   │   ├── core_process.py    # Mesh deduplication
│   │   ├── core_blender.py    # Blender processing
│   │   ├── core_gltf.py       # Direct glTF export
│   │   ├── core_usd.py        # Direct USD export
│   │   ├── core_validate.py   # Export validation
│   │   ├── core_batch.py      # Batch processing
│   │   ├── core_camera.py     # Coordinate transforms
│   │   ├── core_types.py      # Shared data structures
│   │   └── ...                # Additional utilities
│   ├── scripts/               # High-level pipeline scripts
│   │   ├── run_pipeline.py    # Main orchestrator
│   │   └── 01-09_*.py         # Individual steps
│   ├── config.yaml            # Pipeline configuration
│   └── README.md              # Usage documentation
└── docu/                      # Project documentation
    └── project-vision.md      # This document
```

---

## Success Criteria

The project succeeds when:

1. A user can capture a UE4 game scene and have a valid glTF in < 30 minutes
2. Exported scenes view correctly in Blender, three.js, and Unity
3. Batch processing can handle 100+ captures with < 5% failure rate
4. Quality metrics meet thresholds (IoU ≥ 0.92, SSIM ≥ 0.80)
5. All outputs pass format validation (glTF-Validator, USD compliance)

---

## Conclusion

3DvramCap prioritizes reliability and correctness over advanced features. By focusing on post-VS extraction with baked colors, the pipeline produces consistently valid output across the widest range of UE4 titles. The modular architecture allows incremental enhancement while maintaining a working baseline.

The checkpoint-based batch processing ensures human oversight where it matters, making this suitable for both hobbyist use and larger archival projects.

---

*Document Version: 1.0*
*Last Updated: 2025-11-22*
*Based on codebase audit of capture_pipeline/*
