# Test Scene - Example Data

This is a minimal test scene for validating the 3DvramCap pipeline.

## Contents

```
test_scene/
├── Meshes/
│   ├── cube.obj       # 8 vertices, 12 triangles
│   ├── plane.obj      # 4 vertices, 2 triangles (ground)
│   └── pyramid.obj    # 5 vertices, 6 triangles
└── Textures/
    └── final_color.png  # 128x128 test texture
```

## Running the Pipeline

From the `capture_pipeline/` directory:

```bash
# Process meshes only (deduplication)
# Note: Default min-vertices is 50, but test meshes are smaller
python core/core_process.py --input ../export/test_scene/Meshes/ \
  --output ../library/test_scene/ --min-vertices 3

# Or modify config.yaml to set min_vertices: 3 for testing

# Full pipeline (requires Blender and config modification)
# Edit config.yaml first to set lower min_vertices threshold
python pipeline.py run export/test_scene/

# Validate output
python pipeline.py validate targets/gltf/
```

## Expected Results

After running the pipeline:

- **library/test_scene/** - Deduplicated meshes (3 files)
- **targets/gltf/test_scene.glb** - glTF export
- **targets/usd/test_scene.usdc** - USD export
- **targets/blender/test_scene.blend** - Blender file

## Test Criteria

✓ All 3 OBJ files should be processed
✓ No duplicates detected (all meshes are unique)
✓ glTF file should validate successfully
✓ Total vertex count: ~17 vertices
✓ Total triangle count: ~20 triangles

## Notes

**Important**: These test meshes have very low vertex counts (4-8 vertices) to keep them simple. The default pipeline configuration filters out meshes with fewer than 50 vertices (to remove UI elements, debug geometry, etc.). For this test scene, you must:

1. Run core_process.py directly with `--min-vertices 3`, OR
2. Modify `config.yaml` to set a lower threshold temporarily

This is a minimal example. Real UE4 captures will have:
- Hundreds to thousands of meshes
- Complex materials and textures
- Multiple LOD variants (duplicates to remove)
- Higher vertex counts (typically 100+ vertices per mesh)
