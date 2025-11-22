# 3DvramCap - GPU VRAM Capture Pipeline

Extract 3D scene geometry from UE4 games via RenderDoc, export to glTF 2.0 and USD.

## Quick Start

```bash
# 1. Show extraction command (run in RenderDoc environment)
python pipeline.py extract-cmd captures/scene.rdc

# 2. Run full pipeline (after extraction)
python pipeline.py run export/scene/

# Or use shell script
./run_all.sh scene
```

## Prerequisites

- **RenderDoc** (1.25+) - [Download](https://renderdoc.org/)
- **Blender** (3.x+) - [Download](https://blender.org/)
- **Python** (3.8+) with: `pip install numpy`

## Pipeline Usage

### Unified Entry Point

All operations through `pipeline.py`:

```bash
# Show RenderDoc extraction command
python pipeline.py extract-cmd captures/scene.rdc

# Run full pipeline on extracted scene
python pipeline.py run export/my_scene/ --targets targets/

# Individual steps
python pipeline.py process export/Meshes/ --out library/
python pipeline.py export library/ --gltf scene.glb --usd scene.usdc
python pipeline.py validate targets/gltf/
```

### Pipeline Flow

```
1. CAPTURE     Game + RenderDoc → .rdc file
2. EXTRACT     RenderDoc Python → OBJ meshes + textures
3. PROCESS     Deduplicate meshes
4. EXPORT      Blender → glTF/USD
5. VALIDATE    Check output integrity
```

## Project Structure

```
capture_pipeline/
├── pipeline.py            # UNIFIED ENTRY POINT
├── config.yaml            # Configuration
├── run_all.sh/.bat        # Shell wrappers
│
├── core/                  # Essential modules
│   ├── core_extract.py    # RenderDoc extraction
│   ├── core_process.py    # Mesh deduplication
│   ├── core_blender.py    # Blender processing
│   ├── core_gltf.py       # glTF 2.0 export
│   ├── core_usd.py        # USD export
│   ├── core_validate.py   # Output validation
│   ├── core_camera.py     # Coordinate transforms
│   └── core_types.py      # Type definitions
│
├── extras/                # Advanced features (optional)
│   ├── batch.py           # Human-supervised batch processing
│   ├── depth.py           # ReShade depth reconstruction
│   ├── ninja.py           # Ninja Ripper DX9 support
│   ├── quality.py         # QA metrics (SSIM, IoU)
│   ├── texture.py         # DDS conversion
│   └── ...                # More utilities
│
├── captures/              # Input: .rdc files
├── export/                # Intermediate: extracted data
├── library/               # Deduplicated meshes
└── targets/               # Output: glTF, USD, Blender files
```

## Key Concepts

### Post-VS Extraction
Geometry exported **after vertex shader** is already in world space - no transform puzzles.

### Baked Color
Instead of PBR reconstruction, bake captured final color onto geometry. Always imports correctly.

### Units
- UE4: 1 unit = 1 cm
- Blender/glTF: 1 unit = 1 m
- Default scale: 0.01

### Coordinates
```
UE4:     Left-handed, Z-up, X-forward
Blender: Right-handed, Z-up, Y-forward
glTF:    Right-handed, Y-up, Z-forward
```

## Configuration

Edit `config.yaml`:

```yaml
units:
  scale_factor: 0.01

bake:
  resolution: 2048

tools:
  blender: blender
  python: python
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Meshes mirrored | Check scale axis, recalculate normals |
| Wrong scale | Verify `--scale 0.01` |
| RenderDoc won't attach | Launch via RenderDoc, disable overlays |
| Washed out colors | Use correct capture image (final.png) |

## Advanced Features

See `extras/` for:
- Batch processing with human review checkpoints
- Ninja Ripper support (DX9 fallback)
- ReShade depth capture reconstruction
- QA validation metrics

## License

For educational and personal use. Respect game asset usage terms.
