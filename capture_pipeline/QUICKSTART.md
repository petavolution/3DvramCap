# Quick Start: First Test

Capture a UE4 scene and see it in Blender. **~30 minutes**

---

## Prerequisites

1. **RenderDoc** (1.25+) - https://renderdoc.org/
2. **Blender** (3.x+) - https://blender.org/
3. **Python** (3.8+) with: `pip install numpy`
4. **UUU** (for UE4 games) - https://framedsc.com/GeneralGuides/universal_ue4_consoleunlocker.htm

---

## Step 1: Capture Frame (10 min)

### Launch Game
1. Open **RenderDoc** → **File → Launch Application**
2. Set **Executable** to game .exe → Click **Launch**

### Freeze & Position (with UUU)
1. In-game, run **UuuClient.exe** → Inject
2. Enable free camera + timestop
3. Position camera → Press **F12** to capture
4. Save .rdc to `captures/` folder

---

## Step 2: Extract in RenderDoc (10 min)

### Save Final Color
1. Open .rdc in RenderDoc
2. Go to last event in **Event Browser**
3. In **Texture Viewer** → Right-click → **Save Texture** as `export/<scene>/Textures/final.png`

### Export Meshes
For a few draw calls (walls, floor, props):
1. Click draw call in **Event Browser**
2. Open **Mesh Viewer** → Set view to **Post-VS**
3. **Export Mesh** → OBJ → save to `export/<scene>/Meshes/`

Create the scene folder structure:
```
export/my_scene/
├── Meshes/
│   ├── wall.obj
│   └── floor.obj
└── Textures/
    └── final.png
```

---

## Step 3: Run Pipeline (5 min)

```bash
# Full pipeline
python pipeline.py run export/my_scene/

# Or individual steps:
python pipeline.py process export/my_scene/Meshes/ --out library/
python pipeline.py export library/ --gltf scene.glb
```

Or use shell script:
```bash
./run_all.sh my_scene
```

---

## Step 4: View Results

1. Open **Blender**
2. Open `targets/blender/my_scene.blend`
3. Press **Z → Material Preview** to see baked colors

### Troubleshooting

| Issue | Fix |
|-------|-----|
| Tiny meshes | Use `--scale 1.0` instead of 0.01 |
| Missing meshes | Export more draw calls from RenderDoc |
| No colors | Check `final.png` path |

---

## Success Criteria

- [x] .blend file opens in Blender
- [x] Meshes visible in reasonable positions
- [x] Colors approximate what game showed

---

## File Locations After Test

```
capture_pipeline/
├── captures/           # Your .rdc files
├── export/my_scene/    # Extracted OBJs + textures
├── library/            # Deduplicated meshes
└── targets/
    ├── blender/        # my_scene.blend
    ├── gltf/           # my_scene.glb
    └── usd/            # my_scene.usdc
```

---

## Next Steps

1. Capture more angles for complete coverage
2. Use automated extraction with `renderdoccmd python core/core_extract.py`
3. Try `python pipeline.py validate targets/gltf/`
