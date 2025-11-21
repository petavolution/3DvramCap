# Quick Start: First Human Test

This guide walks you through capturing a UE4 scene and seeing it in Blender.
**Time needed: ~30 minutes**

---

## Prerequisites (Install Once)

1. **RenderDoc** - Download from https://renderdoc.org/
2. **Blender 3.x** - Download from https://blender.org/
3. **Universal Unreal Unlocker (UUU)** - For UE4 games:
   https://framedsc.com/GeneralGuides/universal_ue4_consoleunlocker.htm
4. **Python packages**:
   ```
   pip install numpy trimesh opencv-python
   ```

---

## Step 1: Capture a Frame (10 min)

### 1.1 Launch Game Through RenderDoc

1. Open **RenderDoc**
2. Go to **File → Launch Application**
3. Set **Executable** to your game's .exe
4. Click **Launch**

### 1.2 Freeze and Position (with UUU)

1. Once in-game, launch **UuuClient.exe**
2. Select your game process and click **Inject**
3. In game, press the UUU hotkeys:
   - **Toggle free camera** (usually Page Down or a configured key)
   - **Timestop** to freeze the game
4. Move camera to a good view of the scene

### 1.3 Capture

1. Press **F12** to capture a frame
2. Save the .rdc file to `captures/` folder

---

## Step 2: Extract in RenderDoc (10 min)

### 2.1 Save Final Color

1. Open your .rdc in RenderDoc
2. In **Event Browser**, go to the last event
3. In **Texture Viewer**, you'll see the final image
4. Right-click → **Save Texture** as `captures/final.png`

### 2.2 Export Meshes (Manual Method)

For 2-3 big objects (walls, floor, props):

1. Click on a draw call in the **Event Browser**
2. In the viewport, check if it highlights interesting geometry
3. Once you find a good one, open **Mesh Viewer** (tabs at bottom)
4. Make sure view is set to **Post-VS** (or VS Out)
5. Click **Export Mesh** → choose **OBJ** format
6. Save to `export/Meshes/wall.obj` (or similar name)

Repeat for a few objects.

---

## Step 3: Process and Import (5 min)

### 3.1 Deduplicate (Optional for first test)

```bash
python scripts/02_dedupe_meshes.py export/Meshes library/index.json
```

### 3.2 Import to Blender

```bash
blender -b --python scripts/03_blender_import_bake.py -- \
  --meshes export/Meshes \
  --color captures/final.png \
  --out targets/blender/test.blend
```

Or if you want geometry only (faster):

```bash
blender -b --python scripts/03_blender_import_bake.py -- \
  --meshes export/Meshes \
  --out targets/blender/test.blend \
  --no-bake
```

---

## Step 4: View Results (5 min)

1. Open **Blender**
2. Open `targets/blender/test.blend`
3. You should see your extracted geometry!
4. Press **Z → Material Preview** to see baked colors

### Troubleshooting

- **Everything is tiny**: Scale is in UE4 cm, multiply by 100 or use `--scale 1.0`
- **Meshes missing**: Export more draw calls from RenderDoc
- **No colors**: Make sure `final.png` was saved and path is correct

---

## Step 5: Export (Optional)

```bash
# Export to glTF and USD
blender -b targets/blender/test.blend --python scripts/04_blender_export.py -- \
  --gltf targets/gltf/test.glb \
  --usd targets/usd/test.usdc
```

---

## Success Criteria

You've succeeded if:

- [x] You can open the .blend file in Blender
- [x] Meshes are visible and in reasonable positions
- [x] Baked colors show what the game looked like (approximately)

---

## Next Steps

Once this basic test works:

1. **Capture more angles** for complete scene coverage
2. **Use the automated extraction script** with RenderDoc's Python
3. **Run full pipeline** with `run_all.sh` or `run_all.bat`
4. **Try QA validation** to verify accuracy

---

## File Locations After Test

```
capture_pipeline/
├── captures/
│   ├── scene.rdc          # Your capture
│   └── final.png          # Color buffer
├── export/
│   └── Meshes/            # Exported OBJs
├── library/
│   ├── meshes/            # Deduplicated
│   └── index.json
└── targets/
    ├── blender/test.blend # Your reconstructed scene!
    ├── gltf/test.glb
    └── usd/test.usdc
```
