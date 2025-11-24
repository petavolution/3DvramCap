#!/usr/bin/env python3
"""
RenderDoc Scene Extraction Script
==================================
Extracts post-VS meshes and textures from RenderDoc captures (.rdc files).

Usage:
    Run inside RenderDoc's Python shell or with renderdoccmd:
    renderdoccmd python 01_extract_from_rdc.py --rdc captures/scene.rdc --out export

Key Concepts:
    - Post-VS mesh: Geometry after vertex shader, already in world space
    - Final color: The framebuffer at end of frame (what you see on screen)
    - G-Buffer: Deferred rendering targets (albedo, normals, etc.) if available

Author: Capture Pipeline
"""

import argparse
import json
import os
import sys
import struct
from pathlib import Path

# RenderDoc module - only available when running inside RenderDoc
try:
    import renderdoc as rd
except ImportError:
    print("ERROR: This script must be run inside RenderDoc's Python environment.")
    print("Use: renderdoccmd python 01_extract_from_rdc.py --rdc <file.rdc> --out <dir>")
    sys.exit(1)


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path


def open_capture(rdc_path):
    """
    Open a RenderDoc capture file and return the controller for replay.

    Returns:
        tuple: (CaptureFile, ReplayController)
    """
    cap = rd.OpenCaptureFile()
    result = cap.OpenFile(rdc_path, '', None)

    if result != rd.ResultCode.Succeeded:
        raise RuntimeError(f"Failed to open capture file: {result}")

    if not cap.LocalReplaySupport():
        raise RuntimeError("Local replay not supported for this capture")

    # Open capture for replay
    result, controller = cap.OpenCapture(rd.ReplayOptions(), None)

    if result != rd.ResultCode.Succeeded:
        raise RuntimeError(f"Failed to open capture for replay: {result}")

    return cap, controller


def find_final_color_target(controller):
    """
    Find the final color render target (framebuffer) at end of frame.

    Strategy: Go to last event, get the output targets, pick the main color.
    """
    drawcalls = controller.GetRootActions()
    if not drawcalls:
        return None, None

    # Find last actual draw event
    last_event = drawcalls[-1]
    while last_event.children:
        last_event = last_event.children[-1]

    controller.SetFrameEvent(last_event.eventId, True)

    pipe = controller.GetPipelineState()

    # Get output targets
    outputs = pipe.GetOutputTargets()
    color_id = rd.ResourceId.Null()

    if outputs and len(outputs) > 0:
        color_id = outputs[0].resourceId

    # Get depth target
    depth_target = pipe.GetDepthTarget()
    depth_id = depth_target.resourceId if depth_target else rd.ResourceId.Null()

    return color_id, depth_id


def find_gbuffer_targets(controller, min_mrts=3):
    """
    Scan for a G-Buffer pass (deferred rendering) with multiple render targets.

    G-Buffer passes typically have 3+ MRTs: albedo, normals, material props.
    """
    drawcalls = controller.GetRootActions()

    def scan_drawcalls(actions):
        for action in reversed(actions):
            if action.children:
                result = scan_drawcalls(action.children)
                if result:
                    return result

            if not (action.flags & rd.ActionFlags.Drawcall):
                continue

            controller.SetFrameEvent(action.eventId, True)
            pipe = controller.GetPipelineState()
            outputs = pipe.GetOutputTargets()

            valid = [t for t in outputs if t.resourceId != rd.ResourceId.Null()]
            if len(valid) >= min_mrts:
                return [t.resourceId for t in valid]

        return None

    return scan_drawcalls(drawcalls) or []


def save_texture(controller, tex_id, out_path, file_type=rd.FileType.PNG):
    """
    Save a texture resource to disk.
    """
    if tex_id == rd.ResourceId.Null():
        return False

    try:
        ts = rd.TextureSave()
        ts.resourceId = tex_id
        ts.destType = file_type
        ts.alpha = rd.AlphaMapping.Preserve

        controller.SaveTexture(ts, out_path)
        return True
    except Exception as e:
        print(f"  Warning: Could not save texture: {e}")
        return False


def export_mesh_to_obj(controller, event_id, out_path):
    """
    Export post-VS mesh data for a draw call to OBJ format.

    Post-VS means the geometry is already transformed to world space,
    which is exactly what we want for scene reconstruction.
    """
    try:
        controller.SetFrameEvent(event_id, True)

        # Get mesh output data (post vertex shader)
        mesh_out = controller.GetPostVSData(0, 0, rd.MeshDataStage.VSOut)

        if mesh_out is None or mesh_out.numIndices == 0:
            return False

        # Get the actual vertex/index data
        # This requires accessing the buffer data

        # For now, use RenderDoc's built-in mesh export if available
        # The exact API varies by RenderDoc version

        # Alternative: manually construct OBJ from buffer data
        # This is more complex but gives us full control

        pipe = controller.GetPipelineState()

        # Get vertex input layout
        vi = pipe.GetVertexInputs()

        # Get index buffer
        ib = pipe.GetIBuffer()

        # Get vertex buffers
        vbs = pipe.GetVBuffers()

        if not vbs:
            return False

        # Read index data
        if ib.resourceId != rd.ResourceId.Null():
            ib_data = controller.GetBufferData(ib.resourceId, ib.byteOffset, 0)
        else:
            ib_data = None

        # Read first vertex buffer (positions should be here)
        vb = vbs[0]
        if vb.resourceId == rd.ResourceId.Null():
            return False

        vb_data = controller.GetBufferData(vb.resourceId, vb.byteOffset, 0)

        if not vb_data:
            return False

        # Parse vertex layout to find position attribute
        pos_offset = 0
        pos_format = None
        stride = vb.byteStride

        for attr in vi.attributes:
            if attr.name.upper() in ['POSITION', 'SV_POSITION', 'POSITION0']:
                pos_offset = attr.byteOffset
                pos_format = attr.format
                break

        if stride == 0:
            # Estimate stride from format (assuming float3 position)
            stride = 32  # Common stride for pos + normal + uv

        # Calculate vertex count
        num_verts = len(vb_data) // stride

        if num_verts == 0:
            return False

        # Write OBJ file
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(f"# OBJ exported from RenderDoc capture\n")
            f.write(f"# Event ID: {event_id}\n")
            f.write(f"# Vertices: {num_verts}\n\n")

            # Write vertices
            for i in range(num_verts):
                offset = i * stride + pos_offset
                if offset + 12 > len(vb_data):
                    break

                # Unpack as float3 (x, y, z)
                try:
                    x, y, z = struct.unpack_from('fff', vb_data, offset)
                    f.write(f"v {x} {y} {z}\n")
                except struct.error:
                    break

            # Write faces from index buffer
            if ib_data:
                # Determine index format (16-bit or 32-bit)
                idx_size = 2 if ib.byteStride <= 2 else 4
                idx_format = 'H' if idx_size == 2 else 'I'
                num_indices = len(ib_data) // idx_size

                indices = []
                for i in range(num_indices):
                    try:
                        idx = struct.unpack_from(idx_format, ib_data, i * idx_size)[0]
                        indices.append(idx + 1)  # OBJ is 1-indexed
                    except struct.error:
                        break

                # Write triangles
                for i in range(0, len(indices) - 2, 3):
                    f.write(f"f {indices[i]} {indices[i+1]} {indices[i+2]}\n")
            else:
                # No index buffer - assume triangle list
                for i in range(0, num_verts - 2, 3):
                    f.write(f"f {i+1} {i+2} {i+3}\n")

        return True

    except Exception as e:
        print(f"  Warning: Could not export mesh at event {event_id}: {e}")
        return False


def is_geometry_drawcall(action):
    """Check if a draw action is likely geometry (not UI, postFX, etc.)."""
    flags = action.flags

    # Must be an actual draw call
    if not (flags & rd.ActionFlags.Drawcall):
        return False

    # Skip clears
    if flags & rd.ActionFlags.Clear:
        return False

    # Skip very small draws (likely UI or debug)
    if hasattr(action, 'numIndices') and action.numIndices < 60:
        return False

    return True


def extract_capture(rdc_path, out_dir):
    """
    Main extraction function for a single RenderDoc capture.

    Outputs:
        - Meshes/*.obj: Post-VS geometry for each valid draw call
        - Textures/final.png: Final color buffer
        - Textures/depth.exr: Depth buffer (if available)
        - Textures/gbuffer_*.png: G-buffer targets (if deferred)
        - scene.json: Index of all extracted assets
    """
    print(f"\n=== Extracting: {rdc_path} ===")

    cap, controller = open_capture(rdc_path)

    try:
        base_name = Path(rdc_path).stem
        out_base = ensure_dir(os.path.join(out_dir, base_name))
        mesh_dir = ensure_dir(os.path.join(out_base, "Meshes"))
        tex_dir = ensure_dir(os.path.join(out_base, "Textures"))

        scene_index = {
            "capture": base_name,
            "source_file": str(rdc_path),
            "meshes": [],
            "textures": {},
            "units": "UE4_1uu=1cm",
            "notes": "Post-VS world-space export"
        }

        # 1. Save final color buffer
        print("  Finding final color buffer...")
        color_id, depth_id = find_final_color_target(controller)

        if color_id != rd.ResourceId.Null():
            final_path = os.path.join(tex_dir, "final.png")
            if save_texture(controller, color_id, final_path):
                scene_index["textures"]["final_color"] = "Textures/final.png"
                print(f"    Saved final color: {final_path}")

        # 2. Save depth buffer
        if depth_id != rd.ResourceId.Null():
            depth_path = os.path.join(tex_dir, "depth.exr")
            if save_texture(controller, depth_id, depth_path, rd.FileType.EXR):
                scene_index["textures"]["depth"] = "Textures/depth.exr"
                print(f"    Saved depth: {depth_path}")

        # 3. Try to find G-buffer targets
        print("  Scanning for G-buffer targets...")
        gbuffer_ids = find_gbuffer_targets(controller)

        for i, gb_id in enumerate(gbuffer_ids):
            gb_path = os.path.join(tex_dir, f"gbuffer_{i}.png")
            if save_texture(controller, gb_id, gb_path):
                scene_index["textures"][f"gbuffer_{i}"] = f"Textures/gbuffer_{i}.png"
                print(f"    Saved G-buffer {i}: {gb_path}")

        # 4. Export meshes from draw calls
        print("  Exporting meshes from draw calls...")

        drawcalls = controller.GetRootActions()
        mesh_count = 0

        def process_drawcalls(actions, depth=0):
            nonlocal mesh_count

            for action in actions:
                # Process children first (depth-first)
                if action.children:
                    process_drawcalls(action.children, depth + 1)

                # Check if this is a valid geometry draw
                if not is_geometry_drawcall(action):
                    continue

                # Export mesh
                obj_name = f"eid_{action.eventId}.obj"
                obj_path = os.path.join(mesh_dir, obj_name)

                if export_mesh_to_obj(controller, action.eventId, obj_path):
                    scene_index["meshes"].append({
                        "eventId": action.eventId,
                        "file": f"Meshes/{obj_name}",
                        "name": action.customName if hasattr(action, 'customName') else None
                    })
                    mesh_count += 1

                    if mesh_count % 50 == 0:
                        print(f"    Exported {mesh_count} meshes...")

        process_drawcalls(drawcalls)

        print(f"    Total meshes exported: {mesh_count}")

        # 5. Write scene index
        index_path = os.path.join(out_base, "scene.json")
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(scene_index, f, indent=2)

        print(f"  Scene index saved: {index_path}")
        print(f"=== Extraction complete: {base_name} ===\n")

        return scene_index

    finally:
        controller.Shutdown()
        cap.Shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="Extract meshes and textures from RenderDoc captures"
    )
    parser.add_argument(
        '--rdc',
        required=True,
        help="Path to .rdc capture file (or directory of .rdc files)"
    )
    parser.add_argument(
        '--out',
        default='export',
        help="Output directory (default: export)"
    )

    args = parser.parse_args()

    ensure_dir(args.out)

    rdc_path = Path(args.rdc)

    if rdc_path.is_dir():
        # Process all .rdc files in directory
        rdc_files = list(rdc_path.glob("*.rdc"))
        print(f"Found {len(rdc_files)} capture files")

        all_scenes = []
        for rdc_file in rdc_files:
            try:
                scene = extract_capture(str(rdc_file), args.out)
                all_scenes.append(scene)
            except Exception as e:
                print(f"ERROR processing {rdc_file}: {e}")

        # Write master index
        master_index = {
            "scenes": all_scenes,
            "total_captures": len(all_scenes)
        }
        with open(os.path.join(args.out, "master_index.json"), 'w') as f:
            json.dump(master_index, f, indent=2)

    else:
        # Process single file
        extract_capture(str(rdc_path), args.out)

    print("\n[OK] Extraction complete!")


if __name__ == "__main__":
    main()
