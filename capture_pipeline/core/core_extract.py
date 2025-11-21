#!/usr/bin/env python3
"""
Core RenderDoc Extraction Script (Simplified)
==============================================
Extracts post-VS meshes and final color buffer from RenderDoc captures.

MUST be run inside RenderDoc's Python environment:
    renderdoccmd python core_extract.py --rdc capture.rdc --out export/

Output Structure:
    export/scene_name/
        Meshes/
            mesh_0000.obj
            mesh_0001.obj
            ...
        Textures/
            final_color.png
        scene.json

Author: Capture Pipeline
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path

# RenderDoc imports (only available in RenderDoc environment)
try:
    import renderdoc as rd
    HAS_RENDERDOC = True
except ImportError:
    HAS_RENDERDOC = False
    print("WARNING: renderdoc module not found.")
    print("This script must be run inside RenderDoc's Python environment:")
    print("  renderdoccmd python core_extract.py --rdc capture.rdc --out export/")


def open_capture(rdc_path):
    """Open RenderDoc capture file and return replay controller."""
    cap = rd.OpenCaptureFile()
    result = cap.OpenFile(str(rdc_path), '', None)

    if result != rd.ResultCode.Succeeded:
        raise RuntimeError(f"Failed to open capture: {result}")

    if not cap.LocalReplaySupport():
        raise RuntimeError("Capture cannot be replayed locally")

    result, controller = cap.OpenCapture(rd.ReplayOptions(), None)
    if result != rd.ResultCode.Succeeded:
        raise RuntimeError(f"Failed to open replay: {result}")

    return cap, controller


def find_draw_calls(controller):
    """
    Find all draw calls in the capture.

    Returns list of (event_id, action) tuples for indexed/non-indexed draws.
    """
    draws = []

    def walk_actions(actions, depth=0):
        for action in actions:
            # Check if this is a draw call
            flags = action.flags
            if (flags & rd.ActionFlags.Drawcall) and (
                (flags & rd.ActionFlags.Indexed) or
                (flags & rd.ActionFlags.Instanced) or
                action.numIndices > 0
            ):
                draws.append((action.eventId, action))

            # Recurse into children
            if action.children:
                walk_actions(action.children, depth + 1)

    root_actions = controller.GetRootActions()
    walk_actions(root_actions)

    return draws


def get_mesh_data(controller, event_id):
    """
    Extract post-VS mesh data for a specific draw call.

    Returns dict with vertices, normals, uvs, indices or None if extraction fails.
    """
    controller.SetFrameEvent(event_id, False)

    # Get pipeline state
    state = controller.GetPipelineState()

    # Request post-VS data
    postvs = controller.GetPostVSData(0, 0, rd.MeshDataStage.VSOut)

    if postvs.numIndices == 0:
        return None

    # Get position attribute index
    pos_idx = -1
    for i, attr in enumerate(postvs.vertexResourceId):
        # Position is typically the first output
        if i == 0:
            pos_idx = i
            break

    if pos_idx < 0:
        return None

    # Decode vertex data
    vertices = []
    normals = []
    uvs = []

    # Get raw vertex buffer data
    vb_data = controller.GetBufferData(postvs.vertexResourceId,
                                        postvs.vertexByteOffset,
                                        postvs.vertexByteStride * postvs.numIndices)

    stride = postvs.vertexByteStride

    # Parse vertices - assume standard layout: pos(xyz), normal(xyz), uv(xy)
    for i in range(postvs.numIndices):
        offset = i * stride

        if offset + 12 > len(vb_data):
            break

        # Position (3 floats = 12 bytes)
        x, y, z = struct.unpack_from('fff', vb_data, offset)
        vertices.append([x, y, z])

        # Normal (3 floats = 12 bytes) if stride allows
        if stride >= 24:
            try:
                nx, ny, nz = struct.unpack_from('fff', vb_data, offset + 12)
                normals.append([nx, ny, nz])
            except struct.error:
                pass

        # UV (2 floats = 8 bytes) if stride allows
        if stride >= 32:
            try:
                u, v = struct.unpack_from('ff', vb_data, offset + 24)
                uvs.append([u, 1.0 - v])  # Flip V for OpenGL->Blender
            except struct.error:
                pass

    if len(vertices) < 3:
        return None

    # Generate face indices (triangle list)
    indices = []
    for i in range(0, len(vertices) - 2, 3):
        indices.append([i, i + 1, i + 2])

    return {
        'vertices': vertices,
        'normals': normals if len(normals) == len(vertices) else [],
        'uvs': uvs if len(uvs) == len(vertices) else [],
        'indices': indices,
        'event_id': event_id
    }


def export_mesh_to_obj(mesh_data, filepath):
    """Export mesh data to Wavefront OBJ format."""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"# Exported from RenderDoc capture\n")
        f.write(f"# Event ID: {mesh_data['event_id']}\n")
        f.write(f"# Vertices: {len(mesh_data['vertices'])}\n\n")

        # Vertices
        for v in mesh_data['vertices']:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        # Normals
        if mesh_data['normals']:
            f.write("\n")
            for n in mesh_data['normals']:
                f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")

        # UVs
        if mesh_data['uvs']:
            f.write("\n")
            for uv in mesh_data['uvs']:
                f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")

        # Faces
        f.write("\n")
        has_n = bool(mesh_data['normals'])
        has_uv = bool(mesh_data['uvs'])

        for face in mesh_data['indices']:
            if has_uv and has_n:
                f.write(f"f {face[0]+1}/{face[0]+1}/{face[0]+1} "
                        f"{face[1]+1}/{face[1]+1}/{face[1]+1} "
                        f"{face[2]+1}/{face[2]+1}/{face[2]+1}\n")
            elif has_uv:
                f.write(f"f {face[0]+1}/{face[0]+1} "
                        f"{face[1]+1}/{face[1]+1} "
                        f"{face[2]+1}/{face[2]+1}\n")
            elif has_n:
                f.write(f"f {face[0]+1}//{face[0]+1} "
                        f"{face[1]+1}//{face[1]+1} "
                        f"{face[2]+1}//{face[2]+1}\n")
            else:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    return True


def find_final_color_target(controller):
    """
    Find the final color render target in the capture.

    Returns (resource_id, width, height) or None.
    """
    # Get list of textures
    textures = controller.GetTextures()

    # Find largest RGBA texture (likely the final framebuffer)
    best = None
    best_size = 0

    for tex in textures:
        # Skip depth/stencil formats
        fmt = tex.format
        if fmt.compType == rd.CompType.Depth:
            continue

        size = tex.width * tex.height
        if size > best_size and tex.width >= 256 and tex.height >= 256:
            best = tex
            best_size = size

    if best:
        return best.resourceId, best.width, best.height

    return None


def save_texture(controller, resource_id, filepath, width=None, height=None):
    """Save a texture resource to file."""
    # Get texture data
    texsave = rd.TextureSave()
    texsave.resourceId = resource_id
    texsave.mip = 0
    texsave.slice.sliceIndex = 0
    texsave.alpha = rd.AlphaMapping.Preserve
    texsave.destType = rd.FileType.PNG

    controller.SaveTexture(texsave, str(filepath))
    return os.path.exists(filepath)


def extract_capture(rdc_path, output_dir, max_meshes=500, min_vertices=50):
    """
    Main extraction function.

    Args:
        rdc_path: Path to .rdc capture file
        output_dir: Output directory for extracted data
        max_meshes: Maximum number of meshes to extract
        min_vertices: Minimum vertex count to include mesh

    Returns:
        dict with extraction results
    """
    if not HAS_RENDERDOC:
        raise RuntimeError("RenderDoc module not available")

    rdc_path = Path(rdc_path)
    output_dir = Path(output_dir)

    # Create output structure
    scene_name = rdc_path.stem
    scene_dir = output_dir / scene_name
    mesh_dir = scene_dir / "Meshes"
    tex_dir = scene_dir / "Textures"

    mesh_dir.mkdir(parents=True, exist_ok=True)
    tex_dir.mkdir(parents=True, exist_ok=True)

    print(f"Opening capture: {rdc_path}")
    cap, controller = open_capture(rdc_path)

    results = {
        'capture': str(rdc_path),
        'output_dir': str(scene_dir),
        'meshes': [],
        'textures': [],
        'stats': {
            'total_draws': 0,
            'extracted_meshes': 0,
            'skipped_small': 0,
            'skipped_failed': 0
        }
    }

    try:
        # Find draw calls
        print("Finding draw calls...")
        draws = find_draw_calls(controller)
        results['stats']['total_draws'] = len(draws)
        print(f"  Found {len(draws)} draw calls")

        # Extract meshes
        print(f"Extracting meshes (max {max_meshes})...")
        mesh_count = 0

        for i, (event_id, action) in enumerate(draws):
            if mesh_count >= max_meshes:
                break

            if (i + 1) % 100 == 0:
                print(f"  Processing {i + 1}/{len(draws)}...")

            try:
                mesh_data = get_mesh_data(controller, event_id)

                if mesh_data is None:
                    results['stats']['skipped_failed'] += 1
                    continue

                if len(mesh_data['vertices']) < min_vertices:
                    results['stats']['skipped_small'] += 1
                    continue

                # Export mesh
                mesh_name = f"mesh_{mesh_count:04d}.obj"
                mesh_path = mesh_dir / mesh_name

                if export_mesh_to_obj(mesh_data, mesh_path):
                    results['meshes'].append({
                        'name': mesh_name,
                        'path': str(mesh_path),
                        'event_id': event_id,
                        'vertices': len(mesh_data['vertices']),
                        'faces': len(mesh_data['indices'])
                    })
                    mesh_count += 1
                    results['stats']['extracted_meshes'] = mesh_count

            except Exception as e:
                results['stats']['skipped_failed'] += 1
                continue

        print(f"  Extracted {mesh_count} meshes")

        # Extract final color texture
        print("Extracting final color buffer...")
        color_target = find_final_color_target(controller)

        if color_target:
            res_id, width, height = color_target
            tex_path = tex_dir / "final_color.png"

            if save_texture(controller, res_id, tex_path, width, height):
                results['textures'].append({
                    'name': 'final_color.png',
                    'path': str(tex_path),
                    'width': width,
                    'height': height,
                    'type': 'color'
                })
                print(f"  Saved: {tex_path} ({width}x{height})")
            else:
                print("  WARNING: Failed to save color texture")
        else:
            print("  WARNING: No suitable color target found")

    finally:
        controller.Shutdown()
        cap.Shutdown()

    # Save scene manifest
    manifest_path = scene_dir / "scene.json"
    with open(manifest_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nExtraction complete!")
    print(f"  Output: {scene_dir}")
    print(f"  Meshes: {results['stats']['extracted_meshes']}")
    print(f"  Manifest: {manifest_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract meshes and textures from RenderDoc capture"
    )
    parser.add_argument('--rdc', required=True, help="Input .rdc capture file")
    parser.add_argument('--out', default='export', help="Output directory")
    parser.add_argument('--max-meshes', type=int, default=500,
                        help="Maximum meshes to extract")
    parser.add_argument('--min-vertices', type=int, default=50,
                        help="Minimum vertex count per mesh")

    args = parser.parse_args()

    if not os.path.exists(args.rdc):
        print(f"ERROR: Capture file not found: {args.rdc}")
        sys.exit(1)

    try:
        extract_capture(args.rdc, args.out, args.max_meshes, args.min_vertices)
    except Exception as e:
        print(f"ERROR: Extraction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
