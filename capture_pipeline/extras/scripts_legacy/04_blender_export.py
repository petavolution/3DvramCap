#!/usr/bin/env python3
"""
Blender Export Script (glTF 2.0 and USD)
========================================
Exports the reconstructed scene to glTF and USD formats.

Usage:
    blender -b scene.blend --python 04_blender_export.py -- \
        --gltf targets/gltf/scene.glb \
        --usd targets/usd/scene.usdc

Options:
    --gltf PATH     Export to glTF 2.0 format (GLB or GLTF)
    --usd PATH      Export to USD format (.usdc, .usda, or .usd)
    --unlit         Use unlit materials (KHR_materials_unlit for glTF)
    --embed         Embed textures in GLB (default: separate files)

Key Concepts:
    - glTF 2.0: Standard for real-time 3D (web, AR/VR, game engines)
    - USD: Universal Scene Description (film/VFX, Omniverse, UE5)
    - Coordinate systems are handled automatically by exporters
    - Baked materials export as BaseColor / unlit

Author: Capture Pipeline
"""

import sys
import os
from pathlib import Path

# Parse arguments
argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    argv = []


def get_arg(name, default=None):
    """Get argument value from command line."""
    if name in argv:
        idx = argv.index(name)
        if idx + 1 < len(argv):
            return argv[idx + 1]
    return default


def has_flag(name):
    """Check if flag is present."""
    return name in argv


# Parse arguments
GLTF_PATH = get_arg("--gltf", None)
USD_PATH = get_arg("--usd", None)
USE_UNLIT = has_flag("--unlit")
EMBED_TEXTURES = has_flag("--embed")

# Import Blender
try:
    import bpy
except ImportError:
    print("ERROR: This script must be run inside Blender.")
    sys.exit(1)


def export_gltf(filepath, embed=False, unlit=False):
    """
    Export scene to glTF 2.0 format.

    Args:
        filepath: Output path (.glb for binary, .gltf for JSON)
        embed: If True, embed textures in GLB; else save separately
        unlit: If True, use KHR_materials_unlit extension
    """
    print(f"\nExporting glTF: {filepath}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Determine format from extension
    ext = Path(filepath).suffix.lower()
    export_format = 'GLB' if ext == '.glb' else 'GLTF_SEPARATE'

    if embed and ext != '.glb':
        export_format = 'GLTF_EMBEDDED'

    # Export settings
    try:
        bpy.ops.export_scene.gltf(
            filepath=filepath,
            export_format=export_format,

            # Mesh settings
            export_apply=True,  # Apply modifiers
            export_texcoords=True,
            export_normals=True,
            export_tangents=True,  # Required for normal maps

            # Material settings
            export_materials='EXPORT',
            export_image_format='AUTO',

            # Transform settings (glTF is Y-up, right-handed)
            export_yup=True,

            # Include cameras
            export_cameras=True,

            # Compression (optional)
            export_draco_mesh_compression_enable=False,
        )

        print(f"  [OK] Exported: {filepath}")

        # If unlit requested, we'd need to modify materials before export
        # or use a custom extension. For now, note this in the output.
        if unlit:
            print("  Note: For true unlit, set materials to Emission shader before export")

        return True

    except Exception as e:
        print(f"  [ERROR] glTF export failed: {e}")
        return False


def export_usd(filepath):
    """
    Export scene to USD format.

    Args:
        filepath: Output path (.usdc, .usda, or .usd)

    Notes:
        - .usdc = binary crate format (smaller, faster)
        - .usda = ASCII format (human readable)
        - .usd = auto-detect
    """
    print(f"\nExporting USD: {filepath}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    try:
        bpy.ops.wm.usd_export(
            filepath=filepath,

            # What to export
            export_animation=False,
            export_hair=False,
            export_uvmaps=True,
            export_normals=True,

            # Materials
            export_materials=True,
            generate_preview_surface=True,  # Creates UsdPreviewSurface

            # Hierarchy
            export_textures=True,
            relative_paths=True,

            # Transform
            # USD is Z-up by default, which matches UE4
        )

        print(f"  [OK] Exported: {filepath}")
        return True

    except Exception as e:
        print(f"  [ERROR] USD export failed: {e}")
        return False


def setup_unlit_materials():
    """
    Convert all materials to unlit (emission-based) for exact baked look.

    This ensures the exported scene looks exactly like the capture,
    without additional lighting affecting the appearance.
    """
    print("\nConverting materials to unlit...")

    for mat in bpy.data.materials:
        if not mat.use_nodes:
            continue

        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        # Find existing image texture and output
        img_node = None
        output_node = None

        for node in nodes:
            if node.type == 'TEX_IMAGE' and node.image:
                img_node = node
            if node.type == 'OUTPUT_MATERIAL':
                output_node = node

        if not img_node or not output_node:
            continue

        # Clear existing connections to output
        for link in list(links):
            if link.to_node == output_node:
                links.remove(link)

        # Create emission shader if not exists
        emission = None
        for node in nodes:
            if node.type == 'EMISSION':
                emission = node
                break

        if not emission:
            emission = nodes.new('ShaderNodeEmission')

        # Connect: Image -> Emission -> Output
        links.new(img_node.outputs['Color'], emission.inputs['Color'])
        emission.inputs['Strength'].default_value = 1.0
        links.new(emission.outputs['Emission'], output_node.inputs['Surface'])

    print("  Materials converted to unlit")


def print_scene_stats():
    """Print statistics about the scene."""
    mesh_objects = [o for o in bpy.data.objects if o.type == 'MESH']
    total_verts = sum(len(o.data.vertices) for o in mesh_objects)
    total_faces = sum(len(o.data.polygons) for o in mesh_objects)

    print("\n=== Scene Statistics ===")
    print(f"  Objects: {len(mesh_objects)}")
    print(f"  Vertices: {total_verts:,}")
    print(f"  Faces: {total_faces:,}")
    print(f"  Materials: {len(bpy.data.materials)}")
    print(f"  Images: {len(bpy.data.images)}")


def main():
    """Main export function."""
    print("\n=== Blender Export Pipeline ===")

    if not GLTF_PATH and not USD_PATH:
        print("ERROR: No export path specified. Use --gltf and/or --usd")
        sys.exit(1)

    # Print scene info
    print_scene_stats()

    # Convert to unlit if requested
    if USE_UNLIT:
        setup_unlit_materials()

    success = True

    # Export glTF
    if GLTF_PATH:
        if not export_gltf(GLTF_PATH, EMBED_TEXTURES, USE_UNLIT):
            success = False

    # Export USD
    if USD_PATH:
        if not export_usd(USD_PATH):
            success = False

    print("\n=== Export Complete ===")

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
