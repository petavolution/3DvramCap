#!/usr/bin/env python3
"""
Core Blender Processing Script (All-in-One)
============================================
Imports meshes, applies Smart UV, bakes color texture, and exports to glTF/USD.

Usage:
    blender -b --python core_blender.py -- \
        --meshes library/meshes/ \
        --color export/scene/Textures/final_color.png \
        --gltf targets/scene.glb \
        --usd targets/scene.usdc \
        --scale 0.01

Workflow:
    1. Import all OBJ meshes from input directory
    2. Apply UE4->Blender scale (0.01 for cm->m)
    3. Smart UV Project for bake-ready UVs
    4. Camera-project color texture (optional)
    5. Create unlit material with baked texture
    6. Export to glTF 2.0 (GLB) and USD

Author: Capture Pipeline
"""

import argparse
import math
import os
import sys
from pathlib import Path

# Blender imports
try:
    import bpy
    import bmesh
    HAS_BLENDER = True
except ImportError:
    HAS_BLENDER = False
    print("ERROR: This script must be run inside Blender:")
    print("  blender -b --python core_blender.py -- --meshes DIR --gltf OUT.glb")
    sys.exit(1)


def clear_scene():
    """Remove all objects from the current scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # Clear orphan data
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)
    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)


def import_obj_files(mesh_dir, scale=0.01):
    """
    Import all OBJ files from directory.

    Args:
        mesh_dir: Directory containing OBJ files
        scale: Scale factor (0.01 for UE4 cm -> Blender m)

    Returns:
        List of imported objects
    """
    mesh_path = Path(mesh_dir)
    obj_files = sorted(mesh_path.glob("*.obj"))

    if not obj_files:
        print(f"WARNING: No OBJ files found in {mesh_dir}")
        return []

    print(f"Importing {len(obj_files)} OBJ files...")
    imported = []

    for i, obj_path in enumerate(obj_files):
        if (i + 1) % 50 == 0:
            print(f"  Importing {i + 1}/{len(obj_files)}...")

        try:
            # Import OBJ
            bpy.ops.wm.obj_import(
                filepath=str(obj_path),
                forward_axis='NEGATIVE_Z',
                up_axis='Y'
            )

            # Get imported object
            obj = bpy.context.selected_objects[0] if bpy.context.selected_objects else None

            if obj:
                # Apply scale
                obj.scale = (scale, scale, scale)
                bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

                obj.name = obj_path.stem
                imported.append(obj)

        except Exception as e:
            print(f"  Error importing {obj_path.name}: {e}")
            continue

    print(f"  Imported {len(imported)} meshes")
    return imported


def apply_smart_uv(obj, angle_limit=66.0, island_margin=0.02):
    """
    Apply Smart UV Project to an object.

    Args:
        obj: Blender object
        angle_limit: Angle threshold in degrees
        island_margin: Margin between UV islands
    """
    # Ensure object is selected and active
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    # Enter edit mode
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')

    # Create UV layer if needed
    if not obj.data.uv_layers:
        obj.data.uv_layers.new(name="UVMap")

    # Apply Smart UV Project (angle must be in radians)
    try:
        bpy.ops.uv.smart_project(
            angle_limit=math.radians(angle_limit),
            island_margin=island_margin,
            area_weight=0.0,
            correct_aspect=True,
            scale_to_bounds=False
        )
    except Exception as e:
        print(f"  Smart UV failed for {obj.name}: {e}")

    # Return to object mode
    bpy.ops.object.mode_set(mode='OBJECT')


def create_unlit_material(name, texture_path=None, color=(0.8, 0.8, 0.8, 1.0)):
    """
    Create an unlit material using Emission shader.

    This exports as KHR_materials_unlit in glTF.

    Args:
        name: Material name
        texture_path: Optional path to texture image
        color: Fallback RGBA color

    Returns:
        Created material
    """
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear default nodes
    nodes.clear()

    # Create nodes for unlit material
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (400, 0)

    # Use emission shader (renders as unlit)
    emission = nodes.new('ShaderNodeEmission')
    emission.location = (200, 0)
    emission.inputs['Strength'].default_value = 1.0

    if texture_path and os.path.exists(texture_path):
        # Load texture
        img = bpy.data.images.load(texture_path)

        # Texture node
        tex_node = nodes.new('ShaderNodeTexImage')
        tex_node.location = (0, 0)
        tex_node.image = img

        # UV Map node
        uv_node = nodes.new('ShaderNodeUVMap')
        uv_node.location = (-200, 0)
        uv_node.uv_map = "UVMap"

        # Connect: UV -> Texture -> Emission -> Output
        links.new(uv_node.outputs['UV'], tex_node.inputs['Vector'])
        links.new(tex_node.outputs['Color'], emission.inputs['Color'])
    else:
        # Use solid color
        emission.inputs['Color'].default_value = color

    links.new(emission.outputs['Emission'], output.inputs['Surface'])

    return mat


def create_bake_material(obj, bake_image):
    """
    Create material with bake target node for Cycles baking.

    The bake target image node must be selected and active.
    """
    mat_name = f"{obj.name}_bake_mat"
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Get or create output
    output = nodes.get('Material Output')
    if not output:
        output = nodes.new('ShaderNodeOutputMaterial')
        output.location = (400, 0)

    # Create diffuse BSDF
    diffuse = nodes.new('ShaderNodeBsdfDiffuse')
    diffuse.location = (200, 0)
    links.new(diffuse.outputs['BSDF'], output.inputs['Surface'])

    # Create texture node for baking target
    tex_node = nodes.new('ShaderNodeTexImage')
    tex_node.name = 'Bake_Target'
    tex_node.location = (0, 200)
    tex_node.image = bake_image

    # IMPORTANT: Select and make active for baking
    nodes.active = tex_node
    tex_node.select = True

    return mat


def bake_color_to_objects(objects, color_image_path, bake_resolution=2048):
    """
    Bake captured color image onto objects using camera projection.

    This projects the captured screenshot onto the mesh UVs.

    Args:
        objects: List of objects to bake
        color_image_path: Path to captured color image
        bake_resolution: Output texture resolution
    """
    if not os.path.exists(color_image_path):
        print(f"WARNING: Color image not found: {color_image_path}")
        return None

    print(f"Baking color texture to {len(objects)} objects...")

    # Set render engine to Cycles for baking
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'CPU'
    bpy.context.scene.cycles.samples = 1  # Minimal for baking

    # Load source image
    src_image = bpy.data.images.load(color_image_path)

    # Create bake target image
    bake_image = bpy.data.images.new(
        name="baked_color",
        width=bake_resolution,
        height=bake_resolution,
        alpha=True
    )
    bake_image.colorspace_settings.name = 'sRGB'

    # Process each object
    for obj in objects:
        if obj.type != 'MESH':
            continue

        # Ensure UV layer exists
        if not obj.data.uv_layers:
            apply_smart_uv(obj)

        # Create bake material
        bake_mat = create_bake_material(obj, bake_image)

        # Clear existing materials and assign bake material
        obj.data.materials.clear()
        obj.data.materials.append(bake_mat)

    # Select all objects for baking
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objects:
        if obj.type == 'MESH':
            obj.select_set(True)
    if objects:
        bpy.context.view_layer.objects.active = objects[0]

    # Bake diffuse color
    try:
        bpy.ops.object.bake(
            type='DIFFUSE',
            pass_filter={'COLOR'},
            use_selected_to_active=False,
            margin=2
        )
        print(f"  Baking complete: {bake_resolution}x{bake_resolution}")
    except Exception as e:
        print(f"  Baking failed: {e}")
        return None

    # Save baked image
    bake_output_path = str(Path(color_image_path).parent / "baked_color.png")
    bake_image.filepath_raw = bake_output_path
    bake_image.file_format = 'PNG'
    bake_image.save()
    print(f"  Saved: {bake_output_path}")

    return bake_output_path


def assign_material_to_objects(objects, material):
    """Assign a material to all mesh objects."""
    for obj in objects:
        if obj.type != 'MESH':
            continue

        obj.data.materials.clear()
        obj.data.materials.append(material)


def export_gltf(filepath, use_draco=False):
    """
    Export scene to glTF 2.0 (GLB binary format).

    Uses KHR_materials_unlit extension for baked materials.
    """
    print(f"Exporting glTF: {filepath}")

    # Ensure directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    export_settings = {
        'filepath': filepath,
        'export_format': 'GLB',
        'export_texcoords': True,
        'export_normals': True,
        'export_materials': 'EXPORT',
        'export_colors': False,
        'export_cameras': False,
        'export_lights': False,
        'export_extras': False,
        'export_yup': True,  # glTF uses Y-up
    }

    # Add Draco compression if requested
    if use_draco:
        export_settings['export_draco_mesh_compression_enable'] = True
        export_settings['export_draco_mesh_compression_level'] = 6

    try:
        bpy.ops.export_scene.gltf(**export_settings)
        print(f"  Exported: {filepath}")
        return True
    except Exception as e:
        print(f"  Export failed: {e}")
        return False


def export_usd(filepath):
    """
    Export scene to USD format.

    Note: Requires Blender 3.0+ with USD support.
    """
    print(f"Exporting USD: {filepath}")

    # Ensure directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    try:
        bpy.ops.wm.usd_export(
            filepath=filepath,
            selected_objects_only=False,
            export_textures=True,
            relative_paths=True,
            export_materials=True,
            export_meshes=True,
            export_normals=True,
            export_uvmaps=True
        )
        print(f"  Exported: {filepath}")
        return True
    except Exception as e:
        print(f"  Export failed: {e}")
        return False


def save_blend(filepath):
    """Save the current Blender scene."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=filepath)
    print(f"  Saved: {filepath}")


def main():
    # Parse arguments after "--"
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(
        description="Blender: Import, UV, bake, and export captured meshes"
    )
    parser.add_argument('--meshes', required=True,
                        help="Input directory with OBJ meshes")
    parser.add_argument('--color', default=None,
                        help="Captured color image for baking")
    parser.add_argument('--gltf', default=None,
                        help="Output glTF file path (.glb)")
    parser.add_argument('--usd', default=None,
                        help="Output USD file path (.usdc)")
    parser.add_argument('--blend', default=None,
                        help="Output Blender file path (.blend)")
    parser.add_argument('--scale', type=float, default=0.01,
                        help="Scale factor (default: 0.01 for UE4 cm->m)")
    parser.add_argument('--bake-res', type=int, default=2048,
                        help="Bake texture resolution (default: 2048)")
    parser.add_argument('--no-bake', action='store_true',
                        help="Skip texture baking")
    parser.add_argument('--draco', action='store_true',
                        help="Enable Draco compression for glTF")

    args = parser.parse_args(argv)

    print("\n" + "="*60)
    print("Core Blender Processing")
    print("="*60)

    # Step 1: Clear scene
    print("\n[1] Clearing scene...")
    clear_scene()

    # Step 2: Import meshes
    print(f"\n[2] Importing meshes from: {args.meshes}")
    objects = import_obj_files(args.meshes, scale=args.scale)

    if not objects:
        print("ERROR: No meshes imported")
        sys.exit(1)

    # Step 3: Apply Smart UV to all objects
    print(f"\n[3] Applying Smart UV Project...")
    for i, obj in enumerate(objects):
        if (i + 1) % 50 == 0:
            print(f"  UV unwrapping {i + 1}/{len(objects)}...")
        apply_smart_uv(obj)

    # Step 4: Bake or create material
    baked_texture = None
    if args.color and not args.no_bake:
        print(f"\n[4] Baking color texture...")
        baked_texture = bake_color_to_objects(objects, args.color, args.bake_res)

    # Step 5: Create final unlit material
    print(f"\n[5] Creating unlit material...")
    texture_path = baked_texture or args.color
    final_mat = create_unlit_material("CapturedMaterial", texture_path)
    assign_material_to_objects(objects, final_mat)

    # Step 6: Save Blender file if requested
    if args.blend:
        print(f"\n[6] Saving Blender file...")
        save_blend(args.blend)

    # Step 7: Export to glTF
    if args.gltf:
        print(f"\n[7] Exporting to glTF...")
        export_gltf(args.gltf, use_draco=args.draco)

    # Step 8: Export to USD
    if args.usd:
        print(f"\n[8] Exporting to USD...")
        export_usd(args.usd)

    print("\n" + "="*60)
    print("Processing complete!")
    print(f"  Meshes processed: {len(objects)}")
    if args.blend:
        print(f"  Blender file: {args.blend}")
    if args.gltf:
        print(f"  glTF export: {args.gltf}")
    if args.usd:
        print(f"  USD export: {args.usd}")
    print("="*60)


if __name__ == "__main__":
    main()
