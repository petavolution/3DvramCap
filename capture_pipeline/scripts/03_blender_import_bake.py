#!/usr/bin/env python3
"""
Blender Import and Bake Script
==============================
Imports extracted meshes into Blender, optionally bakes color from captured
framebuffer, and prepares the scene for export.

Usage (headless):
    blender -b --python 03_blender_import_bake.py -- \
        --meshes library/meshes \
        --color export/scene/Textures/final.png \
        --out targets/blender/scene.blend \
        --bake-res 2048

Usage (with camera JSON):
    blender -b --python 03_blender_import_bake.py -- \
        --meshes library/meshes \
        --color export/scene/Textures/final.png \
        --camera captures/camera.json \
        --out targets/blender/scene.blend

Key Concepts:
    - UE4 units: 1 UU = 1 cm; Blender units: 1 = 1 m
    - Scale factor: 0.01 (or 1.0 if you want cm-scale in Blender)
    - Post-VS meshes are already in world space - no transform needed
    - Baked color: Project final.png onto geometry for "what you saw" look

Author: Capture Pipeline
"""

import sys
import os
import json
import math
import glob
from pathlib import Path

# Parse arguments before Blender imports (to handle -- separator)
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


# Parse arguments
MESHES_DIR = get_arg("--meshes", "library/meshes")
COLOR_IMAGE = get_arg("--color", None)
CAMERA_JSON = get_arg("--camera", None)
OUTPUT_BLEND = get_arg("--out", "targets/blender/scene.blend")
BAKE_RES = int(get_arg("--bake-res", "2048"))
SCALE_FACTOR = float(get_arg("--scale", "0.01"))  # UE cm -> Blender m
NO_BAKE = "--no-bake" in argv

# Now import Blender modules
try:
    import bpy
    from mathutils import Vector, Euler, Matrix
except ImportError:
    print("ERROR: This script must be run inside Blender.")
    print("Use: blender -b --python 03_blender_import_bake.py -- [args]")
    sys.exit(1)


def clear_scene():
    """Remove all objects from scene."""
    bpy.ops.wm.read_factory_settings(use_empty=True)


def setup_render_engine():
    """Configure Cycles for baking."""
    bpy.context.scene.render.engine = 'CYCLES'

    # Use CPU by default (more reliable)
    bpy.context.scene.cycles.device = 'CPU'

    # Low samples for baking (we're baking from image, not raytracing)
    bpy.context.scene.cycles.samples = 1


def create_camera_from_json(json_path):
    """
    Create camera from exported camera parameters.

    Expected JSON format:
    {
        "fov_deg": 60.0,
        "pos": [x, y, z],
        "rot_deg": [pitch, yaw, roll]
    }
    """
    if not json_path or not os.path.exists(json_path):
        return create_default_camera()

    with open(json_path, 'r') as f:
        data = json.load(f)

    cam = bpy.data.cameras.new("CaptureCamera")
    cam_obj = bpy.data.objects.new("CaptureCamera", cam)
    bpy.context.collection.objects.link(cam_obj)

    # FOV
    fov = data.get("fov_deg", 60.0)
    cam.lens_unit = 'FOV'
    cam.angle = math.radians(fov)

    # Position (apply scale factor for UE -> Blender)
    pos = data.get("pos", [0, 0, 0])
    cam_obj.location = Vector([
        pos[0] * SCALE_FACTOR,
        pos[1] * SCALE_FACTOR,
        pos[2] * SCALE_FACTOR
    ])

    # Rotation
    rot = data.get("rot_deg", [0, 0, 0])
    cam_obj.rotation_euler = Euler([
        math.radians(rot[0]),
        math.radians(rot[1]),
        math.radians(rot[2])
    ], 'XYZ')

    bpy.context.scene.camera = cam_obj
    return cam_obj


def create_default_camera():
    """Create a default camera for the scene."""
    cam = bpy.data.cameras.new("DefaultCamera")
    cam_obj = bpy.data.objects.new("DefaultCamera", cam)
    bpy.context.collection.objects.link(cam_obj)

    cam.lens_unit = 'FOV'
    cam.angle = math.radians(60.0)

    # Default position (looking at origin from front-ish)
    cam_obj.location = Vector((0, -10, 5))
    cam_obj.rotation_euler = Euler((math.radians(70), 0, 0), 'XYZ')

    bpy.context.scene.camera = cam_obj
    return cam_obj


def import_meshes(meshes_dir, scale=0.01):
    """
    Import all OBJ files from directory.

    Args:
        meshes_dir: Directory containing OBJ files
        scale: Scale factor (0.01 converts UE4 cm to Blender m)

    Returns:
        List of imported mesh objects
    """
    mesh_path = Path(meshes_dir)

    if not mesh_path.exists():
        print(f"Warning: Meshes directory not found: {meshes_dir}")
        return []

    obj_files = list(mesh_path.glob("**/*.obj"))
    print(f"Found {len(obj_files)} OBJ files to import")

    imported = []

    for obj_path in obj_files:
        try:
            # Import OBJ
            bpy.ops.wm.obj_import(
                filepath=str(obj_path),
                forward_axis='NEGATIVE_Z',  # Common UE4 -> Blender conversion
                up_axis='Y'
            )

            # Get newly imported objects
            for obj in bpy.context.selected_objects:
                if obj.type == 'MESH':
                    # Apply scale
                    obj.scale = (scale, scale, scale)

                    # Apply the scale transform
                    bpy.context.view_layer.objects.active = obj
                    bpy.ops.object.transform_apply(scale=True)

                    imported.append(obj)

        except Exception as e:
            print(f"Warning: Could not import {obj_path}: {e}")

    print(f"Successfully imported {len(imported)} mesh objects")
    return imported


def ensure_uv_layer(obj, name="BakeUV"):
    """Ensure mesh has a UV layer for baking."""
    mesh = obj.data

    if name in mesh.uv_layers:
        mesh.uv_layers.active = mesh.uv_layers[name]
        return mesh.uv_layers[name]

    # Create new UV layer
    uv_layer = mesh.uv_layers.new(name=name)
    mesh.uv_layers.active = uv_layer
    return uv_layer


def create_smart_uv(obj):
    """Create Smart UV projection for an object."""
    # Select only this object
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    # Enter edit mode
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')

    # Ensure UV layer exists
    ensure_uv_layer(obj, "BakeUV")

    # Smart UV project
    try:
        bpy.ops.uv.smart_project(
            angle_limit=math.radians(66.0),
            island_margin=0.02,
            area_weight=0.0,
            correct_aspect=True,
            scale_to_bounds=True
        )
    except Exception as e:
        print(f"Warning: Smart UV failed for {obj.name}: {e}")

    # Return to object mode
    bpy.ops.object.mode_set(mode='OBJECT')


def create_emission_material(name, image_path):
    """
    Create a material that emits the color from an image texture.
    Used as source for baking.
    """
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear default nodes
    nodes.clear()

    # Create nodes
    output = nodes.new('ShaderNodeOutputMaterial')
    emission = nodes.new('ShaderNodeEmission')
    tex_image = nodes.new('ShaderNodeTexImage')

    # Load image
    if os.path.exists(image_path):
        tex_image.image = bpy.data.images.load(image_path)
    else:
        print(f"Warning: Image not found: {image_path}")

    # Connect nodes
    links.new(tex_image.outputs['Color'], emission.inputs['Color'])
    links.new(emission.outputs['Emission'], output.inputs['Surface'])

    return mat, tex_image


def create_bake_target_image(name, width, height):
    """Create a new image to bake into."""
    img = bpy.data.images.new(name, width=width, height=height, alpha=True)
    img.alpha_mode = 'STRAIGHT'
    return img


def create_principled_material(name, base_color_image):
    """
    Create a Principled BSDF material with baked base color.
    """
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear default nodes
    nodes.clear()

    # Create nodes
    output = nodes.new('ShaderNodeOutputMaterial')
    principled = nodes.new('ShaderNodeBsdfPrincipled')
    tex_image = nodes.new('ShaderNodeTexImage')

    # Assign baked image
    tex_image.image = base_color_image

    # Set material properties for "baked" look
    principled.inputs['Roughness'].default_value = 1.0
    principled.inputs['Metallic'].default_value = 0.0

    # Connect nodes
    links.new(tex_image.outputs['Color'], principled.inputs['Base Color'])
    links.new(principled.outputs['BSDF'], output.inputs['Surface'])

    return mat


def bake_object_color(obj, source_image_path, bake_resolution):
    """
    Bake color from source image onto object using camera projection.

    Process:
        1. Create emission material from source image
        2. UV unwrap object with Smart UV
        3. Create target bake image
        4. Bake emission to target
        5. Replace material with Principled using baked texture

    Returns:
        Path to saved baked texture
    """
    print(f"  Baking color for: {obj.name}")

    # Ensure UV layer
    create_smart_uv(obj)

    # Create emission material (source)
    emit_mat, emit_tex = create_emission_material(
        f"{obj.name}_EmitSource",
        source_image_path
    )

    # Assign emission material
    obj.data.materials.clear()
    obj.data.materials.append(emit_mat)

    # Create bake target image
    bake_img = create_bake_target_image(
        f"{obj.name}_BakedBC",
        bake_resolution,
        bake_resolution
    )

    # Add target texture node to material (must be active for baking)
    nodes = emit_mat.node_tree.nodes
    target_tex = nodes.new('ShaderNodeTexImage')
    target_tex.image = bake_img
    nodes.active = target_tex  # This tells Blender where to bake

    # Select object for baking
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    # Bake settings
    bpy.context.scene.render.bake.use_pass_direct = False
    bpy.context.scene.render.bake.use_pass_indirect = False
    bpy.context.scene.render.bake.margin = 2

    # Bake emission
    try:
        bpy.ops.object.bake(type='EMIT')
    except Exception as e:
        print(f"    Warning: Bake failed for {obj.name}: {e}")
        return None

    # Save baked image
    output_dir = os.path.dirname(OUTPUT_BLEND)
    os.makedirs(os.path.join(output_dir, "textures"), exist_ok=True)
    bake_path = os.path.join(output_dir, "textures", f"{obj.name}_baked.png")

    bake_img.filepath_raw = bake_path
    bake_img.file_format = 'PNG'
    bake_img.save()

    # Replace with Principled material
    final_mat = create_principled_material(f"{obj.name}_Material", bake_img)
    obj.data.materials.clear()
    obj.data.materials.append(final_mat)

    print(f"    Saved baked texture: {bake_path}")
    return bake_path


def join_small_meshes(objects, min_verts=100):
    """
    Join very small meshes together to reduce object count.
    Only joins meshes that are truly tiny.
    """
    small = [o for o in objects if len(o.data.vertices) < min_verts]
    large = [o for o in objects if len(o.data.vertices) >= min_verts]

    if len(small) < 2:
        return objects

    print(f"Joining {len(small)} small meshes...")

    # Select small meshes
    bpy.ops.object.select_all(action='DESELECT')
    for obj in small:
        obj.select_set(True)

    if small:
        bpy.context.view_layer.objects.active = small[0]
        bpy.ops.object.join()
        small[0].name = "SmallMeshes_Combined"

        return large + [small[0]]

    return objects


def main():
    """Main function."""
    print("\n=== Blender Import and Bake Pipeline ===")
    print(f"Meshes directory: {MESHES_DIR}")
    print(f"Color image: {COLOR_IMAGE}")
    print(f"Output: {OUTPUT_BLEND}")
    print(f"Bake resolution: {BAKE_RES}")
    print(f"Scale factor: {SCALE_FACTOR}")

    # Clear scene
    print("\nClearing scene...")
    clear_scene()

    # Setup render engine
    setup_render_engine()

    # Create camera
    print("Creating camera...")
    create_camera_from_json(CAMERA_JSON)

    # Import meshes
    print(f"\nImporting meshes from {MESHES_DIR}...")
    imported = import_meshes(MESHES_DIR, SCALE_FACTOR)

    if not imported:
        print("ERROR: No meshes imported!")
        sys.exit(1)

    print(f"Imported {len(imported)} objects")

    # Optionally join small meshes
    imported = join_small_meshes(imported)

    # Bake color if we have a source image
    if COLOR_IMAGE and os.path.exists(COLOR_IMAGE) and not NO_BAKE:
        print(f"\nBaking color from: {COLOR_IMAGE}")

        for obj in imported:
            bake_object_color(obj, COLOR_IMAGE, BAKE_RES)
    else:
        # Just create simple UV maps
        print("\nCreating UV maps (no baking)...")
        for obj in imported:
            create_smart_uv(obj)

            # Create a simple grey material
            mat = bpy.data.materials.new(f"{obj.name}_Material")
            mat.use_nodes = True
            obj.data.materials.clear()
            obj.data.materials.append(mat)

    # Set smooth shading
    print("\nApplying smooth shading...")
    for obj in imported:
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.shade_smooth()

    # Save blend file
    print(f"\nSaving: {OUTPUT_BLEND}")
    os.makedirs(os.path.dirname(OUTPUT_BLEND), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=OUTPUT_BLEND)

    print("\n=== Import and Bake Complete ===")
    print(f"  Objects: {len(imported)}")
    print(f"  Output: {OUTPUT_BLEND}")


if __name__ == "__main__":
    main()
