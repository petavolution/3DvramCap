#!/usr/bin/env python3
"""
Core Pipeline Orchestrator
==========================
Simple orchestrator for the core capture-to-export pipeline.

Usage:
    # Full pipeline (after RenderDoc extraction)
    python run_core.py --scene export/my_scene/

    # Individual steps
    python run_core.py --extract capture.rdc --out export/
    python run_core.py --process export/my_scene/Meshes/ --library library/
    python run_core.py --blender library/meshes/ --gltf targets/scene.glb

Pipeline Steps:
    1. EXTRACT: RenderDoc capture → OBJ meshes + textures
       (Run in RenderDoc: renderdoccmd python core_extract.py ...)

    2. PROCESS: Deduplicate meshes
       python core_process.py --input ... --output ...

    3. BLENDER: Import, UV, bake, export
       blender -b --python core_blender.py -- ...

    4. VALIDATE: Check exports
       python core_validate.py --dir targets/

Author: Capture Pipeline
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


# Configuration defaults
DEFAULT_CONFIG = {
    'scale': 0.01,           # UE4 cm -> Blender m
    'min_vertices': 50,      # Minimum vertex count
    'bake_resolution': 2048, # Texture bake resolution
    'blender': 'blender',    # Blender executable
    'python': 'python'       # Python executable
}


def get_script_path(script_name):
    """Get path to a core script."""
    script_dir = Path(__file__).parent
    return str(script_dir / script_name)


def run_command(cmd, description):
    """Run a shell command with logging."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True)
        print(f"[OK] {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] {description}: {e}")
        return False
    except FileNotFoundError as e:
        print(f"[ERROR] Command not found: {e}")
        return False


def step_extract(rdc_path, output_dir, config):
    """
    Step 1: Extract from RenderDoc capture.

    NOTE: This must be run in RenderDoc's Python environment.
    This function generates the command to run.
    """
    print("\n" + "="*60)
    print("STEP 1: RenderDoc Extraction")
    print("="*60)
    print("This step must be run in RenderDoc's Python environment.")
    print("\nRun this command:")
    print(f"  renderdoccmd python {get_script_path('core_extract.py')} \\")
    print(f"    --rdc {rdc_path} \\")
    print(f"    --out {output_dir}")
    print()
    return True


def step_process(mesh_dir, library_dir, config):
    """
    Step 2: Deduplicate and prepare meshes.
    """
    cmd = [
        config['python'],
        get_script_path('core_process.py'),
        '--input', str(mesh_dir),
        '--output', str(library_dir),
        '--min-vertices', str(config['min_vertices'])
    ]

    return run_command(cmd, "Mesh Deduplication")


def step_blender(mesh_dir, color_image, output_gltf, output_usd, output_blend, config):
    """
    Step 3: Blender import, UV, bake, and export.
    """
    cmd = [
        config['blender'], '-b',
        '--python', get_script_path('core_blender.py'),
        '--',
        '--meshes', str(mesh_dir),
        '--scale', str(config['scale']),
        '--bake-res', str(config['bake_resolution'])
    ]

    if color_image and os.path.exists(color_image):
        cmd.extend(['--color', str(color_image)])
    else:
        cmd.append('--no-bake')

    if output_gltf:
        cmd.extend(['--gltf', str(output_gltf)])
    if output_usd:
        cmd.extend(['--usd', str(output_usd)])
    if output_blend:
        cmd.extend(['--blend', str(output_blend)])

    return run_command(cmd, "Blender Processing")


def step_validate(targets_dir, config):
    """
    Step 4: Validate exported files.
    """
    cmd = [
        config['python'],
        get_script_path('core_validate.py'),
        '--dir', str(targets_dir)
    ]

    return run_command(cmd, "Export Validation")


def run_full_pipeline(scene_dir, targets_dir, config):
    """
    Run the full pipeline on an extracted scene.

    Assumes RenderDoc extraction has already been done.

    Args:
        scene_dir: Directory with extracted meshes/textures (from step 1)
        targets_dir: Output directory for exports
        config: Configuration dict
    """
    scene_path = Path(scene_dir)
    targets_path = Path(targets_dir)
    scene_name = scene_path.name

    # Paths
    mesh_dir = scene_path / "Meshes"
    tex_dir = scene_path / "Textures"
    color_image = tex_dir / "final_color.png"

    library_dir = targets_path / "library" / scene_name
    gltf_path = targets_path / "gltf" / f"{scene_name}.glb"
    usd_path = targets_path / "usd" / f"{scene_name}.usdc"
    blend_path = targets_path / "blender" / f"{scene_name}.blend"

    # Check extraction exists
    if not mesh_dir.exists():
        print(f"ERROR: Mesh directory not found: {mesh_dir}")
        print("Run RenderDoc extraction first (Step 1)")
        return False

    print("\n" + "#"*60)
    print(f"PIPELINE: {scene_name}")
    print("#"*60)

    # Step 2: Process meshes
    if not step_process(mesh_dir, library_dir, config):
        return False

    # Step 3: Blender processing
    color_path = str(color_image) if color_image.exists() else None
    if not step_blender(library_dir, color_path, gltf_path, usd_path, blend_path, config):
        return False

    # Step 4: Validate
    step_validate(targets_path, config)

    # Summary
    print("\n" + "#"*60)
    print("PIPELINE COMPLETE")
    print("#"*60)
    print(f"  Scene: {scene_name}")
    print(f"  glTF:  {gltf_path}")
    print(f"  USD:   {usd_path}")
    print(f"  Blend: {blend_path}")
    print("#"*60)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Core pipeline orchestrator for UE4 scene capture"
    )

    # Pipeline modes
    parser.add_argument('--scene',
                        help="Run full pipeline on extracted scene directory")
    parser.add_argument('--extract',
                        help="Generate RenderDoc extraction command for .rdc file")
    parser.add_argument('--process',
                        help="Run mesh deduplication on mesh directory")
    parser.add_argument('--blender',
                        help="Run Blender processing on mesh directory")
    parser.add_argument('--validate',
                        help="Validate exports in directory")

    # Output paths
    parser.add_argument('--out', default='export',
                        help="Output directory (default: export)")
    parser.add_argument('--targets', default='targets',
                        help="Targets directory (default: targets)")
    parser.add_argument('--library', default='library',
                        help="Library directory for processed meshes")
    parser.add_argument('--gltf',
                        help="Output glTF path")
    parser.add_argument('--usd',
                        help="Output USD path")

    # Configuration
    parser.add_argument('--scale', type=float, default=0.01,
                        help="Scale factor (default: 0.01)")
    parser.add_argument('--bake-res', type=int, default=2048,
                        help="Bake resolution (default: 2048)")
    parser.add_argument('--blender-path', default='blender',
                        help="Path to Blender executable")

    args = parser.parse_args()

    # Build config
    config = DEFAULT_CONFIG.copy()
    config['scale'] = args.scale
    config['bake_resolution'] = args.bake_res
    config['blender'] = args.blender_path

    # Execute requested mode
    if args.scene:
        # Full pipeline
        run_full_pipeline(args.scene, args.targets, config)

    elif args.extract:
        # Show extraction command
        step_extract(args.extract, args.out, config)

    elif args.process:
        # Mesh processing only
        step_process(args.process, args.library, config)

    elif args.blender:
        # Blender processing only
        step_blender(args.blender, None, args.gltf, args.usd, None, config)

    elif args.validate:
        # Validation only
        step_validate(args.validate, config)

    else:
        print("Usage examples:")
        print()
        print("  # Full pipeline (after RenderDoc extraction):")
        print("  python run_core.py --scene export/my_scene/")
        print()
        print("  # Show RenderDoc extraction command:")
        print("  python run_core.py --extract capture.rdc --out export/")
        print()
        print("  # Deduplicate meshes only:")
        print("  python run_core.py --process export/scene/Meshes/ --library library/")
        print()
        print("  # Blender processing only:")
        print("  python run_core.py --blender library/meshes/ --gltf scene.glb")
        print()
        print("  # Validate exports:")
        print("  python run_core.py --validate targets/")
        print()
        parser.print_help()


if __name__ == "__main__":
    main()
