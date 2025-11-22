#!/usr/bin/env python3
"""
3DvramCap Pipeline - Unified Entry Point
=========================================
Single entry point for the UE4 scene capture-to-export pipeline.

Usage:
    # Full pipeline (after RenderDoc extraction)
    python pipeline.py run <scene_dir>

    # Individual steps
    python pipeline.py extract <capture.rdc> [--out export/]
    python pipeline.py process <mesh_dir> [--out library/]
    python pipeline.py export <mesh_dir> [--gltf out.glb] [--usd out.usdc]
    python pipeline.py validate <file_or_dir>

    # Show extraction command (for RenderDoc environment)
    python pipeline.py extract-cmd <capture.rdc>

Pipeline Flow:
    1. EXTRACT  - RenderDoc capture → OBJ meshes + textures
    2. PROCESS  - Deduplicate meshes
    3. EXPORT   - Blender import → glTF/USD export
    4. VALIDATE - Verify output integrity

Author: 3DvramCap
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CONFIG = {
    'scale': 0.01,           # UE4 cm -> Blender m
    'min_vertices': 50,      # Minimum vertex count
    'bake_resolution': 2048, # Texture bake resolution
    'blender': 'blender',    # Blender executable
}


def get_core_script(name: str) -> Path:
    """Get path to a core module script."""
    return Path(__file__).parent / 'core' / name


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML or use defaults."""
    config = DEFAULT_CONFIG.copy()

    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'

    if Path(config_path).exists():
        try:
            import yaml
            with open(config_path) as f:
                user_config = yaml.safe_load(f)
            if user_config:
                config['scale'] = user_config.get('units', {}).get('scale_factor', config['scale'])
                config['bake_resolution'] = user_config.get('bake', {}).get('resolution', config['bake_resolution'])
                config['blender'] = user_config.get('tools', {}).get('blender', config['blender'])
        except ImportError:
            pass  # YAML not available, use defaults

    return config


# =============================================================================
# Pipeline Steps
# =============================================================================

def run_command(cmd: list, description: str, dry_run: bool = False) -> bool:
    """Run a shell command with logging."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"$ {' '.join(str(c) for c in cmd)}")

    if dry_run:
        print("[DRY RUN] Skipped")
        return True

    try:
        result = subprocess.run(cmd, check=True)
        print(f"[OK] {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] {description}: exit code {e.returncode}")
        return False
    except FileNotFoundError as e:
        print(f"[ERROR] Command not found: {e}")
        return False


def cmd_extract_show(rdc_path: str, output_dir: str = 'export'):
    """Show the RenderDoc extraction command to run."""
    script = get_core_script('core_extract.py')

    print("\n" + "="*60)
    print("RenderDoc Extraction Command")
    print("="*60)
    print("\nRun this command in RenderDoc's Python environment:\n")
    print(f"  renderdoccmd python {script} \\")
    print(f"    --rdc {rdc_path} \\")
    print(f"    --out {output_dir}")
    print()


def cmd_process(input_dir: str, output_dir: str = 'library', config: dict = None) -> bool:
    """Run mesh deduplication."""
    config = config or DEFAULT_CONFIG
    script = get_core_script('core_process.py')

    cmd = [
        sys.executable, str(script),
        '--input', input_dir,
        '--output', output_dir,
        '--min-vertices', str(config['min_vertices'])
    ]

    return run_command(cmd, "Mesh Deduplication")


def cmd_export_blender(mesh_dir: str, color_image: str = None,
                       gltf_path: str = None, usd_path: str = None,
                       blend_path: str = None, config: dict = None) -> bool:
    """Run Blender export."""
    config = config or DEFAULT_CONFIG
    script = get_core_script('core_blender.py')

    cmd = [
        config['blender'], '-b',
        '--python', str(script),
        '--',
        '--meshes', mesh_dir,
        '--scale', str(config['scale']),
        '--bake-res', str(config['bake_resolution'])
    ]

    if color_image and Path(color_image).exists():
        cmd.extend(['--color', color_image])
    else:
        cmd.append('--no-bake')

    if gltf_path:
        Path(gltf_path).parent.mkdir(parents=True, exist_ok=True)
        cmd.extend(['--gltf', gltf_path])

    if usd_path:
        Path(usd_path).parent.mkdir(parents=True, exist_ok=True)
        cmd.extend(['--usd', usd_path])

    if blend_path:
        Path(blend_path).parent.mkdir(parents=True, exist_ok=True)
        cmd.extend(['--blend', blend_path])

    return run_command(cmd, "Blender Processing & Export")


def cmd_validate(path: str) -> bool:
    """Validate exported files."""
    script = get_core_script('core_validate.py')

    if Path(path).is_dir():
        cmd = [sys.executable, str(script), '--dir', path]
    else:
        ext = Path(path).suffix.lower()
        if ext in ['.glb', '.gltf']:
            cmd = [sys.executable, str(script), '--gltf', path]
        elif ext in ['.usd', '.usda', '.usdc']:
            cmd = [sys.executable, str(script), '--usd', path]
        else:
            print(f"Unknown file type: {ext}")
            return False

    return run_command(cmd, "Validation")


def cmd_run_pipeline(scene_dir: str, targets_dir: str = 'targets', config: dict = None) -> bool:
    """
    Run the full pipeline on an extracted scene.

    Expects:
        scene_dir/
            Meshes/     - OBJ files
            Textures/   - PNG files including final_color.png
            scene.json  - Extraction manifest
    """
    config = config or load_config()
    scene_path = Path(scene_dir)
    scene_name = scene_path.name

    # Verify extraction exists
    mesh_dir = scene_path / 'Meshes'
    tex_dir = scene_path / 'Textures'

    if not mesh_dir.exists():
        print(f"ERROR: Mesh directory not found: {mesh_dir}")
        print("Run RenderDoc extraction first.")
        return False

    print(f"\n{'#'*60}")
    print(f"PIPELINE: {scene_name}")
    print(f"{'#'*60}")

    # Step 1: Process meshes (deduplication)
    library_dir = Path(targets_dir) / 'library' / scene_name
    if not cmd_process(str(mesh_dir), str(library_dir), config):
        return False

    # Step 2: Blender export
    color_image = tex_dir / 'final_color.png'
    if not color_image.exists():
        color_image = tex_dir / 'final.png'
    if not color_image.exists():
        color_image = None
    else:
        color_image = str(color_image)

    gltf_path = Path(targets_dir) / 'gltf' / f'{scene_name}.glb'
    usd_path = Path(targets_dir) / 'usd' / f'{scene_name}.usdc'
    blend_path = Path(targets_dir) / 'blender' / f'{scene_name}.blend'

    if not cmd_export_blender(str(library_dir), color_image,
                              str(gltf_path), str(usd_path), str(blend_path), config):
        return False

    # Step 3: Validate
    cmd_validate(str(Path(targets_dir) / 'gltf'))

    # Summary
    print(f"\n{'#'*60}")
    print("PIPELINE COMPLETE")
    print(f"{'#'*60}")
    print(f"  Scene:   {scene_name}")
    print(f"  glTF:    {gltf_path}")
    print(f"  USD:     {usd_path}")
    print(f"  Blender: {blend_path}")
    print(f"{'#'*60}\n")

    return True


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="3DvramCap Pipeline - UE4 Scene Capture to glTF/USD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py run export/my_scene/
  python pipeline.py extract-cmd captures/scene.rdc
  python pipeline.py process export/Meshes/ --out library/
  python pipeline.py export library/meshes/ --gltf scene.glb
  python pipeline.py validate targets/gltf/
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # run - full pipeline
    run_parser = subparsers.add_parser('run', help='Run full pipeline on extracted scene')
    run_parser.add_argument('scene_dir', help='Extracted scene directory')
    run_parser.add_argument('--targets', default='targets', help='Output directory')
    run_parser.add_argument('--config', help='Config file path')

    # extract-cmd - show extraction command
    extract_parser = subparsers.add_parser('extract-cmd', help='Show RenderDoc extraction command')
    extract_parser.add_argument('rdc_file', help='RenderDoc capture file')
    extract_parser.add_argument('--out', default='export', help='Output directory')

    # process - deduplication
    process_parser = subparsers.add_parser('process', help='Deduplicate meshes')
    process_parser.add_argument('input_dir', help='Input mesh directory')
    process_parser.add_argument('--out', default='library', help='Output directory')

    # export - blender export
    export_parser = subparsers.add_parser('export', help='Export to glTF/USD via Blender')
    export_parser.add_argument('mesh_dir', help='Input mesh directory')
    export_parser.add_argument('--gltf', help='Output glTF path')
    export_parser.add_argument('--usd', help='Output USD path')
    export_parser.add_argument('--blend', help='Output Blender file path')
    export_parser.add_argument('--color', help='Color image for baking')
    export_parser.add_argument('--config', help='Config file path')

    # validate
    validate_parser = subparsers.add_parser('validate', help='Validate exports')
    validate_parser.add_argument('path', help='File or directory to validate')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    config = load_config(getattr(args, 'config', None))

    if args.command == 'run':
        success = cmd_run_pipeline(args.scene_dir, args.targets, config)
        sys.exit(0 if success else 1)

    elif args.command == 'extract-cmd':
        cmd_extract_show(args.rdc_file, args.out)

    elif args.command == 'process':
        success = cmd_process(args.input_dir, args.out, config)
        sys.exit(0 if success else 1)

    elif args.command == 'export':
        if not args.gltf and not args.usd and not args.blend:
            print("ERROR: Specify at least one output: --gltf, --usd, or --blend")
            sys.exit(1)
        success = cmd_export_blender(args.mesh_dir, args.color,
                                     args.gltf, args.usd, args.blend, config)
        sys.exit(0 if success else 1)

    elif args.command == 'validate':
        success = cmd_validate(args.path)
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
