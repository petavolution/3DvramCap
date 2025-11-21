#!/usr/bin/env python3
"""
Pipeline Runner Script
======================
Orchestrates the full capture-to-export pipeline.

Usage:
    python run_pipeline.py --capture scene_01.rdc
    python run_pipeline.py --all               # Process all .rdc files

This script coordinates:
    1. RenderDoc extraction (requires RenderDoc environment)
    2. Mesh deduplication
    3. Blender import and baking
    4. glTF/USD export
    5. QA validation

Note: Step 1 (RenderDoc extraction) must be run inside RenderDoc's Python
environment. This script will generate the commands to run.

Author: Capture Pipeline
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def load_config(config_path):
    """Load configuration from YAML or use defaults."""
    defaults = {
        "paths": {
            "captures": "captures",
            "export": "export",
            "library": "library",
            "targets": {
                "blender": "targets/blender",
                "gltf": "targets/gltf",
                "usd": "targets/usd"
            },
            "qa": "qa"
        },
        "units": {
            "scale_factor": 0.01
        },
        "bake": {
            "resolution": 2048
        },
        "tools": {
            "blender": "blender",
            "python": "python"
        }
    }

    if not os.path.exists(config_path):
        print(f"Config not found: {config_path}, using defaults")
        return defaults

    if HAS_YAML:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            # Merge with defaults
            for key, value in defaults.items():
                if key not in config:
                    config[key] = value
            return config
    else:
        print("PyYAML not installed, using defaults")
        return defaults


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path


def run_command(cmd, description, dry_run=False):
    """Run a shell command with logging."""
    print(f"\n{'='*60}")
    print(f"Step: {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")

    if dry_run:
        print("[DRY RUN] Command not executed")
        return True

    try:
        result = subprocess.run(cmd, check=True)
        print(f"[OK] {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {description} failed: {e}")
        return False
    except FileNotFoundError as e:
        print(f"[ERROR] Command not found: {e}")
        return False


def get_script_path(script_name):
    """Get full path to a script in the scripts directory."""
    script_dir = Path(__file__).parent
    return str(script_dir / script_name)


def find_captures(captures_dir):
    """Find all .rdc files in captures directory."""
    captures_path = Path(captures_dir)
    if not captures_path.exists():
        return []
    return list(captures_path.glob("*.rdc"))


def run_pipeline(config, capture_file=None, dry_run=False):
    """
    Run the full pipeline for a capture file.

    Args:
        config: Configuration dict
        capture_file: Specific .rdc file or None to process all
        dry_run: If True, just print commands without running
    """
    paths = config["paths"]
    tools = config["tools"]

    # Find captures
    if capture_file:
        captures = [Path(capture_file)]
    else:
        captures = find_captures(paths["captures"])

    if not captures:
        print("No capture files found!")
        return False

    print(f"\nFound {len(captures)} capture file(s)")

    for capture_path in captures:
        print(f"\n{'#'*60}")
        print(f"Processing: {capture_path.name}")
        print(f"{'#'*60}")

        capture_name = capture_path.stem

        # Output paths for this capture
        export_dir = os.path.join(paths["export"], capture_name)
        mesh_dir = os.path.join(export_dir, "Meshes")
        tex_dir = os.path.join(export_dir, "Textures")
        library_dir = paths["library"]
        blend_path = os.path.join(paths["targets"]["blender"], f"{capture_name}.blend")
        gltf_path = os.path.join(paths["targets"]["gltf"], f"{capture_name}.glb")
        usd_path = os.path.join(paths["targets"]["usd"], f"{capture_name}.usdc")

        # Ensure directories exist
        for d in [export_dir, library_dir, paths["targets"]["blender"],
                  paths["targets"]["gltf"], paths["targets"]["usd"], paths["qa"]]:
            ensure_dir(d)

        # Step 1: RenderDoc Extraction
        # Note: This must be run in RenderDoc's environment
        print("\n" + "="*60)
        print("Step 1: RenderDoc Extraction")
        print("="*60)
        print("Run this command in RenderDoc's Python environment:")
        print(f"  renderdoccmd python {get_script_path('01_extract_from_rdc.py')} \\")
        print(f"    --rdc {capture_path} \\")
        print(f"    --out {paths['export']}")
        print()

        # Check if extraction already done
        scene_json = os.path.join(export_dir, "scene.json")
        if os.path.exists(scene_json):
            print("  [OK] Extraction already completed (scene.json exists)")
        else:
            print("  [PENDING] Run the above command, then re-run this script")
            if not dry_run:
                continue  # Skip to next capture

        # Step 2: Mesh Deduplication
        index_path = os.path.join(library_dir, f"{capture_name}_index.json")
        library_mesh_dir = os.path.join(library_dir, "meshes")

        cmd = [
            tools["python"],
            get_script_path("02_dedupe_meshes.py"),
            mesh_dir,
            index_path,
            "--copy-to", library_mesh_dir
        ]

        if not run_command(cmd, "Mesh Deduplication", dry_run):
            continue

        # Step 3: Blender Import and Bake
        color_image = os.path.join(tex_dir, "final.png")
        if not os.path.exists(color_image):
            # Try gbuffer_0 as fallback
            color_image = os.path.join(tex_dir, "gbuffer_0.png")

        cmd = [
            tools["blender"], "-b", "--python", get_script_path("03_blender_import_bake.py"),
            "--",
            "--meshes", library_mesh_dir,
            "--out", blend_path,
            "--bake-res", str(config["bake"]["resolution"]),
            "--scale", str(config["units"]["scale_factor"])
        ]

        if os.path.exists(color_image):
            cmd.extend(["--color", color_image])
        else:
            cmd.append("--no-bake")
            print("  Note: No color image found, skipping bake")

        if not run_command(cmd, "Blender Import and Bake", dry_run):
            continue

        # Step 4: Export glTF and USD
        cmd = [
            tools["blender"], "-b", blend_path,
            "--python", get_script_path("04_blender_export.py"),
            "--",
            "--gltf", gltf_path,
            "--usd", usd_path
        ]

        if config.get("export", {}).get("gltf", {}).get("use_unlit", False):
            cmd.append("--unlit")

        if not run_command(cmd, "glTF/USD Export", dry_run):
            continue

        # Step 5: QA Validation (if reference available)
        ref_image = color_image if os.path.exists(color_image) else None
        if ref_image:
            # For full QA, we'd need to render from Blender first
            # This is a simplified version that just checks the scene
            report_path = os.path.join(paths["qa"], f"{capture_name}_report.json")

            print("\n" + "="*60)
            print("Step 5: QA Validation")
            print("="*60)
            print(f"To run full validation, render from Blender and run:")
            print(f"  python {get_script_path('05_qa_validate.py')} \\")
            print(f"    {ref_image} \\")
            print(f"    <rendered_image.png> \\")
            print(f"    {report_path}")

        print(f"\n{'#'*60}")
        print(f"Completed: {capture_name}")
        print(f"  Blend: {blend_path}")
        print(f"  glTF:  {gltf_path}")
        print(f"  USD:   {usd_path}")
        print(f"{'#'*60}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run the UE4 scene capture reconstruction pipeline"
    )
    parser.add_argument(
        "--capture",
        help="Specific .rdc capture file to process"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all .rdc files in captures directory"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing"
    )

    args = parser.parse_args()

    if not args.capture and not args.all:
        print("Error: Specify --capture FILE or --all")
        parser.print_help()
        sys.exit(1)

    # Find config file
    config_path = args.config
    if not os.path.exists(config_path):
        # Try in script directory
        script_dir = Path(__file__).parent.parent
        config_path = str(script_dir / "config.yaml")

    config = load_config(config_path)

    run_pipeline(config, args.capture, args.dry_run)


if __name__ == "__main__":
    main()
