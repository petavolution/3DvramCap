#!/usr/bin/env python3
"""
Pipeline Runner Script
======================
Orchestrates the full capture-to-export pipeline.

Usage:
    python run_pipeline.py --capture scene_01.rdc
    python run_pipeline.py --all               # Process all .rdc files
    python run_pipeline.py --ninja-ripper DIR  # Convert Ninja Ripper .rip files
    python run_pipeline.py --depth-capture DIR # Process ReShade depth captures

This script coordinates:
    1. RenderDoc extraction (requires RenderDoc environment)
    2. Mesh deduplication
    3. Texture processing (DDS decompression, PBR classification)
    4. Blender import and baking
    5. glTF/USD export
    6. Export validation
    7. QA validation

Additional tools:
    - Ninja Ripper conversion (DX9 fallback)
    - ReShade depth reconstruction
    - DDS texture utilities

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

try:
    from logging_utils import setup_logging, get_logger, StepLogger
    HAS_LOGGING = True
except ImportError:
    HAS_LOGGING = False
    # Fallback simple logger
    class SimpleLogger:
        def info(self, msg, **kwargs): print(f"[INFO] {msg}")
        def warning(self, msg, **kwargs): print(f"[WARN] {msg}")
        def error(self, msg, **kwargs): print(f"[ERROR] {msg}")
        def debug(self, msg, **kwargs): pass
    def get_logger(name): return SimpleLogger()
    def setup_logging(**kwargs): pass
    class StepLogger:
        def __init__(self, name, logger=None): self.name = name
        def start(self, msg=None): print(f"[START] {self.name}")
        def success(self, msg=None, **kw): print(f"[OK] {msg or self.name}")
        def failure(self, msg=None, **kw): print(f"[FAIL] {msg or self.name}")
        def __enter__(self): self.start(); return self
        def __exit__(self, *args): pass


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

        # Step 3.5: Texture Processing (DDS decompression, classification)
        if os.path.exists(tex_dir):
            run_texture_processing(config, tex_dir, dry_run)

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

        # Step 5: Export Validation
        targets_dir = os.path.dirname(gltf_path)
        run_export_validation(config, targets_dir, dry_run)

        # Step 6: QA Validation (if reference available)
        ref_image = color_image if os.path.exists(color_image) else None
        if ref_image:
            # For full QA, we'd need to render from Blender first
            # This is a simplified version that just checks the scene
            report_path = os.path.join(paths["qa"], f"{capture_name}_report.json")

            print("\n" + "="*60)
            print("Step 6: QA Validation")
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


def run_ninja_ripper(config, input_dir, dry_run=False):
    """
    Run Ninja Ripper conversion for DX9 fallback captures.
    """
    logger = get_logger("pipeline")
    tools = config["tools"]
    paths = config["paths"]

    output_dir = os.path.join(paths["export"], "ninja_rip", "Meshes")

    with StepLogger("ninja_ripper_conversion", logger) as step:
        cmd = [
            tools["python"],
            get_script_path("07_ninja_ripper_convert.py"),
            "--input", input_dir,
            "--output", output_dir,
            "--min-vertices", "50"
        ]

        if run_command(cmd, "Ninja Ripper Conversion", dry_run):
            step.success(f"Converted .rip files to {output_dir}")
            return True
        else:
            step.failure("Ninja Ripper conversion failed")
            return False


def run_depth_capture(config, input_dir, dry_run=False):
    """
    Run ReShade depth capture reconstruction.
    """
    logger = get_logger("pipeline")
    tools = config["tools"]
    paths = config["paths"]

    output_dir = os.path.join(paths["export"], "depth_capture")

    with StepLogger("depth_reconstruction", logger) as step:
        cmd = [
            tools["python"],
            get_script_path("06_depth_reconstruction.py"),
            "--input", input_dir,
            "--output", output_dir,
            "--method", "poisson"
        ]

        if run_command(cmd, "Depth Reconstruction", dry_run):
            step.success(f"Reconstructed depth captures to {output_dir}")
            return True
        else:
            step.failure("Depth reconstruction failed")
            return False


def run_texture_processing(config, tex_dir, dry_run=False):
    """
    Process textures: DDS decompression and PBR classification.
    """
    logger = get_logger("pipeline")
    tools = config["tools"]

    with StepLogger("texture_processing", logger) as step:
        cmd = [
            tools["python"],
            get_script_path("08_texture_utils.py"),
            "--input", tex_dir,
            "--output", tex_dir,
            "--decompress",
            "--classify"
        ]

        if run_command(cmd, "Texture Processing", dry_run):
            step.success(f"Processed textures in {tex_dir}")
            return True
        else:
            step.failure("Texture processing failed")
            return False


def run_export_validation(config, export_dir, dry_run=False):
    """
    Validate exported glTF and USD files.
    """
    logger = get_logger("pipeline")
    tools = config["tools"]
    paths = config["paths"]

    report_path = os.path.join(paths["qa"], "validation_report.json")

    with StepLogger("export_validation", logger) as step:
        cmd = [
            tools["python"],
            get_script_path("09_validate_exports.py"),
            "--all", export_dir,
            "--output", report_path
        ]

        if run_command(cmd, "Export Validation", dry_run):
            step.success(f"Validation report: {report_path}")
            return True
        else:
            step.failure("Export validation failed")
            return False


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
        "--ninja-ripper",
        metavar="DIR",
        help="Convert Ninja Ripper .rip files from directory"
    )
    parser.add_argument(
        "--depth-capture",
        metavar="DIR",
        help="Process ReShade depth captures from directory"
    )
    parser.add_argument(
        "--validate",
        metavar="DIR",
        help="Validate exports in directory"
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
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)
    logger = get_logger("pipeline")

    # Find config file
    config_path = args.config
    if not os.path.exists(config_path):
        # Try in script directory
        script_dir = Path(__file__).parent.parent
        config_path = str(script_dir / "config.yaml")

    config = load_config(config_path)

    # Handle different modes
    if args.ninja_ripper:
        logger.info(f"Running Ninja Ripper conversion: {args.ninja_ripper}")
        run_ninja_ripper(config, args.ninja_ripper, args.dry_run)
        return

    if args.depth_capture:
        logger.info(f"Running depth capture reconstruction: {args.depth_capture}")
        run_depth_capture(config, args.depth_capture, args.dry_run)
        return

    if args.validate:
        logger.info(f"Running export validation: {args.validate}")
        run_export_validation(config, args.validate, args.dry_run)
        return

    if not args.capture and not args.all:
        print("Error: Specify --capture FILE, --all, --ninja-ripper DIR, --depth-capture DIR, or --validate DIR")
        parser.print_help()
        sys.exit(1)

    run_pipeline(config, args.capture, args.dry_run)


if __name__ == "__main__":
    main()
