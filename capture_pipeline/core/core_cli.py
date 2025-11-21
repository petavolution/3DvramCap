#!/usr/bin/env python3
"""
Unified Command-Line Interface
==============================
Single entry point for all pipeline operations.

Usage:
    python -m core_cli extract capture.rdc --output export/
    python -m core_cli process export/meshes/ --deduplicate
    python -m core_cli export library/ --gltf scene.glb
    python -m core_cli batch captures/ --auto-approve
    python -m core_cli library import meshes/
    python -m core_cli status

Author: Capture Pipeline
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Import core modules
try:
    from core_config import Config, get_config, load_config, create_default_config
    from core_logging import init_logging, get_logger, TaskLogger
    from core_database import get_database, CaptureStatus
    from core_library import AssetLibrary
    from core_batch import BatchProcessor, BatchConfig
    from core_quality import QualityChecker, assess_scene_quality
except ImportError:
    # Handle relative imports
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from core_config import Config, get_config, load_config, create_default_config
    from core_logging import init_logging, get_logger, TaskLogger
    from core_database import get_database, CaptureStatus
    from core_library import AssetLibrary
    from core_batch import BatchProcessor, BatchConfig
    from core_quality import QualityChecker, assess_scene_quality


def cmd_init(args):
    """Initialize a new pipeline project."""
    project_dir = Path(args.directory)

    if project_dir.exists() and list(project_dir.iterdir()):
        if not args.force:
            print(f"Directory {project_dir} is not empty. Use --force to override.")
            return 1

    # Create directory structure
    dirs = ['captures', 'export', 'work', 'logs', 'library', 'targets']
    for d in dirs:
        (project_dir / d).mkdir(parents=True, exist_ok=True)

    # Create default config
    config = Config()
    if args.profile:
        config.apply_profile(args.profile)

    config.save(str(project_dir / "pipeline.json"))

    print(f"Initialized pipeline project in {project_dir}")
    print(f"  Config: pipeline.json")
    print(f"  Directories: {', '.join(dirs)}")
    return 0


def cmd_extract(args):
    """Extract meshes from RenderDoc capture."""
    logger = get_logger("cli.extract")

    with TaskLogger("extract", capture=args.capture) as task:
        # This would call the extraction script
        # In practice, this runs via renderdoccmd
        task.info("Starting extraction")

        extract_script = Path(__file__).parent / "core_extract.py"

        if not extract_script.exists():
            task.error("core_extract.py not found")
            return 1

        import subprocess
        cmd = [
            "renderdoccmd", "python",
            str(extract_script),
            args.capture,
            args.output,
            "--max-meshes", str(args.max_meshes),
            "--min-vertices", str(args.min_vertices)
        ]

        task.info(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                task.info("Extraction complete")
                print(result.stdout)
            else:
                task.error(f"Extraction failed: {result.stderr}")
                return 1
        except FileNotFoundError:
            task.error("renderdoccmd not found. Run extraction manually.")
            print(f"\nManual command:")
            print(f"  {' '.join(cmd)}")
            return 1

    return 0


def cmd_process(args):
    """Process extracted meshes (deduplication, optimization)."""
    logger = get_logger("cli.process")

    with TaskLogger("process", input=args.input) as task:
        task.info("Starting processing")

        process_script = Path(__file__).parent / "core_process.py"

        import subprocess
        cmd = [
            sys.executable,
            str(process_script),
            args.input,
            args.output or str(Path(args.input).parent / "deduplicated"),
            "--min-vertices", str(args.min_vertices)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            task.info("Processing complete")
            print(result.stdout)
        else:
            task.error(f"Processing failed: {result.stderr}")
            return 1

    return 0


def cmd_export(args):
    """Export to target formats."""
    logger = get_logger("cli.export")
    config = get_config()

    with TaskLogger("export", input=args.input) as task:
        # Determine export method
        if args.direct:
            # Use direct export (no Blender)
            task.info("Using direct export (no Blender)")

            if args.gltf:
                from core_gltf import GltfBuilder
                from core_merge import load_obj

                builder = GltfBuilder()

                for obj_file in Path(args.input).glob("*.obj"):
                    mesh = load_obj(str(obj_file))
                    if mesh:
                        builder.add_mesh(
                            mesh.name,
                            mesh.vertices,
                            mesh.faces,
                            mesh.normals,
                            mesh.uvs
                        )

                builder.save(args.gltf)
                task.info(f"Saved glTF: {args.gltf}")

            if args.usd:
                from core_usd import UsdBuilder
                from core_merge import load_obj

                builder = UsdBuilder()

                for obj_file in Path(args.input).glob("*.obj"):
                    mesh = load_obj(str(obj_file))
                    if mesh:
                        builder.add_mesh(
                            mesh.name,
                            mesh.vertices.tolist(),
                            mesh.faces.tolist(),
                            mesh.normals.tolist() if mesh.normals is not None else None,
                            mesh.uvs.tolist() if mesh.uvs is not None else None
                        )

                builder.save(args.usd)
                task.info(f"Saved USD: {args.usd}")

        else:
            # Use Blender export
            task.info("Using Blender export")

            blender_script = Path(__file__).parent / "core_blender.py"
            blender_path = config.tools.blender_path or "blender"

            cmd = [
                blender_path, "-b", "--python", str(blender_script),
                "--",
                "--meshes", args.input,
                "--scale", str(config.processing.scale_factor)
            ]

            if args.gltf:
                cmd.extend(["--gltf", args.gltf])
            if args.usd:
                cmd.extend(["--usd", args.usd])
            if args.blend:
                cmd.extend(["--blend", args.blend])
            if config.export.use_unlit:
                cmd.append("--unlit")

            import subprocess
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                task.info("Export complete")
            else:
                task.error(f"Export failed: {result.stderr}")
                return 1

    return 0


def cmd_batch(args):
    """Batch process multiple captures."""
    logger = get_logger("cli.batch")

    # Load or create config
    if args.config:
        batch_config = BatchConfig.load(args.config)
    else:
        batch_config = BatchConfig(
            output_dir=args.output or "export",
            max_parallel=args.parallel
        )

    processor = BatchProcessor(batch_config)

    # Add captures
    input_path = Path(args.input)
    if input_path.is_dir():
        processor.add_captures_from_directory(str(input_path))
    elif input_path.is_file():
        processor.add_capture(str(input_path))
    else:
        print(f"Input not found: {input_path}")
        return 1

    # Run
    results = processor.run(auto_approve=args.auto_approve)

    # Generate report
    if args.report:
        processor.generate_report(args.report)

    # Summary
    completed = sum(1 for r in results.values() if r.status.value == 'completed')
    failed = sum(1 for r in results.values() if r.status.value == 'failed')

    print(f"\nBatch complete: {completed} completed, {failed} failed")

    return 0 if failed == 0 else 1


def cmd_library(args):
    """Manage asset library."""
    library = AssetLibrary(args.library)

    if args.action == 'import':
        if args.type == 'meshes':
            results = library.import_meshes(args.path, tags=args.tags)
        else:
            results = library.import_textures(args.path, tags=args.tags)

        print(f"Added: {results['added']}")
        print(f"Duplicates: {results['duplicates']}")
        print(f"Errors: {results['errors']}")

    elif args.action == 'search':
        meshes = library.search_meshes(
            name_pattern=args.pattern,
            tags=args.tags,
            vertex_count_min=args.min_verts,
            has_uvs=args.has_uvs
        )

        for mesh in meshes:
            print(f"[{mesh.id}] {mesh.name}")
            print(f"    Vertices: {mesh.vertex_count}, Faces: {mesh.face_count}")
            print(f"    UVs: {mesh.has_uvs}, Normals: {mesh.has_normals}")
            if mesh.tags:
                print(f"    Tags: {', '.join(mesh.tags)}")

    elif args.action == 'export':
        count = library.export_meshes(args.output, mesh_ids=args.ids)
        print(f"Exported {count} meshes")

    elif args.action == 'stats':
        stats = library.get_stats()
        print(json.dumps(stats, indent=2))

    return 0


def cmd_validate(args):
    """Validate exports."""
    logger = get_logger("cli.validate")

    if Path(args.path).is_dir():
        report = assess_scene_quality(args.path)
        report.print_summary()

        if args.output:
            if args.output.endswith('.html'):
                report.to_html(args.output)
            else:
                report.to_json(args.output)
    else:
        from core_quality import assess_mesh_quality
        metrics = assess_mesh_quality(args.path)
        print(f"Score: {metrics.score}/100 ({metrics.level.value})")
        if metrics.issues:
            print("Issues:", ', '.join(metrics.issues))

    return 0


def cmd_status(args):
    """Show pipeline status."""
    config = get_config()
    db = get_database()

    print("Pipeline Status")
    print("=" * 40)

    # Tool status
    print("\nTools:")
    found = config.tools.detect_tools()
    for tool, status in found.items():
        path = getattr(config.tools, f"{tool}_path", "")
        indicator = "[OK]" if status else "[--]"
        print(f"  {indicator} {tool}: {path or 'not found'}")

    # Database stats
    print("\nDatabase:")
    stats = db.get_stats()
    print(f"  Captures: {stats['total_captures']}")
    print(f"  Meshes: {stats['total_meshes']} ({stats['unique_meshes']} unique)")
    print(f"  Exports: {stats['total_exports']}")

    # Library stats
    try:
        library = AssetLibrary(config.paths.library_dir)
        lib_stats = library.get_stats()
        print(f"\nLibrary:")
        print(f"  Meshes: {lib_stats['mesh_count']}")
        print(f"  Textures: {lib_stats['texture_count']}")
    except Exception:
        pass

    return 0


def cmd_config(args):
    """Manage configuration."""
    if args.action == 'show':
        config = get_config()
        print(json.dumps(config.to_dict(), indent=2))

    elif args.action == 'create':
        config = Config()
        if args.profile:
            config.apply_profile(args.profile)
        config.save(args.output or "pipeline.json")
        print(f"Created: {args.output or 'pipeline.json'}")

    elif args.action == 'validate':
        config = Config.load(args.file)
        issues = config.validate()
        if issues:
            print("Validation issues:")
            for issue in issues:
                print(f"  - {issue}")
            return 1
        else:
            print("Configuration valid")

    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Capture Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  init       Initialize a new pipeline project
  extract    Extract meshes from RenderDoc capture
  process    Process meshes (deduplicate, optimize)
  export     Export to glTF/USD/Blender formats
  batch      Batch process multiple captures
  library    Manage asset library
  validate   Validate exports
  status     Show pipeline status
  config     Manage configuration

Examples:
  %(prog)s init my_project
  %(prog)s extract capture.rdc --output export/
  %(prog)s batch captures/ --auto-approve
  %(prog)s library import --type meshes meshes/
  %(prog)s validate export/scene/
"""
    )

    parser.add_argument('--config', '-c', help="Config file path")
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--log-format', default='text', choices=['text', 'json'])

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # init
    p_init = subparsers.add_parser('init', help='Initialize pipeline project')
    p_init.add_argument('directory', nargs='?', default='.', help='Project directory')
    p_init.add_argument('--profile', choices=['default', 'fast', 'quality', 'minimal'])
    p_init.add_argument('--force', '-f', action='store_true')

    # extract
    p_extract = subparsers.add_parser('extract', help='Extract from RenderDoc')
    p_extract.add_argument('capture', help='RenderDoc capture file')
    p_extract.add_argument('--output', '-o', default='export', help='Output directory')
    p_extract.add_argument('--max-meshes', type=int, default=500)
    p_extract.add_argument('--min-vertices', type=int, default=50)

    # process
    p_process = subparsers.add_parser('process', help='Process meshes')
    p_process.add_argument('input', help='Input mesh directory')
    p_process.add_argument('--output', '-o', help='Output directory')
    p_process.add_argument('--min-vertices', type=int, default=50)
    p_process.add_argument('--deduplicate', action='store_true', default=True)

    # export
    p_export = subparsers.add_parser('export', help='Export to formats')
    p_export.add_argument('input', help='Input mesh directory')
    p_export.add_argument('--gltf', help='Output glTF path')
    p_export.add_argument('--usd', help='Output USD path')
    p_export.add_argument('--blend', help='Output Blender path')
    p_export.add_argument('--direct', action='store_true',
                         help='Use direct export (no Blender)')

    # batch
    p_batch = subparsers.add_parser('batch', help='Batch process captures')
    p_batch.add_argument('input', help='Capture file or directory')
    p_batch.add_argument('--output', '-o', help='Output directory')
    p_batch.add_argument('--config', dest='batch_config', help='Batch config file')
    p_batch.add_argument('--auto-approve', '-y', action='store_true')
    p_batch.add_argument('--parallel', '-p', type=int, default=1)
    p_batch.add_argument('--report', '-r', help='Generate HTML report')

    # library
    p_library = subparsers.add_parser('library', help='Manage asset library')
    p_library.add_argument('action', choices=['import', 'search', 'export', 'stats'])
    p_library.add_argument('path', nargs='?', help='Path for import/export')
    p_library.add_argument('--library', '-l', default='library')
    p_library.add_argument('--type', choices=['meshes', 'textures'], default='meshes')
    p_library.add_argument('--tags', nargs='+', help='Tags to add/filter')
    p_library.add_argument('--pattern', help='Search pattern')
    p_library.add_argument('--min-verts', type=int, help='Minimum vertices')
    p_library.add_argument('--has-uvs', action='store_true')
    p_library.add_argument('--ids', nargs='+', help='Asset IDs for export')
    p_library.add_argument('--output', '-o', help='Export output directory')

    # validate
    p_validate = subparsers.add_parser('validate', help='Validate exports')
    p_validate.add_argument('path', help='File or directory to validate')
    p_validate.add_argument('--output', '-o', help='Output report path')

    # status
    p_status = subparsers.add_parser('status', help='Show pipeline status')

    # config
    p_config = subparsers.add_parser('config', help='Manage configuration')
    p_config.add_argument('action', choices=['show', 'create', 'validate'])
    p_config.add_argument('--file', '-f', help='Config file path')
    p_config.add_argument('--output', '-o', help='Output path for create')
    p_config.add_argument('--profile', choices=['default', 'fast', 'quality', 'minimal'])

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Initialize logging
    init_logging(level=args.log_level, format=args.log_format)

    # Load config if specified
    if args.config:
        load_config(args.config)

    # Dispatch command
    commands = {
        'init': cmd_init,
        'extract': cmd_extract,
        'process': cmd_process,
        'export': cmd_export,
        'batch': cmd_batch,
        'library': cmd_library,
        'validate': cmd_validate,
        'status': cmd_status,
        'config': cmd_config
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
