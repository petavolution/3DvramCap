#!/usr/bin/env python3
"""
Core Validation Script (Simplified)
====================================
Quick validation of exported glTF and USD files.

Usage:
    python core_validate.py --gltf scene.glb
    python core_validate.py --usd scene.usdc
    python core_validate.py --dir targets/

Features:
    - File integrity checks
    - Basic structure validation
    - Geometry sanity checks (via trimesh)
    - Summary report generation

Author: Capture Pipeline
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Optional dependencies
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


class ValidationResult:
    """Container for validation results."""

    def __init__(self, filepath):
        self.filepath = str(filepath)
        self.filename = Path(filepath).name
        self.format = Path(filepath).suffix.lower()
        self.passed = True
        self.errors = []
        self.warnings = []
        self.info = {}

    def error(self, msg):
        self.errors.append(msg)
        self.passed = False

    def warning(self, msg):
        self.warnings.append(msg)

    def set_info(self, key, value):
        self.info[key] = value

    def to_dict(self):
        return {
            'filepath': self.filepath,
            'filename': self.filename,
            'format': self.format,
            'passed': self.passed,
            'errors': self.errors,
            'warnings': self.warnings,
            'info': self.info
        }


def validate_gltf(filepath):
    """
    Validate glTF/GLB file structure.

    Checks:
        - File exists and is readable
        - Basic JSON structure (for .gltf) or binary header (for .glb)
        - Required asset property
        - Mesh and material references
    """
    result = ValidationResult(filepath)

    if not os.path.exists(filepath):
        result.error("File not found")
        return result

    file_size = os.path.getsize(filepath)
    result.set_info('file_size_bytes', file_size)

    if file_size == 0:
        result.error("File is empty")
        return result

    ext = Path(filepath).suffix.lower()

    try:
        if ext == '.glb':
            # Validate GLB binary header
            with open(filepath, 'rb') as f:
                magic = f.read(4)
                if magic != b'glTF':
                    result.error(f"Invalid GLB magic: {magic}")
                    return result

                version = int.from_bytes(f.read(4), 'little')
                length = int.from_bytes(f.read(4), 'little')

                result.set_info('gltf_version', version)
                result.set_info('glb_length', length)

                if version != 2:
                    result.warning(f"Unexpected glTF version: {version}")

                if length != file_size:
                    result.warning(f"GLB length mismatch: header={length}, file={file_size}")

        elif ext == '.gltf':
            # Validate JSON structure
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if 'asset' not in data:
                result.error("Missing required 'asset' property")
            else:
                result.set_info('gltf_version', data['asset'].get('version', 'unknown'))

            result.set_info('scenes', len(data.get('scenes', [])))
            result.set_info('nodes', len(data.get('nodes', [])))
            result.set_info('meshes', len(data.get('meshes', [])))
            result.set_info('materials', len(data.get('materials', [])))
            result.set_info('textures', len(data.get('textures', [])))

        # Use trimesh for geometry validation
        if HAS_TRIMESH:
            validate_geometry(filepath, result)

    except json.JSONDecodeError as e:
        result.error(f"Invalid JSON: {e}")
    except Exception as e:
        result.error(f"Validation error: {e}")

    return result


def validate_usd(filepath):
    """
    Validate USD file.

    Checks:
        - File exists and is readable
        - Basic USD structure (via pxr if available)
        - Geometry integrity (via trimesh)
    """
    result = ValidationResult(filepath)

    if not os.path.exists(filepath):
        result.error("File not found")
        return result

    file_size = os.path.getsize(filepath)
    result.set_info('file_size_bytes', file_size)

    if file_size == 0:
        result.error("File is empty")
        return result

    # Try OpenUSD validation
    try:
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(str(filepath))
        if not stage:
            result.error("Failed to open USD stage")
            return result

        # Count prims
        prim_count = 0
        mesh_count = 0
        for prim in stage.Traverse():
            prim_count += 1
            if prim.IsA(UsdGeom.Mesh):
                mesh_count += 1

        result.set_info('prims', prim_count)
        result.set_info('meshes', mesh_count)

        if mesh_count == 0:
            result.warning("No mesh geometry found")

    except ImportError:
        result.warning("OpenUSD (pxr) not available - limited validation")

        # Fallback: check file header
        ext = Path(filepath).suffix.lower()
        with open(filepath, 'rb') as f:
            header = f.read(8)

        if ext == '.usdc':
            # USDC binary format
            if not header.startswith(b'PXR-USDC'):
                result.warning("Unexpected USDC header")
        elif ext == '.usda':
            # USDA ASCII format
            if b'#usda' not in header:
                result.warning("Missing USDA header comment")

    except Exception as e:
        result.error(f"USD validation error: {e}")

    # Geometry validation via trimesh
    if HAS_TRIMESH:
        validate_geometry(filepath, result)

    return result


def validate_geometry(filepath, result):
    """
    Validate geometry using trimesh.

    Checks:
        - Geometry loads without errors
        - No NaN/Inf vertices
        - Reasonable bounding box
        - Face/vertex counts
    """
    if not HAS_TRIMESH:
        return

    try:
        scene = trimesh.load(str(filepath))

        # Get geometries
        if isinstance(scene, trimesh.Scene):
            geometries = list(scene.geometry.values())
        else:
            geometries = [scene]

        total_vertices = 0
        total_faces = 0
        has_nan = False
        has_inf = False

        for geom in geometries:
            if not hasattr(geom, 'vertices'):
                continue

            verts = geom.vertices
            total_vertices += len(verts)

            if hasattr(geom, 'faces'):
                total_faces += len(geom.faces)

            if HAS_NUMPY:
                verts = np.array(verts)
                if np.any(np.isnan(verts)):
                    has_nan = True
                if np.any(np.isinf(verts)):
                    has_inf = True

        result.set_info('total_vertices', total_vertices)
        result.set_info('total_faces', total_faces)
        result.set_info('geometry_count', len(geometries))

        if has_nan:
            result.error("Geometry contains NaN vertices")
        if has_inf:
            result.error("Geometry contains Inf vertices")

        if total_vertices == 0:
            result.error("No vertices found")
        if total_faces == 0:
            result.warning("No faces found")

    except Exception as e:
        result.warning(f"Geometry validation failed: {e}")


def validate_file(filepath):
    """Validate a file based on its extension."""
    ext = Path(filepath).suffix.lower()

    if ext in ['.gltf', '.glb']:
        return validate_gltf(filepath)
    elif ext in ['.usd', '.usda', '.usdc', '.usdz']:
        return validate_usd(filepath)
    else:
        result = ValidationResult(filepath)
        result.error(f"Unsupported format: {ext}")
        return result


def validate_directory(dir_path):
    """Validate all supported files in a directory."""
    results = []
    dir_path = Path(dir_path)

    # Find all export files
    patterns = ['*.glb', '*.gltf', '*.usd', '*.usda', '*.usdc']
    files = []
    for pattern in patterns:
        files.extend(dir_path.glob(f"**/{pattern}"))

    print(f"Found {len(files)} files to validate")

    for filepath in sorted(files):
        print(f"  Validating: {filepath.name}")
        result = validate_file(str(filepath))
        results.append(result)

    return results


def print_result(result):
    """Print validation result to console."""
    status = "PASS" if result.passed else "FAIL"
    status_color = "\033[32m" if result.passed else "\033[31m"
    reset = "\033[0m"

    print(f"\n{status_color}[{status}]{reset} {result.filename}")

    # Info
    if result.info:
        for key, value in result.info.items():
            print(f"  {key}: {value}")

    # Errors
    for err in result.errors:
        print(f"  \033[31mERROR: {err}\033[0m")

    # Warnings
    for warn in result.warnings[:3]:  # Limit warnings shown
        print(f"  \033[33mWARN: {warn}\033[0m")
    if len(result.warnings) > 3:
        print(f"  ... and {len(result.warnings) - 3} more warnings")


def main():
    parser = argparse.ArgumentParser(
        description="Validate glTF and USD export files"
    )
    parser.add_argument('--gltf', help="glTF file to validate")
    parser.add_argument('--usd', help="USD file to validate")
    parser.add_argument('--dir', help="Directory to validate")
    parser.add_argument('--output', '-o', help="Output JSON report path")

    args = parser.parse_args()

    results = []

    if args.gltf:
        print(f"Validating glTF: {args.gltf}")
        results.append(validate_file(args.gltf))

    if args.usd:
        print(f"Validating USD: {args.usd}")
        results.append(validate_file(args.usd))

    if args.dir:
        print(f"Validating directory: {args.dir}")
        results.extend(validate_directory(args.dir))

    if not results:
        print("No files to validate. Use --gltf, --usd, or --dir")
        sys.exit(0)

    # Print results
    for result in results:
        print_result(result)

    # Summary
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed

    print(f"\n{'='*50}")
    print(f"Summary: {passed} passed, {failed} failed")
    print(f"{'='*50}")

    # Save report if requested
    if args.output:
        report = {
            'results': [r.to_dict() for r in results],
            'summary': {
                'total': len(results),
                'passed': passed,
                'failed': failed
            }
        }
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved: {args.output}")

    # Exit with error if any failed
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
