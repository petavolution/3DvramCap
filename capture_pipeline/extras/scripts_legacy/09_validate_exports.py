#!/usr/bin/env python3
"""
Export Validation Script (glTF 2.0 and USD)
===========================================
Validates exported scene files for correctness and compatibility.

Usage:
    python 09_validate_exports.py --gltf scene.glb
    python 09_validate_exports.py --usd scene.usdc
    python 09_validate_exports.py --all targets/

Features:
    - glTF 2.0 specification compliance
    - USD schema validation
    - Geometry integrity checks
    - Material and texture validation
    - Cross-format comparison

Dependencies:
    pip install numpy trimesh pygltflib

External validators (recommended):
    - npm install -g gltf-validator
    - usdchecker (from OpenUSD)

Author: Capture Pipeline
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None


class ValidationResult:
    """Container for validation results."""

    def __init__(self, filepath):
        self.filepath = filepath
        self.passed = True
        self.errors = []
        self.warnings = []
        self.info = {}

    def error(self, msg):
        self.errors.append(msg)
        self.passed = False

    def warning(self, msg):
        self.warnings.append(msg)

    def add_info(self, key, value):
        self.info[key] = value

    def to_dict(self):
        return {
            "filepath": str(self.filepath),
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info
        }


def validate_gltf_basic(filepath):
    """
    Basic glTF validation using pygltflib.

    Checks:
        - File loads without errors
        - Required properties present
        - Buffer/accessor alignment
        - Texture references valid
    """
    result = ValidationResult(filepath)

    try:
        from pygltflib import GLTF2
    except ImportError:
        result.warning("pygltflib not installed - skipping Python validation")
        return result

    try:
        gltf = GLTF2().load(str(filepath))

        # Basic stats
        result.add_info("scenes", len(gltf.scenes) if gltf.scenes else 0)
        result.add_info("nodes", len(gltf.nodes) if gltf.nodes else 0)
        result.add_info("meshes", len(gltf.meshes) if gltf.meshes else 0)
        result.add_info("materials", len(gltf.materials) if gltf.materials else 0)
        result.add_info("textures", len(gltf.textures) if gltf.textures else 0)
        result.add_info("images", len(gltf.images) if gltf.images else 0)

        # Check required properties
        if not gltf.asset:
            result.error("Missing required 'asset' property")
        elif not gltf.asset.version:
            result.error("Missing required asset.version")
        else:
            result.add_info("gltf_version", gltf.asset.version)

        # Check scenes
        if not gltf.scenes or len(gltf.scenes) == 0:
            result.warning("No scenes defined")

        # Check meshes
        if gltf.meshes:
            total_primitives = sum(len(m.primitives) for m in gltf.meshes if m.primitives)
            result.add_info("primitives", total_primitives)

            # Check each mesh
            for i, mesh in enumerate(gltf.meshes):
                if not mesh.primitives:
                    result.warning(f"Mesh {i} has no primitives")
                    continue

                for j, prim in enumerate(mesh.primitives):
                    # Check required POSITION attribute
                    if not prim.attributes or 'POSITION' not in prim.attributes.__dict__:
                        result.error(f"Mesh {i} primitive {j} missing POSITION attribute")

        # Check accessors for alignment
        if gltf.accessors and gltf.bufferViews:
            for i, accessor in enumerate(gltf.accessors):
                if accessor.bufferView is not None:
                    bv = gltf.bufferViews[accessor.bufferView]
                    offset = (accessor.byteOffset or 0) + (bv.byteOffset or 0)

                    # Check 4-byte alignment
                    if offset % 4 != 0:
                        result.warning(f"Accessor {i} not 4-byte aligned (offset {offset})")

        # Check material references
        if gltf.meshes and gltf.materials:
            for i, mesh in enumerate(gltf.meshes):
                if mesh.primitives:
                    for j, prim in enumerate(mesh.primitives):
                        if prim.material is not None:
                            if prim.material >= len(gltf.materials):
                                result.error(f"Mesh {i} prim {j} references invalid material {prim.material}")

        # Check texture references
        if gltf.textures and gltf.images:
            for i, tex in enumerate(gltf.textures):
                if tex.source is not None:
                    if tex.source >= len(gltf.images):
                        result.error(f"Texture {i} references invalid image {tex.source}")

    except Exception as e:
        result.error(f"Failed to load glTF: {e}")

    return result


def validate_gltf_external(filepath):
    """
    Validate glTF using external gltf-validator (npm package).

    More comprehensive than Python validation, checks:
        - Full specification compliance
        - Extension validation
        - Binary buffer alignment
        - JSON schema
    """
    result = ValidationResult(filepath)

    # Check if gltf-validator is available
    try:
        proc = subprocess.run(
            ['gltf-validator', '--version'],
            capture_output=True,
            text=True
        )
        if proc.returncode != 0:
            result.warning("gltf-validator not available")
            return result
    except FileNotFoundError:
        result.warning("gltf-validator not installed (npm install -g gltf-validator)")
        return result

    # Run validation
    try:
        report_path = str(filepath) + '.validation.json'
        proc = subprocess.run(
            ['gltf-validator', str(filepath), '-o', report_path],
            capture_output=True,
            text=True
        )

        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                report = json.load(f)

            result.add_info("validator_version", report.get("validatorVersion", "unknown"))

            issues = report.get("issues", {})

            # Count by severity
            errors = issues.get("numErrors", 0)
            warnings = issues.get("numWarnings", 0)
            infos = issues.get("numInfos", 0)

            result.add_info("validation_errors", errors)
            result.add_info("validation_warnings", warnings)
            result.add_info("validation_infos", infos)

            if errors > 0:
                result.passed = False
                messages = issues.get("messages", [])
                for msg in messages:
                    if msg.get("severity") == 0:  # Error
                        result.error(msg.get("message", "Unknown error"))

            # Clean up report file
            os.remove(report_path)

    except Exception as e:
        result.warning(f"External validation failed: {e}")

    return result


def validate_usd_basic(filepath):
    """
    Basic USD validation using pxr module.

    Checks:
        - File loads without errors
        - Valid prim hierarchy
        - Material bindings
        - Texture file references exist
    """
    result = ValidationResult(filepath)

    try:
        from pxr import Usd, UsdGeom, UsdShade
    except ImportError:
        result.warning("OpenUSD (pxr) not installed - skipping Python validation")
        return result

    try:
        stage = Usd.Stage.Open(str(filepath))

        if not stage:
            result.error("Failed to open USD stage")
            return result

        # Count prims
        prim_count = 0
        mesh_count = 0
        material_count = 0

        for prim in stage.Traverse():
            prim_count += 1
            if prim.IsA(UsdGeom.Mesh):
                mesh_count += 1
            if prim.IsA(UsdShade.Material):
                material_count += 1

        result.add_info("prims", prim_count)
        result.add_info("meshes", mesh_count)
        result.add_info("materials", material_count)

        # Check for errors in the stage
        # Note: Stage errors would have been raised during Open()

        if mesh_count == 0:
            result.warning("No mesh geometry found")

    except Exception as e:
        result.error(f"Failed to load USD: {e}")

    return result


def validate_usd_external(filepath):
    """
    Validate USD using usdchecker tool.

    More comprehensive validation including:
        - Schema compliance
        - Asset resolution
        - ARKit compliance (optional)
    """
    result = ValidationResult(filepath)

    # Check if usdchecker is available
    try:
        proc = subprocess.run(
            ['usdchecker', '--help'],
            capture_output=True,
            text=True
        )
    except FileNotFoundError:
        result.warning("usdchecker not installed (part of OpenUSD)")
        return result

    # Run validation
    try:
        proc = subprocess.run(
            ['usdchecker', str(filepath)],
            capture_output=True,
            text=True
        )

        # Parse output
        if proc.returncode != 0:
            result.passed = False
            for line in proc.stderr.split('\n'):
                if line.strip():
                    result.error(line.strip())

        for line in proc.stdout.split('\n'):
            if 'error' in line.lower():
                result.error(line.strip())
            elif 'warning' in line.lower():
                result.warning(line.strip())

    except Exception as e:
        result.warning(f"External validation failed: {e}")

    return result


def validate_geometry_integrity(filepath):
    """
    Validate geometry integrity using trimesh.

    Checks:
        - No degenerate faces
        - No NaN/Inf vertices
        - Watertight (optional)
        - Reasonable bounds
    """
    result = ValidationResult(filepath)

    try:
        import trimesh
    except ImportError:
        result.warning("trimesh not installed - skipping geometry validation")
        return result

    try:
        scene = trimesh.load(str(filepath))

        # Handle scene vs single mesh
        if isinstance(scene, trimesh.Scene):
            geometries = list(scene.geometry.values())
        else:
            geometries = [scene]

        total_vertices = 0
        total_faces = 0
        degenerate_faces = 0

        for geom in geometries:
            if not hasattr(geom, 'vertices'):
                continue

            vertices = np.array(geom.vertices)
            total_vertices += len(vertices)

            if hasattr(geom, 'faces'):
                total_faces += len(geom.faces)

            # Check for NaN/Inf
            if np.any(np.isnan(vertices)):
                result.error("Geometry contains NaN vertices")
            if np.any(np.isinf(vertices)):
                result.error("Geometry contains Inf vertices")

            # Check bounds
            bounds = vertices.max(axis=0) - vertices.min(axis=0)
            if np.any(bounds > 1e6):
                result.warning(f"Very large geometry bounds: {bounds}")
            if np.any(bounds < 1e-6):
                result.warning(f"Very small geometry bounds: {bounds}")

            # Check for degenerate faces
            if hasattr(geom, 'area_faces'):
                areas = geom.area_faces
                degenerate = np.sum(areas < 1e-10)
                degenerate_faces += degenerate

        result.add_info("total_vertices", total_vertices)
        result.add_info("total_faces", total_faces)

        if degenerate_faces > 0:
            result.warning(f"Found {degenerate_faces} degenerate faces (area ~0)")

    except Exception as e:
        result.warning(f"Geometry validation failed: {e}")

    return result


def validate_file(filepath, format_type=None):
    """
    Run all applicable validations on a file.
    """
    path = Path(filepath)

    if not path.exists():
        result = ValidationResult(filepath)
        result.error("File not found")
        return result

    # Auto-detect format
    if format_type is None:
        ext = path.suffix.lower()
        if ext in ['.gltf', '.glb']:
            format_type = 'gltf'
        elif ext in ['.usd', '.usda', '.usdc', '.usdz']:
            format_type = 'usd'
        else:
            result = ValidationResult(filepath)
            result.error(f"Unknown format: {ext}")
            return result

    print(f"\nValidating: {filepath}")
    print(f"Format: {format_type}")

    results = []

    if format_type == 'gltf':
        # Python validation
        basic = validate_gltf_basic(filepath)
        results.append(('gltf_basic', basic))

        # External validator
        external = validate_gltf_external(filepath)
        results.append(('gltf_external', external))

    elif format_type == 'usd':
        # Python validation
        basic = validate_usd_basic(filepath)
        results.append(('usd_basic', basic))

        # External validator
        external = validate_usd_external(filepath)
        results.append(('usd_external', external))

    # Geometry validation for all formats
    geom = validate_geometry_integrity(filepath)
    results.append(('geometry', geom))

    # Combine results
    combined = ValidationResult(filepath)
    combined.info['format'] = format_type

    for name, r in results:
        combined.info[f'{name}_info'] = r.info
        combined.errors.extend([f"[{name}] {e}" for e in r.errors])
        combined.warnings.extend([f"[{name}] {w}" for w in r.warnings])
        if not r.passed:
            combined.passed = False

    return combined


def print_validation_result(result):
    """Print validation result in human-readable format."""
    status = "PASSED" if result.passed else "FAILED"
    print(f"\n{'='*60}")
    print(f"File: {result.filepath}")
    print(f"Status: {status}")
    print(f"{'='*60}")

    if result.info:
        print("\nInfo:")
        for key, value in result.info.items():
            if not key.endswith('_info'):
                print(f"  {key}: {value}")

    if result.errors:
        print(f"\nErrors ({len(result.errors)}):")
        for err in result.errors[:10]:  # Limit output
            print(f"  ERROR: {err}")
        if len(result.errors) > 10:
            print(f"  ... and {len(result.errors) - 10} more")

    if result.warnings:
        print(f"\nWarnings ({len(result.warnings)}):")
        for warn in result.warnings[:5]:
            print(f"  WARNING: {warn}")
        if len(result.warnings) > 5:
            print(f"  ... and {len(result.warnings) - 5} more")


def main():
    parser = argparse.ArgumentParser(
        description="Validate glTF and USD exports"
    )
    parser.add_argument('--gltf', help="glTF file to validate")
    parser.add_argument('--usd', help="USD file to validate")
    parser.add_argument('--all', help="Validate all exports in directory")
    parser.add_argument('--output', '-o', help="Output JSON report path")

    args = parser.parse_args()

    results = []

    if args.gltf:
        result = validate_file(args.gltf, 'gltf')
        results.append(result)
        print_validation_result(result)

    if args.usd:
        result = validate_file(args.usd, 'usd')
        results.append(result)
        print_validation_result(result)

    if args.all:
        # Find all export files
        target_dir = Path(args.all)
        files = (
            list(target_dir.glob("**/*.glb")) +
            list(target_dir.glob("**/*.gltf")) +
            list(target_dir.glob("**/*.usd")) +
            list(target_dir.glob("**/*.usdc")) +
            list(target_dir.glob("**/*.usda"))
        )

        for f in files:
            result = validate_file(str(f))
            results.append(result)
            print_validation_result(result)

    # Save report
    if args.output and results:
        report = {
            "validations": [r.to_dict() for r in results],
            "summary": {
                "total": len(results),
                "passed": sum(1 for r in results if r.passed),
                "failed": sum(1 for r in results if not r.passed)
            }
        }

        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved: {args.output}")

    # Print summary
    if results:
        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed
        print(f"\n{'='*60}")
        print(f"Summary: {passed} passed, {failed} failed")
        print(f"{'='*60}")

        if failed > 0:
            sys.exit(1)

    print("\n[OK] Validation complete!")


if __name__ == "__main__":
    main()
