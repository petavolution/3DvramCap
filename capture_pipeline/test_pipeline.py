#!/usr/bin/env python3
"""
Pipeline Test Framework
=======================
CLI-based testing for the 3DvramCap pipeline.

Usage:
    python test_pipeline.py                    # Run all tests
    python test_pipeline.py --quick            # Run quick tests only
    python test_pipeline.py --module process   # Test specific module
    python test_pipeline.py --list             # List available tests

Tests write results to debug-log.txt and console.

Author: 3DvramCap
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.utils import logger, log_section, timed, start_session, end_session


# =============================================================================
# Test Infrastructure
# =============================================================================

@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    duration: float = 0.0
    message: str = ""
    error: str = ""


@dataclass
class TestSuite:
    """Collection of test results."""
    name: str
    results: List[TestResult] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def add(self, result: TestResult):
        self.results.append(result)

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'passed': self.passed,
            'failed': self.failed,
            'total': self.total,
            'duration': self.duration,
            'results': [
                {
                    'name': r.name,
                    'passed': r.passed,
                    'duration': r.duration,
                    'message': r.message,
                    'error': r.error
                }
                for r in self.results
            ]
        }


class TestRunner:
    """Runs tests and collects results."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.suites: List[TestSuite] = []
        self.temp_dirs: List[Path] = []

    def create_temp_dir(self) -> Path:
        """Create a temporary directory for test artifacts."""
        temp_dir = Path(tempfile.mkdtemp(prefix='3dvram_test_'))
        self.temp_dirs.append(temp_dir)
        return temp_dir

    def cleanup(self):
        """Clean up temporary directories."""
        for temp_dir in self.temp_dirs:
            try:
                shutil.rmtree(temp_dir)
            except (OSError, IOError):
                pass

    def run_test(self, name: str, test_fn: Callable[[], Tuple[bool, str]]) -> TestResult:
        """Run a single test function."""
        start = time.perf_counter()

        try:
            passed, message = test_fn()
            duration = time.perf_counter() - start

            result = TestResult(
                name=name,
                passed=passed,
                duration=duration,
                message=message
            )

        except Exception as e:
            duration = time.perf_counter() - start
            result = TestResult(
                name=name,
                passed=False,
                duration=duration,
                error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            )

        # Log result
        status = "\033[32mPASS\033[0m" if result.passed else "\033[31mFAIL\033[0m"
        logger.info(f"  [{status}] {name} ({result.duration:.3f}s)")

        if result.error and self.verbose:
            logger.error(f"    {result.error}")
        elif result.message and self.verbose:
            logger.debug(f"    {result.message}")

        return result

    def run_suite(self, name: str,
                  tests: List[Tuple[str, Callable]]) -> TestSuite:
        """Run a suite of tests."""
        suite = TestSuite(name=name)
        suite.start_time = time.perf_counter()

        log_section(f"Test Suite: {name}")

        for test_name, test_fn in tests:
            result = self.run_test(test_name, test_fn)
            suite.add(result)

        suite.end_time = time.perf_counter()

        # Summary
        status = "\033[32mPASSED\033[0m" if suite.failed == 0 else "\033[31mFAILED\033[0m"
        logger.info(f"\nSuite {name}: {status} ({suite.passed}/{suite.total})")

        self.suites.append(suite)
        return suite

    def summary(self) -> Tuple[int, int]:
        """Print overall summary and return (passed, failed)."""
        total_passed = sum(s.passed for s in self.suites)
        total_failed = sum(s.failed for s in self.suites)
        total_duration = sum(s.duration for s in self.suites)

        log_section("TEST SUMMARY")
        for suite in self.suites:
            status = "PASS" if suite.failed == 0 else "FAIL"
            logger.info(f"  {suite.name}: {status} ({suite.passed}/{suite.total})")

        logger.info(f"\nTotal: {total_passed} passed, {total_failed} failed")
        logger.info(f"Duration: {total_duration:.2f}s")

        return total_passed, total_failed

    def save_report(self, path: Path):
        """Save test report to JSON."""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'suites': [s.to_dict() for s in self.suites],
            'summary': {
                'passed': sum(s.passed for s in self.suites),
                'failed': sum(s.failed for s in self.suites),
                'duration': sum(s.duration for s in self.suites)
            }
        }

        with open(path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Report saved: {path}")


# =============================================================================
# Test Fixtures
# =============================================================================

def create_test_obj(path: Path, num_vertices: int = 100) -> Path:
    """Create a simple test OBJ file."""
    import random

    with open(path, 'w') as f:
        f.write("# Test OBJ file\n")

        # Random vertices
        for i in range(num_vertices):
            x = random.uniform(-10, 10)
            y = random.uniform(-10, 10)
            z = random.uniform(-10, 10)
            f.write(f"v {x:.4f} {y:.4f} {z:.4f}\n")

        # Simple faces (triangles)
        for i in range(0, num_vertices - 2, 3):
            f.write(f"f {i+1} {i+2} {i+3}\n")

    return path


def create_test_scene(temp_dir: Path) -> Path:
    """Create a complete test scene structure."""
    scene_dir = temp_dir / 'test_scene'
    meshes_dir = scene_dir / 'Meshes'
    textures_dir = scene_dir / 'Textures'

    meshes_dir.mkdir(parents=True)
    textures_dir.mkdir(parents=True)

    # Create test meshes
    for i in range(3):
        create_test_obj(meshes_dir / f'mesh_{i:03d}.obj', num_vertices=100 + i * 50)

    # Create scene.json manifest
    manifest = {
        'capture_name': 'test_scene',
        'meshes': [f'mesh_{i:03d}.obj' for i in range(3)],
        'textures': []
    }
    with open(scene_dir / 'scene.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    return scene_dir


# =============================================================================
# Module Import Tests
# =============================================================================

def test_import_utils() -> Tuple[bool, str]:
    """Test core.utils import."""
    from core.utils import logger, log_error, timed, safe_mkdir
    return True, "All utilities imported successfully"


def test_import_types() -> Tuple[bool, str]:
    """Test core.core_types import."""
    from core.core_types import Mesh, Material, ExtractionResult, PipelineConfig
    return True, "Type definitions imported successfully"


def test_import_camera() -> Tuple[bool, str]:
    """Test core.core_camera import."""
    from core.core_camera import (
        ue4_to_blender_position,
        mat4_multiply,
        vec3_normalize
    )
    # Test basic function
    pos = ue4_to_blender_position([100, 200, 300])
    if len(pos) != 3:
        return False, f"Expected 3-element list, got {len(pos)}"
    return True, f"Camera utilities working: {pos}"


def test_import_process() -> Tuple[bool, str]:
    """Test core.core_process import."""
    from core.core_process import compute_geometry_hash, deduplicate_meshes, parse_obj_file
    return True, "Process module imported successfully"


def test_import_gltf() -> Tuple[bool, str]:
    """Test core.core_gltf import."""
    from core.core_gltf import GltfBuilder, MeshEntry, MaterialEntry
    builder = GltfBuilder()
    return True, f"GltfBuilder created successfully"


def test_import_validate() -> Tuple[bool, str]:
    """Test core.core_validate import."""
    from core.core_validate import ValidationResult, validate_file
    return True, "Validation module imported successfully"


# =============================================================================
# Functionality Tests
# =============================================================================

def test_camera_transforms() -> Tuple[bool, str]:
    """Test coordinate system transformations."""
    from core.core_camera import (
        ue4_to_blender_position,
        ue4_to_gltf_position,
        mat4_identity,
        mat4_multiply,
        mat4_inverse
    )

    # Test UE4 to Blender conversion
    ue4_pos = [100, 200, 300]  # 1m, 2m, 3m in UE4 cm
    blender_pos = ue4_to_blender_position(ue4_pos)

    # Should be scaled by 0.01 (cm -> m) and Y flipped
    expected = [1.0, -2.0, 3.0]
    for i, (got, exp) in enumerate(zip(blender_pos, expected)):
        if abs(got - exp) > 0.001:
            return False, f"Position mismatch at index {i}: {got} != {exp}"

    # Test matrix operations
    identity = mat4_identity()
    result = mat4_multiply(identity, identity)
    for i in range(4):
        if result[i][i] != 1:
            return False, f"Identity multiply failed at [{i}][{i}]"

    # Test matrix inverse
    inv = mat4_inverse(identity)
    if inv is None:
        return False, "Failed to invert identity matrix"

    return True, "All coordinate transforms working correctly"


def test_mesh_processor(runner: TestRunner) -> Tuple[bool, str]:
    """Test mesh processing with deduplication."""
    from core.core_process import deduplicate_meshes

    temp_dir = runner.create_temp_dir()
    input_dir = temp_dir / 'input'
    output_dir = temp_dir / 'output'
    input_dir.mkdir()

    # Create test meshes (including a duplicate)
    create_test_obj(input_dir / 'mesh_001.obj', 100)
    shutil.copy(input_dir / 'mesh_001.obj', input_dir / 'mesh_002.obj')  # Duplicate
    create_test_obj(input_dir / 'mesh_003.obj', 150)

    # Process using deduplicate_meshes function
    result = deduplicate_meshes(str(input_dir), str(output_dir), min_vertices=10)

    if result is None:
        return False, "Processing returned None"

    if 'error' in result:
        return False, f"Processing error: {result['error']}"

    # Should have deduplicated one mesh
    output_files = list(output_dir.glob('*.obj'))
    if len(output_files) < 1:
        return False, f"No output files created"

    return True, f"Processed {len(output_files)} unique meshes"


def test_gltf_builder(runner: TestRunner) -> Tuple[bool, str]:
    """Test glTF building functionality."""
    import numpy as np
    from core.core_gltf import GltfBuilder

    temp_dir = runner.create_temp_dir()
    output_path = temp_dir / 'test.glb'

    # Create simple mesh data
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0.5, 1, 0]
    ], dtype=np.float32)

    faces = np.array([[0, 1, 2]], dtype=np.uint32)

    # Build glTF
    builder = GltfBuilder()
    mat_idx = builder.add_unlit_material("test_material", base_color=[1, 0, 0, 1])
    mesh_idx = builder.add_mesh("test_mesh", vertices, faces, material_index=mat_idx)

    builder.save(str(output_path))

    if not output_path.exists():
        return False, "GLB file was not created"

    file_size = output_path.stat().st_size
    if file_size < 100:
        return False, f"GLB file too small: {file_size} bytes"

    return True, f"Created GLB file: {file_size} bytes"


def test_validation(runner: TestRunner) -> Tuple[bool, str]:
    """Test file validation."""
    import numpy as np
    from core.core_gltf import GltfBuilder
    from core.core_validate import validate_gltf

    temp_dir = runner.create_temp_dir()
    glb_path = temp_dir / 'valid.glb'

    # Create a valid GLB
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.uint32)

    builder = GltfBuilder()
    builder.add_mesh("mesh", vertices, faces)
    builder.save(str(glb_path))

    # Validate
    result = validate_gltf(str(glb_path))

    if not result.passed:
        return False, f"Validation failed: {result.errors}"

    return True, f"Validation passed: {result.info}"


def test_logging() -> Tuple[bool, str]:
    """Test logging to debug-log.txt."""
    from core.utils import logger, LOG_FILE, log_error, log_section

    # Write test messages
    test_msg = f"Test message at {time.time()}"
    logger.info(test_msg)
    log_error("Test error message")
    log_section("Test Section")

    # Verify log file exists and contains message
    if not LOG_FILE.exists():
        return False, f"Log file not created: {LOG_FILE}"

    with open(LOG_FILE, 'r') as f:
        content = f.read()

    if test_msg not in content:
        return False, "Test message not found in log file"

    return True, f"Logging working: {LOG_FILE}"


# =============================================================================
# Pipeline Integration Tests
# =============================================================================

def test_pipeline_help() -> Tuple[bool, str]:
    """Test pipeline.py --help."""
    result = subprocess.run(
        [sys.executable, 'pipeline.py', '--help'],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT
    )

    if result.returncode != 0:
        return False, f"Exit code {result.returncode}: {result.stderr}"

    if 'run' not in result.stdout or 'validate' not in result.stdout:
        return False, "Missing expected commands in help output"

    return True, "pipeline.py --help works correctly"


def test_pipeline_extract_cmd() -> Tuple[bool, str]:
    """Test pipeline.py extract-cmd."""
    result = subprocess.run(
        [sys.executable, 'pipeline.py', 'extract-cmd', 'test.rdc'],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT
    )

    if result.returncode != 0:
        return False, f"Exit code {result.returncode}: {result.stderr}"

    if 'renderdoccmd' not in result.stdout:
        return False, "Missing renderdoccmd in output"

    return True, "extract-cmd works correctly"


def test_pipeline_validate_missing() -> Tuple[bool, str]:
    """Test pipeline.py validate with missing file."""
    result = subprocess.run(
        [sys.executable, 'pipeline.py', 'validate', '/nonexistent/path'],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT
    )

    # Should fail gracefully with non-zero exit
    if result.returncode == 0:
        return False, "Should have failed with missing file"

    return True, "Handles missing files correctly"


# =============================================================================
# Test Suite Definitions
# =============================================================================

def get_import_tests() -> List[Tuple[str, Callable]]:
    """Get module import tests."""
    return [
        ("import_utils", test_import_utils),
        ("import_types", test_import_types),
        ("import_camera", test_import_camera),
        ("import_process", test_import_process),
        ("import_gltf", test_import_gltf),
        ("import_validate", test_import_validate),
    ]


def get_unit_tests(runner: TestRunner) -> List[Tuple[str, Callable]]:
    """Get unit tests."""
    return [
        ("camera_transforms", test_camera_transforms),
        ("mesh_processor", lambda: test_mesh_processor(runner)),
        ("gltf_builder", lambda: test_gltf_builder(runner)),
        ("validation", lambda: test_validation(runner)),
        ("logging", test_logging),
    ]


def get_integration_tests() -> List[Tuple[str, Callable]]:
    """Get integration tests."""
    return [
        ("pipeline_help", test_pipeline_help),
        ("pipeline_extract_cmd", test_pipeline_extract_cmd),
        ("pipeline_validate_missing", test_pipeline_validate_missing),
    ]


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="3DvramCap Pipeline Test Framework")
    parser.add_argument('--quick', action='store_true', help="Run quick tests only")
    parser.add_argument('--module', help="Test specific module (imports, unit, integration)")
    parser.add_argument('--list', action='store_true', help="List available tests")
    parser.add_argument('--verbose', '-v', action='store_true', help="Verbose output")
    parser.add_argument('--report', help="Save JSON report to file")

    args = parser.parse_args()

    if args.list:
        print("Available test modules:")
        print("  imports     - Module import tests")
        print("  unit        - Unit tests")
        print("  integration - Pipeline integration tests")
        print("\nRun with: python test_pipeline.py --module <name>")
        return 0

    # Start session
    session_id = start_session("test_pipeline")

    runner = TestRunner(verbose=args.verbose)

    try:
        # Run selected tests
        if args.module == 'imports' or args.module is None:
            runner.run_suite("Module Imports", get_import_tests())

        if args.module == 'unit' or (args.module is None and not args.quick):
            runner.run_suite("Unit Tests", get_unit_tests(runner))

        if args.module == 'integration' or (args.module is None and not args.quick):
            runner.run_suite("Integration Tests", get_integration_tests())

        # Summary
        passed, failed = runner.summary()

        # Save report if requested
        if args.report:
            runner.save_report(Path(args.report))

        end_session(session_id, success=(failed == 0))

        return 0 if failed == 0 else 1

    finally:
        runner.cleanup()


if __name__ == '__main__':
    sys.exit(main())
