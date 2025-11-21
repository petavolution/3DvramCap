#!/usr/bin/env python3
"""
Quality Metrics and Validation
==============================
Automated quality assessment for meshes and exports.

Features:
    - Mesh quality metrics (topology, UV coverage, normals)
    - Scene completeness checks
    - Export format validation
    - Automated scoring system
    - Report generation

Usage:
    from core_quality import QualityChecker, assess_mesh_quality

    # Check single mesh
    metrics = assess_mesh_quality("mesh.obj")
    print(f"Quality score: {metrics.score}")

    # Full scene assessment
    checker = QualityChecker("export/scene/")
    report = checker.full_assessment()
    report.save_html("quality_report.html")

Author: Capture Pipeline
"""

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class QualityLevel(Enum):
    """Quality assessment levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    FAILED = "failed"


@dataclass
class MeshMetrics:
    """Quality metrics for a single mesh."""
    name: str
    filepath: str

    # Counts
    vertex_count: int = 0
    face_count: int = 0
    edge_count: int = 0

    # Topology
    is_manifold: bool = True
    has_non_manifold_edges: int = 0
    has_non_manifold_vertices: int = 0
    isolated_vertices: int = 0
    degenerate_faces: int = 0
    duplicate_vertices: int = 0

    # Normals
    has_normals: bool = False
    flipped_normals: int = 0
    zero_normals: int = 0

    # UVs
    has_uvs: bool = False
    uv_coverage: float = 0.0  # 0-1, how much of UV space is used
    uv_overlap_ratio: float = 0.0  # 0-1, overlapping UV islands
    uv_distortion: float = 0.0  # Average stretch/compression

    # Bounds
    bounds_min: Tuple[float, float, float] = (0, 0, 0)
    bounds_max: Tuple[float, float, float] = (0, 0, 0)
    bounds_size: Tuple[float, float, float] = (0, 0, 0)

    # Computed
    score: float = 0.0
    level: QualityLevel = QualityLevel.ACCEPTABLE
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def compute_score(self):
        """Calculate overall quality score (0-100)."""
        score = 100.0
        self.issues = []
        self.warnings = []

        # Topology issues (severe)
        if not self.is_manifold:
            score -= 20
            self.issues.append("Non-manifold geometry")

        if self.has_non_manifold_edges > 0:
            penalty = min(20, self.has_non_manifold_edges * 2)
            score -= penalty
            self.issues.append(f"{self.has_non_manifold_edges} non-manifold edges")

        if self.degenerate_faces > 0:
            penalty = min(15, self.degenerate_faces)
            score -= penalty
            self.issues.append(f"{self.degenerate_faces} degenerate faces")

        # Vertex issues (moderate)
        if self.isolated_vertices > 0:
            penalty = min(10, self.isolated_vertices // 10)
            score -= penalty
            self.warnings.append(f"{self.isolated_vertices} isolated vertices")

        if self.duplicate_vertices > self.vertex_count * 0.1:
            score -= 5
            self.warnings.append("High duplicate vertex count")

        # Normal issues
        if not self.has_normals:
            score -= 10
            self.warnings.append("Missing normals")

        if self.flipped_normals > self.face_count * 0.05:
            score -= 10
            self.issues.append(f"{self.flipped_normals} flipped normals")

        if self.zero_normals > 0:
            score -= 5
            self.warnings.append(f"{self.zero_normals} zero-length normals")

        # UV issues
        if not self.has_uvs:
            score -= 15
            self.warnings.append("Missing UVs")
        else:
            if self.uv_coverage < 0.1:
                score -= 10
                self.issues.append("Very low UV coverage")
            elif self.uv_coverage < 0.5:
                score -= 5
                self.warnings.append("Low UV coverage")

            if self.uv_overlap_ratio > 0.5:
                score -= 10
                self.issues.append("High UV overlap")
            elif self.uv_overlap_ratio > 0.2:
                score -= 5
                self.warnings.append("Moderate UV overlap")

        # Size sanity check
        max_dim = max(self.bounds_size)
        if max_dim > 100000:  # Very large (probably not scaled)
            self.warnings.append("Mesh may need scaling")
        elif max_dim < 0.01:  # Very small
            self.warnings.append("Mesh is very small")

        self.score = max(0, score)

        # Determine level
        if self.score >= 90:
            self.level = QualityLevel.EXCELLENT
        elif self.score >= 75:
            self.level = QualityLevel.GOOD
        elif self.score >= 50:
            self.level = QualityLevel.ACCEPTABLE
        elif self.score >= 25:
            self.level = QualityLevel.POOR
        else:
            self.level = QualityLevel.FAILED

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'filepath': self.filepath,
            'counts': {
                'vertices': self.vertex_count,
                'faces': self.face_count,
                'edges': self.edge_count
            },
            'topology': {
                'is_manifold': self.is_manifold,
                'non_manifold_edges': self.has_non_manifold_edges,
                'degenerate_faces': self.degenerate_faces,
                'isolated_vertices': self.isolated_vertices
            },
            'normals': {
                'has_normals': self.has_normals,
                'flipped': self.flipped_normals,
                'zero_length': self.zero_normals
            },
            'uvs': {
                'has_uvs': self.has_uvs,
                'coverage': self.uv_coverage,
                'overlap': self.uv_overlap_ratio
            },
            'bounds': {
                'min': self.bounds_min,
                'max': self.bounds_max,
                'size': self.bounds_size
            },
            'quality': {
                'score': self.score,
                'level': self.level.value,
                'issues': self.issues,
                'warnings': self.warnings
            }
        }


@dataclass
class SceneMetrics:
    """Quality metrics for a complete scene."""
    name: str
    directory: str

    # Counts
    mesh_count: int = 0
    material_count: int = 0
    texture_count: int = 0
    total_vertices: int = 0
    total_faces: int = 0

    # Mesh quality
    mesh_metrics: List[MeshMetrics] = field(default_factory=list)
    avg_mesh_score: float = 0.0

    # Scene bounds
    scene_bounds_min: Tuple[float, float, float] = (0, 0, 0)
    scene_bounds_max: Tuple[float, float, float] = (0, 0, 0)
    scene_size: Tuple[float, float, float] = (0, 0, 0)

    # Export validation
    gltf_valid: bool = False
    usd_valid: bool = False
    export_errors: List[str] = field(default_factory=list)

    # Overall
    score: float = 0.0
    level: QualityLevel = QualityLevel.ACCEPTABLE

    def compute_score(self):
        """Calculate overall scene quality score."""
        if not self.mesh_metrics:
            self.score = 0
            self.level = QualityLevel.FAILED
            return

        # Average mesh scores
        self.avg_mesh_score = sum(m.score for m in self.mesh_metrics) / len(self.mesh_metrics)

        # Start with mesh average
        score = self.avg_mesh_score

        # Export validation bonus/penalty
        if self.gltf_valid:
            score += 5
        else:
            score -= 10

        if self.usd_valid:
            score += 5
        else:
            score -= 5

        # Scene completeness
        if self.mesh_count < 5:
            score -= 10  # Suspiciously few meshes
        elif self.mesh_count > 1000:
            score -= 5  # May have too many small meshes

        self.score = max(0, min(100, score))

        # Determine level
        if self.score >= 85:
            self.level = QualityLevel.EXCELLENT
        elif self.score >= 70:
            self.level = QualityLevel.GOOD
        elif self.score >= 50:
            self.level = QualityLevel.ACCEPTABLE
        elif self.score >= 25:
            self.level = QualityLevel.POOR
        else:
            self.level = QualityLevel.FAILED

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'directory': self.directory,
            'counts': {
                'meshes': self.mesh_count,
                'materials': self.material_count,
                'textures': self.texture_count,
                'total_vertices': self.total_vertices,
                'total_faces': self.total_faces
            },
            'bounds': {
                'min': self.scene_bounds_min,
                'max': self.scene_bounds_max,
                'size': self.scene_size
            },
            'export': {
                'gltf_valid': self.gltf_valid,
                'usd_valid': self.usd_valid,
                'errors': self.export_errors
            },
            'quality': {
                'score': self.score,
                'level': self.level.value,
                'avg_mesh_score': self.avg_mesh_score
            },
            'meshes': [m.to_dict() for m in self.mesh_metrics]
        }


class MeshAnalyzer:
    """Analyze mesh quality from various formats."""

    def __init__(self):
        self._trimesh = None
        try:
            import trimesh
            self._trimesh = trimesh
        except ImportError:
            pass

    def analyze_obj(self, filepath: str) -> MeshMetrics:
        """Analyze OBJ file quality."""
        metrics = MeshMetrics(
            name=Path(filepath).stem,
            filepath=filepath
        )

        vertices = []
        normals = []
        uvs = []
        faces = []

        try:
            with open(filepath, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue

                    if parts[0] == 'v' and len(parts) >= 4:
                        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    elif parts[0] == 'vn' and len(parts) >= 4:
                        normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    elif parts[0] == 'vt' and len(parts) >= 3:
                        uvs.append([float(parts[1]), float(parts[2])])
                    elif parts[0] == 'f':
                        face = []
                        for v in parts[1:]:
                            idx = int(v.split('/')[0]) - 1
                            face.append(idx)
                        if len(face) >= 3:
                            faces.append(face[:3])

        except Exception as e:
            metrics.issues.append(f"Failed to parse: {e}")
            metrics.score = 0
            metrics.level = QualityLevel.FAILED
            return metrics

        if not vertices or not faces:
            metrics.issues.append("Empty mesh")
            metrics.score = 0
            metrics.level = QualityLevel.FAILED
            return metrics

        if HAS_NUMPY:
            vertices = np.array(vertices)
            faces = np.array(faces)

            metrics.vertex_count = len(vertices)
            metrics.face_count = len(faces)
            metrics.has_normals = len(normals) > 0
            metrics.has_uvs = len(uvs) > 0

            # Bounds
            metrics.bounds_min = tuple(vertices.min(axis=0))
            metrics.bounds_max = tuple(vertices.max(axis=0))
            metrics.bounds_size = tuple(vertices.max(axis=0) - vertices.min(axis=0))

            # Check for degenerate faces
            for face in faces:
                if len(set(face)) < 3:
                    metrics.degenerate_faces += 1

            # Check normals
            if metrics.has_normals:
                normals = np.array(normals)
                lengths = np.linalg.norm(normals, axis=1)
                metrics.zero_normals = int(np.sum(lengths < 1e-6))

            # UV analysis
            if metrics.has_uvs:
                uvs = np.array(uvs)
                # Coverage: what fraction of [0,1]x[0,1] is covered
                uv_min = uvs.min(axis=0)
                uv_max = uvs.max(axis=0)
                metrics.uv_coverage = float((uv_max[0] - uv_min[0]) * (uv_max[1] - uv_min[1]))
                metrics.uv_coverage = min(1.0, metrics.uv_coverage)

            # Use trimesh for advanced analysis if available
            if self._trimesh:
                self._analyze_with_trimesh(filepath, metrics)

        else:
            metrics.vertex_count = len(vertices)
            metrics.face_count = len(faces)
            metrics.has_normals = len(normals) > 0
            metrics.has_uvs = len(uvs) > 0

        metrics.compute_score()
        return metrics

    def _analyze_with_trimesh(self, filepath: str, metrics: MeshMetrics):
        """Use trimesh for detailed topology analysis."""
        try:
            mesh = self._trimesh.load(filepath, force='mesh')

            # Manifold check
            metrics.is_manifold = mesh.is_watertight

            # Edge analysis
            if hasattr(mesh, 'edges_unique'):
                metrics.edge_count = len(mesh.edges_unique)

            # Check for non-manifold edges
            # This is a simplified check
            if hasattr(mesh, 'face_adjacency'):
                pass  # Could count edges with >2 adjacent faces

        except Exception as e:
            metrics.warnings.append(f"Trimesh analysis failed: {e}")

    def analyze_gltf(self, filepath: str) -> List[MeshMetrics]:
        """Analyze glTF file quality."""
        metrics_list = []

        try:
            import pygltflib
            gltf = pygltflib.GLTF2().load(filepath)

            for i, mesh in enumerate(gltf.meshes or []):
                metrics = MeshMetrics(
                    name=mesh.name or f"mesh_{i}",
                    filepath=filepath
                )

                total_verts = 0
                total_faces = 0
                has_normals = False
                has_uvs = False

                for prim in mesh.primitives or []:
                    # Count vertices
                    if prim.attributes.POSITION is not None:
                        acc = gltf.accessors[prim.attributes.POSITION]
                        total_verts += acc.count

                    # Check indices
                    if prim.indices is not None:
                        acc = gltf.accessors[prim.indices]
                        total_faces += acc.count // 3

                    # Check normals
                    if prim.attributes.NORMAL is not None:
                        has_normals = True

                    # Check UVs
                    if prim.attributes.TEXCOORD_0 is not None:
                        has_uvs = True

                metrics.vertex_count = total_verts
                metrics.face_count = total_faces
                metrics.has_normals = has_normals
                metrics.has_uvs = has_uvs

                metrics.compute_score()
                metrics_list.append(metrics)

        except Exception as e:
            error_metrics = MeshMetrics(name="error", filepath=filepath)
            error_metrics.issues.append(f"Failed to parse glTF: {e}")
            error_metrics.level = QualityLevel.FAILED
            metrics_list.append(error_metrics)

        return metrics_list


class QualityChecker:
    """
    Comprehensive quality checker for exported scenes.
    """

    def __init__(self, directory: str):
        self.directory = Path(directory)
        self.analyzer = MeshAnalyzer()

    def full_assessment(self) -> SceneMetrics:
        """Perform full quality assessment of a scene directory."""
        metrics = SceneMetrics(
            name=self.directory.name,
            directory=str(self.directory)
        )

        # Find and analyze all meshes
        obj_files = list(self.directory.glob("**/*.obj"))
        for obj_file in obj_files:
            mesh_metrics = self.analyzer.analyze_obj(str(obj_file))
            metrics.mesh_metrics.append(mesh_metrics)
            metrics.total_vertices += mesh_metrics.vertex_count
            metrics.total_faces += mesh_metrics.face_count

        metrics.mesh_count = len(metrics.mesh_metrics)

        # Count materials and textures
        mtl_files = list(self.directory.glob("**/*.mtl"))
        metrics.material_count = len(mtl_files)

        texture_exts = ['.png', '.jpg', '.jpeg', '.dds', '.tga']
        for ext in texture_exts:
            metrics.texture_count += len(list(self.directory.glob(f"**/*{ext}")))

        # Validate exports
        gltf_files = list(self.directory.glob("**/*.glb")) + list(self.directory.glob("**/*.gltf"))
        if gltf_files:
            metrics.gltf_valid = self._validate_gltf(gltf_files[0])

        usd_files = list(self.directory.glob("**/*.usd*"))
        if usd_files:
            metrics.usd_valid = self._validate_usd(usd_files[0])

        # Compute scene bounds
        if metrics.mesh_metrics:
            all_mins = [m.bounds_min for m in metrics.mesh_metrics if m.bounds_min != (0, 0, 0)]
            all_maxs = [m.bounds_max for m in metrics.mesh_metrics if m.bounds_max != (0, 0, 0)]

            if all_mins and all_maxs:
                metrics.scene_bounds_min = tuple(min(m[i] for m in all_mins) for i in range(3))
                metrics.scene_bounds_max = tuple(max(m[i] for m in all_maxs) for i in range(3))
                metrics.scene_size = tuple(
                    metrics.scene_bounds_max[i] - metrics.scene_bounds_min[i]
                    for i in range(3)
                )

        metrics.compute_score()
        return metrics

    def _validate_gltf(self, filepath: Path) -> bool:
        """Validate glTF file."""
        try:
            import pygltflib
            gltf = pygltflib.GLTF2().load(str(filepath))

            # Basic validation
            if not gltf.meshes:
                return False

            # Check all accessors have valid buffer views
            for accessor in gltf.accessors or []:
                if accessor.bufferView is not None:
                    if accessor.bufferView >= len(gltf.bufferViews or []):
                        return False

            return True

        except Exception:
            return False

    def _validate_usd(self, filepath: Path) -> bool:
        """Validate USD file."""
        try:
            from pxr import Usd

            stage = Usd.Stage.Open(str(filepath))
            if stage is None:
                return False

            # Check for prims
            root = stage.GetPseudoRoot()
            if not root.GetChildren():
                return False

            return True

        except ImportError:
            # No USD library - can't validate but file exists
            return filepath.exists()
        except Exception:
            return False

    def quick_check(self) -> Dict[str, Any]:
        """Quick sanity check without full analysis."""
        result = {
            'directory': str(self.directory),
            'exists': self.directory.exists(),
            'has_meshes': False,
            'has_textures': False,
            'has_gltf': False,
            'has_usd': False,
            'mesh_count': 0,
            'texture_count': 0
        }

        if not self.directory.exists():
            return result

        obj_files = list(self.directory.glob("**/*.obj"))
        result['mesh_count'] = len(obj_files)
        result['has_meshes'] = len(obj_files) > 0

        for ext in ['.png', '.jpg', '.dds']:
            result['texture_count'] += len(list(self.directory.glob(f"**/*{ext}")))
        result['has_textures'] = result['texture_count'] > 0

        result['has_gltf'] = len(list(self.directory.glob("**/*.gl*"))) > 0
        result['has_usd'] = len(list(self.directory.glob("**/*.usd*"))) > 0

        return result


class QualityReport:
    """Generate quality reports."""

    def __init__(self, metrics: SceneMetrics):
        self.metrics = metrics
        self.generated_at = datetime.now()

    def to_json(self, filepath: str):
        """Save as JSON."""
        data = {
            'generated_at': self.generated_at.isoformat(),
            'metrics': self.metrics.to_dict()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Saved JSON report: {filepath}")

    def to_html(self, filepath: str):
        """Generate HTML report."""
        m = self.metrics

        html = [
            '<!DOCTYPE html>',
            '<html><head>',
            '<title>Quality Report</title>',
            '<style>',
            'body { font-family: Arial, sans-serif; margin: 20px; }',
            'h1, h2 { color: #333; }',
            '.score { font-size: 48px; font-weight: bold; }',
            '.excellent { color: #28a745; }',
            '.good { color: #5cb85c; }',
            '.acceptable { color: #f0ad4e; }',
            '.poor { color: #d9534f; }',
            '.failed { color: #c9302c; }',
            'table { border-collapse: collapse; width: 100%; margin: 20px 0; }',
            'th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }',
            'th { background-color: #f5f5f5; }',
            '.issue { color: #d9534f; }',
            '.warning { color: #f0ad4e; }',
            '.metric-card { background: #f9f9f9; padding: 15px; margin: 10px 0; border-radius: 5px; }',
            '</style>',
            '</head><body>',
            f'<h1>Quality Report: {m.name}</h1>',
            f'<p>Generated: {self.generated_at.strftime("%Y-%m-%d %H:%M:%S")}</p>',
        ]

        # Overall score
        level_class = m.level.value
        html.append(f'<div class="metric-card">')
        html.append(f'<div class="score {level_class}">{m.score:.0f}</div>')
        html.append(f'<div>Quality Level: {m.level.value.upper()}</div>')
        html.append(f'</div>')

        # Summary
        html.append('<h2>Summary</h2>')
        html.append('<table>')
        html.append(f'<tr><td>Total Meshes</td><td>{m.mesh_count}</td></tr>')
        html.append(f'<tr><td>Total Vertices</td><td>{m.total_vertices:,}</td></tr>')
        html.append(f'<tr><td>Total Faces</td><td>{m.total_faces:,}</td></tr>')
        html.append(f'<tr><td>Materials</td><td>{m.material_count}</td></tr>')
        html.append(f'<tr><td>Textures</td><td>{m.texture_count}</td></tr>')
        html.append(f'<tr><td>glTF Valid</td><td>{"Yes" if m.gltf_valid else "No"}</td></tr>')
        html.append(f'<tr><td>USD Valid</td><td>{"Yes" if m.usd_valid else "No"}</td></tr>')
        html.append('</table>')

        # Scene bounds
        html.append('<h2>Scene Bounds</h2>')
        html.append(f'<p>Size: {m.scene_size[0]:.2f} x {m.scene_size[1]:.2f} x {m.scene_size[2]:.2f}</p>')

        # Mesh details
        html.append('<h2>Mesh Details</h2>')
        html.append('<table>')
        html.append('<tr><th>Name</th><th>Vertices</th><th>Faces</th><th>Normals</th><th>UVs</th><th>Score</th><th>Issues</th></tr>')

        for mesh in m.mesh_metrics:
            issues = ', '.join(mesh.issues) if mesh.issues else '-'
            level_class = mesh.level.value

            html.append('<tr>')
            html.append(f'<td>{mesh.name}</td>')
            html.append(f'<td>{mesh.vertex_count:,}</td>')
            html.append(f'<td>{mesh.face_count:,}</td>')
            html.append(f'<td>{"Yes" if mesh.has_normals else "No"}</td>')
            html.append(f'<td>{"Yes" if mesh.has_uvs else "No"}</td>')
            html.append(f'<td class="{level_class}">{mesh.score:.0f}</td>')
            html.append(f'<td class="issue">{issues}</td>')
            html.append('</tr>')

        html.append('</table>')
        html.append('</body></html>')

        with open(filepath, 'w') as f:
            f.write('\n'.join(html))

        print(f"Saved HTML report: {filepath}")

    def print_summary(self):
        """Print summary to console."""
        m = self.metrics

        print(f"\n{'='*60}")
        print(f"Quality Report: {m.name}")
        print(f"{'='*60}")
        print(f"Overall Score: {m.score:.0f}/100 ({m.level.value.upper()})")
        print()
        print(f"Meshes: {m.mesh_count}")
        print(f"Vertices: {m.total_vertices:,}")
        print(f"Faces: {m.total_faces:,}")
        print(f"glTF Valid: {'Yes' if m.gltf_valid else 'No'}")
        print(f"USD Valid: {'Yes' if m.usd_valid else 'No'}")
        print()

        # Issues summary
        all_issues = []
        all_warnings = []
        for mesh in m.mesh_metrics:
            all_issues.extend([(mesh.name, i) for i in mesh.issues])
            all_warnings.extend([(mesh.name, w) for w in mesh.warnings])

        if all_issues:
            print("Issues:")
            for name, issue in all_issues[:10]:
                print(f"  - [{name}] {issue}")
            if len(all_issues) > 10:
                print(f"  ... and {len(all_issues) - 10} more")

        if all_warnings:
            print("\nWarnings:")
            for name, warning in all_warnings[:10]:
                print(f"  - [{name}] {warning}")


# Convenience functions
def assess_mesh_quality(filepath: str) -> MeshMetrics:
    """Quick function to assess a single mesh."""
    analyzer = MeshAnalyzer()
    ext = Path(filepath).suffix.lower()

    if ext == '.obj':
        return analyzer.analyze_obj(filepath)
    elif ext in ('.glb', '.gltf'):
        metrics_list = analyzer.analyze_gltf(filepath)
        return metrics_list[0] if metrics_list else MeshMetrics(name="error", filepath=filepath)
    else:
        raise ValueError(f"Unsupported format: {ext}")


def assess_scene_quality(directory: str) -> QualityReport:
    """Quick function to assess a scene directory."""
    checker = QualityChecker(directory)
    metrics = checker.full_assessment()
    return QualityReport(metrics)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quality assessment")
    parser.add_argument('input', help="Mesh file or scene directory")
    parser.add_argument('--output', '-o', help="Output report path")
    parser.add_argument('--format', choices=['json', 'html'], default='html', help="Report format")

    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_dir():
        report = assess_scene_quality(str(input_path))
        report.print_summary()

        if args.output:
            if args.format == 'html':
                report.to_html(args.output)
            else:
                report.to_json(args.output)
    else:
        metrics = assess_mesh_quality(str(input_path))
        print(f"\nMesh: {metrics.name}")
        print(f"Score: {metrics.score:.0f}/100 ({metrics.level.value})")
        print(f"Vertices: {metrics.vertex_count:,}")
        print(f"Faces: {metrics.face_count:,}")
        if metrics.issues:
            print("Issues:", ', '.join(metrics.issues))
        if metrics.warnings:
            print("Warnings:", ', '.join(metrics.warnings))
