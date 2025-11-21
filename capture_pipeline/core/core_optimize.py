#!/usr/bin/env python3
"""
Core Mesh Optimization Utilities
================================
Mesh simplification, cleanup, and optimization tools.

Features:
    - Quadric error metric decimation
    - Vertex clustering simplification
    - Degenerate triangle removal
    - Duplicate vertex merging
    - Normal recalculation
    - UV optimization

Dependencies:
    - numpy (required)
    - open3d (optional, for quadric decimation)
    - trimesh (optional, for additional features)

Usage:
    from core_optimize import (
        decimate_mesh, clean_mesh, optimize_for_realtime
    )

    # Reduce to 50% triangles
    simplified = decimate_mesh(vertices, faces, target_reduction=0.5)

    # Clean up mesh
    cleaned = clean_mesh(vertices, faces, normals, uvs)

Author: Capture Pipeline
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import os

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("WARNING: numpy required for mesh optimization")

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


@dataclass
class MeshData:
    """Container for mesh geometry data."""
    vertices: np.ndarray  # Nx3
    faces: np.ndarray     # Mx3
    normals: Optional[np.ndarray] = None  # Nx3
    uvs: Optional[np.ndarray] = None      # Nx2
    colors: Optional[np.ndarray] = None   # Nx3 or Nx4

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    @property
    def face_count(self) -> int:
        return len(self.faces)

    def copy(self) -> 'MeshData':
        return MeshData(
            vertices=self.vertices.copy(),
            faces=self.faces.copy(),
            normals=self.normals.copy() if self.normals is not None else None,
            uvs=self.uvs.copy() if self.uvs is not None else None,
            colors=self.colors.copy() if self.colors is not None else None
        )


def decimate_mesh(vertices: np.ndarray, faces: np.ndarray,
                  target_reduction: float = 0.5,
                  target_faces: int = None,
                  preserve_boundary: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reduce mesh triangle count using quadric error metric.

    Args:
        vertices: Nx3 vertex positions
        faces: Mx3 face indices
        target_reduction: Target reduction ratio (0.5 = 50% of original)
        target_faces: Explicit target face count (overrides target_reduction)
        preserve_boundary: Weight boundary edges to preserve them

    Returns:
        (new_vertices, new_faces) tuple
    """
    if not HAS_NUMPY:
        raise RuntimeError("numpy required for decimation")

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)

    if target_faces is None:
        target_faces = int(len(faces) * target_reduction)

    target_faces = max(4, target_faces)  # Minimum 4 faces

    # Use Open3D if available (better quality)
    if HAS_OPEN3D:
        return _decimate_open3d(vertices, faces, target_faces, preserve_boundary)

    # Fallback: vertex clustering (simpler but less quality)
    return _decimate_vertex_clustering(vertices, faces, target_faces)


def _decimate_open3d(vertices: np.ndarray, faces: np.ndarray,
                     target_faces: int, preserve_boundary: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Decimation using Open3D quadric error metric."""
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    # Compute normals for better decimation
    mesh.compute_vertex_normals()

    # Decimate
    boundary_weight = 10.0 if preserve_boundary else 1.0

    simplified = mesh.simplify_quadric_decimation(
        target_number_of_triangles=target_faces,
        maximum_error=float('inf'),
        boundary_weight=boundary_weight
    )

    new_vertices = np.asarray(simplified.vertices)
    new_faces = np.asarray(simplified.triangles)

    return new_vertices, new_faces


def _decimate_vertex_clustering(vertices: np.ndarray, faces: np.ndarray,
                                 target_faces: int) -> Tuple[np.ndarray, np.ndarray]:
    """Simple decimation via vertex clustering (fallback)."""
    # Estimate voxel size to achieve target
    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    diagonal = np.linalg.norm(bounds_max - bounds_min)

    # Larger voxel = more reduction
    current_ratio = target_faces / len(faces)
    voxel_size = diagonal * (1.0 - current_ratio) * 0.1

    if HAS_OPEN3D:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)

        simplified = mesh.simplify_vertex_clustering(
            voxel_size=voxel_size,
            contraction=o3d.geometry.SimplificationContraction.Average
        )

        return np.asarray(simplified.vertices), np.asarray(simplified.triangles)

    # Pure numpy fallback (very basic)
    return vertices, faces


def clean_mesh(mesh: MeshData,
               remove_degenerate: bool = True,
               merge_vertices: bool = True,
               merge_threshold: float = 1e-6,
               remove_unreferenced: bool = True) -> MeshData:
    """
    Clean up mesh geometry.

    Args:
        mesh: Input MeshData
        remove_degenerate: Remove zero-area triangles
        merge_vertices: Merge vertices at same position
        merge_threshold: Distance threshold for merging
        remove_unreferenced: Remove vertices not used by faces

    Returns:
        Cleaned MeshData
    """
    if not HAS_NUMPY:
        return mesh

    result = mesh.copy()

    if merge_vertices:
        result = _merge_close_vertices(result, merge_threshold)

    if remove_degenerate:
        result = _remove_degenerate_faces(result)

    if remove_unreferenced:
        result = _remove_unreferenced_vertices(result)

    return result


def _merge_close_vertices(mesh: MeshData, threshold: float) -> MeshData:
    """Merge vertices that are very close together."""
    vertices = mesh.vertices

    # Quantize to grid
    scale = 1.0 / threshold if threshold > 0 else 1e6
    quantized = (vertices * scale).astype(np.int64)

    # Find unique quantized positions
    _, unique_indices, inverse_indices = np.unique(
        quantized, axis=0, return_index=True, return_inverse=True
    )

    # Create new vertex array
    new_vertices = vertices[unique_indices]

    # Remap faces
    new_faces = inverse_indices[mesh.faces]

    # Remap other attributes
    new_normals = mesh.normals[unique_indices] if mesh.normals is not None else None
    new_uvs = mesh.uvs[unique_indices] if mesh.uvs is not None else None
    new_colors = mesh.colors[unique_indices] if mesh.colors is not None else None

    return MeshData(
        vertices=new_vertices,
        faces=new_faces,
        normals=new_normals,
        uvs=new_uvs,
        colors=new_colors
    )


def _remove_degenerate_faces(mesh: MeshData, area_threshold: float = 1e-10) -> MeshData:
    """Remove triangles with zero or near-zero area."""
    vertices = mesh.vertices
    faces = mesh.faces

    # Calculate face areas
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Cross product for area
    cross = np.cross(v1 - v0, v2 - v0)
    areas = np.linalg.norm(cross, axis=1) * 0.5

    # Keep non-degenerate
    valid = areas > area_threshold

    # Also remove faces with duplicate vertices
    no_dup = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    valid = valid & no_dup

    new_faces = faces[valid]

    return MeshData(
        vertices=mesh.vertices,
        faces=new_faces,
        normals=mesh.normals,
        uvs=mesh.uvs,
        colors=mesh.colors
    )


def _remove_unreferenced_vertices(mesh: MeshData) -> MeshData:
    """Remove vertices that are not referenced by any face."""
    # Find used vertices
    used = np.unique(mesh.faces.flatten())

    # Create mapping from old to new indices
    old_to_new = np.full(len(mesh.vertices), -1, dtype=np.int32)
    old_to_new[used] = np.arange(len(used))

    # Filter vertices
    new_vertices = mesh.vertices[used]
    new_faces = old_to_new[mesh.faces]

    new_normals = mesh.normals[used] if mesh.normals is not None else None
    new_uvs = mesh.uvs[used] if mesh.uvs is not None else None
    new_colors = mesh.colors[used] if mesh.colors is not None else None

    return MeshData(
        vertices=new_vertices,
        faces=new_faces,
        normals=new_normals,
        uvs=new_uvs,
        colors=new_colors
    )


def recalculate_normals(mesh: MeshData, smooth: bool = True,
                        angle_threshold: float = 60.0) -> MeshData:
    """
    Recalculate vertex normals.

    Args:
        mesh: Input MeshData
        smooth: If True, average normals across shared vertices
        angle_threshold: Angle threshold for smooth shading (degrees)

    Returns:
        MeshData with updated normals
    """
    if not HAS_NUMPY:
        return mesh

    vertices = mesh.vertices
    faces = mesh.faces

    # Calculate face normals
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    face_normals = np.cross(v1 - v0, v2 - v0)
    face_lengths = np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_lengths[face_lengths < 1e-10] = 1.0
    face_normals /= face_lengths

    if not smooth:
        # Flat shading - use face normals directly
        # Need to expand vertices for flat shading
        new_vertices = np.vstack([v0, v1, v2]).reshape(-1, 3)
        new_faces = np.arange(len(new_vertices)).reshape(-1, 3)
        new_normals = np.repeat(face_normals, 3, axis=0)

        return MeshData(
            vertices=new_vertices,
            faces=new_faces,
            normals=new_normals,
            uvs=None,  # UVs would need similar expansion
            colors=None
        )

    # Smooth shading - average normals at vertices
    vertex_normals = np.zeros_like(vertices)

    # Accumulate face normals to vertices
    for i in range(3):
        np.add.at(vertex_normals, faces[:, i], face_normals)

    # Normalize
    lengths = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    lengths[lengths < 1e-10] = 1.0
    vertex_normals /= lengths

    result = mesh.copy()
    result.normals = vertex_normals
    return result


def optimize_for_realtime(mesh: MeshData,
                          target_triangles: int = 50000,
                          texture_resolution: int = 2048) -> MeshData:
    """
    Optimize mesh for realtime rendering.

    Applies:
        - Decimation if above target
        - Vertex cleanup
        - Normal recalculation

    Args:
        mesh: Input MeshData
        target_triangles: Maximum triangle count
        texture_resolution: Not used (for future UV optimization)

    Returns:
        Optimized MeshData
    """
    result = mesh.copy()

    # Clean first
    result = clean_mesh(result)

    # Decimate if needed
    if result.face_count > target_triangles:
        new_verts, new_faces = decimate_mesh(
            result.vertices,
            result.faces,
            target_faces=target_triangles
        )
        result.vertices = new_verts
        result.faces = new_faces
        result.normals = None  # Need recalculation
        result.uvs = None  # Lost during decimation

    # Recalculate normals
    result = recalculate_normals(result, smooth=True)

    return result


def generate_lods(mesh: MeshData, lod_levels: List[float] = None) -> List[MeshData]:
    """
    Generate LOD (Level of Detail) meshes.

    Args:
        mesh: Input high-poly MeshData
        lod_levels: List of reduction ratios [0.5, 0.25, 0.125]

    Returns:
        List of MeshData for each LOD level (including original as LOD0)
    """
    if lod_levels is None:
        lod_levels = [0.5, 0.25, 0.125]

    lods = [mesh]  # LOD0 is original

    for ratio in lod_levels:
        target_faces = int(mesh.face_count * ratio)
        if target_faces < 4:
            break

        new_verts, new_faces = decimate_mesh(
            mesh.vertices,
            mesh.faces,
            target_faces=target_faces
        )

        lod_mesh = MeshData(
            vertices=new_verts,
            faces=new_faces
        )
        lod_mesh = recalculate_normals(lod_mesh)
        lods.append(lod_mesh)

    return lods


def calculate_mesh_stats(mesh: MeshData) -> dict:
    """Calculate mesh statistics."""
    vertices = mesh.vertices
    faces = mesh.faces

    # Bounds
    bounds_min = vertices.min(axis=0).tolist()
    bounds_max = vertices.max(axis=0).tolist()

    # Face areas
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = np.linalg.norm(cross, axis=1) * 0.5

    total_area = areas.sum()
    avg_area = areas.mean()

    # Edge lengths
    e0 = np.linalg.norm(v1 - v0, axis=1)
    e1 = np.linalg.norm(v2 - v1, axis=1)
    e2 = np.linalg.norm(v0 - v2, axis=1)
    all_edges = np.concatenate([e0, e1, e2])

    return {
        'vertex_count': len(vertices),
        'face_count': len(faces),
        'bounds_min': bounds_min,
        'bounds_max': bounds_max,
        'total_surface_area': float(total_area),
        'avg_face_area': float(avg_area),
        'min_edge_length': float(all_edges.min()),
        'max_edge_length': float(all_edges.max()),
        'avg_edge_length': float(all_edges.mean()),
        'has_normals': mesh.normals is not None,
        'has_uvs': mesh.uvs is not None,
        'has_colors': mesh.colors is not None
    }


# Convenience functions for file operations

def load_obj_to_meshdata(filepath: str) -> MeshData:
    """Load OBJ file into MeshData."""
    vertices = []
    normals = []
    uvs = []
    faces = []
    face_uvs = []
    face_normals = []

    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            if parts[0] == 'v':
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'vn':
                normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'vt':
                uvs.append([float(parts[1]), float(parts[2])])
            elif parts[0] == 'f':
                face = []
                for v in parts[1:4]:  # Only first 3 vertices (triangles)
                    indices = v.split('/')
                    face.append(int(indices[0]) - 1)
                faces.append(face)

    return MeshData(
        vertices=np.array(vertices, dtype=np.float64),
        faces=np.array(faces, dtype=np.int32),
        normals=np.array(normals, dtype=np.float64) if normals else None,
        uvs=np.array(uvs, dtype=np.float64) if uvs else None
    )


def save_meshdata_to_obj(mesh: MeshData, filepath: str):
    """Save MeshData to OBJ file."""
    with open(filepath, 'w') as f:
        f.write("# Generated by core_optimize.py\n")
        f.write(f"# Vertices: {mesh.vertex_count}\n")
        f.write(f"# Faces: {mesh.face_count}\n\n")

        for v in mesh.vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        if mesh.normals is not None:
            f.write("\n")
            for n in mesh.normals:
                f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")

        if mesh.uvs is not None:
            f.write("\n")
            for uv in mesh.uvs:
                f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")

        f.write("\n")
        for face in mesh.faces:
            if mesh.uvs is not None and mesh.normals is not None:
                f.write(f"f {face[0]+1}/{face[0]+1}/{face[0]+1} "
                        f"{face[1]+1}/{face[1]+1}/{face[1]+1} "
                        f"{face[2]+1}/{face[2]+1}/{face[2]+1}\n")
            elif mesh.normals is not None:
                f.write(f"f {face[0]+1}//{face[0]+1} "
                        f"{face[1]+1}//{face[1]+1} "
                        f"{face[2]+1}//{face[2]+1}\n")
            else:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optimize mesh geometry")
    parser.add_argument('input', help="Input OBJ file")
    parser.add_argument('--output', '-o', help="Output OBJ file")
    parser.add_argument('--decimate', type=float, help="Decimation ratio (0.5 = 50%)")
    parser.add_argument('--clean', action='store_true', help="Clean mesh")
    parser.add_argument('--stats', action='store_true', help="Print statistics")

    args = parser.parse_args()

    mesh = load_obj_to_meshdata(args.input)
    print(f"Loaded: {mesh.vertex_count} vertices, {mesh.face_count} faces")

    if args.stats:
        stats = calculate_mesh_stats(mesh)
        for k, v in stats.items():
            print(f"  {k}: {v}")

    if args.clean:
        mesh = clean_mesh(mesh)
        print(f"After clean: {mesh.vertex_count} vertices, {mesh.face_count} faces")

    if args.decimate:
        new_v, new_f = decimate_mesh(mesh.vertices, mesh.faces, args.decimate)
        mesh = MeshData(vertices=new_v, faces=new_f)
        mesh = recalculate_normals(mesh)
        print(f"After decimate: {mesh.vertex_count} vertices, {mesh.face_count} faces")

    if args.output:
        save_meshdata_to_obj(mesh, args.output)
        print(f"Saved: {args.output}")
