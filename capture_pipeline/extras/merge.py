#!/usr/bin/env python3
"""
Core Mesh Merging Utilities
===========================
Utilities for merging and combining multiple meshes.

Features:
    - Combine multiple meshes into one
    - Merge by material
    - Merge by spatial proximity
    - Batch mesh combination
    - UV space management during merge

Usage:
    from core_merge import merge_meshes, merge_by_material

    # Combine all meshes into one
    merged = merge_meshes([mesh1, mesh2, mesh3])

    # Merge meshes with same material
    merged_dict = merge_by_material(meshes, materials)

Author: Capture Pipeline
"""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    raise ImportError("numpy required for mesh merging")


@dataclass
class MeshData:
    """Container for mesh geometry data."""
    name: str
    vertices: np.ndarray  # Nx3
    faces: np.ndarray     # Mx3
    normals: Optional[np.ndarray] = None  # Nx3
    uvs: Optional[np.ndarray] = None      # Nx2
    material_id: int = -1

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    @property
    def face_count(self) -> int:
        return len(self.faces)

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounding box (min, max)."""
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def center(self) -> np.ndarray:
        """Get center point."""
        min_b, max_b = self.bounds()
        return (min_b + max_b) / 2


def merge_meshes(meshes: List[MeshData], name: str = "merged") -> MeshData:
    """
    Merge multiple meshes into a single mesh.

    Args:
        meshes: List of MeshData to merge
        name: Name for the merged mesh

    Returns:
        Single merged MeshData
    """
    if not meshes:
        raise ValueError("No meshes to merge")

    if len(meshes) == 1:
        return meshes[0]

    # Collect all data
    all_vertices = []
    all_faces = []
    all_normals = []
    all_uvs = []

    vertex_offset = 0
    has_normals = all(m.normals is not None for m in meshes)
    has_uvs = all(m.uvs is not None for m in meshes)

    for mesh in meshes:
        # Add vertices
        all_vertices.append(mesh.vertices)

        # Offset faces
        faces = mesh.faces + vertex_offset
        all_faces.append(faces)

        # Add normals if all have them
        if has_normals:
            all_normals.append(mesh.normals)
        elif mesh.normals is not None:
            has_normals = False
            all_normals = []

        # Add UVs if all have them
        if has_uvs:
            all_uvs.append(mesh.uvs)
        elif mesh.uvs is not None:
            has_uvs = False
            all_uvs = []

        vertex_offset += mesh.vertex_count

    # Concatenate
    merged_vertices = np.vstack(all_vertices)
    merged_faces = np.vstack(all_faces)
    merged_normals = np.vstack(all_normals) if has_normals else None
    merged_uvs = np.vstack(all_uvs) if has_uvs else None

    return MeshData(
        name=name,
        vertices=merged_vertices,
        faces=merged_faces,
        normals=merged_normals,
        uvs=merged_uvs
    )


def merge_by_material(meshes: List[MeshData]) -> Dict[int, MeshData]:
    """
    Merge meshes that share the same material.

    Args:
        meshes: List of MeshData with material_id set

    Returns:
        Dict mapping material_id -> merged MeshData
    """
    # Group by material
    material_groups: Dict[int, List[MeshData]] = {}

    for mesh in meshes:
        mat_id = mesh.material_id
        if mat_id not in material_groups:
            material_groups[mat_id] = []
        material_groups[mat_id].append(mesh)

    # Merge each group
    result = {}
    for mat_id, group in material_groups.items():
        merged = merge_meshes(group, name=f"merged_mat_{mat_id}")
        merged.material_id = mat_id
        result[mat_id] = merged

    return result


def merge_by_proximity(meshes: List[MeshData],
                       distance_threshold: float = 100.0) -> List[MeshData]:
    """
    Merge meshes that are spatially close together.

    Args:
        meshes: List of MeshData
        distance_threshold: Maximum distance between mesh centers to merge

    Returns:
        List of merged MeshData (fewer than input if merges occurred)
    """
    if len(meshes) <= 1:
        return meshes

    # Calculate centers
    centers = [m.center() for m in meshes]

    # Build groups via distance
    assigned = [False] * len(meshes)
    groups: List[List[int]] = []

    for i in range(len(meshes)):
        if assigned[i]:
            continue

        group = [i]
        assigned[i] = True

        for j in range(i + 1, len(meshes)):
            if assigned[j]:
                continue

            dist = np.linalg.norm(centers[i] - centers[j])
            if dist < distance_threshold:
                group.append(j)
                assigned[j] = True

        groups.append(group)

    # Merge groups
    result = []
    for group_idx, group in enumerate(groups):
        if len(group) == 1:
            result.append(meshes[group[0]])
        else:
            group_meshes = [meshes[i] for i in group]
            merged = merge_meshes(group_meshes, name=f"merged_group_{group_idx}")
            result.append(merged)

    return result


def split_by_material(mesh: MeshData,
                      face_materials: np.ndarray) -> Dict[int, MeshData]:
    """
    Split a single mesh by per-face material assignment.

    Args:
        mesh: Input MeshData
        face_materials: Array of material IDs per face (length = face_count)

    Returns:
        Dict mapping material_id -> MeshData
    """
    unique_materials = np.unique(face_materials)

    result = {}
    for mat_id in unique_materials:
        mat_id = int(mat_id)

        # Find faces with this material
        mask = face_materials == mat_id
        mat_faces = mesh.faces[mask]

        # Find used vertices
        used_verts = np.unique(mat_faces.flatten())

        # Create vertex remapping
        old_to_new = np.full(mesh.vertex_count, -1, dtype=np.int32)
        old_to_new[used_verts] = np.arange(len(used_verts))

        # Extract vertices and remap faces
        new_vertices = mesh.vertices[used_verts]
        new_faces = old_to_new[mat_faces]

        new_normals = mesh.normals[used_verts] if mesh.normals is not None else None
        new_uvs = mesh.uvs[used_verts] if mesh.uvs is not None else None

        result[mat_id] = MeshData(
            name=f"{mesh.name}_mat{mat_id}",
            vertices=new_vertices,
            faces=new_faces,
            normals=new_normals,
            uvs=new_uvs,
            material_id=mat_id
        )

    return result


def merge_with_uv_atlas(meshes: List[MeshData],
                        atlas_layout: List[Tuple[float, float, float, float]]
                        ) -> MeshData:
    """
    Merge meshes while remapping UVs to atlas positions.

    Args:
        meshes: List of MeshData with UVs
        atlas_layout: List of (u_min, v_min, u_max, v_max) for each mesh

    Returns:
        Merged MeshData with remapped UVs
    """
    if len(meshes) != len(atlas_layout):
        raise ValueError("Must provide atlas layout for each mesh")

    remapped_meshes = []

    for mesh, (u_min, v_min, u_max, v_max) in zip(meshes, atlas_layout):
        if mesh.uvs is None:
            remapped_meshes.append(mesh)
            continue

        # Remap UVs from [0,1] to atlas region
        new_uvs = mesh.uvs.copy()
        new_uvs[:, 0] = u_min + new_uvs[:, 0] * (u_max - u_min)
        new_uvs[:, 1] = v_min + new_uvs[:, 1] * (v_max - v_min)

        remapped = MeshData(
            name=mesh.name,
            vertices=mesh.vertices,
            faces=mesh.faces,
            normals=mesh.normals,
            uvs=new_uvs,
            material_id=mesh.material_id
        )
        remapped_meshes.append(remapped)

    return merge_meshes(remapped_meshes, name="atlas_merged")


def weld_vertices(mesh: MeshData, threshold: float = 1e-5) -> MeshData:
    """
    Weld vertices that are very close together.

    This is useful after merging to remove duplicate vertices at seams.

    Args:
        mesh: Input MeshData
        threshold: Distance threshold for welding

    Returns:
        MeshData with welded vertices
    """
    vertices = mesh.vertices

    # Quantize to grid
    scale = 1.0 / threshold if threshold > 0 else 1e6
    quantized = (vertices * scale).astype(np.int64)

    # Find unique positions
    _, unique_indices, inverse = np.unique(
        quantized, axis=0, return_index=True, return_inverse=True
    )

    # Create new vertex array
    new_vertices = vertices[unique_indices]

    # Remap faces
    new_faces = inverse[mesh.faces]

    # Average normals for welded vertices
    new_normals = None
    if mesh.normals is not None:
        new_normals = np.zeros((len(new_vertices), 3), dtype=np.float32)
        counts = np.zeros(len(new_vertices), dtype=np.int32)

        for old_idx, new_idx in enumerate(inverse):
            new_normals[new_idx] += mesh.normals[old_idx]
            counts[new_idx] += 1

        # Normalize
        counts[counts == 0] = 1
        new_normals /= counts[:, np.newaxis]

        # Renormalize
        lengths = np.linalg.norm(new_normals, axis=1, keepdims=True)
        lengths[lengths < 1e-10] = 1.0
        new_normals /= lengths

    # Take first UV for welded vertices
    new_uvs = mesh.uvs[unique_indices] if mesh.uvs is not None else None

    return MeshData(
        name=mesh.name,
        vertices=new_vertices,
        faces=new_faces,
        normals=new_normals,
        uvs=new_uvs,
        material_id=mesh.material_id
    )


def batch_merge(mesh_files: List[str], output_path: str,
                by_material: bool = False,
                weld: bool = True) -> bool:
    """
    Batch merge OBJ files.

    Args:
        mesh_files: List of OBJ file paths
        output_path: Output OBJ path
        by_material: Group by material instead of single merge
        weld: Weld close vertices

    Returns:
        True if successful
    """
    meshes = []

    for filepath in mesh_files:
        mesh = load_obj(filepath)
        if mesh is not None:
            meshes.append(mesh)

    if not meshes:
        print("No valid meshes to merge")
        return False

    if by_material:
        # Group by material and output multiple files
        merged_dict = merge_by_material(meshes)
        for mat_id, merged in merged_dict.items():
            if weld:
                merged = weld_vertices(merged)
            out_file = output_path.replace('.obj', f'_mat{mat_id}.obj')
            save_obj(merged, out_file)
    else:
        # Single merged output
        merged = merge_meshes(meshes, name="merged")
        if weld:
            merged = weld_vertices(merged)
        save_obj(merged, output_path)

    return True


def load_obj(filepath: str) -> Optional[MeshData]:
    """Load OBJ file into MeshData."""
    vertices = []
    normals = []
    uvs = []
    faces = []

    name = Path(filepath).stem

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
                    for v in parts[1:4]:  # Triangles only
                        idx = int(v.split('/')[0]) - 1
                        face.append(idx)
                    if len(face) >= 3:
                        faces.append(face)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

    if not vertices or not faces:
        return None

    return MeshData(
        name=name,
        vertices=np.array(vertices, dtype=np.float64),
        faces=np.array(faces, dtype=np.int32),
        normals=np.array(normals, dtype=np.float64) if normals else None,
        uvs=np.array(uvs, dtype=np.float64) if uvs else None
    )


def save_obj(mesh: MeshData, filepath: str):
    """Save MeshData to OBJ file."""
    with open(filepath, 'w') as f:
        f.write(f"# Merged mesh: {mesh.name}\n")
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

    print(f"Saved: {filepath} ({mesh.vertex_count} verts, {mesh.face_count} faces)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge multiple meshes")
    parser.add_argument('inputs', nargs='+', help="Input OBJ files")
    parser.add_argument('--output', '-o', required=True, help="Output OBJ file")
    parser.add_argument('--weld', action='store_true', help="Weld close vertices")
    parser.add_argument('--by-material', action='store_true', help="Group by material")

    args = parser.parse_args()

    batch_merge(args.inputs, args.output, args.by_material, args.weld)
