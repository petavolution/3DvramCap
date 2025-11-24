#!/usr/bin/env python3
"""
Mesh Deduplication Script
=========================
Conservatively deduplicates exported meshes to avoid redundant geometry.

Usage:
    python 02_dedupe_meshes.py <input_meshes_dir> <output_index_json>

Example:
    python 02_dedupe_meshes.py export/scene/Meshes library/index_meshes.json

Key Concepts:
    - Geometry hash: Unique fingerprint based on vertex positions and topology
    - Conservative dedupe: Only merge meshes with identical hash (no "almost same")
    - LOD handling: When same location but different detail, keep highest-poly

Dependencies:
    pip install numpy trimesh

Author: Capture Pipeline
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

try:
    import numpy as np
    import trimesh
except ImportError:
    print("ERROR: Required packages not found.")
    print("Install with: pip install numpy trimesh")
    sys.exit(1)


def compute_geometry_hash(mesh):
    """
    Compute a stable hash for mesh geometry.

    Strategy:
        1. Normalize vertices to centroid at origin
        2. Scale to unit bounding sphere
        3. Quantize positions to reduce floating-point noise
        4. Hash positions + face topology

    This makes the hash position/scale independent.
    """
    if len(mesh.vertices) == 0:
        return None

    vertices = np.array(mesh.vertices, dtype=np.float64)

    # Center at centroid
    centroid = vertices.mean(axis=0)
    vertices = vertices - centroid

    # Normalize scale
    max_dist = np.linalg.norm(vertices, axis=1).max()
    if max_dist < 1e-9:
        return None  # Degenerate mesh
    vertices = vertices / max_dist

    # Quantize to reduce noise (1000 levels per unit)
    vertices_quantized = (vertices * 1000).round().astype(np.int32)

    # Create hash from position data + face topology
    pos_bytes = vertices_quantized.tobytes()
    face_bytes = np.array(mesh.faces, dtype=np.int32).tobytes()

    combined = pos_bytes + face_bytes
    return hashlib.sha256(combined).hexdigest()


def compute_bbox(mesh):
    """Get bounding box [min, max] for mesh."""
    if len(mesh.vertices) == 0:
        return None
    verts = np.array(mesh.vertices)
    return {
        "min": verts.min(axis=0).tolist(),
        "max": verts.max(axis=0).tolist(),
        "center": verts.mean(axis=0).tolist()
    }


def load_mesh_safe(path):
    """Load mesh with error handling."""
    try:
        mesh = trimesh.load(path, force='mesh', process=False)

        # Handle scene vs single mesh
        if isinstance(mesh, trimesh.Scene):
            # Combine all geometries in scene
            meshes = list(mesh.geometry.values())
            if len(meshes) == 0:
                return None
            mesh = trimesh.util.concatenate(meshes)

        if len(mesh.vertices) < 3 or len(mesh.faces) == 0:
            return None

        return mesh
    except Exception as e:
        print(f"  Warning: Could not load {path}: {e}")
        return None


def find_lod_duplicates(mesh_infos, distance_threshold=1.0):
    """
    Find meshes that are likely LOD variants (same location, different detail).

    Returns dict mapping mesh paths to "keep" or "skip_lod".
    """
    lod_status = {}

    # Group by approximate center location
    location_groups = {}

    for info in mesh_infos:
        if info.get("bbox") is None:
            continue

        center = tuple(round(c, 1) for c in info["bbox"]["center"])

        if center not in location_groups:
            location_groups[center] = []
        location_groups[center].append(info)

    # For each location group, keep highest-poly
    for center, group in location_groups.items():
        if len(group) <= 1:
            for info in group:
                lod_status[info["path"]] = "keep"
            continue

        # Sort by vertex count descending
        group.sort(key=lambda x: x["vertex_count"], reverse=True)

        # Keep highest-poly
        lod_status[group[0]["path"]] = "keep"

        # Mark others as LOD duplicates
        for info in group[1:]:
            # Only mark as LOD if significantly less detail
            ratio = info["vertex_count"] / group[0]["vertex_count"]
            if ratio < 0.7:  # Less than 70% of highest detail
                lod_status[info["path"]] = "skip_lod"
            else:
                lod_status[info["path"]] = "keep"

    return lod_status


def deduplicate_meshes(input_dir, output_index_path, output_mesh_dir=None):
    """
    Main deduplication function.

    Process:
        1. Load all meshes and compute geometry hash
        2. Group by hash (exact duplicates)
        3. For each group, keep highest-poly version
        4. Check for LOD duplicates (same location, different detail)
        5. Copy canonical meshes to output directory
        6. Write index JSON

    Args:
        input_dir: Directory containing OBJ files
        output_index_path: Path to write index JSON
        output_mesh_dir: Optional directory to copy canonical meshes
    """
    input_path = Path(input_dir)

    if not input_path.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        return None

    # Find all OBJ files
    obj_files = list(input_path.glob("**/*.obj"))
    print(f"Found {len(obj_files)} mesh files")

    if len(obj_files) == 0:
        print("No meshes to process")
        return {"meshes": {}, "stats": {"total": 0}}

    # Process each mesh
    mesh_infos = []

    for i, obj_path in enumerate(obj_files):
        if (i + 1) % 50 == 0:
            print(f"  Processing {i + 1}/{len(obj_files)}...")

        mesh = load_mesh_safe(str(obj_path))
        if mesh is None:
            continue

        geom_hash = compute_geometry_hash(mesh)
        if geom_hash is None:
            continue

        bbox = compute_bbox(mesh)

        mesh_infos.append({
            "path": str(obj_path),
            "filename": obj_path.name,
            "hash": geom_hash,
            "vertex_count": len(mesh.vertices),
            "face_count": len(mesh.faces),
            "bbox": bbox
        })

    print(f"Successfully processed {len(mesh_infos)} meshes")

    # Group by geometry hash
    hash_groups = {}
    for info in mesh_infos:
        h = info["hash"]
        if h not in hash_groups:
            hash_groups[h] = []
        hash_groups[h].append(info)

    print(f"Found {len(hash_groups)} unique mesh geometries")

    # For each group, pick canonical (highest poly)
    canonical_meshes = {}

    for h, group in hash_groups.items():
        # Sort by vertex count descending
        group.sort(key=lambda x: x["vertex_count"], reverse=True)

        # Pick highest-poly as canonical
        canonical = group[0]
        canonical_meshes[h] = {
            "canonical": canonical["path"],
            "canonical_filename": canonical["filename"],
            "vertex_count": canonical["vertex_count"],
            "face_count": canonical["face_count"],
            "bbox": canonical["bbox"],
            "duplicates": [g["path"] for g in group[1:]]
        }

    # Check for LOD duplicates
    lod_status = find_lod_duplicates(
        [{"path": v["canonical"], "bbox": v["bbox"], "vertex_count": v["vertex_count"]}
         for v in canonical_meshes.values()]
    )

    # Filter out LOD duplicates
    final_meshes = {}
    skipped_lod = 0

    for h, info in canonical_meshes.items():
        status = lod_status.get(info["canonical"], "keep")
        if status == "skip_lod":
            skipped_lod += 1
            continue
        final_meshes[h] = info

    print(f"After LOD filtering: {len(final_meshes)} canonical meshes")
    print(f"  (Skipped {skipped_lod} lower-LOD duplicates)")

    # Optionally copy canonical meshes to output directory
    if output_mesh_dir:
        os.makedirs(output_mesh_dir, exist_ok=True)

        for h, info in final_meshes.items():
            src_path = info["canonical"]
            dst_filename = info["canonical_filename"]
            dst_path = os.path.join(output_mesh_dir, dst_filename)

            # Copy mesh file
            try:
                import shutil
                shutil.copy2(src_path, dst_path)
                info["library_path"] = dst_path
            except Exception as e:
                print(f"  Warning: Could not copy {src_path}: {e}")

    # Build output index
    output_index = {
        "meshes": final_meshes,
        "stats": {
            "total_input": len(obj_files),
            "processed": len(mesh_infos),
            "unique_geometries": len(hash_groups),
            "after_lod_filter": len(final_meshes),
            "duplicates_removed": len(mesh_infos) - len(hash_groups),
            "lod_removed": skipped_lod
        }
    }

    # Write index
    os.makedirs(os.path.dirname(output_index_path), exist_ok=True)
    with open(output_index_path, 'w', encoding='utf-8') as f:
        json.dump(output_index, f, indent=2)

    print(f"\nIndex written to: {output_index_path}")
    print(f"Stats: {output_index['stats']}")

    return output_index


def main():
    parser = argparse.ArgumentParser(
        description="Deduplicate extracted meshes conservatively"
    )
    parser.add_argument(
        'input_dir',
        help="Directory containing OBJ mesh files"
    )
    parser.add_argument(
        'output_index',
        help="Path for output index JSON"
    )
    parser.add_argument(
        '--copy-to',
        dest='copy_to',
        help="Optional: Copy canonical meshes to this directory"
    )

    args = parser.parse_args()

    # Default copy destination
    copy_dir = args.copy_to or "library/meshes"

    deduplicate_meshes(args.input_dir, args.output_index, copy_dir)

    print("\n[OK] Deduplication complete!")


if __name__ == "__main__":
    main()
