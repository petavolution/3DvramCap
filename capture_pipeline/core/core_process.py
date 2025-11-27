#!/usr/bin/env python3
"""
Core Mesh Processing Script (Simplified)
=========================================
Deduplicates meshes and prepares them for Blender import.

Usage:
    python core_process.py --input export/scene/Meshes/ --output library/

Features:
    - Conservative geometry-based deduplication
    - LOD detection (keeps highest-poly version)
    - Mesh validation and filtering
    - Index file generation for tracking

Author: Capture Pipeline
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

# Import logging utilities
try:
    from core.utils import logger, log_error
    HAS_LOGGING = True
except ImportError:
    # Fallback if utils not available
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    HAS_LOGGING = False
    def log_error(msg, exception=None, context=None):
        if exception:
            logger.error(f"{msg}: {exception}")
        else:
            logger.error(msg)

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    logger.warning("numpy not installed. Install with: pip install numpy")


def parse_obj_file(filepath):
    """
    Parse OBJ file and extract geometry data.

    Returns dict with vertices, normals, uvs, faces, or None on error.
    """
    vertices = []
    normals = []
    uvs = []
    faces = []

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                if not parts:
                    continue

                cmd = parts[0]

                if cmd == 'v' and len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif cmd == 'vn' and len(parts) >= 4:
                    normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif cmd == 'vt' and len(parts) >= 3:
                    uvs.append([float(parts[1]), float(parts[2])])
                elif cmd == 'f':
                    face = []
                    for v in parts[1:]:
                        # Parse v, v/vt, v/vt/vn, v//vn formats
                        idx = v.split('/')[0]
                        face.append(int(idx) - 1)  # OBJ is 1-indexed
                    if len(face) >= 3:
                        faces.append(face)

    except Exception as e:
        log_error(f"Error parsing {filepath}", exception=e)
        return None

    if len(vertices) < 3 or len(faces) < 1:
        return None

    return {
        'vertices': vertices,
        'normals': normals,
        'uvs': uvs,
        'faces': faces,
        'filepath': str(filepath)
    }


def compute_geometry_hash(mesh_data, precision=4):
    """
    Compute a hash based on normalized geometry.

    This identifies meshes that are geometrically identical regardless of
    vertex order or small floating-point differences.
    """
    if not HAS_NUMPY:
        # Fallback: simple vertex count + face count hash
        return hashlib.md5(
            f"{len(mesh_data['vertices'])}_{len(mesh_data['faces'])}".encode()
        ).hexdigest()[:16]

    vertices = np.array(mesh_data['vertices'])

    if len(vertices) == 0:
        return None

    # Normalize to unit cube centered at origin
    centroid = vertices.mean(axis=0)
    centered = vertices - centroid

    # Scale to unit size
    scale = np.abs(centered).max()
    if scale > 1e-6:
        normalized = centered / scale
    else:
        normalized = centered

    # Round to precision and sort
    rounded = np.round(normalized, decimals=precision)
    sorted_verts = np.sort(rounded.flatten())

    # Hash the sorted, normalized vertices
    hash_input = sorted_verts.tobytes()

    # Also include face count for additional discrimination
    hash_input += struct.pack('I', len(mesh_data['faces']))

    return hashlib.md5(hash_input).hexdigest()[:16]


def compute_bounds(mesh_data):
    """Compute bounding box of mesh."""
    if not HAS_NUMPY:
        verts = mesh_data['vertices']
        min_v = [min(v[i] for v in verts) for i in range(3)]
        max_v = [max(v[i] for v in verts) for i in range(3)]
        return min_v, max_v

    vertices = np.array(mesh_data['vertices'])
    return vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()


def is_valid_mesh(mesh_data, min_vertices=50, max_bounds=50000):
    """
    Check if mesh passes validity criteria.

    Args:
        mesh_data: Parsed mesh data
        min_vertices: Minimum vertex count
        max_bounds: Maximum bounding box dimension
    """
    # Check vertex count
    if len(mesh_data['vertices']) < min_vertices:
        return False, "too_small"

    # Check for degenerate geometry
    min_b, max_b = compute_bounds(mesh_data)
    size = [max_b[i] - min_b[i] for i in range(3)]

    if max(size) > max_bounds:
        return False, "too_large"

    if max(size) < 0.001:
        return False, "degenerate"

    return True, "ok"


def deduplicate_meshes(input_dir, output_dir, min_vertices=50):
    """
    Deduplicate meshes using geometry hashing.

    Strategy:
        1. Hash each mesh based on normalized geometry
        2. For duplicate hashes, keep the highest-poly version
        3. Copy unique meshes to output directory

    Returns:
        dict with deduplication results
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all OBJ files
    obj_files = list(input_path.glob("**/*.obj"))
    logger.info(f"Found {len(obj_files)} OBJ files")

    if not obj_files:
        logger.error("No OBJ files found in input directory")
        return {'error': 'No OBJ files found'}

    # Parse and hash all meshes
    mesh_groups = {}  # hash -> list of (mesh_data, vertex_count)
    skipped = {'invalid': 0, 'parse_error': 0}

    logger.info("Parsing meshes...")
    for i, obj_path in enumerate(obj_files):
        if (i + 1) % 100 == 0:
            logger.debug(f"Processing {i + 1}/{len(obj_files)}...")

        mesh_data = parse_obj_file(obj_path)
        if mesh_data is None:
            skipped['parse_error'] += 1
            continue

        valid, reason = is_valid_mesh(mesh_data, min_vertices)
        if not valid:
            skipped['invalid'] += 1
            continue

        # Compute hash
        geom_hash = compute_geometry_hash(mesh_data)
        if geom_hash is None:
            skipped['invalid'] += 1
            continue

        vertex_count = len(mesh_data['vertices'])

        if geom_hash not in mesh_groups:
            mesh_groups[geom_hash] = []
        mesh_groups[geom_hash].append((mesh_data, vertex_count))

    # Select best mesh from each group (highest vertex count)
    unique_meshes = []
    duplicate_count = 0

    for geom_hash, group in mesh_groups.items():
        # Sort by vertex count descending
        group.sort(key=lambda x: x[1], reverse=True)
        best_mesh, best_count = group[0]
        unique_meshes.append({
            'mesh': best_mesh,
            'hash': geom_hash,
            'vertex_count': best_count,
            'duplicates': len(group) - 1
        })
        duplicate_count += len(group) - 1

    logger.info(f"Unique meshes: {len(unique_meshes)}")
    logger.info(f"Duplicates removed: {duplicate_count}")

    # Copy unique meshes to output
    logger.info("Copying unique meshes...")
    results = {
        'input_dir': str(input_dir),
        'output_dir': str(output_dir),
        'meshes': [],
        'stats': {
            'total_input': len(obj_files),
            'unique_output': len(unique_meshes),
            'duplicates_removed': duplicate_count,
            'skipped_invalid': skipped['invalid'],
            'skipped_parse_error': skipped['parse_error']
        }
    }

    for i, item in enumerate(unique_meshes):
        mesh_data = item['mesh']
        src_path = Path(mesh_data['filepath'])

        # Generate output filename
        out_name = f"mesh_{i:04d}.obj"
        out_path = output_path / out_name

        shutil.copy2(src_path, out_path)

        min_b, max_b = compute_bounds(mesh_data)
        results['meshes'].append({
            'name': out_name,
            'path': str(out_path),
            'source': str(src_path),
            'hash': item['hash'],
            'vertices': item['vertex_count'],
            'faces': len(mesh_data['faces']),
            'duplicates': item['duplicates'],
            'bounds': {'min': min_b, 'max': max_b}
        })

    # Save index file
    index_path = output_path / "mesh_index.json"
    with open(index_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info("Deduplication complete!")
    logger.info(f"Output: {output_path}")
    logger.info(f"Unique meshes: {len(unique_meshes)}")
    logger.info(f"Index: {index_path}")

    return results


# Import struct for hash computation
import struct


def main():
    parser = argparse.ArgumentParser(
        description="Deduplicate and prepare meshes for Blender import"
    )
    parser.add_argument('--input', '-i', required=True,
                        help="Input directory with OBJ files")
    parser.add_argument('--output', '-o', required=True,
                        help="Output directory for unique meshes")
    parser.add_argument('--min-vertices', type=int, default=50,
                        help="Minimum vertex count (default: 50)")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        logger.error(f"Input directory not found: {args.input}")
        sys.exit(1)

    results = deduplicate_meshes(args.input, args.output, args.min_vertices)

    if 'error' in results:
        logger.error(results['error'])
        sys.exit(1)


if __name__ == "__main__":
    main()
