#!/usr/bin/env python3
"""
Ninja Ripper Conversion Script
==============================
Converts Ninja Ripper .rip files to OBJ format for DX9/legacy games
where RenderDoc cannot capture.

Usage:
    python 07_ninja_ripper_convert.py --input rips/ --output export/Meshes/

Ninja Ripper Use Cases:
    - DirectX 9 and older games (RenderDoc incompatible)
    - Emulators (Dolphin, PCSX2, Nox)
    - Anti-cheat protected games where RenderDoc fails
    - Quick mesh+texture capture for reference

Known Limitations:
    - CRC32 deduplication bug skips identical meshes
    - No skeleton/bone data (current pose only)
    - Perspective distortion may require FOV correction
    - Hundreds of separate .rip files per capture

Dependencies:
    pip install numpy

Author: Capture Pipeline
"""

import argparse
import os
import struct
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy required")
    print("Install: pip install numpy")
    sys.exit(1)


class RipFile:
    """
    Parser for Ninja Ripper .rip mesh files.

    .rip format structure:
        - Header with version, vertex count, face count
        - Vertex data block with positions, normals, UVs
        - Index data block for triangle faces
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self.vertices = []
        self.normals = []
        self.uvs = []
        self.faces = []
        self.vertex_count = 0
        self.face_count = 0
        self._parse()

    def _parse(self):
        """Parse .rip file format."""
        with open(self.filepath, 'rb') as f:
            # Read header
            magic = f.read(4)
            if magic != b'RIP\x00' and magic[:3] != b'RIP':
                # Try alternative format
                f.seek(0)

            # Version/signature varies - try common patterns
            f.seek(0)
            data = f.read()

        self._parse_rip_data(data)

    def _parse_rip_data(self, data):
        """
        Parse RIP binary data.

        Format varies by Ninja Ripper version. Common structure:
            4 bytes: signature/version
            4 bytes: vertex count
            4 bytes: face count
            4 bytes: vertex stride
            N bytes: vertex data
            M bytes: index data
        """
        offset = 0

        try:
            # Try format 1: Simple header
            if len(data) < 16:
                return

            # Look for vertex data patterns
            # Typically starts with position floats

            # Attempt to detect format by scanning for patterns
            # This is a simplified parser - real implementation would
            # handle multiple Ninja Ripper versions

            # Format detection: look for float patterns
            test_floats = []
            for i in range(0, min(1000, len(data) - 12), 4):
                try:
                    val = struct.unpack_from('f', data, i)[0]
                    if -10000 < val < 10000 and val != 0:
                        test_floats.append((i, val))
                except struct.error:
                    continue

            if not test_floats:
                return

            # Assume vertex data starts at offset 16 (common)
            # with stride of 32 bytes (pos3 + norm3 + uv2 = 32 bytes)

            header_size = 16
            vertex_stride = 32

            # Read header
            if len(data) < header_size:
                return

            try:
                # Try common header format
                sig, vert_count, face_count, stride = struct.unpack_from('4sIII', data, 0)
                if stride > 0 and stride < 256:
                    vertex_stride = stride
                if vert_count > 0 and vert_count < 1000000:
                    self.vertex_count = vert_count
                if face_count > 0 and face_count < 1000000:
                    self.face_count = face_count
            except struct.error:
                # Fallback: estimate from file size
                vertex_stride = 32
                self.vertex_count = (len(data) - header_size) // vertex_stride
                self.face_count = self.vertex_count // 3

            # Parse vertices
            offset = header_size
            for i in range(self.vertex_count):
                if offset + vertex_stride > len(data):
                    break

                try:
                    # Position (3 floats)
                    x, y, z = struct.unpack_from('fff', data, offset)
                    self.vertices.append([x, y, z])

                    # Normal (3 floats) - offset 12
                    if vertex_stride >= 24:
                        nx, ny, nz = struct.unpack_from('fff', data, offset + 12)
                        self.normals.append([nx, ny, nz])

                    # UVs (2 floats) - offset 24
                    if vertex_stride >= 32:
                        u, v = struct.unpack_from('ff', data, offset + 24)
                        self.uvs.append([u, 1.0 - v])  # Flip V

                    offset += vertex_stride
                except struct.error:
                    break

            # Generate faces (triangle list assumed)
            for i in range(0, len(self.vertices) - 2, 3):
                self.faces.append([i, i + 1, i + 2])

            self.face_count = len(self.faces)

        except Exception as e:
            print(f"  Warning: Parse error in {self.filepath}: {e}")

    def is_valid(self):
        """Check if mesh has valid geometry."""
        return len(self.vertices) >= 3 and len(self.faces) >= 1

    def export_obj(self, filepath):
        """Export to Wavefront OBJ format."""
        if not self.is_valid():
            return False

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# Converted from Ninja Ripper .rip\n")
            f.write(f"# Source: {os.path.basename(self.filepath)}\n")
            f.write(f"# Vertices: {len(self.vertices)}\n")
            f.write(f"# Faces: {len(self.faces)}\n\n")

            # Vertices
            for v in self.vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")

            # Normals
            if self.normals:
                f.write("\n")
                for n in self.normals:
                    f.write(f"vn {n[0]} {n[1]} {n[2]}\n")

            # UVs
            if self.uvs:
                f.write("\n")
                for uv in self.uvs:
                    f.write(f"vt {uv[0]} {uv[1]}\n")

            # Faces (1-indexed)
            f.write("\n")
            has_normals = len(self.normals) == len(self.vertices)
            has_uvs = len(self.uvs) == len(self.vertices)

            for face in self.faces:
                if has_uvs and has_normals:
                    f.write(f"f {face[0]+1}/{face[0]+1}/{face[0]+1} "
                            f"{face[1]+1}/{face[1]+1}/{face[1]+1} "
                            f"{face[2]+1}/{face[2]+1}/{face[2]+1}\n")
                elif has_uvs:
                    f.write(f"f {face[0]+1}/{face[0]+1} "
                            f"{face[1]+1}/{face[1]+1} "
                            f"{face[2]+1}/{face[2]+1}\n")
                elif has_normals:
                    f.write(f"f {face[0]+1}//{face[0]+1} "
                            f"{face[1]+1}//{face[1]+1} "
                            f"{face[2]+1}//{face[2]+1}\n")
                else:
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

        return True


def sort_rips_by_size(rip_files):
    """
    Sort .rip files by size descending.

    Larger files typically contain more detailed meshes.
    Character models are usually 600-800KB.
    """
    sized = [(f, os.path.getsize(f)) for f in rip_files]
    sized.sort(key=lambda x: x[1], reverse=True)
    return [f for f, _ in sized]


def convert_rip_directory(input_dir, output_dir, min_vertices=50, max_files=None):
    """
    Convert all .rip files in directory to OBJ.

    Args:
        input_dir: Directory containing .rip files
        output_dir: Output directory for OBJ files
        min_vertices: Minimum vertex count to include
        max_files: Maximum number of files to convert (None = all)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        return []

    output_path.mkdir(parents=True, exist_ok=True)

    # Find all .rip files
    rip_files = list(input_path.glob("**/*.rip"))
    print(f"Found {len(rip_files)} .rip files")

    if not rip_files:
        return []

    # Sort by size (largest first = most detailed)
    rip_files = sort_rips_by_size(rip_files)

    if max_files:
        rip_files = rip_files[:max_files]
        print(f"Processing top {max_files} by size")

    converted = []
    skipped_small = 0
    skipped_invalid = 0

    for i, rip_path in enumerate(rip_files):
        if (i + 1) % 100 == 0:
            print(f"  Processing {i + 1}/{len(rip_files)}...")

        try:
            rip = RipFile(str(rip_path))

            if not rip.is_valid():
                skipped_invalid += 1
                continue

            if len(rip.vertices) < min_vertices:
                skipped_small += 1
                continue

            # Output filename
            obj_name = rip_path.stem + ".obj"
            obj_path = output_path / obj_name

            if rip.export_obj(str(obj_path)):
                converted.append({
                    "source": str(rip_path),
                    "output": str(obj_path),
                    "vertices": len(rip.vertices),
                    "faces": len(rip.faces)
                })

        except Exception as e:
            print(f"  Error converting {rip_path}: {e}")
            skipped_invalid += 1

    print(f"\n=== Conversion Summary ===")
    print(f"  Converted: {len(converted)}")
    print(f"  Skipped (too small): {skipped_small}")
    print(f"  Skipped (invalid): {skipped_invalid}")

    return converted


def find_associated_textures(rip_dir, mesh_name):
    """
    Find DDS textures associated with a mesh.

    Ninja Ripper saves textures alongside meshes with similar names.
    """
    rip_path = Path(rip_dir)
    mesh_base = Path(mesh_name).stem

    # Look for textures with similar names
    patterns = [
        f"{mesh_base}*.dds",
        f"*{mesh_base}*.dds",
        "Tex_*.dds"
    ]

    textures = []
    for pattern in patterns:
        textures.extend(rip_path.glob(pattern))

    return list(set(textures))


def main():
    parser = argparse.ArgumentParser(
        description="Convert Ninja Ripper .rip files to OBJ (fallback for DX9 games)"
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help="Input directory containing .rip files"
    )
    parser.add_argument(
        '--output', '-o',
        default='export/Meshes',
        help="Output directory for OBJ files"
    )
    parser.add_argument(
        '--min-vertices',
        type=int,
        default=50,
        help="Minimum vertex count to include (default: 50)"
    )
    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help="Maximum files to convert (largest first)"
    )

    args = parser.parse_args()

    print("\n=== Ninja Ripper Conversion ===")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")

    converted = convert_rip_directory(
        args.input,
        args.output,
        min_vertices=args.min_vertices,
        max_files=args.max_files
    )

    if converted:
        # Write manifest
        import json
        manifest_path = Path(args.output) / "rip_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump({
                "source_dir": args.input,
                "meshes": converted,
                "total": len(converted)
            }, f, indent=2)
        print(f"\nManifest saved: {manifest_path}")

    print("\n[OK] Ninja Ripper conversion complete!")


if __name__ == "__main__":
    main()
