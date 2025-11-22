#!/usr/bin/env python3
"""
Ninja Ripper File Parser
========================
Parse .rip files from Ninja Ripper for DX9/DX10/DX11 game captures.

Ninja Ripper is a fallback for games that don't work with RenderDoc,
particularly older DirectX 9 titles.

Features:
    - Parse .rip binary format (v4/v5)
    - Extract vertices, normals, UVs, indices
    - Batch process rip directories
    - Convert to OBJ format
    - Texture reference extraction

Usage:
    from core_ninja import RipParser, parse_rip_directory

    # Parse single file
    parser = RipParser("mesh.rip")
    mesh = parser.parse()
    mesh.to_obj("output.obj")

    # Batch process
    meshes = parse_rip_directory("rip_output/")

Author: Capture Pipeline
"""

import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple, BinaryIO
import re

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class RipVertex:
    """Vertex data from a rip file."""
    position: Tuple[float, float, float]
    normal: Optional[Tuple[float, float, float]] = None
    uv: Optional[Tuple[float, float]] = None
    color: Optional[Tuple[int, int, int, int]] = None


@dataclass
class RipMesh:
    """Parsed mesh data from a .rip file."""
    name: str
    vertices: List[RipVertex]
    indices: List[int]
    textures: List[str] = field(default_factory=list)
    shader_hash: str = ""
    vertex_format: List[str] = field(default_factory=list)

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    @property
    def face_count(self) -> int:
        return len(self.indices) // 3

    def to_obj(self, filepath: str, flip_uvs: bool = True):
        """Export to OBJ format."""
        with open(filepath, 'w') as f:
            f.write(f"# Ninja Ripper mesh: {self.name}\n")
            f.write(f"# Vertices: {self.vertex_count}\n")
            f.write(f"# Faces: {self.face_count}\n")
            if self.textures:
                f.write(f"# Textures: {', '.join(self.textures)}\n")
            f.write("\n")

            # Vertices
            for v in self.vertices:
                f.write(f"v {v.position[0]:.6f} {v.position[1]:.6f} {v.position[2]:.6f}\n")

            # Normals
            has_normals = any(v.normal is not None for v in self.vertices)
            if has_normals:
                f.write("\n")
                for v in self.vertices:
                    if v.normal:
                        f.write(f"vn {v.normal[0]:.6f} {v.normal[1]:.6f} {v.normal[2]:.6f}\n")
                    else:
                        f.write("vn 0.0 1.0 0.0\n")

            # UVs
            has_uvs = any(v.uv is not None for v in self.vertices)
            if has_uvs:
                f.write("\n")
                for v in self.vertices:
                    if v.uv:
                        u, t = v.uv
                        if flip_uvs:
                            t = 1.0 - t
                        f.write(f"vt {u:.6f} {t:.6f}\n")
                    else:
                        f.write("vt 0.0 0.0\n")

            # Faces
            f.write("\n")
            for i in range(0, len(self.indices), 3):
                if i + 2 >= len(self.indices):
                    break

                i0 = self.indices[i] + 1
                i1 = self.indices[i + 1] + 1
                i2 = self.indices[i + 2] + 1

                if has_uvs and has_normals:
                    f.write(f"f {i0}/{i0}/{i0} {i1}/{i1}/{i1} {i2}/{i2}/{i2}\n")
                elif has_normals:
                    f.write(f"f {i0}//{i0} {i1}//{i1} {i2}//{i2}\n")
                elif has_uvs:
                    f.write(f"f {i0}/{i0} {i1}/{i1} {i2}/{i2}\n")
                else:
                    f.write(f"f {i0} {i1} {i2}\n")

        print(f"Saved: {filepath}")

    def to_numpy(self) -> Tuple['np.ndarray', 'np.ndarray', Optional['np.ndarray'], Optional['np.ndarray']]:
        """Convert to numpy arrays."""
        if not HAS_NUMPY:
            raise ImportError("numpy required")

        vertices = np.array([v.position for v in self.vertices], dtype=np.float32)

        # Convert indices to triangles
        indices = np.array(self.indices, dtype=np.int32).reshape(-1, 3)

        normals = None
        if any(v.normal is not None for v in self.vertices):
            normals = np.array([v.normal or (0, 1, 0) for v in self.vertices], dtype=np.float32)

        uvs = None
        if any(v.uv is not None for v in self.vertices):
            uvs = np.array([v.uv or (0, 0) for v in self.vertices], dtype=np.float32)

        return vertices, indices, normals, uvs


class RipParser:
    """
    Parser for Ninja Ripper .rip files.

    Supports version 4 and 5 of the rip format.

    The rip format contains:
    - Header with version and counts
    - Vertex format descriptors
    - Raw vertex data
    - Index data (16 or 32 bit)
    - Texture references
    """

    # Known vertex element types
    ELEMENT_TYPES = {
        0: 'POSITION',
        1: 'BLEND_WEIGHT',
        2: 'BLEND_INDICES',
        3: 'NORMAL',
        4: 'PSIZE',
        5: 'TEXCOORD',
        6: 'TANGENT',
        7: 'BINORMAL',
        8: 'TESSFACTOR',
        9: 'POSITIONT',
        10: 'COLOR',
        11: 'FOG',
        12: 'DEPTH',
        13: 'SAMPLE'
    }

    # Data format types
    FORMAT_TYPES = {
        0: ('float', 1),
        1: ('float', 2),
        2: ('float', 3),
        3: ('float', 4),
        4: ('byte4', 4),
        5: ('ubyte4', 4),
        6: ('short2', 2),
        7: ('short4', 4),
        8: ('ubyte4n', 4),
        9: ('short2n', 2),
        10: ('short4n', 4),
        11: ('ushort2n', 2),
        12: ('ushort4n', 4),
        13: ('udec3', 3),
        14: ('dec3n', 3),
        15: ('float16_2', 2),
        16: ('float16_4', 4),
    }

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.name = self.filepath.stem
        self.version = 0
        self.vertex_elements: List[Dict] = []

    def parse(self) -> Optional[RipMesh]:
        """Parse the .rip file and return mesh data."""
        try:
            with open(self.filepath, 'rb') as f:
                return self._parse_file(f)
        except Exception as e:
            print(f"Error parsing {self.filepath}: {e}")
            return None

    def _parse_file(self, f: BinaryIO) -> Optional[RipMesh]:
        """Internal parsing logic."""
        # Read signature
        signature = f.read(4)

        if signature == b'RIP4':
            self.version = 4
        elif signature == b'RIP5':
            self.version = 5
        else:
            # Try to detect format
            f.seek(0)
            header = struct.unpack('<I', f.read(4))[0]
            if header < 1000:  # Probably vertex count
                return self._parse_legacy(f, header)
            else:
                print(f"Unknown rip format: {signature}")
                return None

        if self.version >= 4:
            return self._parse_v4_v5(f)

        return None

    def _parse_v4_v5(self, f: BinaryIO) -> RipMesh:
        """Parse version 4/5 rip format."""
        # Header
        # Face count, Vertex count, VertexSize, TextureCount, ShaderCount, VertexAttributeCount
        face_count = struct.unpack('<I', f.read(4))[0]
        vertex_count = struct.unpack('<I', f.read(4))[0]
        vertex_size = struct.unpack('<I', f.read(4))[0]
        texture_count = struct.unpack('<I', f.read(4))[0]
        shader_count = struct.unpack('<I', f.read(4))[0]
        attrib_count = struct.unpack('<I', f.read(4))[0]

        # Read vertex attributes
        self.vertex_elements = []
        for _ in range(attrib_count):
            semantic = f.read(32).rstrip(b'\x00').decode('utf-8', errors='ignore')
            semantic_index = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<I', f.read(4))[0]
            size = struct.unpack('<I', f.read(4))[0]
            type_id = struct.unpack('<I', f.read(4))[0]

            self.vertex_elements.append({
                'semantic': semantic,
                'index': semantic_index,
                'offset': offset,
                'size': size,
                'type': type_id
            })

        # Read texture names
        textures = []
        for _ in range(texture_count):
            tex_name_len = struct.unpack('<I', f.read(4))[0]
            tex_name = f.read(tex_name_len).decode('utf-8', errors='ignore').rstrip('\x00')
            textures.append(tex_name)

        # Read shader names (skip for now)
        for _ in range(shader_count):
            shader_name_len = struct.unpack('<I', f.read(4))[0]
            f.read(shader_name_len)  # Skip

        # Read indices
        indices = []
        for _ in range(face_count * 3):
            idx = struct.unpack('<I', f.read(4))[0]
            indices.append(idx)

        # Read vertices
        vertices = []
        for _ in range(vertex_count):
            vertex_data = f.read(vertex_size)
            vertex = self._parse_vertex(vertex_data)
            vertices.append(vertex)

        return RipMesh(
            name=self.name,
            vertices=vertices,
            indices=indices,
            textures=textures,
            vertex_format=[e['semantic'] for e in self.vertex_elements]
        )

    def _parse_vertex(self, data: bytes) -> RipVertex:
        """Parse a single vertex from raw bytes."""
        position = None
        normal = None
        uv = None
        color = None

        for elem in self.vertex_elements:
            offset = elem['offset']
            semantic = elem['semantic'].upper()
            size = elem['size']

            try:
                if 'POSITION' in semantic:
                    if size >= 12:
                        position = struct.unpack_from('<fff', data, offset)
                    elif size >= 8:
                        x, y = struct.unpack_from('<ff', data, offset)
                        position = (x, y, 0.0)

                elif 'NORMAL' in semantic:
                    if size >= 12:
                        normal = struct.unpack_from('<fff', data, offset)

                elif 'TEXCOORD' in semantic or 'UV' in semantic:
                    if uv is None:  # Only take first UV set
                        if size >= 8:
                            uv = struct.unpack_from('<ff', data, offset)
                        elif size >= 4:
                            # Half floats
                            uv = self._unpack_half2(data, offset)

                elif 'COLOR' in semantic:
                    if size >= 4:
                        color = struct.unpack_from('<BBBB', data, offset)

            except struct.error:
                pass

        if position is None:
            position = (0.0, 0.0, 0.0)

        return RipVertex(
            position=position,
            normal=normal,
            uv=uv,
            color=color
        )

    def _unpack_half2(self, data: bytes, offset: int) -> Tuple[float, float]:
        """Unpack two half-precision floats."""
        try:
            import struct

            # Read as uint16
            h1, h2 = struct.unpack_from('<HH', data, offset)

            def half_to_float(h):
                """Convert half-precision to float."""
                sign = (h >> 15) & 1
                exp = (h >> 10) & 0x1F
                frac = h & 0x3FF

                if exp == 0:
                    if frac == 0:
                        return -0.0 if sign else 0.0
                    else:
                        # Subnormal
                        return ((-1) ** sign) * (frac / 1024) * (2 ** -14)
                elif exp == 31:
                    if frac == 0:
                        return float('-inf') if sign else float('inf')
                    else:
                        return float('nan')
                else:
                    return ((-1) ** sign) * (1 + frac / 1024) * (2 ** (exp - 15))

            return (half_to_float(h1), half_to_float(h2))
        except Exception:
            return (0.0, 0.0)

    def _parse_legacy(self, f: BinaryIO, vertex_count: int) -> Optional[RipMesh]:
        """Parse older/legacy rip format."""
        f.seek(0)

        # Try to auto-detect format
        # Many legacy rips are just raw vertices + indices

        # Assume 32-byte vertices (pos + normal + uv)
        vertex_size = 32

        try:
            vertices = []
            for _ in range(vertex_count):
                data = f.read(vertex_size)
                if len(data) < vertex_size:
                    break

                pos = struct.unpack_from('<fff', data, 0)
                normal = struct.unpack_from('<fff', data, 12)
                uv = struct.unpack_from('<ff', data, 24)

                vertices.append(RipVertex(
                    position=pos,
                    normal=normal,
                    uv=uv
                ))

            # Read indices (remaining data as 16-bit)
            indices = []
            while True:
                idx_data = f.read(2)
                if len(idx_data) < 2:
                    break
                idx = struct.unpack('<H', idx_data)[0]
                indices.append(idx)

            if vertices:
                return RipMesh(
                    name=self.name,
                    vertices=vertices,
                    indices=indices
                )

        except Exception as e:
            print(f"Legacy parse failed: {e}")

        return None


def parse_rip_file(filepath: str) -> Optional[RipMesh]:
    """Parse a single .rip file."""
    parser = RipParser(filepath)
    return parser.parse()


def parse_rip_directory(directory: str,
                        min_vertices: int = 50,
                        output_dir: Optional[str] = None) -> List[RipMesh]:
    """
    Parse all .rip files in a directory.

    Args:
        directory: Input directory containing .rip files
        min_vertices: Minimum vertices to include
        output_dir: If provided, export OBJ files here

    Returns:
        List of parsed RipMesh objects
    """
    rip_dir = Path(directory)
    meshes = []

    # Find all .rip files
    rip_files = sorted(rip_dir.glob("*.rip"))
    print(f"Found {len(rip_files)} rip files in {directory}")

    for rip_file in rip_files:
        parser = RipParser(str(rip_file))
        mesh = parser.parse()

        if mesh is None:
            continue

        if mesh.vertex_count < min_vertices:
            continue

        meshes.append(mesh)

        # Export if output dir specified
        if output_dir:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            obj_path = out_dir / f"{mesh.name}.obj"
            mesh.to_obj(str(obj_path))

    print(f"Parsed {len(meshes)} valid meshes")
    return meshes


def batch_convert_rips(input_dir: str,
                       output_dir: str,
                       merge: bool = False,
                       min_vertices: int = 50) -> bool:
    """
    Batch convert .rip files to OBJ.

    Args:
        input_dir: Directory with .rip files
        output_dir: Output directory for OBJ files
        merge: Merge all meshes into one OBJ
        min_vertices: Minimum vertex count

    Returns:
        True if successful
    """
    meshes = parse_rip_directory(input_dir, min_vertices)

    if not meshes:
        print("No valid meshes found")
        return False

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if merge:
        # Merge all into one OBJ
        merged_path = out_dir / "merged.obj"

        with open(merged_path, 'w') as f:
            f.write("# Merged Ninja Ripper export\n")
            f.write(f"# Meshes: {len(meshes)}\n\n")

            vertex_offset = 0

            for mesh in meshes:
                f.write(f"# Mesh: {mesh.name}\n")
                f.write(f"g {mesh.name}\n")

                # Vertices
                for v in mesh.vertices:
                    f.write(f"v {v.position[0]:.6f} {v.position[1]:.6f} {v.position[2]:.6f}\n")

                # UVs
                has_uvs = any(v.uv is not None for v in mesh.vertices)
                if has_uvs:
                    for v in mesh.vertices:
                        if v.uv:
                            f.write(f"vt {v.uv[0]:.6f} {1.0 - v.uv[1]:.6f}\n")
                        else:
                            f.write("vt 0.0 0.0\n")

                # Faces
                for i in range(0, len(mesh.indices), 3):
                    if i + 2 >= len(mesh.indices):
                        break

                    i0 = mesh.indices[i] + vertex_offset + 1
                    i1 = mesh.indices[i + 1] + vertex_offset + 1
                    i2 = mesh.indices[i + 2] + vertex_offset + 1

                    if has_uvs:
                        f.write(f"f {i0}/{i0} {i1}/{i1} {i2}/{i2}\n")
                    else:
                        f.write(f"f {i0} {i1} {i2}\n")

                f.write("\n")
                vertex_offset += mesh.vertex_count

        print(f"Saved merged: {merged_path}")
    else:
        # Individual files
        for mesh in meshes:
            obj_path = out_dir / f"{mesh.name}.obj"
            mesh.to_obj(str(obj_path))

    # Copy textures
    tex_dir = Path(input_dir)
    for pattern in ["*.dds", "*.png", "*.tga", "*.jpg"]:
        for tex_file in tex_dir.glob(pattern):
            dest = out_dir / tex_file.name
            if not dest.exists():
                import shutil
                shutil.copy2(tex_file, dest)

    return True


def find_texture_for_mesh(mesh_name: str, texture_dir: str) -> Optional[str]:
    """
    Find texture associated with a mesh.

    Ninja Ripper names textures with mesh index prefixes.
    """
    tex_dir = Path(texture_dir)

    # Extract mesh index from name (e.g., "Mesh_001" -> "001")
    match = re.search(r'(\d+)', mesh_name)
    if not match:
        return None

    mesh_idx = match.group(1)

    # Look for matching textures
    for pattern in [f"*{mesh_idx}*.dds", f"*{mesh_idx}*.png", f"Texture_{mesh_idx}*"]:
        matches = list(tex_dir.glob(pattern))
        if matches:
            return str(matches[0])

    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse Ninja Ripper files")
    parser.add_argument('input', help="Input .rip file or directory")
    parser.add_argument('--output', '-o', help="Output directory")
    parser.add_argument('--merge', '-m', action='store_true', help="Merge all meshes")
    parser.add_argument('--min-vertices', type=int, default=50, help="Minimum vertices")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = args.output or str(input_path.parent / "obj_export")

    if input_path.is_dir():
        batch_convert_rips(str(input_path), output_dir, args.merge, args.min_vertices)
    elif input_path.suffix.lower() == '.rip':
        mesh = parse_rip_file(str(input_path))
        if mesh:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            mesh.to_obj(f"{output_dir}/{mesh.name}.obj")
    else:
        print(f"Invalid input: {input_path}")
