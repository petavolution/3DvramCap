#!/usr/bin/env python3
"""
Core glTF Export Module
=======================
Direct glTF 2.0 export without requiring Blender.

Features:
    - Binary glTF (GLB) export
    - KHR_materials_unlit extension support
    - Texture embedding
    - Multiple meshes and materials
    - Proper buffer/accessor setup

Dependencies:
    pip install pygltflib numpy

Usage:
    from core_gltf import GltfBuilder, export_meshes_to_gltf

    builder = GltfBuilder()
    builder.add_mesh("mesh1", vertices, faces, normals, uvs)
    builder.add_unlit_material("mat1", texture_path="diffuse.png")
    builder.save("output.glb")

Author: Capture Pipeline
"""

import base64
import json
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    raise ImportError("numpy required for glTF export")

try:
    from pygltflib import (
        GLTF2, Scene, Node, Mesh, Primitive, Attributes,
        Accessor, BufferView, Buffer, Material, Image, Texture,
        PbrMetallicRoughness, TextureInfo,
        FLOAT, UNSIGNED_SHORT, UNSIGNED_INT,
        ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER,
        SCALAR, VEC2, VEC3, VEC4
    )
    HAS_PYGLTFLIB = True
except ImportError:
    HAS_PYGLTFLIB = False


@dataclass
class MeshEntry:
    """A mesh to be exported."""
    name: str
    vertices: np.ndarray  # Nx3 float32
    faces: np.ndarray     # Mx3 uint32
    normals: Optional[np.ndarray] = None  # Nx3 float32
    uvs: Optional[np.ndarray] = None      # Nx2 float32
    material_index: int = -1


@dataclass
class MaterialEntry:
    """A material to be exported."""
    name: str
    base_color: List[float] = field(default_factory=lambda: [1, 1, 1, 1])
    metallic: float = 0.0
    roughness: float = 1.0
    emissive: List[float] = field(default_factory=lambda: [0, 0, 0])
    unlit: bool = True
    double_sided: bool = False
    alpha_mode: str = "OPAQUE"  # OPAQUE, MASK, BLEND
    alpha_cutoff: float = 0.5
    base_color_texture: Optional[str] = None
    normal_texture: Optional[str] = None
    emissive_texture: Optional[str] = None


class GltfBuilder:
    """
    Builder for creating glTF 2.0 files.

    Supports:
        - Multiple meshes
        - Multiple materials
        - Embedded textures
        - KHR_materials_unlit extension
    """

    def __init__(self):
        self.meshes: List[MeshEntry] = []
        self.materials: List[MaterialEntry] = []
        self.textures: Dict[str, int] = {}  # filepath -> texture index
        self._texture_data: List[bytes] = []

    def add_mesh(self, name: str,
                 vertices: np.ndarray,
                 faces: np.ndarray,
                 normals: np.ndarray = None,
                 uvs: np.ndarray = None,
                 material_index: int = -1) -> int:
        """
        Add a mesh to the glTF.

        Args:
            name: Mesh name
            vertices: Nx3 vertex positions
            faces: Mx3 triangle indices
            normals: Nx3 vertex normals (optional)
            uvs: Nx2 texture coordinates (optional)
            material_index: Index of material to use

        Returns:
            Mesh index
        """
        vertices = np.asarray(vertices, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.uint32)

        if normals is not None:
            normals = np.asarray(normals, dtype=np.float32)

        if uvs is not None:
            uvs = np.asarray(uvs, dtype=np.float32)

        mesh = MeshEntry(
            name=name,
            vertices=vertices,
            faces=faces,
            normals=normals,
            uvs=uvs,
            material_index=material_index
        )

        self.meshes.append(mesh)
        return len(self.meshes) - 1

    def add_unlit_material(self, name: str,
                           base_color: List[float] = None,
                           texture_path: str = None,
                           alpha_mode: str = "OPAQUE") -> int:
        """
        Add an unlit material (KHR_materials_unlit).

        Args:
            name: Material name
            base_color: RGBA color [r, g, b, a]
            texture_path: Path to base color texture
            alpha_mode: OPAQUE, MASK, or BLEND

        Returns:
            Material index
        """
        material = MaterialEntry(
            name=name,
            base_color=base_color or [1, 1, 1, 1],
            unlit=True,
            alpha_mode=alpha_mode,
            base_color_texture=texture_path
        )

        self.materials.append(material)
        return len(self.materials) - 1

    def add_pbr_material(self, name: str,
                         base_color: List[float] = None,
                         metallic: float = 0.0,
                         roughness: float = 1.0,
                         base_color_texture: str = None,
                         normal_texture: str = None) -> int:
        """
        Add a PBR material.

        Args:
            name: Material name
            base_color: RGBA color
            metallic: Metallic factor
            roughness: Roughness factor
            base_color_texture: Path to albedo texture
            normal_texture: Path to normal map

        Returns:
            Material index
        """
        material = MaterialEntry(
            name=name,
            base_color=base_color or [1, 1, 1, 1],
            metallic=metallic,
            roughness=roughness,
            unlit=False,
            base_color_texture=base_color_texture,
            normal_texture=normal_texture
        )

        self.materials.append(material)
        return len(self.materials) - 1

    def build(self) -> 'GLTF2':
        """
        Build the GLTF2 object.

        Returns:
            pygltflib.GLTF2 object ready for saving
        """
        if not HAS_PYGLTFLIB:
            raise RuntimeError("pygltflib required for glTF export")

        # Collect all binary data
        binary_blob = bytearray()
        buffer_views = []
        accessors = []
        gltf_meshes = []
        gltf_nodes = []
        gltf_materials = []
        gltf_images = []
        gltf_textures = []

        # Process textures first
        texture_indices = {}
        for mat in self.materials:
            for tex_path in [mat.base_color_texture, mat.normal_texture, mat.emissive_texture]:
                if tex_path and tex_path not in texture_indices:
                    tex_idx = self._add_texture(
                        tex_path, binary_blob, buffer_views,
                        gltf_images, gltf_textures
                    )
                    texture_indices[tex_path] = tex_idx

        # Process materials
        uses_unlit = any(m.unlit for m in self.materials)

        for mat in self.materials:
            gltf_mat = self._create_material(mat, texture_indices)
            gltf_materials.append(gltf_mat)

        # Process meshes
        for mesh_idx, mesh in enumerate(self.meshes):
            gltf_mesh, mesh_accessors, mesh_buffer_views = self._create_mesh(
                mesh, binary_blob, len(accessors), len(buffer_views)
            )
            accessors.extend(mesh_accessors)
            buffer_views.extend(mesh_buffer_views)
            gltf_meshes.append(gltf_mesh)

            # Create node for mesh
            node = Node(mesh=mesh_idx, name=mesh.name)
            gltf_nodes.append(node)

        # Create scene
        scene = Scene(nodes=list(range(len(gltf_nodes))))

        # Create buffer
        buffer = Buffer(byteLength=len(binary_blob))

        # Build GLTF
        gltf = GLTF2(
            scene=0,
            scenes=[scene],
            nodes=gltf_nodes,
            meshes=gltf_meshes,
            accessors=accessors,
            bufferViews=buffer_views,
            buffers=[buffer],
            materials=gltf_materials if gltf_materials else None,
            images=gltf_images if gltf_images else None,
            textures=gltf_textures if gltf_textures else None
        )

        # Add extensions if using unlit
        if uses_unlit:
            gltf.extensionsUsed = ["KHR_materials_unlit"]

        # Set binary blob
        gltf.set_binary_blob(bytes(binary_blob))

        return gltf

    def _create_mesh(self, mesh: MeshEntry, binary_blob: bytearray,
                     accessor_offset: int, buffer_view_offset: int
                     ) -> Tuple['Mesh', List['Accessor'], List['BufferView']]:
        """Create glTF mesh with accessors and buffer views."""
        accessors = []
        buffer_views = []

        # Align to 4 bytes
        def align4(blob):
            while len(blob) % 4 != 0:
                blob.append(0)

        # Indices
        align4(binary_blob)
        indices_offset = len(binary_blob)
        indices_data = mesh.faces.flatten().astype(np.uint32).tobytes()
        binary_blob.extend(indices_data)
        indices_count = mesh.faces.size

        buffer_views.append(BufferView(
            buffer=0,
            byteOffset=indices_offset,
            byteLength=len(indices_data),
            target=ELEMENT_ARRAY_BUFFER
        ))

        accessors.append(Accessor(
            bufferView=buffer_view_offset,
            byteOffset=0,
            componentType=UNSIGNED_INT,
            count=indices_count,
            type=SCALAR,
            max=[int(mesh.faces.max())],
            min=[int(mesh.faces.min())]
        ))
        indices_accessor = accessor_offset

        # Positions
        align4(binary_blob)
        positions_offset = len(binary_blob)
        positions_data = mesh.vertices.astype(np.float32).tobytes()
        binary_blob.extend(positions_data)

        buffer_views.append(BufferView(
            buffer=0,
            byteOffset=positions_offset,
            byteLength=len(positions_data),
            target=ARRAY_BUFFER
        ))

        accessors.append(Accessor(
            bufferView=buffer_view_offset + 1,
            byteOffset=0,
            componentType=FLOAT,
            count=len(mesh.vertices),
            type=VEC3,
            max=mesh.vertices.max(axis=0).tolist(),
            min=mesh.vertices.min(axis=0).tolist()
        ))
        position_accessor = accessor_offset + 1

        bv_count = 2
        ac_count = 2

        # Normals
        normal_accessor = None
        if mesh.normals is not None:
            align4(binary_blob)
            normals_offset = len(binary_blob)
            normals_data = mesh.normals.astype(np.float32).tobytes()
            binary_blob.extend(normals_data)

            buffer_views.append(BufferView(
                buffer=0,
                byteOffset=normals_offset,
                byteLength=len(normals_data),
                target=ARRAY_BUFFER
            ))

            accessors.append(Accessor(
                bufferView=buffer_view_offset + bv_count,
                byteOffset=0,
                componentType=FLOAT,
                count=len(mesh.normals),
                type=VEC3,
                max=mesh.normals.max(axis=0).tolist(),
                min=mesh.normals.min(axis=0).tolist()
            ))
            normal_accessor = accessor_offset + ac_count
            bv_count += 1
            ac_count += 1

        # UVs
        texcoord_accessor = None
        if mesh.uvs is not None:
            align4(binary_blob)
            uvs_offset = len(binary_blob)
            uvs_data = mesh.uvs.astype(np.float32).tobytes()
            binary_blob.extend(uvs_data)

            buffer_views.append(BufferView(
                buffer=0,
                byteOffset=uvs_offset,
                byteLength=len(uvs_data),
                target=ARRAY_BUFFER
            ))

            accessors.append(Accessor(
                bufferView=buffer_view_offset + bv_count,
                byteOffset=0,
                componentType=FLOAT,
                count=len(mesh.uvs),
                type=VEC2,
                max=mesh.uvs.max(axis=0).tolist(),
                min=mesh.uvs.min(axis=0).tolist()
            ))
            texcoord_accessor = accessor_offset + ac_count

        # Create attributes
        attrs = Attributes(POSITION=position_accessor)
        if normal_accessor is not None:
            attrs.NORMAL = normal_accessor
        if texcoord_accessor is not None:
            attrs.TEXCOORD_0 = texcoord_accessor

        # Create primitive
        primitive = Primitive(
            attributes=attrs,
            indices=indices_accessor,
            material=mesh.material_index if mesh.material_index >= 0 else None
        )

        gltf_mesh = Mesh(
            name=mesh.name,
            primitives=[primitive]
        )

        return gltf_mesh, accessors, buffer_views

    def _create_material(self, mat: MaterialEntry,
                         texture_indices: Dict[str, int]) -> 'Material':
        """Create glTF material."""
        pbr = PbrMetallicRoughness(
            baseColorFactor=mat.base_color,
            metallicFactor=mat.metallic,
            roughnessFactor=mat.roughness
        )

        # Add base color texture
        if mat.base_color_texture and mat.base_color_texture in texture_indices:
            pbr.baseColorTexture = TextureInfo(
                index=texture_indices[mat.base_color_texture]
            )

        gltf_mat = Material(
            name=mat.name,
            pbrMetallicRoughness=pbr,
            doubleSided=mat.double_sided,
            alphaMode=mat.alpha_mode,
            alphaCutoff=mat.alpha_cutoff if mat.alpha_mode == "MASK" else None
        )

        # Add unlit extension
        if mat.unlit:
            gltf_mat.extensions = {"KHR_materials_unlit": {}}

        # Add normal texture
        if mat.normal_texture and mat.normal_texture in texture_indices:
            gltf_mat.normalTexture = TextureInfo(
                index=texture_indices[mat.normal_texture]
            )

        # Add emissive
        if mat.emissive_texture and mat.emissive_texture in texture_indices:
            gltf_mat.emissiveTexture = TextureInfo(
                index=texture_indices[mat.emissive_texture]
            )
            gltf_mat.emissiveFactor = mat.emissive

        return gltf_mat

    def _add_texture(self, filepath: str, binary_blob: bytearray,
                     buffer_views: List, images: List, textures: List) -> int:
        """Add texture to glTF."""
        if not os.path.exists(filepath):
            return -1

        # Read image data
        with open(filepath, 'rb') as f:
            image_data = f.read()

        # Determine MIME type
        ext = Path(filepath).suffix.lower()
        mime_map = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg'
        }
        mime_type = mime_map.get(ext, 'image/png')

        # Align and add to buffer
        while len(binary_blob) % 4 != 0:
            binary_blob.append(0)

        image_offset = len(binary_blob)
        binary_blob.extend(image_data)

        # Create buffer view for image
        bv_index = len(buffer_views)
        buffer_views.append(BufferView(
            buffer=0,
            byteOffset=image_offset,
            byteLength=len(image_data)
        ))

        # Create image
        img_index = len(images)
        images.append(Image(
            bufferView=bv_index,
            mimeType=mime_type
        ))

        # Create texture
        tex_index = len(textures)
        textures.append(Texture(
            source=img_index
        ))

        return tex_index

    def save(self, filepath: str):
        """
        Save glTF to file.

        Args:
            filepath: Output path (.glb or .gltf)
        """
        gltf = self.build()

        ext = Path(filepath).suffix.lower()
        if ext == '.glb':
            gltf.save_binary(filepath)
        else:
            gltf.save(filepath)

        print(f"Saved: {filepath}")
        print(f"  Meshes: {len(self.meshes)}")
        print(f"  Materials: {len(self.materials)}")


def export_meshes_to_gltf(meshes: List[Tuple[str, np.ndarray, np.ndarray]],
                          output_path: str,
                          texture_path: str = None,
                          use_unlit: bool = True) -> bool:
    """
    Convenience function to export meshes to glTF.

    Args:
        meshes: List of (name, vertices, faces) tuples
        output_path: Output .glb or .gltf path
        texture_path: Optional texture to apply
        use_unlit: Use unlit material

    Returns:
        True if successful
    """
    builder = GltfBuilder()

    # Add material
    if use_unlit:
        mat_idx = builder.add_unlit_material(
            "material",
            texture_path=texture_path
        )
    else:
        mat_idx = builder.add_pbr_material(
            "material",
            base_color_texture=texture_path
        )

    # Add meshes
    for name, vertices, faces in meshes:
        builder.add_mesh(
            name=name,
            vertices=vertices,
            faces=faces,
            material_index=mat_idx if texture_path else -1
        )

    try:
        builder.save(output_path)
        return True
    except Exception as e:
        print(f"Export failed: {e}")
        return False


def convert_obj_to_glb(obj_path: str, glb_path: str,
                       texture_path: str = None) -> bool:
    """
    Convert OBJ file to GLB.

    Args:
        obj_path: Input OBJ file
        glb_path: Output GLB file
        texture_path: Optional texture

    Returns:
        True if successful
    """
    # Simple OBJ parser
    vertices = []
    faces = []
    normals = []
    uvs = []

    with open(obj_path, 'r') as f:
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
                for v in parts[1:4]:
                    idx = int(v.split('/')[0]) - 1
                    face.append(idx)
                faces.append(face)

    vertices = np.array(vertices, dtype=np.float32)
    faces = np.array(faces, dtype=np.uint32)
    normals = np.array(normals, dtype=np.float32) if normals else None
    uvs = np.array(uvs, dtype=np.float32) if uvs else None

    builder = GltfBuilder()

    mat_idx = -1
    if texture_path:
        mat_idx = builder.add_unlit_material("material", texture_path=texture_path)

    builder.add_mesh(
        name=Path(obj_path).stem,
        vertices=vertices,
        faces=faces,
        normals=normals,
        uvs=uvs,
        material_index=mat_idx
    )

    builder.save(glb_path)
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export meshes to glTF")
    parser.add_argument('input', help="Input OBJ file")
    parser.add_argument('--output', '-o', help="Output GLB file")
    parser.add_argument('--texture', '-t', help="Texture file")

    args = parser.parse_args()

    output = args.output or args.input.replace('.obj', '.glb')
    convert_obj_to_glb(args.input, output, args.texture)
