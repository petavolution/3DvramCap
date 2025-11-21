#!/usr/bin/env python3
"""
Direct USD Export
=================
Create USD/USDA/USDC files directly without Blender.

Features:
    - Generate USDA (ASCII) files without OpenUSD dependency
    - Optional USDC (binary) via OpenUSD/pxr
    - Scene hierarchy support
    - Material definitions (UsdPreviewSurface)
    - Texture references
    - Instancing support

Usage:
    from core_usd import UsdBuilder, save_meshes_to_usd

    # Simple API
    save_meshes_to_usd(meshes, "output.usda")

    # Builder API
    builder = UsdBuilder()
    builder.add_mesh("mesh1", vertices, faces, normals, uvs)
    builder.add_material("mat1", diffuse_texture="diffuse.png")
    builder.save("output.usda")

Author: Capture Pipeline
"""

import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import json

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# Try to import OpenUSD for binary export
try:
    from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf, Vt
    HAS_PXR = True
except ImportError:
    HAS_PXR = False


@dataclass
class UsdMeshData:
    """Mesh data for USD export."""
    name: str
    vertices: List[Tuple[float, float, float]]
    faces: List[List[int]]
    normals: Optional[List[Tuple[float, float, float]]] = None
    uvs: Optional[List[Tuple[float, float]]] = None
    material_name: Optional[str] = None
    transform: Optional[List[float]] = None  # 4x4 matrix


@dataclass
class UsdMaterialData:
    """Material data for USD export."""
    name: str
    diffuse_color: Tuple[float, float, float] = (0.8, 0.8, 0.8)
    diffuse_texture: Optional[str] = None
    roughness: float = 0.5
    metallic: float = 0.0
    normal_texture: Optional[str] = None
    opacity: float = 1.0
    emissive_color: Tuple[float, float, float] = (0, 0, 0)
    unlit: bool = True  # Use unlit shading


class UsdaWriter:
    """
    Write USD ASCII (USDA) files without OpenUSD dependency.

    This generates valid USDA files that can be opened in any USD-compatible
    software, including Blender 3.0+, Omniverse, and Houdini.
    """

    def __init__(self):
        self.lines: List[str] = []
        self.indent_level = 0

    def _indent(self) -> str:
        return "    " * self.indent_level

    def _write(self, line: str):
        self.lines.append(f"{self._indent()}{line}")

    def _begin_block(self, header: str):
        self._write(header)
        self._write("{")
        self.indent_level += 1

    def _end_block(self):
        self.indent_level -= 1
        self._write("}")

    def _format_array(self, values: List, type_hint: str = "float") -> str:
        """Format an array for USDA."""
        if not values:
            return "[]"

        if isinstance(values[0], (tuple, list)):
            # Array of tuples (points, vectors, etc.)
            formatted = ", ".join(
                f"({', '.join(f'{v:.6f}' for v in item)})"
                for item in values
            )
        else:
            # Flat array
            if type_hint == "int":
                formatted = ", ".join(str(int(v)) for v in values)
            else:
                formatted = ", ".join(f"{v:.6f}" for v in values)

        return f"[{formatted}]"

    def _format_face_counts_indices(self, faces: List[List[int]]) -> Tuple[str, str]:
        """Format face vertex counts and indices."""
        counts = [len(f) for f in faces]
        indices = [idx for face in faces for idx in face]

        counts_str = self._format_array(counts, "int")
        indices_str = self._format_array(indices, "int")

        return counts_str, indices_str

    def write_header(self, up_axis: str = "Y", meters_per_unit: float = 1.0):
        """Write USDA file header."""
        self.lines = [
            '#usda 1.0',
            '(',
            f'    upAxis = "{up_axis}"',
            f'    metersPerUnit = {meters_per_unit}',
            '    defaultPrim = "World"',
            ')',
            ''
        ]

    def write_xform(self, name: str, transform: Optional[List[float]] = None):
        """Write an Xform (transform) prim."""
        self._begin_block(f'def Xform "{name}"')

        if transform:
            # Format 4x4 matrix
            matrix_str = "( " + ", ".join(
                f"({transform[i*4]:.6f}, {transform[i*4+1]:.6f}, {transform[i*4+2]:.6f}, {transform[i*4+3]:.6f})"
                for i in range(4)
            ) + " )"
            self._write(f'matrix4d xformOp:transform = {matrix_str}')
            self._write('uniform token[] xformOpOrder = ["xformOp:transform"]')

    def write_mesh(self, mesh: UsdMeshData):
        """Write a mesh prim."""
        self._begin_block(f'def Mesh "{mesh.name}"')

        # Points
        points_str = self._format_array(mesh.vertices)
        self._write(f"point3f[] points = {points_str}")

        # Face topology
        counts_str, indices_str = self._format_face_counts_indices(mesh.faces)
        self._write(f"int[] faceVertexCounts = {counts_str}")
        self._write(f"int[] faceVertexIndices = {indices_str}")

        # Normals
        if mesh.normals:
            normals_str = self._format_array(mesh.normals)
            self._write(f"normal3f[] normals = {normals_str}")
            self._write('uniform token normals:interpolation = "vertex"')

        # UVs (primvars:st)
        if mesh.uvs:
            uvs_str = self._format_array(mesh.uvs)
            self._write(f"texCoord2f[] primvars:st = {uvs_str}")
            self._write('uniform token primvars:st:interpolation = "vertex"')

        # Material binding
        if mesh.material_name:
            self._write(f'rel material:binding = </World/Materials/{mesh.material_name}>')

        # Extent (bounding box)
        if mesh.vertices:
            min_pt = [min(v[i] for v in mesh.vertices) for i in range(3)]
            max_pt = [max(v[i] for v in mesh.vertices) for i in range(3)]
            extent_str = f"[({min_pt[0]:.6f}, {min_pt[1]:.6f}, {min_pt[2]:.6f}), ({max_pt[0]:.6f}, {max_pt[1]:.6f}, {max_pt[2]:.6f})]"
            self._write(f"float3[] extent = {extent_str}")

        self._end_block()

    def write_material(self, material: UsdMaterialData):
        """Write a UsdPreviewSurface material."""
        self._begin_block(f'def Material "{material.name}"')

        # Surface output
        self._write('token outputs:surface.connect = </World/Materials/' +
                   f'{material.name}/PreviewSurface.outputs:surface>')

        # PreviewSurface shader
        self._begin_block('def Shader "PreviewSurface"')
        self._write('uniform token info:id = "UsdPreviewSurface"')

        # Diffuse color or texture
        if material.diffuse_texture:
            self._write(f'color3f inputs:diffuseColor.connect = </World/Materials/{material.name}/DiffuseTexture.outputs:rgb>')
        else:
            dc = material.diffuse_color
            self._write(f'color3f inputs:diffuseColor = ({dc[0]:.4f}, {dc[1]:.4f}, {dc[2]:.4f})')

        # PBR properties
        if not material.unlit:
            self._write(f'float inputs:roughness = {material.roughness:.4f}')
            self._write(f'float inputs:metallic = {material.metallic:.4f}')

            if material.normal_texture:
                self._write(f'normal3f inputs:normal.connect = </World/Materials/{material.name}/NormalTexture.outputs:rgb>')

        # Emissive
        if material.emissive_color != (0, 0, 0):
            ec = material.emissive_color
            self._write(f'color3f inputs:emissiveColor = ({ec[0]:.4f}, {ec[1]:.4f}, {ec[2]:.4f})')

        # Opacity
        if material.opacity < 1.0:
            self._write(f'float inputs:opacity = {material.opacity:.4f}')

        self._write('token outputs:surface')
        self._end_block()

        # Diffuse texture reader
        if material.diffuse_texture:
            self._begin_block('def Shader "DiffuseTexture"')
            self._write('uniform token info:id = "UsdUVTexture"')
            self._write(f'asset inputs:file = @{material.diffuse_texture}@')
            self._write('float2 inputs:st.connect = </World/Materials/' +
                       f'{material.name}/TexCoordReader.outputs:result>')
            self._write('float3 outputs:rgb')
            self._end_block()

            # TexCoord reader
            self._begin_block('def Shader "TexCoordReader"')
            self._write('uniform token info:id = "UsdPrimvarReader_float2"')
            self._write('string inputs:varname = "st"')
            self._write('float2 outputs:result')
            self._end_block()

        # Normal texture reader
        if material.normal_texture and not material.unlit:
            self._begin_block('def Shader "NormalTexture"')
            self._write('uniform token info:id = "UsdUVTexture"')
            self._write(f'asset inputs:file = @{material.normal_texture}@')
            self._write('float2 inputs:st.connect = </World/Materials/' +
                       f'{material.name}/TexCoordReader.outputs:result>')
            self._write('float3 outputs:rgb')
            self._end_block()

        self._end_block()

    def write_instance(self, name: str, prototype_path: str,
                      position: Tuple[float, float, float] = (0, 0, 0),
                      rotation: Tuple[float, float, float] = (0, 0, 0),
                      scale: Tuple[float, float, float] = (1, 1, 1)):
        """Write an instance reference."""
        self._begin_block(f'def Xform "{name}" (')
        self._write(f'    references = <{prototype_path}>')
        self.indent_level -= 1
        self._write(')')
        self.indent_level += 1

        # Transform
        self._write(f'double3 xformOp:translate = ({position[0]}, {position[1]}, {position[2]})')
        self._write(f'float3 xformOp:rotateXYZ = ({rotation[0]}, {rotation[1]}, {rotation[2]})')
        self._write(f'float3 xformOp:scale = ({scale[0]}, {scale[1]}, {scale[2]})')
        self._write('uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]')

        self._end_block()

    def end_xform(self):
        """End current Xform block."""
        self._end_block()

    def get_content(self) -> str:
        """Get the complete USDA content."""
        return "\n".join(self.lines)

    def save(self, filepath: str):
        """Save to file."""
        with open(filepath, 'w') as f:
            f.write(self.get_content())


class UsdBuilder:
    """
    High-level USD scene builder.

    Supports both USDA (ASCII, no dependencies) and USDC (binary, requires pxr).
    """

    def __init__(self, up_axis: str = "Y", scale: float = 1.0):
        """
        Initialize USD builder.

        Args:
            up_axis: "Y" (glTF/USD standard) or "Z" (Blender)
            scale: Meters per unit (1.0 = meters, 0.01 = centimeters)
        """
        self.up_axis = up_axis
        self.scale = scale
        self.meshes: List[UsdMeshData] = []
        self.materials: Dict[str, UsdMaterialData] = {}
        self.instances: List[Dict] = []
        self.default_material_counter = 0

    def add_mesh(self,
                 name: str,
                 vertices: Union[List, 'np.ndarray'],
                 faces: Union[List, 'np.ndarray'],
                 normals: Optional[Union[List, 'np.ndarray']] = None,
                 uvs: Optional[Union[List, 'np.ndarray']] = None,
                 material_name: Optional[str] = None,
                 transform: Optional[List[float]] = None) -> str:
        """
        Add a mesh to the scene.

        Args:
            name: Mesh name (will be sanitized)
            vertices: Nx3 vertex positions
            faces: Mx3 triangle indices
            normals: Optional Nx3 vertex normals
            uvs: Optional Nx2 texture coordinates
            material_name: Optional material to bind
            transform: Optional 4x4 transform matrix

        Returns:
            Sanitized mesh name
        """
        # Sanitize name
        safe_name = self._sanitize_name(name)

        # Convert numpy arrays
        if HAS_NUMPY:
            if isinstance(vertices, np.ndarray):
                vertices = vertices.tolist()
            if isinstance(faces, np.ndarray):
                faces = faces.tolist()
            if normals is not None and isinstance(normals, np.ndarray):
                normals = normals.tolist()
            if uvs is not None and isinstance(uvs, np.ndarray):
                uvs = uvs.tolist()

        # Ensure vertices are tuples
        vertices = [tuple(v) for v in vertices]
        faces = [list(f) for f in faces]

        if normals:
            normals = [tuple(n) for n in normals]
        if uvs:
            uvs = [tuple(uv) for uv in uvs]

        mesh = UsdMeshData(
            name=safe_name,
            vertices=vertices,
            faces=faces,
            normals=normals,
            uvs=uvs,
            material_name=material_name,
            transform=transform
        )

        self.meshes.append(mesh)
        return safe_name

    def add_material(self,
                     name: str,
                     diffuse_color: Tuple[float, float, float] = (0.8, 0.8, 0.8),
                     diffuse_texture: Optional[str] = None,
                     roughness: float = 0.5,
                     metallic: float = 0.0,
                     normal_texture: Optional[str] = None,
                     unlit: bool = True) -> str:
        """
        Add a material to the scene.

        Args:
            name: Material name
            diffuse_color: RGB diffuse color (0-1)
            diffuse_texture: Path to diffuse texture
            roughness: Roughness value (0-1)
            metallic: Metallic value (0-1)
            normal_texture: Path to normal map
            unlit: Use unlit shading (no lighting)

        Returns:
            Sanitized material name
        """
        safe_name = self._sanitize_name(name)

        self.materials[safe_name] = UsdMaterialData(
            name=safe_name,
            diffuse_color=diffuse_color,
            diffuse_texture=diffuse_texture,
            roughness=roughness,
            metallic=metallic,
            normal_texture=normal_texture,
            unlit=unlit
        )

        return safe_name

    def add_unlit_material(self,
                          name: str,
                          texture_path: Optional[str] = None,
                          color: Tuple[float, float, float] = (0.8, 0.8, 0.8)) -> str:
        """Add a simple unlit material."""
        return self.add_material(
            name=name,
            diffuse_color=color,
            diffuse_texture=texture_path,
            unlit=True
        )

    def add_default_material(self, texture_path: Optional[str] = None) -> str:
        """Add a default material and return its name."""
        self.default_material_counter += 1
        name = f"Material_{self.default_material_counter:03d}"
        return self.add_unlit_material(name, texture_path)

    def add_instance(self,
                     name: str,
                     prototype_mesh: str,
                     position: Tuple[float, float, float] = (0, 0, 0),
                     rotation: Tuple[float, float, float] = (0, 0, 0),
                     scale: Tuple[float, float, float] = (1, 1, 1)):
        """Add an instance of an existing mesh."""
        self.instances.append({
            'name': self._sanitize_name(name),
            'prototype': prototype_mesh,
            'position': position,
            'rotation': rotation,
            'scale': scale
        })

    def _sanitize_name(self, name: str) -> str:
        """Sanitize name for USD (alphanumeric and underscore only)."""
        import re
        # Replace invalid characters with underscore
        safe = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        # Ensure doesn't start with number
        if safe and safe[0].isdigit():
            safe = '_' + safe
        return safe or '_unnamed'

    def save(self, filepath: str, binary: bool = False):
        """
        Save the USD file.

        Args:
            filepath: Output path (.usda, .usdc, or .usd)
            binary: Force binary format (requires pxr)
        """
        path = Path(filepath)

        # Determine format
        if path.suffix.lower() == '.usdc' or binary:
            if HAS_PXR:
                self._save_usdc(filepath)
            else:
                # Fall back to USDA
                usda_path = path.with_suffix('.usda')
                print(f"Warning: pxr not available, saving as USDA: {usda_path}")
                self._save_usda(str(usda_path))
        else:
            self._save_usda(filepath)

    def _save_usda(self, filepath: str):
        """Save as USDA (ASCII) format."""
        writer = UsdaWriter()
        writer.write_header(up_axis=self.up_axis, meters_per_unit=self.scale)

        # World root
        writer.write_xform("World")

        # Materials scope
        if self.materials:
            writer._begin_block('def Scope "Materials"')
            for material in self.materials.values():
                writer.write_material(material)
            writer._end_block()

        # Meshes scope
        if self.meshes:
            writer._begin_block('def Scope "Geometry"')
            for mesh in self.meshes:
                writer.write_mesh(mesh)
            writer._end_block()

        # Instances scope
        if self.instances:
            writer._begin_block('def Scope "Instances"')
            for inst in self.instances:
                writer.write_instance(
                    inst['name'],
                    f"/World/Geometry/{inst['prototype']}",
                    inst['position'],
                    inst['rotation'],
                    inst['scale']
                )
            writer._end_block()

        writer.end_xform()  # Close World

        writer.save(filepath)
        print(f"Saved USDA: {filepath}")

    def _save_usdc(self, filepath: str):
        """Save as USDC (binary) format using OpenUSD."""
        if not HAS_PXR:
            raise ImportError("pxr (OpenUSD) required for USDC export")

        stage = Usd.Stage.CreateNew(filepath)
        stage.SetMetadata('upAxis', self.up_axis)
        stage.SetMetadata('metersPerUnit', self.scale)

        # Create World root
        world = UsdGeom.Xform.Define(stage, '/World')
        stage.SetDefaultPrim(world.GetPrim())

        # Materials
        if self.materials:
            materials_scope = UsdGeom.Scope.Define(stage, '/World/Materials')

            for mat_data in self.materials.values():
                mat_path = f'/World/Materials/{mat_data.name}'
                material = UsdShade.Material.Define(stage, mat_path)

                # Create PreviewSurface shader
                shader = UsdShade.Shader.Define(stage, f'{mat_path}/PreviewSurface')
                shader.CreateIdAttr('UsdPreviewSurface')

                # Diffuse
                if mat_data.diffuse_texture:
                    # Create texture reader
                    tex_reader = UsdShade.Shader.Define(stage, f'{mat_path}/DiffuseTexture')
                    tex_reader.CreateIdAttr('UsdUVTexture')
                    tex_reader.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(mat_data.diffuse_texture)

                    st_reader = UsdShade.Shader.Define(stage, f'{mat_path}/TexCoordReader')
                    st_reader.CreateIdAttr('UsdPrimvarReader_float2')
                    st_reader.CreateInput('varname', Sdf.ValueTypeNames.String).Set('st')

                    tex_reader.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(
                        st_reader.ConnectableAPI(), 'result')

                    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(
                        tex_reader.ConnectableAPI(), 'rgb')
                else:
                    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(
                        Gf.Vec3f(*mat_data.diffuse_color))

                # Surface output
                shader.CreateOutput('surface', Sdf.ValueTypeNames.Token)
                material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')

        # Meshes
        if self.meshes:
            geo_scope = UsdGeom.Scope.Define(stage, '/World/Geometry')

            for mesh_data in self.meshes:
                mesh_path = f'/World/Geometry/{mesh_data.name}'
                mesh = UsdGeom.Mesh.Define(stage, mesh_path)

                # Points
                mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*v) for v in mesh_data.vertices]))

                # Faces
                face_counts = [len(f) for f in mesh_data.faces]
                face_indices = [idx for face in mesh_data.faces for idx in face]
                mesh.CreateFaceVertexCountsAttr(Vt.IntArray(face_counts))
                mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(face_indices))

                # Normals
                if mesh_data.normals:
                    mesh.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*n) for n in mesh_data.normals]))
                    mesh.SetNormalsInterpolation('vertex')

                # UVs
                if mesh_data.uvs:
                    st = mesh.CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, 'vertex')
                    st.Set(Vt.Vec2fArray([Gf.Vec2f(*uv) for uv in mesh_data.uvs]))

                # Material binding
                if mesh_data.material_name and mesh_data.material_name in self.materials:
                    mat_path = f'/World/Materials/{mesh_data.material_name}'
                    UsdShade.MaterialBindingAPI(mesh).Bind(
                        UsdShade.Material.Get(stage, mat_path))

                # Extent
                mesh.CreateExtentAttr(mesh.ComputeExtent(mesh.GetPointsAttr().Get()))

        stage.Save()
        print(f"Saved USDC: {filepath}")

    def get_stats(self) -> Dict:
        """Get scene statistics."""
        total_verts = sum(len(m.vertices) for m in self.meshes)
        total_faces = sum(len(m.faces) for m in self.meshes)

        return {
            'meshes': len(self.meshes),
            'materials': len(self.materials),
            'instances': len(self.instances),
            'total_vertices': total_verts,
            'total_faces': total_faces
        }


# Convenience functions
def save_meshes_to_usd(meshes: List[Dict],
                       output_path: str,
                       texture_path: Optional[str] = None,
                       up_axis: str = "Y",
                       scale: float = 1.0) -> bool:
    """
    Quick function to save meshes to USD.

    Args:
        meshes: List of mesh dicts with 'vertices', 'faces', 'name'
        output_path: Output .usda or .usdc path
        texture_path: Optional shared texture
        up_axis: "Y" or "Z"
        scale: Meters per unit

    Returns:
        True if successful
    """
    try:
        builder = UsdBuilder(up_axis=up_axis, scale=scale)

        # Add shared material if texture provided
        mat_name = None
        if texture_path:
            mat_name = builder.add_unlit_material("SharedMaterial", texture_path)

        # Add meshes
        for mesh in meshes:
            builder.add_mesh(
                name=mesh.get('name', 'mesh'),
                vertices=mesh['vertices'],
                faces=mesh['faces'],
                normals=mesh.get('normals'),
                uvs=mesh.get('uvs'),
                material_name=mat_name
            )

        builder.save(output_path)
        return True

    except Exception as e:
        print(f"USD export failed: {e}")
        return False


def obj_to_usd(obj_files: List[str],
               output_path: str,
               texture_dir: Optional[str] = None) -> bool:
    """
    Convert multiple OBJ files to USD.

    Args:
        obj_files: List of OBJ file paths
        output_path: Output USD path
        texture_dir: Optional directory containing textures

    Returns:
        True if successful
    """
    from core_merge import load_obj  # Import our OBJ loader

    builder = UsdBuilder()

    for obj_path in obj_files:
        mesh = load_obj(obj_path)
        if mesh is None:
            continue

        # Look for texture
        texture_path = None
        if texture_dir:
            obj_name = Path(obj_path).stem
            for ext in ['.png', '.jpg', '.jpeg']:
                tex_file = Path(texture_dir) / f"{obj_name}{ext}"
                if tex_file.exists():
                    texture_path = str(tex_file)
                    break

        # Add material if texture found
        mat_name = None
        if texture_path:
            mat_name = builder.add_unlit_material(f"Mat_{mesh.name}", texture_path)

        # Add mesh
        builder.add_mesh(
            name=mesh.name,
            vertices=mesh.vertices.tolist(),
            faces=mesh.faces.tolist(),
            normals=mesh.normals.tolist() if mesh.normals is not None else None,
            uvs=mesh.uvs.tolist() if mesh.uvs is not None else None,
            material_name=mat_name
        )

    if builder.meshes:
        builder.save(output_path)
        return True

    return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert meshes to USD")
    parser.add_argument('inputs', nargs='+', help="Input OBJ files")
    parser.add_argument('--output', '-o', required=True, help="Output USD file")
    parser.add_argument('--textures', '-t', help="Texture directory")
    parser.add_argument('--binary', '-b', action='store_true', help="Output USDC binary")
    parser.add_argument('--up-axis', choices=['Y', 'Z'], default='Y', help="Up axis")

    args = parser.parse_args()

    # Determine output format
    output = args.output
    if args.binary and not output.endswith('.usdc'):
        output = output.rsplit('.', 1)[0] + '.usdc'

    success = obj_to_usd(args.inputs, output, args.textures)

    if success:
        print(f"Created: {output}")
    else:
        print("Failed to create USD")
        exit(1)
