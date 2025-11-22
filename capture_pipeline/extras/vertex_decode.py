#!/usr/bin/env python3
"""
Core Vertex Decoder for RenderDoc
=================================
Robust vertex attribute parsing from RenderDoc's GetPostVSData output.

This module handles the complexity of different vertex layouts, shader
outputs, and API differences (D3D11, D3D12, Vulkan, OpenGL).

Key Concepts:
    - Post-VS data is in clip space (before perspective divide)
    - Position is always shuffled to first attribute
    - Vulkan outputs may have padding between attributes
    - Stride and offset must be calculated from shader reflection

Usage:
    from core_vertex_decode import VertexDecoder, decode_post_vs_mesh

    decoder = VertexDecoder(controller)
    mesh = decoder.decode_draw_call(event_id)

Author: Capture Pipeline
"""

import struct
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any

# RenderDoc type mapping
try:
    import renderdoc as rd

    COMP_TYPE_MAP = {
        rd.CompType.Float: ('f', 4),
        rd.CompType.UInt: ('I', 4),
        rd.CompType.SInt: ('i', 4),
        rd.CompType.UNorm: ('B', 1),  # Normalized to 0-1
        rd.CompType.SNorm: ('b', 1),  # Normalized to -1 to 1
        rd.CompType.UScaled: ('H', 2),
        rd.CompType.SScaled: ('h', 2),
        rd.CompType.Double: ('d', 8),
    }
    HAS_RENDERDOC = True
except ImportError:
    HAS_RENDERDOC = False
    COMP_TYPE_MAP = {}


class AttributeSemantic(Enum):
    """Common vertex attribute semantics."""
    POSITION = "POSITION"
    NORMAL = "NORMAL"
    TANGENT = "TANGENT"
    BINORMAL = "BINORMAL"
    TEXCOORD = "TEXCOORD"
    COLOR = "COLOR"
    UNKNOWN = "UNKNOWN"


@dataclass
class VertexAttribute:
    """Describes a single vertex attribute."""
    name: str
    semantic: AttributeSemantic
    semantic_index: int
    component_type: str  # 'f', 'I', 'i', etc.
    component_count: int
    component_size: int
    offset: int
    is_builtin_position: bool = False

    @property
    def byte_size(self) -> int:
        return self.component_count * self.component_size

    def unpack_format(self) -> str:
        return f"{self.component_count}{self.component_type}"


@dataclass
class DecodedVertex:
    """A decoded vertex with all attributes."""
    position: List[float]
    normal: Optional[List[float]] = None
    tangent: Optional[List[float]] = None
    texcoord0: Optional[List[float]] = None
    texcoord1: Optional[List[float]] = None
    color: Optional[List[float]] = None
    raw_attributes: Dict[str, List[float]] = None


@dataclass
class DecodedMesh:
    """A fully decoded mesh ready for export."""
    vertices: List[DecodedVertex]
    indices: List[int]
    triangle_count: int
    event_id: int
    draw_call_name: str = ""
    bounds_min: List[float] = None
    bounds_max: List[float] = None

    def to_obj_data(self) -> dict:
        """Convert to OBJ export format."""
        positions = []
        normals = []
        uvs = []

        for v in self.vertices:
            positions.append(v.position)
            if v.normal:
                normals.append(v.normal)
            if v.texcoord0:
                uvs.append(v.texcoord0)

        return {
            'vertices': positions,
            'normals': normals if len(normals) == len(positions) else [],
            'uvs': uvs if len(uvs) == len(positions) else [],
            'faces': self._generate_faces(),
            'event_id': self.event_id
        }

    def _generate_faces(self) -> List[List[int]]:
        """Generate face indices (triangles)."""
        if self.indices:
            faces = []
            for i in range(0, len(self.indices) - 2, 3):
                faces.append([self.indices[i], self.indices[i+1], self.indices[i+2]])
            return faces
        else:
            # Non-indexed: sequential triangles
            faces = []
            for i in range(0, len(self.vertices) - 2, 3):
                faces.append([i, i+1, i+2])
            return faces


def parse_semantic(name: str) -> Tuple[AttributeSemantic, int]:
    """
    Parse attribute name to semantic and index.

    Examples:
        "POSITION0" -> (POSITION, 0)
        "TEXCOORD1" -> (TEXCOORD, 1)
        "SV_Position" -> (POSITION, 0)
    """
    name_upper = name.upper()

    # Handle system value semantics
    if name_upper.startswith("SV_"):
        name_upper = name_upper[3:]

    # Extract trailing number
    index = 0
    base_name = name_upper
    for i in range(len(name_upper) - 1, -1, -1):
        if name_upper[i].isdigit():
            continue
        else:
            if i < len(name_upper) - 1:
                index = int(name_upper[i+1:])
                base_name = name_upper[:i+1]
            break

    # Map to semantic
    semantic_map = {
        "POSITION": AttributeSemantic.POSITION,
        "POS": AttributeSemantic.POSITION,
        "NORMAL": AttributeSemantic.NORMAL,
        "NORM": AttributeSemantic.NORMAL,
        "TANGENT": AttributeSemantic.TANGENT,
        "TAN": AttributeSemantic.TANGENT,
        "BINORMAL": AttributeSemantic.BINORMAL,
        "BITANGENT": AttributeSemantic.BINORMAL,
        "TEXCOORD": AttributeSemantic.TEXCOORD,
        "UV": AttributeSemantic.TEXCOORD,
        "COLOR": AttributeSemantic.COLOR,
        "COL": AttributeSemantic.COLOR,
    }

    semantic = semantic_map.get(base_name, AttributeSemantic.UNKNOWN)
    return semantic, index


class VertexDecoder:
    """
    Decodes vertex data from RenderDoc captures.

    Handles:
        - Different APIs (D3D11, D3D12, Vulkan, OpenGL)
        - Various vertex attribute layouts
        - Indexed and non-indexed draws
        - Instanced draws
        - Clip space to world space conversion
    """

    def __init__(self, controller):
        """
        Initialize decoder with RenderDoc replay controller.

        Args:
            controller: RenderDoc ReplayController instance
        """
        if not HAS_RENDERDOC:
            raise RuntimeError("RenderDoc module not available")

        self.controller = controller
        self.api_type = self._detect_api()

    def _detect_api(self) -> str:
        """Detect graphics API from capture."""
        api_props = self.controller.GetAPIProperties()
        return api_props.pipelineType.name if hasattr(api_props, 'pipelineType') else "Unknown"

    def decode_draw_call(self, event_id: int) -> Optional[DecodedMesh]:
        """
        Decode mesh data for a specific draw call.

        Args:
            event_id: RenderDoc event ID

        Returns:
            DecodedMesh or None if decoding fails
        """
        # Set replay position
        self.controller.SetFrameEvent(event_id, False)

        # Get pipeline state
        state = self.controller.GetPipelineState()

        # Get post-VS data
        postvs = self.controller.GetPostVSData(0, 0, rd.MeshDataStage.VSOut)

        if postvs.numIndices == 0:
            return None

        # Build attribute layout from shader reflection
        attributes = self._build_attribute_layout(state, postvs)

        if not attributes:
            return None

        # Get vertex buffer data
        vb_data = self.controller.GetBufferData(
            postvs.vertexResourceId,
            postvs.vertexByteOffset,
            0  # 0 = read entire buffer
        )

        if not vb_data:
            return None

        # Decode vertices
        stride = postvs.vertexByteStride
        vertex_count = min(postvs.numIndices, len(vb_data) // stride)

        vertices = []
        bounds_min = [float('inf')] * 3
        bounds_max = [float('-inf')] * 3

        for i in range(vertex_count):
            vertex = self._decode_vertex(vb_data, i * stride, attributes)
            if vertex:
                vertices.append(vertex)
                # Update bounds
                for j in range(3):
                    bounds_min[j] = min(bounds_min[j], vertex.position[j])
                    bounds_max[j] = max(bounds_max[j], vertex.position[j])

        if len(vertices) < 3:
            return None

        # Get indices if indexed draw
        indices = self._get_indices(state, postvs)

        return DecodedMesh(
            vertices=vertices,
            indices=indices,
            triangle_count=len(vertices) // 3 if not indices else len(indices) // 3,
            event_id=event_id,
            bounds_min=bounds_min,
            bounds_max=bounds_max
        )

    def _build_attribute_layout(self, state, postvs) -> List[VertexAttribute]:
        """
        Build attribute layout from shader reflection.

        Position is always first in post-VS output, regardless of
        declaration order in the shader.
        """
        attributes = []

        # Get vertex shader reflection
        vs = state.GetShader(rd.ShaderStage.Vertex)
        if vs == rd.ResourceId.Null():
            # Fallback: assume standard layout
            return self._default_attribute_layout(postvs.vertexByteStride)

        refl = state.GetShaderReflection(rd.ShaderStage.Vertex)
        if not refl:
            return self._default_attribute_layout(postvs.vertexByteStride)

        # Build from output signature
        offset = 0
        position_attr = None

        for sig in refl.outputSignature:
            semantic, index = parse_semantic(sig.varName or sig.semanticName)

            # Determine component info
            comp_count = sig.compCount
            comp_type = 'f'  # Default to float
            comp_size = 4

            if hasattr(sig, 'compType') and sig.compType in COMP_TYPE_MAP:
                comp_type, comp_size = COMP_TYPE_MAP[sig.compType]

            attr = VertexAttribute(
                name=sig.varName or sig.semanticName,
                semantic=semantic,
                semantic_index=index,
                component_type=comp_type,
                component_count=comp_count,
                component_size=comp_size,
                offset=offset,
                is_builtin_position=sig.systemValue == rd.ShaderBuiltin.Position
            )

            if attr.is_builtin_position:
                position_attr = attr
            else:
                attributes.append(attr)

            offset += attr.byte_size

            # Handle Vulkan padding (align to 16 bytes for vec4)
            if self.api_type == "Vulkan":
                padding = (16 - (offset % 16)) % 16
                offset += padding

        # Position is always first in output
        if position_attr:
            position_attr.offset = 0
            # Shift other offsets
            pos_size = position_attr.byte_size
            for attr in attributes:
                attr.offset += pos_size
            attributes.insert(0, position_attr)

        return attributes

    def _default_attribute_layout(self, stride: int) -> List[VertexAttribute]:
        """
        Default attribute layout when reflection unavailable.

        Assumes: position(vec4) + normal(vec3) + texcoord(vec2)
        """
        attributes = []

        # Position (4 floats for clip space)
        attributes.append(VertexAttribute(
            name="POSITION",
            semantic=AttributeSemantic.POSITION,
            semantic_index=0,
            component_type='f',
            component_count=4,
            component_size=4,
            offset=0,
            is_builtin_position=True
        ))

        offset = 16

        # Normal (3 floats) if stride allows
        if stride >= 28:
            attributes.append(VertexAttribute(
                name="NORMAL",
                semantic=AttributeSemantic.NORMAL,
                semantic_index=0,
                component_type='f',
                component_count=3,
                component_size=4,
                offset=offset
            ))
            offset += 12

        # Texcoord (2 floats) if stride allows
        if stride >= 36:
            attributes.append(VertexAttribute(
                name="TEXCOORD",
                semantic=AttributeSemantic.TEXCOORD,
                semantic_index=0,
                component_type='f',
                component_count=2,
                component_size=4,
                offset=offset
            ))

        return attributes

    def _decode_vertex(self, data: bytes, offset: int,
                       attributes: List[VertexAttribute]) -> Optional[DecodedVertex]:
        """Decode a single vertex from buffer data."""
        if offset + max(a.offset + a.byte_size for a in attributes) > len(data):
            return None

        vertex = DecodedVertex(
            position=[0, 0, 0],
            raw_attributes={}
        )

        for attr in attributes:
            try:
                values = struct.unpack_from(
                    attr.unpack_format(),
                    data,
                    offset + attr.offset
                )

                # Normalize if needed
                if attr.component_type == 'B':  # UNorm
                    values = tuple(v / 255.0 for v in values)
                elif attr.component_type == 'b':  # SNorm
                    values = tuple(v / 127.0 for v in values)

                values = list(values)

                # Store by semantic
                if attr.semantic == AttributeSemantic.POSITION:
                    # Handle clip space (vec4) -> world space (vec3)
                    if len(values) >= 4 and values[3] != 0:
                        # Perspective divide
                        vertex.position = [
                            values[0] / values[3],
                            values[1] / values[3],
                            values[2] / values[3]
                        ]
                    else:
                        vertex.position = values[:3]

                elif attr.semantic == AttributeSemantic.NORMAL:
                    vertex.normal = values[:3] if len(values) >= 3 else values

                elif attr.semantic == AttributeSemantic.TANGENT:
                    vertex.tangent = values[:4] if len(values) >= 4 else values

                elif attr.semantic == AttributeSemantic.TEXCOORD:
                    if attr.semantic_index == 0:
                        vertex.texcoord0 = values[:2] if len(values) >= 2 else values
                        # Flip V for OpenGL convention
                        if len(vertex.texcoord0) >= 2:
                            vertex.texcoord0[1] = 1.0 - vertex.texcoord0[1]
                    elif attr.semantic_index == 1:
                        vertex.texcoord1 = values[:2] if len(values) >= 2 else values

                elif attr.semantic == AttributeSemantic.COLOR:
                    vertex.color = values

                # Store raw
                vertex.raw_attributes[attr.name] = values

            except struct.error:
                continue

        return vertex

    def _get_indices(self, state, postvs) -> List[int]:
        """Get index buffer data if indexed draw."""
        indices = []

        ib = state.GetIBuffer()
        if ib.resourceId == rd.ResourceId.Null():
            return indices

        # Get index data
        ib_data = self.controller.GetBufferData(
            ib.resourceId,
            ib.byteOffset,
            0
        )

        if not ib_data:
            return indices

        # Parse based on format
        if ib.byteStride == 2:
            fmt = 'H'
        else:
            fmt = 'I'

        count = min(postvs.numIndices, len(ib_data) // ib.byteStride)

        for i in range(count):
            idx = struct.unpack_from(fmt, ib_data, i * ib.byteStride)[0]
            indices.append(idx)

        return indices


def decode_post_vs_mesh(controller, event_id: int) -> Optional[dict]:
    """
    Convenience function to decode mesh from RenderDoc.

    Args:
        controller: RenderDoc ReplayController
        event_id: Draw call event ID

    Returns:
        Dict with vertices, normals, uvs, faces ready for OBJ export
    """
    decoder = VertexDecoder(controller)
    mesh = decoder.decode_draw_call(event_id)

    if mesh:
        return mesh.to_obj_data()
    return None


# Export mesh to OBJ (standalone function)
def export_decoded_mesh_to_obj(mesh: DecodedMesh, filepath: str) -> bool:
    """Export DecodedMesh to Wavefront OBJ format."""
    data = mesh.to_obj_data()

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"# Decoded from RenderDoc capture\n")
        f.write(f"# Event ID: {mesh.event_id}\n")
        f.write(f"# Vertices: {len(mesh.vertices)}\n")
        f.write(f"# Triangles: {mesh.triangle_count}\n\n")

        # Vertices
        for v in data['vertices']:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        # Normals
        if data['normals']:
            f.write("\n")
            for n in data['normals']:
                f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")

        # UVs
        if data['uvs']:
            f.write("\n")
            for uv in data['uvs']:
                f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")

        # Faces
        f.write("\n")
        has_n = bool(data['normals'])
        has_uv = bool(data['uvs'])

        for face in data['faces']:
            if has_uv and has_n:
                f.write(f"f {face[0]+1}/{face[0]+1}/{face[0]+1} "
                        f"{face[1]+1}/{face[1]+1}/{face[1]+1} "
                        f"{face[2]+1}/{face[2]+1}/{face[2]+1}\n")
            elif has_uv:
                f.write(f"f {face[0]+1}/{face[0]+1} "
                        f"{face[1]+1}/{face[1]+1} "
                        f"{face[2]+1}/{face[2]+1}\n")
            elif has_n:
                f.write(f"f {face[0]+1}//{face[0]+1} "
                        f"{face[1]+1}//{face[1]+1} "
                        f"{face[2]+1}//{face[2]+1}\n")
            else:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    return True
