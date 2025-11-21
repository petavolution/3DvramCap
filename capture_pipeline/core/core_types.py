#!/usr/bin/env python3
"""
Core Type Definitions
=====================
Shared data structures and types used across the capture pipeline.

This module provides the canonical definitions for:
    - Mesh data structures
    - Material definitions
    - Texture information
    - Configuration types
    - Export formats

Author: Capture Pipeline
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
import json


# =============================================================================
# Enums
# =============================================================================

class CoordinateSystem(Enum):
    """Coordinate system conventions."""
    UE4 = "ue4"           # Left-handed, Z-up, X-forward, cm
    BLENDER = "blender"   # Right-handed, Z-up, Y-forward, m
    GLTF = "gltf"         # Right-handed, Y-up, Z-forward, m
    USD = "usd"           # Right-handed, Y-up, cm (configurable)
    OPENGL = "opengl"     # Right-handed, Y-up, -Z-forward


class TextureType(Enum):
    """PBR texture types."""
    ALBEDO = "albedo"
    NORMAL = "normal"
    ROUGHNESS = "roughness"
    METALLIC = "metallic"
    AO = "ao"
    EMISSIVE = "emissive"
    ORM = "orm"  # Packed: Occlusion/Roughness/Metallic
    HEIGHT = "height"
    OPACITY = "opacity"
    UNKNOWN = "unknown"


class ExportFormat(Enum):
    """Supported export formats."""
    OBJ = "obj"
    GLTF = "gltf"
    GLB = "glb"
    USD = "usd"
    USDA = "usda"
    USDC = "usdc"
    FBX = "fbx"
    BLEND = "blend"


class MaterialType(Enum):
    """Material rendering types."""
    UNLIT = "unlit"
    PBR_METALLIC = "pbr_metallic"
    PBR_SPECULAR = "pbr_specular"
    BAKED = "baked"


# =============================================================================
# Basic Geometry Types
# =============================================================================

@dataclass
class Vector2:
    """2D vector."""
    x: float = 0.0
    y: float = 0.0

    def to_list(self) -> List[float]:
        return [self.x, self.y]

    def to_tuple(self) -> tuple:
        return (self.x, self.y)

    @classmethod
    def from_list(cls, data: List[float]) -> 'Vector2':
        return cls(data[0], data[1])


@dataclass
class Vector3:
    """3D vector."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z]

    def to_tuple(self) -> tuple:
        return (self.x, self.y, self.z)

    @classmethod
    def from_list(cls, data: List[float]) -> 'Vector3':
        return cls(data[0], data[1], data[2])

    def length(self) -> float:
        import math
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalized(self) -> 'Vector3':
        l = self.length()
        if l < 1e-10:
            return Vector3(0, 0, 0)
        return Vector3(self.x/l, self.y/l, self.z/l)


@dataclass
class Vector4:
    """4D vector (used for clip space, quaternions, etc.)."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z, self.w]

    def xyz(self) -> Vector3:
        return Vector3(self.x, self.y, self.z)

    def perspective_divide(self) -> Vector3:
        """Convert from clip space to NDC."""
        if abs(self.w) > 1e-10:
            return Vector3(self.x/self.w, self.y/self.w, self.z/self.w)
        return Vector3(self.x, self.y, self.z)


@dataclass
class AABB:
    """Axis-aligned bounding box."""
    min_point: Vector3 = field(default_factory=Vector3)
    max_point: Vector3 = field(default_factory=Vector3)

    @property
    def center(self) -> Vector3:
        return Vector3(
            (self.min_point.x + self.max_point.x) / 2,
            (self.min_point.y + self.max_point.y) / 2,
            (self.min_point.z + self.max_point.z) / 2
        )

    @property
    def size(self) -> Vector3:
        return Vector3(
            self.max_point.x - self.min_point.x,
            self.max_point.y - self.min_point.y,
            self.max_point.z - self.min_point.z
        )

    def to_dict(self) -> dict:
        return {
            'min': self.min_point.to_list(),
            'max': self.max_point.to_list()
        }


# =============================================================================
# Mesh Types
# =============================================================================

@dataclass
class Vertex:
    """A single vertex with all attributes."""
    position: Vector3
    normal: Optional[Vector3] = None
    tangent: Optional[Vector4] = None
    texcoord0: Optional[Vector2] = None
    texcoord1: Optional[Vector2] = None
    color: Optional[Vector4] = None

    def to_dict(self) -> dict:
        d = {'position': self.position.to_list()}
        if self.normal:
            d['normal'] = self.normal.to_list()
        if self.tangent:
            d['tangent'] = self.tangent.to_list()
        if self.texcoord0:
            d['texcoord0'] = self.texcoord0.to_list()
        if self.texcoord1:
            d['texcoord1'] = self.texcoord1.to_list()
        if self.color:
            d['color'] = self.color.to_list()
        return d


@dataclass
class MeshPrimitive:
    """A mesh primitive (submesh with single material)."""
    vertices: List[Vertex]
    indices: List[int]
    material_id: int = -1
    name: str = ""

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    @property
    def face_count(self) -> int:
        return len(self.indices) // 3


@dataclass
class Mesh:
    """A complete mesh with primitives."""
    name: str
    primitives: List[MeshPrimitive]
    bounds: AABB = None
    source_event_id: int = 0
    geometry_hash: str = ""
    filepath: str = ""

    @property
    def total_vertices(self) -> int:
        return sum(p.vertex_count for p in self.primitives)

    @property
    def total_faces(self) -> int:
        return sum(p.face_count for p in self.primitives)

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'vertices': self.total_vertices,
            'faces': self.total_faces,
            'primitives': len(self.primitives),
            'bounds': self.bounds.to_dict() if self.bounds else None,
            'event_id': self.source_event_id,
            'hash': self.geometry_hash,
            'filepath': self.filepath
        }


# =============================================================================
# Material Types
# =============================================================================

@dataclass
class TextureRef:
    """Reference to a texture."""
    filepath: str
    texture_type: TextureType
    uv_channel: int = 0
    # Transform
    offset: Vector2 = field(default_factory=Vector2)
    scale: Vector2 = field(default_factory=lambda: Vector2(1, 1))

    def to_dict(self) -> dict:
        return {
            'filepath': self.filepath,
            'type': self.texture_type.value,
            'uv_channel': self.uv_channel,
            'offset': self.offset.to_list(),
            'scale': self.scale.to_list()
        }


@dataclass
class Material:
    """Material definition."""
    name: str
    material_type: MaterialType = MaterialType.UNLIT
    # Base properties
    base_color: Vector4 = field(default_factory=lambda: Vector4(0.8, 0.8, 0.8, 1.0))
    metallic: float = 0.0
    roughness: float = 0.5
    emissive: Vector3 = field(default_factory=Vector3)
    emissive_strength: float = 1.0
    # Textures
    textures: Dict[TextureType, TextureRef] = field(default_factory=dict)
    # Flags
    double_sided: bool = False
    alpha_mode: str = "OPAQUE"  # OPAQUE, MASK, BLEND
    alpha_cutoff: float = 0.5

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'type': self.material_type.value,
            'base_color': self.base_color.to_list(),
            'metallic': self.metallic,
            'roughness': self.roughness,
            'emissive': self.emissive.to_list(),
            'double_sided': self.double_sided,
            'alpha_mode': self.alpha_mode,
            'textures': {
                k.value: v.to_dict() for k, v in self.textures.items()
            }
        }


# =============================================================================
# Configuration Types
# =============================================================================

@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    # Paths
    input_dir: str = "captures"
    output_dir: str = "export"
    library_dir: str = "library"
    targets_dir: str = "targets"

    # Processing
    scale_factor: float = 0.01  # UE4 cm -> Blender m
    min_vertices: int = 50
    max_meshes: int = 500
    deduplicate: bool = True

    # Baking
    bake_resolution: int = 2048
    bake_samples: int = 1

    # Export
    export_gltf: bool = True
    export_usd: bool = True
    export_blend: bool = True
    use_draco: bool = False
    use_unlit: bool = True

    # Tools
    blender_path: str = "blender"
    python_path: str = "python"

    def to_dict(self) -> dict:
        return {
            'paths': {
                'input': self.input_dir,
                'output': self.output_dir,
                'library': self.library_dir,
                'targets': self.targets_dir
            },
            'processing': {
                'scale_factor': self.scale_factor,
                'min_vertices': self.min_vertices,
                'max_meshes': self.max_meshes,
                'deduplicate': self.deduplicate
            },
            'baking': {
                'resolution': self.bake_resolution,
                'samples': self.bake_samples
            },
            'export': {
                'gltf': self.export_gltf,
                'usd': self.export_usd,
                'blend': self.export_blend,
                'draco': self.use_draco,
                'unlit': self.use_unlit
            },
            'tools': {
                'blender': self.blender_path,
                'python': self.python_path
            }
        }

    def save(self, filepath: str):
        """Save config to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'PipelineConfig':
        """Load config from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        return cls(
            input_dir=data.get('paths', {}).get('input', 'captures'),
            output_dir=data.get('paths', {}).get('output', 'export'),
            library_dir=data.get('paths', {}).get('library', 'library'),
            targets_dir=data.get('paths', {}).get('targets', 'targets'),
            scale_factor=data.get('processing', {}).get('scale_factor', 0.01),
            min_vertices=data.get('processing', {}).get('min_vertices', 50),
            max_meshes=data.get('processing', {}).get('max_meshes', 500),
            deduplicate=data.get('processing', {}).get('deduplicate', True),
            bake_resolution=data.get('baking', {}).get('resolution', 2048),
            bake_samples=data.get('baking', {}).get('samples', 1),
            export_gltf=data.get('export', {}).get('gltf', True),
            export_usd=data.get('export', {}).get('usd', True),
            export_blend=data.get('export', {}).get('blend', True),
            use_draco=data.get('export', {}).get('draco', False),
            use_unlit=data.get('export', {}).get('unlit', True),
            blender_path=data.get('tools', {}).get('blender', 'blender'),
            python_path=data.get('tools', {}).get('python', 'python')
        )


# =============================================================================
# Result Types
# =============================================================================

@dataclass
class ExtractionResult:
    """Result of RenderDoc extraction."""
    capture_path: str
    output_dir: str
    meshes: List[Mesh]
    textures: List[str]
    camera_data: Optional[dict] = None
    stats: dict = field(default_factory=dict)
    success: bool = True
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'capture': self.capture_path,
            'output': self.output_dir,
            'meshes': [m.to_dict() for m in self.meshes],
            'textures': self.textures,
            'camera': self.camera_data,
            'stats': self.stats,
            'success': self.success,
            'errors': self.errors
        }


@dataclass
class ProcessingResult:
    """Result of mesh processing."""
    input_dir: str
    output_dir: str
    total_input: int
    unique_output: int
    duplicates_removed: int
    meshes: List[dict]
    success: bool = True
    errors: List[str] = field(default_factory=list)


@dataclass
class ExportResult:
    """Result of export operation."""
    gltf_path: Optional[str] = None
    usd_path: Optional[str] = None
    blend_path: Optional[str] = None
    atlas_path: Optional[str] = None
    mesh_count: int = 0
    material_count: int = 0
    success: bool = True
    errors: List[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Result of validation."""
    filepath: str
    format: str
    passed: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'filepath': self.filepath,
            'format': self.format,
            'passed': self.passed,
            'errors': self.errors,
            'warnings': self.warnings,
            'info': self.info
        }


# =============================================================================
# Conversion Helpers
# =============================================================================

def convert_coordinate_system(point: Vector3,
                               from_system: CoordinateSystem,
                               to_system: CoordinateSystem) -> Vector3:
    """
    Convert a point between coordinate systems.

    Common conversions:
        UE4 -> Blender: (X, Y, Z) -> (X*0.01, -Y*0.01, Z*0.01)
        UE4 -> glTF:    (X, Y, Z) -> (X*0.01, Z*0.01, -Y*0.01)
        Blender -> glTF: (X, Y, Z) -> (X, Z, -Y)
    """
    if from_system == to_system:
        return point

    x, y, z = point.x, point.y, point.z

    # Convert to intermediate (Blender-like)
    if from_system == CoordinateSystem.UE4:
        # UE4: X-forward, Y-right, Z-up (left-handed, cm)
        # To: Y-forward, -X-right, Z-up (right-handed, m)
        x, y, z = x * 0.01, -y * 0.01, z * 0.01

    # Convert from intermediate to target
    if to_system == CoordinateSystem.GLTF:
        # Blender Z-up -> glTF Y-up
        x, y, z = x, z, -y
    elif to_system == CoordinateSystem.UE4:
        # Back to UE4
        x, y, z = x * 100, -y * 100, z * 100

    return Vector3(x, y, z)


# Export all types
__all__ = [
    'CoordinateSystem', 'TextureType', 'ExportFormat', 'MaterialType',
    'Vector2', 'Vector3', 'Vector4', 'AABB',
    'Vertex', 'MeshPrimitive', 'Mesh',
    'TextureRef', 'Material',
    'PipelineConfig',
    'ExtractionResult', 'ProcessingResult', 'ExportResult', 'ValidationResult',
    'convert_coordinate_system'
]
