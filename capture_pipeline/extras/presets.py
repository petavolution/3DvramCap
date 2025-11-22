#!/usr/bin/env python3
"""
Export Presets
==============
Predefined export configurations for different targets and use cases.

Features:
    - Target platform presets (Web, Unity, Unreal, Blender)
    - Quality presets (Draft, Production, Archive)
    - Custom preset creation and saving
    - Format-specific optimizations

Usage:
    from core_presets import get_preset, apply_preset, list_presets

    # Get a preset
    preset = get_preset("web_optimized")

    # Apply to export
    export_with_preset(meshes, preset, "output/")

Author: Capture Pipeline
"""

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any


class TargetPlatform(Enum):
    """Target platform types."""
    WEB = "web"
    UNITY = "unity"
    UNREAL = "unreal"
    BLENDER = "blender"
    THREE_JS = "threejs"
    GODOT = "godot"
    ARCHIVE = "archive"
    CUSTOM = "custom"


class QualityLevel(Enum):
    """Quality level types."""
    DRAFT = "draft"
    PREVIEW = "preview"
    PRODUCTION = "production"
    ARCHIVE = "archive"


@dataclass
class TexturePreset:
    """Texture export settings."""
    format: str = "png"  # png, jpg, webp
    max_size: int = 2048
    quality: int = 90
    generate_mipmaps: bool = False
    power_of_two: bool = True
    compress: bool = False


@dataclass
class MeshPreset:
    """Mesh export settings."""
    # Optimization
    decimate: bool = False
    decimate_ratio: float = 0.5
    weld_vertices: bool = True
    weld_threshold: float = 1e-5

    # LOD
    generate_lods: bool = False
    lod_levels: List[float] = field(default_factory=lambda: [0.5, 0.25])

    # Normals
    recalculate_normals: bool = False
    smooth_angle: float = 60.0

    # Transform
    center_origin: bool = False
    apply_scale: float = 1.0


@dataclass
class MaterialPreset:
    """Material export settings."""
    type: str = "unlit"  # unlit, pbr
    embed_textures: bool = True
    bake_vertex_colors: bool = False


@dataclass
class FormatPreset:
    """Format-specific settings."""
    # glTF
    gltf_binary: bool = True
    gltf_draco: bool = False
    gltf_draco_compression: int = 7
    gltf_embed_images: bool = True

    # USD
    usd_binary: bool = False
    usd_up_axis: str = "Y"
    usd_meters_per_unit: float = 1.0

    # OBJ
    obj_separate_materials: bool = False
    obj_triangulate: bool = True


@dataclass
class ExportPreset:
    """Complete export preset."""
    name: str
    description: str = ""
    platform: TargetPlatform = TargetPlatform.CUSTOM
    quality: QualityLevel = QualityLevel.PRODUCTION

    # Output formats
    output_gltf: bool = True
    output_usd: bool = False
    output_obj: bool = False
    output_blend: bool = False

    # Sub-presets
    textures: TexturePreset = field(default_factory=TexturePreset)
    meshes: MeshPreset = field(default_factory=MeshPreset)
    materials: MaterialPreset = field(default_factory=MaterialPreset)
    formats: FormatPreset = field(default_factory=FormatPreset)

    # Coordinate system
    up_axis: str = "Y"  # Y or Z
    forward_axis: str = "-Z"
    scale_factor: float = 1.0

    # Scene options
    merge_meshes: bool = False
    single_material: bool = False

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'description': self.description,
            'platform': self.platform.value,
            'quality': self.quality.value,
            'output': {
                'gltf': self.output_gltf,
                'usd': self.output_usd,
                'obj': self.output_obj,
                'blend': self.output_blend
            },
            'textures': asdict(self.textures),
            'meshes': asdict(self.meshes),
            'materials': asdict(self.materials),
            'formats': asdict(self.formats),
            'coordinate_system': {
                'up_axis': self.up_axis,
                'forward_axis': self.forward_axis,
                'scale_factor': self.scale_factor
            },
            'scene': {
                'merge_meshes': self.merge_meshes,
                'single_material': self.single_material
            }
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ExportPreset':
        preset = cls(
            name=data.get('name', 'custom'),
            description=data.get('description', ''),
            platform=TargetPlatform(data.get('platform', 'custom')),
            quality=QualityLevel(data.get('quality', 'production'))
        )

        if 'output' in data:
            preset.output_gltf = data['output'].get('gltf', True)
            preset.output_usd = data['output'].get('usd', False)
            preset.output_obj = data['output'].get('obj', False)
            preset.output_blend = data['output'].get('blend', False)

        if 'textures' in data:
            for k, v in data['textures'].items():
                if hasattr(preset.textures, k):
                    setattr(preset.textures, k, v)

        if 'meshes' in data:
            for k, v in data['meshes'].items():
                if hasattr(preset.meshes, k):
                    setattr(preset.meshes, k, v)

        if 'materials' in data:
            for k, v in data['materials'].items():
                if hasattr(preset.materials, k):
                    setattr(preset.materials, k, v)

        if 'formats' in data:
            for k, v in data['formats'].items():
                if hasattr(preset.formats, k):
                    setattr(preset.formats, k, v)

        if 'coordinate_system' in data:
            preset.up_axis = data['coordinate_system'].get('up_axis', 'Y')
            preset.forward_axis = data['coordinate_system'].get('forward_axis', '-Z')
            preset.scale_factor = data['coordinate_system'].get('scale_factor', 1.0)

        if 'scene' in data:
            preset.merge_meshes = data['scene'].get('merge_meshes', False)
            preset.single_material = data['scene'].get('single_material', False)

        return preset

    def save(self, filepath: str):
        """Save preset to file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'ExportPreset':
        """Load preset from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


# ============================================================================
# Built-in Presets
# ============================================================================

def _create_web_optimized() -> ExportPreset:
    """Web-optimized preset for browsers/WebGL."""
    preset = ExportPreset(
        name="web_optimized",
        description="Optimized for web browsers and WebGL",
        platform=TargetPlatform.WEB,
        quality=QualityLevel.PRODUCTION
    )

    # Small textures, compressed
    preset.textures.format = "webp"
    preset.textures.max_size = 1024
    preset.textures.quality = 80
    preset.textures.power_of_two = True

    # Reduced geometry
    preset.meshes.decimate = True
    preset.meshes.decimate_ratio = 0.5
    preset.meshes.weld_vertices = True

    # glTF with Draco
    preset.output_gltf = True
    preset.formats.gltf_binary = True
    preset.formats.gltf_draco = True
    preset.formats.gltf_draco_compression = 7

    # Unlit for performance
    preset.materials.type = "unlit"
    preset.materials.embed_textures = True

    # Y-up for WebGL
    preset.up_axis = "Y"
    preset.scale_factor = 1.0

    return preset


def _create_web_draft() -> ExportPreset:
    """Quick preview for web."""
    preset = ExportPreset(
        name="web_draft",
        description="Quick draft for web preview",
        platform=TargetPlatform.WEB,
        quality=QualityLevel.DRAFT
    )

    preset.textures.format = "jpg"
    preset.textures.max_size = 512
    preset.textures.quality = 70

    preset.meshes.decimate = True
    preset.meshes.decimate_ratio = 0.25

    preset.output_gltf = True
    preset.formats.gltf_binary = True
    preset.formats.gltf_draco = True

    preset.materials.type = "unlit"

    return preset


def _create_unity() -> ExportPreset:
    """Unity-compatible preset."""
    preset = ExportPreset(
        name="unity",
        description="Optimized for Unity import",
        platform=TargetPlatform.UNITY,
        quality=QualityLevel.PRODUCTION
    )

    preset.textures.format = "png"
    preset.textures.max_size = 2048
    preset.textures.power_of_two = True

    preset.output_gltf = True
    preset.formats.gltf_binary = True

    # Unity uses Y-up, left-handed
    preset.up_axis = "Y"
    preset.scale_factor = 1.0

    preset.materials.type = "pbr"

    return preset


def _create_unreal() -> ExportPreset:
    """Unreal Engine preset."""
    preset = ExportPreset(
        name="unreal",
        description="Optimized for Unreal Engine import",
        platform=TargetPlatform.UNREAL,
        quality=QualityLevel.PRODUCTION
    )

    preset.textures.format = "png"
    preset.textures.max_size = 4096
    preset.textures.power_of_two = True

    preset.output_gltf = True
    preset.formats.gltf_binary = True

    # Unreal uses Z-up, centimeters
    preset.up_axis = "Z"
    preset.scale_factor = 100.0  # meters to centimeters

    preset.materials.type = "pbr"

    return preset


def _create_blender() -> ExportPreset:
    """Blender-native preset."""
    preset = ExportPreset(
        name="blender",
        description="Native Blender format",
        platform=TargetPlatform.BLENDER,
        quality=QualityLevel.PRODUCTION
    )

    preset.textures.format = "png"
    preset.textures.max_size = 4096

    preset.output_blend = True
    preset.output_gltf = False

    # Blender uses Z-up, meters
    preset.up_axis = "Z"
    preset.scale_factor = 1.0

    return preset


def _create_threejs() -> ExportPreset:
    """Three.js optimized preset."""
    preset = ExportPreset(
        name="threejs",
        description="Optimized for Three.js",
        platform=TargetPlatform.THREE_JS,
        quality=QualityLevel.PRODUCTION
    )

    preset.textures.format = "webp"
    preset.textures.max_size = 2048
    preset.textures.power_of_two = True

    preset.meshes.generate_lods = True
    preset.meshes.lod_levels = [0.5, 0.25, 0.125]

    preset.output_gltf = True
    preset.formats.gltf_binary = True
    preset.formats.gltf_draco = True

    preset.materials.type = "unlit"
    preset.up_axis = "Y"

    return preset


def _create_archive() -> ExportPreset:
    """Archival preset - maximum quality, multiple formats."""
    preset = ExportPreset(
        name="archive",
        description="Maximum quality for archival",
        platform=TargetPlatform.ARCHIVE,
        quality=QualityLevel.ARCHIVE
    )

    preset.textures.format = "png"
    preset.textures.max_size = 8192
    preset.textures.quality = 100
    preset.textures.compress = False

    preset.meshes.decimate = False
    preset.meshes.weld_vertices = False

    preset.output_gltf = True
    preset.output_usd = True
    preset.output_obj = True
    preset.output_blend = True

    preset.formats.gltf_binary = False  # Text for archival
    preset.formats.gltf_draco = False
    preset.formats.usd_binary = False

    return preset


def _create_godot() -> ExportPreset:
    """Godot Engine preset."""
    preset = ExportPreset(
        name="godot",
        description="Optimized for Godot Engine",
        platform=TargetPlatform.GODOT,
        quality=QualityLevel.PRODUCTION
    )

    preset.textures.format = "png"
    preset.textures.max_size = 2048

    preset.output_gltf = True
    preset.formats.gltf_binary = True

    # Godot uses Y-up
    preset.up_axis = "Y"
    preset.scale_factor = 1.0

    return preset


# Built-in preset registry
BUILTIN_PRESETS: Dict[str, ExportPreset] = {}


def _init_builtin_presets():
    """Initialize built-in presets."""
    global BUILTIN_PRESETS
    BUILTIN_PRESETS = {
        'web_optimized': _create_web_optimized(),
        'web_draft': _create_web_draft(),
        'unity': _create_unity(),
        'unreal': _create_unreal(),
        'blender': _create_blender(),
        'threejs': _create_threejs(),
        'godot': _create_godot(),
        'archive': _create_archive()
    }


_init_builtin_presets()


# ============================================================================
# Public API
# ============================================================================

def get_preset(name: str) -> Optional[ExportPreset]:
    """Get a preset by name."""
    return BUILTIN_PRESETS.get(name)


def list_presets() -> List[str]:
    """List available preset names."""
    return list(BUILTIN_PRESETS.keys())


def get_preset_info() -> List[Dict[str, str]]:
    """Get info about all presets."""
    return [
        {
            'name': p.name,
            'description': p.description,
            'platform': p.platform.value,
            'quality': p.quality.value
        }
        for p in BUILTIN_PRESETS.values()
    ]


def create_custom_preset(name: str,
                         base: str = None,
                         **overrides) -> ExportPreset:
    """
    Create a custom preset, optionally based on an existing one.

    Args:
        name: Name for the new preset
        base: Name of preset to use as base
        **overrides: Settings to override

    Returns:
        New ExportPreset instance
    """
    if base and base in BUILTIN_PRESETS:
        # Clone base preset
        base_preset = BUILTIN_PRESETS[base]
        preset_dict = base_preset.to_dict()
        preset_dict['name'] = name

        # Apply overrides
        for key, value in overrides.items():
            if '.' in key:
                # Nested key like "textures.max_size"
                parts = key.split('.')
                d = preset_dict
                for part in parts[:-1]:
                    if part not in d:
                        d[part] = {}
                    d = d[part]
                d[parts[-1]] = value
            else:
                preset_dict[key] = value

        return ExportPreset.from_dict(preset_dict)
    else:
        preset = ExportPreset(name=name)
        return preset


def save_preset(preset: ExportPreset, directory: str = "presets"):
    """Save a preset to a file."""
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)

    filepath = dir_path / f"{preset.name}.json"
    preset.save(str(filepath))
    print(f"Saved preset: {filepath}")


def load_custom_presets(directory: str = "presets") -> Dict[str, ExportPreset]:
    """Load custom presets from a directory."""
    dir_path = Path(directory)
    presets = {}

    if dir_path.exists():
        for filepath in dir_path.glob("*.json"):
            try:
                preset = ExportPreset.load(str(filepath))
                presets[preset.name] = preset
            except Exception as e:
                print(f"Failed to load preset {filepath}: {e}")

    return presets


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export presets")
    parser.add_argument('--list', '-l', action='store_true', help="List presets")
    parser.add_argument('--show', '-s', help="Show preset details")
    parser.add_argument('--save', help="Save preset to directory")
    parser.add_argument('--preset', '-p', help="Preset name")

    args = parser.parse_args()

    if args.list:
        print("Available presets:")
        for info in get_preset_info():
            print(f"  {info['name']:20} - {info['description']}")

    elif args.show:
        preset = get_preset(args.show)
        if preset:
            print(json.dumps(preset.to_dict(), indent=2))
        else:
            print(f"Preset not found: {args.show}")

    elif args.save and args.preset:
        preset = get_preset(args.preset)
        if preset:
            save_preset(preset, args.save)
        else:
            print(f"Preset not found: {args.preset}")

    else:
        print("Built-in presets:")
        for name in list_presets():
            preset = get_preset(name)
            print(f"  {name:20} ({preset.platform.value}, {preset.quality.value})")
