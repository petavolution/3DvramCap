"""
3DvramCap Core Modules
======================
Essential modules for the capture-to-export pipeline.

Core Modules:
    - extract: RenderDoc capture extraction
    - process: Mesh deduplication and processing
    - blender: Blender import, bake, and export
    - gltf: Direct glTF 2.0 export
    - usd: Direct USD export
    - validate: Export validation
    - camera: Coordinate transforms
    - types: Shared data structures

Usage:
    from core import extract, process, blender
    from core.types import Mesh, Material
    from core.camera import ue4_to_blender_position
"""

# Version
__version__ = '1.0.0'

# Core module names for lazy loading
__all__ = [
    'core_extract',
    'core_process',
    'core_blender',
    'core_gltf',
    'core_usd',
    'core_validate',
    'core_camera',
    'core_types',
]
