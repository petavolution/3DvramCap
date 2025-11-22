"""
3DvramCap Extras - Advanced Features
=====================================
Optional advanced modules for extended functionality.

These modules provide additional capabilities beyond the core pipeline:

Capture Sources:
    - depth: ReShade depth buffer reconstruction
    - ninja: Ninja Ripper .rip file parser (DX9 fallback)
    - vertex_decode: Advanced vertex attribute parsing

Processing:
    - atlas: Texture atlas packing
    - materials: PBR material classification
    - merge: Mesh merging utilities
    - optimize: Mesh decimation and LOD
    - scene: Scene hierarchy reconstruction
    - texture: DDS conversion, texture processing

Workflow:
    - batch: Human-supervised batch processing
    - database: SQLite asset tracking
    - library: Asset library management
    - presets: Export preset definitions
    - quality: QA metrics (SSIM, IoU)

Infrastructure:
    - cli: Alternative CLI interface
    - config: Extended configuration
    - logging_ext: Structured JSON logging

Legacy:
    - scripts_legacy/: Original numbered scripts (deprecated)

Usage:
    from extras.batch import BatchProcessor
    from extras.depth import DepthProcessor
    from extras.quality import assess_scene_quality
"""

__version__ = '1.0.0'
