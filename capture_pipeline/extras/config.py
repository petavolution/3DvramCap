#!/usr/bin/env python3
"""
Unified Configuration Management
================================
Centralized configuration for the entire capture pipeline.

Features:
    - Hierarchical config (defaults -> file -> env -> cli)
    - Config validation with type checking
    - Environment variable override
    - Multiple config profiles
    - Auto-discovery of tools (Blender, RenderDoc)

Usage:
    from core_config import Config, get_config

    # Load default config
    config = get_config()

    # Load from file
    config = Config.load("pipeline.yaml")

    # Access settings
    print(config.paths.output_dir)
    print(config.processing.scale_factor)

Author: Capture Pipeline
"""

import os
import json
import shutil
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from enum import Enum


class ConfigProfile(Enum):
    """Predefined configuration profiles."""
    DEFAULT = "default"
    FAST = "fast"  # Quick processing, lower quality
    QUALITY = "quality"  # Slower, higher quality
    MINIMAL = "minimal"  # Minimum dependencies
    DEBUG = "debug"  # Verbose logging


@dataclass
class PathsConfig:
    """Path configuration."""
    # Input/Output
    input_dir: str = "captures"
    output_dir: str = "export"
    work_dir: str = "work"
    log_dir: str = "logs"

    # Asset management
    library_dir: str = "library"
    targets_dir: str = "targets"
    cache_dir: str = ".cache"

    # Database
    database_path: str = "pipeline.db"

    def ensure_dirs(self):
        """Create all directories if they don't exist."""
        for attr in ['input_dir', 'output_dir', 'work_dir', 'log_dir',
                     'library_dir', 'targets_dir', 'cache_dir']:
            path = Path(getattr(self, attr))
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class ToolsConfig:
    """External tool paths and settings."""
    # Tool paths (auto-detected if empty)
    blender_path: str = ""
    renderdoc_path: str = ""
    python_path: str = ""

    # Tool versions (populated during detection)
    blender_version: str = ""
    renderdoc_version: str = ""

    # Blender settings
    blender_threads: int = 0  # 0 = auto

    def detect_tools(self) -> Dict[str, bool]:
        """Auto-detect tool paths and versions."""
        found = {}

        # Blender
        if not self.blender_path:
            self.blender_path = self._find_executable([
                "blender",
                "/usr/bin/blender",
                "/Applications/Blender.app/Contents/MacOS/Blender",
                "C:\\Program Files\\Blender Foundation\\Blender 3.6\\blender.exe",
                "C:\\Program Files\\Blender Foundation\\Blender 4.0\\blender.exe"
            ])

        if self.blender_path:
            found['blender'] = True
            self.blender_version = self._get_blender_version()
        else:
            found['blender'] = False

        # RenderDoc
        if not self.renderdoc_path:
            self.renderdoc_path = self._find_executable([
                "renderdoccmd",
                "qrenderdoc",
                "/usr/bin/renderdoccmd",
                "/Applications/RenderDoc.app/Contents/MacOS/qrenderdoc",
                "C:\\Program Files\\RenderDoc\\renderdoccmd.exe"
            ])
        found['renderdoc'] = bool(self.renderdoc_path)

        # Python
        if not self.python_path:
            self.python_path = self._find_executable([
                "python3",
                "python",
                "/usr/bin/python3"
            ])
        found['python'] = bool(self.python_path)

        return found

    def _find_executable(self, candidates: List[str]) -> str:
        """Find first available executable."""
        for candidate in candidates:
            if shutil.which(candidate):
                return shutil.which(candidate)
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
        return ""

    def _get_blender_version(self) -> str:
        """Get Blender version string."""
        try:
            result = subprocess.run(
                [self.blender_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines:
                    return lines[0]
        except Exception:
            pass
        return ""


@dataclass
class ProcessingConfig:
    """Mesh processing settings."""
    # Scale
    scale_factor: float = 0.01  # UE4 cm -> m

    # Mesh filtering
    min_vertices: int = 50
    max_vertices: int = 0  # 0 = no limit
    max_meshes: int = 500

    # Deduplication
    deduplicate: bool = True
    hash_precision: int = 4  # Decimal places for vertex hashing

    # Optimization
    decimate: bool = False
    decimate_ratio: float = 0.5
    weld_threshold: float = 1e-5

    # LOD
    generate_lods: bool = False
    lod_levels: List[float] = field(default_factory=lambda: [0.5, 0.25, 0.125])


@dataclass
class BakingConfig:
    """Texture baking settings."""
    enabled: bool = True
    resolution: int = 2048
    samples: int = 1
    margin: int = 2

    # Atlas
    use_atlas: bool = True
    atlas_size: int = 4096
    atlas_padding: int = 2


@dataclass
class ExportConfig:
    """Export format settings."""
    # Formats
    export_gltf: bool = True
    export_usd: bool = True
    export_blend: bool = False
    export_obj: bool = False

    # glTF settings
    gltf_binary: bool = True  # .glb vs .gltf
    gltf_draco: bool = False
    gltf_embed_textures: bool = True

    # USD settings
    usd_binary: bool = False  # .usdc vs .usda
    usd_up_axis: str = "Y"

    # Material settings
    use_unlit: bool = True
    texture_format: str = "png"  # png, jpg, webp
    texture_quality: int = 90

    # Coordinate system
    target_up_axis: str = "Y"  # Y or Z


@dataclass
class CheckpointConfig:
    """Human supervision checkpoint settings."""
    enabled: bool = True

    # Checkpoint locations
    after_extract: bool = True
    after_dedup: bool = False
    before_export: bool = True
    after_export: bool = True
    on_quality_fail: bool = True

    # Automation
    auto_approve_timeout: Optional[float] = None  # Seconds, None = manual
    auto_approve_quality_threshold: float = 70.0  # Auto-approve if score >= threshold

    # Notifications
    webhook_url: Optional[str] = None
    email_on_complete: Optional[str] = None


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    format: str = "json"  # json, text
    console: bool = True
    file: bool = True
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


@dataclass
class Config:
    """
    Main configuration container.

    Hierarchical configuration with override support:
    1. Built-in defaults
    2. Config file (YAML/JSON)
    3. Environment variables
    4. Command-line arguments
    """
    # Sub-configs
    paths: PathsConfig = field(default_factory=PathsConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    baking: BakingConfig = field(default_factory=BakingConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    checkpoints: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Meta
    profile: str = "default"
    version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'profile': self.profile,
            'version': self.version,
            'paths': asdict(self.paths),
            'tools': {k: v for k, v in asdict(self.tools).items()
                     if not k.endswith('_version')},  # Don't save detected versions
            'processing': asdict(self.processing),
            'baking': asdict(self.baking),
            'export': asdict(self.export),
            'checkpoints': asdict(self.checkpoints),
            'logging': asdict(self.logging)
        }

    def save(self, filepath: str):
        """Save configuration to file."""
        path = Path(filepath)
        data = self.to_dict()

        if path.suffix in ('.yaml', '.yml'):
            try:
                import yaml
                with open(filepath, 'w') as f:
                    yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            except ImportError:
                # Fall back to JSON
                filepath = str(path.with_suffix('.json'))
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
        else:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

        print(f"Config saved: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'Config':
        """Load configuration from file."""
        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")

        if path.suffix in ('.yaml', '.yml'):
            try:
                import yaml
                with open(filepath, 'r') as f:
                    data = yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML required for YAML config files")
        else:
            with open(filepath, 'r') as f:
                data = json.load(f)

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """Create config from dictionary."""
        config = cls()
        config.profile = data.get('profile', 'default')
        config.version = data.get('version', '1.0')

        if 'paths' in data:
            for k, v in data['paths'].items():
                if hasattr(config.paths, k):
                    setattr(config.paths, k, v)

        if 'tools' in data:
            for k, v in data['tools'].items():
                if hasattr(config.tools, k):
                    setattr(config.tools, k, v)

        if 'processing' in data:
            for k, v in data['processing'].items():
                if hasattr(config.processing, k):
                    setattr(config.processing, k, v)

        if 'baking' in data:
            for k, v in data['baking'].items():
                if hasattr(config.baking, k):
                    setattr(config.baking, k, v)

        if 'export' in data:
            for k, v in data['export'].items():
                if hasattr(config.export, k):
                    setattr(config.export, k, v)

        if 'checkpoints' in data:
            for k, v in data['checkpoints'].items():
                if hasattr(config.checkpoints, k):
                    setattr(config.checkpoints, k, v)

        if 'logging' in data:
            for k, v in data['logging'].items():
                if hasattr(config.logging, k):
                    setattr(config.logging, k, v)

        return config

    def apply_profile(self, profile: Union[str, ConfigProfile]):
        """Apply a predefined profile."""
        if isinstance(profile, str):
            profile = ConfigProfile(profile)

        self.profile = profile.value

        if profile == ConfigProfile.FAST:
            self.processing.decimate = True
            self.processing.decimate_ratio = 0.3
            self.processing.max_meshes = 200
            self.baking.resolution = 1024
            self.baking.samples = 1
            self.export.gltf_draco = True
            self.checkpoints.enabled = False

        elif profile == ConfigProfile.QUALITY:
            self.processing.decimate = False
            self.processing.max_meshes = 1000
            self.processing.min_vertices = 20
            self.baking.resolution = 4096
            self.baking.samples = 4
            self.export.texture_quality = 95
            self.checkpoints.auto_approve_quality_threshold = 85.0

        elif profile == ConfigProfile.MINIMAL:
            self.export.export_usd = False
            self.export.export_blend = False
            self.baking.enabled = False
            self.processing.generate_lods = False
            self.checkpoints.enabled = False

        elif profile == ConfigProfile.DEBUG:
            self.logging.level = "DEBUG"
            self.checkpoints.enabled = True
            self.checkpoints.auto_approve_timeout = None

    def apply_env_overrides(self):
        """Apply environment variable overrides."""
        env_prefix = "PIPELINE_"

        # Path overrides
        env_map = {
            f"{env_prefix}INPUT_DIR": ("paths", "input_dir"),
            f"{env_prefix}OUTPUT_DIR": ("paths", "output_dir"),
            f"{env_prefix}WORK_DIR": ("paths", "work_dir"),
            f"{env_prefix}BLENDER_PATH": ("tools", "blender_path"),
            f"{env_prefix}RENDERDOC_PATH": ("tools", "renderdoc_path"),
            f"{env_prefix}SCALE_FACTOR": ("processing", "scale_factor"),
            f"{env_prefix}MAX_MESHES": ("processing", "max_meshes"),
            f"{env_prefix}BAKE_RESOLUTION": ("baking", "resolution"),
            f"{env_prefix}LOG_LEVEL": ("logging", "level"),
        }

        for env_var, (section, attr) in env_map.items():
            value = os.environ.get(env_var)
            if value is not None:
                section_obj = getattr(self, section)
                current = getattr(section_obj, attr)

                # Type conversion
                if isinstance(current, bool):
                    value = value.lower() in ('true', '1', 'yes')
                elif isinstance(current, int):
                    value = int(value)
                elif isinstance(current, float):
                    value = float(value)

                setattr(section_obj, attr, value)

    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []

        # Check scale factor
        if self.processing.scale_factor <= 0:
            issues.append("scale_factor must be positive")

        # Check resolution
        if self.baking.resolution < 64 or self.baking.resolution > 8192:
            issues.append("bake_resolution should be between 64 and 8192")

        # Check LOD levels
        for lod in self.processing.lod_levels:
            if not 0 < lod < 1:
                issues.append(f"LOD level {lod} should be between 0 and 1")

        # Check texture quality
        if not 1 <= self.export.texture_quality <= 100:
            issues.append("texture_quality should be between 1 and 100")

        # Check checkpoint threshold
        if not 0 <= self.checkpoints.auto_approve_quality_threshold <= 100:
            issues.append("auto_approve_quality_threshold should be between 0 and 100")

        return issues

    def initialize(self):
        """Full initialization: create dirs, detect tools, validate."""
        self.paths.ensure_dirs()
        self.tools.detect_tools()
        self.apply_env_overrides()

        issues = self.validate()
        if issues:
            print("Configuration warnings:")
            for issue in issues:
                print(f"  - {issue}")


# Global config instance
_global_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance."""
    global _global_config

    if _global_config is None:
        _global_config = Config()
        _global_config.initialize()

    return _global_config


def set_config(config: Config):
    """Set global configuration instance."""
    global _global_config
    _global_config = config


def load_config(filepath: str) -> Config:
    """Load and set global configuration from file."""
    config = Config.load(filepath)
    config.initialize()
    set_config(config)
    return config


def create_default_config(filepath: str = "pipeline.json"):
    """Create a default configuration file."""
    config = Config()
    config.save(filepath)
    return config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Configuration management")
    parser.add_argument('--create', '-c', help="Create default config file")
    parser.add_argument('--show', '-s', action='store_true', help="Show current config")
    parser.add_argument('--profile', '-p', choices=['default', 'fast', 'quality', 'minimal', 'debug'],
                       help="Apply profile")
    parser.add_argument('--validate', '-v', help="Validate config file")

    args = parser.parse_args()

    if args.create:
        config = Config()
        if args.profile:
            config.apply_profile(args.profile)
        config.save(args.create)

    elif args.validate:
        config = Config.load(args.validate)
        issues = config.validate()
        if issues:
            print("Validation issues:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("Configuration valid")

    elif args.show:
        config = get_config()
        print(json.dumps(config.to_dict(), indent=2))

    else:
        # Show tool detection
        config = Config()
        found = config.tools.detect_tools()
        print("Tool detection:")
        for tool, status in found.items():
            path = getattr(config.tools, f"{tool}_path", "")
            print(f"  {tool}: {'Found' if status else 'Not found'} - {path}")
