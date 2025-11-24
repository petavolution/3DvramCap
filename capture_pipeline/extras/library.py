#!/usr/bin/env python3
"""
Asset Library Management
========================
Manage deduplicated asset library with search and organization.

Features:
    - Organize meshes by geometry hash
    - Texture library with deduplication
    - Search by name, tags, properties
    - Export sets for selective export
    - Library statistics and reports

Usage:
    from core_library import AssetLibrary

    library = AssetLibrary("library/")

    # Import meshes from extraction
    library.import_meshes("export/scene/Meshes/")

    # Search
    results = library.search(vertex_count_min=1000, has_uvs=True)

    # Create export set
    library.create_export_set("hero_assets", [mesh_id1, mesh_id2])

Author: Capture Pipeline
"""

import json
import shutil
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Set, Tuple
import re

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class MeshAsset:
    """A mesh asset in the library."""
    id: str
    name: str
    filepath: str
    geometry_hash: str
    vertex_count: int
    face_count: int
    has_normals: bool = False
    has_uvs: bool = False
    bounds_min: Tuple[float, float, float] = (0, 0, 0)
    bounds_max: Tuple[float, float, float] = (0, 0, 0)
    tags: List[str] = field(default_factory=list)
    source_capture: str = ""
    created_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'MeshAsset':
        # Handle tuple conversion
        if 'bounds_min' in data and isinstance(data['bounds_min'], list):
            data['bounds_min'] = tuple(data['bounds_min'])
        if 'bounds_max' in data and isinstance(data['bounds_max'], list):
            data['bounds_max'] = tuple(data['bounds_max'])
        return cls(**data)

    @property
    def bounds_size(self) -> Tuple[float, float, float]:
        return tuple(self.bounds_max[i] - self.bounds_min[i] for i in range(3))


@dataclass
class TextureAsset:
    """A texture asset in the library."""
    id: str
    name: str
    filepath: str
    content_hash: str
    width: int
    height: int
    format: str
    texture_type: str = "unknown"  # albedo, normal, etc.
    tags: List[str] = field(default_factory=list)
    source_capture: str = ""
    created_at: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'TextureAsset':
        return cls(**data)


@dataclass
class ExportSet:
    """A named set of assets for export."""
    name: str
    description: str = ""
    mesh_ids: List[str] = field(default_factory=list)
    texture_ids: List[str] = field(default_factory=list)
    created_at: str = ""
    export_config: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


class AssetLibrary:
    """
    Manage organized asset library.

    Directory structure:
        library/
            meshes/
                <hash>/
                    mesh.obj
                    metadata.json
            textures/
                <hash>/
                    texture.png
                    metadata.json
            index.json
            export_sets/
                <name>.json
    """

    def __init__(self, library_dir: str = "library"):
        self.library_dir = Path(library_dir)
        self.meshes_dir = self.library_dir / "meshes"
        self.textures_dir = self.library_dir / "textures"
        self.export_sets_dir = self.library_dir / "export_sets"
        self.index_path = self.library_dir / "index.json"

        # In-memory index
        self.meshes: Dict[str, MeshAsset] = {}
        self.textures: Dict[str, TextureAsset] = {}
        self.hash_to_mesh: Dict[str, str] = {}  # geometry_hash -> mesh_id
        self.hash_to_texture: Dict[str, str] = {}  # content_hash -> texture_id

        self._ensure_dirs()
        self._load_index()

    def _ensure_dirs(self):
        """Create library directories."""
        for d in [self.library_dir, self.meshes_dir, self.textures_dir, self.export_sets_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _load_index(self):
        """Load library index from disk."""
        if self.index_path.exists():
            try:
                with open(self.index_path, 'r') as f:
                    data = json.load(f)

                for mesh_data in data.get('meshes', []):
                    mesh = MeshAsset.from_dict(mesh_data)
                    self.meshes[mesh.id] = mesh
                    self.hash_to_mesh[mesh.geometry_hash] = mesh.id

                for tex_data in data.get('textures', []):
                    tex = TextureAsset.from_dict(tex_data)
                    self.textures[tex.id] = tex
                    self.hash_to_texture[tex.content_hash] = tex.id

            except Exception as e:
                print(f"Warning: Failed to load index: {e}")

    def _save_index(self):
        """Save library index to disk."""
        data = {
            'version': '1.0',
            'updated_at': datetime.now().isoformat(),
            'meshes': [m.to_dict() for m in self.meshes.values()],
            'textures': [t.to_dict() for t in self.textures.values()]
        }

        with open(self.index_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _generate_id(self, prefix: str = "asset") -> str:
        """Generate unique asset ID."""
        import uuid
        return f"{prefix}_{uuid.uuid4().hex[:12]}"

    def _compute_geometry_hash(self, filepath: str) -> str:
        """Compute geometry hash from OBJ file."""
        vertices = []

        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        # Round to reduce floating point noise
                        v = tuple(round(float(parts[i]), 4) for i in range(1, 4))
                        vertices.append(v)

        # Sort vertices for consistent hash
        vertices.sort()
        data = json.dumps(vertices).encode()
        return hashlib.md5(data).hexdigest()

    def _analyze_mesh(self, filepath: str) -> Dict[str, Any]:
        """Analyze mesh properties."""
        result = {
            'vertex_count': 0,
            'face_count': 0,
            'has_normals': False,
            'has_uvs': False,
            'bounds_min': [float('inf')] * 3,
            'bounds_max': [float('-inf')] * 3
        }

        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue

                if parts[0] == 'v' and len(parts) >= 4:
                    result['vertex_count'] += 1
                    for i in range(3):
                        v = float(parts[i + 1])
                        result['bounds_min'][i] = min(result['bounds_min'][i], v)
                        result['bounds_max'][i] = max(result['bounds_max'][i], v)

                elif parts[0] == 'vn':
                    result['has_normals'] = True

                elif parts[0] == 'vt':
                    result['has_uvs'] = True

                elif parts[0] == 'f':
                    result['face_count'] += 1

        # Convert to tuples
        result['bounds_min'] = tuple(result['bounds_min'])
        result['bounds_max'] = tuple(result['bounds_max'])

        return result

    def add_mesh(self, filepath: str,
                 name: str = None,
                 source_capture: str = "",
                 tags: List[str] = None,
                 force: bool = False) -> Optional[str]:
        """
        Add a mesh to the library.

        Returns:
            Mesh ID if added, None if duplicate (unless force=True)
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Mesh not found: {filepath}")

        # Compute hash
        geometry_hash = self._compute_geometry_hash(str(filepath))

        # Check for duplicate
        if geometry_hash in self.hash_to_mesh and not force:
            return None  # Already in library

        # Analyze mesh
        analysis = self._analyze_mesh(str(filepath))

        # Generate ID and paths
        mesh_id = self._generate_id("mesh")
        mesh_dir = self.meshes_dir / geometry_hash
        mesh_dir.mkdir(parents=True, exist_ok=True)

        dest_path = mesh_dir / "mesh.obj"

        # Copy file
        shutil.copy2(filepath, dest_path)

        # Create asset record
        asset = MeshAsset(
            id=mesh_id,
            name=name or filepath.stem,
            filepath=str(dest_path.relative_to(self.library_dir)),
            geometry_hash=geometry_hash,
            vertex_count=analysis['vertex_count'],
            face_count=analysis['face_count'],
            has_normals=analysis['has_normals'],
            has_uvs=analysis['has_uvs'],
            bounds_min=analysis['bounds_min'],
            bounds_max=analysis['bounds_max'],
            tags=tags or [],
            source_capture=source_capture,
            created_at=datetime.now().isoformat()
        )

        # Save metadata
        with open(mesh_dir / "metadata.json", 'w') as f:
            json.dump(asset.to_dict(), f, indent=2)

        # Update index
        self.meshes[mesh_id] = asset
        self.hash_to_mesh[geometry_hash] = mesh_id
        self._save_index()

        return mesh_id

    def add_texture(self, filepath: str,
                    name: str = None,
                    texture_type: str = "unknown",
                    source_capture: str = "",
                    tags: List[str] = None) -> Optional[str]:
        """Add a texture to the library."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Texture not found: {filepath}")

        # Compute content hash
        with open(filepath, 'rb') as f:
            content_hash = hashlib.md5(f.read()).hexdigest()

        # Check for duplicate
        if content_hash in self.hash_to_texture:
            return None

        # Get dimensions
        width, height = 0, 0
        try:
            from PIL import Image
            with Image.open(filepath) as img:
                width, height = img.size
        except ImportError:
            pass

        # Generate ID and paths
        tex_id = self._generate_id("tex")
        tex_dir = self.textures_dir / content_hash
        tex_dir.mkdir(parents=True, exist_ok=True)

        dest_path = tex_dir / f"texture{filepath.suffix}"
        shutil.copy2(filepath, dest_path)

        # Create asset record
        asset = TextureAsset(
            id=tex_id,
            name=name or filepath.stem,
            filepath=str(dest_path.relative_to(self.library_dir)),
            content_hash=content_hash,
            width=width,
            height=height,
            format=filepath.suffix.lstrip('.'),
            texture_type=texture_type,
            tags=tags or [],
            source_capture=source_capture,
            created_at=datetime.now().isoformat()
        )

        # Update index
        self.textures[tex_id] = asset
        self.hash_to_texture[content_hash] = tex_id
        self._save_index()

        return tex_id

    def import_meshes(self, directory: str,
                      source_capture: str = "",
                      tags: List[str] = None) -> Dict[str, int]:
        """Import all meshes from a directory."""
        dir_path = Path(directory)
        results = {'added': 0, 'duplicates': 0, 'errors': 0}

        for obj_file in dir_path.glob("*.obj"):
            try:
                mesh_id = self.add_mesh(
                    str(obj_file),
                    source_capture=source_capture,
                    tags=tags
                )
                if mesh_id:
                    results['added'] += 1
                else:
                    results['duplicates'] += 1
            except Exception as e:
                print(f"Error importing {obj_file}: {e}")
                results['errors'] += 1

        return results

    def import_textures(self, directory: str,
                        source_capture: str = "",
                        tags: List[str] = None) -> Dict[str, int]:
        """Import all textures from a directory."""
        dir_path = Path(directory)
        results = {'added': 0, 'duplicates': 0, 'errors': 0}

        for ext in ['*.png', '*.jpg', '*.jpeg', '*.dds', '*.tga']:
            for tex_file in dir_path.glob(ext):
                try:
                    tex_id = self.add_texture(
                        str(tex_file),
                        source_capture=source_capture,
                        tags=tags
                    )
                    if tex_id:
                        results['added'] += 1
                    else:
                        results['duplicates'] += 1
                except Exception as e:
                    print(f"Error importing {tex_file}: {e}")
                    results['errors'] += 1

        return results

    def get_mesh(self, mesh_id: str) -> Optional[MeshAsset]:
        """Get mesh by ID."""
        return self.meshes.get(mesh_id)

    def get_mesh_path(self, mesh_id: str) -> Optional[Path]:
        """Get absolute path to mesh file."""
        mesh = self.meshes.get(mesh_id)
        if mesh:
            return self.library_dir / mesh.filepath
        return None

    def search_meshes(self,
                      name_pattern: str = None,
                      tags: List[str] = None,
                      vertex_count_min: int = None,
                      vertex_count_max: int = None,
                      has_uvs: bool = None,
                      has_normals: bool = None,
                      source_capture: str = None) -> List[MeshAsset]:
        """Search meshes with filters."""
        results = []

        for mesh in self.meshes.values():
            # Name pattern
            if name_pattern and not re.search(name_pattern, mesh.name, re.IGNORECASE):
                continue

            # Tags (must have all)
            if tags and not all(t in mesh.tags for t in tags):
                continue

            # Vertex count
            if vertex_count_min and mesh.vertex_count < vertex_count_min:
                continue
            if vertex_count_max and mesh.vertex_count > vertex_count_max:
                continue

            # Properties
            if has_uvs is not None and mesh.has_uvs != has_uvs:
                continue
            if has_normals is not None and mesh.has_normals != has_normals:
                continue

            # Source
            if source_capture and mesh.source_capture != source_capture:
                continue

            results.append(mesh)

        return results

    def add_tags(self, mesh_id: str, tags: List[str]):
        """Add tags to a mesh."""
        if mesh_id in self.meshes:
            mesh = self.meshes[mesh_id]
            mesh.tags = list(set(mesh.tags + tags))
            self._save_index()

    def remove_mesh(self, mesh_id: str):
        """Remove mesh from library."""
        mesh = self.meshes.get(mesh_id)
        if mesh:
            # Remove files
            mesh_dir = self.meshes_dir / mesh.geometry_hash
            if mesh_dir.exists():
                shutil.rmtree(mesh_dir)

            # Update index
            del self.meshes[mesh_id]
            if mesh.geometry_hash in self.hash_to_mesh:
                del self.hash_to_mesh[mesh.geometry_hash]
            self._save_index()

    def create_export_set(self, name: str,
                          mesh_ids: List[str] = None,
                          texture_ids: List[str] = None,
                          description: str = "",
                          export_config: Dict = None) -> ExportSet:
        """Create a named export set."""
        export_set = ExportSet(
            name=name,
            description=description,
            mesh_ids=mesh_ids or [],
            texture_ids=texture_ids or [],
            created_at=datetime.now().isoformat(),
            export_config=export_config or {}
        )

        # Save to file
        set_path = self.export_sets_dir / f"{name}.json"
        with open(set_path, 'w') as f:
            json.dump(export_set.to_dict(), f, indent=2)

        return export_set

    def get_export_set(self, name: str) -> Optional[ExportSet]:
        """Load an export set."""
        set_path = self.export_sets_dir / f"{name}.json"
        if set_path.exists():
            with open(set_path, 'r') as f:
                data = json.load(f)
            return ExportSet(**data)
        return None

    def list_export_sets(self) -> List[str]:
        """List all export set names."""
        return [p.stem for p in self.export_sets_dir.glob("*.json")]

    def get_stats(self) -> Dict[str, Any]:
        """Get library statistics."""
        total_verts = sum(m.vertex_count for m in self.meshes.values())
        total_faces = sum(m.face_count for m in self.meshes.values())

        return {
            'mesh_count': len(self.meshes),
            'texture_count': len(self.textures),
            'total_vertices': total_verts,
            'total_faces': total_faces,
            'meshes_with_uvs': sum(1 for m in self.meshes.values() if m.has_uvs),
            'meshes_with_normals': sum(1 for m in self.meshes.values() if m.has_normals),
            'export_sets': len(self.list_export_sets()),
            'unique_sources': len(set(m.source_capture for m in self.meshes.values() if m.source_capture))
        }

    def export_meshes(self, output_dir: str,
                      mesh_ids: List[str] = None,
                      flatten: bool = True) -> int:
        """Export meshes to a directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        meshes_to_export = mesh_ids or list(self.meshes.keys())
        exported = 0

        for mesh_id in meshes_to_export:
            mesh = self.meshes.get(mesh_id)
            if not mesh:
                continue

            src_path = self.library_dir / mesh.filepath
            if not src_path.exists():
                continue

            if flatten:
                dest_path = output_path / f"{mesh.name}.obj"
            else:
                dest_path = output_path / mesh.geometry_hash / "mesh.obj"
                dest_path.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(src_path, dest_path)
            exported += 1

        return exported


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Asset library management")
    parser.add_argument('--library', '-l', default='library', help="Library directory")
    parser.add_argument('--import-meshes', '-m', help="Import meshes from directory")
    parser.add_argument('--import-textures', '-t', help="Import textures from directory")
    parser.add_argument('--search', '-s', help="Search by name pattern")
    parser.add_argument('--stats', action='store_true', help="Show statistics")
    parser.add_argument('--export', '-e', help="Export to directory")

    args = parser.parse_args()

    library = AssetLibrary(args.library)

    if args.import_meshes:
        results = library.import_meshes(args.import_meshes)
        print(f"Imported: {results['added']}, Duplicates: {results['duplicates']}")

    elif args.import_textures:
        results = library.import_textures(args.import_textures)
        print(f"Imported: {results['added']}, Duplicates: {results['duplicates']}")

    elif args.search:
        meshes = library.search_meshes(name_pattern=args.search)
        for m in meshes:
            print(f"[{m.id}] {m.name} - {m.vertex_count} verts, {m.face_count} faces")

    elif args.export:
        count = library.export_meshes(args.export)
        print(f"Exported {count} meshes")

    elif args.stats:
        stats = library.get_stats()
        print(json.dumps(stats, indent=2))

    else:
        stats = library.get_stats()
        print(f"Library: {args.library}")
        print(f"Meshes: {stats['mesh_count']}")
        print(f"Textures: {stats['texture_count']}")
