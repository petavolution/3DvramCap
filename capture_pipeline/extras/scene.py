#!/usr/bin/env python3
"""
Core Scene Hierarchy Reconstruction
====================================
Reconstructs scene hierarchy from captured draw calls and meshes.

Features:
    - Spatial clustering for object grouping
    - Material-based mesh grouping
    - LOD detection and management
    - Render pass identification
    - Instanced mesh detection

Usage:
    from core_scene import SceneBuilder, reconstruct_scene

    builder = SceneBuilder()
    for mesh in extracted_meshes:
        builder.add_mesh(mesh)
    scene = builder.build()

Author: Capture Pipeline
"""

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class RenderPassType(Enum):
    """Common render pass types in game engines."""
    UNKNOWN = "unknown"
    DEPTH_PREPASS = "depth_prepass"
    GBUFFER = "gbuffer"
    FORWARD = "forward"
    SHADOW = "shadow"
    TRANSPARENT = "transparent"
    POST_PROCESS = "post_process"
    UI = "ui"


class MeshCategory(Enum):
    """Mesh categorization based on properties."""
    STATIC = "static"
    SKELETAL = "skeletal"
    INSTANCED = "instanced"
    TERRAIN = "terrain"
    FOLIAGE = "foliage"
    PARTICLE = "particle"
    UNKNOWN = "unknown"


@dataclass
class BoundingBox:
    """Axis-aligned bounding box."""
    min_point: List[float] = field(default_factory=lambda: [0, 0, 0])
    max_point: List[float] = field(default_factory=lambda: [0, 0, 0])

    @property
    def center(self) -> List[float]:
        return [
            (self.min_point[i] + self.max_point[i]) / 2
            for i in range(3)
        ]

    @property
    def size(self) -> List[float]:
        return [
            self.max_point[i] - self.min_point[i]
            for i in range(3)
        ]

    @property
    def diagonal(self) -> float:
        s = self.size
        return math.sqrt(s[0]**2 + s[1]**2 + s[2]**2)

    def expand(self, point: List[float]):
        """Expand bounds to include point."""
        for i in range(3):
            self.min_point[i] = min(self.min_point[i], point[i])
            self.max_point[i] = max(self.max_point[i], point[i])

    def merge(self, other: 'BoundingBox'):
        """Merge with another bounding box."""
        for i in range(3):
            self.min_point[i] = min(self.min_point[i], other.min_point[i])
            self.max_point[i] = max(self.max_point[i], other.max_point[i])

    def distance_to(self, other: 'BoundingBox') -> float:
        """Calculate distance between bounding box centers."""
        c1 = self.center
        c2 = other.center
        return math.sqrt(sum((c1[i] - c2[i])**2 for i in range(3)))

    def intersects(self, other: 'BoundingBox') -> bool:
        """Check if bounding boxes intersect."""
        for i in range(3):
            if self.max_point[i] < other.min_point[i]:
                return False
            if self.min_point[i] > other.max_point[i]:
                return False
        return True


@dataclass
class SceneMesh:
    """A mesh within the scene."""
    name: str
    filepath: str
    event_id: int
    vertex_count: int
    face_count: int
    bounds: BoundingBox
    geometry_hash: str = ""
    material_id: int = -1
    category: MeshCategory = MeshCategory.UNKNOWN
    lod_level: int = 0
    instance_group: int = -1
    render_pass: RenderPassType = RenderPassType.UNKNOWN
    parent_node: Optional[str] = None


@dataclass
class SceneNode:
    """A node in the scene hierarchy."""
    name: str
    transform: List[List[float]] = None
    bounds: BoundingBox = None
    children: List['SceneNode'] = field(default_factory=list)
    meshes: List[str] = field(default_factory=list)  # Mesh names
    node_type: str = "group"

    def add_child(self, child: 'SceneNode'):
        self.children.append(child)

    def add_mesh(self, mesh_name: str):
        self.meshes.append(mesh_name)


@dataclass
class Scene:
    """Complete reconstructed scene."""
    name: str
    root: SceneNode
    meshes: Dict[str, SceneMesh]
    materials: Dict[int, dict]
    bounds: BoundingBox
    stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""

        def node_to_dict(node: SceneNode) -> dict:
            return {
                'name': node.name,
                'type': node.node_type,
                'meshes': node.meshes,
                'children': [node_to_dict(c) for c in node.children],
                'bounds': {
                    'min': node.bounds.min_point if node.bounds else None,
                    'max': node.bounds.max_point if node.bounds else None
                } if node.bounds else None
            }

        return {
            'name': self.name,
            'hierarchy': node_to_dict(self.root),
            'meshes': {
                name: {
                    'filepath': m.filepath,
                    'event_id': m.event_id,
                    'vertices': m.vertex_count,
                    'faces': m.face_count,
                    'category': m.category.value,
                    'lod': m.lod_level,
                    'bounds': {
                        'min': m.bounds.min_point,
                        'max': m.bounds.max_point
                    }
                }
                for name, m in self.meshes.items()
            },
            'stats': self.stats
        }

    def save(self, filepath: str):
        """Save scene to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class SceneBuilder:
    """
    Builds scene hierarchy from extracted meshes.

    Strategies:
        1. Spatial clustering - group nearby meshes
        2. Material grouping - meshes with same material
        3. LOD detection - meshes with similar geometry at same location
        4. Instance detection - identical meshes at different locations
    """

    def __init__(self, scene_name: str = "captured_scene"):
        self.scene_name = scene_name
        self.meshes: Dict[str, SceneMesh] = {}
        self.materials: Dict[int, dict] = {}
        self.geometry_hashes: Dict[str, List[str]] = {}  # hash -> mesh names

    def add_mesh(self, name: str, filepath: str, event_id: int,
                 vertex_count: int, face_count: int,
                 bounds_min: List[float], bounds_max: List[float],
                 geometry_hash: str = "", material_id: int = -1):
        """Add a mesh to the scene."""
        bounds = BoundingBox(
            min_point=bounds_min.copy(),
            max_point=bounds_max.copy()
        )

        mesh = SceneMesh(
            name=name,
            filepath=filepath,
            event_id=event_id,
            vertex_count=vertex_count,
            face_count=face_count,
            bounds=bounds,
            geometry_hash=geometry_hash,
            material_id=material_id
        )

        self.meshes[name] = mesh

        # Track by geometry hash for instance detection
        if geometry_hash:
            if geometry_hash not in self.geometry_hashes:
                self.geometry_hashes[geometry_hash] = []
            self.geometry_hashes[geometry_hash].append(name)

    def add_material(self, material_id: int, properties: dict):
        """Add material information."""
        self.materials[material_id] = properties

    def build(self, clustering_threshold: float = 500.0,
              lod_distance_threshold: float = 50.0) -> Scene:
        """
        Build the scene hierarchy.

        Args:
            clustering_threshold: Distance threshold for spatial clustering
            lod_distance_threshold: Distance threshold for LOD grouping

        Returns:
            Reconstructed Scene
        """
        # Step 1: Detect instances
        self._detect_instances()

        # Step 2: Detect LODs
        self._detect_lods(lod_distance_threshold)

        # Step 3: Categorize meshes
        self._categorize_meshes()

        # Step 4: Spatial clustering
        clusters = self._spatial_clustering(clustering_threshold)

        # Step 5: Build hierarchy
        root = self._build_hierarchy(clusters)

        # Calculate scene bounds
        scene_bounds = BoundingBox(
            min_point=[float('inf')] * 3,
            max_point=[float('-inf')] * 3
        )
        for mesh in self.meshes.values():
            scene_bounds.merge(mesh.bounds)

        # Stats
        stats = {
            'total_meshes': len(self.meshes),
            'unique_geometries': len(self.geometry_hashes),
            'instance_groups': sum(1 for h, m in self.geometry_hashes.items() if len(m) > 1),
            'clusters': len(clusters),
            'materials': len(self.materials)
        }

        return Scene(
            name=self.scene_name,
            root=root,
            meshes=self.meshes,
            materials=self.materials,
            bounds=scene_bounds,
            stats=stats
        )

    def _detect_instances(self):
        """Detect instanced meshes (same geometry, different locations)."""
        instance_group = 0

        for geom_hash, mesh_names in self.geometry_hashes.items():
            if len(mesh_names) > 1:
                # This is an instanced group
                for name in mesh_names:
                    self.meshes[name].instance_group = instance_group
                    self.meshes[name].category = MeshCategory.INSTANCED
                instance_group += 1

    def _detect_lods(self, distance_threshold: float):
        """Detect LOD chains (similar position, decreasing detail)."""
        # Group meshes by approximate position
        position_groups: Dict[Tuple[int, int, int], List[str]] = {}
        cell_size = distance_threshold

        for name, mesh in self.meshes.items():
            center = mesh.bounds.center
            cell = (
                int(center[0] / cell_size),
                int(center[1] / cell_size),
                int(center[2] / cell_size)
            )
            if cell not in position_groups:
                position_groups[cell] = []
            position_groups[cell].append(name)

        # Within each position group, detect LODs
        for cell, names in position_groups.items():
            if len(names) <= 1:
                continue

            # Sort by vertex count (highest first)
            sorted_meshes = sorted(
                names,
                key=lambda n: self.meshes[n].vertex_count,
                reverse=True
            )

            # Check if they're at very similar positions (LOD candidates)
            base_center = self.meshes[sorted_meshes[0]].bounds.center

            lod_chain = []
            for name in sorted_meshes:
                mesh = self.meshes[name]
                dist = math.sqrt(sum(
                    (mesh.bounds.center[i] - base_center[i])**2
                    for i in range(3)
                ))

                if dist < distance_threshold:
                    lod_chain.append(name)

            # If we have multiple meshes at same position with different vertex counts
            if len(lod_chain) > 1:
                # Check if vertex counts are significantly different (LOD-like)
                counts = [self.meshes[n].vertex_count for n in lod_chain]
                if max(counts) > min(counts) * 1.5:  # At least 50% difference
                    for i, name in enumerate(lod_chain):
                        self.meshes[name].lod_level = i

    def _categorize_meshes(self):
        """Categorize meshes based on properties."""
        for name, mesh in self.meshes.items():
            if mesh.category != MeshCategory.UNKNOWN:
                continue  # Already categorized

            # Large meshes with few instances are likely terrain
            if mesh.vertex_count > 50000 and mesh.instance_group < 0:
                mesh.category = MeshCategory.TERRAIN
                continue

            # Small meshes with many instances might be foliage
            if mesh.vertex_count < 1000 and mesh.instance_group >= 0:
                instances = sum(
                    1 for m in self.meshes.values()
                    if m.instance_group == mesh.instance_group
                )
                if instances > 10:
                    mesh.category = MeshCategory.FOLIAGE
                    continue

            # Default to static
            mesh.category = MeshCategory.STATIC

    def _spatial_clustering(self, threshold: float) -> List[List[str]]:
        """Cluster meshes spatially using simple distance-based clustering."""
        if not self.meshes:
            return []

        # Simple clustering: merge nearby meshes
        clusters: List[Set[str]] = []
        assigned: Set[str] = set()

        mesh_list = list(self.meshes.keys())

        for name in mesh_list:
            if name in assigned:
                continue

            # Start new cluster
            cluster = {name}
            assigned.add(name)
            mesh = self.meshes[name]

            # Find nearby meshes
            for other_name in mesh_list:
                if other_name in assigned:
                    continue

                other = self.meshes[other_name]
                dist = mesh.bounds.distance_to(other.bounds)

                if dist < threshold:
                    cluster.add(other_name)
                    assigned.add(other_name)

            clusters.append(cluster)

        return [list(c) for c in clusters]

    def _build_hierarchy(self, clusters: List[List[str]]) -> SceneNode:
        """Build node hierarchy from clusters."""
        root = SceneNode(name=self.scene_name, node_type="root")

        # Create category groups
        category_nodes: Dict[MeshCategory, SceneNode] = {}

        for category in MeshCategory:
            if category != MeshCategory.UNKNOWN:
                node = SceneNode(name=category.value, node_type="category")
                category_nodes[category] = node

        # Assign meshes to category nodes
        for cluster_idx, cluster in enumerate(clusters):
            # Determine dominant category for cluster
            categories = [self.meshes[n].category for n in cluster]
            dominant = max(set(categories), key=categories.count)

            if len(cluster) == 1:
                # Single mesh - add directly to category
                mesh_name = cluster[0]
                mesh = self.meshes[mesh_name]
                category_nodes[mesh.category].add_mesh(mesh_name)
            else:
                # Multiple meshes - create cluster node
                cluster_node = SceneNode(
                    name=f"cluster_{cluster_idx:03d}",
                    node_type="cluster"
                )

                # Calculate cluster bounds
                cluster_bounds = BoundingBox(
                    min_point=[float('inf')] * 3,
                    max_point=[float('-inf')] * 3
                )
                for mesh_name in cluster:
                    mesh = self.meshes[mesh_name]
                    cluster_bounds.merge(mesh.bounds)
                    cluster_node.add_mesh(mesh_name)
                    mesh.parent_node = cluster_node.name

                cluster_node.bounds = cluster_bounds
                category_nodes[dominant].add_child(cluster_node)

        # Add non-empty category nodes to root
        for category, node in category_nodes.items():
            if node.meshes or node.children:
                # Calculate category bounds
                bounds = BoundingBox(
                    min_point=[float('inf')] * 3,
                    max_point=[float('-inf')] * 3
                )
                for mesh_name in node.meshes:
                    bounds.merge(self.meshes[mesh_name].bounds)
                for child in node.children:
                    if child.bounds:
                        bounds.merge(child.bounds)
                node.bounds = bounds
                root.add_child(node)

        return root


def reconstruct_scene(mesh_index_path: str, output_path: str = None) -> Scene:
    """
    Reconstruct scene from mesh index JSON file.

    Args:
        mesh_index_path: Path to mesh_index.json from core_process.py
        output_path: Optional path to save scene JSON

    Returns:
        Reconstructed Scene
    """
    with open(mesh_index_path, 'r') as f:
        index = json.load(f)

    scene_name = Path(mesh_index_path).parent.name
    builder = SceneBuilder(scene_name)

    for mesh_info in index.get('meshes', []):
        builder.add_mesh(
            name=mesh_info['name'],
            filepath=mesh_info['path'],
            event_id=mesh_info.get('event_id', 0),
            vertex_count=mesh_info['vertices'],
            face_count=mesh_info['faces'],
            bounds_min=mesh_info['bounds']['min'],
            bounds_max=mesh_info['bounds']['max'],
            geometry_hash=mesh_info.get('hash', '')
        )

    scene = builder.build()

    if output_path:
        scene.save(output_path)
        print(f"Scene saved: {output_path}")
        print(f"  Meshes: {scene.stats['total_meshes']}")
        print(f"  Unique geometries: {scene.stats['unique_geometries']}")
        print(f"  Clusters: {scene.stats['clusters']}")

    return scene


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reconstruct scene hierarchy")
    parser.add_argument('mesh_index', help="Path to mesh_index.json")
    parser.add_argument('--output', '-o', help="Output scene JSON path")

    args = parser.parse_args()

    output = args.output or args.mesh_index.replace('mesh_index.json', 'scene.json')
    reconstruct_scene(args.mesh_index, output)
