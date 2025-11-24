#!/usr/bin/env python3
"""
Depth Buffer Integration
========================
Process depth buffers from ReShade for mesh reconstruction and camera projection.

Features:
    - Load depth maps from ReShade DisplayDepth shader
    - Depth linearization for various game engines
    - Point cloud generation from depth
    - Depth-based UV projection
    - Camera frustum reconstruction
    - Depth map filtering and cleanup

Usage:
    from core_depth import DepthProcessor, load_depth_map

    # Load and process depth map
    processor = DepthProcessor(depth_image, camera_params)
    point_cloud = processor.to_point_cloud()
    uvs = processor.project_to_depth(world_positions)

Author: Capture Pipeline
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    raise ImportError("numpy required for depth processing")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


@dataclass
class CameraParams:
    """Camera parameters for depth projection."""
    # Intrinsics
    fov_y: float = 90.0  # Vertical FOV in degrees
    aspect_ratio: float = 16 / 9
    near_plane: float = 10.0  # Near clip (game units)
    far_plane: float = 100000.0  # Far clip (game units)

    # Image dimensions
    width: int = 1920
    height: int = 1080

    # Extrinsics (optional)
    position: Tuple[float, float, float] = (0, 0, 0)
    rotation: Tuple[float, float, float] = (0, 0, 0)  # Euler angles

    # Depth format hints
    reversed_z: bool = False  # Some games use reversed-z depth
    logarithmic: bool = False  # Logarithmic depth buffer

    @property
    def fov_y_rad(self) -> float:
        return math.radians(self.fov_y)

    @property
    def fov_x_rad(self) -> float:
        return 2 * math.atan(math.tan(self.fov_y_rad / 2) * self.aspect_ratio)

    def focal_length_pixels(self) -> Tuple[float, float]:
        """Get focal length in pixels (fx, fy)."""
        fy = self.height / (2 * math.tan(self.fov_y_rad / 2))
        fx = self.width / (2 * math.tan(self.fov_x_rad / 2))
        return (fx, fy)


class DepthProcessor:
    """
    Process depth maps for 3D reconstruction.

    Handles depth map loading, linearization, and projection operations.
    """

    def __init__(self,
                 depth_data: Union[str, np.ndarray, 'Image.Image'],
                 camera: Optional[CameraParams] = None):
        """
        Initialize depth processor.

        Args:
            depth_data: Depth image path, numpy array, or PIL Image
            camera: Camera parameters (uses defaults if not provided)
        """
        self.camera = camera or CameraParams()
        self.raw_depth = self._load_depth(depth_data)
        self.linear_depth: Optional[np.ndarray] = None

    def _load_depth(self, depth_data) -> np.ndarray:
        """Load depth data from various sources."""
        if isinstance(depth_data, str):
            return self._load_depth_image(depth_data)
        elif isinstance(depth_data, np.ndarray):
            return depth_data.astype(np.float32)
        elif HAS_PIL and isinstance(depth_data, Image.Image):
            return np.array(depth_data.convert('L')).astype(np.float32) / 255.0
        else:
            raise TypeError(f"Unsupported depth data type: {type(depth_data)}")

    def _load_depth_image(self, filepath: str) -> np.ndarray:
        """Load depth from image file."""
        if not HAS_PIL:
            raise ImportError("PIL required to load depth images")

        img = Image.open(filepath)

        # Handle different formats
        if img.mode == 'I;16':
            # 16-bit depth
            depth = np.array(img).astype(np.float32) / 65535.0
        elif img.mode == 'F':
            # 32-bit float (EXR)
            depth = np.array(img)
        elif img.mode in ('L', 'LA'):
            # 8-bit grayscale
            depth = np.array(img.convert('L')).astype(np.float32) / 255.0
        elif img.mode == 'RGB':
            # RGB depth (ReShade often outputs this)
            # Usually R channel contains depth
            depth = np.array(img)[:, :, 0].astype(np.float32) / 255.0
        else:
            depth = np.array(img.convert('L')).astype(np.float32) / 255.0

        # Update camera dimensions
        self.camera.height, self.camera.width = depth.shape[:2]

        return depth

    def linearize(self, method: str = 'auto') -> np.ndarray:
        """
        Linearize depth buffer to world-space depth.

        Args:
            method: Linearization method:
                - 'auto': Auto-detect
                - 'linear': Already linear (0-1 = near-far)
                - 'perspective': Standard perspective projection
                - 'reversed_z': Reversed-z buffer
                - 'logarithmic': Logarithmic depth

        Returns:
            Linearized depth in world units
        """
        if method == 'auto':
            method = self._detect_depth_type()

        near = self.camera.near_plane
        far = self.camera.far_plane
        d = self.raw_depth

        if method == 'linear':
            # Already linear 0-1 -> near-far
            self.linear_depth = near + d * (far - near)

        elif method == 'perspective':
            # Standard NDC depth: d = (f*(z-n)) / (z*(f-n))
            # Solve for z: z = f*n / (f - d*(f-n))
            denom = far - d * (far - near)
            denom = np.maximum(denom, 1e-6)  # Prevent division by zero
            self.linear_depth = (far * near) / denom

        elif method == 'reversed_z':
            # Reversed-z: near=1, far=0
            d = 1.0 - d
            denom = far - d * (far - near)
            denom = np.maximum(denom, 1e-6)
            self.linear_depth = (far * near) / denom

        elif method == 'logarithmic':
            # Logarithmic: d = log(z/n) / log(f/n)
            # Solve for z: z = n * (f/n)^d
            ratio = far / near
            self.linear_depth = near * np.power(ratio, d)

        else:
            raise ValueError(f"Unknown linearization method: {method}")

        return self.linear_depth

    def _detect_depth_type(self) -> str:
        """Auto-detect depth buffer type from statistics."""
        d = self.raw_depth

        # Check value distribution
        near_vals = d[d < 0.1]
        far_vals = d[d > 0.9]

        # Reversed-z typically has more values near 1 (close objects)
        if len(near_vals) > len(far_vals) * 2:
            return 'reversed_z'

        # Check for logarithmic distribution
        hist, _ = np.histogram(d.flatten(), bins=10)
        if hist[0] > sum(hist[1:]) / 2:
            return 'logarithmic'

        return 'perspective'

    def to_point_cloud(self, subsample: int = 1) -> np.ndarray:
        """
        Convert depth map to 3D point cloud.

        Args:
            subsample: Subsample factor (1 = full resolution)

        Returns:
            Nx3 array of world-space points
        """
        if self.linear_depth is None:
            self.linearize()

        depth = self.linear_depth
        h, w = depth.shape

        # Create pixel coordinate grids
        u = np.arange(0, w, subsample)
        v = np.arange(0, h, subsample)
        u, v = np.meshgrid(u, v)

        # Get depth values
        z = depth[::subsample, ::subsample]

        # Mask invalid depths
        valid = (z > self.camera.near_plane) & (z < self.camera.far_plane)

        # Convert to camera coordinates
        fx, fy = self.camera.focal_length_pixels()
        cx, cy = w / 2, h / 2

        # x = (u - cx) * z / fx
        # y = (v - cy) * z / fy
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        # Stack and filter
        points = np.stack([x, y, z], axis=-1)
        points = points[valid]

        return points

    def project_points(self, world_points: np.ndarray) -> np.ndarray:
        """
        Project 3D world points to depth map UV coordinates.

        Args:
            world_points: Nx3 array of world positions

        Returns:
            Nx2 array of UV coordinates (0-1 range)
        """
        fx, fy = self.camera.focal_length_pixels()
        cx, cy = self.camera.width / 2, self.camera.height / 2

        # Assume points are in camera space (transform if needed)
        x, y, z = world_points[:, 0], world_points[:, 1], world_points[:, 2]

        # Prevent division by zero
        z = np.maximum(z, 1e-6)

        # Project to pixel coordinates
        u = (x * fx / z) + cx
        v = (y * fy / z) + cy

        # Convert to UV (0-1)
        uv = np.stack([
            u / self.camera.width,
            v / self.camera.height
        ], axis=-1)

        return np.clip(uv, 0, 1)

    def sample_depth_at_points(self, world_points: np.ndarray) -> np.ndarray:
        """
        Sample depth values at projected point locations.

        Useful for depth-based visibility checks.
        """
        if self.linear_depth is None:
            self.linearize()

        uvs = self.project_points(world_points)

        # Convert to pixel coordinates
        px = (uvs[:, 0] * (self.camera.width - 1)).astype(np.int32)
        py = (uvs[:, 1] * (self.camera.height - 1)).astype(np.int32)

        # Clamp
        px = np.clip(px, 0, self.camera.width - 1)
        py = np.clip(py, 0, self.camera.height - 1)

        return self.linear_depth[py, px]

    def compute_visibility(self,
                          world_points: np.ndarray,
                          tolerance: float = 10.0) -> np.ndarray:
        """
        Compute visibility of points based on depth comparison.

        Args:
            world_points: Nx3 world positions
            tolerance: Depth tolerance in world units

        Returns:
            Boolean array indicating visibility
        """
        if self.linear_depth is None:
            self.linearize()

        # Get point depths
        point_depths = world_points[:, 2]  # Assuming camera-space Z

        # Sample depth buffer
        buffer_depths = self.sample_depth_at_points(world_points)

        # Point is visible if its depth matches buffer depth (within tolerance)
        visible = np.abs(point_depths - buffer_depths) < tolerance

        return visible

    def generate_normals(self) -> np.ndarray:
        """
        Generate normal map from depth gradients.

        Returns:
            HxWx3 normal map in camera space
        """
        if self.linear_depth is None:
            self.linearize()

        # Compute gradients
        dz_dx = np.gradient(self.linear_depth, axis=1)
        dz_dy = np.gradient(self.linear_depth, axis=0)

        # Construct normals
        normals = np.stack([
            -dz_dx,
            -dz_dy,
            np.ones_like(dz_dx)
        ], axis=-1)

        # Normalize
        lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
        lengths = np.maximum(lengths, 1e-6)
        normals /= lengths

        return normals.astype(np.float32)

    def bilateral_filter(self,
                        spatial_sigma: float = 5.0,
                        range_sigma: float = 0.1) -> np.ndarray:
        """
        Apply bilateral filter to reduce noise while preserving edges.
        """
        if self.linear_depth is None:
            self.linearize()

        try:
            import cv2
            # OpenCV bilateral filter
            filtered = cv2.bilateralFilter(
                self.linear_depth.astype(np.float32),
                d=int(spatial_sigma * 2) | 1,  # Must be odd
                sigmaColor=range_sigma * (self.camera.far_plane - self.camera.near_plane),
                sigmaSpace=spatial_sigma
            )
            self.linear_depth = filtered
        except ImportError:
            # Fallback: simple Gaussian blur
            from scipy.ndimage import gaussian_filter
            self.linear_depth = gaussian_filter(self.linear_depth, sigma=spatial_sigma)

        return self.linear_depth

    def fill_holes(self, max_hole_size: int = 10) -> np.ndarray:
        """
        Fill small holes in depth map (invalid depth regions).
        """
        if self.linear_depth is None:
            self.linearize()

        # Find invalid regions (typically 0 or very far)
        invalid = (self.linear_depth < self.camera.near_plane) | \
                  (self.linear_depth >= self.camera.far_plane * 0.99)

        try:
            from scipy.ndimage import binary_dilation, distance_transform_edt

            # Only fill small holes
            struct = np.ones((3, 3))
            dilated = binary_dilation(~invalid, structure=struct, iterations=max_hole_size)
            small_holes = invalid & dilated

            # Fill using distance transform
            if np.any(small_holes):
                _, indices = distance_transform_edt(invalid, return_indices=True)
                self.linear_depth[small_holes] = self.linear_depth[
                    indices[0][small_holes],
                    indices[1][small_holes]
                ]

        except ImportError:
            pass  # Skip hole filling without scipy

        return self.linear_depth

    def save_linear_depth(self, filepath: str, normalize: bool = True):
        """Save linearized depth as image."""
        if self.linear_depth is None:
            self.linearize()

        if not HAS_PIL:
            raise ImportError("PIL required to save images")

        depth = self.linear_depth

        if normalize:
            # Normalize to 0-1 range
            d_min = self.camera.near_plane
            d_max = min(self.camera.far_plane, np.percentile(depth, 99))
            depth = (depth - d_min) / (d_max - d_min)
            depth = np.clip(depth, 0, 1)

        # Convert to 16-bit
        depth_16 = (depth * 65535).astype(np.uint16)

        img = Image.fromarray(depth_16, mode='I;16')
        img.save(filepath)
        print(f"Saved depth: {filepath}")


def load_depth_map(filepath: str,
                   camera: Optional[CameraParams] = None,
                   linearize: bool = True) -> DepthProcessor:
    """
    Load depth map from file.

    Args:
        filepath: Path to depth image
        camera: Camera parameters
        linearize: Auto-linearize depth

    Returns:
        DepthProcessor instance
    """
    processor = DepthProcessor(filepath, camera)

    if linearize:
        processor.linearize()

    return processor


def depth_to_mesh(depth_processor: DepthProcessor,
                  subsample: int = 4,
                  max_edge_length: float = 100.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert depth map to triangle mesh.

    Args:
        depth_processor: Depth processor with linearized depth
        subsample: Pixel subsample factor
        max_edge_length: Maximum triangle edge length (removes stretched triangles)

    Returns:
        (vertices Nx3, faces Mx3)
    """
    if depth_processor.linear_depth is None:
        depth_processor.linearize()

    depth = depth_processor.linear_depth
    h, w = depth.shape
    camera = depth_processor.camera

    # Subsample
    depth = depth[::subsample, ::subsample]
    sh, sw = depth.shape

    # Create vertex grid
    fx, fy = camera.focal_length_pixels()
    cx, cy = w / 2, h / 2

    u = np.arange(0, w, subsample).astype(np.float32)
    v = np.arange(0, h, subsample).astype(np.float32)
    uu, vv = np.meshgrid(u, v)

    # Back-project to 3D
    z = depth
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy

    vertices = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    # Create face indices
    faces = []
    for i in range(sh - 1):
        for j in range(sw - 1):
            v0 = i * sw + j
            v1 = v0 + 1
            v2 = v0 + sw
            v3 = v2 + 1

            # Get vertex positions
            p0, p1, p2, p3 = vertices[v0], vertices[v1], vertices[v2], vertices[v3]

            # Check edge lengths
            def edge_ok(a, b):
                return np.linalg.norm(a - b) < max_edge_length

            # Check validity (valid depth)
            near, far = camera.near_plane, camera.far_plane
            valid = [near < p[2] < far for p in [p0, p1, p2, p3]]

            if all(valid[:3]) and edge_ok(p0, p1) and edge_ok(p1, p2) and edge_ok(p0, p2):
                faces.append([v0, v1, v2])

            if all(valid[1:]) and edge_ok(p1, p3) and edge_ok(p3, p2) and edge_ok(p1, p2):
                faces.append([v1, v3, v2])

    return vertices, np.array(faces, dtype=np.int32)


def project_color_to_depth(color_image: Union[str, np.ndarray],
                           depth_processor: DepthProcessor,
                           world_points: np.ndarray) -> np.ndarray:
    """
    Project color image onto 3D points using depth-based projection.

    Returns vertex colors for each point.
    """
    if HAS_PIL and isinstance(color_image, str):
        img = Image.open(color_image)
        color = np.array(img)
    else:
        color = color_image

    h, w = color.shape[:2]

    # Project points to image UV
    uvs = depth_processor.project_points(world_points)

    # Sample colors
    px = (uvs[:, 0] * (w - 1)).astype(np.int32)
    py = (uvs[:, 1] * (h - 1)).astype(np.int32)

    px = np.clip(px, 0, w - 1)
    py = np.clip(py, 0, h - 1)

    if len(color.shape) == 3:
        vertex_colors = color[py, px, :3]
    else:
        vertex_colors = np.stack([color[py, px]] * 3, axis=-1)

    return vertex_colors


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process depth maps")
    parser.add_argument('input', help="Input depth image")
    parser.add_argument('--output', '-o', help="Output file")
    parser.add_argument('--fov', type=float, default=90, help="Vertical FOV")
    parser.add_argument('--near', type=float, default=10, help="Near plane")
    parser.add_argument('--far', type=float, default=100000, help="Far plane")
    parser.add_argument('--points', action='store_true', help="Export point cloud")
    parser.add_argument('--mesh', action='store_true', help="Export mesh")
    parser.add_argument('--subsample', type=int, default=4, help="Subsample factor")

    args = parser.parse_args()

    camera = CameraParams(
        fov_y=args.fov,
        near_plane=args.near,
        far_plane=args.far
    )

    processor = DepthProcessor(args.input, camera)
    processor.linearize()

    if args.points:
        points = processor.to_point_cloud(args.subsample)
        output = args.output or args.input.rsplit('.', 1)[0] + '_points.ply'

        # Save as PLY
        with open(output, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            for p in points:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")

        print(f"Saved {len(points)} points to {output}")

    elif args.mesh:
        vertices, faces = depth_to_mesh(processor, args.subsample)
        output = args.output or args.input.rsplit('.', 1)[0] + '_mesh.obj'

        with open(output, 'w') as f:
            f.write("# Depth mesh\n")
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

        print(f"Saved mesh: {len(vertices)} verts, {len(faces)} faces")

    else:
        # Just linearize and save
        output = args.output or args.input.rsplit('.', 1)[0] + '_linear.png'
        processor.save_linear_depth(output)
