#!/usr/bin/env python3
"""
ReShade Depth Capture and 3D Reconstruction
============================================
Reconstructs geometry from ReShade depth buffer captures when RenderDoc
extraction is insufficient (occluded areas, depth-only reconstruction).

Usage:
    python 06_depth_reconstruction.py \
        --color captures/color.png \
        --depth captures/depth.exr \
        --fov 60 \
        --out export/depth_mesh.ply

Features:
    - Depth map to point cloud conversion
    - Surface normal estimation from depth
    - Poisson surface reconstruction
    - Multi-view TSDF fusion for complete scenes
    - ICP registration for aligning multiple captures

Dependencies:
    pip install open3d numpy opencv-python

Key Concepts:
    - ReShade exports 32-bit EXR depth for full precision
    - Camera intrinsics derived from FOV and resolution
    - Point-to-plane ICP for robust alignment
    - TSDF fusion for watertight mesh reconstruction

Author: Capture Pipeline
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    import numpy as np
    import cv2
except ImportError:
    print("ERROR: numpy and opencv-python required")
    print("Install: pip install numpy opencv-python")
    sys.exit(1)

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("WARNING: open3d not installed. Advanced reconstruction unavailable.")
    print("Install: pip install open3d")


def load_depth_exr(path):
    """
    Load 32-bit EXR depth map from ReShade Frame Capture.

    ReShade's depth is typically linearized but may need inversion
    depending on game's depth buffer configuration.
    """
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if depth is None:
        raise ValueError(f"Could not load depth: {path}")

    # Handle multi-channel EXR (take first channel)
    if len(depth.shape) == 3:
        depth = depth[:, :, 0]

    return depth.astype(np.float32)


def load_color_image(path):
    """Load color image and convert to RGB."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not load color: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def compute_camera_intrinsics(width, height, fov_deg=60.0):
    """
    Compute camera intrinsic matrix from FOV and resolution.

    For games with ~90° horizontal FOV:
        fx = width / (2 * tan(fov/2))

    Returns:
        K: 3x3 intrinsic matrix
        fx, fy, cx, cy: individual parameters
    """
    fov_rad = np.radians(fov_deg)
    fx = width / (2 * np.tan(fov_rad / 2))
    fy = fx  # Assume square pixels

    cx = width / 2.0
    cy = height / 2.0

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)

    return K, fx, fy, cx, cy


def depth_to_point_cloud(depth, color, fx, fy, cx, cy, depth_scale=1.0, max_depth=100.0):
    """
    Convert depth map to colored point cloud.

    Mathematical reconstruction:
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        Z = depth_value * depth_scale

    Args:
        depth: HxW depth map (meters or game units)
        color: HxWx3 RGB image
        fx, fy: Focal lengths
        cx, cy: Principal point
        depth_scale: Scale factor to convert to meters
        max_depth: Maximum depth to include (filters skybox)

    Returns:
        points: Nx3 array of XYZ positions
        colors: Nx3 array of RGB colors [0-1]
    """
    h, w = depth.shape

    # Create pixel coordinate grids
    u = np.arange(w)
    v = np.arange(h)
    u, v = np.meshgrid(u, v)

    # Scale depth
    z = depth * depth_scale

    # Compute 3D coordinates
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # Stack into point array
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    # Get corresponding colors
    colors = color.reshape(-1, 3).astype(np.float32) / 255.0

    # Filter invalid points
    valid = (z.flatten() > 0.01) & (z.flatten() < max_depth)
    points = points[valid]
    colors = colors[valid]

    return points, colors


def estimate_normals_from_depth(depth, fx, fy):
    """
    Estimate surface normals from depth map using finite differences.

    Uses 3-tap method for speed:
        ddx = [1, 0, dZ/dx]
        ddy = [0, 1, dZ/dy]
        normal = normalize(cross(ddx, ddy))
    """
    # Compute depth gradients
    dzdx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3) / (8.0 * fx)
    dzdy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3) / (8.0 * fy)

    # Construct normal vectors
    h, w = depth.shape
    normals = np.zeros((h, w, 3), dtype=np.float32)

    normals[:, :, 0] = -dzdx
    normals[:, :, 1] = -dzdy
    normals[:, :, 2] = 1.0

    # Normalize
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    norm[norm < 1e-6] = 1.0
    normals = normals / norm

    return normals


def create_open3d_point_cloud(points, colors, normals=None):
    """Create Open3D point cloud from numpy arrays."""
    if not HAS_OPEN3D:
        raise RuntimeError("Open3D required for point cloud creation")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    if normals is not None:
        # Flatten normals to match points
        n = normals.reshape(-1, 3)
        # Filter to match valid points
        if len(n) != len(points):
            n = n[:len(points)]
        pcd.normals = o3d.utility.Vector3dVector(n)
    else:
        # Estimate normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )

    return pcd


def create_rgbd_image(color_path, depth_path, depth_scale=1.0, depth_trunc=100.0):
    """
    Create Open3D RGBD image for TSDF integration.

    Args:
        color_path: Path to color image
        depth_path: Path to depth EXR
        depth_scale: Depth value multiplier
        depth_trunc: Maximum depth to include
    """
    if not HAS_OPEN3D:
        raise RuntimeError("Open3D required for RGBD images")

    color = o3d.io.read_image(color_path)

    # Load depth manually and convert
    depth_np = load_depth_exr(depth_path) * depth_scale

    # Convert to Open3D depth image
    depth_o3d = o3d.geometry.Image(depth_np.astype(np.float32))

    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth_o3d,
        depth_scale=1.0,  # Already scaled
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False
    )


def register_point_clouds_icp(source, target, initial_transform=None, threshold=0.05):
    """
    Align two point clouds using point-to-plane ICP.

    Point-to-plane ICP converges 4-10x faster than point-to-point
    for structured environments with planar surfaces.

    Args:
        source: Source point cloud to transform
        target: Target point cloud (reference)
        initial_transform: Optional 4x4 initial guess
        threshold: Distance threshold for correspondences

    Returns:
        transformation: 4x4 transformation matrix
        fitness: Alignment quality (0-1)
    """
    if not HAS_OPEN3D:
        raise RuntimeError("Open3D required for ICP registration")

    if initial_transform is None:
        initial_transform = np.eye(4)

    # Ensure normals exist
    if not source.has_normals():
        source.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
    if not target.has_normals():
        target.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )

    # Point-to-plane ICP
    result = o3d.pipelines.registration.registration_icp(
        source, target, threshold, initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
    )

    return result.transformation, result.fitness


def global_registration_fpfh(source, target, voxel_size=0.05):
    """
    Global registration using Fast Point Feature Histograms (FPFH).

    Use this for initial alignment when captures have unknown relative positions.

    Returns:
        transformation: 4x4 transformation matrix for initial alignment
    """
    if not HAS_OPEN3D:
        raise RuntimeError("Open3D required for FPFH registration")

    def preprocess(pcd, voxel_size):
        # Downsample
        pcd_down = pcd.voxel_down_sample(voxel_size)
        # Estimate normals
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
        )
        # Compute FPFH features
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
        )
        return pcd_down, fpfh

    source_down, source_fpfh = preprocess(source, voxel_size)
    target_down, target_fpfh = preprocess(target, voxel_size)

    # RANSAC registration
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=voxel_size * 1.5,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.5)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )

    return result.transformation


def tsdf_fusion(rgbd_images, intrinsics, poses, voxel_size=0.01, sdf_trunc=0.04):
    """
    Fuse multiple RGBD images into watertight mesh using TSDF.

    Truncated Signed Distance Function (TSDF) volumetric fusion:
    - Accumulates depth observations in 3D voxel grid
    - Weights contributions by view quality
    - Extracts isosurface via marching cubes

    Args:
        rgbd_images: List of RGBD images
        intrinsics: Camera intrinsic matrix
        poses: List of 4x4 camera poses (camera-to-world)
        voxel_size: Voxel size in meters
        sdf_trunc: Truncation distance

    Returns:
        mesh: Reconstructed triangle mesh
    """
    if not HAS_OPEN3D:
        raise RuntimeError("Open3D required for TSDF fusion")

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    # Create Open3D intrinsics
    h, w = rgbd_images[0].depth.shape if hasattr(rgbd_images[0], 'depth') else (1080, 1920)

    o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=w, height=h,
        fx=intrinsics[0, 0], fy=intrinsics[1, 1],
        cx=intrinsics[0, 2], cy=intrinsics[1, 2]
    )

    # Integrate each frame
    for i, (rgbd, pose) in enumerate(zip(rgbd_images, poses)):
        # TSDF expects world-to-camera, so invert camera-to-world
        extrinsic = np.linalg.inv(pose)
        volume.integrate(rgbd, o3d_intrinsics, extrinsic)
        print(f"  Integrated frame {i+1}/{len(rgbd_images)}")

    # Extract mesh
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    return mesh


def poisson_reconstruction(pcd, depth=9):
    """
    Poisson surface reconstruction for watertight mesh.

    Args:
        pcd: Point cloud with normals
        depth: Octree depth (higher = more detail, more memory)

    Returns:
        mesh: Reconstructed triangle mesh
    """
    if not HAS_OPEN3D:
        raise RuntimeError("Open3D required for Poisson reconstruction")

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )

    # Remove low-density vertices (likely artifacts)
    densities = np.asarray(densities)
    density_threshold = np.quantile(densities, 0.01)
    vertices_to_remove = densities < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)

    return mesh


def single_view_reconstruction(color_path, depth_path, output_path,
                                fov_deg=60.0, depth_scale=1.0, max_depth=100.0):
    """
    Reconstruct mesh from single color+depth capture.

    Workflow:
        1. Load color and depth
        2. Compute intrinsics from FOV
        3. Convert to point cloud
        4. Estimate normals
        5. Poisson surface reconstruction
        6. Save mesh
    """
    print(f"\n=== Single View Reconstruction ===")
    print(f"Color: {color_path}")
    print(f"Depth: {depth_path}")
    print(f"FOV: {fov_deg}°")

    # Load images
    color = load_color_image(color_path)
    depth = load_depth_exr(depth_path)

    h, w = depth.shape
    print(f"Resolution: {w}x{h}")

    # Compute intrinsics
    K, fx, fy, cx, cy = compute_camera_intrinsics(w, h, fov_deg)
    print(f"Intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")

    # Convert to point cloud
    print("Converting depth to point cloud...")
    points, colors = depth_to_point_cloud(
        depth, color, fx, fy, cx, cy,
        depth_scale=depth_scale, max_depth=max_depth
    )
    print(f"Points: {len(points):,}")

    if not HAS_OPEN3D:
        # Fallback: save as PLY manually
        print("Saving point cloud (Open3D not available for mesh reconstruction)...")
        save_ply_simple(output_path.replace('.ply', '_points.ply'), points, colors)
        return

    # Create Open3D point cloud
    pcd = create_open3d_point_cloud(points, colors)

    # Statistical outlier removal
    print("Removing outliers...")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"Points after outlier removal: {len(pcd.points):,}")

    # Poisson reconstruction
    print("Running Poisson surface reconstruction...")
    mesh = poisson_reconstruction(pcd, depth=9)
    print(f"Mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f"Saved: {output_path}")

    return mesh


def save_ply_simple(path, points, colors):
    """Simple PLY export without Open3D."""
    with open(path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for p, c in zip(points, colors):
            r, g, b = (c * 255).astype(np.uint8)
            f.write(f"{p[0]} {p[1]} {p[2]} {r} {g} {b}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct 3D geometry from ReShade depth captures"
    )
    parser.add_argument('--color', required=True, help="Path to color image")
    parser.add_argument('--depth', required=True, help="Path to depth EXR")
    parser.add_argument('--out', default='export/depth_mesh.ply', help="Output mesh path")
    parser.add_argument('--fov', type=float, default=60.0, help="Horizontal FOV in degrees")
    parser.add_argument('--depth-scale', type=float, default=1.0, help="Depth value scale")
    parser.add_argument('--max-depth', type=float, default=100.0, help="Maximum depth to include")

    args = parser.parse_args()

    single_view_reconstruction(
        args.color, args.depth, args.out,
        fov_deg=args.fov,
        depth_scale=args.depth_scale,
        max_depth=args.max_depth
    )

    print("\n[OK] Depth reconstruction complete!")


if __name__ == "__main__":
    main()
