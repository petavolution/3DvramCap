#!/usr/bin/env python3
"""
Core Camera and Projection Utilities
=====================================
Camera matrix extraction, coordinate system transforms, and projection utilities.

Key Transformations:
    - NDC (Normalized Device Coordinates) -> World Space
    - Clip Space -> World Space (via inverse view-projection)
    - UE4 coordinate system -> Blender/glTF coordinate system
    - Screen UV -> World Ray for texture projection

Coordinate Systems:
    UE4:     Left-handed, Z-up, X-forward
    Blender: Right-handed, Z-up, Y-forward
    glTF:    Right-handed, Y-up, Z-forward
    OpenGL:  Right-handed, Y-up, -Z-forward

Author: Capture Pipeline
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class CameraParams:
    """Camera parameters extracted from capture."""
    # View matrix (world -> view space)
    view_matrix: List[List[float]]
    # Projection matrix (view -> clip space)
    projection_matrix: List[List[float]]
    # Combined view-projection matrix
    view_projection: List[List[float]]
    # Inverse matrices for unprojection
    inverse_view: List[List[float]]
    inverse_projection: List[List[float]]
    inverse_view_projection: List[List[float]]
    # Camera position in world space
    position: List[float]
    # Camera forward direction
    forward: List[float]
    # Field of view (degrees)
    fov_y: float
    # Aspect ratio
    aspect: float
    # Near/far planes
    near: float
    far: float


def mat4_identity() -> List[List[float]]:
    """Create 4x4 identity matrix."""
    return [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]


def mat4_multiply(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    """Multiply two 4x4 matrices."""
    if HAS_NUMPY:
        return np.matmul(a, b).tolist()

    result = [[0]*4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            for k in range(4):
                result[i][j] += a[i][k] * b[k][j]
    return result


def mat4_inverse(m: List[List[float]]) -> Optional[List[List[float]]]:
    """Compute inverse of 4x4 matrix."""
    if HAS_NUMPY:
        try:
            return np.linalg.inv(m).tolist()
        except np.linalg.LinAlgError:
            return None

    # Manual 4x4 matrix inversion (Gaussian elimination)
    # This is a fallback - numpy is much more reliable
    n = 4
    augmented = [row[:] + [1 if i == j else 0 for j in range(n)]
                 for i, row in enumerate(m)]

    for col in range(n):
        # Find pivot
        max_row = col
        for row in range(col + 1, n):
            if abs(augmented[row][col]) > abs(augmented[max_row][col]):
                max_row = row
        augmented[col], augmented[max_row] = augmented[max_row], augmented[col]

        if abs(augmented[col][col]) < 1e-10:
            return None  # Singular matrix

        # Scale pivot row
        pivot = augmented[col][col]
        for j in range(2 * n):
            augmented[col][j] /= pivot

        # Eliminate column
        for row in range(n):
            if row != col:
                factor = augmented[row][col]
                for j in range(2 * n):
                    augmented[row][j] -= factor * augmented[col][j]

    # Extract inverse
    return [row[n:] for row in augmented]


def mat4_transpose(m: List[List[float]]) -> List[List[float]]:
    """Transpose 4x4 matrix."""
    return [[m[j][i] for j in range(4)] for i in range(4)]


def vec4_transform(m: List[List[float]], v: List[float]) -> List[float]:
    """Transform vec4 by 4x4 matrix."""
    result = [0, 0, 0, 0]
    for i in range(4):
        for j in range(4):
            result[i] += m[i][j] * v[j]
    return result


def vec3_normalize(v: List[float]) -> List[float]:
    """Normalize 3D vector."""
    length = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    if length < 1e-10:
        return [0, 0, 0]
    return [v[0]/length, v[1]/length, v[2]/length]


def vec3_cross(a: List[float], b: List[float]) -> List[float]:
    """Cross product of two 3D vectors."""
    return [
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ]


def vec3_dot(a: List[float], b: List[float]) -> float:
    """Dot product of two 3D vectors."""
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


# =============================================================================
# Coordinate System Conversions
# =============================================================================

def ue4_to_blender_position(pos: List[float]) -> List[float]:
    """
    Convert position from UE4 to Blender coordinate system.

    UE4: Left-handed, Z-up, X-forward, units in cm
    Blender: Right-handed, Z-up, Y-forward, units in m

    Transform: (X, Y, Z) -> (X, -Y, Z) * 0.01
    """
    scale = 0.01  # cm to m
    return [
        pos[0] * scale,
        -pos[1] * scale,  # Flip Y for handedness
        pos[2] * scale
    ]


def ue4_to_gltf_position(pos: List[float]) -> List[float]:
    """
    Convert position from UE4 to glTF coordinate system.

    UE4: Left-handed, Z-up, X-forward
    glTF: Right-handed, Y-up, Z-forward

    Transform: (X, Y, Z) -> (X, Z, -Y) * 0.01
    """
    scale = 0.01
    return [
        pos[0] * scale,
        pos[2] * scale,   # Z -> Y (up axis swap)
        -pos[1] * scale   # -Y -> Z (forward + handedness)
    ]


def blender_to_gltf_position(pos: List[float]) -> List[float]:
    """
    Convert position from Blender to glTF coordinate system.

    Blender: Right-handed, Z-up
    glTF: Right-handed, Y-up

    Transform: (X, Y, Z) -> (X, Z, -Y)
    """
    return [pos[0], pos[2], -pos[1]]


def ue4_to_blender_matrix(m: List[List[float]]) -> List[List[float]]:
    """Convert UE4 4x4 matrix to Blender coordinate system."""
    # Coordinate conversion matrix (Y flip + scale)
    scale = 0.01
    conv = [
        [scale, 0, 0, 0],
        [0, -scale, 0, 0],
        [0, 0, scale, 0],
        [0, 0, 0, 1]
    ]
    conv_inv = [
        [1/scale, 0, 0, 0],
        [0, -1/scale, 0, 0],
        [0, 0, 1/scale, 0],
        [0, 0, 0, 1]
    ]
    return mat4_multiply(conv, mat4_multiply(m, conv_inv))


# =============================================================================
# Projection Utilities
# =============================================================================

def create_perspective_matrix(fov_y: float, aspect: float,
                               near: float, far: float) -> List[List[float]]:
    """
    Create perspective projection matrix.

    Args:
        fov_y: Vertical field of view in degrees
        aspect: Width / height
        near: Near clip plane
        far: Far clip plane

    Returns:
        4x4 perspective matrix (OpenGL convention)
    """
    f = 1.0 / math.tan(math.radians(fov_y) / 2.0)

    return [
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), -1],
        [0, 0, (2 * far * near) / (near - far), 0]
    ]


def extract_frustum_planes(view_proj: List[List[float]]) -> List[Tuple[List[float], float]]:
    """
    Extract frustum planes from view-projection matrix.

    Uses Gribb/Hartmann method for direct plane extraction.

    Returns:
        List of (normal, distance) tuples for left, right, bottom, top, near, far planes
    """
    m = view_proj
    planes = []

    # Left plane
    planes.append(([
        m[0][3] + m[0][0],
        m[1][3] + m[1][0],
        m[2][3] + m[2][0]
    ], m[3][3] + m[3][0]))

    # Right plane
    planes.append(([
        m[0][3] - m[0][0],
        m[1][3] - m[1][0],
        m[2][3] - m[2][0]
    ], m[3][3] - m[3][0]))

    # Bottom plane
    planes.append(([
        m[0][3] + m[0][1],
        m[1][3] + m[1][1],
        m[2][3] + m[2][1]
    ], m[3][3] + m[3][1]))

    # Top plane
    planes.append(([
        m[0][3] - m[0][1],
        m[1][3] - m[1][1],
        m[2][3] - m[2][1]
    ], m[3][3] - m[3][1]))

    # Near plane
    planes.append(([
        m[0][3] + m[0][2],
        m[1][3] + m[1][2],
        m[2][3] + m[2][2]
    ], m[3][3] + m[3][2]))

    # Far plane
    planes.append(([
        m[0][3] - m[0][2],
        m[1][3] - m[1][2],
        m[2][3] - m[2][2]
    ], m[3][3] - m[3][2]))

    # Normalize planes
    normalized = []
    for normal, d in planes:
        length = math.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)
        if length > 1e-10:
            normalized.append((
                [normal[0]/length, normal[1]/length, normal[2]/length],
                d / length
            ))

    return normalized


def ndc_to_world(ndc: List[float], inverse_vp: List[List[float]]) -> List[float]:
    """
    Convert NDC coordinates to world space.

    Args:
        ndc: [x, y, z] in NDC space (-1 to 1 for x,y; 0 to 1 or -1 to 1 for z)
        inverse_vp: Inverse view-projection matrix

    Returns:
        [x, y, z] in world space
    """
    # Convert to clip space (w = 1)
    clip = [ndc[0], ndc[1], ndc[2], 1.0]

    # Transform by inverse VP
    world = vec4_transform(inverse_vp, clip)

    # Perspective divide
    if abs(world[3]) > 1e-10:
        return [world[0]/world[3], world[1]/world[3], world[2]/world[3]]

    return [world[0], world[1], world[2]]


def world_to_ndc(world: List[float], view_proj: List[List[float]]) -> List[float]:
    """
    Convert world coordinates to NDC.

    Args:
        world: [x, y, z] in world space
        view_proj: View-projection matrix

    Returns:
        [x, y, z] in NDC space
    """
    clip = vec4_transform(view_proj, [world[0], world[1], world[2], 1.0])

    if abs(clip[3]) > 1e-10:
        return [clip[0]/clip[3], clip[1]/clip[3], clip[2]/clip[3]]

    return [clip[0], clip[1], clip[2]]


def screen_to_world_ray(screen_uv: List[float], inverse_vp: List[List[float]],
                        camera_pos: List[float]) -> Tuple[List[float], List[float]]:
    """
    Convert screen UV to world space ray.

    Args:
        screen_uv: [u, v] in 0-1 range (top-left origin)
        inverse_vp: Inverse view-projection matrix
        camera_pos: Camera position in world space

    Returns:
        (origin, direction) ray in world space
    """
    # Convert UV to NDC
    ndc_x = screen_uv[0] * 2.0 - 1.0
    ndc_y = 1.0 - screen_uv[1] * 2.0  # Flip Y

    # Get near and far points in world space
    near_world = ndc_to_world([ndc_x, ndc_y, 0.0], inverse_vp)
    far_world = ndc_to_world([ndc_x, ndc_y, 1.0], inverse_vp)

    # Calculate direction
    direction = [
        far_world[0] - near_world[0],
        far_world[1] - near_world[1],
        far_world[2] - near_world[2]
    ]
    direction = vec3_normalize(direction)

    return camera_pos, direction


def project_texture_uv(world_pos: List[float], view_proj: List[List[float]],
                       texture_width: int, texture_height: int) -> Tuple[float, float]:
    """
    Project 3D world position to texture UV coordinates.

    Used for camera-projecting captured screenshots onto meshes.

    Args:
        world_pos: Position in world space
        view_proj: View-projection matrix used during capture
        texture_width: Width of captured texture
        texture_height: Height of captured texture

    Returns:
        (u, v) texture coordinates in 0-1 range
    """
    ndc = world_to_ndc(world_pos, view_proj)

    # NDC to UV (flip Y for texture coordinates)
    u = (ndc[0] + 1.0) * 0.5
    v = (1.0 - ndc[1]) * 0.5  # Flip Y

    return u, v


# =============================================================================
# Camera Extraction from Constant Buffers
# =============================================================================

def extract_camera_from_cbuffer(cbuffer_data: bytes,
                                 matrix_offset: int = 0) -> Optional[CameraParams]:
    """
    Extract camera parameters from a constant buffer.

    UE4 typically stores View, Projection, and ViewProjection matrices
    in constant buffers. Common layouts:

    SceneView CBV:
        float4x4 ViewMatrix        (offset 0)
        float4x4 ProjectionMatrix  (offset 64)
        float4x4 ViewProjection    (offset 128)
        float3   CameraPosition    (offset 192)

    Args:
        cbuffer_data: Raw constant buffer bytes
        matrix_offset: Byte offset to first matrix

    Returns:
        CameraParams or None if extraction fails
    """
    import struct

    if len(cbuffer_data) < matrix_offset + 208:  # Need at least 3 matrices + position
        return None

    try:
        def read_matrix(data, offset):
            values = struct.unpack_from('16f', data, offset)
            return [
                list(values[0:4]),
                list(values[4:8]),
                list(values[8:12]),
                list(values[12:16])
            ]

        view = read_matrix(cbuffer_data, matrix_offset)
        proj = read_matrix(cbuffer_data, matrix_offset + 64)
        view_proj = read_matrix(cbuffer_data, matrix_offset + 128)

        # Camera position (may be stored after matrices)
        pos_offset = matrix_offset + 192
        if len(cbuffer_data) >= pos_offset + 12:
            cam_pos = list(struct.unpack_from('3f', cbuffer_data, pos_offset))
        else:
            # Extract from inverse view matrix
            inv_view = mat4_inverse(view)
            cam_pos = [inv_view[0][3], inv_view[1][3], inv_view[2][3]] if inv_view else [0, 0, 0]

        # Compute inverses
        inv_view = mat4_inverse(view)
        inv_proj = mat4_inverse(proj)
        inv_vp = mat4_inverse(view_proj)

        # Extract FOV and aspect from projection matrix
        # proj[1][1] = 1 / tan(fov_y/2)
        # proj[0][0] = proj[1][1] / aspect
        fov_y = 2.0 * math.degrees(math.atan(1.0 / proj[1][1])) if proj[1][1] != 0 else 90.0
        aspect = proj[1][1] / proj[0][0] if proj[0][0] != 0 else 16/9

        # Near/far from projection
        # For perspective: proj[2][2] = -(f+n)/(f-n), proj[3][2] = -2fn/(f-n)
        near = 10.0  # Default
        far = 100000.0

        # Forward direction from view matrix (negative Z axis of view)
        forward = [-view[0][2], -view[1][2], -view[2][2]]
        forward = vec3_normalize(forward)

        return CameraParams(
            view_matrix=view,
            projection_matrix=proj,
            view_projection=view_proj,
            inverse_view=inv_view,
            inverse_projection=inv_proj,
            inverse_view_projection=inv_vp,
            position=cam_pos,
            forward=forward,
            fov_y=fov_y,
            aspect=aspect,
            near=near,
            far=far
        )

    except struct.error:
        return None


def estimate_camera_from_vertices(vertices: List[List[float]],
                                   screen_width: int,
                                   screen_height: int) -> Optional[CameraParams]:
    """
    Estimate camera parameters from post-VS vertex positions.

    When we have clip-space vertices (before perspective divide),
    we can estimate the camera frustum from the vertex distribution.

    This is useful when constant buffers aren't accessible.
    """
    if not HAS_NUMPY or len(vertices) < 10:
        return None

    verts = np.array(vertices)

    # Assume vertices with W != 0 are valid
    valid = verts[:, 3] != 0
    if not np.any(valid):
        return None

    # Perspective divide to get NDC
    ndc = verts[valid, :3] / verts[valid, 3:4]

    # Estimate bounds
    ndc_min = ndc.min(axis=0)
    ndc_max = ndc.max(axis=0)

    # Create approximate inverse VP (identity as fallback)
    # This is very rough without actual matrix data
    inv_vp = mat4_identity()

    return CameraParams(
        view_matrix=mat4_identity(),
        projection_matrix=mat4_identity(),
        view_projection=mat4_identity(),
        inverse_view=mat4_identity(),
        inverse_projection=mat4_identity(),
        inverse_view_projection=inv_vp,
        position=[0, 0, 0],
        forward=[0, 0, -1],
        fov_y=90.0,
        aspect=screen_width / screen_height,
        near=0.1,
        far=10000.0
    )
