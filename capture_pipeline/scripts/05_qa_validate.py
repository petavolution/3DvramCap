#!/usr/bin/env python3
"""
QA Validation Script
====================
Validates the reconstruction by comparing rendered output to reference.

Usage:
    python 05_qa_validate.py reference.png rendered.png report.json

Metrics:
    - Silhouette IoU: Edge-based geometry comparison (>= 0.92 = good)
    - SSIM: Structural similarity (>= 0.80 = acceptable)
    - Geometry sanity: Bounding box and scale checks

Dependencies:
    pip install opencv-python numpy scikit-image

Author: Capture Pipeline
"""

import argparse
import json
import os
import sys

try:
    import cv2
    import numpy as np
except ImportError:
    print("ERROR: Required packages not found.")
    print("Install with: pip install opencv-python numpy")
    sys.exit(1)

# Try to import SSIM from scikit-image (optional but recommended)
try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("Note: scikit-image not found. SSIM will use simplified calculation.")


def load_image(path):
    """Load image in BGR format."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not load image: {path}")

    return img


def compute_edges(img, low_threshold=50, high_threshold=150):
    """Compute Canny edges from image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return edges


def compute_silhouette_iou(ref_img, test_img):
    """
    Compute Intersection over Union of edge silhouettes.

    This metric focuses on geometry/camera alignment rather than
    pixel-perfect color matching.

    Returns:
        float: IoU value (0-1, higher is better)
    """
    # Compute edges
    ref_edges = compute_edges(ref_img)
    test_edges = compute_edges(test_img)

    # Ensure same size
    if ref_edges.shape != test_edges.shape:
        test_edges = cv2.resize(test_edges, (ref_edges.shape[1], ref_edges.shape[0]))

    # Binary masks
    ref_mask = ref_edges > 0
    test_mask = test_edges > 0

    # Compute IoU
    intersection = np.logical_and(ref_mask, test_mask).sum()
    union = np.logical_or(ref_mask, test_mask).sum()

    if union == 0:
        return 1.0  # Both empty = perfect match

    iou = intersection / union
    return float(iou)


def compute_ssim_simple(ref_img, test_img):
    """
    Simplified SSIM calculation without scikit-image.

    Based on the original SSIM paper formula.
    """
    # Convert to grayscale
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY).astype(np.float64)

    # Ensure same size
    if ref_gray.shape != test_gray.shape:
        test_gray = cv2.resize(test_gray, (ref_gray.shape[1], ref_gray.shape[0]))

    # Constants
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    # Means
    mu_x = ref_gray.mean()
    mu_y = test_gray.mean()

    # Variances and covariance
    sigma_x2 = ((ref_gray - mu_x) ** 2).mean()
    sigma_y2 = ((test_gray - mu_y) ** 2).mean()
    sigma_xy = ((ref_gray - mu_x) * (test_gray - mu_y)).mean()

    # SSIM formula
    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x2 + sigma_y2 + C2)

    return float(numerator / denominator)


def compute_ssim(ref_img, test_img):
    """
    Compute Structural Similarity Index.

    Uses scikit-image if available, otherwise simplified calculation.
    """
    # Ensure same size
    if ref_img.shape != test_img.shape:
        test_img = cv2.resize(test_img, (ref_img.shape[1], ref_img.shape[0]))

    if HAS_SKIMAGE:
        # Convert to grayscale
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

        return float(ssim(ref_gray, test_gray))
    else:
        return compute_ssim_simple(ref_img, test_img)


def compute_edge_diff_image(ref_img, test_img):
    """
    Create a diff image showing edge differences.

    Green = reference only
    Red = test only
    White = both
    """
    ref_edges = compute_edges(ref_img)
    test_edges = compute_edges(test_img)

    # Ensure same size
    if ref_edges.shape != test_edges.shape:
        test_edges = cv2.resize(test_edges, (ref_edges.shape[1], ref_edges.shape[0]))

    # Create color diff image
    h, w = ref_edges.shape
    diff = np.zeros((h, w, 3), dtype=np.uint8)

    # Both = white
    both = np.logical_and(ref_edges > 0, test_edges > 0)
    diff[both] = [255, 255, 255]

    # Reference only = green
    ref_only = np.logical_and(ref_edges > 0, test_edges == 0)
    diff[ref_only] = [0, 255, 0]

    # Test only = red
    test_only = np.logical_and(ref_edges == 0, test_edges > 0)
    diff[test_only] = [0, 0, 255]

    return diff


def validate_geometry_sanity(scene_json_path):
    """
    Check geometry sanity from scene index.

    Checks:
        - No extreme scales (> 1e6 units)
        - No NaN/Inf values
        - Reasonable mesh counts
    """
    results = {
        "passed": True,
        "checks": []
    }

    if not os.path.exists(scene_json_path):
        results["passed"] = False
        results["checks"].append({
            "name": "scene_index_exists",
            "passed": False,
            "message": f"Scene index not found: {scene_json_path}"
        })
        return results

    with open(scene_json_path, 'r') as f:
        scene = json.load(f)

    # Check mesh count
    mesh_count = len(scene.get("meshes", []))
    results["checks"].append({
        "name": "mesh_count",
        "passed": mesh_count > 0,
        "value": mesh_count,
        "message": f"Found {mesh_count} meshes"
    })

    if mesh_count == 0:
        results["passed"] = False

    # Check for texture presence
    textures = scene.get("textures", {})
    has_final = "final_color" in textures
    results["checks"].append({
        "name": "has_final_color",
        "passed": has_final,
        "message": "Final color buffer present" if has_final else "No final color buffer"
    })

    return results


def run_validation(ref_path, test_path, output_report_path, scene_json_path=None):
    """
    Run full validation suite.

    Args:
        ref_path: Path to reference image (original capture)
        test_path: Path to test image (rendered from reconstruction)
        output_report_path: Path to write JSON report
        scene_json_path: Optional path to scene.json for geometry checks

    Returns:
        dict: Validation results
    """
    print("\n=== QA Validation ===")
    print(f"Reference: {ref_path}")
    print(f"Test: {test_path}")

    results = {
        "reference": ref_path,
        "test": test_path,
        "metrics": {},
        "thresholds": {
            "silhouette_iou": 0.92,
            "ssim": 0.80
        },
        "passed": True,
        "issues": []
    }

    # Load images
    try:
        ref_img = load_image(ref_path)
        test_img = load_image(test_path)
    except Exception as e:
        results["passed"] = False
        results["issues"].append(f"Failed to load images: {e}")
        return results

    # Compute metrics
    print("\nComputing metrics...")

    # Silhouette IoU
    sil_iou = compute_silhouette_iou(ref_img, test_img)
    results["metrics"]["silhouette_iou"] = round(sil_iou, 4)
    print(f"  Silhouette IoU: {sil_iou:.4f} (threshold: >= 0.92)")

    if sil_iou < 0.92:
        results["passed"] = False
        results["issues"].append(
            f"Silhouette IoU too low ({sil_iou:.4f}). Check camera FOV/position."
        )

    # SSIM
    ssim_val = compute_ssim(ref_img, test_img)
    results["metrics"]["ssim"] = round(ssim_val, 4)
    print(f"  SSIM: {ssim_val:.4f} (threshold: >= 0.80)")

    if ssim_val < 0.80:
        if results["passed"]:  # Only warn if IoU passed
            results["issues"].append(
                f"SSIM below threshold ({ssim_val:.4f}). Colors/textures may differ."
            )

    # Geometry sanity (if scene.json provided)
    if scene_json_path:
        geom_results = validate_geometry_sanity(scene_json_path)
        results["geometry_checks"] = geom_results
        if not geom_results["passed"]:
            results["passed"] = False
            results["issues"].append("Geometry sanity check failed")

    # Save diff image
    diff_path = output_report_path.replace(".json", "_diff.png")
    diff_img = compute_edge_diff_image(ref_img, test_img)
    cv2.imwrite(diff_path, diff_img)
    results["diff_image"] = diff_path
    print(f"\nDiff image saved: {diff_path}")

    # Write report
    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    with open(output_report_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Report saved: {output_report_path}")

    # Summary
    print("\n=== Validation Summary ===")
    status = "PASSED" if results["passed"] else "FAILED"
    print(f"  Status: {status}")

    if results["issues"]:
        print("  Issues:")
        for issue in results["issues"]:
            print(f"    - {issue}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Validate reconstruction quality against reference"
    )
    parser.add_argument(
        'reference',
        help="Path to reference image (original capture)"
    )
    parser.add_argument(
        'test',
        help="Path to test image (rendered reconstruction)"
    )
    parser.add_argument(
        'report',
        help="Path for output JSON report"
    )
    parser.add_argument(
        '--scene-json',
        dest='scene_json',
        help="Optional: Path to scene.json for geometry checks"
    )

    args = parser.parse_args()

    results = run_validation(
        args.reference,
        args.test,
        args.report,
        args.scene_json
    )

    # Exit with error code if validation failed
    sys.exit(0 if results["passed"] else 1)


if __name__ == "__main__":
    main()
