"""
3D point triangulation from two camera views.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def triangulate_matched_key_pts_to_3D_pts(
    K: np.ndarray,
    R1: np.ndarray,
    t1: np.ndarray,
    R2: np.ndarray,
    t2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Triangulate 3D points from matched 2D correspondences in two views.

    Args:
        K: Intrinsic camera matrix (3x3).
        R1: Rotation matrix for first camera (3x3).
        t1: Translation vector for first camera (3, 1) or (3,).
        R2: Rotation matrix for second camera (3x3).
        t2: Translation vector for second camera (3, 1) or (3,).
        pts1: Points in first image (N, 2).
        pts2: Points in second image (N, 2).

    Returns:
        Tuple of (points_3d, reprojection_errors) where:
        - points_3d: Triangulated 3D points (N, 3) in world coordinates.
        - reprojection_errors: Per-point reprojection errors (N,) (average of both views).
    """
    if len(pts1) == 0:
        return np.array([]).reshape(0, 3), np.array([])

    # Ensure t1 and t2 are column vectors
    if t1.ndim == 1:
        t1 = t1.reshape(3, 1)
    if t2.ndim == 1:
        t2 = t2.reshape(3, 1)

    # Construct projection matrices
    # P = K [R | t]
    P1 = K @ np.hstack([R1, t1])
    P2 = K @ np.hstack([R2, t2])

    # Triangulate points (OpenCV expects points as (N, 2) and transposes internally)
    points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)

    # Convert from homogeneous to inhomogeneous coordinates
    points_3d = points_4d[:3] / points_4d[3]  # (3, N)
    points_3d = points_3d.T  # (N, 3)

    # Debug: inspect raw triangulated points before any filtering.
    print("[DEBUG] Triangulated points: shape", points_3d.shape)
    if points_3d.shape[0] > 0:
        print("[DEBUG] First 5 points:\\n", points_3d[:5])
        unique_count = np.unique(points_3d.round(4), axis=0).shape[0]
        print("[DEBUG] Unique points (rounded to 4dp):", unique_count)

    # Compute reprojection errors
    reprojection_errors = _compute_reprojection_errors(
        K, R1, t1, R2, t2, pts1, pts2, points_3d
    )

    return points_3d, reprojection_errors


def _compute_reprojection_errors(
    K: np.ndarray,
    R1: np.ndarray,
    t1: np.ndarray,
    R2: np.ndarray,
    t2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    points_3d: np.ndarray,
) -> np.ndarray:
    """
    Compute reprojection errors for triangulated 3D points.

    Args:
        K: Intrinsic camera matrix (3x3).
        R1, t1: Pose of first camera.
        R2, t2: Pose of second camera.
        pts1, pts2: Observed 2D points (N, 2).
        points_3d: Triangulated 3D points (N, 3).

    Returns:
        Array of reprojection errors (N,) - average error across both views.
    """
    if len(points_3d) == 0:
        return np.array([])

    # Ensure t vectors are column vectors
    if t1.ndim == 1:
        t1 = t1.reshape(3, 1)
    if t2.ndim == 1:
        t2 = t2.reshape(3, 1)

    # Project 3D points to image coordinates
    rvec1, _ = cv2.Rodrigues(R1)
    rvec2, _ = cv2.Rodrigues(R2)

    projected1, _ = cv2.projectPoints(
        points_3d.reshape(-1, 1, 3), rvec1, t1, K, None
    )
    projected2, _ = cv2.projectPoints(
        points_3d.reshape(-1, 1, 3), rvec2, t2, K, None
    )

    projected1 = projected1.reshape(-1, 2)
    projected2 = projected2.reshape(-1, 2)

    # Compute errors
    error1 = np.linalg.norm(pts1 - projected1, axis=1)
    error2 = np.linalg.norm(pts2 - projected2, axis=1)

    # Average error across both views
    avg_errors = (error1 + error2) / 2.0

    return avg_errors


__all__ = ["triangulate_matched_key_pts_to_3D_pts"]

