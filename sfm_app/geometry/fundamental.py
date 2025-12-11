"""
Fundamental matrix estimation using normalized 8-point algorithm and RANSAC.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def normalize_points(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize 2D points by centering and scaling.

    Args:
        pts: Array of points (N, 2).

    Returns:
        Tuple of (normalized_pts, T) where:
        - normalized_pts: Normalized points (N, 2).
        - T: Transformation matrix (3x3) that normalizes pts.
    """
    mean = np.mean(pts, axis=0)
    centered = pts - mean
    scale = np.sqrt(2.0) / np.mean(np.linalg.norm(centered, axis=1))
    if np.isnan(scale) or scale == 0:
        scale = 1.0

    T = np.array(
        [
            [scale, 0, -scale * mean[0]],
            [0, scale, -scale * mean[1]],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    ones = np.ones((pts.shape[0], 1))
    pts_homogeneous = np.hstack([pts, ones])
    normalized_pts = (T @ pts_homogeneous.T).T[:, :2]

    return normalized_pts, T


def estimate_fundamental_matrix(
    pts1: np.ndarray,
    pts2: np.ndarray,
) -> np.ndarray:
    """
    Estimate fundamental matrix using normalized 8-point algorithm.

    Args:
        pts1: Points in first image (N, 2).
        pts2: Points in second image (N, 2).

    Returns:
        Fundamental matrix F (3x3).

    Raises:
        ValueError: If fewer than 8 point correspondences are provided.
    """
    if len(pts1) < 8:
        raise ValueError(f"Need at least 8 point correspondences, got {len(pts1)}")

    # Normalize points
    pts1_norm, T1 = normalize_points(pts1)
    pts2_norm, T2 = normalize_points(pts2)

    # Build coefficient matrix A (N x 9)
    n = len(pts1_norm)
    A = np.zeros((n, 9))

    for i in range(n):
        x1, y1 = pts1_norm[i]
        x2, y2 = pts2_norm[i]
        A[i] = [
            x2 * x1,
            x2 * y1,
            x2,
            y2 * x1,
            y2 * y1,
            y2,
            x1,
            y1,
            1,
        ]

    # Solve Af=0 via SVD
    _, _, Vt = np.linalg.svd(A)
    f = Vt[-1]  # Last row of V^T (smallest singular value)
    F = f.reshape(3, 3)

    # Enforce rank-2 constraint
    F = constrain_F(F)

    # Denormalize: F = T2^T @ F @ T1
    F = T2.T @ F @ T1

    return F


def constrain_F(F: np.ndarray) -> np.ndarray:
    """
    Enforce rank-2 constraint on fundamental matrix using SVD.

    Args:
        F: Fundamental matrix (3x3).

    Returns:
        Rank-2 constrained fundamental matrix (3x3).
    """
    U, S, Vt = np.linalg.svd(F)
    # Zero out smallest singular value
    S[2] = 0
    # Reconstruct F
    F_rank2 = U @ np.diag(S) @ Vt

    return F_rank2


def fundamental_matrix_ransac(
    pts1: np.ndarray,
    pts2: np.ndarray,
    reproj_threshold: float = 1.0,
    confidence: float = 0.999,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate fundamental matrix using RANSAC.

    Args:
        pts1: Points in first image (N, 2).
        pts2: Points in second image (N, 2).
        reproj_threshold: Maximum distance from a point to an epipolar line
                          for it to be considered an inlier.
        confidence: Confidence level for RANSAC.

    Returns:
        Tuple of (F, inlier_mask) where:
        - F: Fundamental matrix (3x3).
        - inlier_mask: Boolean array (N,) indicating inlier correspondences.
    """
    if len(pts1) < 8:
        # If too few points, return identity-like matrix and all-False boolean mask.
        F = np.eye(3)
        inlier_mask = np.zeros(len(pts1), dtype=bool)
        return F, inlier_mask

    F, inlier_mask = cv2.findFundamentalMat(
        pts1,
        pts2,
        cv2.FM_RANSAC,
        reproj_threshold,
        confidence,
    )

    if F is None:
        F = np.eye(3)
        inlier_mask = np.zeros(len(pts1), dtype=bool)
    else:
        # OpenCV returns an uint8 mask with values 0 or 1 (or 0/255).
        # We must convert this to a boolean mask before using it for
        # boolean indexing; otherwise NumPy will interpret it as an
        # index array and effectively mix only rows 0/1.
        inlier_mask = inlier_mask.astype(bool)

    return F, inlier_mask.ravel()


__all__ = [
    "estimate_fundamental_matrix",
    "fundamental_matrix_ransac",
    "constrain_F",
]

