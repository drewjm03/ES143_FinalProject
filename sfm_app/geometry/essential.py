"""
Essential matrix computation and camera pose extraction.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def compute_essential_matrix(K: np.ndarray, F: np.ndarray) -> np.ndarray:
    """
    Compute essential matrix from fundamental matrix and camera intrinsics.

    Args:
        K: Intrinsic camera matrix (3x3).
        F: Fundamental matrix (3x3).

    Returns:
        Essential matrix E (3x3), where E = K^T @ F @ K.
    """
    E = K.T @ F @ K
    return E


def extract_RT_essential_matrix(
    E: np.ndarray,
    K: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract camera rotation and translation from essential matrix.

    Args:
        E: Essential matrix (3x3).
        K: Intrinsic camera matrix (3x3).
        pts1: Points in first image (N, 2).
        pts2: Points in second image (N, 2).

    Returns:
        Tuple of (R, t, mask) where:
        - R: Rotation matrix (3x3) from first to second camera.
        - t: Translation vector (3, 1) from first to second camera.
        - mask: Inlier mask (N,) indicating which points are in front of both cameras.
    """
    if len(pts1) == 0 or len(pts2) == 0:
        R = np.eye(3)
        t = np.zeros((3, 1))
        mask = np.array([]).astype(np.uint8)
        return R, t, mask

    # Recover pose using OpenCV
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)

    # OpenCV returns mask as uint8 (0 or 255). Convert to boolean mask of shape (N,).
    mask_bool = mask.ravel().astype(bool)

    return R, t, mask_bool


__all__ = ["compute_essential_matrix", "extract_RT_essential_matrix"]

