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
        (R, t, mask) where:
        - R: 3x3 rotation from cam1 to cam2
        - t: 3x1 translation from cam1 to cam2
        - mask: boolean inlier mask (N,), here we simply treat all points
                as inliers and let later filters (z>0, reproj error) do the pruning.
    """
    if len(pts1) == 0 or len(pts2) == 0:
        R = np.eye(3)
        t = np.zeros((3, 1))
        mask_bool = np.array([], dtype=bool)
        return R, t, mask_bool

    # Ensure correct dtype/shape for OpenCV
    pts1_cv = pts1.astype(np.float32).reshape(-1, 1, 2)
    pts2_cv = pts2.astype(np.float32).reshape(-1, 1, 2)

    # Call recoverPose in a way that works whether it returns 3 or 4 values.
    result = cv2.recoverPose(E, pts1_cv, pts2_cv, K)

    if isinstance(result, tuple):
        if len(result) == 4:
            retval, R, t, _mask = result
        elif len(result) == 3:
            retval, R, t = result
        else:
            # Unexpected form; fall back to identity transform.
            print("[WARN] recoverPose returned unexpected tuple length; using identity pose.")
            R = np.eye(3)
            t = np.zeros((3, 1))
    else:
        # Extremely weird case (non-tuple); fall back.
        print("[WARN] recoverPose returned non-tuple; using identity pose.")
        R = np.eye(3)
        t = np.zeros((3, 1))

    # IMPORTANT: we do NOT trust recoverPose's mask here.
    # We treat all F-RANSAC inliers as pose inliers and let triangulation
    # + reprojection-error filtering clean things up.
    mask_bool = np.ones(len(pts1), dtype=bool)

    print(f"[DEBUG] recoverPose called on {len(pts1)} points; "
          f"t norm = {np.linalg.norm(t):.4f}")

    return R, t, mask_bool


__all__ = ["compute_essential_matrix", "extract_RT_essential_matrix"]

