"""
Perspective-n-Point (PnP) pose estimation.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def estimate_camera_pose_pnp(
    K: np.ndarray,
    points_3d: np.ndarray,
    points_2d: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate camera pose from 3D-2D correspondences using PnP RANSAC.

    Args:
        K: Intrinsic camera matrix (3x3).
        points_3d: 3D points in world coordinates (N, 3).
        points_2d: Corresponding 2D points in image coordinates (N, 2).

    Returns:
        Tuple of (R, t, inlier_mask) where:
        - R: Rotation matrix (3x3) from world to camera coordinates.
        - t: Translation vector (3, 1) from world to camera coordinates.
        - inlier_mask: Boolean array (N,) indicating inlier correspondences.
    """
    if len(points_3d) < 4:
        # PnP requires at least 4 points
        R = np.eye(3)
        t = np.zeros((3, 1))
        inlier_mask = np.zeros(len(points_3d), dtype=bool)
        return R, t, inlier_mask

    # Solve PnP using RANSAC
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        points_3d.reshape(-1, 1, 3),
        points_2d.reshape(-1, 1, 2),
        K,
        None,  # dist_coeffs
        flags=cv2.SOLVEPNP_EPNP,
        reprojectionError=4.0,
        confidence=0.99,
        iterationsCount=2000,
    )

    if not success or inliers is None or len(inliers) == 0:
        # Fallback: identity pose
        R = np.eye(3)
        t = np.zeros((3, 1))
        inlier_mask = np.zeros(len(points_3d), dtype=bool)
        return R, t, inlier_mask

    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3, 1)

    # Create full inlier mask
    inlier_mask = np.zeros(len(points_3d), dtype=bool)
    inlier_mask[inliers.ravel()] = True

    # Debug: log the estimated camera center from PnP.
    C = -R.T @ t
    print(
        "[DEBUG] New cam via PnP: center",
        C.ravel(),
        "norm",
        float(np.linalg.norm(C)),
        "num_inliers",
        int(inlier_mask.sum()),
    )

    return R, t, inlier_mask


__all__ = ["estimate_camera_pose_pnp"]

