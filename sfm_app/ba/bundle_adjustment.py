"""
Bundle adjustment for refining camera poses and 3D point positions.
"""

from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares

from sfm_app.sfm_inc.data_structures import SceneGraph


def pack_parameters(scene: SceneGraph) -> Tuple[np.ndarray, Dict]:
    """
    Pack all camera extrinsics and 3D point positions into a 1D parameter vector.

    Args:
        scene: SceneGraph containing cameras and 3D points.

    Returns:
        Tuple of (params, meta) where:
        - params: 1D array of all parameters.
        - meta: Dictionary with slice information for unpacking:
            - meta["camera_slice"][camera_id] -> slice object
            - meta["point_slice"][point_id] -> slice object
            - meta["camera_params_start"]: Starting index for camera parameters
            - meta["point_params_start"]: Starting index for point parameters
    """
    param_list = []
    meta = {
        "camera_slice": {},
        "point_slice": {},
    }

    # Pack camera parameters (6 per camera: rvec (3) + t (3))
    camera_params_start = 0
    for cam in scene.cameras:
        # Convert R to rvec using Rodrigues
        rvec, _ = cv2.Rodrigues(cam.R)
        rvec = rvec.flatten()

        # Get translation
        t = cam.t.flatten()

        # Pack: [rvec[0], rvec[1], rvec[2], t[0], t[1], t[2]]
        start_idx = len(param_list)
        param_list.extend([rvec[0], rvec[1], rvec[2], t[0], t[1], t[2]])
        end_idx = len(param_list)

        meta["camera_slice"][cam.id] = slice(start_idx, end_idx)

    # Pack point parameters (3 per point: X, Y, Z)
    point_params_start = len(param_list)
    for pt in scene.points3d:
        xyz = pt.xyz
        start_idx = len(param_list)
        param_list.extend([xyz[0], xyz[1], xyz[2]])
        end_idx = len(param_list)

        meta["point_slice"][pt.id] = slice(start_idx, end_idx)

    meta["camera_params_start"] = camera_params_start
    meta["point_params_start"] = point_params_start

    params = np.array(param_list, dtype=np.float64)
    return params, meta


def unpack_parameters(
    params: np.ndarray,
    scene: SceneGraph,
    meta: Dict,
) -> None:
    """
    Unpack optimized parameters back into the SceneGraph.

    Args:
        params: 1D array of optimized parameters.
        scene: SceneGraph to update in-place.
        meta: Dictionary with slice information from pack_parameters.
    """
    # Unpack camera parameters
    for cam in scene.cameras:
        cam_slice = meta["camera_slice"][cam.id]
        cam_params = params[cam_slice]

        rvec = cam_params[:3]
        t = cam_params[3:6]

        # Convert rvec to R
        R, _ = cv2.Rodrigues(rvec)
        cam.R = R
        cam.t = t.reshape(3, 1)

    # Unpack point parameters
    for pt in scene.points3d:
        pt_slice = meta["point_slice"][pt.id]
        pt_params = params[pt_slice]

        pt.xyz = pt_params


def project_point(
    K: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    X: np.ndarray,
) -> np.ndarray:
    """
    Project a 3D point into image coordinates.

    Args:
        K: Intrinsic camera matrix (3x3).
        rvec: Rotation vector (3,).
        tvec: Translation vector (3,).
        X: 3D point (3,).

    Returns:
        Projected 2D point (u, v).
    """
    # cv2.projectPoints expects (N, 1, 3) shape
    points_3d = X.reshape(1, 1, 3)
    rvec_reshaped = rvec.reshape(3, 1)
    tvec_reshaped = tvec.reshape(3, 1)

    projected, _ = cv2.projectPoints(points_3d, rvec_reshaped, tvec_reshaped, K, None)

    uv = projected[0, 0]  # Extract (u, v)
    return uv


def reprojection_residuals(
    params: np.ndarray,
    scene: SceneGraph,
    meta: Dict,
    K: np.ndarray,
) -> np.ndarray:
    """
    Compute reprojection residuals for all observations.

    Args:
        params: 1D parameter vector (camera poses + 3D points).
        scene: SceneGraph with observations.
        meta: Dictionary with slice information for unpacking.
        K: Intrinsic camera matrix (3x3).

    Returns:
        1D array of residuals (2 per observation: [du, dv]).
    """
    residuals = []

    for obs in scene.observations:
        # Get camera and point indices
        camera_id = obs.camera_id
        point_id = obs.point_id

        # Extract camera parameters
        cam_slice = meta["camera_slice"][camera_id]
        cam_params = params[cam_slice]
        rvec = cam_params[:3]
        tvec = cam_params[3:6]

        # Extract point parameters
        pt_slice = meta["point_slice"][point_id]
        pt_params = params[pt_slice]
        X = pt_params

        # Project point
        uv_projected = project_point(K, rvec, tvec, X)

        # Compute residual
        uv_observed = obs.uv
        residual = uv_projected - uv_observed
        residuals.extend([residual[0], residual[1]])

    return np.array(residuals, dtype=np.float64)


def run_bundle_adjustment(
    scene: SceneGraph,
    K: np.ndarray,
    max_nfev: int = 10,
) -> SceneGraph:
    """
    Run bundle adjustment to refine camera poses and 3D point positions.

    Args:
        scene: SceneGraph to optimize.
        K: Intrinsic camera matrix (3x3).
        max_nfev: Maximum number of function evaluations.

    Returns:
        Updated SceneGraph with optimized poses and points.
    """
    if len(scene.cameras) == 0 or len(scene.points3d) == 0:
        return scene

    # Optionally subsample points to keep BA lightweight.
    # We target roughly 200–300 points; use at most 250 randomly selected points.
    max_points_for_ba = 250
    n_total_points = len(scene.points3d)

    if n_total_points > max_points_for_ba:
        # Randomly select a subset of 3D points for BA.
        rng = np.random.default_rng()
        selected_indices = rng.choice(n_total_points, size=max_points_for_ba, replace=False)
        selected_ids = {scene.points3d[i].id for i in selected_indices}

        # Build a lightweight SceneGraph "view" that shares Camera/Point3D
        # objects with the original scene but only includes the sampled points
        # and their observations.
        scene_for_ba = SceneGraph(
            cameras=list(scene.cameras),
            points3d=[pt for pt in scene.points3d if pt.id in selected_ids],
            observations=[obs for obs in scene.observations if obs.point_id in selected_ids],
        )

        print(
            f"[ba] Subsampling {n_total_points} points down to "
            f"{len(scene_for_ba.points3d)} for bundle adjustment"
        )
    else:
        scene_for_ba = scene

    # Pack parameters
    params, meta = pack_parameters(scene_for_ba)

    n_cams = len(scene_for_ba.cameras)
    n_pts = len(scene_for_ba.points3d)
    n_obs = len(scene_for_ba.observations)
    n_params = params.size
    print(
        f"[ba] Starting bundle adjustment with {n_cams} cameras, "
        f"{n_pts} points, {n_obs} observations, {n_params} parameters, "
        f"max_nfev={max_nfev}"
    )

    # Run optimization with limited iterations (to keep runtime manageable)
    result = least_squares(
        reprojection_residuals,
        params,
        args=(scene_for_ba, meta, K),
        method="trf",
        loss="soft_l1",
        verbose=2,  # more detailed progress info
        max_nfev=max_nfev,
    )

    print(
        f"[ba] Done. Status={result.status}, nfev={result.nfev}, "
        f"initial_cost={result.cost:.3e}"
    )

    # Unpack optimized parameters
    unpack_parameters(result.x, scene_for_ba, meta)

    return scene


__all__ = [
    "pack_parameters",
    "unpack_parameters",
    "project_point",
    "reprojection_residuals",
    "run_bundle_adjustment",
]

