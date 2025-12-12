"""
Bundle adjustment for refining camera poses and 3D point positions.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares

from sfm_app.sfm_inc.data_structures import SceneGraph


def pack_parameters(
    scene: SceneGraph,
    fixed_camera_ids: Optional[Iterable[int]] = None,
) -> Tuple[np.ndarray, Dict]:
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
    fixed_ids = set(fixed_camera_ids) if fixed_camera_ids is not None else set()
    meta = {
        "camera_slice": {},
        "point_slice": {},
    }

    # Pack camera parameters (6 per camera: rvec (3) + t (3)), skipping
    # any cameras whose poses are held fixed (for gauge fixing).
    camera_params_start = 0
    for cam in scene.cameras:
        if cam.id in fixed_ids:
            # Mark this camera as fixed by storing a None slice; no
            # parameters are added to the vector for this camera.
            meta["camera_slice"][cam.id] = None
            continue
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
    # Unpack camera parameters (skip any cameras that were held fixed)
    for cam in scene.cameras:
        cam_slice = meta["camera_slice"][cam.id]
        if cam_slice is None:
            continue  # fixed camera: do not overwrite its pose

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
    # Precompute fixed camera extrinsics (those with no parameter slice).
    camera_slices = meta["camera_slice"]
    fixed_cam_rt: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for cam in scene.cameras:
        sl = camera_slices.get(cam.id)
        if sl is None:
            rvec, _ = cv2.Rodrigues(cam.R)
            fixed_cam_rt[cam.id] = (rvec.flatten(), cam.t.flatten())

    for obs in scene.observations:
        # Get camera and point indices
        camera_id = obs.camera_id
        point_id = obs.point_id

        # Extract camera parameters
        cam_slice = camera_slices[camera_id]
        if cam_slice is None:
            # This camera is held fixed: use its current R,t directly.
            rvec, tvec = fixed_cam_rt[camera_id]
        else:
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
    max_nfev: int = 15,
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
    # We target a *smaller* subset of long tracks to reduce runtime:
    #  - only consider points observed in at least `min_track_len` cameras
    #  - cap at `max_points_for_ba` points, sampled with probability
    #    proportional to track length.
    max_points_for_ba = 500
    min_track_len = 4
    n_total_points = len(scene.points3d)

    # Identify "long track" points first.
    track_lengths = np.array(
        [len(pt.observations) for pt in scene.points3d],
        dtype=np.int32,
    )
    eligible_indices = np.where(track_lengths >= min_track_len)[0]

    if eligible_indices.size == 0:
        # Fall back to using all points if nothing meets the track-length threshold.
        eligible_indices = np.arange(n_total_points)

    n_eligible = eligible_indices.size

    if n_eligible > max_points_for_ba:
        tl_eligible = track_lengths[eligible_indices].astype(np.float64)
        tl_eligible[tl_eligible <= 0] = 1.0
        weights = tl_eligible / np.sum(tl_eligible)

        rng = np.random.default_rng()
        chosen_local = rng.choice(
            n_eligible,
            size=max_points_for_ba,
            replace=False,
            p=weights,
        )
        selected_indices = eligible_indices[chosen_local]
    else:
        selected_indices = eligible_indices

    if selected_indices.size < n_total_points:
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
            f"{len(scene_for_ba.points3d)} for bundle adjustment "
            f"(track-length-weighted, min_track_len={min_track_len})"
        )
    else:
        scene_for_ba = scene

    # Pack parameters, holding camera 0 fixed to remove gauge freedom.
    fixed_cam_ids = {0} if any(cam.id == 0 for cam in scene_for_ba.cameras) else set()
    params, meta = pack_parameters(scene_for_ba, fixed_camera_ids=fixed_cam_ids)

    n_cams = len(scene_for_ba.cameras)
    n_pts = len(scene_for_ba.points3d)
    n_obs = len(scene_for_ba.observations)
    n_params = params.size

    # Compute initial residual statistics.
    res0 = reprojection_residuals(params, scene_for_ba, meta, K)
    rms0 = float(np.sqrt(np.mean(res0 ** 2))) if res0.size > 0 else 0.0
    max0 = float(np.max(np.abs(res0))) if res0.size > 0 else 0.0

    print(
        f"[ba] Starting bundle adjustment with {n_cams} cameras, "
        f"{n_pts} points, {n_obs} observations, {n_params} parameters, "
        f"max_nfev={max_nfev}"
    )
    print(
        f"[ba] Initial reprojection error: RMS={rms0:.2f} px, "
        f"max={max0:.2f} px over {res0.size // 2} observations"
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

    # Compute final residual statistics.
    res1 = reprojection_residuals(result.x, scene_for_ba, meta, K)
    rms1 = float(np.sqrt(np.mean(res1 ** 2))) if res1.size > 0 else 0.0
    max1 = float(np.max(np.abs(res1))) if res1.size > 0 else 0.0

    print(
        f"[ba] Done. Status={result.status}, nfev={result.nfev}, "
        f"cost={result.cost:.3e}"
    )
    print(
        f"[ba] Final reprojection error: RMS={rms1:.2f} px, "
        f"max={max1:.2f} px (ΔRMS={rms1 - rms0:+.2f} px)"
    )

    # Unpack optimized parameters back into the (subsampled) scene.
    unpack_parameters(result.x, scene_for_ba, meta)

    return scene


__all__ = [
    "pack_parameters",
    "unpack_parameters",
    "project_point",
    "reprojection_residuals",
    "run_bundle_adjustment",
]

