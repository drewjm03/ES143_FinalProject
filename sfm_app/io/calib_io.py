"""
Calibration I/O utilities for saving and loading camera intrinsics and scenes.
"""

from __future__ import annotations

import numpy as np

from sfm_app.sfm.data_structures import SceneGraph


def save_calibration(
    output_path: str,
    K: np.ndarray,
    dist_coeffs: np.ndarray,
) -> None:
    """
    Save camera intrinsics and distortion coefficients to a .npz file.

    Args:
        output_path: Path where the calibration data will be saved (.npz file).
        K: Intrinsic camera matrix (3x3).
        dist_coeffs: Distortion coefficients array.
    """
    np.savez(output_path, K=K, dist_coeffs=dist_coeffs)


def load_calibration(
    input_path: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load camera intrinsics and distortion coefficients from a .npz file.

    Args:
        input_path: Path to the .npz file containing calibration data.

    Returns:
        Tuple of (K, dist_coeffs) where:
        - K: Intrinsic camera matrix (3x3).
        - dist_coeffs: Distortion coefficients array.
    """
    data = np.load(input_path)
    K = data["K"]
    dist_coeffs = data["dist_coeffs"]
    return K, dist_coeffs


def save_scene_npz(
    output_path: str,
    scene: SceneGraph,
    K: np.ndarray,
) -> None:
    """
    Serialize a SceneGraph and camera intrinsics to a .npz file.

    Args:
        output_path: Path where the scene data will be saved (.npz file).
        scene: SceneGraph containing cameras, 3D points, and observations.
        K: Intrinsic camera matrix (3x3).
    """
    # Extract camera poses
    n_cameras = len(scene.cameras)
    camera_Rs = np.zeros((n_cameras, 3, 3))
    camera_ts = np.zeros((n_cameras, 3))
    camera_image_indices = np.zeros(n_cameras, dtype=int)

    for i, cam in enumerate(scene.cameras):
        camera_Rs[i] = cam.R
        camera_ts[i] = cam.t.flatten()
        camera_image_indices[i] = cam.image_idx

    # Extract 3D points
    n_points = len(scene.points3d)
    points_xyz = np.zeros((n_points, 3))
    points_colors = np.zeros((n_points, 3), dtype=np.uint8)

    for i, pt in enumerate(scene.points3d):
        points_xyz[i] = pt.xyz
        points_colors[i] = pt.color

    # Extract observations
    n_observations = len(scene.observations)
    obs_camera_ids = np.zeros(n_observations, dtype=int)
    obs_point_ids = np.zeros(n_observations, dtype=int)
    obs_uvs = np.zeros((n_observations, 2))

    for i, obs in enumerate(scene.observations):
        obs_camera_ids[i] = obs.camera_id
        obs_point_ids[i] = obs.point_id
        obs_uvs[i] = obs.uv

    np.savez(
        output_path,
        K=K,
        camera_Rs=camera_Rs,
        camera_ts=camera_ts,
        camera_image_indices=camera_image_indices,
        points_xyz=points_xyz,
        points_colors=points_colors,
        obs_camera_ids=obs_camera_ids,
        obs_point_ids=obs_point_ids,
        obs_uvs=obs_uvs,
    )


__all__ = ["save_calibration", "load_calibration", "save_scene_npz"]

