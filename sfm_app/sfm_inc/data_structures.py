"""
Shared core data structures for the SfM pipeline.

These dataclasses are intentionally simple containers used across:
- incremental SfM
- bundle adjustment
- visualization
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class Camera:
    """Represents a single calibrated camera and its image measurements."""

    id: int
    # Intrinsic matrix (3x3).
    K: np.ndarray
    # Rotation (3x3) and translation (3x1) from world to camera coordinates.
    R: np.ndarray
    t: np.ndarray

    # Index into the original frames list / video.
    image_idx: int

    # Keypoints and descriptors for this image.
    # keypoints: (N, 2) array (x, y) in pixel coordinates, dtype=float32.
    # These are ALL detected features for this image.
    keypoints: np.ndarray
    # descriptors: (N, D) array, dtype=float32 (SIFT) or uint8 (ORB),
    # aligned with `keypoints` by row index.
    descriptors: np.ndarray
    # point_ids: (N,) int array mapping each keypoint to a 3D point id in
    # SceneGraph.points3d, or -1 if that keypoint has no associated 3D point.
    point_ids: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=int))


@dataclass
class Observation:
    """
    A 2D observation of a 3D point in a particular camera.

    `uv` is (2,) numpy array in pixel coordinates.
    """

    camera_id: int
    point_id: int
    uv: np.ndarray


@dataclass
class Point3D:
    """A single 3D point in the global scene."""

    id: int
    # 3D location (X, Y, Z) in world coordinates.
    xyz: np.ndarray
    # RGB color (3,) uint8, typically sampled from one of the observing images.
    color: np.ndarray
    # Indices into the global SceneGraph.observations list.
    observations: List[int] = field(default_factory=list)


@dataclass
class SceneGraph:
    """
    Global container for all cameras, points, and observations.

    This is the main structure passed between SfM, bundle adjustment, and
    visualization code.
    """

    cameras: List[Camera] = field(default_factory=list)
    points3d: List[Point3D] = field(default_factory=list)
    observations: List[Observation] = field(default_factory=list)


__all__ = ["Camera", "Point3D", "Observation", "SceneGraph"]


