"""
Keypoint detection and descriptor extraction.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


def detect_keypoints(
    image: np.ndarray,
    use_sift: bool = True,
) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """
    Detect keypoints and compute descriptors in an image.

    Args:
        image: Input image (H, W, 3) or (H, W), dtype=uint8.
        use_sift: If True, use SIFT detector; otherwise use ORB.

    Returns:
        Tuple of (keypoints, descriptors) where:
        - keypoints: List of cv2.KeyPoint objects.
        - descriptors: Array of descriptors (N, D), dtype=float32 (SIFT) or uint8 (ORB).
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    if use_sift:
        detector = cv2.SIFT_create()
    else:
        detector = cv2.ORB_create()

    keypoints, descriptors = detector.detectAndCompute(gray, None)

    if descriptors is None:
        descriptors = np.array([]).reshape(0, detector.descriptorSize())

    return keypoints, descriptors


__all__ = ["detect_keypoints"]

