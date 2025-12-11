"""
Feature matching utilities using FLANN or brute-force matchers.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


def match_keypoints(
    descriptors1: np.ndarray,
    descriptors2: np.ndarray,
    use_flann: bool = True,
) -> List[List[cv2.DMatch]]:
    """
    Match keypoint descriptors between two images using k-NN matching.

    Args:
        descriptors1: Descriptors from first image (N1, D).
        descriptors2: Descriptors from second image (N2, D).
        use_flann: If True, use FLANN matcher (for float descriptors);
                   otherwise use BFMatcher with HAMMING (for binary descriptors).

    Returns:
        List of k-NN match candidates. Each element is a list of cv2.DMatch objects
        (typically 2 for k=2 ratio test).
    """
    if len(descriptors1) == 0 or len(descriptors2) == 0:
        return []

    # Determine descriptor type
    is_float = descriptors1.dtype == np.float32

    if use_flann and is_float:
        # FLANN matcher for SIFT (float descriptors)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    else:
        # BFMatcher for ORB (binary descriptors) or as fallback
        norm_type = cv2.NORM_HAMMING if not is_float else cv2.NORM_L2
        matcher = cv2.BFMatcher(norm_type, crossCheck=False)

    # k-NN matching with k=2 for ratio test
    knn_matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    return knn_matches


def filter_matches_ratio_test(
    keypoints1: List[cv2.KeyPoint],
    keypoints2: List[cv2.KeyPoint],
    knn_matches: List[List[cv2.DMatch]],
    ratio: float = 0.75,
) -> Tuple[np.ndarray, np.ndarray, List[cv2.DMatch]]:
    """
    Filter matches using Lowe's ratio test.

    Args:
        keypoints1: Keypoints from first image.
        keypoints2: Keypoints from second image.
        knn_matches: List of k-NN match candidates (typically k=2).
        ratio: Ratio threshold for Lowe's test (default: 0.75).

    Returns:
        Tuple of (pts1, pts2, good_matches) where:
        - pts1: Array of matched points from first image (N, 2).
        - pts2: Array of matched points from second image (N, 2).
        - good_matches: List of filtered cv2.DMatch objects.
    """
    good_matches = []
    pts1_list = []
    pts2_list = []

    for match_pair in knn_matches:
        if len(match_pair) < 2:
            continue

        m, n = match_pair[0], match_pair[1]

        # Lowe's ratio test: keep if distance ratio is below threshold
        if m.distance < ratio * n.distance:
            good_matches.append(m)
            pts1_list.append(keypoints1[m.queryIdx].pt)
            pts2_list.append(keypoints2[m.trainIdx].pt)

    if len(good_matches) == 0:
        pts1 = np.array([]).reshape(0, 2)
        pts2 = np.array([]).reshape(0, 2)
    else:
        pts1 = np.array(pts1_list, dtype=np.float32)
        pts2 = np.array(pts2_list, dtype=np.float32)

    return pts1, pts2, good_matches


__all__ = ["match_keypoints", "filter_matches_ratio_test"]

