#!/usr/bin/env python
"""
Small debugging script to inspect keypoint detection and matching
between two keyframes of a scene video.

Usage (from repo root, inside your venv):

    python debug_features.py --scene-video data/test_vids/Library.MOV --output debug_matches.png

This will:
  - Extract frames from the scene video (same parameters as the CLI: every_n=5, max_frames=500)
  - Select up to 40 keyframes uniformly
  - Take the first and last keyframes as a base pair
  - Detect SIFT keypoints/descriptors
  - Match them with FLANN + Lowe ratio test
  - Estimate F with RANSAC and count inliers
  - Save an image with inlier matches drawn to the output path
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from sfm_app.features.keypoints import detect_keypoints
from sfm_app.features.matching import filter_matches_ratio_test, match_keypoints
from sfm_app.geometry.fundamental import fundamental_matrix_ransac
from sfm_app.io.video_io import extract_frames_from_video, select_keyframes


def draw_matches(
    img1: np.ndarray,
    img2: np.ndarray,
    kp1: list[cv2.KeyPoint],
    kp2: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
) -> np.ndarray:
    """
    Draw matched keypoints between two images.

    Args:
        img1, img2: RGB images (H, W, 3)
        kp1, kp2: keypoints for each image
        matches: list of cv2.DMatch (typically inliers)
    """
    # Convert RGB to BGR for OpenCV drawing
    img1_bgr = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    img2_bgr = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

    vis = cv2.drawMatches(
        img1_bgr,
        kp1,
        img2_bgr,
        kp2,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return vis


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug keypoint detection and matching between two keyframes."
    )
    parser.add_argument(
        "--scene-video",
        type=str,
        required=True,
        help="Path to scene video file (e.g., data/test_vids/Library.MOV)",
    )
    parser.add_argument(
        "--every-n",
        type=int,
        default=5,
        help="Extract every Nth frame from the video (default: 5, like CLI).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=500,
        help="Maximum number of frames to extract (default: 500, like CLI).",
    )
    parser.add_argument(
        "--max-keyframes",
        type=int,
        default=40,
        help="Maximum number of keyframes to select (default: 40, like CLI).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="debug_matches.png",
        help="Path to output image with drawn matches (default: debug_matches.png).",
    )

    args = parser.parse_args()

    scene_path = args.scene_video
    out_path = Path(args.output)

    print(f"[debug] Extracting frames from {scene_path} ...")
    frames = extract_frames_from_video(
        scene_path, every_n=args.every_n, max_frames=args.max_frames
    )
    print(f"[debug] Extracted {len(frames)} frames")

    if len(frames) < 2:
        print("[debug] Not enough frames extracted; aborting.")
        return

    # For debugging, use the same approximate timestamps as the SfM base pair.
    # With FPS ≈ 59.97 and stride every_n=5, frames around 26s and 27s in the
    # original video correspond to indices ~312 and ~324 in the subsampled list.
    approx_i0 = int(round(1559 / args.every_n))  # ~26s
    approx_i1 = int(round(1619 / args.every_n))  # ~27s
    n_frames = len(frames)
    i0 = max(0, min(n_frames - 1, approx_i0))
    i1 = max(0, min(n_frames - 1, approx_i1))
    print(f"[debug] Using base frames indices {i0} and {i1} in subsampled list")

    img1 = frames[i0]
    img2 = frames[i1]

    # Detect keypoints
    kp1, desc1 = detect_keypoints(img1, use_sift=True)
    kp2, desc2 = detect_keypoints(img2, use_sift=True)
    print(
        f"[debug] Detected {len(kp1)} keypoints in frame {i0}, "
        f"{len(kp2)} keypoints in frame {i1}"
    )

    if len(kp1) == 0 or len(kp2) == 0 or desc1 is None or desc2 is None:
        print("[debug] No features or descriptors in one of the frames; aborting.")
        return

    # Match descriptors
    knn_matches = match_keypoints(desc1, desc2, use_flann=True)
    pts1, pts2, good_matches = filter_matches_ratio_test(
        kp1, kp2, knn_matches, ratio=0.75
    )
    print(
        f"[debug] Matching stats: {len(knn_matches)} knn matches, "
        f"{len(good_matches)} after ratio test"
    )

    if len(pts1) < 8:
        print("[debug] Too few good matches (<8); cannot estimate F robustly.")
    else:
        # Estimate F with RANSAC to see how many inliers we have
        F, inlier_mask = fundamental_matrix_ransac(pts1, pts2)
        num_inliers = int(np.sum(inlier_mask))
        print(f"[debug] F RANSAC inliers: {num_inliers}")

        # Keep only inlier matches for drawing
        inlier_matches = [good_matches[i] for i in range(len(good_matches)) if inlier_mask[i]]

        vis = draw_matches(img1, img2, kp1, kp2, inlier_matches)
        cv2.imwrite(str(out_path), vis)
        print(f"[debug] Wrote inlier match visualization to {out_path}")


if __name__ == "__main__":
    main()


