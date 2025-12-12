"""
Video I/O utilities for extracting frames and selecting keyframes.
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np


def extract_frames_from_video(
    video_path: str,
    every_n: int = 5,
    max_frames: int | None = None,
) -> List[np.ndarray]:
    """
    Extract frames from a video file.

    Args:
        video_path: Path to the input video file.
        every_n: Extract every Nth frame (default: 5).
        max_frames: Maximum number of frames to extract (None = no limit).

    Returns:
        List of frames as numpy arrays (H, W, 3), dtype=uint8, RGB format.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frames = []
    frame_count = 0
    extracted_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Subsample by every_n
        if frame_count % every_n == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb.astype(np.uint8))
            extracted_count += 1

            # Check max_frames limit
            if max_frames is not None and extracted_count >= max_frames:
                break

        frame_count += 1

    cap.release()
    return frames


def select_keyframes(
    frames: List[np.ndarray],
    max_num: int = 20,
) -> List[int]:
    """
    Select keyframe indices from a list of frames.

    Improvement:
    - Always include the first few adjacent frames (0, 1, 2)
      because these are essential for stable initialization,
      especially on low-parallax datasets like Dino.

    - Uniformly sample the remaining frames if needed.
    """

    n = len(frames)
    if n <= max_num:
        return list(range(n))

    # Always include adjacent initial frames
    base = [0, 1, 2]  # guarantees strong overlap and stable SfM init

    remaining_max = max_num - len(base)
    if remaining_max <= 0:
        return base[:max_num]

    # Uniformly sample the rest (excluding the first 3 frames)
    remaining_indices = np.linspace(
        3, n - 1, remaining_max, dtype=int
    ).tolist()  # <<< convert to Python list

    # Combine and deduplicate
    result = []
    seen = set()
    for idx in base + remaining_indices:  # now list + list = concatenation
        if idx not in seen:
            seen.add(int(idx))  # ensure it's an int, not np.int64
            result.append(int(idx))

    return result


__all__ = ["extract_frames_from_video", "select_keyframes"]

