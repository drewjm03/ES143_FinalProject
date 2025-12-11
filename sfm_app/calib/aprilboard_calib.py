"""
AprilBoard-based camera calibration from video.
"""

from __future__ import annotations

import pickle
from typing import Tuple

import cv2
import numpy as np
import requests

from sfm_app.io.calib_io import save_calibration

# Detection uses pupil_apriltags directly in the calibration function

# URL to download AprilBoards pickle file from CS283 pset data
APRILBOARD_URL = (
    "https://github.com/Harvard-CS283/pset-data/raw/"
    "f1a90573ae88cd530a3df3cd0cea71aa2363b1b3/april/AprilBoards.pickle"
)


def load_aprilboards() -> Tuple[object, object]:
    """
    Download and load AprilBoard definitions from the CS283 repository.

    Returns:
        Tuple of (at_coarseboard, at_fineboard) AprilBoard objects.
    """
    response = requests.get(APRILBOARD_URL)
    response.raise_for_status()
    data = pickle.loads(response.content)

    at_coarseboard = data["at_coarseboard"]
    at_fineboard = data["at_fineboard"]

    return at_coarseboard, at_fineboard


def calibrate_camera_from_aprilboard_video(
    calib_video_path: str,
    board_type: str = "coarse",
    max_frames: int = 100,
    output_path: str | None = None,
    stride: int = 8,
    min_points: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calibrate a camera from a video containing AprilBoard patterns.

    Based on the CS283 calibration pattern:
    - Extracts frames from video with specified stride
    - Detects AprilBoard corners in each frame
    - Accumulates 2D-3D correspondences
    - Runs cv2.calibrateCamera to get intrinsics and distortion

    Args:
        calib_video_path: Path to the calibration video file.
        board_type: Type of board to use ("coarse" or "fine").
        max_frames: Maximum number of frames to use for calibration.
        output_path: Optional path to save the calibration results.
        stride: Extract every Nth frame (default: 8).
        min_points: Minimum number of points required per frame (default: 30).

    Returns:
        Tuple of (K, dist_coeffs) where:
        - K: Intrinsic camera matrix (3x3).
        - dist_coeffs: Distortion coefficients array.

    Note:
        This function requires the apriltag library and a detect_aprilboard helper.
        If detect_aprilboard is not available, you may need to implement it based on
        your AprilBoard detection code or import it from your CS283 pset utilities.
    """
    try:
        from pupil_apriltags import Detector
    except ImportError:
        raise ImportError(
            "pupil_apriltags library is required for AprilBoard calibration. "
            "Install with: pip install pupil-apriltags"
        )

    # Load AprilBoard definitions (lists of dicts with 'tag_id' and 'center')
    at_coarseboard, at_fineboard = load_aprilboards()

    # Choose board list
    board_list = at_fineboard if board_type.lower().startswith("f") else at_coarseboard

    # Build a simple mapping: tag_id -> 3D center (X, Y, Z) in board coordinates.
    board_centers: dict[int, np.ndarray] = {}
    for entry in board_list:
        if not isinstance(entry, dict):
            continue
        if "tag_id" not in entry or "center" not in entry:
            continue
        tag_id = int(entry["tag_id"])
        center = np.asarray(entry["center"], dtype=np.float32).reshape(3)
        board_centers[tag_id] = center

    if not board_centers:
        raise RuntimeError("AprilBoard definitions do not contain tag_id/center entries.")

    # Extract frames from video
    cap = cv2.VideoCapture(calib_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {calib_video_path}")

    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % max(1, int(stride)) == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            if max_frames and len(frames) >= max_frames:
                break
        idx += 1
    cap.release()

    if not frames:
        raise RuntimeError(
            f"No frames extracted from {calib_video_path}. Try a smaller stride."
        )

    # Get image size from first frame
    h, w = frames[0].shape[:2]
    image_size = (w, h)  # OpenCV expects (width, height)

    # Create AprilTag detector
    at_detector = Detector(
        families="tag36h11",
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )

    # Collect 2D-3D correspondences (per-frame object/image point sets)
    calObjPoints, calImgPoints = [], []
    total_valid = 0
    max_points_seen = 0

    # Process frames; we've already subsampled by `stride` and `max_frames`,
    # so just iterate over the collected frames here.
    frames_to_process = frames

    for count, frame in enumerate(frames_to_process):
        # Convert RGB to grayscale for detection
        if len(frame.shape) == 3:
            img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            img = frame

        # Detect AprilTags using pupil_apriltags directly
        detections = at_detector.detect(img)
        print(f"[calib] Frame {count}: {len(detections)} raw tag detections")
        
        imgpoints_list = []  # 2D centers in image (u, v)
        objpoints_list = []  # 3D centers in board coordinates (X, Y, Z)

        for detection in detections:
            tag_id = int(detection.tag_id)

            # Use tag center in image as 2D point
            center_2d = np.asarray(detection.center, dtype=np.float32).reshape(2)

            # Look up corresponding 3D center from board definitions
            if tag_id not in board_centers:
                continue
            center_3d = board_centers[tag_id]  # shape (3,)

            imgpoints_list.append(center_2d)
            objpoints_list.append(center_3d)

        if not imgpoints_list:
            # No valid tag with a known 3D center for this frame
            continue

        imgpoints = np.vstack(imgpoints_list)  # (N, 2)
        objpoints = np.vstack(objpoints_list)  # (N, 3)
        num_pts = len(imgpoints)
        max_points_seen = max(max_points_seen, num_pts)
        print(
            f"[calib] Frame {count}: {num_pts} correspondences "
            f"(min_points={min_points})"
        )

        if num_pts >= min_points and len(objpoints) >= min_points:
            total_valid += 1
            calObjPoints.append(objpoints.astype("float32"))
            calImgPoints.append(imgpoints.astype("float32"))

    if not calObjPoints:
        raise RuntimeError(
            f"No valid frames met min_points={min_points}. "
            f"Maximum correspondences in any frame: {max_points_seen}. "
            f"Try lowering min_points or stride (more frames)."
        )

    # Perform camera calibration
    reprojerr, calMatrix, distCoeffs, calRotations, calTranslations = cv2.calibrateCamera(
        calObjPoints,
        calImgPoints,
        image_size,
        None,
        None,
        flags=None,
    )

    np.set_printoptions(precision=5, suppress=True)
    print("RMSE of reprojected points:", reprojerr)
    print("Distortion coefficients:", distCoeffs.ravel())

    np.set_printoptions(precision=2, suppress=True)
    print("Intrinsic camera matrix:\n", calMatrix)
    print(f"Total images used for calibration: {total_valid}")

    # Optionally save calibration
    if output_path:
        save_calibration(output_path, calMatrix, distCoeffs)

    return calMatrix, distCoeffs


__all__ = ["load_aprilboards", "calibrate_camera_from_aprilboard_video"]
