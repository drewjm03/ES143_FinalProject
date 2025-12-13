"""
Command-line interface for the SfM pipeline.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

from sfm_app.ba.bundle_adjustment import run_bundle_adjustment
from sfm_app.calib.aprilboard_calib import calibrate_camera_from_aprilboard_video
from sfm_app.io.calib_io import save_calibration, save_scene_npz
from sfm_app.io.video_io import extract_frames_from_video, select_keyframes
from sfm_app.sfm_inc.incremental_sfm import run_sfm_from_frames
from sfm_app.viz.plotly_viz import plot_sfm_reconstruct


def main() -> None:
    """
    Main CLI entry point for SfM pipeline.

    Usage:
        sfm-from-video --calib-video path/to/calib.mp4 \\
                      --scene-video path/to/scene.mp4 \\
                      --board-type coarse \\
                      --output-dir out/
    """
    parser = argparse.ArgumentParser(
        description="Structure-from-Motion from video using calibration and scene videos"
    )
    parser.add_argument(
        "--calib-video",
        type=str,
        required=True,
        help="Path to calibration video file",
    )
    parser.add_argument(
        "--scene-video",
        type=str,
        required=True,
        help="Path to scene video file",
    )
    parser.add_argument(
        "--board-type",
        type=str,
        default="coarse",
        choices=["coarse", "fine"],
        help="Type of AprilBoard to use for calibration (default: coarse)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for calibration and scene files (default: output)",
    )
    parser.add_argument(
        "--max-keyframes",
        type=int,
        default=20,
        help="Maximum number of keyframes to use (default: 20)",
    )
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Skip calibration step and load existing calibration from output_dir",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate HTML visualization of the reconstruction",
    )
    parser.add_argument(
        "--baseframe1",
        type=int,
        default=None,
        help=(
            "Optional manual base frame index (in the extracted/undistorted frame list). "
            "If both --baseframe1 and --baseframe2 are provided, they override the "
            "default base keyframe selection."
        ),
    )
    parser.add_argument(
        "--baseframe2",
        type=int,
        default=None,
        help=(
            "Optional second manual base frame index (in the extracted/undistorted "
            "frame list). Must be used together with --baseframe1 to take effect."
        ),
    )
    parser.add_argument(
        "--skip-ba",
        action="store_true",
        help="Skip global bundle adjustment (use raw triangulated poses/points)",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Calibrate (or load existing calibration)
    calib_path = output_dir / "calibration.npz"

    if args.skip_calibration and calib_path.exists():
        print(f"Loading existing calibration from {calib_path}")
        from sfm_app.io.calib_io import load_calibration

        K, dist_coeffs = load_calibration(str(calib_path))
    else:
        print(f"Calibrating camera from {args.calib_video}...")
        try:
            K, dist_coeffs = calibrate_camera_from_aprilboard_video(
                args.calib_video,
                board_type=args.board_type,
                max_frames=100,
                output_path=str(calib_path),
            )
            print(f"Calibration saved to {calib_path}")
        except NotImplementedError as e:
            print(f"Warning: {e}")
            print(
                "Calibration requires AprilBoard detection implementation. "
                "Please implement the detection code or use --skip-calibration "
                "with an existing calibration file."
            )
            return

    print(f"Camera intrinsics K:\n{K}")

    # Step 2: Extract frames from scene video
    print(f"Extracting frames from {args.scene_video}...")
    frames = extract_frames_from_video(args.scene_video, every_n=5, max_frames=500)

    if len(frames) == 0:
        print("Error: No frames extracted from scene video")
        return

    print(f"Extracted {len(frames)} frames")

    # Step 3: Undistort frames
    print("Undistorting frames...")
    undistorted_frames = []
    h, w = frames[0].shape[:2]

    # Get optimal new camera matrix for undistortion
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), 1, (w, h))

    for frame in frames:
        # Convert RGB to BGR for cv2.undistort
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        undistorted_bgr = cv2.undistort(frame_bgr, K, dist_coeffs, None, new_K)
        # Convert back to RGB
        undistorted_rgb = cv2.cvtColor(undistorted_bgr, cv2.COLOR_BGR2RGB)
        undistorted_frames.append(undistorted_rgb)

    # Use the optimal K for further processing
    K = new_K

    # Step 4: Select keyframes
    print(f"Selecting up to {args.max_keyframes} keyframes...")
    keyframe_indices = select_keyframes(undistorted_frames, max_num=args.max_keyframes)
    print(f"Selected {len(keyframe_indices)} keyframes: {keyframe_indices}")

    if len(keyframe_indices) < 2:
        print("Error: Need at least 2 keyframes for SfM")
        return

    # Step 5: Run SfM
    print("Running incremental SfM...")
    scene = run_sfm_from_frames(
        undistorted_frames,
        K,
        keyframe_indices,
        baseframe1=args.baseframe1,
        baseframe2=args.baseframe2,
    )

    if len(scene.cameras) == 0 or len(scene.points3d) == 0:
        print("Error: SfM failed to reconstruct scene")
        return

    num_cams = len(scene.cameras)
    num_pts = len(scene.points3d)
    print(f"SfM reconstructed {num_cams} cameras and {num_pts} 3D points")

    # Step 6: Run bundle adjustment (optional)
    if args.skip_ba:
        print("Skipping bundle adjustment (--skip-ba set)")
    else:
        print("Running bundle adjustment...")
        # Run BA on a random subset of points with a small iteration limit
        # to keep runtime reasonable.
        scene = run_bundle_adjustment(scene, K, max_nfev=10)
        print("Bundle adjustment completed")

    # ------------------------------------------------------------------
    # Optional final 3D point filtering based on scene scale.
    #
    # Idea:
    #   - Compute camera centers C_i.
    #   - Let R_cam = max_i ||C_i|| (typical camera radius).
    #   - Discard 3D points whose norm ||X|| is far beyond the camera scale,
    #     e.g. ||X|| > 5 * R_cam.
    #
    # This removes extreme outliers that can dominate visualizations or
    # downstream processing while keeping points at a reasonable distance
    # from the camera rig.
    # ------------------------------------------------------------------
    if len(scene.cameras) > 0 and len(scene.points3d) > 0:
        Rs = np.stack([cam.R for cam in scene.cameras])
        ts = np.stack([cam.t for cam in scene.cameras])
        C = -(Rs.transpose(0, 2, 1) @ ts).squeeze(-1)  # (N_cams, 3)
        cam_norms = np.linalg.norm(C, axis=1)

        max_cam_radius = float(np.max(cam_norms)) if cam_norms.size > 0 else 0.0

        if max_cam_radius > 0.0:
            pts_xyz = np.array([pt.xyz for pt in scene.points3d])
            pt_norms = np.linalg.norm(pts_xyz, axis=1)

            max_allowed_norm = 5.0 * max_cam_radius
            keep_mask = pt_norms <= max_allowed_norm

            if not np.all(keep_mask):
                # Build mapping from old point IDs to new compact IDs.
                old_to_new: dict[int, int] = {}
                new_points = []

                for keep, pt in zip(keep_mask, scene.points3d):
                    if not keep:
                        continue
                    new_id = len(new_points)
                    old_to_new[pt.id] = new_id
                    pt.id = new_id
                    new_points.append(pt)

                # Filter observations and remap their point_ids.
                new_observations = []
                for obs in scene.observations:
                    if obs.point_id in old_to_new:
                        obs.point_id = old_to_new[obs.point_id]
                        new_observations.append(obs)

                # Update per-camera point_ids arrays if present.
                for cam in scene.cameras:
                    if cam.point_ids.size == 0:
                        continue
                    new_point_ids = -np.ones_like(cam.point_ids)
                    for idx, pid in enumerate(cam.point_ids):
                        if pid in old_to_new:
                            new_point_ids[idx] = old_to_new[pid]
                    cam.point_ids = new_point_ids

                removed = int((~keep_mask).sum())
                scene.points3d = new_points
                scene.observations = new_observations

                print(
                    "[sfm] Filtered 3D points by norm: "
                    f"removed {removed}, kept {len(scene.points3d)} "
                    f"with ||X|| <= {max_allowed_norm:.3f} "
                    f"(5x max camera radius {max_cam_radius:.3f})"
                )

    # Step 7: Save outputs
    scene_path = output_dir / "scene.npz"
    print(f"Saving scene to {scene_path}...")
    save_scene_npz(str(scene_path), scene, K)

    # Optional: Generate visualization
    if args.visualize:
        print("Generating visualization...")
        fig = plot_sfm_reconstruct(scene)
        viz_path = output_dir / "reconstruction.html"
        fig.write_html(str(viz_path))
        print(f"Visualization saved to {viz_path}")

    print("Pipeline completed successfully!")


if __name__ == "__main__":
    main()

