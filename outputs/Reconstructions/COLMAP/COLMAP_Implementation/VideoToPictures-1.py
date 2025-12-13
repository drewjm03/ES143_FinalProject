#!/usr/bin/env python
"""
Extract frames from a video (.mp4, .mov, etc.) into a sequence of .jpg images.

Usage:
    # Save one frame every 100 frames
    python VideoToPictures-1.py \
        --input-video /path/to/video.MOV \
        --output-dir /path/to/output \
        --prefix frame \
        --step 100

    # Or use FPS-based sampling instead:
    python VideoToPictures-1.py \
        --input-video /path/to/video.MOV \
        --output-dir /path/to/output \
        --prefix frame \
        --fps 5
"""

from __future__ import annotations

import argparse
from pathlib import Path
import cv2


def video_to_images(
    input_video: str,
    output_dir: str,
    prefix: str = "frame",
    fps: float | None = None,
    step: int | None = None,
) -> str:
    """
    Extract a sequence of .jpg images from a video file.

    Args:
        input_video: Path to the .mp4 or .mov file.
        output_dir: Directory where extracted frames will be written.
        prefix: Prefix for output frame names (default: "frame").
        fps: Optional target extraction FPS. If None, no FPS-based downsampling.
        step: Optional frame step. If set, save one frame every `step` frames.
              If both step and fps are None, save every frame.

    Returns:
        Output directory path as a string.
    """
    video_path = Path(input_video)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

    # Decide how often to save a frame
    if step is not None:
        frame_interval = max(1, step)  # e.g. 100
    elif fps is not None:
        frame_interval = max(1, int(orig_fps / fps))
    else:
        frame_interval = 1  # save every frame

    print(f"[info] Video: {video_path}")
    print(f"[info] Original FPS: {orig_fps:.2f}")
    print(f"[info] Saving one frame every {frame_interval} frame(s)")

    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            out_path = out_dir / f"{prefix}_{saved_idx:05d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved_idx += 1

            if saved_idx % 50 == 0:
                print(f"[info] Saved {saved_idx} frames...")

        frame_idx += 1

    cap.release()

    print(f"[done] Extracted {saved_idx} frames to {out_dir}")
    return str(out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract JPG frames from a video.")
    parser.add_argument(
        "--input-video",
        type=str,
        required=True,
        help="Path to a .mp4 or .mov video file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where frames will be saved.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="frame",
        help="Prefix for saved frame names (default: frame).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Target extraction FPS. Ignored if --step is provided.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Save one frame every this many frames (e.g. 100). "
             "Overrides --fps if set.",
    )

    args = parser.parse_args()
    video_to_images(
        input_video=args.input_video,
        output_dir=args.output_dir,
        prefix=args.prefix,
        fps=args.fps,
        step=args.step,
    )


if __name__ == "__main__":
    main()
