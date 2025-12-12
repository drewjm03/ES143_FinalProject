#!/usr/bin/env python
"""
Convert a folder of .jpg/.jpeg/.png images into an MP4 video.

Usage:

    python images_to_video.py \
        --input-dir data/frames \
        --output-dir outputs \
        --output-name timelapse.mp4 \
        --fps 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def images_to_mp4(
    input_dir: str,
    output_dir: str,
    output_name: str = "output.mp4",
    fps: int = 30,
) -> str:
    """
    Convert all .jpg/.jpeg/.png images in a folder to an MP4 video.
    """

    in_path = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.is_dir():
        raise ValueError(f"Input directory does not exist: {in_path}")

    # ------------------------------------------------------------
    # Collect images: JPG, JPEG, PNG
    # ------------------------------------------------------------
    exts = {".jpg", ".jpeg", ".png"}
    image_files = sorted([p for p in in_path.iterdir() if p.suffix.lower() in exts])

    if not image_files:
        raise ValueError(f"No .jpg/.jpeg/.png files found in {in_path}")

    # ------------------------------------------------------------
    # Read first image to determine video resolution
    # ------------------------------------------------------------
    first_img = cv2.imread(str(image_files[0]))
    if first_img is None:
        raise RuntimeError(f"Failed to read first image: {image_files[0]}")

    height, width = first_img.shape[:2]

    # ------------------------------------------------------------
    # Set up the MP4 writer
    # ------------------------------------------------------------
    output_path = out_dir / output_name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {output_path}")

    # ------------------------------------------------------------
    # Write all frames
    # ------------------------------------------------------------
    try:
        for idx, img_path in enumerate(image_files):
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[warn] Skipping unreadable image: {img_path}")
                continue

            # Resize if needed
            if img.shape[:2] != (height, width):
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

            writer.write(img)

            if idx % 50 == 0:
                print(f"[info] Wrote frame {idx + 1}/{len(image_files)}")
    finally:
        writer.release()

    print(f"[done] Wrote video to {output_path}")
    return str(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a folder of .jpg/.jpeg/.png images into an MP4 video."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to folder containing image files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where the MP4 will be written.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="output.mp4",
        help="Name of the output video file.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second.",
    )

    args = parser.parse_args()

    images_to_mp4(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        output_name=args.output_name,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
