#!/usr/bin/env python
"""
Extract and save two frames from a video at approximately given timestamps.

Usage (from repo root, in your venv):

    python extract_two_frames.py --video data/test_vids/Library.MOV --t1 25 --t2 28

This will:
  - Open the video.
  - Read its FPS.
  - Compute frame indices around t1 and t2 seconds.
  - Seek to those frames and save them as PNG images.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def grab_frame_at_time(video_path: str, t_seconds: float, out_path: Path) -> bool:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[extract] Could not open video: {video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print(f"[extract] Could not read FPS from video (got {fps}); aborting.")
        cap.release()
        return False

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = int(round(t_seconds * fps))
    frame_idx = max(0, min(frame_idx, frame_count - 1))

    print(
        f"[extract] Video FPS={fps:.3f}, total_frames={frame_count}, "
        f"t={t_seconds:.2f}s -> frame_idx={frame_idx}"
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        print(f"[extract] Failed to read frame at index {frame_idx}")
        return False

    # Save as PNG (BGR to RGB not needed for viewing, but we keep BGR for cv2.imwrite)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame)
    print(f"[extract] Saved frame at ~{t_seconds:.2f}s to {out_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract two frames from a video at approximately given timestamps."
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file.",
    )
    parser.add_argument(
        "--t1",
        type=float,
        default=25.0,
        help="First timestamp in seconds (default: 25).",
    )
    parser.add_argument(
        "--t2",
        type=float,
        default=28.0,
        help="Second timestamp in seconds (default: 28).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="debug_frames",
        help="Directory to save extracted frames (default: debug_frames).",
    )

    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    vpath = args.video
    print(f"[extract] Using video: {vpath}")

    f1 = out_dir / f"frame_{int(args.t1)}s.png"
    f2 = out_dir / f"frame_{int(args.t2)}s.png"

    ok1 = grab_frame_at_time(vpath, args.t1, f1)
    ok2 = grab_frame_at_time(vpath, args.t2, f2)

    if not (ok1 and ok2):
        print("[extract] One or both frames failed to extract.")
    else:
        print(f"[extract] Done. Frames saved to {out_dir}")


if __name__ == "__main__":
    main()


