"""
General calibration extractor for datasets that provide a `*_par.txt` file.

The expected format of each line in the par file is:

    imgname.png k11 k12 k13 k21 k22 k23 k31 k32 k33 r11 ... t3

This script extracts ONLY the intrinsic matrix K from the first line,
assumes zero distortion, and saves `calibration.npz` to the given output directory.
"""

import argparse
import numpy as np
from pathlib import Path
from sfm_app.io.calib_io import save_calibration


def load_intrinsics_from_par(par_path: Path) -> np.ndarray:
    """Reads the first valid data line of a *_par.txt file and extracts K."""
    with open(par_path, "r") as f:
        first_data_line = None
        for line in f:
            line = line.strip()
            if not line:
                continue  # skip empty lines
            if line.startswith("#"):
                continue  # skip comments if any
            parts = line.split()
            # We expect at least: imgname + 9 K entries
            if len(parts) >= 10:
                first_data_line = parts
                break

    if first_data_line is None:
        raise ValueError(f"Could not find a valid parameter line in {par_path}")

    # first token = image name, next 9 tokens = K
    try:
        k_vals = [float(x) for x in first_data_line[1:10]]
    except ValueError as e:
        raise ValueError(
            f"Failed to parse K entries from line in {par_path}:\n"
            f"{' '.join(first_data_line)}"
        ) from e

    K = np.array(k_vals, dtype=np.float64).reshape(3, 3)

    print(f"[calib] Loaded intrinsics K from {par_path}")
    print(K)

    return K


def main():
    parser = argparse.ArgumentParser(description="Save calibration.npz from *_par.txt")
    parser.add_argument(
        "--par-file",
        type=str,
        required=True,
        help="Path to the *_par.txt file containing K and extrinsics.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where calibration.npz will be saved.",
    )
    args = parser.parse_args()

    par_path = Path(args.par_file)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Extract intrinsic matrix
    K = load_intrinsics_from_par(par_path)

    # Dataset provides rectified images → assume zero distortion
    dist_coeffs = np.zeros(5, dtype=np.float64)

    # Save calibration
    calib_path = out_dir / "calibration.npz"
    save_calibration(str(calib_path), K, dist_coeffs)

    print(f"[calib] Saved calibration to {calib_path}")


if __name__ == "__main__":
    main()
