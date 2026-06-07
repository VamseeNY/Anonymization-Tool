"""
nifti_viewer.py
---------------
Quick visual check of a NIfTI file.

Plots 9 evenly spaced axial slices from the volume using matplotlib.
Also prints key header information (shape, voxel sizes, data range,
cal_min/cal_max window values).

Usage
-----
    python nifti_viewer.py <path_to_nifti>

Optional: pass a second argument to set the output PNG path.
If not provided, the plot is displayed interactively.

Example
-------
    python nifti_viewer.py output/nifti/patient_5CDDFE3D5457/series_CT_8788982331181411.nii.gz
    python nifti_viewer.py series.nii.gz slices.png
"""

import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
import matplotlib.pyplot as plt


def inspect_and_plot(nifti_path: str, output_png: str = None) -> None:
    path = Path(nifti_path)
    if not path.exists():
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)

    img = nib.load(str(path))
    hdr = img.header
    data = img.get_fdata()

    # ── Print header info ──
    print("=" * 55)
    print("NIfTI HEADER INFO")
    print("=" * 55)
    print(f"  File       : {path.name}")
    print(f"  Shape      : {data.shape}")
    print(f"  Voxel size : {hdr.get_zooms()}")
    print(f"  Data type  : {data.dtype}")
    print(f"  Data range : min={data.min():.1f}  max={data.max():.1f}")
    print(f"  cal_min    : {hdr['cal_min']}")
    print(f"  cal_max    : {hdr['cal_max']}")
    print(f"  descrip    : {hdr['descrip'].tobytes().decode('utf-8', errors='replace').strip()}")
    print(f"  db_name    : {hdr['db_name'].tobytes().decode('utf-8', errors='replace').strip()}")
    print()

    # ── Determine display window ──
    cal_min = float(hdr["cal_min"])
    cal_max = float(hdr["cal_max"])

    if cal_min == 0.0 and cal_max == 0.0:
        # No window set — use 2nd/98th percentile to avoid outlier stretching
        vmin = float(np.percentile(data, 2))
        vmax = float(np.percentile(data, 98))
        window_label = f"Auto window (2-98th pct): [{vmin:.0f}, {vmax:.0f}]"
    else:
        vmin = cal_min
        vmax = cal_max
        window_label = f"NIfTI cal window: [{vmin:.0f}, {vmax:.0f}]"

    print(f"  Display window: {window_label}")
    print()

    # ── Handle 2D edge case ──
    if data.ndim == 2:
        data = data[:, :, np.newaxis]

    n_slices = data.shape[2]

    # ── Pick 9 evenly spaced slices ──
    n_plots   = min(9, n_slices)
    indices   = np.linspace(0, n_slices - 1, n_plots, dtype=int)

    # ── Plot ──
    cols = 3
    rows = (n_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 4))
    axes = np.array(axes).flatten()

    fig.suptitle(
        f"{path.name}\n{window_label}",
        fontsize=10,
        y=1.01,
    )

    for i, slice_idx in enumerate(indices):
        ax = axes[i]
        # Axial slice — transpose so left/right is correct
        slice_data = data[:, :, slice_idx].T
        ax.imshow(
            slice_data,
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
            origin="lower",
            aspect="equal",
        )
        ax.set_title(f"Slice {slice_idx + 1}/{n_slices}", fontsize=9)
        ax.axis("off")

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    if output_png:
        plt.savefig(output_png, dpi=150, bbox_inches="tight")
        print(f"[INFO] Saved plot to: {output_png}")
    else:
        plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python nifti_viewer.py <path_to_nifti> [output.png]")
        print("Example:")
        print("  python nifti_viewer.py output/nifti/patient_5CDDFE3D5457/series_CT_xyz.nii.gz")
        print("  python nifti_viewer.py series.nii.gz slices.png")
        sys.exit(1)

    nifti_path = sys.argv[1]
    output_png = sys.argv[2] if len(sys.argv) > 2 else None

    # Use non-interactive backend if saving to file
    if output_png:
        matplotlib.use("Agg")

    inspect_and_plot(nifti_path, output_png)