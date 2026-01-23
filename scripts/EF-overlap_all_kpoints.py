"""
EF-overlap_all_kpoints
======================

Plot overlap matrices at all unique high-symmetry k-points along
a chosen k-path, for each material.


For each high-symmetry point, this script displays:
- old@new overlap matrices
- new@new overlap matrices

stacked in rows, allowing direct comparison across k-points.

Use case
--------
This script is intended for:
- diagnosing k-dependent wavefunction mixing
- checking consistency of DFPT wavefunctions
- visual inspection of overlap structure across symmetry points

Notes
-----
- High-symmetry points are deduplicated automatically.
- Overlaps are shown on a logarithmic color scale.
- Separate figures are produced for each material.
- The script is intentionally verbose to mirror the original
  exploratory analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from ef_dfpt.io import load_h5_data, load_overlap_data

fontsize = 18
nv = 4
nc = 4


def unique_hsp(hsk):
    """
    Deduplicate high-symmetry point entries while preserving order.

    Parameters
    ----------
    hsk : list
        High-symmetry points as [[label, kx, ky, kz], ...].

    Returns
    -------
    list
        Deduplicated list, preserving first occurrence order.

    Notes
    -----
    Matches original logic:
    `if point not in unique_high_symmetry_points: append`.
    """
    out = []
    for p in hsk:
        if p not in out:
            out.append(p)
    return out


def find_hsp_indices(rk: np.ndarray, hsp_list: list) -> list:
    """
    Map each high-symmetry point in `hsp_list` to its k-point index in `rk`.

    Parameters
    ----------
    rk : ndarray
        Fractional k-point grid (nk, 3).
    hsp_list : list
        High-symmetry points [[label, kx, ky, kz], ...].

    Returns
    -------
    list of int
        Indices in rk for each high-symmetry point.
    """
    indices = []
    for p in hsp_list:
        idx = np.where(np.all(np.isclose(rk, p[1:]), axis=1))[0][0]
        indices.append(idx)
    return indices



def plot_overlap_grid(data_dir: str, hsk: list, cmap: str):
    """
    Plot old@new and new@new overlaps at all unique high-symmetry points.

    Parameters
    ----------
    data_dir : str
        Directory containing EF_data.h5 and overlaps.h5.
    hsk : list
        High-symmetry points [[label, kx, ky, kz], ...].
    cmap : str
        Matplotlib colormap name.

    Notes
    -----
    Preserves the original figure layout and shared colorbar positioning.
    """
    rk, occ, *_ = load_h5_data(data_dir + "EF_data.h5")
    print(f"Number of occupied states: {occ}")

    overlaps = load_overlap_data(data_dir + "overlaps.h5", occ=occ, nv=nv, nc=nc)
    overlap_old_new = overlaps["old_new"]
    overlap_new_new = overlaps["new_new"]

    hsp_list = unique_hsp(hsk)
    hsp_indices = find_hsp_indices(rk, hsp_list)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=len(hsp_indices),
        figsize=(4 * len(hsp_indices), 8),
        constrained_layout=False,
    )
    axes = np.atleast_2d(axes)

    mats = [overlap_old_new, overlap_new_new]
    names = ["old-new", "new-new"]

    im = None
    for row_idx, (mat, name) in enumerate(zip(mats, names)):
        for col_idx, (idx, point) in enumerate(zip(hsp_indices, hsp_list)):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(
                np.abs(mat[idx]),
                cmap=cmap,
                norm=mcolors.LogNorm(vmin=1e-3, vmax=1),
            )
            ax.set_title(f"{name} @ {point[0]}", fontsize=fontsize)
            ax.set_xlabel("Band index", fontsize=fontsize)
            ax.set_ylabel("Band index", fontsize=fontsize)
            ax.tick_params(axis="both", labelsize=fontsize - 2)

    # Match original colorbar placement
    fig.tight_layout(rect=[0, 0, 0.86, 1])
    cbar_ax = fig.add_axes([0.88, 0.25, 0.02, 0.5])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Magnitude", fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize - 2)

    plt.show()
    

def main():
    """
    Run overlap-grid plots for MoS2 and Pentacene (same as original script).
    """
    data_dir_Mos2 = "/work/rachels/phd/MoS2/36x36/6-EF-tests/test-broadening/lorentzian/D5/"
    data_dir_pen = "/work/rachels/phd/pentacene/original_struct/encut110/884/4-EF/lorentzian20/"

    hsk_Mos2 = [
        [r"$\Gamma$", 0.0, 0.0, 0.0],
        [r"$\Lambda$", 1/6, 1/6, 0.0],
        [r"$K$", 1/3, 1/3, 0.0],
        [r"$M$", 0.5, 0.0, 0.0],
        [r"$\Gamma$", 0.0, 0.0, 0.0],
    ]

    hsk_pen = [
        [r"$\Gamma$", 0.0, 0.0, 0.0],
        [r"$X$", 0.5, 0, 0.0],
        [r"$Y$", 0, 0.5, 0.0],
        [r"$C$", 0.5, 0.5, 0.0],
        [r"$\Gamma$", 0.0, 0.0, 0.0],
    ]

    plot_overlap_grid(data_dir=data_dir_Mos2, hsk=hsk_Mos2, cmap="Oranges")
    plot_overlap_grid(data_dir=data_dir_pen, hsk=hsk_pen, cmap="Blues")


if __name__ == "__main__":
    main()
