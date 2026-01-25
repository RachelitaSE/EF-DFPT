"""
EF-overlap-gap
==============

Visualize overlap matrices between DFPT-corrected wavefunctions
near the band gap.

This script plots the magnitude of the ``new@new`` overlap matrix
for bands around the VBM and CBM at a selected high-symmetry k-point.

The overlap matrix elements are shown on a logarithmic color scale
to emphasize small but non-zero couplings.

Use case
--------
This visualization is used to:
- assess band mixing induced by EF corrections
- identify coupling strength near the band gap
- compare behavior between different materials

Notes
-----
- Only a small window of bands around the gap is shown.
- Overlaps are read from ``overlaps.h5``.
- The choice of k-point and band window is hard-coded.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from ef_dfpt.io import load_h5_data, load_overlap_data
from ef_dfpt.paths import get_data_dir

fontsize = 18
nv = 4
nc = 4


def find_k_index(rk: np.ndarray, k_target: np.ndarray) -> int:
    """
    Find index of `k_target` in `rk` using np.isclose matching.

    Parameters
    ----------
    rk : ndarray
        Fractional k-points (nk, 3).
    k_target : ndarray
        Fractional k-point coordinates (3,).

    Returns
    -------
    int
        Index of matching k-point.
    """
    return np.where(np.all(np.isclose(rk, k_target), axis=1))[0][0]


def plot_new_new_overlap(data_dir: str, k_target: np.ndarray, cmap: str):
    """
    Plot |new_new| overlap submatrix around the gap at a target k-point.

    Parameters
    ----------
    data_dir : str
        Directory containing EF_data.h5 and overlaps.h5.
    k_target : ndarray
        Fractional target k-point (3,).
    cmap : str
        Matplotlib colormap name.

    Notes
    -----
    Matches original tick placement:
    - x/y ticks at VBM index (nv-1) and CBM index (nv).
    """
    rk, occ, *_ = load_h5_data(data_dir / "EF_data.h5")
    overlaps = load_overlap_data(data_dir / "overlaps.h5", occ=occ, nv=nv, nc=nc)

    idx = find_k_index(rk, k_target)
    overlap_new_new = overlaps["new_new"][idx]
    
    nbnd = overlaps["new_new"].shape[1]
    vbm_idx = nbnd // 2 - 1
    cbm_idx = nbnd // 2

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(
        np.abs(overlap_new_new),
        cmap=cmap,
        norm=mcolors.LogNorm(vmin=1e-3, vmax=1),
    )

    ax.set_xticks([vbm_idx, cbm_idx])
    ax.set_yticks([vbm_idx, cbm_idx])
    ax.set_xticklabels(["VBM", "CBM"], fontsize=fontsize - 2, rotation=45)
    ax.set_yticklabels(["VBM", "CBM"], fontsize=fontsize - 2)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Magnitude", fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize - 2)

    plt.tight_layout()
    plt.show()

def main():
    """
    Produce overlap plots for Pentacene@C and MoS2@K (same as original script).
    """
    
    DATA_DIR = get_data_dir()
    data_dir_Mos2 = DATA_DIR / "MoS2"
    data_dir_pen =  DATA_DIR / "pentacene" 

    # =======================
    # High-symmetry points
    # =======================
    k_Mos2 = np.array([1/3, 1/3, 0.0])  # K
    k_pen = np.array([0.5, 0.5, 0.0])   # C

    # Pentacene new@new @ C
    plot_new_new_overlap(
        data_dir=data_dir_pen,
        k_target=k_pen,
        # title=r'Pentacene new@new @ $C$',
        cmap="Blues"
    )

    # MoS2 new@new @ K
    plot_new_new_overlap(
        data_dir=data_dir_Mos2,
        k_target=k_Mos2,
        # title=r'MoS$_2$ new@new @ $K$',
        cmap="Oranges"
    )


if __name__ == "__main__":
    main()
