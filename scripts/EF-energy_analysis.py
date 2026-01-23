"""
EF-energy_analysis
==================

Analyze and compare magnitudes of second-order energy corrections as a function
of band energy, at a chosen high-symmetry k-point for each material.


This script plots:
- |εₙₖ^(2)| vs εₙₖ - ε_VBM

on a logarithmic scale for both:
- adiabatic corrections
- non-adiabatic (EF) corrections

and compares different materials at selected k-points.

Use case
--------
This plot is intended to highlight:
- the energy dependence of second-order corrections
- differences between adiabatic and non-adiabatic treatments
- material-specific trends near and away from the band edges

Notes
-----
- Energies are aligned to the valence band maximum (VBM).
- A single representative high-symmetry k-point is selected
  for each material.
- Band truncation (e.g. first 152 bands) is intentional and
  mirrors the original analysis.
"""

import numpy as np
import matplotlib.pyplot as plt

from ef_dfpt.io import load_h5_data
from ef_dfpt.energies import normalize_energies


def find_k_index(rk: np.ndarray, k_target: np.ndarray) -> int:
    """
    Find the index of the k-point in `rk` matching `k_target`.

    Parameters
    ----------
    rk : ndarray
        Fractional k-points (nk, 3).
    k_target : ndarray
        Fractional k-point coordinates (3,).

    Returns
    -------
    int
        Index in rk where rk[idx] == k_target (within isclose tolerance).

    Notes
    -----
    This matches the original selection logic:
    `np.where(np.all(np.isclose(rk, k_target), axis=1))[0][0]`.
    """
    return np.where(np.all(np.isclose(rk, k_target), axis=1))[0][0]


def main():
    """
    Load MoS2 and Pentacene datasets and plot correction magnitudes vs energy.

    Preserves original plot styling and band truncation behavior.
    """
    data_dir_mos2 = "/work/rachels/phd/MoS2/36x36/6-EF-tests/test-broadening/lorentzian/D5/"
    data_dir_pen = "/work/rachels/phd/pentacene/original_struct/encut110/884/4-EF/lorentzian20/"

    # Load using package IO
    rk_mos2, occ_mos2, e0_mos2, e2_mos2, e2_adiab_mos2 = load_h5_data(data_dir_mos2 + "EF_data.h5")
    rk_pen, occ_pen, e0_pen, e2_pen, e2_adiab_pen = load_h5_data(data_dir_pen + "EF_data.h5")

    # Normalize to VBM = 0 (same logic as before)
    e0_mos2 = normalize_energies(e0_mos2, occ_mos2)
    e0_pen = normalize_energies(e0_pen, occ_pen)

    # Target k-points (same as before)
    k_Mos2 = np.array([1 / 3, 1 / 3, 0.0])  # K
    k_pen = np.array([0.5, 0.5, 0.0])       # C

    idx_mos2 = find_k_index(rk_mos2, k_Mos2)
    idx_pen = find_k_index(rk_pen, k_pen)

    # ---- Plotting (same look/behavior) ----
    fig, ax = plt.subplots(figsize=(4, 6))

    # Pentacene band truncation to :152 matches the old script
    ax.semilogy(e0_pen[idx_pen, :152], np.abs(e2_adiab_pen[idx_pen, :152]),
                "o", color="#034C53", label="Pen Adiabatic")
    ax.semilogy(e0_pen[idx_pen, :152], np.abs(e2_pen[idx_pen, :152]),
                "x", color="#57B4BA", label="Pen Non-Adiabatic")

    ax.semilogy(e0_mos2[idx_mos2], np.abs(e2_adiab_mos2[idx_mos2]),
                "o", color="#E16A54", label="MoS₂ Adiabatic")
    ax.semilogy(e0_mos2[idx_mos2], np.abs(e2_mos2[idx_mos2]),
                "x", color="#F39E60", label="MoS₂ Non-Adiabatic")

    ax.axvline(0.0, color="k", linestyle="--", linewidth=1.2)
    ax.set_xlabel(r"$\epsilon_{nk} - \epsilon_{\mathrm{VBM}}$ [eV]", fontsize=20)
    ax.set_ylabel(r"$|\epsilon^{(2)}_{nk}|$ [eV]", fontsize=22)
    ax.set_xlim(-3, 4)
    ax.tick_params(axis="both", labelsize=20)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
