import h5py
import numpy as np
from pathlib import Path
from .constants import Ry2eV


def load_h5_data(path):
    """
    Load DFPT electronic structure and EF corrections from an HDF5 file.

    This function supports both:
    - full production EF_data.h5 files
    - reduced demo EF_data.h5 files created by EF_data_demo.py

    Parameters
    ----------
    path : str or Path
        Path to the ``EF_data.h5`` file.

    Returns
    -------
    rk : ndarray
        Fractional k-point coordinates (nk, 3).
    occ : int
        Number of occupied bands.
    e0 : ndarray
        Unperturbed band energies (nk, nbnd) in eV.
    e2 : ndarray
        Non-adiabatic second-order energy corrections (nk, nbnd) in eV.
    e2_adiab : ndarray
        Adiabatic second-order energy corrections (nk, nbnd) in eV.
    """
    path = Path(path)

    with h5py.File(path, "r") as f:
        rk = f["mf_header/kpoints/rk"][()]

        # -------------------------
        # Handle occ (full vs demo)
        # -------------------------
        occ_ds = f["mf_header/kpoints/occ"][()]
        if occ_ds.ndim == 3:
            # Full data: (1, nk, nbnd)
            occ_arr = np.sum(occ_ds[0], axis=1)
            assert np.all(occ_arr[0] == occ_arr)
            occ = int(occ_arr[0])
        else:
            # Demo data: scalar occ
            occ = int(occ_ds[0])

        # -------------------------
        # Handle energies (full vs demo)
        # -------------------------
        el = f["mf_header/kpoints/el"][()]
        if el.ndim == 3:
            # Full data: (1, nk, nbnd)
            e0 = el[0] * Ry2eV
        else:
            # Demo data: (nk, nbnd)
            e0 = el * Ry2eV

        e2 = f["e2_corrections/energy_corrections"][()] * Ry2eV
        e2_adiab = f["e2_corrections/e2_adiab"][()] * Ry2eV

    return rk, occ, e0, e2, e2_adiab


def load_overlap_data(path, occ=None, nv=4, nc=4):
    """
    Load overlap matrices around the band gap.

    Supports both:
    - full overlap files (need slicing using occ)
    - demo overlap files (already sliced)

    Parameters
    ----------
    path : str or Path
        Path to ``overlaps.h5`` file.
    occ : int or None
        Number of occupied bands in the full calculation.
        If None or inconsistent with overlap size, slicing is skipped.
    nv : int
        Number of valence bands below VBM.
    nc : int
        Number of conduction bands above CBM.

    Returns
    -------
    dict
        Dictionary with keys ``'old_new'`` and ``'new_new'``.
    """
    with h5py.File(path, "r") as f:
        old_new = f["overlaps/old_new"][()]
        new_new = f["overlaps/new_new"][()]

    nbnd = old_new.shape[1]

    # Demo data: already sliced
    if occ is None or occ >= nbnd:
        return {
            "old_new": old_new,
            "new_new": new_new,
        }

    # Full data: slice around gap
    return {
        "old_new": old_new[:, occ - nv : occ + nc, occ - nv : occ + nc],
        "new_new": new_new[:, occ - nv : occ + nc, occ - nv : occ + nc],
    }
