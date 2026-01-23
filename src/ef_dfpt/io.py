import h5py
import numpy as np
from .constants import Ry2eV

def load_h5_data(path):
    """
    Load DFPT electronic structure and EF corrections from an HDF5 file.

    Parameters
    ----------
    path : str
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
    with h5py.File(path, "r") as f:
        rk = f["mf_header/kpoints/rk"][()]
        occ_raw = f["mf_header/kpoints/occ"][0]
        e0 = f["mf_header/kpoints/el"][0] * Ry2eV #shape (nk, nbnd)
        e2 = f["e2_corrections/energy_corrections"][()] * Ry2eV #shape (nk, nbnd)
        e2_adiab = f["e2_corrections/e2_adiab"][()] * Ry2eV #shape (nk, nbnd)

    occ = np.sum(occ_raw, axis=1)
    assert np.all(occ[0] == occ)

    return rk, int(occ[0]), e0, e2, e2_adiab


def load_overlap_data(path, occ, nv=4, nc=4):
    """
    Load overlap matrices around the band gap.

    Parameters
    ----------
    path : str
        Path to ``overlaps.h5`` file.
    occ : int
        Number of occupied bands.
    nv : int, optional
        Number of valence bands below VBM (default: 4).
    nc : int, optional
        Number of conduction bands above CBM (default: 4).

    Returns
    -------
    dict
        Dictionary with keys ``'old_new'`` and ``'new_new'`` containing
        overlap matrices of shape (nk, nv+nc, nv+nc).
    """
    with h5py.File(path, "r") as f:
        return {
            "old_new": f["overlaps/old_new"][:, occ-nv:occ+nc, occ-nv:occ+nc],
            "new_new": f["overlaps/new_new"][:, occ-nv:occ+nc, occ-nv:occ+nc],
        }
