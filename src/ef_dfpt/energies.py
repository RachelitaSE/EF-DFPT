import numpy as np
from scipy.interpolate import make_interp_spline

import numpy as np

def normalize_energies(e0, occ=None):
    """
    Align energies such that the valence band maximum (VBM) is at zero.

    Parameters
    ----------
    e0 : ndarray
        Band energies of shape (nk, nbnd).
    occ : int or None, optional
        Number of occupied bands in the *full* calculation.
        If None or inconsistent with e0.shape, the VBM is assumed
        to be at the center of the band window.

    Returns
    -------
    ndarray
        Energy array shifted so that the VBM is at 0 eV.
    """
    nbnd = e0.shape[1]

    if occ is None or occ - 1 >= nbnd:
        # Demo / truncated data: VBM at center of window
        vbm_idx = nbnd // 2 - 1
    else:
        # Full data
        vbm_idx = occ - 1

    vmax = np.max(e0[:, vbm_idx])
    return e0 - vmax


def smooth_bands(e, n_points=1000):
    """
    Align energies such that the VBM is set to zero.

    Parameters
    ----------
    e0 : ndarray
        Band energies (nk, nbnd).
    occ : int
        Number of occupied bands.

    Returns
    -------
    ndarray
        Energy array shifted so that max(e0[:, occ-1]) = 0.
    """
    x = np.arange(len(e))
    xs = np.linspace(0, len(e) - 1, n_points)
    spline = make_interp_spline(x, e, k=3)
    return xs, spline(xs)
