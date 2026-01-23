import numpy as np
from scipy.interpolate import make_interp_spline

def normalize_energies(e0, occ):
    """
    Shift energies so that the valence band maximum (VBM) is at zero.

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

    Notes
    -----
    - This normalization is performed independently for each material.
    """
    vmax = np.max(e0[:, occ - 1])
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
