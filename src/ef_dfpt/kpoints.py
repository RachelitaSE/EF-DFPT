import numpy as np

def wrap_kpoint(k):
    """
    Wrap fractional k-point coordinates into the interval [0, 1).

    Parameters
    ----------
    k : ndarray
        Fractional k-point coordinates (..., 3).

    Returns
    -------
    ndarray
        Wrapped k-point coordinates with values in [0, 1).
    """
    k = np.round(k, 6)
    k[k < 0] += 1
    k[k >= 1] -= 1
    return k

def p2p(kpts, pi, pf):
    """
    Find k-point indices lying on a straight path between two points.

    The function identifies all k-points collinear with the segment
    connecting `pi` and `pf`, and orders them along the path.

    Parameters
    ----------
    kpts : ndarray
        Full k-point grid of shape (nk, 3).
    pi : ndarray
        Initial fractional k-point (3,).
    pf : ndarray
        Final fractional k-point (3,).

    Returns
    -------
    ndarray
        Indices of k-points lying on the path, ordered from pi to pf.
    """
    pi = wrap_kpoint(pi)
    pf = wrap_kpoint(pf)
    grid = wrap_kpoint(kpts)

    online = np.squeeze(
        np.argwhere(
            np.isclose(
                np.linalg.norm(np.cross(pi - pf, pi - grid), axis=1),
                0,
                atol=1e-5,
            )
        )
    )
    normals = np.einsum("k,pk", pf - pi, grid[online] - pi)
    mask = (normals >= 0) & (normals < np.dot(pf - pi, pf - pi))
    return online[mask][np.argsort(normals[mask])]

def make_kpath(rk, hsp):
    """
    Construct a concatenated k-point path from high-symmetry points.

    Parameters
    ----------
    rk : ndarray
        Full k-point grid of shape (nk, 3).
    hsp : list
        High-symmetry points of the form
        [[label, kx, ky, kz], ...].

    Returns
    -------
    kpath_conc : ndarray
        Concatenated indices of all k-points along the path.
    kpath_segments : list of ndarray
        List of index arrays for each path segment.
    """
    kpath = [p2p(rk, hsp[i][1:], hsp[i + 1][1:]) for i in range(len(hsp) - 1)]
    kpath.append([0])
    return np.concatenate(kpath), kpath
