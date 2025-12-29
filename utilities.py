import numpy as np
import h5py
from scipy.interpolate import make_interp_spline

Ry2eV = 13.605698066  # BEGIN: Constants

def get_color_schemes():
    """
    Return a dictionary containing all available color schemes
    for EF and adiabatic corrections, including 'set1' and 'set2'.
    You can easily extend this to more palettes.
    """
    return {
        "set1": {
            "ef": "#F39E60",
            "adiab": "#E16A54",
            "ef_dot": "#F39E60",
            "adiab_dot": "#E16A54",
        },
        "set2": {
            "ef": "#57B4BA",
            "adiab": "#034C53",
            "ef_dot": "#57B4BA",
            "adiab_dot": "#034C53",
        },
    }
def p2p(kpts, pi, pf):
    pi = np.round(pi, 6)
    pi[pi < 0] += 1
    pi[pi >= 1] -= 1
    pf = np.round(pf, 6)
    pf[pf < 0] += 1
    pf[pf >= 1] -= 1
    grid = np.round(kpts, 6)
    grid[grid < 0] += 1
    grid[grid >= 1] -= 1
    online = np.squeeze(np.argwhere(np.isclose(np.linalg.norm(np.cross(pi - pf, pi - grid), axis = 1), 0, atol = 1E-5)))
    normals = np.einsum('k, pk', pf - pi, grid[online] - pi)
    online = online[normals >= 0]
    normals = normals[normals >= 0]
    online = online[normals < np.dot(pf - pi, pf - pi)]
    normals = normals[normals < np.dot(pf - pi, pf - pi)]
    path = online[np.argsort(normals)]
    
    return path


def wrap_kpoint(k):
    """Wrap fractional coordinates into [0,1)."""
    k = np.round(k, 6)
    k[k < 0] += 1
    k[k >= 1] -= 1
    return k


def get_smooth(y, n_points=1000):
    x = np.arange(len(y))
    spline = make_interp_spline(x, y, k=3)
    x_s = np.linspace(0, len(y) - 1, n_points)
    y_s = spline(x_s)
    return x_s, y_s


def load_h5_data(path):
    with h5py.File(path, "r") as f:
        rk = f["mf_header/kpoints/rk"][()]
        occ_raw = f["mf_header/kpoints/occ"][0, :, :]
        e0 = f["mf_header/kpoints/el"][0, :, :] * Ry2eV
        e2 = f["e2_corrections/energy_corrections"][()] * Ry2eV
        e2_adiab = f["e2_corrections/e2_adiab"][()] * Ry2eV

    occ = np.sum(occ_raw, axis=1)
    assert np.all(occ[0] == occ)
    return rk, int(occ[0]), e0, e2, e2_adiab


def make_kpath(rk, hsp):
    kpath = [p2p(rk, hsp[i][1:], hsp[i + 1][1:]) for i in range(len(hsp) - 1)]
    kpath.append([0])
    return np.concatenate(kpath), kpath
