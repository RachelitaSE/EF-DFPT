"""
EF-bandstructure
================

Plot electronic band structures with second-order DFPT corrections.

This script compares:
- unperturbed Kohn-Sham band energies
- non-adiabatic electron-phonon (EF) corrections
- adiabatic second-order corrections

for selected materials along high-symmetry k-point paths.

The band structure is plotted as:
- smooth black lines: unperturbed energies
- open circles: EF-corrected energies at high-symmetry points
- crosses: adiabatic corrections at high-symmetry points

All energies are aligned such that the valence band maximum (VBM)
is set to zero.

Notes
-----
- Input data are read from ``EF_data.h5`` files.
- High-symmetry paths are hard-coded in the script.
- The script reproduces exactly the figures used in analysis
  without any data filtering or reweighting.
"""

from ef_dfpt import (
    load_h5_data, 
    make_kpath, 
    normalize_energies, 
    smooth_bands, 
    plot_bandstructure
)
from ef_dfpt.paths import get_data_dir
import numpy as np



DATA_DIR = get_data_dir()


data_dir_Mos2 = {
    "label": "MoS₂ - EF corrections",
    'data_path': DATA_DIR / "MoS2" / "EF_data.h5",
    'hsp': [
        [r"$\Gamma$", 0.0, 0.0, 0.0],
        [r"$\Lambda$", 1/6, 1/6, 0.0],
        [r"$K$", 1/3, 1/3, 0.0],
        [r"$M$", 0.5, 0.0, 0.0],
        [r"$\Gamma$", 0.0, 0.0, 0.0]
    ],
    'ylim': (-3, 4),
    # 'colors': {'ef': 'red', 'adiab': 'blue'},
    "colors": {"ef": "#F39E60", "adiab": "#E16A54"},
    "truncate_bands": None,  # keep all
}
data_dir_pen  = {
    "label": "Pentacene - EF corrections",
    'data_path': DATA_DIR / "pentacene" / "EF_data.h5",
    'hsp': [
        [r"$\Gamma$", 0.0, 0.0, 0.0],
        [r"$X$", 0.5, 0.0, 0.0],
        [r"$Y$", 0.0, 0.5, 0.0],
        [r"$C$", 0.5, 0.5, 0.0],
        [r"$\Gamma$", 0.0, 0.0, 0.0]
    ],
    # 'ylim': (-1.5, 2.5),
    'ylim': (-1.6, 2.5),
    # 'colors': {'ef': 'red', 'adiab': 'blue'},
    "colors": {"ef": "#57B4BA", "adiab": "#034C53"},
    "truncate_bands": 152, # match original analysis
}

data_cases = [data_dir_Mos2, data_dir_pen]


def compute_tick_positions(hsp, kpath_segments):
    """
    Compute x-axis tick positions and labels for a k-path plot.

    Parameters
    ----------
    hsp : list
        High-symmetry points of the form [[label, kx, ky, kz], ...].
    kpath_segments : list of ndarray
        List of k-point index arrays corresponding to each path segment.

    Returns
    -------
    tick_positions : list of int
        Cumulative positions for each high-symmetry point along the concatenated path.
    labels : list of str
        Tick labels (LaTeX strings) for each high-symmetry point.
    """
    tick_positions = []
    labels = []
    offset = 0
    for point, seg in zip(hsp, kpath_segments):
        tick_positions.append(offset)
        labels.append(point[0])
        offset += len(seg)
    return tick_positions, labels


def run_case(cfg): 
    """
    Generate a band structure plot for a single material/configuration.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary describing the system to plot.
        Expected keys include:
        - ``data_dir`` : str
            Path to the ``EF_data.h5`` file.
        - ``hsp`` : list
            High-symmetry k-point path in the form
            ``[[label, kx, ky, kz], ...]``.
        - ``colors`` : dict
            Color specification with keys ``'ef'`` and ``'adiab'``.
        - ``ylim`` : tuple
            y-axis limits for the band structure plot.

    Notes
    -----
    For the given configuration, this function:
    - loads unperturbed and corrected band energies
    - aligns energies to the valence band maximum (VBM)
    - constructs the k-point path
    - interpolates band energies for smooth plotting
    - overlays EF and adiabatic corrections at high-symmetry points

    The resulting figure is shown interactively and not saved to disk.
    """
    rk, occ, e0, e2, e2_adiab = load_h5_data(cfg["data_path"])
    print(f"{cfg['label']} — occupied states: {occ}")

    # # Energy alignment
    # e0 = normalize_energies(e0, occ)
    # e02 = normalize_energies(e0 + e2, occ)
    # e02_adiab = normalize_energies(e0 + e2_adiab, occ)

    # # Optional band truncation (Pentacene: first 152 bands)
    # if cfg.get("truncate_bands") is not None:
    #     nb = int(cfg["truncate_bands"])
    #     e0 = e0[:, :nb]
    #     e02 = e02[:, :nb]
    #     e02_adiab = e02_adiab[:, :nb]

    # --- Unperturbed energies ---
    e0 = normalize_energies(e0, occ)

    # Optional band truncation (Pentacene)
    if cfg.get("truncate_bands") is not None:
        nb = int(cfg["truncate_bands"])
        e0 = e0[:, :nb]
        e2 = e2[:, :nb]
        e2_adiab = e2_adiab[:, :nb]

    assert e0.shape == e2.shape == e2_adiab.shape, "Energy arrays must have matching shapes before combination"
    # --- Corrected energies (match original script order) ---
    e02 = normalize_energies(e0 + e2, occ)
    e02_adiab = normalize_energies(e0 + e2_adiab, occ)
    


    # Build k-path and high-symmetry indices
    kpath_conc, kpath_segments = make_kpath(rk, cfg["hsp"])
    hs_indices = [seg[0] for seg in kpath_segments]

    # Smooth curves for unperturbed energies along the path
    x_s, e0_s = smooth_bands(e0[kpath_conc])

    # Ticks/labels
    tick_positions, labels = compute_tick_positions(cfg["hsp"], kpath_segments)

    # Plot using package plotting helper
    plot_bandstructure(
        x_s=x_s,
        e0_s=e0_s,
        ticks=tick_positions,
        labels=labels,
        e02=e02,
        e02_adiab=e02_adiab,
        hs_idx=hs_indices,
        colors=cfg["colors"],
        ylim=cfg["ylim"],
    )


def main():
    """
    Generate band structure plots with EF and adiabatic corrections.

    This function loops over all predefined data cases and produces
    band structure plots along the specified high-symmetry k-point paths.

    For each case, it:
    - loads unperturbed and corrected band energies
    - aligns energies to the valence band maximum (VBM)
    - constructs the k-point path
    - interpolates bands for smooth visualization
    - overlays EF and adiabatic corrections at high-symmetry points

    Notes
    -----
    - All configuration (paths, materials, colors) is hard-coded.
    - Figures are shown interactively and not saved to disk.
    - The numerical behavior matches the original analysis exactly.
    """
    for cfg in data_cases:
        run_case(cfg)


if __name__ == "__main__":
    main()



# def run_case(cfg):
#     """
#     Generate a band structure plot for a single material/configuration.

#     Parameters
#     ----------
#     cfg : dict
#         Configuration dictionary describing the system to plot.
#         Expected keys include:
#         - ``data_dir`` : str
#             Path to the ``EF_data.h5`` file.
#         - ``hsp`` : list
#             High-symmetry k-point path in the form
#             ``[[label, kx, ky, kz], ...]``.
#         - ``colors`` : dict
#             Color specification with keys ``'ef'`` and ``'adiab'``.
#         - ``ylim`` : tuple
#             y-axis limits for the band structure plot.

#     Notes
#     -----
#     For the given configuration, this function:
#     - loads unperturbed and corrected band energies
#     - aligns energies to the valence band maximum (VBM)
#     - constructs the k-point path
#     - interpolates band energies for smooth plotting
#     - overlays EF and adiabatic corrections at high-symmetry points

#     The resulting figure is shown interactively and not saved to disk.
#     """
#     rk, occ, e0, e2, e2_adiab = load_h5_data(cfg["data_dir"])

#     e0 = normalize_energies(e0, occ)
#     e02 = normalize_energies(e0 + e2, occ)
#     e02_adiab = normalize_energies(e0 + e2_adiab, occ)

#     kpath, segments = make_kpath(rk, cfg["hsp"])
#     hs_indices = [seg[0] for seg in segments]

#     x_s, e0_s = smooth_bands(e0[kpath])

#     tick_positions, labels, offset = [], [], 0
#     for label, seg in zip(cfg["hsp"], segments):
#         tick_positions.append(offset)
#         labels.append(label[0])
#         offset += len(seg)

#     plot_bandstructure(
#         x_s,
#         e0_s,
#         tick_positions,
#         labels,
#         e02,
#         e02_adiab,
#         hs_indices,
#         cfg["colors"],
#         cfg["ylim"],
#     )

# def main():
#     """
#     Generate band structure plots with EF and adiabatic corrections.

#     This function loops over all predefined data cases and produces
#     band structure plots along the specified high-symmetry k-point paths.

#     For each case, it:
#     - loads unperturbed and corrected band energies
#     - aligns energies to the valence band maximum (VBM)
#     - constructs the k-point path
#     - interpolates bands for smooth visualization
#     - overlays EF and adiabatic corrections at high-symmetry points

#     Notes
#     -----
#     - All configuration (paths, materials, colors) is hard-coded.
#     - Figures are shown interactively and not saved to disk.
#     - The numerical behavior matches the original analysis exactly.
#     """
#     for cfg in data_cases:
#         run_case(cfg)

# if __name__ == "__main__":
#     main()
