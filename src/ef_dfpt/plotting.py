import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def plot_bandstructure(
    x_s, e0_s, ticks, labels,
    e02, e02_adiab, hs_idx, colors, ylim
):
    """
    Plot band structure with EF and adiabatic corrections.

    Parameters
    ----------
    x_s : ndarray
        Interpolated k-path coordinate.
    e0_s : ndarray
        Smoothed unperturbed band energies.
    ticks : list
        Positions of high-symmetry points along the path.
    labels : list
        Labels of high-symmetry points.
    e02 : ndarray
        Energies including non-adiabatic EF corrections.
    e02_adiab : ndarray
        Energies including adiabatic corrections.
    hs_idx : list
        Indices of high-symmetry k-points.
    colors : dict
        Dictionary with keys ``'ef'`` and ``'adiab'``.
    ylim : tuple
        y-axis limits (emin, emax).
    """
    plt.figure(figsize=(7, 5))

    for i, band in enumerate(e0_s.T):
        plt.plot(x_s, band, "k", lw=1, label="unperturbed" if i == 0 else None)

    for b in range(e02.shape[1]):
        first = b == 0
        plt.scatter(
            ticks, e02[hs_idx, b],
            s=60, marker="o", facecolors="none",
            edgecolors=colors["ef"],
            label="EF corrections" if first else None,
        )
        plt.scatter(
            ticks, e02_adiab[hs_idx, b],
            s=60, marker="x",
            c=colors["adiab"],
            label="adiabatic corrections" if first else None,
        )

    for t in ticks:
        plt.axvline(t, color="grey", lw=0.7)

    plt.axhline(0, color="grey", ls="--")
    plt.xticks(ticks, labels, fontsize=20)
    plt.ylabel(r"$E - E_f$ [eV]", fontsize=22)
    plt.ylim(*ylim)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_overlap(mat, title, cmap):
    """
    Plot an overlap matrix on a logarithmic color scale.

    Parameters
    ----------
    mat : ndarray
        Overlap matrix (nbnd, nbnd).
    title : str
        Plot title.
    cmap : str
        Matplotlib colormap name.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(
        np.abs(mat),
        cmap=cmap,
        norm=mcolors.LogNorm(vmin=1e-3, vmax=1),
    )
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()
