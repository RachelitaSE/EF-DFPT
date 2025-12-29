import numpy as np
import matplotlib.pyplot as plt
from utilities import (load_h5_data,make_kpath,get_smooth)
import sys
# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

colors = {
    "ef_dot": "#57B4BA",
    "adiab_dot": "#034C53",
}

# Example: modify as needed
data_cases = [
    {
        "label": "MoS₂ - EF corrections",
        "data_dir": "/work/rachels/phd/MoS2/4-EF/u_overl_1/EF_data.h5",
        "fermi": 0.8272,
        "hsp": [
            [r"$\Gamma$", 0, 0, 0],
            [r"$\Lambda$", 1/6, 1/6, 0],
            [r"$K$", 1/3, 1/3, 0],
            [r"$M$", 0.5, 0, 0],
            [r"$\Gamma$", 0, 0, 0],
        ],
        "ylim": (-3, 4),
        "colors": {"ef": "#F39E60", "adiab": "#E16A54"},
    },

    {
        "label": "Pentacene - EF corrections",
        "data_dir": "/work/rachels/phd/pentacene/original_struct/encut110/4-EF/EF_data.h5",
        "fermi": 2.328,
        "hsp": [
            [r"$\Gamma$", 0, 0, 0],
            [r"$X$", 0.5, 0, 0],
            [r"$C$", 0.5, 0.5, 0],
            [r"$Y$", 0, 0.5, 0],
            [r"$\Gamma$", 0, 0, 0],
        ],
        "ylim": (-1.6, 2.5),
        "colors": {"ef": "#57B4BA", "adiab": "#034C53"},
    },
]

def main():

   

    cfg = data_cases[0]
    rk, occ, e0, e2, e2_adiab = load_h5_data(cfg["data_dir"])
    print(f"{cfg['label']} — occupied states: {occ}")   
    # Normalize energies
    vmax = np.max(e0[:, occ - 1])
    e0 -= vmax 
    e02 = e0 + e2
    e02 -= np.max(e02[:, occ - 1])
    e02_adiab = e0 + e2_adiab
    e02_adiab -= np.max(e02_adiab[:, occ - 1])
    # Build k-path
    kpath_conc, kpath_segments = make_kpath(rk, cfg["hsp"])
    hs_indices = [seg[0] for seg in kpath_segments]
    # Smooth curves
    x_s, e0_s = get_smooth(e0[kpath_conc])
    # Prepare ticks
    tick_positions = []
    labels = []
    offset = 0
    for label, seg in zip(cfg["hsp"], kpath_segments):
        tick_positions.append(offset)
        labels.append(label[0])
        offset += len(seg)
    # ----- PLOT -----
    plt.figure(figsize=(7, 5))
    for i, band in enumerate(e0_s.T):   # assuming e0_s is shape (npoints, nbands)
        if i == 0:
            plt.plot(x_s, band, color="k", lw=1, label="unperturbed")
        else:
            plt.plot(x_s, band, color="k", lw=1)
    nbands = e0.shape[1]
    for b in range(nbands):
        first = (b == 0)
        plt.scatter(
            tick_positions,
            e02[hs_indices, b],
            s=40, marker="o", facecolors="none",
            edgecolors=cfg["colors"]["ef"],
            label="EF corrections" if first else None
        )
        plt.scatter(
            tick_positions,
            e02_adiab[hs_indices, b],
            s=40, marker="x",
            c=cfg["colors"]["adiab"],
            label="adiabatic corrections" if first else None
        )
    # Decorations
    for t in tick_positions:
        plt.axvline(t, color="grey", lw=0.5)
    plt.axhline(0, color="grey", ls="--", lw=1)
    plt.xticks(tick_positions, labels, fontsize=16)
    plt.ylabel(r"$E - E_f$ [eV]", fontsize=20)
    plt.ylim(*cfg["ylim"])
    plt.yticks(fontsize=16)
    plt.legend(loc="upper right", fontsize=14)
    plt.title(cfg["label"])
    plt.tight_layout()
    plt.show()

    cfg = data_cases[1]
    rk, occ, e0, e2, e2_adiab = load_h5_data(cfg["data_dir"])
    print(f"{cfg['label']} — occupied states: {occ}")
    
    # Normalize energies
    vmax = np.max(e0[:, occ - 1])
    e0 -= vmax - cfg["fermi"]
    e0 = e0[:,:152]
    e02 = e0 + e2 - cfg["fermi"]
    e02 -= np.max(e02[:, occ - 1])

    e02_adiab = e0 + e2_adiab - cfg["fermi"]
    e02_adiab -= np.max(e02_adiab[:, occ - 1])

    # Build k-path
    kpath_conc, kpath_segments = make_kpath(rk, cfg["hsp"])
    hs_indices = [seg[0] for seg in kpath_segments]

    # Smooth curves
    x_s, e0_s = get_smooth(e0[kpath_conc])

    # Prepare ticks
    tick_positions = []
    labels = []
    offset = 0
    for label, seg in zip(cfg["hsp"], kpath_segments):
        tick_positions.append(offset)
        labels.append(label[0])
        offset += len(seg)

    # ----- PLOT -----
    plt.figure(figsize=(7, 5))
    for i, band in enumerate(e0_s.T):   # assuming e0_s is shape (npoints, nbands)
        if i == 0:
            plt.plot(x_s, band, color="k", lw=1, label="unperturbed")
        else:
            plt.plot(x_s, band, color="k", lw=1)

    nbands = e0.shape[1]

    for b in range(nbands):
        first = (b == 0)
        plt.scatter(
            tick_positions,
            e02[hs_indices, b],
            s=40, marker="o", facecolors="none",
            edgecolors=cfg["colors"]["ef"],
            label="EF corrections" if first else None
        )
        plt.scatter(
            tick_positions,
            e02_adiab[hs_indices, b],
            s=40, marker="x",
            c=cfg["colors"]["adiab"],
            label="adiabatic corrections" if first else None
        )

    # Decorations
    for t in tick_positions:
        plt.axvline(t, color="grey", lw=0.5)
    plt.axhline(0, color="grey", ls="--", lw=1)

    plt.xticks(tick_positions, labels, fontsize=16)
    plt.ylabel(r"$E - E_f$ [eV]", fontsize=20)
    plt.ylim(*cfg["ylim"])
    plt.yticks(fontsize=16)

    plt.legend(loc="upper right", fontsize=14)
    plt.title(cfg["label"])
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# RUN BOTH CASES
# ------------------------------------------------------------

if __name__ == "__main__":
    
    main()
