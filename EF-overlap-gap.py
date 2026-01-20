import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

Ry2eV = 13.605698066
fontsize = 18

# =======================
# Paths
# =======================
data_dir_Mos2 = '/work/rachels/phd/MoS2/36x36/4-EF/lorentzian_normalized_d5/'
data_dir_pen  = '/work/rachels/phd/pentacene/original_struct/encut110/884/4-EF/lorentzian20/'

# =======================
# High-symmetry points
# =======================
k_Mos2 = np.array([1/3, 1/3, 0.0])  # K
k_pen  = np.array([0.5, 0.5, 0.0])  # C

nv = 4
nc = 4

# =======================
# Helper function
# =======================
def plot_new_new_overlap(data_dir, k_target, cmap):
    with h5py.File(data_dir + 'EF_data.h5', 'r') as f:
        rk = f['mf_header/kpoints/rk'][()]
        occ = f['mf_header/kpoints/occ'][0, :, :]

    occ = np.sum(occ, axis=1)
    assert np.all(occ[0] == occ)
    occ = int(occ[0])

    with h5py.File(data_dir + 'overlaps.h5', 'r') as f:
        overlap_new_new = f['overlaps/new_new'][
            :, occ - nv : occ + nc, occ - nv : occ + nc
        ]

    idx = np.where(np.all(np.isclose(rk, k_target), axis=1))[0][0]

    vbm_idx = nv-1
    cbm_idx = nv 

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(
        np.abs(overlap_new_new[idx]),
        cmap=cmap,
        norm=mcolors.LogNorm(vmin=1e-3, vmax=1)
    )



    ax.set_xticks([vbm_idx, cbm_idx])
    ax.set_yticks([vbm_idx, cbm_idx])
    ax.set_xticklabels(['VBM', 'CBM'], fontsize=fontsize-2, rotation=45)
    ax.set_yticklabels(['VBM', 'CBM'], fontsize=fontsize-2)

    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Magnitude', fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize - 2)

    plt.tight_layout()
    plt.show()

# Pentacene new@new @ C
plot_new_new_overlap(
    data_dir=data_dir_pen,
    k_target=k_pen,
    # title=r'Pentacene new@new @ $C$',
    cmap='Blues'
)

# MoS2 new@new @ K
plot_new_new_overlap(
    data_dir=data_dir_Mos2,
    k_target=k_Mos2,
    # title=r'MoS$_2$ new@new @ $K$',
    cmap='Oranges'
)

