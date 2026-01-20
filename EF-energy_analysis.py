import matplotlib.pyplot as plt
import numpy as np
import h5py
from matplotlib.gridspec import GridSpec

ryd2ev = 13.605698066
data_dir_mos2 = '/work/rachels/phd/MoS2/36x36/6-EF-tests/test-broadening/lorentzian/D5/'
data_dir_pen  = '/work/rachels/phd/pentacene/original_struct/encut110/884/4-EF/lorentzian20/'

def read_data(data_dir):
    with h5py.File(data_dir + 'EF_data.h5', 'r') as f:    
        rk = f['mf_header/kpoints/rk'][()]
        occ_raw = f['mf_header/kpoints/occ'][0, :, :]
        e2_nk = f['e2_corrections/energy_corrections'][()] * ryd2ev #shape (nk, nbnd)
        e2_adiab_nk = f['e2_corrections/e2_adiab'][()] * ryd2ev #shape (nk, nbnd)
        e_nk = f['mf_header/kpoints/el'][0, :, :]* ryd2ev #shape (nk, nbnd)
    occ = np.sum(occ_raw, axis=1)
    assert np.all(occ[0] == occ)
    return e2_nk, e2_adiab_nk, e_nk, int(occ[0]),rk

def normalize_energies(e0, occ):
    vmax = np.max(e0[:, occ - 1])
    e0 -= vmax 
    return e0



e2_mos2, e2_adiab_mos2, e0_mos2, occ_mos2,rk_mos2 = read_data(data_dir_mos2)
e2_pen, e2_adiab_pen, e0_pen, occ_pen, rk_pen = read_data(data_dir_pen)
e0_mos2 = normalize_energies(e0_mos2, occ_mos2)
e0_pen = normalize_energies(e0_pen, occ_pen)

k_Mos2 = np.array([1/3, 1/3, 0.0])  # K
k_pen  = np.array([0.5, 0.5, 0.0])  # C


nv = 4
nc = 4

idx_pen = np.where(np.all(np.isclose(rk_pen, k_pen), axis=1))[0][0]
idx_mos2 = np.where(np.all(np.isclose(rk_mos2, k_Mos2), axis=1))[0][0]
vbm_idx = nv-1
cbm_idx = nv 
# ---- Plotting ----
fig, ax = plt.subplots(figsize=(4, 6))  # rectangular

ax.semilogy(e0_pen[idx_pen,:152], np.abs(e2_adiab_pen[idx_pen]), 'o', color='#034C53', label='Pen Adiabatic')
ax.semilogy(e0_pen[idx_pen,:152], np.abs(e2_pen[idx_pen]), 'x', color='#57B4BA', label='Pen Non-Adiabatic')
ax.semilogy(e0_mos2[idx_mos2], np.abs(e2_adiab_mos2[idx_mos2]), 'o', color='#E16A54', label='MoS₂ Adiabatic')
ax.semilogy(e0_mos2[idx_mos2], np.abs(e2_mos2[idx_mos2]), 'x', color='#F39E60', label='MoS₂ Non-Adiabatic')

# Vertical dashed line at HOB = 0
ax.axvline(0.0, color='k', linestyle='--', linewidth=1.2)

ax.set_xlabel(r'$\epsilon_{nk} - \epsilon_{\mathrm{VBM}}$ [eV]', fontsize=20)
ax.set_ylabel(r'$|\epsilon^{(2)}_{nk}|$ [eV]', fontsize=22)
# ax.set_title(r'$\epsilon^{(2)}_{nk}$ Magnitudes', fontsize=18)

ax.set_xlim(-3, 4)
ax.tick_params(axis='both', labelsize=20)

plt.tight_layout()
plt.show()

