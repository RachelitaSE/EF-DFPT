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
        occ = f['mf_header/kpoints/occ'][0, :, :]
        e2_nk = f['e2_corrections/energy_corrections'][()] * ryd2ev #shape (nk, nbnd)
        e2_adiab_nk = f['e2_corrections/e2_adiab'][()] * ryd2ev #shape (nk, nbnd)
        e_nk = f['mf_header/kpoints/el'][0, :, :]* ryd2ev #shape (nk, nbnd)
        #nk = f['headers/nk'][()]
        #nq = f['headers/nq'][()]
        #nbnd = f['headers/nbnd'][()]
    return e2_nk, e2_adiab_nk, e_nk




def compute_avg_data_with_band_filter(e_nk, e2_adiab_nk, e2_na_nk,
                                      Emin=-2.0, Emax=6.0):
    """
    Average over k-points **only for bands whose mean energy lies between Emin and Emax**.
    """
    # Average over k for selection
    avg_e_all = np.mean(e_nk, axis=0)
    # Select band indices
    band_mask = (avg_e_all >= Emin) & (avg_e_all <= Emax)
    print(f"Selected {np.sum(band_mask)} bands out of {len(avg_e_all)}")
    # Apply mask
    e_nk_sel = e_nk[:, band_mask]
    e2_adiab_nk_sel = e2_adiab_nk[:, band_mask]
    e2_na_nk_sel = e2_na_nk[:, band_mask]
    # Recompute averages only for selected bands
    avg_e = np.mean(e_nk_sel, axis=0)
    avg_e_adiab = np.mean(e_nk_sel + e2_adiab_nk_sel, axis=0)
    avg_e_na = np.mean(e_nk_sel + e2_na_nk_sel, axis=0)
    # Corrections
    delta_adiab = np.abs(avg_e_adiab - avg_e)
    delta_na = np.abs(avg_e_na - avg_e)
    return avg_e, avg_e_adiab, avg_e_na, delta_adiab, delta_na

# Load data
e2_nk_pen, e2_adiab_nk_pen, e_nk_pen = read_data(data_dir_pen)
print("Data loaded for Pen")
print('e2_nk',e2_nk_pen.shape,'e2_nk_adi', e2_adiab_nk_pen.shape,'e_nk', e_nk_pen.shape)
e2_nk_mose2, e2_adiab_nk_mose2, e_nk_mose2 = read_data(data_dir_mos2)
print("Data loaded for MoSe2")
print('e2_nk',e2_nk_mose2.shape,'e2_nk_adi', e2_adiab_nk_mose2.shape,'e_nk', e_nk_mose2.shape)

avg_e_pen, avg_e_adiab_pen, avg_e_na_pen, delta_adiab_pen, delta_na_pen = \
    compute_avg_data_with_band_filter(e_nk_pen, e2_adiab_nk_pen, e2_nk_pen)
avg_e_mose2, avg_e_adiab_mose2, avg_e_na_mose2, delta_adiab_mose2, delta_na_mose2 = \
    compute_avg_data_with_band_filter(e_nk_mose2, e2_adiab_nk_mose2, e2_nk_mose2)

# ---- Plotting ----
fig = plt.figure(figsize=(12, 6))
gs = GridSpec(1, 2, width_ratios=[3,1], figure=fig)  # 3:2 ratio
ax1 = fig.add_subplot(gs[0])  # Main plot
ax2 = fig.add_subplot(gs[1])  # Log scale plot

# --- Left Plot: Identity with error bars ---
# Identity line
identity_min = min(avg_e_pen.min(), avg_e_mose2.min())
identity_max = max(avg_e_pen.max(), avg_e_mose2.max())
ax1.plot([identity_min, identity_max], [identity_min, identity_max], 'k--', linewidth=1.2, label='Identity')

# Adiabatic
ax1.errorbar(avg_e_pen, avg_e_adiab_pen, yerr=delta_adiab_pen, fmt='o', color='#034C53', label='Pen Adiabatic')
ax1.errorbar(avg_e_mose2, avg_e_adiab_mose2, yerr=delta_adiab_mose2, fmt='o', color='#E16A54', label='MoSe₂ Adiabatic')

# Non-Adiabatic
ax1.errorbar(avg_e_pen, avg_e_na_pen, yerr=delta_na_pen, fmt='x', color='#57B4BA', label='Pen Non-Adiabatic')
ax1.errorbar(avg_e_mose2, avg_e_na_mose2, yerr=delta_na_mose2, fmt='x', color='#F39E60', label='MoSe₂ Non-Adiabatic')
ax1.set_xlim(-4,7)
ax1.set_ylim(-4,7)
ax1.set_xlabel('Original Energy $ \epsilon_{nk} $ [eV]', fontsize=16)
ax1.set_ylabel('Corrected Energy $\epsilon_{nk} +  \epsilon^{(2)}_{nk}$ [eV]', fontsize=16)
ax1.set_title('$\epsilon_{nk} +  \epsilon^{(2)}_{nk}$', fontsize=18)
#ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(fontsize=14)

# --- Right Plot: Log-scale magnitudes ---
ax2.semilogy(avg_e_pen, delta_adiab_pen, 'o', color='#034C53', label='Pen Adiabatic')
ax2.semilogy(avg_e_pen, delta_na_pen, 'x', color='#57B4BA', label='Pen Non-Adiabatic')
ax2.semilogy(avg_e_mose2, delta_adiab_mose2, 'o', color='#E16A54', label='MoSe₂ Adiabatic')
ax2.semilogy(avg_e_mose2, delta_na_mose2, 'x', color='#F39E60', label='MoSe₂ Non-Adiabatic')

ax2.set_xlabel('$\epsilon_{nk}$ [eV]', fontsize=16)
ax2.set_ylabel('$|\epsilon^{(2)}_{nk}|$ [eV]', fontsize=16)
ax2.set_title('$\epsilon^{(2)}_{nk}$ Magnitudes', fontsize=18)
#ax2.legend(loc='center left', bbox_to_anchor=(1.2, 0.5), fontsize=14, borderaxespad=0.)
ax2.set_xlim(-4,7)
ax2.set_xticks([-2, 0, 2, 4, 6])
ax2.tick_params(axis='x', labelsize=16)
ax2.tick_params(axis='y', labelsize=16)
plt.tight_layout()
# plt.savefig("corrections_side_by_side.pdf")
plt.show()
