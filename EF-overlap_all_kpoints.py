import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

Ry2eV = 13.605698066
data_dir_Mos2 = '/work/rachels/phd/MoS2/36x36/6-EF-tests/test-broadening/lorentzian/D5/'
data_dir_pen = '/work/rachels/phd/pentacene/original_struct/encut110/884/4-EF/lorentzian20/'
hsk_Mos2 = [[r"$\Gamma$", 0.0, 0.0, 0.0], \
                        [r"$\Lambda$", 1/6, 1/6, 0.0], \
                        [r"$K$", 1/3, 1/3, 0.0], \
                        [r"$M$", 0.5, 0.0, 0.0], \
                        [r"$\Gamma$", 0.0, 0.0, 0.0]]
hsk_pen = [[r"$\Gamma$", 0.0, 0.0, 0.0], \
                        [r"$X$", 0.5, 0, 0.0], \
                        [r"$Y$", 0, 0.5, 0.0], \
                        [r"$C$", 0.5, 0.5, 0.0],\
                        [r"$\Gamma$", 0.0, 0.0, 0.0]]

fontsize = 14 



with h5py.File(data_dir_Mos2+ 'EF_data.h5', 'r') as f:
    rk = f['mf_header/kpoints/rk'][()]
    occ = f['mf_header/kpoints/occ'][0, :, :]
    e0 = f['mf_header/kpoints/el'][0, :, :] * Ry2eV
    e2 = f['e2_corrections/energy_corrections'][()] * Ry2eV
    e2_adiab = f['e2_corrections/e2_adiab'][()] * Ry2eV

occ = np.sum(occ, axis = 1)
assert np.all(occ[0] == occ)
occ = int(occ[0])
print(f'Number of occupied states: {occ}')
nv =4
nc = 4
with h5py.File(data_dir_Mos2+ 'overlaps.h5', 'r') as f:
    overlap_old_new = f['overlaps/old_new'][:, occ - nv:occ + nc, occ - nv:occ + nc]
    overlap_new_new = f['overlaps/new_new'][:, occ - nv:occ + nc, occ - nv:occ + nc]
unique_high_symmetry_points=[]
for point in hsk_Mos2:
    if point not in unique_high_symmetry_points:
        unique_high_symmetry_points.append(point)

High_symmetry_points_indices = []
for point in unique_high_symmetry_points:
    idx = np.where(np.all(np.isclose(rk, point[1:]), axis=1))[0][0]
    High_symmetry_points_indices.append(idx)
fig2, axes2 = plt.subplots(
    nrows=len([overlap_old_new, overlap_new_new]),
    ncols=len(High_symmetry_points_indices),
    figsize=(4 * len(High_symmetry_points_indices), 8),
    constrained_layout=False
)
axes2 = np.atleast_2d(axes2)

for row_idx, (mat, name) in enumerate(zip(
        [overlap_old_new, overlap_new_new],
        ['old-new', 'new-new'])):
    for col_idx, (idx, point) in enumerate(zip(
            High_symmetry_points_indices,
            unique_high_symmetry_points)):
        ax = axes2[row_idx, col_idx]
        im2 = ax.imshow(np.abs(mat[idx]), cmap='Oranges',
                        norm=mcolors.LogNorm(vmin=1e-3, vmax=1))
        ax.set_title(f'{name} @ {point[0]}', fontsize=fontsize)
        ax.set_xlabel('Band index', fontsize=fontsize)
        ax.set_ylabel('Band index', fontsize=fontsize)
        ax.tick_params(axis='both', labelsize=fontsize-2)

fig2.tight_layout(rect=[0, 0, 0.86, 1])
cbar_ax2 = fig2.add_axes([0.88, 0.25, 0.02, 0.5])
cbar2 = fig2.colorbar(im2, cax=cbar_ax2)
cbar2.set_label('Magnitude', fontsize=fontsize)
cbar2.ax.tick_params(labelsize=fontsize-2)
#fig2.suptitle('Pentacene Overlap-same wfn', fontsize=fontsize+2, y=0.99)

plt.show()



with h5py.File(data_dir_pen+ 'EF_data.h5', 'r') as f:
    rk = f['mf_header/kpoints/rk'][()]
    occ = f['mf_header/kpoints/occ'][0, :, :]
    e0 = f['mf_header/kpoints/el'][0, :, :] * Ry2eV
    e2 = f['e2_corrections/energy_corrections'][()] * Ry2eV
    e2_adiab = f['e2_corrections/e2_adiab'][()] * Ry2eV

occ = np.sum(occ, axis = 1)
assert np.all(occ[0] == occ)
occ = int(occ[0])
print(f'Number of occupied states: {occ}')
nv =4
nc = 4
with h5py.File(data_dir_pen+ 'overlaps.h5', 'r') as f:
    overlap_old_new = f['overlaps/old_new'][:, occ - nv:occ + nc, occ - nv:occ + nc]
    overlap_new_new = f['overlaps/new_new'][:, occ - nv:occ + nc, occ - nv:occ + nc]
unique_high_symmetry_points=[]
for point in hsk_pen:
    if point not in unique_high_symmetry_points:
        unique_high_symmetry_points.append(point)

High_symmetry_points_indices = []
for point in unique_high_symmetry_points:
    idx = np.where(np.all(np.isclose(rk, point[1:]), axis=1))[0][0]
    High_symmetry_points_indices.append(idx)
fig2, axes2 = plt.subplots(
    nrows=len([overlap_old_new, overlap_new_new]),
    ncols=len(High_symmetry_points_indices),
    figsize=(4 * len(High_symmetry_points_indices), 8),
    constrained_layout=False
)
axes2 = np.atleast_2d(axes2)

for row_idx, (mat, name) in enumerate(zip(
        [overlap_old_new, overlap_new_new],
        ['old-new', 'new-new'])):
    for col_idx, (idx, point) in enumerate(zip(
            High_symmetry_points_indices,
            unique_high_symmetry_points)):
        ax = axes2[row_idx, col_idx]
        im2 = ax.imshow(np.abs(mat[idx]), cmap='Blues',
                        norm=mcolors.LogNorm(vmin=1e-3, vmax=1))
        ax.set_title(f'{name} @ {point[0]}', fontsize=fontsize)
        ax.set_xlabel('Band index', fontsize=fontsize)
        ax.set_ylabel('Band index', fontsize=fontsize)
        ax.tick_params(axis='both', labelsize=fontsize-2)

fig2.tight_layout(rect=[0, 0, 0.86, 1])
cbar_ax2 = fig2.add_axes([0.88, 0.25, 0.02, 0.5])
cbar2 = fig2.colorbar(im2, cax=cbar_ax2)
cbar2.set_label('Magnitude', fontsize=fontsize)
cbar2.ax.tick_params(labelsize=fontsize-2)
#fig2.suptitle('Pentacene Overlap-same wfn', fontsize=fontsize+2, y=0.99)

plt.show()