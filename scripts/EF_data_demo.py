import h5py
import numpy as np

# SRC = "/work/rachels/phd/pentacene/original_struct/encut110/884/4-EF/lorentzian20/"
# DST = "/work/rachels/phd/EF-DFPT/data/pentacene/"
SRC = "/work/rachels/phd/MoS2/36x36/6-EF-tests/test-broadening/lorentzian/D5/"
DST = "/work/rachels/phd/EF-DFPT/data/MoS2/"


def load_h5_data(path):
    with h5py.File(path + 'EF_data.h5', "r") as f:
        rk = f["mf_header/kpoints/rk"][()]
        occ_raw = f["mf_header/kpoints/occ"][0, :, :]
        e0 = f["mf_header/kpoints/el"][0, :, :] 
        e2 = f["e2_corrections/energy_corrections"][()] 
        e2_adiab = f["e2_corrections/e2_adiab"][()] 

    occ_arr = np.sum(occ_raw, axis=1)
    assert np.all(occ_arr[0] == occ_arr)
    occ = int(occ_arr[0])
    nv = 4
    nc = 4
    with h5py.File(path + 'overlaps.h5', "r") as g:
        overlap_old_new = g['overlaps/old_new'][:, occ - nv:occ + nc, occ - nv:occ + nc]
        overlap_new_new = g['overlaps/new_new'][:, occ - nv:occ + nc, occ - nv:occ + nc]
    return rk, occ, e0, e2, e2_adiab, overlap_old_new, overlap_new_new


rk, occ, e0, e2, e2_adiab,overlap_old_new, overlap_new_new = load_h5_data(SRC)
cond_bands = occ+10
val_bands= occ - 10
print(f"Original number of bands: {e0.shape}, truncating to {e0[:, val_bands:cond_bands].shape} bands.")
with h5py.File(DST+ 'EF_data.h5', "w") as f:
    f.create_dataset("mf_header/kpoints/rk", data=rk)
    f.create_dataset("mf_header/kpoints/occ", data=np.array([occ]))
    f.create_dataset("mf_header/kpoints/el", data=e0[:,val_bands:cond_bands])
    f.create_dataset("e2_corrections/energy_corrections", data=e2[:,val_bands:cond_bands])
    f.create_dataset("e2_corrections/e2_adiab", data=e2_adiab[:,val_bands:cond_bands])
print(f"Demo EF data written to {DST}")
print(f"rk shape: {rk.shape}, occ: {occ}, e0 shape: {e0.shape}, e2 shape: {e2.shape}, e2_adiab shape: {e2_adiab.shape}")

with h5py.File(DST + 'overlaps.h5', "w") as g:
    g.create_dataset('overlaps/old_new', data=overlap_old_new)
    g.create_dataset('overlaps/new_new', data=overlap_new_new)
print(f"Demo overlaps.h5 written to {DST}")