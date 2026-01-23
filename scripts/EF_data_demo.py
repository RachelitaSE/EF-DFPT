import h5py
import numpy as np

# SRC = "/work/rachels/phd/pentacene/original_struct/encut110/4-EF/EF_data.h5"
# DST = "EF_data_pen.h5"
SRC = "/work/rachels/phd/MoS2/4-EF/u_overl_1/EF_data.h5"
DST = "EF_data_MoS2.h5"


def load_h5_data(path):
    with h5py.File(path, "r") as f:
        rk = f["mf_header/kpoints/rk"][()]
        occ_raw = f["mf_header/kpoints/occ"][0, :, :]
        e0 = f["mf_header/kpoints/el"][0, :, :] 
        e2 = f["e2_corrections/energy_corrections"][()] 
        e2_adiab = f["e2_corrections/e2_adiab"][()] 

    occ = np.sum(occ_raw, axis=1)
    assert np.all(occ[0] == occ)
    return rk, int(occ[0]), e0, e2, e2_adiab

rk, occ, e0, e2, e2_adiab = load_h5_data(SRC)
cond_bands = occ+10
val_bands= occ - 10
print(f"Original number of bands: {e0.shape}, truncating to {e0[:, val_bands:cond_bands].shape} bands.")
with h5py.File(DST, "w") as f:
    f.create_dataset("mf_header/kpoints/rk", data=rk)
    f.create_dataset("mf_header/kpoints/occ", data=np.array([occ]))
    f.create_dataset("mf_header/kpoints/el", data=e0[:,val_bands:cond_bands])
    f.create_dataset("e2_corrections/energy_corrections", data=e2[:,val_bands:cond_bands])
    f.create_dataset("e2_corrections/e2_adiab", data=e2_adiab[:,val_bands:cond_bands])
print(f"Demo EF data written to {DST}")
print(f"rk shape: {rk.shape}, occ: {occ}, e0 shape: {e0.shape}, e2 shape: {e2.shape}, e2_adiab shape: {e2_adiab.shape}")

