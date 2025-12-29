import h5py
import numpy as np
'''
This script creates a demo HDF5 file "EF_data_demo.h5" from the full EF data file "EF_data.h5" 
(this is just to check the scripts. Note: not physical results).
'''
src_file = "/work/rachels/phd/pentacene/original_struct/encut110/4-EF/EF_data.h5"
dst_file = "/work/rachels/phd/EF-DFPT/data/EF_data_demo.h5"

# --- demo sizes ---
nk_demo   = 5
nbnd_demo = 4
nq_demo   = 3
ns_demo   = 1
ngk_demo  = 10

def copy_attrs(src, dst):
    for k, v in src.attrs.items():
        dst.attrs[k] = v

with h5py.File(src_file, "r") as src, h5py.File(dst_file, "w") as dst:

    # -------------------------
    # Copy headers completely
    # -------------------------
    for grp in ["mf_header", "gkq_header", "gkq_mappings", "ef_header"]:
        src.copy(grp, dst)

    # -------------------------
    # gkq_data
    # -------------------------
    gkq_src = src["gkq_data"]
    gkq_dst = dst.create_group("gkq_data")
    copy_attrs(gkq_src, gkq_dst)

    gkq_dst.create_dataset(
        "energies",
        data=gkq_src["energies"][:nk_demo, :nbnd_demo]
    )

    gkq_dst.create_dataset(
        "frequencies",
        data=gkq_src["frequencies"][:nq_demo]
    )

    # -------------------------
    # corrections
    # -------------------------
    corr_src = src["corrections"]
    corr_dst = dst.create_group("corrections")
    copy_attrs(corr_src, corr_dst)

    # 1st order corrections
    dset = corr_src["1st_order_corrections"]
    corr_dst.create_dataset(
        "1st_order_corrections",
        data=dset[:nk_demo, :nbnd_demo, :ns_demo, :ngk_demo]
    )

    # 1st order v_NA
    dset = corr_src["1st_order_v_NA"]
    corr_dst.create_dataset(
        "1st_order_v_NA",
        data=dset[:nq_demo, :nk_demo, :nbnd_demo, :nbnd_demo]
    )

    # -------------------------
    # e2_corrections
    # -------------------------
    e2_src = src["e2_corrections"]
    e2_dst = dst.create_group("e2_corrections")
    copy_attrs(e2_src, e2_dst)

    for name in e2_src:
        dset = e2_src[name]
        e2_dst.create_dataset(
            name,
            data=dset[:nk_demo, :nbnd_demo]
        )

print("Demo file created:", dst_file)
