# EF-DFPT
Exact factorization density functional perturbation theory code analysis. Based on the paper: DOI: https://doi.org/10.1103/dmpv-zqdh

# EF-DFPT HDF5 Output Structure

This document describes the structure and meaning of the HDF5 output files produced by the EF-DFPT code.

Outputs:
* `EF_data.h5` — electron–phonon correction factors and energy shifts
* `new_WFN.h5` — corrected wavefunctions and energies
* `overlaps.h5` — overlap diagnostics between original and corrected wavefunctions
Inputs include WFN.h5, eph.h5, and phonon mode files.

## 1. `EF_data.h5` — Electron–Phonon Corrections

This file contains first- and second-order EF-DFPT quantities used to construct corrected energies and wavefunctions.
```
EF_data.h5
│
├── mf_header/
├── gkq_header/
├── gkq_mappings/
├── gkq_data/
│   ├── energies
│   └── frequencies
│
├── ef_header/
│   └── delta
│
├── corrections/
│   ├── 1st_order_corrections
|   |   Shape: (nk, nbnd, ns, ngk_max)
|   |   Type: complex128
|   |   Description: Contains the first-order prefactor to the wavefunction correction
│   ├── 1st_order_v_NA
|   |   Shape: (nq, nk, nbnd, nbnd)
|   |   Type: complex128
|   └── Description: The non-adiabatic 1st order additive potential matrix 
│
└── e2_corrections/
    ├── energy_corrections
    |   Shape: (nk, nbnd)
    |   Type: float64
    |   Description: non- adiabatic 2nd order energy correction term built from 4 terms listed bellow
    ├── e2_adiab
    |   Shape: (nk, nbnd)
    |   Type: float64
    |   Description: adiabatic 2nd order energy correction term following Guistino2017
    ├── e2_first_term
    ├── e2_second_term
    ├── e2_third_term
    └── e2_forth_term
```
### shape of the energy terms:
| Dataset          | Shape                  | Meaning                  |
| ---------------- | ---------------------- | ------------------------ |
| `e2_first_term`  | `(nq, nk, nbnd, nbnd)` | NA-FM term               | 
| `e2_second_term` | `(nq, nk, nbnd, nbnd)` | correction to NA-FM term |
| `e2_third_term`  | `(nq, nk, nbnd, nbnd)` | mixed two-phonon term    |
| `e2_forth_term`  | `(nq, nk, nbnd)`       | diagonal DW term         |   

## 2. `new_WFN.h5` — Corrected Wavefunctions
This file is a BerkeleyGW-style wavefunction file [wfn.h5](http://manual.berkeleygw.org/2.1/wfn_h5_spec/) containing corrected energies and coefficients.
```
new_WFN.h5
│
├── mf_header/
│   └── kpoints/el    ← corrected energies
│
└── wfns/
    ├── gvecs
    └── coeffs        ← corrected ψ + δψ (possibly normalized)
```

