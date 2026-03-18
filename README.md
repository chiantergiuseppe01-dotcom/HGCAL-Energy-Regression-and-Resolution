# Electron Energy Regression in the CMS High-Granularity Calorimeter Prototype

A 3D convolutional neural network (CNN) for electron energy regression in the CMS High-Granularity Calorimeter (HGCAL) prototype, trained on simulated testbeam data. The model learns to correct for shower leakage and non-uniformities, achieving better energy resolution than the simple raw-sum baseline across the full beam energy range (10–350 GeV). Developed as a university project for a Machine Learning Techniques for Physics course, with training and inference running on Google Colab free-tier resources. Inspired by the original work: arXiv:2309.06582

---

## Physics context

The HGCAL is a sampling calorimeter with 28 silicon layers providing fine 3D spatial resolution. Each electron shower produces thousands of silicon hits (`rechits`) per event, recording position (x, y, z) and deposited energy. The naive energy estimator — summing all rechit energies — is degraded by shower leakage, dead material, and non-uniform response. This project treats the 3D hit pattern as a voxel image and trains a CNN to regress the true beam energy directly.

---

## Pipeline overview

```
hgcal_electron_data.h5
        │
        ▼
01_converting_into_chunks.ipynb   →  parquet_chunks/chunk_*.parquet
        │
        ├──▶ 02_data_analysis.ipynb          (EDA, Longo profiles, Molière radius)
        │
        ▼
03_voxelization.ipynb             →  dataset_voxel*.npy  +  target_energy*.npy
        │
        ▼
04_cnn_training.ipynb             →  cnn3d_final_model.pth
        │
        ▼
05_inference.ipynb                →  y_pred_gev.npy  +  y_true_gev.npy  +  energies_reco.npy
        │
        ▼
scripts/resolution_fit.py         →  resolution plots (PNG/PDF)
```

---

## Repository structure

```
hgcal-electron-energy-regression/
├── README.md
├── requirements.txt
├── environment.yml            # conda env with ROOT included
├── .gitignore
├── notebooks/
│   ├── 01_converting_into_chunks.ipynb
│   ├── 02_data_analysis.ipynb
│   ├── 03_voxelization.ipynb
│   ├── 04_cnn_training.ipynb
│   └── 05_inference.ipynb
├── scripts/
│   └── resolution_fit.py
└── data/
    └── README.md              # data format description and access instructions
```

---

## Model architecture (`HGCAL_Net`)

Input: `(B, 1, 28, 15, 15)` voxel tensor — one channel, 28 layers, 15×15 transverse grid.  
The network combines learned convolutional features with hand-crafted physics observables:

| Block | Details |
|---|---|
| Conv1 | `Conv3d(1→32, k=(1,3,3), padding=(0,1,1)` — captures transverse patterns within a single layer |
| Conv2 | `Conv3d(32→64, k=(3,1,1)), padding=(1,0,0)` + BatchNorm — captures longitudinal shower development |
| Conv3 | `Conv3d(64→128, k=3, padding=(1,2,2) dilation=(1,2,2))` + BatchNorm — wider receptive field |
| Bottleneck | `Conv3d(128→32, k=1)` |
| Spatial pool | `AdaptiveAvgPool3d((2,4,4))` |
| Global pool | `AdaptiveAvgPool3d(1)` |
| Physics features | Total energy (log-normalised), longitudinal centre-of-gravity, per-layer energy profile `E(z)`, lateral spread |
| FC head | `Linear(1065→256→128→1)` with SiLU activations and dropout |

The input voxels and target energies are log-transformed before training: `log1p(E × 0.01)` for voxels and `log1p(E)` for targets. Predictions are inverted with `expm1` at inference time.

---

## Results

Energy resolution is extracted by fitting the residual distribution `(E_pred − E_true)` in each energy bin with a Double-Sided Crystal Ball (DSCB) function using RooFit. The stochastic (S) and constant (C) terms are obtained from a fit to `σ(E)/E` vs `1/√E`:

| Model | S (%) | C (%) |
|---|---|---|
| CNN | 20.51% | 0.19% |
| SUM baseline | 21.19% | 0.62% |


---

## Setup

### Python environment (PyTorch + data processing)

```bash
pip install -r requirements.txt
```

### ROOT / RooFit (required only for `resolution_fit.py`)

ROOT is not pip-installable in the standard way. The recommended approach is via conda:

```bash
conda env create -f environment.yml
conda activate hgcal
```

Or install a pre-built binary from [https://root.cern/install](https://root.cern/install) and ensure `import ROOT` works in your Python environment.

### Google Colab

Notebooks 03–05 were developed on Google Colab with GPU acceleration (T4). Mount your Google Drive and update the path variables at the top of each notebook. Install `awkward` if not present:

```python
!pip install awkward
```

---

## Data

The raw dataset is publicly available on Zenodo:

**Bhargav Joshi, Alpana Alpana (2023). *Electron Energy Regression in High-Granularity Calorimeter Prototype*. Zenodo.**
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7504164.svg)](https://doi.org/10.5281/zenodo.7504164)

The file used in this project is `hgcal_electron_data_0001.h5.gz` (2.8 GB compressed). Extract it before use:
```bash
gunzip hgcal_electron_data_0001.h5.gz
```

The dataset consists of ~648k simulated electron events in the energy range 10–350 GeV. See [`data/README.md`](data/README.md) for the full HDF5 schema. Large files (`.h5`, `.npy`, `.parquet`, `.pth`) are excluded from this repository via `.gitignore`.
---

## Reproducing results

```bash
# 1. Convert HDF5 to Parquet chunks (run locally, ~648k events)
jupyter nbconvert --to notebook --execute notebooks/01_converting_into_chunks.ipynb

# 2. Exploratory data analysis
jupyter nbconvert --to notebook --execute notebooks/02_data_analysis.ipynb

# 3. Voxelization (run on Colab or GPU machine)
#    Output: dataset_voxel*.npy, target_energy*.npy

# 4. Train the CNN (Colab A100 recommended)
#    Output: cnn3d_final_model.pth

# 5. Inference
#    Output: y_pred_gev.npy, y_true_gev.npy, energies_reco.npy

# 6. Resolution fits (requires ROOT)
python scripts/resolution_fit.py
```

---

## Citation / acknowledgements

This work is inspired by and uses the dataset from:
> Bhargav Joshi et al. (2023). *Electron Energy Regression in High-Granularity Calorimeter Prototype*. [arXiv:2309.06582](https://arxiv.org/abs/2309.06582)

The HGCAL detector is described in the CMS Technical Design Report: [CMS-TDR-019](https://cds.cern.ch/record/2293646).
