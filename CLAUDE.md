# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

V2V3D is a deep learning framework for **view-to-view denoised 3D reconstruction from Light-Field Microscopy (LFM)** images. The model takes multi-view light-field images (LFIs) and PSFs as input, and reconstructs a denoised 3D volumetric output.

## Commands

### Training
```bash
python train_model.py
# Or with a custom config:
python train_model.py -c Config/train.yaml
```

### Inference
```bash
python test_model.py
# Or with a custom config:
python test_model.py -c Config/test.yaml
```

Both scripts use `configargparse` — CLI flags override YAML config values.

## Architecture

### Data Flow
1. **Input**: Multi-view LF images (`Nnum × H × W` float32 tensors), normalized to mean ~0.2
2. **PSF loading**: `Nnum` PSF files from `PSF/` dir → stacked into `(Nnum, z_slices, psf_h, psf_w)` tensor
3. **View splitting**: Even-indexed views → `select_v`; odd-indexed → `remain_v` (cross-view self-supervision)
4. **Output**: 3D volume `(z_slices, H, W)` saved as `.tif`

### Model (`model.py`)
- **`Feature`**: Shared 2D feature extractor (multi-scale with pooling branches) applied per view
- **`warp_feats`** (in `utils.py`): FFT-based convolution of features with PSFs to align views in 3D space
- **`Unet`**: U-Net decoder that takes warped features from one view-set and reconstructs a 3D volume
- **`V2V3D`**: Top-level model — runs `Feature` on all views, warps with PSFs, then runs two parallel `Unet` branches (one per view subset), averages their outputs

### Training Strategy (`train_model.py`)
- Self-supervised: `unet1` reconstructs from `select_v` views, supervised by re-projecting via PSF against `remain_v` LF images (and vice versa)
- Loss = MSE + 0.5×FFT loss + 1e-3×positivity loss + `dc_weight`×de-crosstalk loss + `tv_weight`×z-direction TV loss
- LR: constant until `decay_init` epoch, then decays by `lr_decay` factor every `decay_every` epochs
- Checkpoints saved to `./Checkpoints/<timestamp>_V2V3D_F{feat_ch}_{dataset}/`
- TensorBoard logs saved to `./Log/`

### Key Files
- `model.py` — All network modules (`Feature`, `Unet`, `V2V3D`)
- `dataset.py` — `SyntheticData` dataset class; handles PSF loading, LF normalization, random crop/flip augmentation; `test_lf_names` list must be updated to match test data
- `utils.py` — FFT-based forward projection (`generate_fps`), PSF warping (`warp_feats`, `genWarpPSFs`), checkpoint saving, LR scheduling
- `loss.py` — Custom losses: `fftloss`, `deCrosstalk_loss`, `posloss` (positivity), `zloss` (z-TV)
- `Config/train.yaml` / `Config/test.yaml` — All hyperparameters

### Data Conventions
- LF `.tif` files are named with `fp` in the filename; the postfix after `fp` is auto-detected
- PSF files named `psf_1.tif` … `psf_{Nnum}.tif` inside the PSF directory
- Raw uint16 values >32767 are offset-corrected: `img = img - 32767`
- Amplitude normalization: each LF is scaled so its mean = 0.2
- To add new test samples, edit `self.test_lf_names` list in `dataset.py:SyntheticData.getTestLF`

### Output
- Training: intermediate reconstructions saved every `epochs/10` epochs to `./Results/`
- Testing: per-sample `{name}_recon.tif` files in `./Results/{checkpoint_name}_test/`
- `use_amp` flag in test config rescales output by the per-image amplitude factor before saving
