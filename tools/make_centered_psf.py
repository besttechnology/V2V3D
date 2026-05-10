"""Generate PSF_centered.pt for Phase 2 of the Hybrid Renderer.

Two products:
1. psf_centered: each PSF[u, z] is bilinear-shifted so its (measured) centroid
   lands at V2V3D's discrete kernel center ((ph-1)//2, (pw-1)//2). After this,
   the centered PSF can be used in `mode='centered_affine'` and contributes
   no extra shift beyond what the affine center provides.
2. centroid offset tables (in (H-axis, W-axis) order, as a shift to be added
   to the Gaussian's voxel position to obtain the projected image position):
     - centroid_offset_measured[u, z]  = measured_centroid_abs - kernel_center
     - centroid_offset_affine[u, z]    = affine prediction (from psf_params_M1.pt)
       Phase 2 production uses `_affine`. The `_measured` table is for the
       Phase 2 sanity baseline (Phase 2 vs Phase 1 — should match closely).

Convention notes:
- `compute_moments` here computes centroid in **(H-axis, W-axis) absolute
  PSF index** order. (The hybrid renderer uses gaussian_utils mu=(H,W,D),
  so we keep the centroid table in that same order.)
- `psf_fit.py` measured `(d_x, d_y) = (W-axis, H-axis)` shifts per z. We
  swap them when building `centroid_offset_affine`.
- Two distinct kernel centers:
    `discrete_kc  = ((ph-1)//2, (pw-1)//2) = (63, 63)`
        — V2V3D's `generate_fps` crops with `int((rb-1)/2)`, so this is
          where the centered PSF's centroid must land.
    `continuous_kc = ((ph-1)/2, (pw-1)/2) = (63.5, 63.5)`
        — the geometric center of the PSF tensor (psf_fit's image_center).
  The de-centering bilinear shift targets `discrete_kc`, but the saved
  `centroid_offset` table stores offsets relative to `continuous_kc`. The
  0.5 px difference is exactly the (mu_world − voxel_idx) offset, so the
  renderer's splat formula `round(mu + offset)` works directly without
  any -0.5 fudge factor.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

# Make project root importable when running from `tools/`.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import load_psfs


def compute_centroid_abs(psf_2d: torch.Tensor) -> tuple[float, float]:
    """Return (centroid_h, centroid_w) in absolute PSF index, weighted by mass.

    Negative values are clipped (centroid only meaningful for non-negative).
    """
    p = psf_2d.clamp(min=0)
    H, W = p.shape
    mass = p.sum().clamp_min(1e-20)
    yy = torch.arange(H, dtype=p.dtype, device=p.device).view(H, 1)
    xx = torch.arange(W, dtype=p.dtype, device=p.device).view(1, W)
    cy = (p * yy).sum() / mass
    cx = (p * xx).sum() / mass
    return cy.item(), cx.item()


def fft_shift_2d(img: torch.Tensor, shift_h: float, shift_w: float) -> torch.Tensor:
    """FFT phase shift: out[i,j] = img[i - shift_h, j - shift_w] for non-integer
    shifts via Fourier-domain phase ramp. Exact for band-limited signals
    (modulo periodic-boundary wrap). Real input → real output (small float
    imag part discarded).

    img: (..., H, W).
    """
    H, W = img.shape[-2:]
    device = img.device
    dtype = img.dtype
    # Phase ramps. fftfreq returns frequencies in cycles/sample (range [-0.5, 0.5)).
    fy = torch.fft.fftfreq(H, device=device, dtype=dtype).view(H, 1)
    fx = torch.fft.fftfreq(W, device=device, dtype=dtype).view(1, W)
    # exp(-2πi (fy*shift_h + fx*shift_w))
    phase = torch.exp(-2j * torch.pi * (fy * shift_h + fx * shift_w))
    spec = torch.fft.fft2(img.to(torch.float64))
    out = torch.real(torch.fft.ifft2(spec * phase)).to(dtype)
    return out


def bilinear_shift_2d(img: torch.Tensor, shift_h: float, shift_w: float) -> torch.Tensor:
    """Bilinear shift: out[i,j] = img[i - shift_h, j - shift_w]; out-of-bound = 0.

    img: (..., H, W). Vectorized; only relies on integer-index gather (so it
    runs correctly on cpu / cuda without grid_sample's normalized-coord traps).
    """
    H, W = img.shape[-2:]
    device = img.device
    dtype = img.dtype

    i_src = torch.arange(H, device=device, dtype=dtype).view(H, 1) - shift_h
    j_src = torch.arange(W, device=device, dtype=dtype).view(1, W) - shift_w

    i0 = torch.floor(i_src).long(); i1 = i0 + 1
    j0 = torch.floor(j_src).long(); j1 = j0 + 1
    wi = (i_src - i0.to(dtype))
    wj = (j_src - j0.to(dtype))

    valid_i0 = (i0 >= 0) & (i0 < H)
    valid_i1 = (i1 >= 0) & (i1 < H)
    valid_j0 = (j0 >= 0) & (j0 < W)
    valid_j1 = (j1 >= 0) & (j1 < W)

    i0c = i0.clamp(0, H - 1).expand(H, W)
    i1c = i1.clamp(0, H - 1).expand(H, W)
    j0c = j0.clamp(0, W - 1).expand(H, W)
    j1c = j1.clamp(0, W - 1).expand(H, W)

    v00 = img[..., i0c, j0c]
    v01 = img[..., i0c, j1c]
    v10 = img[..., i1c, j0c]
    v11 = img[..., i1c, j1c]

    m00 = (valid_i0 & valid_j0).expand(H, W).to(dtype)
    m01 = (valid_i0 & valid_j1).expand(H, W).to(dtype)
    m10 = (valid_i1 & valid_j0).expand(H, W).to(dtype)
    m11 = (valid_i1 & valid_j1).expand(H, W).to(dtype)

    w00 = (1 - wi) * (1 - wj)
    w01 = (1 - wi) * wj
    w10 = wi * (1 - wj)
    w11 = wi * wj

    return (v00 * m00 * w00 + v01 * m01 * w01
            + v10 * m10 * w10 + v11 * m11 * w11)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--psf_dir', default='PSF/PSF_zoom2_39dz1_N13')
    p.add_argument('--Nnum', type=int, default=13)
    p.add_argument('--affine_params', default='psf_params_M1.pt',
                   help='File with psf_fit affine prediction (used for centroid_offset_affine)')
    p.add_argument('--out', default='PSF_centered.pt')
    p.add_argument('--shift_method', choices=('bilinear', 'fft'), default='fft',
                   help='Sub-pixel shift method for de-centering. fft is exact '
                        'for band-limited signals; bilinear smears slightly.')
    args = p.parse_args()

    psfs = load_psfs(args.psf_dir, args.Nnum).float()  # (U, Z, ph, pw) on cpu
    U, Z, ph, pw = psfs.shape
    print(f'[make_centered_psf] PSF: U={U} Z={Z} ph={ph} pw={pw}')

    # Two centers (see header). Bilinear de-centering targets the discrete
    # one; the centroid_offset table is relative to the continuous one.
    kc_h_disc = (ph - 1) // 2
    kc_w_disc = (pw - 1) // 2
    kc_h_cont = (ph - 1) / 2.0
    kc_w_cont = (pw - 1) / 2.0
    print(f'[make_centered_psf] discrete kc = ({kc_h_disc}, {kc_w_disc}) '
          f'(de-centering target); continuous kc = ({kc_h_cont}, {kc_w_cont}) '
          f'(offset reference)')

    # 1. Per-(u, z) measured centroid (absolute index, H-axis & W-axis).
    centroid_abs = torch.zeros(U, Z, 2)  # (cy, cx) = (H-axis, W-axis)
    for u in range(U):
        for z in range(Z):
            cy, cx = compute_centroid_abs(psfs[u, z])
            centroid_abs[u, z, 0] = cy
            centroid_abs[u, z, 1] = cx

    centroid_offset_measured = centroid_abs - torch.tensor(
        [kc_h_cont, kc_w_cont], dtype=torch.float32
    )

    # 2. Bilinear-shift each PSF so its centroid lands at the discrete kc.
    #    shift = discrete_kc - centroid (so out_center = in_centroid - shift = kc).
    psf_centered = torch.zeros_like(psfs)
    energy_raw = torch.zeros(U, Z)
    energy_centered = torch.zeros(U, Z)
    centroid_centered_abs = torch.zeros(U, Z, 2)
    for u in range(U):
        for z in range(Z):
            cy, cx = centroid_abs[u, z, 0].item(), centroid_abs[u, z, 1].item()
            shift_h = kc_h_disc - cy
            shift_w = kc_w_disc - cx
            if args.shift_method == 'fft':
                psf_centered[u, z] = fft_shift_2d(psfs[u, z], shift_h, shift_w)
            else:
                psf_centered[u, z] = bilinear_shift_2d(psfs[u, z], shift_h, shift_w)
            energy_raw[u, z] = psfs[u, z].sum()
            energy_centered[u, z] = psf_centered[u, z].sum()
            cy_after, cx_after = compute_centroid_abs(psf_centered[u, z])
            centroid_centered_abs[u, z, 0] = cy_after
            centroid_centered_abs[u, z, 1] = cx_after

    # 3. Energy + residual centroid diagnostics.
    eps = 1e-12
    energy_err = (energy_centered - energy_raw).abs() / (energy_raw.abs() + eps)
    centroid_err = (centroid_centered_abs - torch.tensor(
        [kc_h_disc, kc_w_disc], dtype=torch.float32
    ))
    centroid_err_norm = centroid_err.norm(dim=-1)  # (U, Z) per-(u,z) px error

    print(f'[make_centered_psf] energy err: mean={energy_err.mean().item():.3e} '
          f'max={energy_err.max().item():.3e}')
    print(f'[make_centered_psf] residual centroid err (px after centering): '
          f'mean={centroid_err_norm.mean().item():.3e} '
          f'max={centroid_err_norm.max().item():.3e}')

    # Top-5 worst (u, z).
    flat = energy_err.flatten()
    topk = torch.topk(flat, k=min(5, flat.numel()))
    print('[make_centered_psf] top-5 (u, z) by energy err:')
    for v, idx in zip(topk.values, topk.indices):
        u_ = (idx // Z).item(); z_ = (idx % Z).item()
        print(f'   u={u_} z={z_}: energy_err={v.item():.3e} '
              f'shift=({(kc_h_disc - centroid_abs[u_, z_, 0]).item():+.3f},'
              f'{(kc_w_disc - centroid_abs[u_, z_, 1]).item():+.3f}) px')

    # 4. Affine centroid prediction (from psf_params_M1.pt).
    centroid_offset_affine = None
    affine_path = args.affine_params
    if not os.path.isabs(affine_path):
        affine_path = os.path.abspath(affine_path)
    if os.path.exists(affine_path):
        ap = torch.load(affine_path, weights_only=False)
        # psf_fit: x=W, y=H. A[u, 0, 0, 2] = d_x = W shift/z, A[u, 0, 1, 2] = d_y = H shift/z.
        # b[u, 0] = (b_x, b_y) = (W bias, H bias).
        # image_center = (cx0, cy0) = (W center, H center) = (63.5, 63.5).
        # psf_fit z_grid = arange(Z) - (Z-1)/2.
        # We want centroid_offset_affine[u, z] = (h_offset, w_offset) where h_offset is H axis.
        d_x = ap['A'][:, 0, 0, 2]   # (U,)
        d_y = ap['A'][:, 0, 1, 2]   # (U,)
        b_x = ap['b'][:, 0, 0]       # (U,)
        b_y = ap['b'][:, 0, 1]       # (U,)
        image_center = ap['image_center']  # (cx0, cy0) = (W, H) center; psf_fit's continuous center
        z_grid = ap['z_grid']                # (Z,) centered around 0
        # Predicted centroid (absolute PSF idx):
        #   centroid_w_abs = image_center[0] + z_grid * d_x + b_x
        #   centroid_h_abs = image_center[1] + z_grid * d_y + b_y
        # Offset from continuous kc = image_center (psf_fit's reference):
        h_offset = (image_center[1] + z_grid.unsqueeze(0) * d_y.unsqueeze(1)
                    + b_y.unsqueeze(1) - kc_h_cont)
        w_offset = (image_center[0] + z_grid.unsqueeze(0) * d_x.unsqueeze(1)
                    + b_x.unsqueeze(1) - kc_w_cont)
        centroid_offset_affine = torch.stack([h_offset, w_offset], dim=-1)  # (U, Z, 2)

        # Diagnostic: how much does affine prediction differ from measurement?
        diff = (centroid_offset_affine - centroid_offset_measured).norm(dim=-1)
        print(f'[make_centered_psf] affine vs measured centroid diff (px): '
              f'mean={diff.mean().item():.3f} max={diff.max().item():.3f}')
    else:
        print(f'[make_centered_psf] (no affine params at {affine_path}; skipping affine table)')

    # 5. Save.
    out = {
        'psf_centered': psf_centered,
        'centroid_raw_abs': centroid_abs,
        'centroid_centered_abs': centroid_centered_abs,
        'centroid_offset_measured': centroid_offset_measured,
        'energy_raw': energy_raw,
        'energy_centered': energy_centered,
        'energy_err': energy_err,
        'centroid_err_after': centroid_err_norm,
        'discrete_kernel_center': torch.tensor([kc_h_disc, kc_w_disc],
                                               dtype=torch.float32),
        'continuous_kernel_center': torch.tensor([kc_h_cont, kc_w_cont],
                                                 dtype=torch.float32),
        'psf_shape': torch.tensor([U, Z, ph, pw]),
    }
    if centroid_offset_affine is not None:
        out['centroid_offset_affine'] = centroid_offset_affine
    torch.save(out, args.out)
    print(f'[make_centered_psf] saved -> {args.out}')


if __name__ == '__main__':
    main()
