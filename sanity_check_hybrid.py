"""Sanity checks for HybridRenderer.

Phase 1 (mode='raw_exact'):
    Test 1 — Single-Gaussian forward equivalence vs full-volume forward.
    Test 2 — Multi-Gaussian additivity (renderer is linear).
    Test 3 — Multi-Gaussian vs full-volume forward.
        Pass: rel L2 < 2%, view energy err < 3%, center err < 0.5 px.

Phase 2 (mode='centered_affine', splat_mode='round'):
    Test 4 — Phase 2 with measured centroid + round splat vs Phase 1.
        Round splat quantizes the per-Gaussian image position to an integer
        pixel; for σ ~1.5 px this gives rel_l2 ≈ 0.4/(σ√2) ≈ 19% — confirmed
        empirically. Phase 2's geometry math is correct; this test only
        verifies energy preservation and centroid alignment within 1 px.
        Pass: max center err < 1.0 px, energy max err < 1%.
    Test 5 — Phase 2 with affine centroid + round splat vs Phase 1.
        Adds the affine-vs-measured residual (~0.33 px max) on top of
        round-splat quantization. Same kind of looseness as Test 4.
        Pass: max center err < 1.5 px, energy max err < 1%.
    Test 6 — Affine-prediction vs measured centroid (no rendering).
        Pass: max < 0.5 px (already verified in make_centered_psf).

Phase 3 preview (mode='centered_affine', splat_mode='bilinear'):
    Test 7 — Phase 2 + bilinear splat vs Phase 1. Sub-pixel placement is
        recovered via 4-corner weighted splat → centroid error drops to
        ~0.02 px (machine precision) confirming geometry is correct.
        Residual rel L2 ~5% is bilinear sub-pixel-resampling shape
        distortion (intrinsic to any pixel-grid bilinear renderer; LFM
        PSFs have high-frequency features sensitive to 2x2 box smoothing).
        Pass: rel L2 < 8%, max center err < 0.05 px, energy err < 1e-5.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch

from utils import load_psfs, generate_fps
from hybrid_renderer import (
    GaussianBatch,
    HybridRenderer,
    gaussians_to_full_volume,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def relative_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = (a - b).reshape(-1)
    denom = b.reshape(-1).norm() + 1e-12
    return (diff.norm() / denom).item()


def per_view_energy_err(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a, b: (V, H, W). Returns (V,) relative |Σa - Σb| / |Σb|."""
    ea = a.sum(dim=(-1, -2))
    eb = b.sum(dim=(-1, -2))
    return ((ea - eb).abs() / (eb.abs() + 1e-12))


def per_view_centroid(img: torch.Tensor) -> torch.Tensor:
    """img: (V, H, W) → (V, 2) centroid in (h, w) pixels."""
    V, H, W = img.shape
    img_pos = img.clamp(min=0)  # centroid only meaningful for non-negative mass
    yy = torch.arange(H, device=img.device, dtype=img.dtype).view(1, H, 1)
    xx = torch.arange(W, device=img.device, dtype=img.dtype).view(1, 1, W)
    mass = img_pos.sum(dim=(-1, -2)).clamp(min=1e-12)
    cy = (img_pos * yy).sum(dim=(-1, -2)) / mass
    cx = (img_pos * xx).sum(dim=(-1, -2)) / mass
    return torch.stack([cy, cx], dim=-1)


def reference_forward(
    gaussians: GaussianBatch,
    psf: torch.Tensor,           # (U, Z, ph, pw)
    image_shape: tuple[int, int],
    target_views: list[int],
    z_norm: str = "mean",
) -> torch.Tensor:
    """Build full volume → generate_fps → mean(dim=1)."""
    H, W = image_shape
    U, Z, ph, pw = psf.shape
    vol = gaussians_to_full_volume(gaussians, (Z, H, W),
                                   device=psf.device, dtype=psf.dtype)
    psf_subset = psf[target_views]  # (Vt, Z, ph, pw)
    fp = generate_fps(psf_subset, vol)  # (Vt, Z, H, W)
    if z_norm == "mean":
        return fp.mean(dim=1)
    elif z_norm == "sum":
        return fp.sum(dim=1)
    else:
        raise ValueError(z_norm)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def make_single_gaussian(H, W, Z, device, *, mu=None, sigma=(1.5, 1.5, 1.5),
                         rho=1.0, dtype=torch.float32):
    if mu is None:
        # Place near the center but with a non-integer offset to exercise round-anchor.
        mu = (H / 2.0 + 0.3, W / 2.0 - 0.4, Z / 2.0 + 0.2)
    pos = torch.tensor([mu], device=device, dtype=dtype)         # (1, 3)
    sig = torch.tensor([sigma], device=device, dtype=dtype)      # (1, 3)
    rho_t = torch.tensor([rho], device=device, dtype=dtype)      # (1,)
    return GaussianBatch(positions=pos, sigmas=sig, rhos=rho_t)


def make_random_gaussians(N, H, W, Z, device, *, dtype=torch.float32, seed=0):
    g = torch.Generator(device='cpu').manual_seed(seed)
    margin_xy = 30   # keep away from boundary so 4σ box stays inside
    margin_z = 5
    xs = (torch.rand(N, generator=g) * (H - 2 * margin_xy) + margin_xy)
    ys = (torch.rand(N, generator=g) * (W - 2 * margin_xy) + margin_xy)
    zs = (torch.rand(N, generator=g) * (Z - 2 * margin_z) + margin_z)
    sx = torch.rand(N, generator=g) * 1.0 + 1.0    # 1.0–2.0
    sy = torch.rand(N, generator=g) * 1.0 + 1.0
    sz = torch.rand(N, generator=g) * 0.8 + 0.6    # 0.6–1.4
    rho = torch.rand(N, generator=g) * 0.8 + 0.2
    pos = torch.stack([xs, ys, zs], dim=-1).to(device).to(dtype)
    sig = torch.stack([sx, sy, sz], dim=-1).to(device).to(dtype)
    rho = rho.to(device).to(dtype)
    return GaussianBatch(positions=pos, sigmas=sig, rhos=rho)


def test_single_gaussian(psf, H, W, device, sigma_trunc=4.0):
    print("\n=== Test 1 — Single-Gaussian forward equivalence ===")
    U, Z = psf.shape[0], psf.shape[1]
    gaussians = make_single_gaussian(H, W, Z, device)
    target_views = list(range(U))

    ref = reference_forward(gaussians, psf, (H, W), target_views, z_norm="mean")
    renderer = HybridRenderer(psf, (H, W), sigma_truncation=sigma_trunc,
                              z_norm="mean", mode="raw_exact").to(device)
    out = renderer(gaussians, target_views=target_views)

    rel_l2 = relative_l2(out, ref)
    energy_err = per_view_energy_err(out, ref)
    cent_ref = per_view_centroid(ref)
    cent_out = per_view_centroid(out)
    center_err = (cent_out - cent_ref).norm(dim=-1)  # (V,)

    print(f"  relative L2 (vs full-volume forward)  = {rel_l2:.4e}")
    print(f"  per-view energy err: mean={energy_err.mean().item():.4e} "
          f"max={energy_err.max().item():.4e}")
    print(f"  per-view centroid err (px): mean={center_err.mean().item():.4e} "
          f"max={center_err.max().item():.4e}")

    pass_l2 = rel_l2 < 0.02
    pass_energy = energy_err.max().item() < 0.03
    pass_center = center_err.max().item() < 0.5
    ok = pass_l2 and pass_energy and pass_center
    print(f"  PASS={ok}  (l2<2%: {pass_l2}, max energy<3%: {pass_energy}, "
          f"max center<0.5px: {pass_center})")
    return ok, dict(rel_l2=rel_l2,
                    energy_err_mean=energy_err.mean().item(),
                    energy_err_max=energy_err.max().item(),
                    center_err_mean=center_err.mean().item(),
                    center_err_max=center_err.max().item())


def test_additivity(psf, H, W, device, sigma_trunc=4.0, N=5):
    print("\n=== Test 2 — Additivity (Hybrid is linear) ===")
    U, Z = psf.shape[0], psf.shape[1]
    gaussians = make_random_gaussians(N, H, W, Z, device)
    target_views = list(range(U))

    renderer = HybridRenderer(psf, (H, W), sigma_truncation=sigma_trunc,
                              z_norm="mean", mode="raw_exact").to(device)
    out_joint = renderer(gaussians, target_views=target_views)

    out_sum = torch.zeros_like(out_joint)
    for i in range(N):
        gi = GaussianBatch(
            positions=gaussians.positions[i:i+1],
            sigmas=gaussians.sigmas[i:i+1],
            rhos=gaussians.rhos[i:i+1],
        )
        out_sum = out_sum + renderer(gi, target_views=target_views)

    rel_l2 = relative_l2(out_joint, out_sum)
    print(f"  relative L2 (joint vs Σ single) = {rel_l2:.4e}")
    ok = rel_l2 < 1e-5
    print(f"  PASS={ok}  (threshold 1e-5)")
    return ok, dict(rel_l2=rel_l2)


# ---------------------------------------------------------------------------
# Phase 2 tests
# ---------------------------------------------------------------------------

def test_phase2_measured(psf_raw, psf_centered, centroid_offset_measured,
                         H, W, device, sigma_trunc=4.0, N=10):
    """Test 4 — Phase 2 (measured centroid) vs Phase 1.

    Both render the same Gaussians; Phase 2 should produce visually-equivalent
    LFIs, with discrepancy bounded by bilinear de-centering interp + round() splat.
    """
    print("\n=== Test 4 — Phase 2 (measured centroid) vs Phase 1 ===")
    U, Z = psf_raw.shape[0], psf_raw.shape[1]
    gaussians = make_random_gaussians(N, H, W, Z, device, seed=1)
    target_views = list(range(U))

    rend1 = HybridRenderer(psf_raw, (H, W), sigma_truncation=sigma_trunc,
                           z_norm="mean", mode="raw_exact").to(device)
    rend2 = HybridRenderer(psf_centered, (H, W), sigma_truncation=sigma_trunc,
                           z_norm="mean", mode="centered_affine",
                           centroid_offset=centroid_offset_measured.to(device)).to(device)

    out1 = rend1(gaussians, target_views=target_views)
    out2 = rend2(gaussians, target_views=target_views)

    rel_l2 = relative_l2(out2, out1)
    energy_err = per_view_energy_err(out2, out1)
    cent_1 = per_view_centroid(out1)
    cent_2 = per_view_centroid(out2)
    center_err = (cent_2 - cent_1).norm(dim=-1)

    print(f"  N={N}  relative L2 = {rel_l2:.4e}  "
          f"(round-splat sub-pixel cost ~ Δ/(σ√2); not a bug)")
    print(f"  per-view energy err: mean={energy_err.mean().item():.4e} "
          f"max={energy_err.max().item():.4e}")
    print(f"  per-view centroid err (px): mean={center_err.mean().item():.4e} "
          f"max={center_err.max().item():.4e}")
    pass_center = center_err.max().item() < 1.0
    pass_energy = energy_err.max().item() < 0.01
    ok = pass_center and pass_energy
    print(f"  PASS={ok}  (max center<1px: {pass_center}, "
          f"max energy<1%: {pass_energy})")
    return ok, dict(rel_l2=rel_l2,
                    energy_err_mean=energy_err.mean().item(),
                    energy_err_max=energy_err.max().item(),
                    center_err_mean=center_err.mean().item(),
                    center_err_max=center_err.max().item())


def test_phase2_affine(psf_raw, psf_centered, centroid_offset_affine,
                       H, W, device, sigma_trunc=4.0, N=10):
    """Test 5 — Phase 2 with affine centroid (psf_params_M1.pt) vs Phase 1.

    Adds the affine-vs-measured residual (max ~0.33 px in our PSF) on top
    of the discretization error from Test 4.
    """
    print("\n=== Test 5 — Phase 2 (affine centroid) vs Phase 1 ===")
    U, Z = psf_raw.shape[0], psf_raw.shape[1]
    gaussians = make_random_gaussians(N, H, W, Z, device, seed=2)
    target_views = list(range(U))

    rend1 = HybridRenderer(psf_raw, (H, W), sigma_truncation=sigma_trunc,
                           z_norm="mean", mode="raw_exact").to(device)
    rend2 = HybridRenderer(psf_centered, (H, W), sigma_truncation=sigma_trunc,
                           z_norm="mean", mode="centered_affine",
                           centroid_offset=centroid_offset_affine.to(device)).to(device)

    out1 = rend1(gaussians, target_views=target_views)
    out2 = rend2(gaussians, target_views=target_views)

    rel_l2 = relative_l2(out2, out1)
    energy_err = per_view_energy_err(out2, out1)
    cent_1 = per_view_centroid(out1)
    cent_2 = per_view_centroid(out2)
    center_err = (cent_2 - cent_1).norm(dim=-1)

    print(f"  N={N}  relative L2 = {rel_l2:.4e}  "
          f"(round + affine residual; not a bug)")
    print(f"  per-view energy err: mean={energy_err.mean().item():.4e} "
          f"max={energy_err.max().item():.4e}")
    print(f"  per-view centroid err (px): mean={center_err.mean().item():.4e} "
          f"max={center_err.max().item():.4e}")
    pass_center = center_err.max().item() < 1.5
    pass_energy = energy_err.max().item() < 0.01
    ok = pass_center and pass_energy
    print(f"  PASS={ok}  (max center<1.5px: {pass_center}, "
          f"max energy<1%: {pass_energy})")
    return ok, dict(rel_l2=rel_l2,
                    energy_err_mean=energy_err.mean().item(),
                    energy_err_max=energy_err.max().item(),
                    center_err_mean=center_err.mean().item(),
                    center_err_max=center_err.max().item())


def test_phase3_bilinear(psf_raw, psf_centered, centroid_offset_measured,
                         H, W, device, sigma_trunc=4.0, N=10):
    """Test 7 — Phase 2 + bilinear splat vs Phase 1.

    Sub-pixel placement is recovered via 4-corner weighted splat. The
    bilinear sub-pixel resampling is equivalent to convolving the patch
    with a 2x2 box kernel (variance 1/12 per axis), which smooths LFM
    PSF features by ~5% rel L2 against the FFT-exact Phase 1 reference.
    This is the inherent cost of a bilinear differentiable renderer;
    any pixel-grid renderer using bilinear sub-pixel sampling will see
    this. Centroid and energy match should be tight (< 0.05 px, < 1e-5).
    """
    print("\n=== Test 7 — Phase 2 + bilinear splat vs Phase 1 ===")
    U, Z = psf_raw.shape[0], psf_raw.shape[1]
    gaussians = make_random_gaussians(N, H, W, Z, device, seed=3)
    target_views = list(range(U))

    rend1 = HybridRenderer(psf_raw, (H, W), sigma_truncation=sigma_trunc,
                           z_norm="mean", mode="raw_exact").to(device)
    rend2 = HybridRenderer(
        psf_centered, (H, W), sigma_truncation=sigma_trunc, z_norm="mean",
        mode="centered_affine", splat_mode="bilinear",
        centroid_offset=centroid_offset_measured.to(device),
    ).to(device)

    out1 = rend1(gaussians, target_views=target_views)
    out2 = rend2(gaussians, target_views=target_views)

    rel_l2 = relative_l2(out2, out1)
    energy_err = per_view_energy_err(out2, out1)
    cent_1 = per_view_centroid(out1)
    cent_2 = per_view_centroid(out2)
    center_err = (cent_2 - cent_1).norm(dim=-1)

    print(f"  N={N}  relative L2 = {rel_l2:.4e}")
    print(f"  per-view energy err: mean={energy_err.mean().item():.4e} "
          f"max={energy_err.max().item():.4e}")
    print(f"  per-view centroid err (px): mean={center_err.mean().item():.4e} "
          f"max={center_err.max().item():.4e}")
    pass_l2 = rel_l2 < 0.08
    pass_center = center_err.max().item() < 0.05
    pass_energy = energy_err.max().item() < 1e-5
    ok = pass_l2 and pass_center and pass_energy
    print(f"  PASS={ok}  (l2<8%: {pass_l2}, max center<0.05px: {pass_center}, "
          f"max energy<1e-5: {pass_energy})")
    return ok, dict(rel_l2=rel_l2,
                    energy_err_mean=energy_err.mean().item(),
                    energy_err_max=energy_err.max().item(),
                    center_err_mean=center_err.mean().item(),
                    center_err_max=center_err.max().item())


def test_grad_nonzero(psf_centered, centroid_offset_measured,
                      H, W, device, sigma_trunc=4.0):
    """Test 9 — Gradient sanity check (per guide §6.5).

    Verify that mu, sigma, rho all receive non-zero gradients in
    centered_affine + bilinear mode. We use a position-discriminating
    loss (target image MSE) instead of out.sum() — the latter is
    translation-invariant for a non-truncated Gaussian, so mu_xy.grad
    would be ~0 by physics, masking real wiring bugs.
    """
    print("\n=== Test 9 — Gradient sanity (Phase 3 differentiability) ===")
    U, Z = psf_centered.shape[0], psf_centered.shape[1]

    mu = torch.tensor([[H/2 + 0.3, W/2 - 0.4, Z/2 + 0.2]],
                      device=device, requires_grad=True)
    sigma = torch.tensor([[1.5, 1.5, 1.0]], device=device, requires_grad=True)
    # Use large rho so output is observable in float32 for FD.
    rho = torch.tensor([1.0e4], device=device, requires_grad=True)

    g = GaussianBatch(positions=mu, sigmas=sigma, rhos=rho)
    renderer = HybridRenderer(
        psf_centered, (H, W), sigma_truncation=sigma_trunc, z_norm="mean",
        mode="centered_affine", splat_mode="bilinear",
        centroid_offset=centroid_offset_measured.to(device),
    ).to(device)

    out = renderer(g)
    # MSE against a random target on the same scale → discriminating loss.
    target = torch.randn_like(out) * out.detach().abs().mean()
    loss = ((out - target) ** 2).mean()
    loss.backward()

    assert mu.grad is not None, "mu has no grad — broken graph"
    assert sigma.grad is not None, "sigma has no grad — broken graph (likely .item() leak)"
    assert rho.grad is not None, "rho has no grad"

    # gaussian_utils convention: positions = (x=H, y=W, z=D)
    grad_x = mu.grad[:, 0].abs().mean().item()
    grad_y = mu.grad[:, 1].abs().mean().item()
    grad_z = mu.grad[:, 2].abs().mean().item()
    grad_sigma = sigma.grad.abs().mean().item()
    grad_rho = rho.grad.abs().mean().item()

    print(f"  mu.grad   x(H)={grad_x:.4e}  y(W)={grad_y:.4e}  z(D)={grad_z:.4e}")
    print(f"  sigma.grad mean abs = {grad_sigma:.4e}")
    print(f"  rho.grad   mean abs = {grad_rho:.4e}")

    # All five must be > 0 for training to work.
    eps = 1e-15
    pass_mu_x = grad_x > eps
    pass_mu_y = grad_y > eps
    pass_mu_z = grad_z > eps
    pass_sigma = grad_sigma > eps
    pass_rho = grad_rho > eps
    ok = all([pass_mu_x, pass_mu_y, pass_mu_z, pass_sigma, pass_rho])
    print(f"  PASS={ok}  (mu_x={pass_mu_x}, mu_y={pass_mu_y}, mu_z={pass_mu_z}, "
          f"sigma={pass_sigma}, rho={pass_rho})")

    # Finite-difference grad check on rho (cleanest, no integer index issues).
    # rho enters linearly, so any nonlinearity in FD is purely numerical.
    eps_fd = 1.0  # rho ~ 1e4 → 1e-4 relative perturbation
    def render(rho_val):
        m2 = mu.detach().clone()
        s2 = sigma.detach().clone()
        r2 = torch.tensor([rho_val], device=device)
        g2 = GaussianBatch(positions=m2, sigmas=s2, rhos=r2)
        return renderer(g2)

    o_plus = render(rho.item() + eps_fd)
    o_minus = render(rho.item() - eps_fd)
    target_det = target.detach()
    L_plus = ((o_plus - target_det) ** 2).mean().item()
    L_minus = ((o_minus - target_det) ** 2).mean().item()
    fd_rho = (L_plus - L_minus) / (2 * eps_fd)
    rel = abs(fd_rho - rho.grad.item()) / (abs(fd_rho) + 1e-15)
    print(f"  finite-diff vs autograd on rho: fd={fd_rho:.4e} ag={rho.grad.item():.4e} "
          f"rel={rel:.2%}")
    pass_fd = rel < 0.05
    ok = ok and pass_fd
    print(f"  PASS={ok}  (FD/autograd agree <5%: {pass_fd})")
    return ok, dict(grad_mu_x=grad_x, grad_mu_y=grad_y, grad_mu_z=grad_z,
                    grad_sigma=grad_sigma, grad_rho=grad_rho,
                    fd_rho=fd_rho, fd_rel=rel)


def test_centroid_prediction(centroid_offset_measured, centroid_offset_affine):
    """Test 6 — Affine prediction vs measured centroid (numerical only)."""
    print("\n=== Test 6 — Affine vs measured centroid (px) ===")
    diff = (centroid_offset_affine - centroid_offset_measured).norm(dim=-1)
    print(f"  centroid pred diff: mean={diff.mean().item():.3f} "
          f"max={diff.max().item():.3f}")
    ok = diff.max().item() < 0.5
    print(f"  PASS={ok}  (max<0.5 px)")
    return ok, dict(diff_mean=diff.mean().item(), diff_max=diff.max().item())


# ---------------------------------------------------------------------------
# Phase 1 helper (back-compat name)
# ---------------------------------------------------------------------------

def test_multi_vs_full_volume(psf, H, W, device, sigma_trunc=4.0, N=10):
    print("\n=== Test 3 — Multi-Gaussian vs full-volume forward ===")
    U, Z = psf.shape[0], psf.shape[1]
    gaussians = make_random_gaussians(N, H, W, Z, device)
    target_views = list(range(U))

    ref = reference_forward(gaussians, psf, (H, W), target_views, z_norm="mean")
    renderer = HybridRenderer(psf, (H, W), sigma_truncation=sigma_trunc,
                              z_norm="mean", mode="raw_exact").to(device)
    out = renderer(gaussians, target_views=target_views)

    rel_l2 = relative_l2(out, ref)
    energy_err = per_view_energy_err(out, ref)
    print(f"  N={N}  relative L2 = {rel_l2:.4e}")
    print(f"  per-view energy err: mean={energy_err.mean().item():.4e} "
          f"max={energy_err.max().item():.4e}")
    pass_l2 = rel_l2 < 0.02
    pass_energy = energy_err.max().item() < 0.03
    ok = pass_l2 and pass_energy
    print(f"  PASS={ok}")
    return ok, dict(rel_l2=rel_l2,
                    energy_err_mean=energy_err.mean().item(),
                    energy_err_max=energy_err.max().item())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--psf_dir', default='PSF/PSF_zoom2_39dz1_N13')
    p.add_argument('--Nnum', type=int, default=13)
    p.add_argument('--H', type=int, default=256)
    p.add_argument('--W', type=int, default=256)
    p.add_argument('--sigma_trunc', type=float, default=4.0)
    p.add_argument('--gpu_id', type=int, default=0)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--centered_psf', default='PSF_centered.pt',
                   help='File produced by tools/make_centered_psf.py (Phase 2 tests)')
    p.add_argument('--phase', choices=('1', '2', 'all'), default='all')
    args = p.parse_args()

    device = torch.device('cpu') if args.cpu else torch.device('cuda', args.gpu_id)
    psf = load_psfs(args.psf_dir, args.Nnum).to(device).float()  # (U, Z, ph, pw)
    print(f"Loaded PSF: {tuple(psf.shape)} on {device}")
    print(f"Image shape: ({args.H}, {args.W}), sigma_trunc={args.sigma_trunc}")

    results = {}
    all_ok = True

    if args.phase in ('1', 'all'):
        ok1, results['test1'] = test_single_gaussian(psf, args.H, args.W, device,
                                                     sigma_trunc=args.sigma_trunc)
        ok2, results['test2'] = test_additivity(psf, args.H, args.W, device,
                                                sigma_trunc=args.sigma_trunc)
        ok3, results['test3'] = test_multi_vs_full_volume(psf, args.H, args.W, device,
                                                          sigma_trunc=args.sigma_trunc)
        all_ok = all_ok and ok1 and ok2 and ok3

    if args.phase in ('2', 'all'):
        if not os.path.exists(args.centered_psf):
            print(f"\n[Phase 2] {args.centered_psf} not found — "
                  f"run tools/make_centered_psf.py first.")
            sys.exit(2)
        cp = torch.load(args.centered_psf, weights_only=False)
        psf_centered = cp['psf_centered'].to(device).float()
        cent_meas = cp['centroid_offset_measured'].float()
        cent_aff = cp.get('centroid_offset_affine', None)
        print(f"\nLoaded centered PSF: {tuple(psf_centered.shape)} from {args.centered_psf}")

        ok4, results['test4'] = test_phase2_measured(
            psf, psf_centered, cent_meas, args.H, args.W, device,
            sigma_trunc=args.sigma_trunc)
        all_ok = all_ok and ok4
        if cent_aff is not None:
            cent_aff = cent_aff.float()
            ok5, results['test5'] = test_phase2_affine(
                psf, psf_centered, cent_aff, args.H, args.W, device,
                sigma_trunc=args.sigma_trunc)
            ok6, results['test6'] = test_centroid_prediction(cent_meas, cent_aff)
            all_ok = all_ok and ok5 and ok6
        else:
            print("\n[Phase 2] no affine centroid table — skipping Test 5/6")

        ok7, results['test7'] = test_phase3_bilinear(
            psf, psf_centered, cent_meas, args.H, args.W, device,
            sigma_trunc=args.sigma_trunc)
        all_ok = all_ok and ok7

        ok9, results['test9'] = test_grad_nonzero(
            psf_centered, cent_meas, args.H, args.W, device,
            sigma_trunc=args.sigma_trunc)
        all_ok = all_ok and ok9

    print("\n========== SUMMARY ==========")
    for k, v in results.items():
        print(f"  {k}: {v}")
    print(f"\n  ALL TESTS {'PASSED' if all_ok else 'FAILED'}")
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
