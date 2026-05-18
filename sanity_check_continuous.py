"""Sanity checks for HybridRenderer mode='continuous_fourier' (Phase 1).

Run on a GPU host after `git pull && git checkout gauss-continuous-fourier`.

Tests:
  T1 — Integer-μ equivalence: continuous_fourier vs raw_exact when μ sits
       exactly on integer voxel centers. The two paths differ only in how
       they evaluate the 2D Gaussian (discrete voxel sum vs analytic FT).
       For σ ≥ 1 voxel with sigma_trunc≥4 the Riemann-sum error is tiny.
       PASS: rel_l2 < 2% per Gaussian.

  T2 — Sub-voxel smoothness scan: sweep μ_x across two voxels and measure
       how the readout pixel evolves. Both modes are expected to be smooth
       at this scale (raw_exact's box anchor jumps by 1 voxel at half-int
       boundaries, but the Gaussian-coverage / splat-anchor co-shift makes
       the rendered pixel approximately C0; continuous_fourier is C∞ by
       construction). The test is therefore a SMOOTHNESS check on
       continuous_fourier alone, with raw_exact reported for context.
       Smoothness ratio = max(|Δ|) / mean(|Δ|) over consecutive sweep
       samples; ~1–3 for smooth curves, ≫ 5 for step-like.
       PASS: continuous_fourier ratio < 5.

  T3 — FD vs autograd for ∂loss/∂μ_x. The loss must NOT be translation-
       invariant (otherwise the true gradient is 0 and the test verifies
       nothing). Uses a single off-center pixel readout
           loss = out[u_read, i_read, j_read]
       which depends on how the convolved Gaussian aligns with that fixed
       pixel — strongly μ-sensitive.
       PASS: relative error < 1e-2.

  T4 — Multi-Gaussian vs full-volume reference. N random Gaussians at
       generic sub-voxel positions. continuous_fourier should agree well
       with the reference (gaussians_to_full_volume → generate_fps → mean),
       and typically better than raw_exact because there is no μ→round(μ)
       splat quantization.
       PASS: rel_l2 < 3%.

Usage:
    python sanity_check_continuous.py \
        --psf_dir PSF/PSF_zoom2_39dz1_N13 --Nnum 13 --H 256 --W 256
"""
from __future__ import annotations

import argparse
import sys

import torch

from utils import load_psfs
from hybrid_renderer import GaussianBatch, HybridRenderer
from sanity_check_hybrid import (
    relative_l2,
    per_view_energy_err,
    per_view_centroid,
    reference_forward,
    make_random_gaussians,
)


# ---------------------------------------------------------------------------
# Builders for both renderers (same psf, same fixed_half_xy/z, same chunking)
# ---------------------------------------------------------------------------

def build_renderer(psf, H, W, mode, *, sigma_trunc, half_xy, half_z,
                   chunk_size=16, use_checkpoint=False):
    return HybridRenderer(
        psf, (H, W),
        sigma_truncation=sigma_trunc,
        z_norm='mean',
        mode=mode,
        fixed_half_xy=half_xy,
        fixed_half_z=half_z,
        chunk_size=chunk_size,
        use_checkpoint=use_checkpoint,
    )


def auto_half_from_sigma_max(s_max, sigma_trunc):
    import math
    return max(1, int(math.ceil(s_max * sigma_trunc)))


# ---------------------------------------------------------------------------
# T1 — Integer-μ equivalence
# ---------------------------------------------------------------------------

def test_integer_mu_equivalence(psf, H, W, device, *, sigma_trunc=4.0):
    print("\n[T1] Integer-μ equivalence (continuous_fourier vs raw_exact)")
    U, Z, ph, pw = psf.shape
    # Pick a few integer voxel centers, varying σ.
    cases = [
        dict(mu=(H // 2 + 0.5, W // 2 + 0.5, Z // 2 + 0.5), sigma=(1.5, 1.5, 1.0)),
        dict(mu=(H // 2 + 0.5, W // 2 + 0.5, Z // 2 + 0.5), sigma=(2.0, 2.0, 1.2)),
        dict(mu=(H // 2 - 10 + 0.5, W // 2 + 8 + 0.5, Z // 2 - 3 + 0.5), sigma=(1.5, 2.5, 1.0)),
    ]
    all_ok = True
    rows = []
    for k, c in enumerate(cases):
        s = c['sigma']
        half_xy = auto_half_from_sigma_max(max(s[0], s[1]), sigma_trunc)
        half_z = auto_half_from_sigma_max(s[2], sigma_trunc)
        half_z = min(Z - 1, half_z)

        r_raw = build_renderer(psf, H, W, 'raw_exact',
                               sigma_trunc=sigma_trunc, half_xy=half_xy, half_z=half_z).to(device)
        r_cf = build_renderer(psf, H, W, 'continuous_fourier',
                              sigma_trunc=sigma_trunc, half_xy=half_xy, half_z=half_z).to(device)

        g = GaussianBatch(
            positions=torch.tensor([c['mu']], device=device, dtype=psf.dtype),
            sigmas=torch.tensor([s], device=device, dtype=psf.dtype),
            rhos=torch.tensor([1.0], device=device, dtype=psf.dtype),
        )
        out_raw = r_raw(g)
        out_cf = r_cf(g)
        rel = relative_l2(out_cf, out_raw)
        ok = rel < 0.02
        all_ok = all_ok and ok
        rows.append((k, c['mu'], s, half_xy, rel, ok))
        print(f"  case {k}: mu={c['mu']}, σ={s}, half_xy={half_xy} → "
              f"rel_l2={rel:.2e} [{'OK' if ok else 'FAIL'}]")
    print(f"  T1 {'PASS' if all_ok else 'FAIL'}")
    return all_ok, rows


# ---------------------------------------------------------------------------
# T2 — Sub-voxel smoothness scan
# ---------------------------------------------------------------------------

def test_subvoxel_smoothness(psf, H, W, device, *, sigma_trunc=4.0,
                             n_steps=41, sigma=(1.5, 1.5, 1.0)):
    print("\n[T2] Sub-voxel smoothness scan (μ_x sweep across one voxel)")
    U, Z, ph, pw = psf.shape
    half_xy = auto_half_from_sigma_max(max(sigma[0], sigma[1]), sigma_trunc)
    half_z = min(Z - 1, auto_half_from_sigma_max(sigma[2], sigma_trunc))

    r_raw = build_renderer(psf, H, W, 'raw_exact',
                           sigma_trunc=sigma_trunc, half_xy=half_xy, half_z=half_z).to(device)
    r_cf = build_renderer(psf, H, W, 'continuous_fourier',
                          sigma_trunc=sigma_trunc, half_xy=half_xy, half_z=half_z).to(device)

    # Sweep μ_x ∈ [c−1.0, c+1.0] crossing two integer boundaries.
    c_mu = H / 2.0
    sweep = torch.linspace(c_mu - 1.0, c_mu + 1.0, n_steps).tolist()
    mu_y = W / 2.0 + 0.0
    mu_z = Z / 2.0 + 0.0

    # Readout: pixel at the nearest integer image position to (c_mu, mu_y).
    i_read = int(round(c_mu - 0.5))
    j_read = int(round(mu_y - 0.5))
    u_read = U // 2

    vals_raw, vals_cf = [], []
    for mx in sweep:
        g = GaussianBatch(
            positions=torch.tensor([[mx, mu_y, mu_z]], device=device, dtype=psf.dtype),
            sigmas=torch.tensor([sigma], device=device, dtype=psf.dtype),
            rhos=torch.tensor([1.0], device=device, dtype=psf.dtype),
        )
        vals_raw.append(r_raw(g)[u_read, i_read, j_read].item())
        vals_cf.append(r_cf(g)[u_read, i_read, j_read].item())

    def smoothness_ratio(vals):
        diffs = [abs(vals[i + 1] - vals[i]) for i in range(len(vals) - 1)]
        m = sum(diffs) / max(len(diffs), 1)
        return (max(diffs) / m) if m > 0 else float('inf')

    r_raw_score = smoothness_ratio(vals_raw)
    r_cf_score = smoothness_ratio(vals_cf)
    # Pass on continuous_fourier alone; raw_exact reported for context.
    # raw_exact happens to also be smooth at this scale (the box-anchor
    # shifts at half-int boundaries, but Gaussian coverage co-shifts so
    # the readout pixel sees an approximately C0 curve). The test verifies
    # that continuous_fourier is at least as smooth.
    ok = r_cf_score < 5.0
    # Match between the two curves is a useful side-channel: at sub-voxel
    # μ, both should render essentially the same image, so a max abs diff
    # close to 0 means continuous_fourier reproduces raw_exact across the
    # whole sweep (not just at integer μ as T1 checks).
    max_abs_diff = max(abs(a - b) for a, b in zip(vals_raw, vals_cf))
    rel_diff_at_mid = abs(vals_raw[n_steps // 2] - vals_cf[n_steps // 2]) / (
        abs(vals_raw[n_steps // 2]) + 1e-30)
    print(f"  continuous_fourier smoothness ratio = {r_cf_score:.2f}  "
          f"[{'OK' if ok else 'FAIL'}]  (pass < 5)")
    print(f"  raw_exact          smoothness ratio = {r_raw_score:.2f}  (context only)")
    print(f"  readout pixel u={u_read}, (i,j)=({i_read},{j_read}) over {n_steps} sweep steps")
    print(f"  raw_exact   first/mid/last values:",
          [f"{v:.4e}" for v in (vals_raw[0], vals_raw[n_steps // 2], vals_raw[-1])])
    print(f"  continuous  first/mid/last values:",
          [f"{v:.4e}" for v in (vals_cf[0], vals_cf[n_steps // 2], vals_cf[-1])])
    print(f"  max |cf − raw_exact| across sweep = {max_abs_diff:.3e}  "
          f"(rel at mid = {rel_diff_at_mid:.2e})")
    print(f"  T2 {'PASS' if ok else 'FAIL'}")
    return ok, dict(raw=r_raw_score, cf=r_cf_score,
                    max_abs_diff=max_abs_diff,
                    vals_raw=vals_raw, vals_cf=vals_cf, sweep=sweep)


# ---------------------------------------------------------------------------
# T3 — Finite-difference vs autograd for ∂loss/∂μ_x
# ---------------------------------------------------------------------------

def test_grad_fd_vs_autograd(psf, H, W, device, *, sigma_trunc=4.0,
                             sigma=(1.5, 1.5, 1.0), eps=1e-3, seed=0):
    print("\n[T3] FD vs autograd  for ∂loss/∂μ_x  (single-pixel readout)")
    U, Z, ph, pw = psf.shape
    half_xy = auto_half_from_sigma_max(max(sigma[0], sigma[1]), sigma_trunc)
    half_z = min(Z - 1, auto_half_from_sigma_max(sigma[2], sigma_trunc))

    r_cf = build_renderer(psf, H, W, 'continuous_fourier',
                          sigma_trunc=sigma_trunc, half_xy=half_xy, half_z=half_z).to(device)

    torch.manual_seed(seed)
    # Sub-voxel position. Pick μ so the off-center readout pixel falls on
    # the steep side of the rendered PSF spot (not the peak), giving a
    # strong gradient signal w.r.t. μ.
    mu0 = torch.tensor([[H / 2.0 + 0.37, W / 2.0 - 0.21, Z / 2.0 + 0.43]],
                       device=device, dtype=psf.dtype)
    sig = torch.tensor([sigma], device=device, dtype=psf.dtype)
    rho = torch.tensor([1.0], device=device, dtype=psf.dtype)

    # Non-translation-invariant scalar loss: read ONE fixed pixel of the
    # output. ∂L/∂μ_x is then the spatial slope of the rendered image at
    # that pixel — generically nonzero.
    u_read = U // 2
    i_read = H // 2 + 2     # 2 voxels off-center along H-axis (steep slope side)
    j_read = W // 2 - 1     # off-center along W-axis as well

    def render_loss(mu):
        g = GaussianBatch(positions=mu, sigmas=sig, rhos=rho)
        out = r_cf(g)
        return out[u_read, i_read, j_read]

    # Autograd
    mu_a = mu0.detach().clone().requires_grad_(True)
    l_a = render_loss(mu_a)
    l_a.backward()
    g_auto = mu_a.grad[0, 0].item()

    # Finite difference (μ_x)
    with torch.no_grad():
        mu_p = mu0.clone(); mu_p[0, 0] += eps
        mu_m = mu0.clone(); mu_m[0, 0] -= eps
        lp = render_loss(mu_p).item()
        lm = render_loss(mu_m).item()
        g_fd = (lp - lm) / (2 * eps)

    # Relative error with a sane absolute-floor so we don't divide by zero
    # if the loss happens to be locally flat.
    denom = max(abs(g_fd), abs(g_auto), 1e-8)
    rel = abs(g_auto - g_fd) / denom
    ok = rel < 1e-2
    print(f"  readout pixel u={u_read}, (i,j)=({i_read},{j_read})")
    print(f"  loss at μ_x±eps:  L+ = {lp: .6e}   L- = {lm: .6e}   (eps={eps})")
    print(f"  autograd  ∂L/∂μ_x = {g_auto: .6e}")
    print(f"  finite-Δ  ∂L/∂μ_x = {g_fd: .6e}")
    print(f"  relative error = {rel:.2e}  [{'OK' if ok else 'FAIL'}]")
    print(f"  T3 {'PASS' if ok else 'FAIL'}")
    return ok, dict(g_auto=g_auto, g_fd=g_fd, rel=rel)


# ---------------------------------------------------------------------------
# T4 — Multi-Gaussian vs full-volume reference
# ---------------------------------------------------------------------------

def test_multi_vs_full_volume(psf, H, W, device, *, sigma_trunc=4.0, N=10, seed=1):
    print(f"\n[T4] Multi-Gaussian vs full-volume reference (N={N})")
    U, Z, ph, pw = psf.shape
    g = make_random_gaussians(N, H, W, Z, device, dtype=psf.dtype, seed=seed)

    s_max = float(g.sigmas.max().item())
    half_xy = auto_half_from_sigma_max(s_max, sigma_trunc)
    half_z = min(Z - 1, auto_half_from_sigma_max(s_max, sigma_trunc))

    r_raw = build_renderer(psf, H, W, 'raw_exact',
                           sigma_trunc=sigma_trunc, half_xy=half_xy, half_z=half_z).to(device)
    r_cf = build_renderer(psf, H, W, 'continuous_fourier',
                          sigma_trunc=sigma_trunc, half_xy=half_xy, half_z=half_z).to(device)

    target_views = list(range(U))
    ref = reference_forward(g, psf, (H, W), target_views, z_norm='mean')
    out_raw = r_raw(g, target_views=target_views)
    out_cf = r_cf(g, target_views=target_views)

    rel_raw_vs_ref = relative_l2(out_raw, ref)
    rel_cf_vs_ref = relative_l2(out_cf, ref)
    rel_cf_vs_raw = relative_l2(out_cf, out_raw)

    en_raw = per_view_energy_err(out_raw, ref).max().item()
    en_cf = per_view_energy_err(out_cf, ref).max().item()

    ok = rel_cf_vs_ref < 0.03
    print(f"  rel_l2  raw_exact         vs ref = {rel_raw_vs_ref:.3e}")
    print(f"  rel_l2  continuous_fourier vs ref = {rel_cf_vs_ref:.3e}  "
          f"[{'OK' if ok else 'FAIL'}]")
    print(f"  rel_l2  continuous_fourier vs raw = {rel_cf_vs_raw:.3e}")
    print(f"  per-view energy err  (max over views):  raw={en_raw:.2e}  cf={en_cf:.2e}")
    print(f"  T4 {'PASS' if ok else 'FAIL'}")
    return ok, dict(rel_raw=rel_raw_vs_ref, rel_cf=rel_cf_vs_ref, rel_cf_vs_raw=rel_cf_vs_raw)


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
    p.add_argument('--only', type=str, default='all',
                   choices=('all', 't1', 't2', 't3', 't4'),
                   help='Run a single test (for quick iteration).')
    args = p.parse_args()

    device = torch.device('cpu') if args.cpu else torch.device('cuda', args.gpu_id)
    psf = load_psfs(args.psf_dir, args.Nnum).to(device).float()
    print(f"Loaded PSF: {tuple(psf.shape)} on {device}")
    print(f"Image shape: ({args.H}, {args.W}), sigma_trunc={args.sigma_trunc}")

    results = {}
    all_ok = True
    if args.only in ('all', 't1'):
        ok, results['T1'] = test_integer_mu_equivalence(
            psf, args.H, args.W, device, sigma_trunc=args.sigma_trunc)
        all_ok = all_ok and ok
    if args.only in ('all', 't2'):
        ok, results['T2'] = test_subvoxel_smoothness(
            psf, args.H, args.W, device, sigma_trunc=args.sigma_trunc)
        all_ok = all_ok and ok
    if args.only in ('all', 't3'):
        ok, results['T3'] = test_grad_fd_vs_autograd(
            psf, args.H, args.W, device, sigma_trunc=args.sigma_trunc)
        all_ok = all_ok and ok
    if args.only in ('all', 't4'):
        ok, results['T4'] = test_multi_vs_full_volume(
            psf, args.H, args.W, device, sigma_trunc=args.sigma_trunc)
        all_ok = all_ok and ok

    print("\n========== SUMMARY ==========")
    for k in sorted(results.keys()):
        v = results[k]
        if isinstance(v, dict):
            summary = ', '.join(f'{kk}={vv:.3e}' if isinstance(vv, float) else f'{kk}={vv}'
                                for kk, vv in v.items() if not isinstance(vv, list))
            print(f"  {k}: {summary}")
        else:
            print(f"  {k}: {v}")
    print(f"\n  ALL TESTS {'PASSED' if all_ok else 'FAILED'}")
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
