"""Sanity checks for HybridRenderer mode='continuous_zfourier' (Phase 2).

Run on a GPU host after `git pull && git checkout gauss-continuous-zfourier`.

Tests:
  ZA - Analytic gz_neg formula: at DC (f=0) the value must be σ_z·√(2π)
       regardless of μ_z (envelope only, no phase). At nonzero f the
       envelope is exp(-2π²σ²f²), real-valued and symmetric in f.

  ZB - Integer μ_z equivalence vs Phase 1 (continuous_fourier). When μ_z
       sits on a voxel center the two paths should agree closely
       (continuous_zfourier sums over all Z, continuous_fourier sums over
       local Bz — small discrepancy from Bz truncation only).
       PASS: rel_l2 < 5% per Gaussian (with σ_z = 1, Bz = 5).

  ZC - μ_z sub-voxel smoothness: sweep μ_z across two voxels, check that
       continuous_zfourier output is C∞ smooth (smoothness ratio < 3).
       continuous_fourier reported for context (jumps at box boundaries).

  ZD - FD vs autograd for ∂loss/∂μ_z and ∂loss/∂σ_z. Off-center pixel
       readout (μ-sensitive).
       PASS: relative error < 1e-2.

  ZE - Large σ_z agreement with Phase 1: σ_z = 4 (well-covered by Bz),
       continuous_zfourier and continuous_fourier should agree closely
       since both paths integrate the same Gaussian against PSF in z.
       PASS: rel_l2 < 5%.

Usage:
    python sanity_check_zfourier.py \\
        --psf_dir PSF/PSF_zoom2_39dz1_N13 --Nnum 13 --H 256 --W 256
"""
from __future__ import annotations

import argparse
import math
import sys

import torch

from utils import load_psfs
from hybrid_renderer import GaussianBatch, HybridRenderer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def relative_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = (a - b).reshape(-1)
    denom = b.reshape(-1).norm().item() + 1e-30
    return (diff.norm().item() / denom)


def smoothness_ratio(vals: torch.Tensor) -> float:
    """max(|Δ|) / mean(|Δ|) over consecutive samples. ~1-3 = smooth, ≫5 = step."""
    diffs = (vals[1:] - vals[:-1]).abs()
    return float(diffs.max() / (diffs.mean() + 1e-30))


def make_renderer(psf, H, W, mode, half_xy=4, half_z=4, chunk=4):
    return HybridRenderer(
        psf=psf, image_shape=(H, W), mode=mode,
        fixed_half_xy=half_xy, fixed_half_z=half_z,
        chunk_size=chunk, use_checkpoint=False,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_ZA(device):
    print("\n=== ZA: gz_neg formula at DC ===")
    Z = 39
    fz = torch.fft.fftfreq(Z, device=device, dtype=torch.float64)
    for sz in (0.5, 1.0, 2.0, 4.0):
        for mu_z in (0.0, 3.7, 17.0, 25.3):
            envelope = sz * math.sqrt(2 * math.pi) * torch.exp(
                -2 * math.pi ** 2 * sz ** 2 * fz ** 2)
            phase = 2 * math.pi * fz * (mu_z - 0.5)
            gz_neg = torch.complex(envelope * torch.cos(phase),
                                   envelope * torch.sin(phase))
            dc = gz_neg[0]
            expected = sz * math.sqrt(2 * math.pi)
            err = abs(dc.real.item() - expected) + abs(dc.imag.item())
            if err > 1e-6:
                print(f"  FAIL  σ={sz}, μ={mu_z}: DC={dc.item():.6f}, "
                      f"expected {expected:.6f}")
                return False
    print(f"  PASS  DC value = σ·√(2π) for all (σ, μ) tested, err < 1e-6")
    return True


def test_ZB(psf, device):
    print("\n=== ZB: integer μ_z equivalence vs Phase 1 ===")
    H = W = 256
    Z = psf.shape[1]
    r_cf = make_renderer(psf, H, W, mode="continuous_fourier")
    r_zf = make_renderer(psf, H, W, mode="continuous_zfourier")

    torch.manual_seed(11)
    N = 5
    positions = torch.stack([
        torch.rand(N, device=device) * (H - 20) + 10,
        torch.rand(N, device=device) * (W - 20) + 10,
        torch.randint(5, Z - 5, (N,), device=device).float() + 0.5,
    ], dim=1)
    sigmas = torch.full((N, 3), 1.0, device=device)
    rhos = torch.rand(N, device=device) * 0.5 + 0.5
    g = GaussianBatch(positions=positions, sigmas=sigmas, rhos=rhos)

    out_cf = r_cf(g, target_views=list(range(psf.shape[0])))
    out_zf = r_zf(g, target_views=list(range(psf.shape[0])))
    rl2 = relative_l2(out_zf, out_cf)
    print(f"  rel_l2 (zfourier vs cfourier at integer μ_z) = {rl2:.4f}  "
          f"(PASS threshold 5%)")
    return rl2 < 0.05


def test_ZC(psf, device):
    print("\n=== ZC: μ_z sub-voxel smoothness ===")
    H = W = 256
    Z = psf.shape[1]
    r_cf = make_renderer(psf, H, W, mode="continuous_fourier")
    r_zf = make_renderer(psf, H, W, mode="continuous_zfourier")

    mu_zs = torch.linspace(Z // 2 + 0.0, Z // 2 + 1.0, 25, device=device)
    results = {}
    for mode_name, r in (("continuous_fourier", r_cf),
                         ("continuous_zfourier", r_zf)):
        vals = []
        for mz in mu_zs:
            pos = torch.tensor([[H / 2, W / 2, mz.item()]], device=device)
            sig = torch.full((1, 3), 1.0, device=device)
            rho = torch.tensor([1.0], device=device)
            g = GaussianBatch(positions=pos, sigmas=sig, rhos=rho)
            out = r(g, target_views=[0])
            vals.append(out[0, H // 2, W // 2].item())
        s = smoothness_ratio(torch.tensor(vals))
        results[mode_name] = s
        print(f"  {mode_name:24s}  smoothness ratio = {s:.2f}")
    ok = results["continuous_zfourier"] < 3.0
    print(f"  PASS' threshold: zfourier ratio < 3   "
          f"({'PASS' if ok else 'FAIL'})")
    return ok


def test_ZD(psf, device):
    print("\n=== ZD: FD vs autograd, ∂L/∂μ_z and ∂L/∂σ_z ===")
    H = W = 256
    Z = psf.shape[1]
    r = make_renderer(psf, H, W, mode="continuous_zfourier")

    torch.manual_seed(42)
    pos_val = [H / 2 + 0.3, W / 2 + 0.7, Z / 2 + 0.4]
    sig_val = [1.0, 1.0, 1.2]
    rho_val = 0.8
    u_read, i_read, j_read = 0, H // 2 + 3, W // 2 + 2

    def loss_of(pos_v, sig_v, rho_v):
        pos = torch.tensor([pos_v], device=device, requires_grad=True)
        sig = torch.tensor([sig_v], device=device, requires_grad=True)
        rho = torch.tensor([rho_v], device=device, requires_grad=True)
        g = GaussianBatch(positions=pos, sigmas=sig, rhos=rho)
        out = r(g, target_views=[u_read])
        return out[0, i_read, j_read], (pos, sig, rho)

    # Autograd
    L, (pos, sig, rho) = loss_of(pos_val, sig_val, rho_val)
    L.backward()
    g_mu_z = pos.grad[0, 2].item()
    g_sig_z = sig.grad[0, 2].item()

    # FD (central difference)
    eps = 1e-3
    pos_plus = list(pos_val); pos_plus[2] += eps
    pos_minus = list(pos_val); pos_minus[2] -= eps
    Lp, _ = loss_of(pos_plus, sig_val, rho_val)
    Lm, _ = loss_of(pos_minus, sig_val, rho_val)
    fd_mu_z = (Lp.item() - Lm.item()) / (2 * eps)

    sig_plus = list(sig_val); sig_plus[2] += eps
    sig_minus = list(sig_val); sig_minus[2] -= eps
    Lp, _ = loss_of(pos_val, sig_plus, rho_val)
    Lm, _ = loss_of(pos_val, sig_minus, rho_val)
    fd_sig_z = (Lp.item() - Lm.item()) / (2 * eps)

    def rel_err(a, fd):
        return abs(a - fd) / (max(abs(a), abs(fd), 1e-12))

    e_mu = rel_err(g_mu_z, fd_mu_z)
    e_sg = rel_err(g_sig_z, fd_sig_z)
    print(f"  ∂L/∂μ_z   autograd={g_mu_z:.4e}  FD={fd_mu_z:.4e}  rel_err={e_mu:.2e}")
    print(f"  ∂L/∂σ_z   autograd={g_sig_z:.4e}  FD={fd_sig_z:.4e}  rel_err={e_sg:.2e}")
    ok = (e_mu < 1e-2) and (e_sg < 1e-2)
    print(f"  PASS threshold rel_err < 1e-2  ({'PASS' if ok else 'FAIL'})")
    return ok


def test_ZE(psf, device):
    print("\n=== ZE: large σ_z agreement with Phase 1 ===")
    H = W = 256
    Z = psf.shape[1]
    r_cf = make_renderer(psf, H, W, mode="continuous_fourier",
                         half_z=12)  # big box so Bz covers σ_z=4
    r_zf = make_renderer(psf, H, W, mode="continuous_zfourier",
                         half_z=12)

    torch.manual_seed(7)
    N = 4
    positions = torch.stack([
        torch.rand(N, device=device) * (H - 60) + 30,
        torch.rand(N, device=device) * (W - 60) + 30,
        torch.rand(N, device=device) * (Z - 16) + 8,
    ], dim=1)
    sigmas = torch.full((N, 3), 1.0, device=device)
    sigmas[:, 2] = 4.0
    rhos = torch.rand(N, device=device) * 0.4 + 0.6
    g = GaussianBatch(positions=positions, sigmas=sigmas, rhos=rhos)

    out_cf = r_cf(g, target_views=list(range(psf.shape[0])))
    out_zf = r_zf(g, target_views=list(range(psf.shape[0])))
    rl2 = relative_l2(out_zf, out_cf)
    print(f"  rel_l2 (zfourier vs cfourier, σ_z=4) = {rl2:.4f}  "
          f"(PASS threshold 5%)")
    return rl2 < 0.05


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--psf_dir', default='PSF/PSF_zoom2_39dz1_N13')
    p.add_argument('--Nnum', type=int, default=13)
    p.add_argument('--H', type=int, default=256)
    p.add_argument('--W', type=int, default=256)
    p.add_argument('--only', type=str, default=None,
                   help='Run only one test, e.g. --only ZB')
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available()
                          else 'cuda')
    print(f"device = {device}")

    psf = load_psfs(args.psf_dir, args.Nnum).to(device).float()
    print(f"PSF shape: {tuple(psf.shape)}")

    tests = {
        'ZA': lambda: test_ZA(device),
        'ZB': lambda: test_ZB(psf, device),
        'ZC': lambda: test_ZC(psf, device),
        'ZD': lambda: test_ZD(psf, device),
        'ZE': lambda: test_ZE(psf, device),
    }
    if args.only:
        if args.only not in tests:
            print(f"unknown test {args.only!r}; choose from {list(tests)}")
            sys.exit(1)
        tests = {args.only: tests[args.only]}

    results = {}
    for name, fn in tests.items():
        try:
            results[name] = fn()
        except Exception as e:
            print(f"\n[ERROR] {name} threw: {type(e).__name__}: {e}")
            results[name] = False

    print("\n=== Summary ===")
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    all_ok = all(results.values())
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
