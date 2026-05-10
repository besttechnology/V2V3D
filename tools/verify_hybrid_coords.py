"""Verify V2V3D_Gauss decoder output renders correctly via HybridRenderer.

Two checks:
  1. Hybrid renderer (raw_exact) vs `gaussians_to_full_volume + generate_fps`.
     If the math is right, these match at machine precision regardless of how
     wild the gp values are. This validates the (x,y,z) ↔ (H,W,D) convention
     end-to-end against V2V3D_Gauss's actual output format.
  2. Untrained model output → render at remain_v views: should produce a
     non-zero, non-NaN image with reasonable mass distribution.
"""
import os, sys, math
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_psfs, generate_fps
from hybrid_renderer import HybridRenderer, GaussianBatch, gaussians_to_full_volume
from gaussian_utils import make_grid_centers, decoder_output_to_gaussians


def check_consistency(device, H=64, W=64, Z=39, U=13, n_views=6, seed=0):
    """Match-test: V2V3D_Gauss-like raw output → gp → hybrid vs full-volume."""
    print('\n=== Check 1: V2V3D_Gauss raw → gp → hybrid vs full-volume ===')
    torch.manual_seed(seed)

    # Synthetic V2V3D_Gauss-style raw output: (B=1, 11, D, H, W)
    raw = torch.randn(1, 11, Z, H, W, device=device)
    # Bias rho channel down so most are small but a few are big enough to render.
    raw[0, 0] -= 3.0
    grid = make_grid_centers(D=Z, H=H, W=W, device=device)

    gp = decoder_output_to_gaussians(raw, grid, scale_init=0.5,
                                     max_offset=0.5, s_min=0.5, s_max=2.0)
    print(f'  gp positions range: x=[{gp["positions"][:,0].min():.2f},{gp["positions"][:,0].max():.2f}] '
          f'y=[{gp["positions"][:,1].min():.2f},{gp["positions"][:,1].max():.2f}] '
          f'z=[{gp["positions"][:,2].min():.2f},{gp["positions"][:,2].max():.2f}]')
    print(f'  gp scales range: [{gp["scales"].min():.2f}, {gp["scales"].max():.2f}]')
    print(f'  gp rho range: [{gp["densities"].min():.4e}, {gp["densities"].max():.4e}]')

    # Filter to top-K by density for tractable comparison.
    rho = gp['densities'].reshape(-1)
    K = 200
    top_idx = torch.topk(rho, K).indices
    g = GaussianBatch(
        positions=gp['positions'][top_idx],
        sigmas=gp['scales'][top_idx],
        rhos=rho[top_idx],
    )

    # Build PSF (random for the consistency check — both paths use the same PSF)
    psf = torch.rand(U, Z, 32, 32, device=device) * 1e-3
    target_views = list(range(n_views))

    # Path A: hybrid renderer (per-G loop, no fixed half — use most accurate)
    rend = HybridRenderer(psf, (H, W), z_norm='mean', mode='raw_exact').to(device)
    out_hybrid = rend(g, target_views=target_views)

    # Path B: gaussians_to_full_volume + generate_fps + mean(z)
    vol = gaussians_to_full_volume(g, (Z, H, W), device=device)
    fp = generate_fps(psf[target_views], vol)
    out_ref = fp.mean(dim=1)

    rel = ((out_hybrid - out_ref).norm() / out_ref.norm()).item()
    print(f'  rel L2 (hybrid vs full-vol+fps) = {rel:.4e}')
    print(f'  hybrid energy = {out_hybrid.sum().item():.4e}, ref energy = {out_ref.sum().item():.4e}')
    ok = rel < 1e-3
    print(f'  PASS={ok}  (threshold rel L2 < 1e-3)')
    return ok


def check_realistic_render(device, input_size=64, seed=0):
    """Forward an actual untrained V2V3D_Gauss-style decoder output and verify
    the rendered LFI is non-pathological (no NaN, non-zero, peak interior)."""
    print('\n=== Check 2: Realistic render scale + sanity ===')
    torch.manual_seed(seed)

    psf_dir = 'PSF/PSF_zoom2_39dz1_N13'
    psfs = load_psfs(psf_dir, 13).to(device).float()
    U, Z = psfs.shape[:2]
    H = W = input_size

    # Synthetic gp (mimics V2V3D_Gauss with rho_bias=-5 default)
    raw = torch.randn(1, 11, Z, H, W, device=device) * 0.1
    raw[0, 0] -= 5.0  # rho bias as init
    grid = make_grid_centers(D=Z, H=H, W=W, device=device)
    gp = decoder_output_to_gaussians(raw, grid, scale_init=0.5,
                                     max_offset=0.5, s_min=0.1, s_max=3.0)

    rho = gp['densities'].reshape(-1)
    n_active = (rho > 0.005).sum().item()
    print(f'  total Gaussians = {rho.numel()}, ρ>0.005 count = {n_active}')
    K = min(2000, rho.numel())
    top_idx = torch.topk(rho, K).indices
    g = GaussianBatch(
        positions=gp['positions'][top_idx],
        sigmas=gp['scales'][top_idx],
        rhos=rho[top_idx],
    )

    rend = HybridRenderer(
        psfs, (H, W), z_norm='mean', mode='raw_exact',
        fixed_half_xy=5, fixed_half_z=5, chunk_size=32, use_checkpoint=False,
    ).to(device)
    target_views = [0, 6, 12]   # one extreme + center + other extreme
    out = rend(g, target_views=target_views)
    print(f'  out shape={tuple(out.shape)} dtype={out.dtype}')
    print(f'  out: min={out.min().item():.4e} max={out.max().item():.4e} '
          f'mean={out.mean().item():.4e} std={out.std().item():.4e}')
    has_nan = torch.isnan(out).any().item()
    is_const = (out.max() - out.min()).item() < 1e-12

    # Locate peak in center view to check it's not pinned at a corner.
    img_c = out[1]
    max_idx = img_c.flatten().argmax().item()
    py, px = max_idx // W, max_idx % W
    peak_at_edge = py < 2 or py >= H - 2 or px < 2 or px >= W - 2
    print(f'  center-view peak at ({py},{px})  edge={peak_at_edge}')

    ok = (not has_nan) and (not is_const) and (not peak_at_edge)
    print(f'  PASS={ok}  (NaN={has_nan}, const={is_const}, peak_edge={peak_at_edge})')
    return ok


def main():
    device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')
    ok1 = check_consistency(device)
    ok2 = check_realistic_render(device)
    print(f'\n=== ALL: {"PASSED" if (ok1 and ok2) else "FAILED"} ===')
    sys.exit(0 if (ok1 and ok2) else 1)


if __name__ == '__main__':
    main()
