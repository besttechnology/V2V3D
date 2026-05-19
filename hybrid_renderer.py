"""Hybrid Gaussian → real-PSF renderer.

Three modes implemented:

mode="raw_exact" (Phase 1):
    Use raw PSF; splat each Gaussian's local conv patch around its rounded
    voxel position. The raw PSF carries the full view/depth-dependent shift,
    so the image peak naturally lands at (mu_xy + raw_centroid_offset).
    Numerically equivalent to V2V3D's full-volume `generate_fps(...).mean(1)`.

mode="centered_affine" (Phase 2/3):
    Use *centered* PSF (centroid pre-shifted to V2V3D kernel center); place
    each Gaussian's patch at the affine-projected image position
        anchor_continuous[u] = mu_xy + centroid_offset[u, cz_idx]
    where `centroid_offset` is a table (U, Z, 2) supplied by the caller.
    Geometry (centroid_offset) is decoupled from shape (centered PSF), so
    no double shift.

    splat_mode controls how the continuous anchor is materialized:
      "round"    — anchor = round(continuous). Phase 2 sanity only;
                   incurs sub-pixel quantization error ~Δ/(σ√2).
      "bilinear" — 4-corner weighted splat using floor/frac of continuous
                   anchor. Recovers PSF sub-pixel info; this is what
                   Phase 3 differentiable training will use.

mode="continuous_fourier" (Phase 1 of the continuization plan):
    Use *raw* PSF (no centering, no offset table). Skip voxel-grid Gaussian
    sampling entirely: build the Gaussian's analytical 2D Fourier transform
    directly on the (r_x, r_y) DFT grid that matches psf_freq, then multiply
    and irfft2. The sub-voxel position of mu_xy is encoded as a phase ramp
    exp(-2πi(fx·mu_local_x + fy·mu_local_y)) — exact, smoothly differentiable.
    The xy envelope is the closed-form Gaussian FT
        ρ · 2π σx σy · exp(-2π²(σx² fx² + σy² fy²)).
    Splat anchor stays at the integer (cx_idx, cy_idx) — the raw PSF carries
    the view/depth-dependent shift, just like raw_exact.

    Phase 1 of this mode: z dimension stays nearest-z (Gaussian z-density
    evaluated at integer voxel centers in the local box, same as raw_exact's
    discrete sampling). Sub-voxel mu_z gradient flows through the analytic
    density formula but is quantized at the box level. Phase 2 (future)
    will replace this with an analytic erf-based integral over each slice
    interval, giving full mu_z sub-voxel grad continuity.

    Requires fixed_half_xy and fixed_half_z (uses pre-FFT'd psf_freq cache).
    centroid_offset / splat_mode are not used.

Coordinate convention (matches gaussian_utils.make_grid_centers):
    position = (x, y, z) where
        x  ←→  H axis (rows of image)
        y  ←→  W axis (cols of image)
        z  ←→  D axis (depth slices)
    voxel center at (i + 0.5) for integer index i.
    centroid_offset[u, z] is in (H-axis, W-axis) order — same as mu[0:2].

Conv convention (matches utils.generate_fps):
    LFI_u = (1/Z_total) · Σ_z (V_z ⊛ PSF_{u,z})
    `⊛` is mathematical convolution implemented via FFT zero-pad.
    PSF kernel center is placed at index ((ph-1)//2, (pw-1)//2);
    for the typical even ph=pw=128 this is index 63 — asymmetric, but
    matches V2V3D's `int((rb-1)/2)` crop offset.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
from torch.fft import fft2, ifft2


@dataclass
class GaussianBatch:
    """Phase 1 only supports diagonal covariance.

    positions: (N, 3) — (x=H-axis, y=W-axis, z=D-axis), voxel-world units
    sigmas:    (N, 3) — (s_x, s_y, s_z), positive
    rhos:      (N,)  or (N, 1) — non-negative intensity
    """
    positions: torch.Tensor
    sigmas: torch.Tensor
    rhos: torch.Tensor


def gaussians_from_decoder(gp: dict) -> GaussianBatch:
    """Convert gaussian_utils decoder_output_to_gaussians dict → GaussianBatch.

    Phase 1 ignores rotations (diagonal cov only).
    """
    rhos = gp['densities'].reshape(-1)
    return GaussianBatch(
        positions=gp['positions'],
        sigmas=gp['scales'],
        rhos=rhos,
    )


# ---------------------------------------------------------------------------
# Local FFT conv that matches generate_fps exactly
# ---------------------------------------------------------------------------

def _fft_conv2d_full(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Linear 2D convolution via FFT zero-pad. a[..., ra, ca], b[..., rb, cb].

    Returns shape [..., ra+rb-1, ca+cb-1]. Mirrors `generate_fps`'s pad+fft+ifft.
    Real part is taken (matches `generate_fps` for the multi-view branch).
    """
    *lead, ra, ca = a.shape
    *_, rb, cb = b.shape
    r = ra + rb - 1
    c = ca + cb - 1
    a_pad = a.new_zeros(*a.shape[:-2], r, c)
    b_pad = b.new_zeros(*b.shape[:-2], r, c)
    a_pad[..., :ra, :ca] = a
    b_pad[..., :rb, :cb] = b
    out = torch.real(ifft2(fft2(a_pad) * fft2(b_pad)))
    return out


# ---------------------------------------------------------------------------
# Diagonal-Gaussian local box voxelize (analytic eval at integer voxel centers)
# ---------------------------------------------------------------------------

def _box_indices(center_idx: int, half: int, lo: int, hi: int):
    """Return (idxs, valid_mask) where idxs spans [center_idx-half, center_idx+half]
    and valid_mask flags entries inside [lo, hi). Out-of-range entries are kept
    so tensor shapes stay regular; we mask them to zero before conv.
    """
    idxs = torch.arange(center_idx - half, center_idx + half + 1)
    mask = (idxs >= lo) & (idxs < hi)
    return idxs, mask


# ---------------------------------------------------------------------------
# The renderer
# ---------------------------------------------------------------------------

class HybridRenderer(nn.Module):
    """Phase 1 raw-PSF exact renderer.

    Args:
        psf:               (U, Z, ph, pw) PSF tensor. For mode='raw_exact'
                           this is the raw PSF; for 'centered_affine' it
                           must be the centered PSF (each (u,z) centroid at
                           V2V3D's discrete kernel center ((ph-1)//2, (pw-1)//2)).
        image_shape:       (H, W) target LFI spatial shape.
        sigma_truncation:  box half-extent in σ (4.0 recommended for Phase 1/2).
        z_norm:            "mean" → divide by total Z (matches train_model.py
                           `gen_*.mean(dim=1)`), "sum" → no division.
        mode:              "raw_exact" (Phase 1) or "centered_affine" (Phase 2).
        centroid_offset:   (U, Z, 2) table giving image-space offset
                           (h_offset, w_offset) to add to mu_xy for each
                           (view, integer-z-slice). Required for
                           'centered_affine'. Ignored for 'raw_exact'.
    """

    def __init__(
        self,
        psf: torch.Tensor,
        image_shape: tuple[int, int],
        sigma_truncation: float = 4.0,
        z_norm: str = "mean",
        mode: str = "raw_exact",
        centroid_offset: torch.Tensor | None = None,
        splat_mode: str = "round",
        fixed_half_xy: int | None = None,
        fixed_half_z: int | None = None,
        chunk_size: int = 64,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        assert psf.ndim == 4, f"psf shape must be (U,Z,ph,pw), got {psf.shape}"
        assert mode in ("raw_exact", "centered_affine", "continuous_fourier"), \
            f"unsupported mode={mode!r}"
        assert z_norm in ("mean", "sum")
        assert splat_mode in ("round", "bilinear"), \
            f"unsupported splat_mode={splat_mode!r}"
        if mode == "raw_exact" and splat_mode != "round":
            raise ValueError(
                "raw_exact mode requires splat_mode='round': the raw PSF "
                "already carries sub-pixel shift via FFT conv, so bilinear "
                "splat would double-count the offset.")
        if mode == "continuous_fourier":
            if fixed_half_xy is None or fixed_half_z is None:
                raise ValueError(
                    "continuous_fourier mode requires fixed_half_xy and "
                    "fixed_half_z (uses pre-FFT'd psf_freq cache for the "
                    "analytic-Gaussian freq-domain compute path).")
            if splat_mode != "round":
                raise ValueError(
                    "continuous_fourier mode requires splat_mode='round': "
                    "the analytic Gaussian's phase shift already encodes the "
                    "sub-voxel mu_xy position inside the patch content.")
        self.register_buffer('psf', psf, persistent=False)
        self.U, self.Z, self.ph, self.pw = psf.shape
        self.H, self.W = image_shape
        self.sigma_trunc = float(sigma_truncation)
        self.z_norm = z_norm
        self.mode = mode
        self.splat_mode = splat_mode
        self.chunk_size = int(chunk_size)
        self.use_checkpoint = bool(use_checkpoint)

        # Vectorized batch path: enabled when both fixed_half_xy and fixed_half_z
        # are supplied. Pre-FFT the PSF on the resulting (r_x, r_y) so per-Gaussian
        # PSF FFT is amortized to a single up-front cost. Per-Gaussian path
        # (the original) still works when these are None — useful for tests
        # where σ varies a lot or you don't want to pre-compute a fixed cache.
        self.fixed_half_xy = fixed_half_xy
        self.fixed_half_z = fixed_half_z
        if fixed_half_xy is not None and fixed_half_z is not None:
            Bx = 2 * fixed_half_xy + 1
            By = Bx
            r_x = Bx + self.ph - 1
            r_y = By + self.pw - 1
            psf_pad = psf.new_zeros(self.U, self.Z, r_x, r_y)
            psf_pad[:, :, :self.ph, :self.pw] = psf
            # Use rfft2 — for real input the spectrum is Hermitian, so we
            # only need half the W-axis bins (r_y // 2 + 1). Halves the
            # complex tensor memory + bandwidth in the per-chunk mul step.
            psf_freq = torch.fft.rfft2(psf_pad)
            self.register_buffer('psf_freq', psf_freq, persistent=False)
            self.r_x = r_x
            self.r_y = r_y
        else:
            self.psf_freq = None
            self.r_x = None
            self.r_y = None

        if mode == "centered_affine":
            assert centroid_offset is not None, \
                "centered_affine requires centroid_offset table"
            assert centroid_offset.shape == (self.U, self.Z, 2), \
                f"centroid_offset shape must be ({self.U},{self.Z},2), " \
                f"got {tuple(centroid_offset.shape)}"
            self.register_buffer('centroid_offset', centroid_offset.float(),
                                 persistent=False)
        else:
            self.centroid_offset = None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        gaussians: GaussianBatch,
        target_views: Sequence[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Render the given Gaussians into LFI views.

        Returns: (Vt, H, W) where Vt = len(target_views) (default: all views).
        """
        if target_views is None:
            target_views = list(range(self.U))
        if isinstance(target_views, torch.Tensor):
            target_views = target_views.tolist()
        target_views = list(target_views)

        device = gaussians.positions.device
        out = torch.zeros(len(target_views), self.H, self.W, device=device,
                          dtype=gaussians.positions.dtype)

        N = gaussians.positions.shape[0]

        if N > 0:
            if self.psf_freq is not None:
                # Vectorized path: process Gaussians in chunks of chunk_size.
                for chunk_start in range(0, N, self.chunk_size):
                    chunk_end = min(chunk_start + self.chunk_size, N)
                    self._render_chunk(out, gaussians, chunk_start, chunk_end,
                                       target_views)
            else:
                # Per-Gaussian Python loop (original path).
                for i in range(N):
                    mu = gaussians.positions[i]      # (3,) (x, y, z)
                    sigma = gaussians.sigmas[i]      # (3,) (sx, sy, sz)
                    rho = gaussians.rhos[i]
                    self._render_one(out, mu, sigma, rho, target_views)

        # Ghost-grad guard: when every _splat call early-returned (all
        # anchors fell outside (H, W)) or N==0, `out` is still the leaf
        # zeros from above with no grad_fn, and loss.backward() would die
        # with "element 0 ... does not have a grad_fn". The 0.0 multiplier
        # keeps values unchanged but reconnects autograd to Gaussian params,
        # so degenerate batches receive zero gradient instead of crashing.
        zero_link = 0.0 * (
            gaussians.positions.sum()
            + gaussians.sigmas.sum()
            + gaussians.rhos.sum()
        )
        return out + zero_link

    # ------------------------------------------------------------------
    # Per-Gaussian render
    # ------------------------------------------------------------------
    def _render_one(
        self,
        out: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        rho: torch.Tensor,
        target_views: list[int],
    ):
        device = mu.device
        dtype = mu.dtype

        # Keep sigmas as tensors so autograd flows through v's analytic eval.
        # Box half-extent only needs a Python int — use detached values.
        sigma_clamped = sigma.clamp(min=1e-6)
        sx = sigma_clamped[0]; sy = sigma_clamped[1]; sz = sigma_clamped[2]
        sx_det = sx.detach().item()
        sy_det = sy.detach().item()
        sz_det = sz.detach().item()
        half_x = max(1, int(math.ceil(self.sigma_trunc * sx_det)))
        half_y = max(1, int(math.ceil(self.sigma_trunc * sy_det)))
        half_z = max(1, int(math.ceil(self.sigma_trunc * sz_det)))

        # Anchor: nearest integer voxel center to mu. Detached — gradient on
        # mu flows through v (analytic Gaussian eval at integer voxel centers),
        # not through the integer box position.
        with torch.no_grad():
            cx_idx = int(torch.round(mu[0] - 0.5).item())
            cy_idx = int(torch.round(mu[1] - 0.5).item())
            cz_idx = int(torch.round(mu[2] - 0.5).item())

        # Local box global integer voxel indices.
        bx_idxs, bx_valid = _box_indices(cx_idx, half_x, 0, self.H)
        by_idxs, by_valid = _box_indices(cy_idx, half_y, 0, self.W)
        bz_idxs, bz_valid = _box_indices(cz_idx, half_z, 0, self.Z)
        Bx = bx_idxs.numel(); By = by_idxs.numel(); Bz = bz_idxs.numel()

        # World coords of voxel centers (continuous).
        xs = (bx_idxs.to(device).to(dtype) + 0.5)
        ys = (by_idxs.to(device).to(dtype) + 0.5)
        zs = (bz_idxs.to(device).to(dtype) + 0.5)

        # Diagonal Gaussian eval at each voxel center.
        # v shape: (Bz, Bx, By) — matches volume[D, H, W] order.
        zz = zs.view(Bz, 1, 1)
        xx = xs.view(1, Bx, 1)
        yy = ys.view(1, 1, By)
        v = rho * torch.exp(
            -0.5 * (
                ((zz - mu[2]) / sz) ** 2
                + ((xx - mu[0]) / sx) ** 2
                + ((yy - mu[1]) / sy) ** 2
            )
        )

        # Mask out-of-volume voxels (their conv contribution would not be in
        # the original full-volume forward either).
        if not (bx_valid.all() and by_valid.all() and bz_valid.all()):
            mask = (
                bz_valid.to(device).view(Bz, 1, 1)
                & bx_valid.to(device).view(1, Bx, 1)
                & by_valid.to(device).view(1, 1, By)
            )
            v = v * mask.to(v.dtype)

        # PSF gather: for each local z, take psf[u, z_global]. With nearest-z,
        # z_global is just the (clamped, masked) integer index.
        # Phase 1 only: out-of-range z is masked above, so we clamp the gather
        # index but those voxels contribute zero anyway.
        bz_global_clamped = bz_idxs.clamp(0, self.Z - 1).to(device)

        # Stack PSFs for the requested views: (Vt, Bz, ph, pw)
        view_idx = torch.as_tensor(target_views, device=device)
        psf_block = self.psf.index_select(0, view_idx)         # (Vt, Z, ph, pw)
        psf_block = psf_block.index_select(1, bz_global_clamped)  # (Vt, Bz, ph, pw)

        # FFT-based linear conv per (view, local_z): output (Vt, Bz, r_x, r_y).
        # v needs shape (1, Bz, Bx, By) to broadcast against (Vt, Bz, ph, pw).
        v_b = v.unsqueeze(0)                                    # (1, Bz, Bx, By)
        conv_out = _fft_conv2d_full(v_b, psf_block)             # (Vt, Bz, r_x, r_y)
        # Sum over local z and apply 1/Z normalization to match generate_fps.
        patch = conv_out.sum(dim=1)                             # (Vt, r_x, r_y)
        if self.z_norm == "mean":
            patch = patch / float(self.Z)

        # Splat: place each per-view patch into the output image. Patch idx
        # (i, j) maps to image idx (i + offset_x, j + offset_y), with offset
        # chosen so the PSF peak (kernel idx (ph-1)//2 in patch coords) lands
        # at the splat anchor in image space.
        #
        # raw_exact:        anchor = (cx_idx, cy_idx)            (raw PSF carries shift)
        # centered_affine:  anchor_continuous = mu_xy + centroid_offset[u, cz]
        r_x = Bx + self.ph - 1
        r_y = By + self.pw - 1

        if self.mode == "raw_exact":
            for u_out_idx in range(len(target_views)):
                self._splat(out, u_out_idx, patch[u_out_idx],
                            cx_idx, cy_idx, half_x, half_y, r_x, r_y, 1.0)
            return

        # centered_affine
        # Continuous splat anchor in pixel-idx units: (cx_idx + 0.5) + offs.
        # The (cx_idx + 0.5) term is the world coord of the local-box center
        # voxel; mu's *sub-voxel* offset is already encoded in the patch
        # content (v's analytic Gaussian eval), so re-adding it via mu would
        # double-count the offset. See derivation in the module docstring.
        cz_lookup = max(0, min(self.Z - 1, cz_idx))
        offs = self.centroid_offset[:, cz_lookup, :]   # (U, 2)
        anchor_x_world = cx_idx + 0.5
        anchor_y_world = cy_idx + 0.5

        for u_out_idx, u in enumerate(target_views):
            cont_h = anchor_x_world + offs[u, 0].item()
            cont_w = anchor_y_world + offs[u, 1].item()
            if self.splat_mode == "round":
                ah = int(round(cont_h))
                aw = int(round(cont_w))
                self._splat(out, u_out_idx, patch[u_out_idx],
                            ah, aw, half_x, half_y, r_x, r_y, 1.0)
            else:  # bilinear
                lo_h = int(math.floor(cont_h))
                lo_w = int(math.floor(cont_w))
                fh = cont_h - lo_h
                fw = cont_w - lo_w
                # Four-corner weighted splat. Identity check: if fh=fw=0,
                # only (lo,lo) gets weight 1, recovering the round() case
                # (with anchor exactly at integer cont).
                for dh, dw, weight in (
                    (0, 0, (1.0 - fh) * (1.0 - fw)),
                    (0, 1, (1.0 - fh) * fw),
                    (1, 0, fh * (1.0 - fw)),
                    (1, 1, fh * fw),
                ):
                    if weight == 0.0:
                        continue
                    self._splat(out, u_out_idx, patch[u_out_idx],
                                lo_h + dh, lo_w + dw,
                                half_x, half_y, r_x, r_y, weight)

    # ------------------------------------------------------------------
    # Vectorized chunk render (FFT path with precomputed PSF spectrum)
    # ------------------------------------------------------------------
    def _render_chunk(
        self,
        out: torch.Tensor,
        gaussians: GaussianBatch,
        chunk_start: int,
        chunk_end: int,
        target_views: list[int],
    ):
        device = gaussians.positions.device
        dtype = gaussians.positions.dtype
        N_c = chunk_end - chunk_start
        half_xy = self.fixed_half_xy
        half_z = self.fixed_half_z
        Bx = 2 * half_xy + 1
        By = Bx
        Bz = 2 * half_z + 1
        r_x = self.r_x
        r_y = self.r_y

        mu = gaussians.positions[chunk_start:chunk_end]      # (N_c, 3)
        sigma = gaussians.sigmas[chunk_start:chunk_end]      # (N_c, 3)
        rho = gaussians.rhos[chunk_start:chunk_end]          # (N_c,)
        sigma_c = sigma.clamp(min=1e-6)                      # keep grad

        # Integer voxel anchor (no grad)
        with torch.no_grad():
            cx_idx = torch.round(mu[:, 0] - 0.5).long()       # (N_c,)
            cy_idx = torch.round(mu[:, 1] - 0.5).long()
            cz_idx = torch.round(mu[:, 2] - 0.5).long()

        # Local box global voxel indices
        bx_off = torch.arange(-half_xy, half_xy + 1, device=device, dtype=torch.long)
        by_off = bx_off
        bz_off = torch.arange(-half_z, half_z + 1, device=device, dtype=torch.long)
        bx_global = cx_idx[:, None] + bx_off[None, :]   # (N_c, Bx)
        by_global = cy_idx[:, None] + by_off[None, :]   # (N_c, By)
        bz_global = cz_idx[:, None] + bz_off[None, :]   # (N_c, Bz)

        bx_valid = (bx_global >= 0) & (bx_global < self.H)
        by_valid = (by_global >= 0) & (by_global < self.W)
        bz_valid = (bz_global >= 0) & (bz_global < self.Z)

        # World coords of voxel centers
        xs_world = bx_global.to(dtype) + 0.5            # (N_c, Bx)
        ys_world = by_global.to(dtype) + 0.5            # (N_c, By)
        zs_world = bz_global.to(dtype) + 0.5            # (N_c, Bz)

        view_idx = torch.as_tensor(target_views, device=device, dtype=torch.long)

        # Heavy freq-domain compute. Optionally checkpointed: backward
        # re-runs this function instead of saving the (N_c, Vt, Bz, r_x, ry2)
        # complex tensor (~hundreds of MB per chunk) — essential for any
        # nontrivial Gaussian count during training.
        if self.mode == "continuous_fourier":
            # Analytic Gaussian FT path: skip xs/ys/bx/by — μ_xy sub-voxel
            # info is encoded as phase shift in the closed-form FT.
            compute_args = (mu, sigma_c, rho, cx_idx, cy_idx,
                            zs_world, bz_valid, bz_global, view_idx)
            compute_fn = self._chunk_freq_compute_continuous
        else:
            compute_args = (mu, sigma_c, rho, xs_world, ys_world, zs_world,
                            bx_valid, by_valid, bz_valid, bz_global, view_idx)
            compute_fn = self._chunk_freq_compute
        if self.use_checkpoint and torch.is_grad_enabled() and rho.requires_grad:
            from torch.utils.checkpoint import checkpoint
            patches = checkpoint(
                compute_fn, *compute_args,
                use_reentrant=False,
            )
        else:
            patches = compute_fn(*compute_args)

        Vt = view_idx.numel()

        # Splat per Gaussian × view (Python loop kept; cheap relative to FFT).
        for g in range(N_c):
            cx = int(cx_idx[g].item())
            cy = int(cy_idx[g].item())
            cz = int(cz_idx[g].item())
            patch_g = patches[g]                                  # (Vt, r_x, r_y)

            if self.mode in ("raw_exact", "continuous_fourier"):
                # Both: integer anchor at (cx, cy). For raw_exact the raw PSF
                # carries the centroid shift; for continuous_fourier the
                # analytic FT's phase term carries the sub-voxel mu_xy shift
                # inside the patch.
                for u_out_idx in range(Vt):
                    self._splat(out, u_out_idx, patch_g[u_out_idx],
                                cx, cy, half_xy, half_xy, r_x, r_y, 1.0)
            else:  # centered_affine
                cz_lookup = max(0, min(self.Z - 1, cz))
                offs = self.centroid_offset[:, cz_lookup, :]      # (U, 2)
                anchor_x_world = cx + 0.5
                anchor_y_world = cy + 0.5
                for u_out_idx, u in enumerate(target_views):
                    cont_h = anchor_x_world + offs[u, 0].item()
                    cont_w = anchor_y_world + offs[u, 1].item()
                    if self.splat_mode == "round":
                        self._splat(out, u_out_idx, patch_g[u_out_idx],
                                    int(round(cont_h)), int(round(cont_w)),
                                    half_xy, half_xy, r_x, r_y, 1.0)
                    else:  # bilinear
                        lo_h = int(math.floor(cont_h))
                        lo_w = int(math.floor(cont_w))
                        fh = cont_h - lo_h
                        fw = cont_w - lo_w
                        for dh, dw, weight in (
                            (0, 0, (1.0 - fh) * (1.0 - fw)),
                            (0, 1, (1.0 - fh) * fw),
                            (1, 0, fh * (1.0 - fw)),
                            (1, 1, fh * fw),
                        ):
                            if weight == 0.0:
                                continue
                            self._splat(out, u_out_idx, patch_g[u_out_idx],
                                        lo_h + dh, lo_w + dw,
                                        half_xy, half_xy, r_x, r_y, weight)

    # ------------------------------------------------------------------
    # Heavy freq-domain compute (checkpointable)
    # ------------------------------------------------------------------
    def _chunk_freq_compute(
        self,
        mu, sigma_c, rho,
        xs_world, ys_world, zs_world,
        bx_valid, by_valid, bz_valid,
        bz_global, view_idx,
    ):
        """Pure freq-domain path: 3D Gaussian eval → rfft2 → freq mul-sum
        over Bz → irfft2 → patches.

        All inputs are tensors (so torch.utils.checkpoint can save/restore
        them). Returns patches of shape (N_c, Vt, r_x, r_y).
        """
        dtype = mu.dtype
        N_c, Bz = bz_global.shape
        Bx = bx_valid.shape[1]
        By = by_valid.shape[1]
        r_x = self.r_x
        r_y = self.r_y

        zz = zs_world[:, :, None, None]
        xx = xs_world[:, None, :, None]
        yy = ys_world[:, None, None, :]
        mu_x = mu[:, 0][:, None, None, None]
        mu_y = mu[:, 1][:, None, None, None]
        mu_z = mu[:, 2][:, None, None, None]
        sx = sigma_c[:, 0][:, None, None, None]
        sy = sigma_c[:, 1][:, None, None, None]
        sz = sigma_c[:, 2][:, None, None, None]

        v = rho[:, None, None, None] * torch.exp(
            -0.5 * (
                ((zz - mu_z) / sz) ** 2
                + ((xx - mu_x) / sx) ** 2
                + ((yy - mu_y) / sy) ** 2
            )
        )

        valid_3d = (
            bz_valid[:, :, None, None]
            & bx_valid[:, None, :, None]
            & by_valid[:, None, None, :]
        )
        v = v * valid_3d.to(dtype)

        v_pad = v.new_zeros(N_c, Bz, r_x, r_y)
        v_pad[:, :, :Bx, :By] = v
        v_freq = torch.fft.rfft2(v_pad)

        Vt = view_idx.numel()
        psf_freq_views = self.psf_freq.index_select(0, view_idx)
        ry2 = psf_freq_views.shape[-1]
        bz_global_clamped = bz_global.clamp(0, self.Z - 1)
        bz_idx = bz_global_clamped[:, None, :, None, None].expand(
            N_c, Vt, Bz, r_x, ry2)
        psf_freq_exp = psf_freq_views.unsqueeze(0).expand(
            N_c, Vt, self.Z, r_x, ry2)
        psf_g = torch.take_along_dim(psf_freq_exp, bz_idx, dim=2)

        mask_bz = bz_valid.to(v_freq.dtype)
        v_freq_b = v_freq.unsqueeze(1) * mask_bz[:, None, :, None, None]
        out_freq = (v_freq_b * psf_g).sum(dim=2)

        patches = torch.fft.irfft2(out_freq, s=(r_x, r_y))
        if self.z_norm == "mean":
            patches = patches / float(self.Z)
        return patches

    # ------------------------------------------------------------------
    # Continuous-Fourier freq-domain compute (Phase 1: analytic xy, nearest-z)
    # ------------------------------------------------------------------
    def _chunk_freq_compute_continuous(
        self,
        mu, sigma_c, rho,
        cx_idx, cy_idx,
        zs_world, bz_valid, bz_global, view_idx,
    ):
        """Analytic 2D-Gaussian FT path. Skips voxel-grid xy sampling.

        For each Gaussian, builds the continuous 2D Gaussian Fourier transform
        directly on the (r_x, r_y) DFT grid that matches psf_freq:

            G_hat(fx, fy) = ρ · 2π σx σy
                          · exp(-2π² (σx² fx² + σy² fy²))      [envelope]
                          · exp(-2πi (fx · mu_local_x
                                    + fy · mu_local_y))         [phase]

        where mu_local_* is the Gaussian center expressed in the local box
        coordinate system (same as the implicit coord used by rfft2 of v_pad):
            mu_local_x = mu_x - (cx_idx - half_xy + 0.5)
                       = (mu_x - cx_idx - 0.5) + half_xy

        When mu lies exactly on an integer voxel center, mu_local equals
        half_xy and the result reduces (modulo Gaussian truncation error) to
        the discretely-sampled path. When mu has a sub-voxel offset δ, the
        phase ramp exp(-2πi · f · δ) materializes that offset in the patch
        — fully smooth and analytically differentiable in mu_xy.

        Phase 1 of this mode keeps z handling identical to raw_exact: nearest
        PSF z slice + Gaussian z-density sampled at integer voxel centers.
        Sub-voxel mu_z grad flows through the density formula but is
        quantized at the box level. (Phase 2 will replace with erf-based
        slice integral.)
        """
        device = mu.device
        dtype = mu.dtype
        N_c, Bz = bz_global.shape
        r_x = self.r_x
        r_y = self.r_y
        half_xy = self.fixed_half_xy

        # --- Local-box mu position ------------------------------------------
        # mu_local = (mu_world − (cx_idx + 0.5)) + half_xy
        # When mu_world = cx_idx + 0.5 (integer voxel center), mu_local = half_xy.
        mu_local_x = (mu[:, 0] - cx_idx.to(dtype) - 0.5) + float(half_xy)
        mu_local_y = (mu[:, 1] - cy_idx.to(dtype) - 0.5) + float(half_xy)

        # --- Frequency grids matching psf_freq layout -----------------------
        # rfft2 over (r_x, r_y): full bins for the inner H-axis (fx via
        # fftfreq) and half bins for the outer W-axis (fy via rfftfreq).
        # Units: cycles per voxel.
        fx = torch.fft.fftfreq(r_x, device=device, dtype=dtype)        # (r_x,)
        fy = torch.fft.rfftfreq(r_y, device=device, dtype=dtype)       # (r_y//2+1,)

        sx = sigma_c[:, 0][:, None, None]    # (N_c, 1, 1)
        sy = sigma_c[:, 1][:, None, None]
        fx_g = fx[None, :, None]             # (1, r_x, 1)
        fy_g = fy[None, None, :]             # (1, 1, r_y//2+1)

        # --- Envelope (real, magnitude of continuous Gaussian FT) -----------
        # 2π · σx · σy · exp(-2π² (σx² fx² + σy² fy²))
        two_pi = 2.0 * math.pi
        two_pi_sq = 2.0 * math.pi * math.pi
        envelope = (two_pi * sx * sy) * torch.exp(
            -two_pi_sq * (sx * sx * fx_g * fx_g + sy * sy * fy_g * fy_g)
        )                                   # (N_c, r_x, r_y//2+1), real

        # --- Phase ramp encoding sub-voxel mu position ----------------------
        # exp(-2πi (fx · mu_local_x + fy · mu_local_y))
        phase_arg = (
            fx_g * mu_local_x[:, None, None]
            + fy_g * mu_local_y[:, None, None]
        )                                   # (N_c, r_x, r_y//2+1)
        # Build complex via real cos/sin (autograd-friendly; equivalent to
        # torch.complex but lets us multiply with the real envelope first to
        # cut one complex op).
        amp = rho[:, None, None] * envelope             # real, (N_c, r_x, ry2)
        cos_p = torch.cos(-two_pi * phase_arg)
        sin_p = torch.sin(-two_pi * phase_arg)
        v_freq = torch.complex(amp * cos_p, amp * sin_p)  # complex, (N_c, r_x, ry2)

        # --- z-direction weighting (Phase 1: nearest-z + sampled density) ---
        # Gaussian z density evaluated at integer voxel z centers in the
        # local box. Identical to what _chunk_freq_compute samples on the
        # z-axis; the difference vs that path is only in xy.
        sz = sigma_c[:, 2][:, None]                                    # (N_c, 1)
        mu_z = mu[:, 2][:, None]                                       # (N_c, 1)
        z_density = torch.exp(-0.5 * ((zs_world - mu_z) / sz) ** 2)    # (N_c, Bz)
        z_density = z_density * bz_valid.to(dtype)

        # --- Gather & weight PSF spectra over local z -----------------------
        Vt = view_idx.numel()
        psf_freq_views = self.psf_freq.index_select(0, view_idx)        # (Vt, Z, r_x, ry2)
        ry2 = psf_freq_views.shape[-1]
        bz_clamped = bz_global.clamp(0, self.Z - 1)
        bz_idx = bz_clamped[:, None, :, None, None].expand(
            N_c, Vt, Bz, r_x, ry2)
        psf_freq_exp = psf_freq_views.unsqueeze(0).expand(
            N_c, Vt, self.Z, r_x, ry2)
        psf_g = torch.take_along_dim(psf_freq_exp, bz_idx, dim=2)       # (N_c, Vt, Bz, r_x, ry2)

        z_w = z_density[:, None, :, None, None].to(psf_g.dtype)         # complex weights (imag=0)
        psf_weighted = (psf_g * z_w).sum(dim=2)                         # (N_c, Vt, r_x, ry2)

        # --- Final spectrum, irfft → patches --------------------------------
        out_freq = v_freq.unsqueeze(1) * psf_weighted                   # (N_c, Vt, r_x, ry2)
        patches = torch.fft.irfft2(out_freq, s=(r_x, r_y))              # (N_c, Vt, r_x, r_y)
        if self.z_norm == "mean":
            patches = patches / float(self.Z)
        return patches

    # ------------------------------------------------------------------
    # Splat helper
    # ------------------------------------------------------------------
    def _splat(
        self,
        out: torch.Tensor,
        u_out_idx: int,
        patch_uv: torch.Tensor,    # (r_x, r_y)
        anchor_h: int,
        anchor_w: int,
        half_x: int,
        half_y: int,
        r_x: int,
        r_y: int,
        weight: float,
    ):
        offset_x = anchor_h - half_x - (self.ph - 1) // 2
        offset_y = anchor_w - half_y - (self.pw - 1) // 2
        i0_img = max(0, offset_x);   i1_img = min(self.H, offset_x + r_x)
        j0_img = max(0, offset_y);   j1_img = min(self.W, offset_y + r_y)
        if i1_img <= i0_img or j1_img <= j0_img:
            return
        i0_p = i0_img - offset_x;    i1_p = i1_img - offset_x
        j0_p = j0_img - offset_y;    j1_p = j1_img - offset_y
        if weight == 1.0:
            out[u_out_idx, i0_img:i1_img, j0_img:j1_img] += \
                patch_uv[i0_p:i1_p, j0_p:j1_p]
        else:
            out[u_out_idx, i0_img:i1_img, j0_img:j1_img] += \
                weight * patch_uv[i0_p:i1_p, j0_p:j1_p]


# ---------------------------------------------------------------------------
# Reference: build full discrete-eval volume from Gaussians
# ---------------------------------------------------------------------------

def gaussians_to_full_volume(
    gaussians: GaussianBatch,
    volume_shape: tuple[int, int, int],
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """Evaluate Gaussian sum at every integer voxel center → (D, H, W) volume.

    This is the reference voxelizer for sanity-check. It uses the SAME analytic
    Gaussian evaluation as `_render_one`, so the only thing being tested is
    "local conv + splat" vs "global volume + global conv".
    """
    D, H, W = volume_shape
    if device is None:
        device = gaussians.positions.device
    xs = torch.arange(H, device=device, dtype=dtype) + 0.5  # H axis = x
    ys = torch.arange(W, device=device, dtype=dtype) + 0.5  # W axis = y
    zs = torch.arange(D, device=device, dtype=dtype) + 0.5  # D axis = z

    vol = torch.zeros(D, H, W, device=device, dtype=dtype)
    N = gaussians.positions.shape[0]
    for i in range(N):
        mu = gaussians.positions[i]
        sigma = gaussians.sigmas[i]
        rho = gaussians.rhos[i]
        sx = max(sigma[0].item(), 1e-6)
        sy = max(sigma[1].item(), 1e-6)
        sz = max(sigma[2].item(), 1e-6)
        zz = zs.view(D, 1, 1)
        xx = xs.view(1, H, 1)
        yy = ys.view(1, 1, W)
        vol += rho * torch.exp(
            -0.5 * (
                ((zz - mu[2]) / sz) ** 2
                + ((xx - mu[0]) / sx) ** 2
                + ((yy - mu[1]) / sy) ** 2
            )
        )
    return vol
