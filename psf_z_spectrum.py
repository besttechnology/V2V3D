"""PSF z-direction energy distribution and interpolation-order diagnostic.

Answers two questions to decide Phase 2 interpolation order:
  1. Is PSF bandlimited along z?  (decides whether sinc is physically valid)
  2. How much does cubic beat linear at reconstructing held-out z slices?
     (decides whether the complexity step from linear → cubic pays off)

Outputs:
  - psf_z_spectrum.png  per-view + mean power spectrum and cumulative energy
  - Console verdict      with thresholded recommendation
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import load_psfs


def spectrum_analysis(psf_np):
    """psf_np: (U, Z, H, W) -> (per_view_spec, per_view_cum, freqs).

    For each view, take 1D rFFT along z at every (h, w), sum |F|^2 over
    spatial dims, normalize so total = 1. Cumulative fraction is the
    integral of the normalized spectrum up to each cutoff.
    """
    U, Z, _, _ = psf_np.shape
    Nfreq = Z // 2 + 1
    freqs = np.fft.rfftfreq(Z)  # cycles per voxel, [0, 0.5]

    per_view_spec = np.zeros((U, Nfreq))
    per_view_cum = np.zeros((U, Nfreq))

    for u in range(U):
        spec = np.fft.rfft(psf_np[u], axis=0)              # (Nfreq, H, W)
        power = (spec * spec.conj()).real                  # (Nfreq, H, W)
        view_power = power.sum(axis=(1, 2))                # (Nfreq,)
        view_power = view_power / (view_power.sum() + 1e-30)
        per_view_spec[u] = view_power
        per_view_cum[u] = np.cumsum(view_power)

    return per_view_spec, per_view_cum, freqs


def _linear_interp_z(psf_kept, z_kept, z_test):
    """numpy-only linear interpolation along z axis."""
    U, Zk, H, W = psf_kept.shape
    out = np.zeros((U, len(z_test), H, W), dtype=psf_kept.dtype)
    for i, z in enumerate(z_test):
        # find interval
        j = np.searchsorted(z_kept, z) - 1
        j = max(0, min(Zk - 2, j))
        t = (z - z_kept[j]) / (z_kept[j + 1] - z_kept[j])
        out[:, i] = (1 - t) * psf_kept[:, j] + t * psf_kept[:, j + 1]
    return out


def decimation_test(psf_np):
    """Downsample PSF in z by 2, interpolate back, report rel L2 error.

    z_kept = [0, 2, ..., Z-1 if even else Z-2]
    z_test = held-out odd indices, endpoints excluded (no extrapolation).
    Always reports linear; cubic only if scipy is available.
    """
    U, Z, H, W = psf_np.shape
    z_kept = np.arange(0, Z, 2)
    z_test = np.arange(1, Z - 1, 2)

    psf_kept = psf_np[:, z_kept, :, :]
    psf_true = psf_np[:, z_test, :, :]
    norm = np.sqrt((psf_true ** 2).sum()) + 1e-30

    results = {}
    # Linear: numpy-only
    psf_lin = _linear_interp_z(psf_kept, z_kept, z_test)
    results['linear'] = float(np.sqrt(((psf_lin - psf_true) ** 2).sum()) / norm)

    # Cubic: scipy-only
    try:
        from scipy.interpolate import interp1d
        f = interp1d(z_kept, psf_kept, axis=1, kind='cubic')
        psf_cub = f(z_test)
        results['cubic'] = float(np.sqrt(((psf_cub - psf_true) ** 2).sum()) / norm)
    except ImportError:
        print('  [skip] scipy not available, cubic comparison disabled')
        results['cubic'] = None
    return results


def print_verdict(per_view_spec, per_view_cum, freqs, decim, Z):
    avg = per_view_spec.mean(axis=0)
    avg_cum = per_view_cum.mean(axis=0)
    Nfreq = len(freqs)

    print(f"\n=== PSF z-direction analysis (Z={Z} slices) ===\n")

    print("Average normalized power at key z-frequencies:")
    for f_bin, label in [(1, 'lowest non-DC'),
                         (Nfreq // 4, 'quarter-Nyquist'),
                         (Nfreq // 2, 'half-Nyquist'),
                         (Nfreq - 1, 'Nyquist')]:
        print(f"  f = {freqs[f_bin]:.3f} cyc/vox ({label:>17}): "
              f"power = {avg[f_bin]:.3e}")

    print("\nCumulative energy fraction (mean across views):")
    for cutoff in [0.1, 0.2, 0.3, 0.4, 0.5]:
        idx = min(Nfreq - 1, int(round(cutoff * 2 * (Nfreq - 1))))
        print(f"  energy below {cutoff:.1f} cyc/vox: {avg_cum[idx] * 100:5.1f}%")

    print("\nDecimation test (downsample z by 2 -> interpolate -> rel L2):")
    for kind, err in decim.items():
        if err is None:
            print(f"  {kind:>8}: (skipped)")
        else:
            print(f"  {kind:>8}: {err:.4f}  ({err * 100:.2f}%)")

    # ------------------- verdict -------------------
    print("\n=== Verdict ===")
    ratio_nyq = avg[-1] / (avg[1] + 1e-30)
    if ratio_nyq < 0.01:
        print(f"STRONGLY BANDLIMITED  (Nyquist/lowest = {ratio_nyq:.2e})")
        print("  -> Section 3.2 sinc interpolation is physically well-justified.")
    elif ratio_nyq < 0.1:
        print(f"MODERATELY BANDLIMITED  (Nyquist/lowest = {ratio_nyq:.2e})")
        print("  -> Section 3.1 cubic spline is robust; sinc plausible but watch ringing.")
    else:
        print(f"NOT BANDLIMITED  (Nyquist/lowest = {ratio_nyq:.2e})")
        print("  -> dz=1 likely undersampled. sinc would alias.")
        print("  -> Stick with linear/cubic, or pursue Section 3.6 Fresnel parametric.")

    print()
    if decim['cubic'] is None:
        print("Cubic comparison skipped (no scipy). Re-run with scipy installed "
              "to decide linear vs cubic.")
    else:
        ratio_lc = decim['cubic'] / (decim['linear'] + 1e-30)
        if ratio_lc < 0.5:
            print(f"Cubic significantly beats linear  (cubic/linear = {ratio_lc:.2f})")
            print("  -> cubic spline worth the implementation cost over linear.")
        else:
            print(f"Cubic ~ linear  (cubic/linear = {ratio_lc:.2f})")
            print("  -> linear (Phase 2 baseline) may already be sufficient.")


def plot(per_view_spec, per_view_cum, freqs, out_path):
    U = per_view_spec.shape[0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for u in range(U):
        ax.semilogy(freqs[1:], per_view_spec[u, 1:], alpha=0.3, color='C0')
    ax.semilogy(freqs[1:], per_view_spec.mean(axis=0)[1:],
                'k-', lw=2, label='mean over views')
    ax.set_xlabel('z-frequency (cycles per voxel)')
    ax.set_ylabel('normalized power (log)')
    ax.set_title('PSF z-power-spectrum')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

    ax = axes[1]
    for u in range(U):
        ax.plot(freqs, per_view_cum[u], alpha=0.3, color='C1')
    ax.plot(freqs, per_view_cum.mean(axis=0), 'k-', lw=2, label='mean over views')
    for level, ls in [(0.95, ':'), (0.99, '--')]:
        ax.axhline(level, color='gray', ls=ls, alpha=0.5,
                   label=f'{int(level * 100)}%')
    ax.set_xlabel('z-frequency cutoff (cycles per voxel)')
    ax.set_ylabel('cumulative energy fraction')
    ax.set_title('Cumulative z-energy vs cutoff')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--psf_dir', default='PSF/PSF_zoom2_39dz1_N13')
    p.add_argument('--Nnum', type=int, default=13)
    p.add_argument('--out', default='psf_z_spectrum.png')
    args = p.parse_args()

    psf = load_psfs(args.psf_dir, args.Nnum)
    psf_np = psf.numpy()
    print(f"\nPSF tensor shape: {psf_np.shape}, dtype: {psf_np.dtype}")
    print(f"PSF range: [{psf_np.min():.3e}, {psf_np.max():.3e}]")

    spec, cum, freqs = spectrum_analysis(psf_np)
    decim = decimation_test(psf_np)
    print_verdict(spec, cum, freqs, decim, psf_np.shape[1])
    plot(spec, cum, freqs, args.out)
    print(f"\nSaved plot: {args.out}")


if __name__ == '__main__':
    main()
