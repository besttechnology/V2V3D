"""离线 PSF affine 拟合（v3 阶段 A）。

按 v3 §4.2 推荐路线，对每个 view 拟合:
    A_u = [[1, 0, d_x^(u)],
           [0, 1, d_y^(u)]]      # 前 2x2 固定为 I
    b_u, C_u                      # 2D bias + 2x2 covariance
    a_u(z)                        # depth-dependent energy attenuation

输入: PSF/<dir>/psf_{1..U}.tif，每个文件 (Z, H, W) float32。
输出: psf_params.pt 含 A, b, C, a, z_grid, image_center。
      psf_fit_report.md 含每 view 拟合误差与诊断图引用。
      psf_fit_vis/ 含可视化对比图。

判定门 (v3 §4.4):
    Err < 3%  优秀
    Err < 5%  合格
    Err < 10% 需要 mixture 或 depth-dependent C
    Err > 10% 不应直接使用 Gaussian channel — 改 hybrid renderer

与 v3 §6.1 的小偏离: a 存成 [U, M, Z]，因为实测 LFM PSF 轴向能量有显著
衰减 (~4x)，丢掉这层会让远端 z 的强度系统性偏弱。M=1 时 [U, 1, Z]，
渲染器按 Gaussian 的 μ_z 插值 a。
"""
import argparse
import os

import numpy as np
import tifffile as tf
import torch


def load_psfs(psf_dir, U):
    """读 PSF/<dir>/psf_{1..U}.tif，返回 (U, Z, H, W) torch.float32。"""
    psfs = []
    postfix = os.listdir(psf_dir)[0].split('.')[-1]
    for u in range(1, U + 1):
        p = tf.imread(os.path.join(psf_dir, f'psf_{u}.{postfix}')).astype(np.float32)
        psfs.append(p)
    return torch.from_numpy(np.stack(psfs, 0))


def compute_moments(psf_uz, image_center):
    """对单张 (H, W) PSF 计算能量、相对中心的 centroid、2x2 covariance。

    centroid 已减去 image_center。
    """
    H, W = psf_uz.shape
    cx0, cy0 = image_center
    ys = torch.arange(H, dtype=psf_uz.dtype) - cy0
    xs = torch.arange(W, dtype=psf_uz.dtype) - cx0
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')

    e = psf_uz.sum()
    if e <= 0:
        return torch.tensor(0.0), torch.zeros(2), torch.eye(2) * 1e-6

    p = psf_uz / e
    cx = (xx * p).sum()
    cy = (yy * p).sum()
    dx = xx - cx
    dy = yy - cy
    Cxx = (dx * dx * p).sum()
    Cyy = (dy * dy * p).sum()
    Cxy = (dx * dy * p).sum()
    cov = torch.tensor([[Cxx, Cxy], [Cxy, Cyy]])
    centroid = torch.tensor([cx, cy])
    return e, centroid, cov


def fit_view(centroids_z, covs_z, energies_z, z_grid):
    """对单个 view 拟合 (d_u, b_u, C_u, a_u(z))。

    centroids_z: (Z, 2) — 已减去 image_center
    covs_z:      (Z, 2, 2)
    energies_z:  (Z,)
    z_grid:      (Z,) — 已中心化的 z 坐标
    返回 dict: d (2,), b (2,), C (2, 2), a_per_z (Z,), residual_per_z (Z, 2)
    """
    Z = z_grid.shape[0]

    # least-squares: c_{u,k} = z_k · d + b
    # design matrix [Z, 2]: columns [z_k, 1]
    # 用能量做权重（远端 z 噪声更大）
    w = energies_z.clamp_min(1e-12)
    w = w / w.sum()
    A_design = torch.stack([z_grid, torch.ones_like(z_grid)], dim=1)    # (Z, 2)

    # weighted normal equations: (A^T W A) θ = A^T W c, per axis
    WA = A_design * w.unsqueeze(1)
    AtWA = A_design.T @ WA                                              # (2, 2)
    AtWc = WA.T @ centroids_z                                           # (2, 2): rows = [d, b], cols = [x, y]
    theta = torch.linalg.solve(AtWA, AtWc)                              # (2, 2)
    d = theta[0]                                                        # (2,) [dx, dy]
    b = theta[1]                                                        # (2,) [bx, by]

    # C_u: weighted mean of cov across z
    C = (covs_z * w.view(-1, 1, 1)).sum(0)                              # (2, 2)
    # 强制对称 + 微小 jitter 防退化
    C = 0.5 * (C + C.T) + torch.eye(2) * 1e-6

    # 拟合残差
    pred_centroids = z_grid.unsqueeze(1) * d.unsqueeze(0) + b.unsqueeze(0)  # (Z, 2)
    residual = centroids_z - pred_centroids                             # (Z, 2)

    return {
        'd': d, 'b': b, 'C': C,
        'a_per_z': energies_z.clone(),
        'residual': residual,
    }


def reconstruct_psf(d, b, C, a, z, image_center, H, W):
    """按拟合参数渲染单张 (H, W) PSF：N_2D(z·d + b + image_center, C) · a。"""
    cx0, cy0 = image_center
    ys = torch.arange(H, dtype=torch.float32) - cy0
    xs = torch.arange(W, dtype=torch.float32) - cx0
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')

    mu = z * d + b                                                      # (2,)
    diff = torch.stack([xx - mu[0], yy - mu[1]], dim=-1)                # (H, W, 2)

    Cinv = torch.linalg.inv(C)
    detC = torch.det(C).clamp_min(1e-12)
    norm = 1.0 / (2 * np.pi * torch.sqrt(detC))

    # quadratic form
    q = (diff @ Cinv * diff).sum(-1)                                    # (H, W)
    g = norm * torch.exp(-0.5 * q)
    return a * g


def relative_l2(a, b):
    return ((a - b) ** 2).sum().sqrt() / (b ** 2).sum().sqrt().clamp_min(1e-12)


def fit_view_mixture(psfs_u, z_grid, image_center, M):
    """对单个 view 用残差迭代拟 M 个 affine Gaussian channel components。

    psfs_u: (Z, H, W)
    返回: list of M dicts (每个含 d, b, C, a_per_z)
    """
    Z, H, W = psfs_u.shape
    residual = psfs_u.clone()
    components = []
    for m in range(M):
        # 在残差上提矩（clip 负值，moments 才有意义）
        centroids = torch.zeros(Z, 2)
        covs = torch.zeros(Z, 2, 2)
        energies = torch.zeros(Z)
        for k in range(Z):
            r_k = residual[k].clamp_min(0)
            e, c, cov = compute_moments(r_k, image_center)
            energies[k] = e
            centroids[k] = c
            covs[k] = cov

        # 能量太小说明已无可拟成分，组件后续置零（不再继续 peel）
        if energies.sum() < 1e-20:
            for _ in range(M - m):
                components.append({
                    'd': torch.zeros(2), 'b': torch.zeros(2),
                    'C': torch.eye(2) * 1e-6,
                    'a_per_z': torch.zeros(Z),
                    'residual': torch.zeros(Z, 2),
                })
            break

        fit = fit_view(centroids, covs, energies, z_grid)
        components.append(fit)

        # 从残差减去这个 component
        for k in range(Z):
            g = reconstruct_psf(
                fit['d'], fit['b'], fit['C'], fit['a_per_z'][k],
                z_grid[k], image_center, H, W,
            )
            residual[k] = residual[k] - g

    return components


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--psf_dir', type=str,
                        default='PSF/PSF_zoom2_39dz1_N13')
    parser.add_argument('--Nnum', type=int, default=13)
    parser.add_argument('--M', type=int, default=1,
                        help='Mixture components per view (v3 §4.3)')
    parser.add_argument('--out_params', type=str, default='psf_params.pt')
    parser.add_argument('--out_report', type=str, default='psf_fit_report.md')
    parser.add_argument('--out_vis_dir', type=str, default='psf_fit_vis')
    parser.add_argument('--save_vis', action='store_true',
                        help='保存每 view 的 real vs fit 对比图（matplotlib）')
    args = parser.parse_args()

    print(f'[psf_fit] loading PSFs from {args.psf_dir} (U={args.Nnum}, M={args.M})')
    psfs = load_psfs(args.psf_dir, args.Nnum)
    U, Z, H, W = psfs.shape
    print(f'[psf_fit] PSF tensor shape: {tuple(psfs.shape)}')

    image_center = (W / 2.0 - 0.5, H / 2.0 - 0.5)                       # (cx, cy) — 像素中心约定
    z_grid = torch.arange(Z, dtype=torch.float32) - (Z - 1) / 2.0        # 中心化
    M = args.M

    # Step 1+2: 每 view 用残差迭代拟 M 个 component
    A_out = torch.zeros(U, M, 2, 3)
    b_out = torch.zeros(U, M, 2)
    C_out = torch.zeros(U, M, 2, 2)
    a_out = torch.zeros(U, M, Z)
    all_components = []                                                  # [U][M] dicts
    for u in range(U):
        comps = fit_view_mixture(psfs[u], z_grid, image_center, M)
        all_components.append(comps)
        for m, fit in enumerate(comps):
            A_out[u, m, 0, 0] = 1.0
            A_out[u, m, 1, 1] = 1.0
            A_out[u, m, 0, 2] = fit['d'][0]
            A_out[u, m, 1, 2] = fit['d'][1]
            b_out[u, m] = fit['b']
            C_out[u, m] = fit['C']
            a_out[u, m] = fit['a_per_z']

    # Step 3: 重建 PSF + 计算每 (u, z) 误差
    psf_fit_full = torch.zeros_like(psfs)
    err_per_uz = torch.zeros(U, Z)
    err_per_u = torch.zeros(U)
    for u in range(U):
        for k in range(Z):
            recon = torch.zeros(H, W)
            for m in range(M):
                fit = all_components[u][m]
                recon = recon + reconstruct_psf(
                    fit['d'], fit['b'], fit['C'], fit['a_per_z'][k],
                    z_grid[k], image_center, H, W,
                )
            psf_fit_full[u, k] = recon
            err_per_uz[u, k] = relative_l2(psf_fit_full[u, k], psfs[u, k])
        err_per_u[u] = relative_l2(psf_fit_full[u], psfs[u])
    overall_err = relative_l2(psf_fit_full, psfs)

    # Step 4: 保存参数
    out = {
        'A': A_out, 'b': b_out, 'C': C_out, 'a': a_out,
        'z_grid': z_grid,
        'image_center': torch.tensor(image_center),
        'psf_shape': torch.tensor([U, Z, H, W]),
        'M': M,
        'err_per_u': err_per_u,
        'err_per_uz': err_per_uz,
        'overall_err': overall_err,
    }
    torch.save(out, args.out_params)
    print(f'[psf_fit] saved params -> {args.out_params}')

    # Step 5: 报告
    def grade(e):
        if e < 0.03:
            return '优秀'
        if e < 0.05:
            return '合格'
        if e < 0.10:
            return '需要 mixture / depth-dep C'
        return '不应直接用 Gaussian channel — 改 hybrid'

    lines = []
    lines.append('# PSF affine 拟合报告\n')
    lines.append(f'- PSF dir: `{args.psf_dir}`')
    lines.append(f'- Tensor shape: U={U}, Z={Z}, H={H}, W={W}')
    lines.append(f'- z_grid: 中心化于 0，范围 [{z_grid[0].item():.1f}, {z_grid[-1].item():.1f}]')
    lines.append(f'- image_center: {image_center}')
    lines.append(f'- 模型: A_u 前 2x2 = I, 残差迭代拟 M={M} 个 component\n')
    lines.append(f'## 总体误差: **{overall_err.item() * 100:.2f}%** — {grade(overall_err.item())}\n')

    lines.append('## 每 view 拟合 (按 component 分行)')
    lines.append('| view | m | d_x (px/z) | d_y (px/z) | b_x (px) | b_y (px) | sqrt(C_xx) | sqrt(C_yy) | corr | a(z) max | a 占总能量 |')
    lines.append('|------|---|-----------|-----------|---------|---------|-----------|-----------|------|---------|-----------|')
    for u in range(U):
        view_total_a = sum(all_components[u][m]['a_per_z'].sum() for m in range(M)).clamp_min(1e-20)
        for m in range(M):
            fit = all_components[u][m]
            d = fit['d']; b_ = fit['b']; C = fit['C']; a = fit['a_per_z']
            sxx = C[0, 0].clamp_min(0).sqrt().item()
            syy = C[1, 1].clamp_min(0).sqrt().item()
            corr = (C[0, 1] / (C[0, 0] * C[1, 1]).clamp_min(1e-12).sqrt()).item()
            a_share = (a.sum() / view_total_a).item()
            lines.append(
                f'| {u} | {m} | {d[0].item():+.3f} | {d[1].item():+.3f} | '
                f'{b_[0].item():+.3f} | {b_[1].item():+.3f} | '
                f'{sxx:.2f} | {syy:.2f} | {corr:+.2f} | '
                f'{a.max().item():.2e} | {a_share * 100:.1f}% |'
            )
        lines.append(f'| {u} | **all** | | | | | | | | | Err = **{err_per_u[u].item() * 100:.2f}%** |')

    # 主分量轴向能量衰减诊断（仅 m=0）
    a_main = a_out[:, 0, :]                                              # (U, Z)
    a_var_ratio = a_main.max(dim=-1).values / a_main.min(dim=-1).values.clamp_min(1e-12)
    lines.append('\n## 主分量 (m=0) 轴向能量衰减')
    lines.append('每 view 的 a(z_max)/a(z_min) 比值（>2 时建议保留 depth-dependent a）:\n')
    for u in range(U):
        lines.append(f'- view {u}: {a_var_ratio[u].item():.2f}x')

    # 残差最大的几个 (u, z)
    flat_err = err_per_uz.flatten()
    topk = torch.topk(flat_err, k=min(10, flat_err.numel()))
    lines.append('\n## 误差最大的 10 个 (u, z) 切片')
    for val, idx in zip(topk.values, topk.indices):
        u_ = (idx // Z).item()
        k_ = (idx % Z).item()
        lines.append(f'- (u={u_}, z={z_grid[k_].item():+.1f}): Err = {val.item() * 100:.2f}%')

    lines.append('\n## 判定门 (v3 §4.4)')
    lines.append('- Err < 3% : 优秀，直接进入训练')
    lines.append('- Err < 5% : 合格，sanity check 验证')
    lines.append('- Err < 10%: 需要增加 mixture 或 depth-dependent C_u')
    lines.append('- Err > 10%: 不应直接使用 Gaussian channel — 改 hybrid renderer')

    with open(args.out_report, 'w') as f:
        f.write('\n'.join(lines))
    print(f'[psf_fit] saved report -> {args.out_report}')

    # Step 6: 可视化（可选）
    if args.save_vis:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        os.makedirs(args.out_vis_dir, exist_ok=True)
        z_show = [0, Z // 4, Z // 2, 3 * Z // 4, Z - 1]
        for u in range(U):
            fig, axes = plt.subplots(3, len(z_show), figsize=(3 * len(z_show), 9))
            for col, k in enumerate(z_show):
                real = psfs[u, k].numpy()
                fit = psf_fit_full[u, k].numpy()
                vmax = max(real.max(), fit.max())
                axes[0, col].imshow(real, vmin=0, vmax=vmax); axes[0, col].set_title(f'real z={z_grid[k].item():+.0f}')
                axes[1, col].imshow(fit, vmin=0, vmax=vmax); axes[1, col].set_title(f'fit  err={err_per_uz[u, k].item()*100:.1f}%')
                axes[2, col].imshow(real - fit, cmap='RdBu_r'); axes[2, col].set_title('real - fit')
                for ax in axes[:, col]:
                    ax.axis('off')
            fig.suptitle(f'view {u}  Err_view = {err_per_u[u].item()*100:.2f}%', fontsize=14)
            fig.tight_layout()
            fig.savefig(os.path.join(args.out_vis_dir, f'view_{u:02d}.png'), dpi=80)
            plt.close(fig)
        print(f'[psf_fit] saved vis -> {args.out_vis_dir}/')


if __name__ == '__main__':
    main()
