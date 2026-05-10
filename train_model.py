import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import os
import configargparse
from utils import test2b, adjust_lr, generate_fps, save_ckpt, normal
from loss import fftloss, deCrosstalk_loss, posloss, zloss, gauss_total_reg
from dataset import SyntheticData
from model import V2V3D, V2V3D_Gauss

def parse_args():
    parser = configargparse.ArgumentParser(
        description='V2V3D Training',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
        default_config_files=['Config/train.yaml'] 
    )
    parser.add('-c', '--config', required=False, is_config_file=True)
    
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--psf_dir', type=str)
    parser.add_argument('--lf_dir', type=str)
    
    parser.add_argument('--Nnum', type=int, default=13, help='Num of all views')
    parser.add_argument('--input_size', type=int, default=256)
    parser.add_argument('--feat_ch', type=int, default=4)
    parser.add_argument('--dc_weight', type=float, default=0.1)
    parser.add_argument('--tv_weight', type=float, default=1e-3)
    
    parser.add_argument('--lr_init', type=float, default=8e-5)
    parser.add_argument('--lr_decay', type=float, default=0.3)
    parser.add_argument('--decay_init', type=int, default=60)
    parser.add_argument('--decay_every', type=int, default=20)
    
    parser.add_argument('--log', type=str, default='', help='specific log')

    # --- Gaussian Decoder 相关 ---
    parser.add_argument('--use_gaussian', action='store_true',
                        help='使用 V2V3D_Gauss（Gaussian Decoder + Intensity Voxelizer）')
    parser.add_argument('--gauss_scale_init', type=float, default=0.5)
    parser.add_argument('--gauss_s_min', type=float, default=0.1)
    parser.add_argument('--gauss_s_max', type=float, default=3.0)
    parser.add_argument('--gauss_lambda_scale', type=float, default=1e-2)
    parser.add_argument('--gauss_lambda_sparse', type=float, default=1e-3)
    parser.add_argument('--gauss_lambda_aniso', type=float, default=1e-4)
    parser.add_argument('--gauss_reg_warmup', type=int, default=2,
                        help='前几个 epoch 不施加 Gaussian 正则，只用 MSE')

    # --- Hybrid renderer 相关 (gauss → LFI 直接投影) ---
    parser.add_argument('--use_hybrid', action='store_true',
                        help='用 HybridRenderer 直接 gauss→LFI，跳过 voxelizer+generate_fps '
                             '路径。仅在 --use_gaussian 时生效。')
    parser.add_argument('--hybrid_rho_threshold', type=float, default=0.01,
                        help='只渲染 ρ > threshold 的 Gaussian（控制速度）')
    parser.add_argument('--hybrid_max_gaussians', type=int, default=20000,
                        help='单次 forward 最多渲染的 Gaussian 数（top-K by ρ）')
    parser.add_argument('--hybrid_chunk_size', type=int, default=16)
    parser.add_argument('--hybrid_half_xy', type=int, default=0,
                        help='覆盖 ceil(4*s_max) 计算的 half_xy；0 表示自动')
    parser.add_argument('--hybrid_half_z', type=int, default=0,
                        help='覆盖 ceil(4*s_max) 计算的 half_z；0 表示自动')
    parser.add_argument('--hybrid_no_checkpoint', action='store_true',
                        help='关闭 gradient checkpoint。默认开（forward 慢 ~1.5x，'
                             '但 backward 内存大幅下降，可处理更多 Gaussian）')
    parser.add_argument('--hybrid_rho_bias', type=float, default=None,
                        help='覆盖 init_gaussian_head_bias 默认 rho_bias=-5。Hybrid '
                             '路径建议设置为 0~-2（让初始 ρ ≈ 0.1~0.7，渲染输出量级'
                             '匹配 target，避免 bootstrap 困境）。None=不覆盖。')
    parser.add_argument('--hybrid_explore_frac', type=float, default=0.3,
                        help='Soft filter 中随机探索的比例（top-K 为主，剩下做随机采样）。'
                             '0=纯 top-K，1=纯随机。默认 0.3 让 dormant Gaussian 有机会被'
                             '渲染并拿到梯度。')
    parser.add_argument('--hybrid_soft_temp', type=float, default=0.001,
                        help='Soft mask 的温度。被渲染的 Gaussian 的实际 ρ = ρ * '
                             'sigmoid((ρ-threshold)/temp)。temp 越小越接近 hard 阈值。')
    parser.add_argument('--hybrid_smoke', type=int, default=0,
                        help='只跑前 N 个 iter 后退出（0=正常训练，>0=smoke 模式）')

    return parser.parse_args()


def _hybrid_filter_topk(gp, rho_threshold, max_gaussians):
    """Filter Gaussians: keep ρ > threshold, then top-K by ρ. Differentiable
    in the kept Gaussians' params (filtering itself is non-diff but stable
    when rho stays above the threshold)."""
    rho = gp['densities'].reshape(-1)            # (N,)
    pos = gp['positions']                         # (N, 3)
    sca = gp['scales']                            # (N, 3)
    keep = rho > rho_threshold
    if keep.sum() == 0:
        K = min(max_gaussians, rho.numel())
        idx = torch.topk(rho, K).indices
    else:
        idx = torch.nonzero(keep, as_tuple=True)[0]
        if idx.numel() > max_gaussians:
            rho_kept = rho[idx]
            top = torch.topk(rho_kept, max_gaussians).indices
            idx = idx[top]
    return pos[idx], sca[idx], rho[idx], idx.numel(), rho.numel()


def _hybrid_soft_filter(gp, rho_threshold, max_gaussians, explore_frac=0.3,
                        soft_temp=0.001):
    """Soft top-K filter for hybrid renderer.

    Selects max_gaussians Gaussians per iter:
      - (1-explore_frac) * max_gaussians  picked by top-ρ (the "active" ones)
      - explore_frac     * max_gaussians  sampled uniformly from the rest
        ("exploration" — gives dormant Gaussians a chance to receive
        gradient and wake up).

    All picked Gaussians' ρ is multiplied by sigmoid((ρ-threshold)/temp)
    so the threshold is a smooth gate (Gaussians with ρ ≪ threshold
    contribute ~0, but their ρ still gets gradient flow when picked).

    Returns (positions, scales, rho_effective, n_picked, n_total).
    """
    rho = gp['densities'].reshape(-1)
    pos = gp['positions']
    sca = gp['scales']
    N = rho.numel()
    K = min(max_gaussians, N)
    K_explore = int(K * explore_frac)
    K_active = K - K_explore

    # Hard top-K by ρ (no grad — selection is non-differentiable).
    rho_det = rho.detach()
    top_idx = torch.topk(rho_det, K_active).indices

    if K_explore > 0 and N > K_active:
        mask = torch.ones(N, dtype=torch.bool, device=rho.device)
        mask[top_idx] = False
        rest = torch.nonzero(mask, as_tuple=True)[0]
        n_rest = rest.numel()
        n_pick = min(K_explore, n_rest)
        if n_pick > 0:
            perm = torch.randperm(n_rest, device=rho.device)[:n_pick]
            explore_idx = rest[perm]
            idx = torch.cat([top_idx, explore_idx])
        else:
            idx = top_idx
    else:
        idx = top_idx

    rho_picked = rho[idx]                        # tensor with grad
    soft_mask = torch.sigmoid((rho_picked - rho_threshold) / soft_temp)
    rho_eff = rho_picked * soft_mask
    return pos[idx], sca[idx], rho_eff, idx.numel(), N

def train(args):
    device = torch.device('cuda', args.gpu_id)
    
    curtime = time.strftime('%m_%d_%H_%M', time.localtime(time.time()))
    projection_name = args.lf_dir.split('/')[-1]
    notes = '_V2V3D_F%d_'%(args.feat_ch)+args.log+'_' if args.log != '' else '_V2V3D_F%d_'%(args.feat_ch)
    project_name = curtime+notes+projection_name.split('.tif')[0] 

    ckpt_root = './Checkpoints'
    result_root = './Results'
    tb_root = './Log'
    ckpt_dir = os.path.join(ckpt_root, project_name)
    result_dir = os.path.join(result_root, project_name)
    tb_dir = os.path.join(tb_root, project_name)
    
    for dir_path in [ckpt_dir, result_dir, tb_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    with open(os.path.join(ckpt_dir, 'config.txt'), 'w') as f:
        f.write(str(args))
    
    train_db = SyntheticData(args.lf_dir, args.psf_dir, device, args.Nnum, args.input_size)
    train_loader = DataLoader(train_db, batch_size=1, shuffle=True)

    test_lfs = train_db.test_lf_imgs.to(device)
    psfs, warp_psfs, energy_rate = train_db.getPSF()
    psf_energy_mean = torch.mean(psfs[0,...].sum(-1).sum(-1))

    u_res, z_res, psf_res, _ = psfs.shape
    select_v = np.arange(0, u_res, 2)
    remain_v = np.arange(1, u_res, 2)

    if args.use_gaussian:
        model = V2V3D_Gauss(
            warp_psfs, z_res, select_v, remain_v,
            input_size=args.input_size, use_views=u_res, feat_ch=args.feat_ch,
            scale_init=args.gauss_scale_init,
            s_min=args.gauss_s_min, s_max=args.gauss_s_max,
            skip_voxelizer=args.use_hybrid,
        ).to(device)
        which = 'HybridRenderer' if args.use_hybrid else 'Intensity Voxelizer'
        print(f'[V2V3D] use_gaussian=True — Gaussian Decoder + {which}')
    else:
        model = V2V3D(warp_psfs, z_res, select_v, remain_v, u_res, args.feat_ch).to(device)

    hybrid_renderer = None
    if args.use_hybrid:
        assert args.use_gaussian, "--use_hybrid requires --use_gaussian"
        from hybrid_renderer import HybridRenderer, GaussianBatch
        # σ_max → half_xy/z so the analytic Gaussian box covers ~4σ.
        # Bigger half = bigger FFT but no information loss. half_z capped at
        # z_res-1 since the volume only has z_res depth slices.
        import math as _math
        h_xy = (args.hybrid_half_xy if args.hybrid_half_xy > 0
                else max(1, int(_math.ceil(args.gauss_s_max * 4))))
        h_z = (args.hybrid_half_z if args.hybrid_half_z > 0
               else max(1, int(_math.ceil(args.gauss_s_max * 4))))
        h_z = min(z_res - 1, h_z)
        hybrid_renderer = HybridRenderer(
            psfs, (args.input_size, args.input_size),
            z_norm='mean', mode='raw_exact',
            fixed_half_xy=h_xy, fixed_half_z=h_z,
            chunk_size=args.hybrid_chunk_size,
            use_checkpoint=not args.hybrid_no_checkpoint,
        ).to(device)

        # Override rho_bias for hybrid mode: default -5 makes init ρ ≈ 0.007,
        # which is too small for the hybrid render path (rendered output ~1e-6
        # smaller than target → gradient bootstrap problem). Setting to ~0
        # gives init ρ ≈ 0.7 so rendered scale is comparable to target.
        if args.hybrid_rho_bias is not None:
            with torch.no_grad():
                for unet in (model.unet1, model.unet2):
                    b = unet.conv_final[-1].bias
                    b[0:z_res] = args.hybrid_rho_bias
            print(f'[V2V3D] override rho_bias = {args.hybrid_rho_bias} '
                  f'(init ρ ≈ {torch.nn.functional.softplus(torch.tensor(float(args.hybrid_rho_bias))).item():.3f})')
        print(f'[V2V3D] use_hybrid=True — HybridRenderer (half_xy={h_xy}, '
              f'half_z={h_z}, chunk={args.hybrid_chunk_size}, '
              f'ρ_thr={args.hybrid_rho_threshold}, max_g={args.hybrid_max_gaussians})')

    optimizer = optim.Adam(model.parameters(), lr=args.lr_init)
    epochs = args.decay_init + args.decay_every
    db_size = len(train_db)
    loss_fn = nn.MSELoss()

    writer = SummaryWriter(tb_dir)

    loops = round(2000/db_size) if db_size<2000 else 1
    iters = loops*db_size
    
    for epoch in range(epochs):
        loss_total_mse = 0; iter = 0
        optimizer = adjust_lr(args.lr_init, args.lr_decay, epoch, args.decay_init, args.decay_every, optimizer)
        model.train()
        for loop in range(loops):
            for step, lf_all in enumerate(train_loader):
                lf_all = torch.squeeze(lf_all, 0).to(device)
                select_lfs = lf_all[select_v,...]
                remain_lfs = lf_all[remain_v,...]
                cv = lf_all.mean(dim=0)
                xguess1, xguess2, xguess = model(lf_all)

                if hybrid_renderer is not None:
                    # Hybrid path: render gauss params directly to LFI; skip
                    # full-volume generate_fps. The volumes (xguess*) are still
                    # used downstream by posloss / dc / tv terms.
                    gp1, gp2 = model.last_gauss_params
                    p1, s1, r1, n_kept1, n_total = _hybrid_soft_filter(
                        gp1, args.hybrid_rho_threshold, args.hybrid_max_gaussians,
                        explore_frac=args.hybrid_explore_frac,
                        soft_temp=args.hybrid_soft_temp)
                    p2, s2, r2, n_kept2, _ = _hybrid_soft_filter(
                        gp2, args.hybrid_rho_threshold, args.hybrid_max_gaussians,
                        explore_frac=args.hybrid_explore_frac,
                        soft_temp=args.hybrid_soft_temp)
                    g1 = GaussianBatch(positions=p1, sigmas=s1, rhos=r1)
                    g2 = GaussianBatch(positions=p2, sigmas=s2, rhos=r2)
                    gen_remain_lfs = hybrid_renderer(g1, target_views=remain_v.tolist())
                    gen_select_lfs = hybrid_renderer(g2, target_views=select_v.tolist())
                else:
                    gen_remain_fps = generate_fps(psfs[remain_v,...], torch.squeeze(xguess1))
                    gen_select_fps = generate_fps(psfs[select_v,...], torch.squeeze(xguess2))
                    gen_remain_lfs = gen_remain_fps.mean(dim=1)
                    gen_select_lfs = gen_select_fps.mean(dim=1)

                loss_mse = loss_fn(gen_remain_lfs, remain_lfs) + loss_fn(gen_select_lfs, select_lfs)
                loss_fft = fftloss(gen_remain_lfs, remain_lfs) + fftloss(gen_select_lfs, select_lfs)
                loss = loss_mse + loss_fft * 0.5
                if hybrid_renderer is None:
                    # Volume-dependent losses (positivity / cross-talk / z-TV)
                    # are skipped in hybrid mode because xguess* are zero
                    # placeholders. Gaussian-param regularizers below provide
                    # the equivalent constraints.
                    loss = loss + (posloss(xguess1) + posloss(xguess2)) * 1e-3
                    if args.dc_weight > 0:
                        loss += deCrosstalk_loss(cv, xguess, psf_energy_mean) * args.dc_weight
                    if args.tv_weight > 0:
                        loss += zloss(xguess) * args.tv_weight

                if args.use_gaussian and epoch >= args.gauss_reg_warmup:
                    gp1, gp2 = model.last_gauss_params
                    loss = loss + gauss_total_reg(
                        [gp1, gp2],
                        lambda_s=args.gauss_lambda_scale,
                        lambda_sp=args.gauss_lambda_sparse,
                        lambda_a=args.gauss_lambda_aniso,
                        s_max=args.gauss_s_max,
                    )
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                iter += 1

                loss_total_mse += loss_mse.cpu().item()
                lfmean = torch.mean(lf_all).item()
                hybrid_info = ''
                if hybrid_renderer is not None:
                    # Show render-vs-target scale to diagnose under-render.
                    rmean = gen_remain_lfs.detach().abs().mean().item()
                    tmean = remain_lfs.detach().abs().mean().item()
                    rmax = gen_remain_lfs.detach().abs().max().item()
                    tmax = remain_lfs.detach().abs().max().item()
                    rho_kept_max = r1.detach().max().item()
                    hybrid_info = (f' Hybrid:[{n_kept1}+{n_kept2}/{n_total}] '
                                   f'rend/tgt mean={rmean:.2e}/{tmean:.2e} '
                                   f'max={rmax:.2e}/{tmax:.2e} '
                                   f'ρmax={rho_kept_max:.3f}')
                print('Epoch:[%d/%d]'%(epoch+1,epochs),'Iters:[%d/%d]'%(iter,iters),
                      'MSE:',loss_mse.item(),'LFMean:',lfmean, hybrid_info)

                if args.hybrid_smoke > 0 and iter >= args.hybrid_smoke:
                    print(f'[smoke] {iter} iters complete; final loss={loss.item():.4e} '
                          f'(finite={torch.isfinite(loss).item()}); exiting.')
                    return

        loss_avg_mse = loss_total_mse/(iters)
        print('Epoch:[%d/%d]'%(epoch+1,epochs),' AVG MSE:',loss_avg_mse)
        
        writer.add_scalar('MSE Loss', loss_avg_mse, epoch)
        writer.add_image('CV Gen', normal(gen_remain_lfs[0,...].unsqueeze(0)), epoch)
        writer.add_image('CV GT', normal(remain_lfs[0,...].unsqueeze(0)), epoch)
        
        if epoch%int(epochs/10)==0 or epoch+1 == epochs: test2b(test_lfs, model, epoch, result_dir, args.log)
        if (epoch+1)%int(epochs/5)==0 or epoch+1 == epochs: save_ckpt(model, ckpt_dir, epoch)

if __name__ == '__main__':
    args = parse_args()
    train(args)
