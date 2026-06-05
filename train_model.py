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
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gaussian 分支梯度裁剪上限，防止 voxelizer 反传梯度爆炸')
    parser.add_argument('--gauss_coarse', type=int, default=1,
                        help='高斯锚点 xy 粗化倍数（128→128/coarse），减少高斯数加速 voxelizer；1=每voxel一个')
    parser.add_argument('--gauss_coarse_z', type=int, default=1,
                        help='高斯锚点 z 粗化倍数（默认1=保留全部z切片，LFM中z分辨率重要）')
    parser.add_argument('--gauss_lambda_vol', type=float, default=0.0,
                        help='体素级L1稀疏权重(作用于渲染体积xguess，打破累加渲染铺底)；0=关闭')

    return parser.parse_args()

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
            gauss_coarse=args.gauss_coarse, gauss_coarse_z=args.gauss_coarse_z,
        ).to(device)
        print('[V2V3D] use_gaussian=True — Gaussian Decoder + Intensity Voxelizer')
    else:
        model = V2V3D(warp_psfs, z_res, select_v, remain_v, u_res, args.feat_ch).to(device)
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

                gen_remain_fps = generate_fps(psfs[remain_v,...], torch.squeeze(xguess1))
                gen_select_fps = generate_fps(psfs[select_v,...], torch.squeeze(xguess2))
                gen_remain_lfs = gen_remain_fps.mean(dim=1)
                gen_select_lfs = gen_select_fps.mean(dim=1)

                loss_mse = loss_fn(gen_remain_lfs, remain_lfs) + loss_fn(gen_select_lfs, select_lfs)
                loss_fft = fftloss(gen_remain_lfs, remain_lfs) + fftloss(gen_select_lfs, select_lfs)
                loss_pos = posloss(xguess1) + posloss(xguess2)
                loss = loss_mse + loss_fft * 0.5 + loss_pos * 1e-3
                if args.dc_weight > 0: loss += deCrosstalk_loss(cv, xguess, psf_energy_mean) * args.dc_weight
                if args.tv_weight > 0: loss += zloss(xguess) * args.tv_weight

                if args.use_gaussian and epoch >= args.gauss_reg_warmup:
                    gp1, gp2 = model.last_gauss_params
                    loss = loss + gauss_total_reg(
                        [gp1, gp2],
                        lambda_s=args.gauss_lambda_scale,
                        lambda_sp=args.gauss_lambda_sparse,
                        lambda_a=args.gauss_lambda_aniso,
                        s_max=args.gauss_s_max,
                    )
                    # 体素级 L1：约束渲染后的体积(而非高斯参数)，直接惩罚铺底——
                    # 高斯参数无法绕过(三次在rho/scale上的约束都被另一参数补偿了)
                    if args.gauss_lambda_vol > 0:
                        loss = loss + args.gauss_lambda_vol * (xguess1.abs().mean() + xguess2.abs().mean())
                
                optimizer.zero_grad()
                # NaN/Inf 守卫：loss 非有限时跳过本步，避免污染权重
                if args.use_gaussian and not torch.isfinite(loss):
                    print('[WARN] non-finite loss, skip step at epoch %d iter %d' % (epoch+1, iter+1))
                    iter += 1
                    continue
                loss.backward()
                if args.use_gaussian:
                    # 梯度裁剪：拦截 voxelizer 反传的爆炸梯度（防止权重发散成 NaN）
                    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                    if not torch.isfinite(gnorm):
                        print('[WARN] non-finite grad norm, skip step at epoch %d iter %d' % (epoch+1, iter+1))
                        optimizer.zero_grad()
                        iter += 1
                        continue
                optimizer.step()
                iter += 1

                loss_total_mse += loss_mse.cpu().item()
                lfmean = torch.mean(lf_all).item()
                print('Epoch:[%d/%d]'%(epoch+1,epochs),'Iters:[%d/%d]'%(iter,iters),'MSE:',loss_mse.item(),'LFMean:',lfmean)

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
