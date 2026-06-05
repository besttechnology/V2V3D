import tifffile as tf
import numpy as np
import torch
import os
import configargparse
from tqdm import tqdm
from dataset import SyntheticData
from model import V2V3D, V2V3D_Gauss

def parse_args():
    parser = configargparse.ArgumentParser(
        description='V2V3D Testing',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
        default_config_files=['Config/test.yaml'] 
    )
    parser.add('-c', '--config', required=False, is_config_file=True)
    
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--psf_dir', type=str)
    parser.add_argument('--lf_dir', type=str)
    parser.add_argument('--model_path', type=str, required=True)

    parser.add_argument('--Nnum', type=int, default=13)
    parser.add_argument('--input_size', type=int, default=256)
    parser.add_argument('--feat_ch', type=int, default=4)

    parser.add_argument('--result_dir', type=str, default=None)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--use_gaussian', action='store_true',
                        help='使用 V2V3D_Gauss 模型（与训练时一致）')
    parser.add_argument('--gauss_coarse', type=int, default=1,
                        help='高斯 xy 粗化倍数，必须与训练时一致')
    parser.add_argument('--gauss_coarse_z', type=int, default=1,
                        help='高斯 z 粗化倍数，必须与训练时一致')
    parser.add_argument('--gauss_s_min', type=float, default=0.1,
                        help='高斯 scale 下限，必须与训练时一致')
    parser.add_argument('--gauss_s_max', type=float, default=None,
                        help='高斯 scale 上限，必须与训练时一致（否则推理 scale 会撑大铺底）')

    return parser.parse_args()

def infer_patched(model, lf_img, patch, z_res):
    """分块推理：把整图按 patch 切块逐块重建再拼，避免 Gaussian voxelizer 在大图上 OOM。
    patch 应等于训练 input_size，以命中缓存的 grid/voxelizer。"""
    V, H, W = lf_img.shape
    if H <= patch and W <= patch:
        _, _, xg = model(lf_img)
        return torch.squeeze(xg)
    vol = torch.zeros((z_res, H, W))
    for i in range(0, H, patch):
        for j in range(0, W, patch):
            ie, je = min(i + patch, H), min(j + patch, W)
            _, _, xg = model(lf_img[:, i:ie, j:je])
            vol[:, i:ie, j:je] = torch.squeeze(xg).cpu()
    return vol

def test(args):
    device = torch.device('cuda', args.gpu_id)

    projection_name = args.model_path.split('/')[-2]
    result_dir = os.path.join('./Results', projection_name + '_test')
    
    os.makedirs(result_dir, exist_ok=True)
    
    with open(os.path.join(result_dir, 'test_config.txt'), 'w') as f:
        f.write(str(args))
    
    train_db = SyntheticData(args.lf_dir, args.psf_dir, device, args.Nnum, args.input_size, test_all=True)
    
    test_lfs = train_db.test_lf_imgs.to(device)
    test_lf_names = train_db.test_lf_names
    
    if test_lfs.ndim == 3: test_lfs = torch.unsqueeze(test_lfs, dim=0)
    
    psfs, warp_psfs, energy_rate = train_db.getPSF()
    
    u_res, z_res, psf_res, _ = psfs.shape
    select_v = np.arange(0, u_res, 2)
    remain_v = np.arange(1, u_res, 2)
    
    print(f"Loading: {args.model_path}")
    if args.use_gaussian:
        model = V2V3D_Gauss(warp_psfs, z_res, select_v, remain_v,
                            input_size=args.input_size, use_views=u_res,
                            feat_ch=args.feat_ch,
                            gauss_coarse=args.gauss_coarse,
                            gauss_coarse_z=args.gauss_coarse_z,
                            s_min=args.gauss_s_min, s_max=args.gauss_s_max).to(device)
    else:
        model = V2V3D(warp_psfs, z_res, select_v, remain_v, u_res, args.feat_ch).to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    
    with torch.no_grad():
        for i in tqdm(range(test_lfs.shape[0]), desc="Test Bar"):
            test_lf_img = test_lfs[i, ...]
            if args.use_gaussian:
                xguess = infer_patched(model, test_lf_img, args.input_size, z_res)
            else:
                _, _, xguess = model(test_lf_img)
            base_name = test_lf_names[i]
            recon_save_path = os.path.join(result_dir, base_name + '.tif')
            if args.use_amp:
                recon_data = torch.squeeze(xguess / train_db.amp[i]).cpu().numpy().astype(np.float32)
            else:
                recon_data = torch.squeeze(xguess).cpu().numpy().astype(np.float32)
            tf.imwrite(recon_save_path, recon_data)
    
    print(f"Saved: {result_dir}")

if __name__ == '__main__':
    args = parse_args()
    test(args)