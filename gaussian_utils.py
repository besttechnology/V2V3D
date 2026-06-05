import torch
import torch.nn.functional as F


def make_grid_centers(D, H, W, vol_D=None, vol_H=None, vol_W=None,
                      device=None, dtype=torch.float32):
    """预计算锚点世界坐标 (D, H, W, 3)。

    D,H,W 是锚点网格尺寸；vol_* 是目标体积（voxelizer 输出）尺寸，默认 = D,H,W。
    锚点落在它所覆盖块的中心：coord = (i + 0.5) * (vol_N / N)。
    当 vol == grid（每 voxel 一个锚点）时退化为 (i + 0.5)，即原行为。
    这里 3-vector 顺序 = (x, y, z) = (H 轴, W 轴, D 轴)。
    """
    vol_D = D if vol_D is None else vol_D
    vol_H = H if vol_H is None else vol_H
    vol_W = W if vol_W is None else vol_W
    xs = (torch.arange(H, device=device, dtype=dtype) + 0.5) * (vol_H / H)
    ys = (torch.arange(W, device=device, dtype=dtype) + 0.5) * (vol_W / W)
    zs = (torch.arange(D, device=device, dtype=dtype) + 0.5) * (vol_D / D)
    zz, xx, yy = torch.meshgrid(zs, xs, ys, indexing='ij')  # (D, H, W)
    return torch.stack([xx, yy, zz], dim=-1)  # (D, H, W, 3)


def decoder_output_to_gaussians(raw, grid_centers, scale_init=0.5,
                                max_offset=0.5, s_min=0.1, s_max=None):
    """把 (B, 11, D, H, W) 的 raw decoder 输出转成 Gaussian 参数张量。

    返回字段形状（B=1 场景下直接 squeeze 到无 batch 维）：
        positions: (N, 3) 世界坐标
        densities: (N, 1) 非负强度
        scales:    (N, 3) 正值
        rotations: (N, 4) 单位四元数
    其中 N = D*H*W。
    """
    assert raw.ndim == 5 and raw.shape[1] == 11, f"expect (B,11,D,H,W), got {raw.shape}"
    B, _, D, H, W = raw.shape
    assert B == 1, "当前训练 batch_size=1；多 batch 情况请在外层循环调用"
    N = D * H * W

    rho_logit = raw[:, 0]                 # (B, D, H, W)
    mu_offset = raw[:, 1:4]               # (B, 3, D, H, W)
    s_log     = raw[:, 4:7]               # (B, 3, D, H, W)
    q_raw     = raw[:, 7:11]              # (B, 4, D, H, W)

    # 激活（所有参数均做 NaN/Inf 保护，防止 CUDA voxelizer 产生全 NaN 输出）
    rho = torch.nan_to_num(F.softplus(rho_logit), nan=0.0)
    mu_delta = torch.nan_to_num(torch.tanh(mu_offset), nan=0.0) * max_offset
    s = torch.exp(s_log.clamp(-10, 10)) * scale_init
    s_clamp_max = s_max if s_max is not None else 1e6
    s_clamp_min = s_min if s_min is not None else 1e-6
    s = s.clamp(min=s_clamp_min, max=s_clamp_max)
    s = torch.nan_to_num(s, nan=scale_init, posinf=s_clamp_max, neginf=s_clamp_min)
    q_raw = torch.nan_to_num(q_raw, nan=0.0)               # NaN → 0 后再归一化
    q = F.normalize(q_raw, dim=1, eps=1e-8)

    # flatten 到 (N, *)
    rho = rho.reshape(N, 1)                                # (N, 1)
    # mu_delta: (B, 3, D, H, W) -> (D, H, W, 3) -> (N, 3)
    mu_delta = mu_delta[0].permute(1, 2, 3, 0).reshape(N, 3)
    positions = grid_centers.reshape(N, 3) + mu_delta      # (N, 3)
    scales = s[0].permute(1, 2, 3, 0).reshape(N, 3)        # (N, 3)
    rotations = q[0].permute(1, 2, 3, 0).reshape(N, 4)     # (N, 4)

    return {
        'positions': positions,
        'densities': rho,
        'scales': scales,
        'rotations': rotations,
    }


def init_gaussian_head_bias(final_conv, Dp,
                            rho_bias=-5.0, scale_bias=0.0, q_bias=(1.0, 0.0, 0.0, 0.0)):
    """按文档 §4.4 初始化 GaussianHead 最后一层 bias。

    Dp 是（粗化后的）锚点网格深度 D'，输出通道数 = 11 * Dp。
    final_conv 的输出顺序必须是 [rho, dx, dy, dz, sx, sy, sz, qw, qx, qy, qz] × Dp。
    我们按通道分组写 bias：
        channels [0 .. Dp)          -> rho_bias
        channels [Dp .. 4*Dp)       -> 0 (位置偏移)
        channels [4*Dp .. 7*Dp)     -> scale_bias
        channels [7*Dp .. 11*Dp)    -> q_bias (重复 Dp 次)
    """
    with torch.no_grad():
        b = final_conv.bias
        assert b.numel() == 11 * Dp, f"bias size {b.numel()} != 11*{Dp}"
        b.zero_()
        b[0:Dp] = rho_bias
        # 1..4: 位置偏移保持 0
        b[4 * Dp:7 * Dp] = scale_bias
        q_bias_t = torch.tensor(q_bias, dtype=b.dtype, device=b.device)
        for i, v in enumerate(q_bias_t):
            b[(7 + i) * Dp:(8 + i) * Dp] = v
