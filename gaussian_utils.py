import torch
import torch.nn.functional as F


def make_grid_centers(D, H, W, device=None, dtype=torch.float32):
    """预计算网格锚点世界坐标 (D, H, W, 3)。

    约定：sVoxel = nVoxel、center = nVoxel/2，每个 voxel 边长 = 1。
    voxel (i,j,k) 中心世界坐标 = (i + 0.5, j + 0.5, k + 0.5)。
    这里 3-vector 顺序 = (x, y, z) = (H 轴, W 轴, D 轴)。
    """
    xs = torch.arange(H, device=device, dtype=dtype) + 0.5
    ys = torch.arange(W, device=device, dtype=dtype) + 0.5
    zs = torch.arange(D, device=device, dtype=dtype) + 0.5
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

    # 激活
    rho = F.softplus(rho_logit)
    mu_delta = torch.tanh(mu_offset) * max_offset          # 限制在 voxel 内
    # 用 softplus 代替 exp：softplus(s_log) 对任意有限 s_log 都返回有限值，
    # 上极限 backward grad = sigmoid(s_log) ∈ (0,1) 永不爆炸。exp 在 s_log>87
    # 时 fp32 溢出 +Inf，下游 clamp 把前向 mask 住但 backward 链 d(clamp)/d(s)=0
    # 配 d(exp)=Inf 产出 NaN，连环污染参数。
    s = F.softplus(s_log) * scale_init                     # 有界 grad、防溢出
    if s_min is not None or s_max is not None:
        s = s.clamp(min=s_min if s_min is not None else 1e-6,
                    max=s_max if s_max is not None else 1e6)
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


def init_gaussian_head_bias(final_conv, n_slice,
                            rho_bias=-5.0, scale_bias=0.0, q_bias=(1.0, 0.0, 0.0, 0.0)):
    """按文档 §4.4 初始化 GaussianHead 最后一层 bias。

    final_conv 的输出顺序必须是 [rho, dx, dy, dz, sx, sy, sz, qw, qx, qy, qz] × n_slice。
    我们按通道分组写 bias：
        channels [0 .. n_slice)          -> rho_bias
        channels [n_slice .. 4*n_slice)  -> 0 (位置偏移)
        channels [4*n_slice .. 7*n_slice) -> scale_bias
        channels [7*n_slice .. 11*n_slice) -> q_bias (重复 n_slice 次)
    """
    with torch.no_grad():
        b = final_conv.bias
        assert b.numel() == 11 * n_slice, f"bias size {b.numel()} != 11*{n_slice}"
        b.zero_()
        b[0:n_slice] = rho_bias
        # 1..4: 位置偏移保持 0
        b[4 * n_slice:7 * n_slice] = scale_bias
        q_bias_t = torch.tensor(q_bias, dtype=b.dtype, device=b.device)
        for i, v in enumerate(q_bias_t):
            b[(7 + i) * n_slice:(8 + i) * n_slice] = v
