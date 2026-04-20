# GaussV2V3D：将3D Gaussian Splatting引入V2V3D的技术方案与实施指南

> 方案一（Gaussian Decoder）+ 方案三（Gaussian Feature Alignment）整合版 | 技术文档 v1.0

---

## 1. 总体改进思路

### 1.1 问题分析：V2V3D的当前局限

V2V3D是一个优秀的光场显微镜（LFM）去噪重建框架，但存在以下可改进之处：

- **体素表示的局限：** Decoder直接输出密集的3D体素网格（512×512×39），无论荧光信号是否存在，每个体素都需要计算和存储。对于荧光样本中常见的稀疏信号，这造成了大量计算浪费。

- **Feature Alignment的粗糙近似：** 原始方法将PSF转化为质心位置为1、其余为0的卷积核（Eq. 6）。这种"一刀切"的简化完全丢弃了PSF的空间扩展信息，仅保留了最粗糙的位移估计，导致在细节丰富区域的特征聚合不充分。

- **双分支融合策略简单：** 两个分支生成的3D体积直接取平均，未能在结构化的参数空间中进行更精细的融合。

### 1.2 改进方案总纲：GaussV2V3D

我们提出GaussV2V3D，核心改造包含两个相互协同的模块：

**改造一 — Gaussian Decoder（方案一）：** 将Decoder的输出从密集体素改为一组可学习的3D Gaussian基元参数，通过可微分的Gaussian-to-Voxel转换层得到最终3D体积。这使得网络能够以连续、稀疏、自适应的方式表示荧光信号。

**改造二 — Gaussian Feature Alignment（方案三）：** 将原始的PSF质心卷积核对齐替换为基于3D Gaussian Unprojection的可学习对齐模块。每个视图的特征被解释为2D Gaussian Splat，通过预测深度和空间参数反投影到统一的3D特征空间。

**两个改造的协同关系：** 方案三在Encoder端通过Gaussian Unprojection将2D特征精确汇聚到3D空间，方案一在Decoder端通过3D Gaussian基元高效表达和输出3D信号。整个pipeline形成了一个"Gaussian-centric"的统一框架：

```
输入特征 → Gaussian Unprojection聚合 → Decoder通过Gaussian基元输出3D结构
→ PSF前向投影生成模拟视图 → view2view loss驱动自监督训练
```

---

## 2. 整体架构设计

### 2.1 修改后的完整Pipeline

以下描述GaussV2V3D的完整数据流（以分支1为例，分支2对称）：

1. **输入：** 从U个光场视图中取出子集U₁的LFI图像集合

2. **Encoder（不变）：** 金字塔结构逐视图提取多尺度特征 F_u (u ∈ U₁)

3. **Gaussian Feature Alignment（改造）：**
   - a. 对每个视图的特征图预测逐像素的深度偏移和空间参数
   - b. 利用PSF先验初始化反投影参数
   - c. 通过可微分Gaussian Unprojection将所有视图特征汇聚到统一3D特征体

4. **Gaussian Decoder（改造）：**
   - a. U-Net处理3D特征体，输出每个空间位置的Gaussian参数
   - b. 可微分Gaussian-to-Voxel转换层生成3D强度体积

5. **前向投影（不变）：** 通过PSF卷积生成子集U₂的模拟LFI

6. **Loss计算：** MSE + FFT + De-crosstalk + 新增Gaussian正则化Loss

7. **融合：** 两个分支的Gaussian参数在参数空间融合，再转换为最终体素输出

### 2.2 与原始V2V3D的对比

**保持不变的部分：**
- Encoder权重共享机制
- view2view分割策略（奇偶分组）
- PSF前向投影物理模型
- FFT Loss和De-crosstalk Loss

**发生改变的部分：**
- Feature Alignment模块（质心卷积核 → Gaussian Unprojection）
- Decoder输出（密集体素 → Gaussian参数 + G2V转换）
- 融合策略（体素平均 → 参数空间融合）
- 新增Gaussian正则化损失

---

## 3. 改造一：Gaussian Feature Alignment模块

### 3.1 原始方法回顾

V2V3D原始的Feature Alignment过程可表达为：

```
Feature_aligned = Feature * Kernel_{PSF^{-1}}
```

其中Kernel是将PSF简化为质心位置为1、其余为0的卷积核。具体做法是：对每个深度z的PSF切片，计算其质心坐标，将该位置设为1，其余设为0，然后翻转用于反投影。

这种方法的核心问题是：PSF通常具有一定的空间扩展（尤其在远离焦面的深度），将其简化为单点会导致远离焦面深度的特征对齐严重失准。

### 3.2 改进方案：Gaussian Unprojection

#### 3.2.1 核心思想

将每个视图特征图的每个像素位置视为一个"2D Gaussian Splat"，通过预测其对应的3D空间参数（深度、空间扩展），将2D特征反投影到3D特征空间。PSF先验用于参数初始化，但允许网络在训练中自适应优化。

#### 3.2.2 模块设计细节

**Step 1 — PSF先验提取：**

对每个视图u、每个深度z的PSF切片PSF_{u,x,y,z}，预计算以下先验信息：

- 质心位置 (cx_uz, cy_uz)：即原始V2V3D的质心计算
- 空间扩展 (σx_uz, σy_uz)：对PSF切片拟合2D Gaussian，提取标准差
- 峰值强度 w_uz：PSF切片的最大值（用于权重初始化）

这些先验在训练前一次性计算并缓存，不参与梯度计算。

**Step 2 — 可学习参数预测网络：**

在Encoder输出的特征图上，附加一个轻量级的参数预测头（2-3层卷积），对每个像素位置(x,y)预测：

- Δd_{u,x,y}：深度偏移残差（相对于PSF先验质心给出的深度估计）
- Δσ_{u,x,y}：空间扩展调整因子（乘法残差，初始化为1.0）
- α_{u,x,y}：反投影置信度权重（sigmoid激活，初始化偏向0.5）

> **设计要点：** 使用残差预测而非直接预测，确保初始行为接近原始PSF先验，训练初期不会发散。

**Step 3 — Gaussian Unprojection操作：**

对视图u的特征图F_u中的每个位置(x,y)：

**(a) 计算该特征在3D空间中每个深度z的贡献权重：**

```
w(x,y,z,u) = α_{u,x,y} × G(z; d_prior_uz + Δd_{u,x,y}, σ_prior_uz × Δσ_{u,x,y})
```

其中G(·)是1D Gaussian函数，d_prior是PSF先验给出的深度映射。

**(b) 计算该特征在3D空间中的空间偏移：**

```
x_3d = x - cx_uz,  y_3d = y - cy_uz    // PSF先验的质心偏移
```

这一步与原始方法等价（质心位移），但权重分配更精细。

**(c) 将加权特征累加到3D特征体的对应位置：**

```
Feature3D[x_3d, y_3d, z] += w(x,y,z,u) × F_u[x, y]
```

**(d) 对所有视图的贡献进行归一化：**

```
Feature3D_aligned = Feature3D / (Σ_u w + ε)
```

#### 3.2.3 高效实现策略

直接逐像素计算代价过高。实际实现中采用以下优化：

- **分组卷积实现：** 将Gaussian Unprojection重新表述为"可学习权重的分组深度卷积"。对每个深度z，构造一个与原始PSF核大小相同的软卷积核（权重由Gaussian函数决定而非硬设为0/1），然后应用分组卷积。

- **稀疏计算：** 由于PSF在远离质心的位置权重极低，可以设置截断半径（如3σ），只在有效区域内计算，避免全图卷积。

- **深度并行：** 所有深度z的Unprojection可以并行计算（每个深度对应一个独立的软卷积核）。

#### 3.2.4 伪代码

```python
class GaussianFeatureAlignment(nn.Module):
    def __init__(self, psf_priors, n_depths, feat_dim):
        super().__init__()
        # psf_priors: dict包含centroid, sigma, weight (预计算)
        self.centroid = psf_priors['centroid']   # [U, Z, 2]
        self.sigma = psf_priors['sigma']         # [U, Z, 2]
        self.weight = psf_priors['weight']       # [U, Z]
        self.n_depths = n_depths

        # 可学习参数预测头
        self.param_head = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim // 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feat_dim // 4, 3, 1)  # 输出: delta_d, delta_sigma, alpha
        )

    def forward(self, features_per_view, view_indices):
        """
        features_per_view: [B, U, C, H, W]
        """
        B, U, C, H, W = features_per_view.shape
        Z = self.n_depths
        feature_3d = torch.zeros(B, C, Z, H, W, device=features_per_view.device)
        weight_sum = torch.zeros(B, 1, Z, H, W, device=features_per_view.device)

        for u in range(U):
            feat_u = features_per_view[:, u]   # [B, C, H, W]
            params = self.param_head(feat_u)    # [B, 3, H, W]
            delta_d = params[:, 0:1]
            delta_sigma = torch.exp(params[:, 1:2])   # 保证正数
            alpha = torch.sigmoid(params[:, 2:3])

            for z in range(Z):
                # 深度方向Gaussian加权
                d_prior = self.centroid[u, z, :]   # 质心位移
                s_prior = self.sigma[u, z, :]
                depth_weight = gaussian_1d(
                    z, d_prior + delta_d, s_prior * delta_sigma
                )
                w = alpha * depth_weight   # [B, 1, H, W]

                # 空间偏移 (使用grid_sample实现亚像素偏移)
                shifted_feat = spatial_shift(feat_u, d_prior)

                feature_3d[:, :, z] += w * shifted_feat
                weight_sum[:, :, z] += w

        return feature_3d / (weight_sum + 1e-8)
```

---

## 4. 改造二：Gaussian Decoder模块

### 4.1 原始Decoder回顾

V2V3D原始Decoder是一个标准的3D U-Net，输入为Feature Alignment后的3D特征体，直接输出单通道的3D强度体积 Î(x,y,z)，分辨率为512×512×39。

### 4.2 改进方案：Gaussian参数化输出

#### 4.2.1 核心思想

将Decoder的输出从"每个体素一个强度值"改为"每个体素位置一组Gaussian参数"，然后通过可微分的Gaussian-to-Voxel（G2V）层转换为最终的3D强度体积。

#### 4.2.2 Gaussian参数定义

对于每个体素位置(x,y,z)，Decoder输出以下参数：

- **强度 α(x,y,z)：** 该位置荧光信号的峰值强度。通过Softplus激活保证非负。

- **位置偏移 Δμ(x,y,z) = (Δx, Δy, Δz)：** 相对于体素中心的亚体素偏移。通过Tanh激活限制在[-0.5, 0.5]体素范围内。这允许网络以亚体素精度定位荧光信号。

- **各向异性Scale s(x,y,z) = (sx, sy, sz)：** 控制该Gaussian的空间扩展。通过Softplus激活保证正值。初始化约为1.0体素宽度。注意：轴向sz可以与横向sx, sy不同，适配LFM固有的各向异性分辨率。

> **简化设计选择：** 不引入旋转参数。荧光信号通常不具有旋转的各向异性（不同于自然场景中的表面法线），保持轴对齐的各向异性就足够了，而且避免了旋转参数的优化困难。

因此，Decoder输出通道数从原来的1变为 **7**（1 + 3 + 3）。

#### 4.2.3 Gaussian-to-Voxel（G2V）转换层

G2V层将Gaussian参数转换为密集的3D强度体积，以兼容后续的PSF前向投影：

```python
def gaussian_to_voxel(alpha, delta_mu, scale, grid_size):
    """
    alpha:    [B, 1, D, H, W]  -- 强度
    delta_mu: [B, 3, D, H, W]  -- 亚体素偏移
    scale:    [B, 3, D, H, W]  -- Gaussian宽度
    """
    B, _, D, H, W = alpha.shape
    volume = torch.zeros(B, 1, D, H, W, device=alpha.device)

    # 对每个体素位置，计算其Gaussian对邻域的贡献
    # 使用局部窗口（如3x3x3）避免全局计算
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                dist_sq = ((dx - delta_mu[:, 0:1]) / scale[:, 0:1]) ** 2 \
                        + ((dy - delta_mu[:, 1:2]) / scale[:, 1:2]) ** 2 \
                        + ((dz - delta_mu[:, 2:3]) / scale[:, 2:3]) ** 2
                gauss = alpha * torch.exp(-0.5 * dist_sq)
                # scatter_add到偏移后的位置
                volume = scatter_add(volume, gauss, offsets=(dx, dy, dz))

    return volume
```

> **实现优化：** 上述伪代码用循环展示逻辑。实际实现中可以用3D Unfold操作+向量化计算替代循环，在GPU上高效执行。窗口大小3×3×3在scale初始化为1.0时已覆盖99.7%的Gaussian质量。

#### 4.2.4 U-Net改造细节

对原始U-Net的改动极小：

- 最后一层卷积的输出通道从1改为7
- 在输出后添加参数分离和激活函数层
- 所有中间层结构不变

```python
class GaussianDecoder(nn.Module):
    def __init__(self, original_unet):
        super().__init__()
        self.unet = original_unet
        # 修改最后一层输出通道: 1 -> 7
        in_ch = self.unet.final_conv.in_channels
        self.unet.final_conv = nn.Conv3d(in_ch, 7, kernel_size=1)
        self.g2v = GaussianToVoxel()

    def forward(self, aligned_features):
        raw = self.unet(aligned_features)   # [B, 7, D, H, W]

        # 参数分离与激活
        alpha = F.softplus(raw[:, 0:1])                   # 强度 (非负)
        delta_mu = 0.5 * torch.tanh(raw[:, 1:4])          # 亚体素偏移 ([-0.5, 0.5])
        scale = F.softplus(raw[:, 4:7]) + 0.1             # 最小scale=0.1

        # G2V转换
        volume = self.g2v(alpha, delta_mu, scale)

        # 返回体积和参数（参数用于额外的loss计算）
        return volume, (alpha, delta_mu, scale)
```

---

## 5. 损失函数设计

### 5.1 保留的原始Loss

以下Loss完全保留，不做修改：

- **L_MSE：** 模拟LFI与真实LFI之间的均方误差
- **L_FFT：** 频域损失，恢复高频细节
- **L_DC：** De-crosstalk损失，抑制Z轴信号串扰

### 5.2 新增Loss 1：Gaussian稀疏性正则化 L_sparse

荧光样本的3D信号通常是稀疏的（大部分体素是背景）。我们引入稀疏性正则化来鼓励Gaussian基元的稀疏分布：

```
L_sparse = λ_s × mean(alpha)
```

这是对Gaussian强度参数alpha的L1正则化。它鼓励网络只在真正有信号的位置激活高强度的Gaussian，背景区域的alpha趋近于零。

> **超参数建议：** λ_s = 0.01。过大会抑制弱信号，过小则无法有效稀疏化。建议在验证集上调整。

### 5.3 新增Loss 2：Gaussian平滑性正则化 L_smooth

相邻空间位置的Gaussian参数应具有空间连续性（荧光结构是连续的）：

```
L_smooth = λ_m × (||∇_x scale||₂ + ||∇_y scale||₂ + ||∇_z scale||₂)
```

这约束了相邻体素的scale参数变化平缓，避免出现不自然的Gaussian形状突变。

> **超参数建议：** λ_m = 0.005。对delta_mu不加平滑约束（允许亚体素偏移的不连续，对应信号边缘）。

### 5.4 新增Loss 3：双分支Gaussian一致性损失 L_consist

这是GaussV2V3D最重要的新增Loss。两个分支从不同的视图子集出发，应该重建出一致的3D Gaussian场：

```
L_consist = λ_c × ||alpha_1 - alpha_2||₂ / N
```

其中alpha_1和alpha_2分别是两个分支输出的强度参数图。注意只约束强度一致性，不约束delta_mu和scale（这两者可以在不同视角信息下有合理差异）。

> **超参数建议：** λ_c = 0.1。该loss的作用是提供比体素级平均更强的跨分支正则化，但不应过强以免抑制两个分支各自的去噪能力。

### 5.5 总Loss公式

```
L_all = L_MSE + 0.1 × L_FFT + 1.0 × L_DC
      + λ_s × L_sparse + λ_m × L_smooth + λ_c × L_consist
```

---

## 6. 参数空间融合策略

### 6.1 原始方法

V2V3D原始融合：

```
Volume_final = (Volume_1 + Volume_2) / 2
```

即简单的体素级平均。

### 6.2 改进：参数空间加权融合

由于两个分支各自输出了Gaussian参数，可以在参数空间进行更智能的融合：

```python
def parameter_fusion(params_1, params_2):
    alpha_1, mu_1, s_1 = params_1
    alpha_2, mu_2, s_2 = params_2

    # 以置信度（强度）为权重的加权融合
    eps = 1e-8
    w1 = alpha_1 / (alpha_1 + alpha_2 + eps)
    w2 = alpha_2 / (alpha_1 + alpha_2 + eps)

    alpha_fused = (alpha_1 + alpha_2) / 2         # 强度取平均
    mu_fused = w1 * mu_1 + w2 * mu_2              # 位置加权平均
    scale_fused = w1 * s_1 + w2 * s_2             # scale加权平均

    # 最终G2V转换
    volume_final = G2V(alpha_fused, mu_fused, scale_fused)
    return volume_final
```

这种融合方式的优势是：在信号强的位置，两个分支的贡献根据各自的置信度加权；在信号弱的位置，两者趋于均等贡献，起到类似平均去噪的效果。

---

## 7. 训练策略与实施建议

### 7.1 分阶段训练策略（强烈推荐）

一次性训练所有改动可能导致不稳定。建议采用三阶段渐进式训练：

#### 阶段一：热身（Epoch 1-50）

- 冻结Gaussian Feature Alignment的可学习参数（Δd, Δσ, α），使其退化为原始PSF质心对齐
- Gaussian Decoder中，delta_mu初始化为0、scale初始化为1.0，强制G2V退化为恒等映射
- 此阶段等价于原始V2V3D的训练，确保网络先学到基本的重建能力
- Loss：仅使用原始的 L_MSE + L_FFT + L_DC

#### 阶段二：Gaussian激活（Epoch 50-150）

- 解冻Gaussian Decoder的所有参数，允许学习亚体素偏移和自适应scale
- Feature Alignment仍然冻结，使用PSF先验
- 逐步加入 L_sparse 和 L_smooth，权重从0线性增加到目标值
- 学习率降低为阶段一的1/2

#### 阶段三：全面优化（Epoch 150-300）

- 解冻Feature Alignment的可学习参数
- 加入 L_consist
- 学习率降低为阶段一的1/5
- 此阶段允许网络自适应优化PSF对齐参数

### 7.2 关键超参数

以下给出建议的超参数设置（基于V2V3D原始配置和3DGS的经验）：

| 超参数 | 建议值 | 备注 |
|--------|--------|------|
| 学习率 | 1e-4 → 5e-5 → 2e-5 | 三阶段递减 |
| Optimizer | Adam | 与原始V2V3D一致 |
| G2V窗口大小 | 3×3×3 | 可尝试5×5×5获得更平滑结果 |
| Scale初始化 | sx=sy=1.0, sz=1.5 | 反映LFM轴向分辨率低于横向 |
| λ_s (稀疏) | 0.01 | 验证集上调整 |
| λ_m (平滑) | 0.005 | 对scale参数施加 |
| λ_c (一致性) | 0.1 | 仅约束alpha |
| Batch size | 与原始V2V3D相同 | 受GPU内存限制 |

### 7.3 内存开销评估

GaussV2V3D相比原始V2V3D的额外内存开销来源于：

- Decoder输出通道从1增到7：约增加6倍的最后一层参数和梯度存储
- G2V转换层：常数级额外内存（3×3×3窗口的临时变量）
- Feature Alignment参数预测头：约增加0.5M参数

总体估计：在A100 GPU上，额外内存开销约15-20%，完全可控。如果内存紧张，可以通过梯度累积或降低batch size解决。

---

## 8. 建议的实验设计

### 8.1 消融实验矩阵

建议按以下消融实验验证每个组件的贡献：

| 配置 | Gauss Decoder | Gauss Align | L_sparse | L_consist | 对比基线 |
|------|:---:|:---:|:---:|:---:|------|
| V2V3D (baseline) | ✗ | ✗ | ✗ | ✗ | PSNR/SSIM |
| + Gauss Decoder only | ✓ | ✗ | ✗ | ✗ | vs baseline |
| + Gauss Align only | ✗ | ✓ | ✗ | ✗ | vs baseline |
| + Both (no extra loss) | ✓ | ✓ | ✗ | ✗ | vs above |
| + L_sparse | ✓ | ✓ | ✓ | ✗ | vs above |
| Full GaussV2V3D | ✓ | ✓ | ✓ | ✓ | vs all |

### 8.2 对比实验

在V2V3D的合成数据集和真实数据集上，与以下方法对比：

- **RLD：** 传统Richardson-Lucy反卷积
- **VCDNet：** 监督学习方法
- **DINER：** NeRF-based方法
- **V2V3D：** 原始baseline
- **3DGAT：** 3DGS-based光场显微重建（如有代码可复现）
- **DeepCAD + RLD/DINER：** 预去噪+重建的两阶段方法

### 8.3 重点分析方向

- **噪声鲁棒性：** 在不同噪声等级下（σ = 25, 50, 100）对比PSNR/SSIM
- **轴向分辨率：** 亚体素偏移机制是否改善了Z轴方向的信号定位精度
- **高频细节恢复：** Gaussian Feature Alignment是否比原始PSF质心对齐恢复了更多细节
- **稀疏性分析：** 可视化Gaussian强度图alpha的分布，验证稀疏化效果
- **计算效率：** 与V2V3D和3DGAT对比训练和推理时间

---

## 9. 风险评估与应对方案

### 9.1 潜在风险

**风险1 — 训练不稳定：** G2V层中的exp操作可能导致梯度爆炸。

- 应对：限制scale的最大值（如clamp到[0.1, 5.0]）
- 应对：使用梯度裁剪（max_norm=1.0）
- 应对：分阶段训练（第7节）

**风险2 — G2V与PSF前向投影的重复模糊：** G2V将Gaussian渲染为体素时已引入了空间扩展，后续PSF卷积又引入模糊。两次模糊可能过度平滑。

- 应对：将scale初始化为较小值（0.5-0.8而非1.0），让G2V层的模糊幅度小于体素分辨率
- 应对：或者在PSF前向投影前，对G2V输出做自适应锐化

**风险3 — Feature Alignment过拟合：** 可学习的对齐参数可能过拟合到训练数据的特定噪声模式。

- 应对：使用强L2正则化
- 应对：冻结预测头的BN层
- 应对：或在推理时使用PSF先验作为fallback

### 9.2 保底方案

如果方案一+方案三的联合改造在实验中效果不佳，可以退化为仅保留其中一个改造：

- **仅保留方案三（Gaussian Feature Alignment）：** 对原始V2V3D的改动最小，风险最低
- **仅保留方案一的简化版：** 只输出alpha（强度）和delta_mu（偏移），不输出scale，G2V退化为可微分的亚体素偏移插值

---

## 10. 总结与创新点梳理

### 10.1 论文贡献点（可用于摘要和Introduction）

1. 提出GaussV2V3D，**首次将3D Gaussian Splatting的显式表示引入端到端的光场显微镜去噪重建网络**，实现了连续、稀疏、自适应的3D荧光信号表示。

2. 设计了**Gaussian Feature Alignment模块**，用基于Gaussian Unprojection的可学习对齐替代了V2V3D的粗糙PSF质心对齐，在保留物理先验的同时允许自适应优化，显著提升了细节恢复能力。

3. 提出了**Gaussian参数化Decoder和Gaussian-to-Voxel转换层**，支持亚体素精度的信号定位和荧光信号的稀疏化表示，同时保持与物理前向投影模型的兼容性。

4. 提出**参数空间融合策略和Gaussian一致性损失**，为view2view框架的双分支融合提供了比体素级平均更有效的跨分支正则化。

### 10.2 与相关工作的差异化

- **vs V2V3D：** GaussV2V3D在V2V3D的去噪重建框架基础上，引入3DGS的显式表示思想，改进了Feature Alignment和Decoder两个核心模块，实现了更精确的特征对齐和更灵活的3D信号表示。

- **vs 3DGAT：** 3DGAT是纯优化方法（每个场景单独优化，无去噪能力），GaussV2V3D是端到端的网络方法，结合了view2view的去噪能力和3DGS的表示优势，适用于低信噪比的实时/快照成像场景。

- **vs DINER：** DINER使用NeRF的隐式表示，计算成本高且噪声鲁棒性差。GaussV2V3D使用3DGS的显式表示，在保持高重建质量的同时大幅提升计算效率。

### 10.3 推荐的论文标题方向

- GaussV2V3D: Gaussian Splatting Enhanced View-to-View Denoised 3D Reconstruction for Light-Field Microscopy
- Towards Sub-Voxel Precision: Integrating 3D Gaussian Representations into Self-Supervised LFM Reconstruction
- Gaussian-Centric Light-Field Microscopy: Bridging Explicit 3D Representation and Unsupervised Denoising Reconstruction
