# PSF Affine Gaussian Channel 失败证据报告

> 用于记录 v3 文档(`V2V3D_GaussianPile_Integration_Guide_v3.md`)所提"view-dependent affine Gaussian channel" PSF 模型在本项目数据集 `PSF_zoom2_39dz1_N13` 上的实验失败,作为后续走 voxel-based scheme 的物理依据,以及未来 paper 中的 negative result 支撑。

- 数据集: `PSF/PSF_zoom2_39dz1_N13/`,U=13 view,Z=39 z-slice,空间 128×128 px
- 拟合脚本: `psf_fit.py`
- 拟合参数文件: `psf_params_M{1,2,3}.pt`
- 详细每 view 报告: `psf_fit_report_M{1,2,3}.md`
- 判定门 (v3 §4.4): Err ≥ 10% 即判定不应直接用 Gaussian channel

---

## 1. 拟合模型回顾(v3 §3.2 / §4.2)

每个 view 的 PSF 建模为 mixture of affine Gaussian channels:

```
p | x ~ Σ_m a_{u,m}(z) · N_2D(p ; A_{u,m} x + b_{u,m}, C_{u,m})
A_{u,m} = [ 1  0  d_x^{(u,m)} ;
            0  1  d_y^{(u,m)} ]    (前 2×2 固定为 I,见 v3 §4.2)
```

每个 component 7 自由参数(d_x, d_y, b_x, b_y, C_xx, C_xy, C_yy),`a` 按 z 离散存储。

---

## 2. 总体拟合误差

| Mixture 阶数 M | Overall Err | v3 §4.4 判定 |
|---|---|---|
| 1 | **50.99%** | 不应直接用 Gaussian channel |
| 2 | **51.00%** | 不应直接用 Gaussian channel |
| 3 | **54.03%** | 不应直接用 Gaussian channel(更差) |

误差定义: `Err = ‖PSF_real - PSF_fit‖_2 / ‖PSF_real‖_2`,在所有 (u, z, h, w) 上聚合。

**关键观察**: 增加 mixture 分量不仅没有改善,反而更差。这与 v3 §4.3 期望(M=1~3 通常够用,M≥5 才认为模型失败)直接矛盾,说明**问题不是"主瓣 + sidelobe"分解,而是主瓣本身就不能被高斯描述**。

---

## 3. 误差按 |z| 分层

按到焦面距离分箱后的平均 Err:

| \|z\| 范围 | M=1 | M=2 | M=3 |
|---|---|---|---|
| [0, 3)  | 53.65% | **61.74%** | **70.06%** |
| [3, 7)  | 49.98% | 50.26% | 53.10% |
| [7, 12) | 41.38% | 39.01% | 40.37% |
| [12, 20)| **29.44%** | 37.32% | 43.70% |

**两条结论**:

1. **离焦面越近,误差越大**——焦面 PSF (\|z\|<3) 误差是远焦面 (\|z\|>12) 的 ~1.8 倍。
2. **远焦面(|z|>12)的 M=1 拟合误差 29%——其实接近"勉强可用"区**。增加 M 反而恶化,说明残差迭代在远焦面扔掉了真高斯结构;在近焦面则纯粹是高斯模型表达不了的问题,peel 一次只能加重错误。

---

## 4. 像素级反例(决定性证据)

### 4.1 焦面 PSF 违反高斯径向单调性

view 0 在 z=mid (z=0,焦面)峰值附近 5×5 像素值,以峰值归一:

```
[[0.258  0.414  0.591  0.414  0.258]
 [0.414  0.819  0.714  0.819  0.414]
 [0.591  0.714  1.000  0.714  0.591]
 [0.414  0.819  0.714  0.819  0.414]
 [0.258  0.414  0.591  0.414  0.258]]
```

定量对比:

| 位置 | 距峰 (px) | 值/峰 |
|---|---|---|
| 中心 | 0     | 1.000 |
| 主轴邻居 | 1     | **0.714** |
| 对角邻居 | √2≈1.41 | **0.819** |

**对角/主轴 = 1.148,高斯必然 < 1**(因为高斯仅依赖 Mahalanobis 距离,沿对角方向距离更远,值必然更低)。这是任何 axis-aligned 高斯——以及任何**和**高斯——都不可能产生的图样。

### 4.2 离焦 PSF 是好高斯

同一 view 在 z=-19(远离焦面)峰值附近 5×5,归一后:

```
[[0.199  0.387  0.458  0.387  0.199]
 [0.387  0.751  0.871  0.751  0.387]
 [0.458  0.871  1.000  0.871  0.458]
 [0.387  0.751  0.871  0.751  0.387]
 [0.199  0.387  0.458  0.387  0.199]]
```

| 位置 | 距峰 (px) | 值/峰 |
|---|---|---|
| 主轴邻居 | 1     | **0.871** |
| 对角邻居 | √2    | **0.751** |

主轴 > 对角,**满足高斯单调性**。这解释了 §3 中"|z|>12 误差最低"——远焦面 PSF 接近高斯,只在那个 z 区段拟合勉强可接受。

### 4.3 物理解读

近焦面 PSF 呈现 4-fold 对称的菱形/十字结构,这是 LFM lenslet array 衍射图样的标准形态(Bessel-like / sinc-like rings),非高斯。远焦面被光束扩散平滑后才趋于高斯。

---

## 5. 能量集中度与 mixture 失效原因

各 view 在焦面切片(z=mid),99.9%~100% 能量集中在 **r ≤ 3 px** 范围内:

| view | r ≤ 3 内能量占比 | 峰值 |
|---|---|---|
| 0  | 99.9%  | 9.31e-05 |
| 1  | 100.0% | 8.89e-05 |
| 2  | 100.0% | 1.09e-04 |
| 3-9 | 100.0% | ~1.07e-04 |
| 10-12 | 100.0% | 8.89e-05 |

**这意味着**:

- 没有"sidelobe 在远处"的能量给后续 mixture 分量去拟。
- 第 2、3 个分量只能在主瓣同一个区域内 peel——但主瓣是非高斯的菱形/十字,任何高斯减出来的残差还是同一个对称形状,新的高斯 fit 又落回同一处,**peel 只能损失能量、加重误差**,不能减少误差。这正是 §2 看到 M=3 比 M=1 更差的原因。

v3 §4.3 假设的 mixture 分量"先主瓣、后 sidelobe"机制在这套数据上根本不存在。

---

## 6. v3 §3.2 中除"形状"外的物理量,affine 模型确实抓住了

为公平起见: v3 affine 模型对 PSF 的 **disparity 几何**和**轴向能量衰减**确实拟合得不错。这两项**与 PSF 形状解耦**,即使形状非高斯也能正确估计。

### 6.1 视差斜率 d_u

各 view 拟出的 d_u(单位 px/z):

```
view  d_x      d_y
 0   -0.000   -0.000     ← 中心 view,无视差 ✓
 1   -0.000   +0.305
 2   -0.000   +0.630
 3   -0.469   +0.469
 4   -0.630    0.000
 5   -0.469   -0.469
 6    0.000   -0.630
 7   +0.469   -0.469
 8   +0.630    0.000
 9   +0.469   +0.469
10   +0.305    0.000
11    0.000   -0.305
12   -0.305    0.000
```

13 view 围绕中心呈对称 hex pattern,d_u 大小、符号、对称性完全合乎 LFM 几何。这部分可信。

### 6.2 轴向能量衰减 a_u(z)

每 view 的 `a(z_max) / a(z_min)` 比值都在 4.0~4.3x 范围内,说明远焦面 PSF 能量约为焦面的 1/4。这一项与 v3 文档中"标量 a_u"的假设(§6.1)不一致——必须保留 depth-dependent a。

---

## 7. Hybrid 路线为何也不可行

v3 §10 风险 1 给出的退路:

> 主瓣用 Gaussian + sidelobe 用查表,混合 forward
> 退而求其次: Gaussian → 局部 voxelize → 对该 voxel 区域做 PSF conv

但本数据集**没有可拆分的"高斯主瓣"**:

- 99.9%+ 能量集中在 r ≤ 3 px,**主瓣本身就是非高斯的菱形结构**。
- 把它分裂为"高斯主瓣 + 残差 sidelobe"等价于把绝大部分能量丢给 sidelobe,等同于全走 voxel+conv,Gaussian 部分只剩名义上的几何监督。

---

## 8. 结论与处理建议

### 8.1 结论

- v3 文档的核心创新——把 forward renderer 替换为 view-dependent affine Gaussian channel——**在本项目 PSF 上 fundamentally 不成立**。
- 失败原因不在工程实现,而在**物理假设**: PSF ≈ Gaussian 在焦面附近不满足。
- Mixture (M=2,3) 不能挽救;Hybrid renderer 也只能退化为 voxel+conv。

### 8.2 路线决定

继续在当前分支实现 v3 affine renderer **不应推进**。建议:

1. 当前分支 `gauss-affine-projection` 保留 `psf_fit.py`、本报告、三份 `psf_fit_report_M{1,2,3}.md` 和 `psf_params_M{1,2,3}.pt` 作为 negative-result 的可复现证据。
2. 训练相关工作切回 `3dgat-voxelization` 分支的 `V2V3D_Gauss`(Gaussian → voxel → conv),它**不依赖 PSF 高斯假设**,是这套数据上的物理正解。
3. 后续 paper 中,本报告内容可以作为 "为什么我们没采用 affine projection renderer"的方法论说明,反而是论文的可信度加分项。

### 8.3 复现步骤

```bash
cd /root/autodl-tmp/V2V3D
python psf_fit.py --M 1 --out_params psf_params_M1.pt --out_report psf_fit_report_M1.md
python psf_fit.py --M 2 --out_params psf_params_M2.pt --out_report psf_fit_report_M2.md
python psf_fit.py --M 3 --out_params psf_params_M3.pt --out_report psf_fit_report_M3.md
python psf_fit.py --M 1 --save_vis    # 生成 psf_fit_vis/view_*.png 可视化对比
```
