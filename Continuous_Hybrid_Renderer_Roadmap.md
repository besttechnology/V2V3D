# Continuous Hybrid Renderer：阶段路线与数学推导

> 本文档记录 `gauss-continuous-fourier` 分支当前进展（Phase 1 已完成），下一步要实现的 Phase 2（PSF 沿 z 连续化），以及未来若干可探索方向。每一阶段附完整数学推导与正确性校验。

---

## 0. 总览：统一公式视角

V2V3D 的物理 forward 是

$$
\text{LFI}_u(x', y') \;=\; \iiint G(x, y, z)\;\text{PSF}_u(x' - x,\, y' - y;\, z)\;\mathrm{d}x\,\mathrm{d}y\,\mathrm{d}z
$$

其中 $G$ 是体积强度，PSF 沿 view $u$ 和 depth $z$ 变化。当体积用 Gaussian 基元参数化：

$$
G(x, y, z) \;=\; \sum_i \rho_i \, \exp\!\left[-\tfrac{1}{2}\Big(\tfrac{(x-\mu_{i,x})^2}{\sigma_{i,x}^2}+\tfrac{(y-\mu_{i,y})^2}{\sigma_{i,y}^2}+\tfrac{(z-\mu_{i,z})^2}{\sigma_{i,z}^2}\Big)\right]
$$

由强度可加性可换序求和与积分：

$$
\text{LFI}_u \;=\; \sum_i \iiint G_i(x,y,z)\;\text{PSF}_u(x'-x,\,y'-y;\,z)\,\mathrm{d}x\,\mathrm{d}y\,\mathrm{d}z
$$

**Hybrid renderer 的核心命题**：每个 Gaussian 在自己的 3σ 局部盒子内独立计算上述积分，最后加性 splat 回全局。问题归结为：**如何让这个三重积分对 $(\mu_x, \mu_y, \mu_z, \sigma_x, \sigma_y, \sigma_z, \rho)$ 全部连续可微？**

三个空间维度可以独立连续化：

| 维度 | Phase 1（已完成） | Phase 2（待做） | 进一步 |
|---|---|---|---|
| $x, y$ | 连续 2D Fourier + 相位编码 $\mu_{xy}$ | — | — |
| $z$ | nearest-$z$ 切片 + 整数处采样 | **PSF 沿 $z$ 线性插值 + Gaussian 解析 $z$ 积分** | spline / sinc 插值；全 3D 解析 |
| 视图 $u$ | 离散视图，PSF 自带 view-dependent shift | 不变 | 不变 |

整篇路线的**统一思想**是：

> *先在连续域写解析式，最后才落到离散网格。*

Phase 1 把这件事做在了 $(x, y)$，Phase 2 把它推广到 $z$。

---

## 1. Phase 1：xy 解析 Fourier（已完成）

### 1.1 设计动机

原 `raw_exact` 模式按以下流程：

```
对每个 Gaussian:
    在局部 box [r_x, r_y, B_z] 的整数 voxel 上采样 g(x, y, z)
    → rfft2 得 V̂
    → 与 PSF_u_z 的 FFT 相乘
    → irfft2 → patch
    → splat 回整数 anchor (round(μ_x), round(μ_y))
```

问题：

1. splat 的整数 anchor 让 $\mu_{xy}$ 的子像素分量对输出**完全没有梯度**（anchor 不动）。
2. voxel 上离散采样 Gaussian 引入双线性级别的离散误差，对 $\mu_{xy}$ 子像素移动的响应不严格光滑。

### 1.2 数学公式

连续 2D Gaussian：

$$
g(x, y) \;=\; \rho \, \exp\!\left[-\tfrac{1}{2}\Big(\tfrac{(x-\mu_x)^2}{\sigma_x^2}+\tfrac{(y-\mu_y)^2}{\sigma_y^2}\Big)\right]
$$

其连续 2D Fourier 变换（频率单位：cycles per voxel）：

$$
\boxed{\;\widehat{g}(f_x, f_y) \;=\; \rho \cdot 2\pi \sigma_x \sigma_y \cdot \underbrace{\exp\!\big[-2\pi^2(\sigma_x^2 f_x^2 + \sigma_y^2 f_y^2)\big]}_{\text{envelope}} \cdot \underbrace{\exp\!\big[-2\pi i (f_x \mu_x + f_y \mu_y)\big]}_{\text{phase ramp}}\;}
$$

**两条性质**：

- envelope 仅依赖 $\sigma$，对 $\mu$ 无关。
- phase ramp 把 $\mu$ 的**子像素**位置精确编码为相位。位移定理 $\mathcal{F}[g(x - \delta)] = e^{-2\pi i f \delta}\,\widehat{g}(f)$ 保证这是严格无近似的。

### 1.3 局部坐标与 DFT 频率约定

局部 box 在世界坐标系下的索引范围是 $[c_x - h, c_x + h]$，其中 $c_x = \lfloor \mu_x \rfloor$（splat anchor）、$h$ = `fixed_half_xy`。Box 内 voxel 中心位于 $c_x - h + 0.5,\,\ldots,\,c_x + h - 0.5$。

DFT 把 box 的局部索引 $0, 1, \ldots, r_x-1$ 映射为频率：

$$
f_x^{(k)} \;=\; \begin{cases} k / r_x & 0 \le k \le r_x/2 \\ (k - r_x)/r_x & k > r_x/2 \end{cases} \quad (\text{cycles per voxel})
$$

`rfftfreq` 在 $f_y$ 上只保留正半轴（共 $r_y/2 + 1$ 个频率）。

box-local 坐标下 Gaussian 中心为

$$
\mu_x^{\text{local}} \;=\; (\mu_x - c_x - 0.5) + h
$$

当 $\mu_x = c_x + 0.5$（box 中心 voxel 中心）时 $\mu_x^{\text{local}} = h$。phase ramp 用 $\mu^{\text{local}}$，envelope 用 $\sigma$ 即可。

### 1.4 实现位点

文件 `hybrid_renderer.py::_chunk_freq_compute_continuous`（commit `290a7d4`）。`train_model.py` 的 `--hybrid_mode {raw_exact, continuous_fourier}` 控制路径切换；splat 阶段仍走 raw_exact 的整数 anchor，避免与 PSF 自带 view-dependent shift 发生双重位移。

### 1.5 验证（`sanity_check_continuous.py`，已全部通过）

- **T1**：整数 $\mu$ 时 continuous_fourier 与 raw_exact rel L2 差 $\sim 1\mathrm{e}{-5}$。
- **T2**：sub-voxel $\mu_x$ 扫描下输出 C∞ 平滑，smoothness ratio < 5。
- **T3**：偏心 pixel readout loss 下 $\partial L/\partial \mu_x$ 的 FD vs autograd rel err < 1e-2。
- **T4**：多 Gaussian vs 全 volume reference rel L2 $\sim 2\mathrm{e}{-4}$。

### 1.6 当前局限

| 维度 | 是否子像素可微 | 备注 |
|---|---|---|
| $\mu_x, \mu_y$ | ✅ | phase ramp 完全编码 |
| $\sigma_x, \sigma_y$ | ✅ | envelope 闭式可微 |
| $\rho$ | ✅ | 线性乘子 |
| **$\mu_z$** | ❌ | **nearest-$z$ 整数采样，box 边界处有阶跃** |
| $\sigma_z$ | 部分 | 仅通过整数 z 上的密度采样间接影响 |

50-iter smoke 中 raw_exact 与 continuous_fourier 收敛行为几乎一致（ρmax 从 0.7 涨到 0.21，渲染均值仍差 target 四个量级），说明**当前瓶颈不在 xy 子像素可微，而在 z 维度的不连续**。这是 Phase 2 的直接动机。

---

## 2. Phase 2：PSF 沿 $z$ 连续化（规划中）

### 2.1 设计动机

物理事实：

- PSF 沿 defocus（$z$）变化是**光学衍射模式的连续演化**，不是阶梯函数。
- 一个深度 $z = 4.49$ 的发光点和 $z = 4.51$ 的发光点应该被几乎相同的 PSF 成像，而不是 PSF[4] 和 PSF[5] 这两张完全不同的图。

Phase 1 的 z 处理（与 raw_exact 共享）：

$$
\text{output}_u(x', y') \;\approx\; \sum_k g_z(k;\,\mu_z, \sigma_z)\; \cdot\; \text{PSF}_u(x', y';\, k)
$$

这等价于把 PSF 看作"在每个整数 $k$ 处取值，离散点列"，再用 Gaussian 在整数处的离散采样去做加权求和。它**既不是 PSF 的真实连续行为，也不是 Gaussian 与连续 PSF 的精确卷积**。

Phase 2 的目标是把 PSF 显式建模为 $z$ 的连续函数 $\text{PSF}_u(x',y';\,z)$（对任意实数 $z$ 有定义），然后做精确积分。

### 2.2 PSF $z$ 连续化的建模选择

最简单也最自然的选择：**沿 $z$ 分段线性插值**。

定义 tent 函数 $\Lambda(t) = \max(1 - |t|, 0)$，则

$$
\boxed{\;\text{PSF}_u(x', y';\,z) \;=\; \sum_{k=0}^{Z-1} \text{PSF}_u[k](x', y') \cdot \Lambda(z - k)\;}
$$

- $\Lambda_k(z) = \Lambda(z-k)$ 在 $[k-1, k+1]$ 上非零，在 $z = k$ 处取 $1$；
- $\sum_k \Lambda_k(z) = 1$（partition of unity），所以总能量守恒；
- 这是 PSF $z$ 维度上**最低阶的连续插值**（$C^0$，分段线性），更高阶（cubic spline / Fourier interp）留到 §3。

### 2.3 整体 forward 推导

把 (2.1) 代入 §0 的总积分，对 $z$ 维求积：

$$
\text{LFI}_u \;=\; \sum_i \iint G_i^{xy}(x, y)\;\Big[\textstyle\int g_i^z(z) \cdot \sum_k \text{PSF}_u[k]\,\Lambda_k(z) \,\mathrm{d}z\Big]\,\text{PSF}^{xy}\;\mathrm{d}x\,\mathrm{d}y
$$

其中 $G_i^{xy}$ 与 $G_i^z$ 分别是 Gaussian 在 xy 和 z 上的两个边缘（因 Gaussian 可分离）。交换求和与积分：

$$
\boxed{\;\text{LFI}_u(x',y') \;=\; \sum_i \sum_{k=0}^{Z-1}\; w_k(\mu_{i,z}, \sigma_{i,z})\;\cdot\;\Big[G_i^{xy} \otimes \text{PSF}_u[k]\Big](x'-\mu_{i,x},\,y'-\mu_{i,y})\;}
$$

其中

$$
w_k(\mu, \sigma) \;=\; \int_{-\infty}^{\infty} g_z(z;\mu,\sigma)\;\Lambda(z - k)\,\mathrm{d}z
\;=\; \int_{k-1}^{k+1} g_z(z)\;\Lambda(z-k)\,\mathrm{d}z
$$

$g_z(z;\mu,\sigma) := \exp\!\big[-\tfrac{1}{2}\big(\tfrac{z - \mu}{\sigma}\big)^2\big]$（与 renderer 保持一致的**未归一化**形式，归一化常数已被 $\rho$ 吸收）。

**关键性质**：
- $w_k$ 是 $\mu_z$ 的 $C^\infty$ 光滑函数（erf 平滑）。
- 对 PSF 部分**无任何近似**：所有 PSF 切片 $\text{PSF}_u[k]$ 都按解析权重 $w_k$ 进入求和，PSF 沿 $z$ 的连续性由 $\Lambda_k$ 的 partition-of-unity 表达。
- 与 Phase 1 兼容：xy 部分仍可用 §1.2 的连续 Fourier；只是 z 维度的求和权重从 $g_z(k)$ 替换成 $w_k$。

### 2.4 $w_k$ 的闭式推导

将 tent 分两段：

$$
\Lambda(z - k) \;=\; \begin{cases} z - k + 1, & z \in [k-1, k]\\ k + 1 - z, & z \in [k, k+1]\\ 0, & \text{otherwise} \end{cases}
$$

定义零阶与一阶 truncated moment：

$$
\mathcal{M}_0(a, b;\,\mu,\sigma) := \int_a^b g_z(z;\mu,\sigma)\,\mathrm{d}z,\quad
\mathcal{M}_1(a, b;\,\mu,\sigma) := \int_a^b z\,g_z(z;\mu,\sigma)\,\mathrm{d}z
$$

则

$$
w_k \;=\; \underbrace{\big[\mathcal{M}_1(k\!-\!1, k) - (k\!-\!1)\mathcal{M}_0(k\!-\!1, k)\big]}_{\text{left tent}} \;+\; \underbrace{\big[(k\!+\!1)\mathcal{M}_0(k, k\!+\!1) - \mathcal{M}_1(k, k\!+\!1)\big]}_{\text{right tent}}
$$

**闭式表达**：

零阶矩用 erf：

$$
\mathcal{M}_0(a, b;\,\mu,\sigma) \;=\; \sigma\sqrt{\tfrac{\pi}{2}}\left[\mathrm{erf}\!\Big(\tfrac{b-\mu}{\sigma\sqrt{2}}\Big) - \mathrm{erf}\!\Big(\tfrac{a-\mu}{\sigma\sqrt{2}}\Big)\right]
$$

一阶矩由分部积分给出。利用 $g_z'(z) = -\tfrac{z-\mu}{\sigma^2}\,g_z(z)$，即 $z\,g_z(z) = \mu\,g_z(z) - \sigma^2\,g_z'(z)$：

$$
\mathcal{M}_1(a, b;\,\mu,\sigma) \;=\; \mu\,\mathcal{M}_0(a,b) \;+\; \sigma^2\,\big[g_z(a) - g_z(b)\big]
$$

代回 $w_k$ 并整理：

$$
\boxed{\;w_k(\mu,\sigma) \;=\; (\mu - k + 1)\,\mathcal{M}_0(k\!-\!1, k) \;+\; (k + 1 - \mu)\,\mathcal{M}_0(k, k\!+\!1) \;+\; \sigma^2\,\Delta g_z(k)\;}
$$

其中 $\Delta g_z(k) := g_z(k\!-\!1) - 2g_z(k) + g_z(k\!+\!1)$ 是 $g_z$ 在整数 $k$ 处的**离散 Laplacian**。

整个 $w_k$ 对 $(\mu_z, \sigma_z)$ 全部解析可微（erf 在 PyTorch / CUDA 都有原生 differentiable 实现，`torch.erf` 已开 autograd）。

### 2.5 公式自洽性与极限校验

#### A. partition-of-unity 守恒

对任意 $\mu, \sigma$：

$$
\sum_{k=-\infty}^{\infty} w_k(\mu, \sigma) \;=\; \int_{-\infty}^{\infty} g_z(z;\mu,\sigma) \cdot \Big[\sum_k \Lambda_k(z)\Big]\,\mathrm{d}z \;=\; \int g_z \,\mathrm{d}z \;=\; \sigma\sqrt{2\pi}
$$

即所有 $w_k$ 之和恰为 Gaussian 的归一积分。这给出一个免费的**单元测试**。

#### B. 极限 $\sigma_z \to \infty$（宽 Gaussian）

$g_z(z) \to 1$ 处处，$\mathcal{M}_0(k\!-\!1, k) \to 1$，$\mathcal{M}_0(k, k\!+\!1) \to 1$。

需要展开 $g_z$ 二阶项才能算 Laplacian：
$g_z(z) \approx 1 - \tfrac{(z-\mu)^2}{2\sigma^2} + O(\sigma^{-4})$，
$\Delta g_z(k) = -\tfrac{1}{2\sigma^2}\big[(k\!-\!1\!-\!\mu)^2 - 2(k\!-\!\mu)^2 + (k\!+\!1\!-\!\mu)^2\big] = -\tfrac{1}{\sigma^2}$。

代入：

$$
w_k \;\to\; (\mu - k + 1) + (k + 1 - \mu) + \sigma^2 \cdot (-\tfrac{1}{\sigma^2}) \;=\; 2 - 1 \;=\; 1
$$

合理：宽 Gaussian 在长度 1 的 tent 上接近常值 $1$，积分恰为 tent 面积 $1$。 ✅

#### C. 极限 $\sigma_z \to 0$（窄 Gaussian / Dirac）

$g_z \to \delta(z - \mu)$，于是 $w_k \to \Lambda(\mu - k)$。

具体地 $\mu \in (k_0, k_0+1)$ 时：
$w_{k_0} = k_0 + 1 - \mu$，$w_{k_0+1} = \mu - k_0$，其他 $w_k = 0$。

这正是**对 $\mu_z$ 子像素位置的线性插值权重**——即"位于 $\mu_z$ 的点光源被 PSF 在 $\mu_z$ 处的线性插值成像"。Phase 2 在 Dirac 极限下退化为对 PSF 的线性插值评估。 ✅

#### D. 退化到 nearest-$z$（"Phase 1 限制"）

如果把 $\Lambda$ 换成 indicator $\mathbb{1}_{[k-1/2,\,k+1/2]}$（阶 0 piecewise constant，而不是阶 1 piecewise linear），就退化到**原 commit 里写的 erf slice integral 计划**。$w_k$ 变为

$$
w_k^{\text{step}} \;=\; \mathcal{M}_0(k - 1/2,\, k + 1/2;\,\mu,\sigma)
$$

进一步若把整个 $g_z$ 替换为 $g_z(k) \cdot 1$（点采样而不是积分），就退化到 Phase 1 当前路径

$$
w_k^{\text{nearest}} \;=\; g_z(k;\,\mu, \sigma)
$$

三者关系：

| 路径 | PSF $z$ 模型 | Gaussian $z$ 处理 | $\mu_z$ 子像素可微 |
|---|---|---|---|
| Phase 1 现状 | 离散切片 | 整数点采样 $g_z(k)$ | 弱（仅密度变化） |
| 原 erf 计划 | 离散切片（piecewise constant） | erf slice integral | 中（slice 边界平滑） |
| **Phase 2 本方案** | **分段线性插值** | **解析 tent 积分** | **强（Dirac 极限即线性插值）** |
| 进一步方向 | spline / sinc | 高阶矩 / convolution | 更强 |

### 2.6 实现规划

#### 代码位点

新增 `_chunk_freq_compute_zlinear`（或在 `_chunk_freq_compute_continuous` 中加分支），覆盖现有的 `z_density = exp(-½...)` 计算段：

```text
原（Phase 1）:
    z_density[i, k] = exp(-½((zs_world[i, k] - μ_z[i]) / σ_z[i])²)
    psf_weighted = Σ_k z_density[i, k] · PSF[u, bz_global[i, k], :, :]

新（Phase 2, z-linear）:
    w[i, k] = (μ_z[i] - k + 1) · M0(k-1, k; μ_z[i], σ_z[i])
            + (k + 1 - μ_z[i]) · M0(k, k+1; μ_z[i], σ_z[i])
            + σ_z[i]² · (g_z(k-1) - 2 g_z(k) + g_z(k+1))
    psf_weighted = Σ_k w[i, k] · PSF[u, k, :, :]
```

`M0` 用 `torch.erf` 实现，整段对 $\mu_z, \sigma_z$ 全 autograd。

#### Box 范围

当前 box 沿 $z$ 取 $[\mu_z - 3\sigma_z, \mu_z + 3\sigma_z]$ 范围的整数 voxel。Phase 2 下需要把 box 扩 1：因为 $w_k$ 依赖 $g_z(k\!-\!1)$ 和 $g_z(k\!+\!1)$，box 边缘的 $k$ 需要邻居存在。最简单的处理是 PSF 在 $k = -1$ 和 $k = Z$ 处补零，即 $\Lambda_{-1}$、$\Lambda_Z$ 不参与求和。

#### 训练接入

新增 CLI flag：`--hybrid_mode {raw_exact, continuous_fourier, continuous_3d}`，或者复用 `continuous_fourier` 加 `--z_continuous {nearest, zlinear}`。建议后者，因为 xy 与 z 是两个正交开关。

#### 性能代价

- $w_k$ 计算每个 Gaussian × 每个 z slice 4 次 erf + 几次乘加，相对原 `exp` 多 ~2x flops，但 z 方向 slice 数远小于 xy 像素数，总开销增量 < 5%。
- 不增加显存（无 box 扩张大头）。

### 2.7 验证计划

| 测试 | 内容 | PASS 阈值 |
|---|---|---|
| **C1**（积分守恒） | $\sum_k w_k(\mu, \sigma) = \sigma\sqrt{2\pi}$ | rel err < 1e-6 |
| **C2**（$\sigma_z$ 极限） | $\sigma_z = 10$ 时 $w_k \to g_z(k) \cdot 1$；$\sigma_z = 0.05$ 时 $w_k \to \Lambda(\mu_z - k)$ | rel err < 1e-3 |
| **C3**（$\mu_z$ 子像素平滑） | 扫描 $\mu_z \in [5.0, 6.0]$，输出 C∞，smoothness ratio < 3 | — |
| **C4**（FD vs autograd） | $\partial L/\partial \mu_z$ 与 $\partial L/\partial \sigma_z$ | rel err < 1e-2 |
| **C5**（vs upsampled reference） | 将 PSF 沿 z 用 linear interp 上采样到 $10\times$ fine grid，做 dense forward 比较 | rel L2 < 0.5% |
| **C6**（与 Phase 1 在 $\mu_z = $ 整数时） | 不要求完全相等；预期差距 $O(\sigma_z^2 \cdot \|\partial^2_z g_z\|)$ | $\sim 1\%$ 级别 |

C5 是 Phase 2 的**黄金参考**：物理上正确的 forward 就是把 PSF 视为 z 的连续函数（用 fine-grid 离散逼近）后做 dense 卷积。Phase 2 公式与 fine-grid 直接逼近的差距应该接近 0。

### 2.8 插值阶选择讨论：线性的定位、局限、升阶路径

线性 tent 是把 PSF 沿 $z$ 从"阶梯"升到"连续 $C^0$"最便宜的一步。作为 Phase 2 的起点合理，但**不应作为最终方案**。这一节集中讨论它的定位与升阶路径。

#### A. 为什么先选线性

- **实现成本最低**：erf 一行 + Laplacian 一行，autograd 直接通。
- **物理已定性正确**：PSF 沿 $z$ 不再是阶梯函数，发光点 voxel 间位置的渐变能被分辨。
- **作为诊断工具的价值**：若它已能解开 $\rho$ 收敛瓶颈，说明"$z$ 不可微"就是主因，更高阶意义不大；若不够，正好量化差距决定升阶到什么程度。

#### B. 线性插值的失真模式

| 场景 | 失真机制 | 严重程度 |
|---|---|---|
| 焦面附近 | PSF 沿 $z$ 高曲率变化（defocus 模式快速反转），线性在两切片中点处误差最大 | 焦面能量被低估或高估 |
| 远焦面 sidelobe | PSF 有环状振荡，相邻切片可能同位置一正一负，线性会平均掉振荡 | 远焦能量被"抹平" |
| $\sigma_z \to 0$ | $w_k \to \Lambda(\mu_z - k)$，渲染 PSF 就是真实 PSF 的线性插值 | 子像素 $\mu_z$ 看到的 PSF 始终是低通版本 |

#### C. xy 与 z 的精度阶不对称

Phase 1 在 xy 上用了**连续 Fourier 相位项**——精确无近似的位移定理；Phase 2 在 z 上用 tent——分段线性近似。两个维度的精度阶差好几档：

| 维度 | 方法 | 精度阶 |
|---|---|---|
| $x, y$ | 连续 Fourier | exact（仅 DFT 截断误差 $\sim e^{-2\pi^2 \sigma^2 / 4}$） |
| $z$ | tent 线性 | $O(h^2)$，$h = 1$ voxel |

工程上不对称不致命（瓶颈维度决定整体精度），但**汇报时如果被问"xy 是 exact 的 Fourier，z 为什么用最朴素的线性"，需要在方法论上有解释**——即"用最低成本看 z 可微是否瓶颈，是再决定升阶到 cubic / sinc"。

#### D. 插值阶统一谱

把所有备选方案排在一条连续谱上：

| 阶 | 插值核 $\phi(z)$ | 连续性 | $w_k$ 闭式难度 | 文档位置 |
|---|---|---|---|---|
| $-1$ | $\delta_k$ (nearest) | $C^{-1}$ | 平凡（Phase 1 现状） | §1 |
| $0$ (step) | $\mathbb{1}_{[-1/2, 1/2]}$ | $C^{-1}$（仍阶梯） | erf 单段 | §2.5.D（原 erf 计划） |
| **$1$ (linear)** | **tent $\Lambda$** | **$C^0$** | **erf + Laplacian** | **§2.4（Phase 2 本方案）** |
| $2$ (Hermite) | 三次 Hermite + 数值导 | $C^1$ | erf + $\mathcal{M}_0..\mathcal{M}_3$，需 cache 导数 | §3.7 |
| $3$ (B-spline) | cubic B-spline $B_3$ | $C^2$ | erf + $\mathcal{M}_0..\mathcal{M}_3$ | §3.1 |
| $\infty$ (sinc) | $\mathrm{sinc}(z)$ | $C^\infty$（bandlimited） | 需走频域路径 | §3.2 |
| $\infty$ (parametric) | Fresnel defocus 作用于焦面 PSF | $C^\infty$（解析） | 完全闭式 | §3.6 |

#### E. 升阶决策路径

```
Phase 2a (linear)  ── 实现 + 验证 z 可微解开训练瓶颈
       │
       ▼
Phase 2b (cubic / Hermite) ── 实现 + 对比 2a，量化高阶收益（消融实验）
       │
       ▼
   决策分叉
       ├── 收益小: 停在 cubic
       ├── 收益大且 PSF 看起来 bandlimited: 上 sinc (§3.2)
       └── 物理建模可行: 走 Fresnel 参数化 (§3.6，论文价值最大)
```

**建议**：Phase 2 的产出不应只是"实现一个 linear 版本"，而是"实现 linear + cubic 两档 + 一个 ablation 表"，量化插值阶 vs 重建质量 / 收敛速度的关系。这本身就是一个完整的实验故事。

---

## 3. 未来可能方向

### 3.1 高阶 PSF $z$ 插值（cubic spline / B-spline）

线性插值（Phase 2）在 PSF 沿 $z$ 二阶导数大的区域（例如焦面附近 defocus 模式快速变化）有 $O(h^2)$ 截断误差。Cubic B-spline 把误差降到 $O(h^4)$。

需要的矩升到三阶：

$$
\mathcal{M}_n(a, b;\,\mu, \sigma) := \int_a^b z^n\,g_z(z;\mu,\sigma)\,\mathrm{d}z
$$

递推关系（由 $z^n g_z = \mu z^{n-1} g_z + \tfrac{1}{n+1} \tfrac{d}{dz}[z^{n+1} g_z] + \ldots$ 类似分部）：

$$
\mathcal{M}_{n+1} \;=\; \mu\,\mathcal{M}_n \;+\; n\sigma^2 \mathcal{M}_{n-1} \;+\; \sigma^2\,\big[a^n g_z(a) - b^n g_z(b)\big]
$$

cubic spline 基函数支持 $[k-2, k+2]$，每段是 $z$ 的三次多项式；需要 $\mathcal{M}_0, \mathcal{M}_1, \mathcal{M}_2, \mathcal{M}_3$。代码量翻 2-3 倍，但仍是闭式可微。

**何时上 cubic**：Phase 2 跑通后，如果 C5 测试在某些 z 区段 rel L2 > 0.5%，说明线性插值不够，再升阶。

### 3.2 sinc / Fourier $z$ 插值（理论最优但实现难）

若假设 PSF 沿 $z$ 是 bandlimited（光学衍射模型在 NA 限制下确实如此），最优插值是 sinc：

$$
\text{PSF}_u(z) \;=\; \sum_k \text{PSF}_u[k] \cdot \mathrm{sinc}(z - k)
$$

代入 forward 积分需要

$$
\int g_z(z;\mu,\sigma)\,\mathrm{sinc}(z - k)\,\mathrm{d}z
$$

= Gaussian 与 sinc 的卷积在 $z = k$ 处的值。

这没有 erf 形式的闭式解，但等价于 Gaussian 在频域被 rect 截断后再反变换：

$$
\int g \cdot \mathrm{sinc} = \mathcal{F}^{-1}\!\big[\widehat{g}(f) \cdot \mathrm{rect}(f)\big]\!\Big|_{z=k}
$$

实现路径：把 Gaussian 的 1D Fourier $\widehat{g_z}(f) = \sigma\sqrt{2\pi}\,e^{-2\pi^2\sigma^2 f^2} e^{-2\pi i f \mu}$ 在 $|f| \le 1/2$ 范围内取样、与 rect 相乘、再 irfft。等价于沿 $z$ 也走 Fourier 路径——和 Phase 1 的 xy 解析 Fourier **结构对称**。

**判断**：sinc 在数值上更准但工程复杂度大，Phase 2 线性版若在 C5 上已够好，可不必走 sinc。

### 3.3 与 `centered_affine` 合流：全可微 3D forward

现有 `gauss-hybrid-renderer` 分支里的 `centered_affine` 模式做的是**xy 维度的另一种连续化**：用 affine projection $\mu_{2D}^u = A_u\,\mu + b_u$ 给出 xy 子像素位置 + bilinear splat，PSF 用去中心化版本以避免 double shift。它和 continuous_fourier 是两条平行思路：

| | continuous_fourier | centered_affine |
|---|---|---|
| xy 子像素来源 | DFT 相位项 | 解析 affine projection + bilinear splat |
| PSF | raw（自带 view shift） | centered（shift 表外置） |
| 优点 | 与 raw_exact 完全数值对齐 | view shift 可微 |
| 缺点 | view shift 离散（取 PSF 切片 shift） | 需精确 affine fit |

合流方向：把 **z 连续化 (Phase 2) + xy continuous_fourier** 作为基线；若以后需要 view-dependent shift 也可微（例如 $u$ 也参数化为连续 view direction），再切换到 centered_affine。

### 3.4 全 3D 联合 Fourier（理论极致，但 PSF 是测量数据）

理想形式：把整个 forward 写在 3D 频域

$$
\text{LFI}_u \;=\; \mathcal{F}_{3D}^{-1}\!\big[\widehat{G}(\mathbf{f}) \cdot \widehat{\text{PSF}_u}(\mathbf{f})\big]
$$

Gaussian 有 3D 解析 FT；问题在 **$\widehat{\text{PSF}_u}$ 沿 $f_z$ 怎么定义**——PSF 测量数据沿 $z$ 是有限离散的，3D DFT 假定 $z$ 周期，不符合物理。可行做法：

1. 沿 $z$ 用 §3.2 的 sinc 插值得到连续 PSF；
2. 对连续 PSF 做沿 $z$ 的 1D Fourier（数值），再与 Gaussian 频域相乘；
3. 沿 $z$ 反变换回空间域。

这本质上等价于 §3.2，只是写在了频域里。**没有真正新的物理**。

### 3.5 CUDA 融合

Phase 1 / Phase 2 的所有计算（envelope + phase + erf weights + PSF gather + irfft2 + splat）目前是若干 PyTorch op 串行，显存搬动占大头。融合到一个 CUDA kernel：

- Gaussian 数 N × view 数 V × box voxel 数 $r_x r_y$ 的单 kernel；
- shared memory 缓存当前 chunk 的 PSF z-stack；
- output 直接累加到全局 LFI（atomic add 或 scatter）；
- 后向用 autograd recipe 反推（envelope/phase/erf 的解析梯度都已有）。

代码规模 1000+ 行 CUDA，**先把数值验证全部跑通再做**。

### 3.6 物理参数化 PSF 沿 $z$（Fresnel defocus，论文价值最大）

不做通用插值，而是**对 PSF 沿 $z$ 的演化做物理建模**。Fresnel 衍射理论给出

$$
\text{PSF}_u(x, y;\,z) \;=\; \mathcal{D}(z - z_0) \,*\, \text{PSF}_u(x, y;\,z_0)
$$

其中 $z_0$ 是焦面、$\mathcal{D}(\Delta z)$ 是 Fresnel 传播算子，在傍轴近似下有解析形式（频域为相位 chirp $e^{-i\pi\lambda \Delta z (f_x^2 + f_y^2)}$，空间域为 Gaussian-like Fresnel kernel）。若分解成立：

- **PSF 沿 $z$ 不需要插值**：任意 $z$ 处的 PSF 由焦面 PSF + 解析传播算子直接给出，$z$ 是真正连续的。
- **存储骤降**：只存一张焦面 PSF + 系统物理参数（NA、$\lambda$、$n$）。
- **解析可微**：传播算子对 $z$ 解析可微，autograd 自然通。

**前提**：要先用 PSF 测量数据拟合验证。若焦面 PSF + Fresnel 传播能在全部 $z$ 上拟合到残差 < 5%，这条路成立；若残差大（系统色散、像差、非傍轴效应显著），则放弃。

**为什么是"论文价值最大"**：把 PSF $z$ 行为从"数据驱动的插值"变成"物理驱动的解析模型"，与 Phase 1 的"Gaussian 解析 FT"在哲学上对称——整个 forward 都是从第一性原理推导出来的解析形式，没有任何离散插值的 ad hoc 选择。这种工作适合写成 method paper 的核心 contribution，相比"我们用 cubic 比 linear 好"这种叙事强得多。

### 3.7 Hermite 插值（cubic 之外的折中档）

线性和 cubic spline 之间还有一档：**带数值导数的三次 Hermite 插值**。在每个 $z = k$ 处保留 PSF 值 $\text{PSF}[k]$ 与数值一阶导 $\text{PSF}'[k] \approx (\text{PSF}[k+1] - \text{PSF}[k-1])/2$，用三次 Hermite 基函数插值：

$$
\text{PSF}(z) = h_{00}(t)\,\text{PSF}[k] + h_{10}(t)\,\text{PSF}'[k] + h_{01}(t)\,\text{PSF}[k+1] + h_{11}(t)\,\text{PSF}'[k+1]
$$

其中 $t = z - k \in [0, 1]$，$h_{ij}$ 是 Hermite 基函数。

| 方面 | Hermite | cubic B-spline |
|---|---|---|
| 连续性 | $C^1$（一阶导连续） | $C^2$（二阶导连续） |
| 局部支持 | 仅依赖相邻 2 切片 + 导数 | 依赖 4 切片 |
| 预处理 | 一次性 cache PSF z 方向数值导 | 无 |
| $w_k$ 矩阶 | $\mathcal{M}_0..\mathcal{M}_3$ | $\mathcal{M}_0..\mathcal{M}_3$ |

代码量与 cubic B-spline 接近，精度比 linear 显著提升。适合作为 cubic 之前的"中间档"消融对照，且数值导只在初始化时算一次，运行期开销与 linear 持平。

---

## 4. 整体路线收口

```
        ┌─────────────────────────────────────────┐
        │  raw_exact (V2V3D 黄金参考)              │
        │  μ_xy 不可微，μ_z 不可微                  │
        └────────────────┬────────────────────────┘
                         │ xy 解析 Fourier
                         ▼
        ┌─────────────────────────────────────────┐
        │  continuous_fourier (Phase 1, DONE)     │
        │  μ_xy 全可微 ✓ ， μ_z 仍不可微            │
        └────────────────┬────────────────────────┘
                         │ PSF 沿 z 线性插值
                         │ + Gaussian erf 矩
                         ▼
        ┌─────────────────────────────────────────┐
        │  continuous_3d (Phase 2, NEXT)          │
        │  μ_xy 全可微 ✓ ， μ_z 全可微 ✓             │
        │  整个 forward 在连续域闭式                │
        └────────────────┬────────────────────────┘
                         │ 高阶插值 / sinc / CUDA
                         ▼
                  Phase 3+（按需）
```

**第一里程碑**：Phase 2 完成 + 在真实数据集上和 raw_exact / Phase 1 做长训练 (50+ epoch) 对比，验证 z 可微是否能解开 ρ 收敛瓶颈。

**第二里程碑**：Phase 2 的插值阶消融——同时实现 linear 与 cubic / Hermite 两档，在 C5（vs upsampled reference）和真实数据集训练曲线上做对比，量化插值阶对精度与收敛的影响。

**决策分支**：

```
Phase 2a 跑通 (linear)
   │
   ├─ 训练收敛速度 / 重建质量比 Phase 1 显著提升？
   │   ├─ 是 → "z 可微是主因"假设成立，Phase 2 立项成功
   │   └─ 否 → 瓶颈在别处（top-K filter / ρ 初始化 / 数据 SNR），回去诊断
   │
   ▼
Phase 2b (cubic / Hermite) 对照
   │
   ├─ 在 C5 / 训练曲线上 cubic vs linear 差距大且收益落地？
   │   ├─ 大 → 继续 sinc (§3.2) / Fresnel 物理模型 (§3.6)
   │   └─ 小 → 停在 cubic，把工程精力转向 CUDA (§3.5) / 训练 scale up
```

---

## 附录 A：符号与约定

| 符号 | 含义 | 单位 |
|---|---|---|
| $x, y, z$ | 世界坐标 | voxel |
| $\mu_x, \mu_y, \mu_z$ | Gaussian 中心 | voxel |
| $\sigma_x, \sigma_y, \sigma_z$ | Gaussian 各向方差 | voxel |
| $\rho$ | Gaussian 强度（未归一化） | dimensionless |
| $u$ | view 索引 | $0, \ldots, U\!-\!1$ |
| $k$ | $z$ 整数切片索引 | $0, \ldots, Z\!-\!1$ |
| $f_x, f_y$ | 频率（rfft2 grid） | cycles per voxel |
| $r_x, r_y$ | 局部 box 边长 | voxel |
| $h = $ `fixed_half_xy` | box 半宽 | voxel |
| $c_x = \lfloor \mu_x \rfloor$ | splat anchor | integer |

## 附录 B：关键闭式

**Gaussian 1D moments**（在 $g_z(z) = e^{-\tfrac{1}{2}((z-\mu)/\sigma)^2}$ 下）：

$$
\mathcal{M}_0(a, b) = \sigma\sqrt{\tfrac{\pi}{2}}\Big[\mathrm{erf}\big(\tfrac{b-\mu}{\sigma\sqrt{2}}\big) - \mathrm{erf}\big(\tfrac{a-\mu}{\sigma\sqrt{2}}\big)\Big]
$$

$$
\mathcal{M}_1(a, b) = \mu\,\mathcal{M}_0(a, b) + \sigma^2\big[g_z(a) - g_z(b)\big]
$$

$$
\mathcal{M}_{n+1}(a, b) = \mu\,\mathcal{M}_n + n\sigma^2\mathcal{M}_{n-1} + \sigma^2\big[a^n g_z(a) - b^n g_z(b)\big]
$$

**Tent 卷积权重**（Phase 2 核心）：

$$
w_k(\mu, \sigma) = (\mu - k + 1)\mathcal{M}_0(k\!-\!1, k) + (k + 1 - \mu)\mathcal{M}_0(k, k\!+\!1) + \sigma^2 \Delta g_z(k)
$$

**Gaussian 连续 2D FT**（Phase 1 核心）：

$$
\widehat{g}(f_x, f_y) = \rho\,(2\pi\sigma_x\sigma_y) \cdot e^{-2\pi^2(\sigma_x^2 f_x^2 + \sigma_y^2 f_y^2)} \cdot e^{-2\pi i(f_x\mu_x + f_y\mu_y)}
$$

**位移定理**：$\mathcal{F}[g(\mathbf{x} - \boldsymbol{\delta})] = e^{-2\pi i \mathbf{f}\cdot\boldsymbol{\delta}}\,\widehat{g}(\mathbf{f})$

**Parseval（守恒校验）**：$\sum_k w_k = \int g_z = \sigma\sqrt{2\pi}$。

---

## 附录 C：训练稳定性 — 已知失败模式与修复

> 来源：500-iter smoke 测试（`rho_bias=0`、`rho_max=20`、`soft_temp=0.001` 默认）。前 161 iter 正常，iter 161 ρmax 顶到 cap=20.000，iter 162 ρmax=19.795，iter 163 backward 报错。

### C.1 错误现象

```
Epoch:[1/80] Iters:[161/1618] ... ρmax=20.000
Epoch:[1/80] Iters:[162/1618] ... ρmax=19.795
Traceback (most recent call last):
  File "train_model.py", line 335, in train
    loss.backward()
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

注意：loss 数值有限（MSE=0.075），所以**不是 NaN**——`commit 5fe4d81` 加的 NaN guard 抓不到。失败的不是 loss **数值**，而是 loss **autograd 图结构**断了。

### C.2 根因链

```
ρ 撞 cap (rho_max=20)
     │
     ▼
soft_filter 里 sigmoid((ρ-threshold)/soft_temp=0.001) 饱和到 1.0
导数 sig·(1-sig) = 0  →  ρ 路径梯度被清零
     │
     ▼
网络只能靠 position / sigma 的梯度推动
     │
     ▼
position decoder 漂移（或上游 activation 溢出产生 NaN）
     │
     ▼
所有 Gaussian 的 anchor (cx_idx, cy_idx) 落到 [0, H) × [0, W) 之外
     │
     ▼
_splat 的 `if i1_img <= i0_img or j1_img <= j0_img: return` 全部命中
out 张量从未被 in-place 修改
     │
     ▼
hybrid_renderer.forward() 返回的是 line 261 的 torch.zeros(...) leaf
该 tensor requires_grad=False, grad_fn=None
     │
     ▼
loss = MSE(zeros_leaf, target) + ...   ← 数值有限，但无 grad
loss.backward() 报错
```

关键洞察：**NaN 是数值问题，graph 断裂是结构问题**，它们需要不同的防护层。

### C.3 已应用修复（防御层 / 已 patched）

`hybrid_renderer.py::HybridRenderer.forward()` 末尾加 ghost-grad guard：

```python
zero_link = 0.0 * (
    gaussians.positions.sum()
    + gaussians.sigmas.sum()
    + gaussians.rhos.sum()
)
return out + zero_link
```

性质：

- **数值不变**：乘 0 不改 `out`。
- **图保持连通**：`out` 现在通过 `+ 0 * input.sum()` 与 Gaussian 参数挂上 autograd 链。
- **退化 batch 收到 0 梯度而不是 crash**：训练能继续，grad_clip / NaN guard / optimizer 后续兜底。

这是**纯防御**——只保证 backward 不死，不解决根因。

### C.4 建议参数调整（治根，未默认改动）

ghost-grad 只防止 crash，不解决"ρ 饱和 → 训练失稳"。建议同时调整训练 flag：

| flag | 当前默认 | 建议 | 原因 |
|---|---|---|---|
| `--rho_max` | 20.0 | **5.0** | 真实 target 强度 max ≈ 3；cap=5 已够大。cap 越低 → 饱和概率越低 → 梯度死区越窄 |
| `--hybrid_soft_temp` | 0.001 | **0.05–0.1** | 当前 temp 太硬，sigmoid 接近 step。提高 temp 让 gate 在阈值附近有可观梯度带宽 |
| `--gauss_lambda_sparse` | 0 | **1e-4** | 主动 L2 把大 ρ 拉回，比 clamp 被动 cap 更平滑 |

（默认值**未改动**，以免影响在跑实验；上述值在新一轮 long training 时手动加 CLI 即可。）

### C.5 诊断建议

复现时若想确认根因，在 `_render_chunk` 末尾加：

```python
if not torch.isfinite(mu).all():
    print(f'[warn] non-finite mu in chunk: has_nan={torch.isnan(mu).any().item()}')
elif (mu[:, 0].max() >= self.H + 10) or (mu[:, 1].max() >= self.W + 10):
    print(f'[warn] mu drift OOB: x=[{mu[:,0].min():.1f}, {mu[:,0].max():.1f}], '
          f'y=[{mu[:,1].min():.1f}, {mu[:,1].max():.1f}]')
```

- `has_nan=True` → 上游 decoder 溢出，问题在 position head 的 tanh/激活
- `OOB 但 finite` → max_offset 配错，或 ρ 饱和反推 position 飞出

### C.6 与之前 commit 的关系

`commit 5fe4d81` 加了三层防护：grad_clip / rho_max / NaN guard。这次发现的 graph-break 是**第四层**没覆盖的失败模式：

| 失败模式 | 哪层防护抓 |
|---|---|
| 梯度爆炸 | grad_clip ✓ |
| ρ 渲染溢出 | rho_max ✓ |
| loss 变 NaN | NaN guard ✓ |
| **loss 没 grad_fn**（图断）| **ghost-grad（本次修复）** ✓ |

至此四层防护构成完整的训练稳定性 envelope：数值层（前三层）+ 结构层（第四层）。

---

## 附录 D：与现有分支与文档的对应

| 文档 / 分支 | 对应内容 |
|---|---|
| `V2V3D_Hybrid_Renderer_Implementation_Guide_v2.md` | 上游：定义了 raw_exact / centered_affine / bilinear 三阶段路线 |
| `PSF_Affine_Channel_Failure_Report.md` | 排除路线 B（PSF Gaussian fit），证明 PSF 不能闭式高斯化，是本路线选择"保留 raw PSF + 数学连续化包装"的物理依据 |
| `gauss-hybrid-renderer` 分支 | raw_exact (Phase 1) + centered_affine 实现 + 训练接入 + 稳定性补丁 |
| `gauss-continuous-fourier` 分支（本文档对应） | continuous_fourier Phase 1（xy 解析 Fourier）已落地；Phase 2（z 连续化）规划在此 |
| `sanity_check_continuous.py` | Phase 1 验证（T1-T4 已 PASS） |
| 待新增 `sanity_check_zlinear.py` | Phase 2 验证（C1-C6） |
