"""Intensity voxelizer wrapper — 迁移自 3DGAT 的 us_gaussian_voxelization。

先决条件（一次性）：
    cd /root/autodl-tmp/3DGAT/submodules/us_gaussian_voxelization
    pip install -e .
"""
import torch
import torch.nn as nn


class IntensityVoxelizer(nn.Module):
    """单场景 Gaussian → Volume 渲染器。

    约定：
        - 输入 positions / scales 都在 "voxel 世界坐标" 下（见 gaussian_utils.make_grid_centers）
        - 输出 volume 形状 (D, H, W)，与 V2V3D 原 Unet 输出一致
    """

    def __init__(self, n_slice, H, W, scale_modifier=1.0, debug=False):
        super().__init__()
        from us_gaussian_voxelization import (
            GaussianVoxelizationSettings,
            GaussianVoxelizer,
        )
        self.D, self.H, self.W = n_slice, H, W
        self.settings = GaussianVoxelizationSettings(
            scale_modifier=float(scale_modifier),
            nVoxel_x=int(H),
            nVoxel_y=int(W),
            nVoxel_z=int(n_slice),
            sVoxel_x=float(H),
            sVoxel_y=float(W),
            sVoxel_z=float(n_slice),
            center_x=float(H) / 2.0,
            center_y=float(W) / 2.0,
            center_z=float(n_slice) / 2.0,
            prefiltered=False,
            debug=bool(debug),
        )
        self._vox = GaussianVoxelizer(voxel_settings=self.settings)

    def forward(self, positions, densities, scales, rotations):
        """
        Args:
            positions: (N, 3)
            densities: (N, 1)
            scales:    (N, 3)
            rotations: (N, 4) unit quaternion
        Returns:
            volume: (D, H, W)   -- 与 V2V3D 原 Unet 输出对齐
            radii:  (N,) 可见性半径（可忽略）
        """
        fields, radii = self._vox(
            means3D=positions,
            opacities=densities,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,
        )
        # fields 形状按 (nVoxel_x, nVoxel_y, nVoxel_z) = (H, W, D)
        # 需要 permute 成 (D, H, W) 对齐原 V2V3D volume 约定
        volume = fields.permute(2, 0, 1).contiguous()
        return volume, radii
