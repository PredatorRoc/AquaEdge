# NeuFlow/mnv4_adapter.py -- MobileNetV4 adapter (robust to AMP & pos size)
import math
import torch
import torch.nn as nn
from typing import Tuple, Optional

from .MNV4 import (
    MobileNetV4ConvSmall,
    MobileNetV4ConvMedium,
    MobileNetV4ConvLarge,
)

__all__ = ["MobileNetV4Encoder"]


# 可选主干
_MNV4_FACTORY = {
    "MobileNetV4ConvSmall":  MobileNetV4ConvSmall,
    "MobileNetV4ConvMedium": MobileNetV4ConvMedium,
    "MobileNetV4ConvLarge":  MobileNetV4ConvLarge,
}


class MobileNetV4Encoder(nn.Module):
    """
    作为 NeuFlow 的 backbone 适配器，导出两路特征：
        forward(x_cat) -> (x_16_with_pos, x_8)

    - 输入:
        x_cat: [B*2, 3, H, W]  (外部通常把两帧在 dim=0 拼接)
    - 输出:
        x_16_with_pos: [B*2, context_dim_s16 + feature_dim_s16, H/16, W/16]
                       = [context, feature(-2), pos(2)]
        x_8          : [B*2, context_dim_s8  + feature_dim_s8 , H/8 , W/8 ]

    设计要点：
      * 投影层保持 FP32，更稳定；特征在投影前自动转换到同 dtype。
      * 位置编码不依赖外部传入的 batch，大/小误传都不会炸；forward 内对齐当前特征的 B,H16,W16。
      * 支持 ImageNet 归一化（按输入 dtype 运算，兼容 AMP）。
    """

    def __init__(
        self,
        feature_dim_s16: int,
        context_dim_s16: int,
        feature_dim_s8: int,
        context_dim_s8: int,
        model_name: str = "MobileNetV4ConvSmall",
        imagenet_norm: bool = True,
    ):
        super().__init__()
        self.feature_dim_s16 = feature_dim_s16
        self.context_dim_s16 = context_dim_s16
        self.feature_dim_s8  = feature_dim_s8
        self.context_dim_s8  = context_dim_s8
        self.imagenet_norm   = imagenet_norm

        if model_name not in _MNV4_FACTORY:
            raise ValueError(
                f"Unsupported model_name={model_name}. "
                f"Choose from {list(_MNV4_FACTORY.keys())}."
            )

        # 1) 主干：forward 需返回 [F4, F8, F16, F32]，至少包含 F8/F16
        self.backbone = _MNV4_FACTORY[model_name]()

        # 2) ImageNet 归一化 buffer（在 forward 时以输入 dtype 进行）
        if self.imagenet_norm:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer("img_mean", mean, persistent=False)
            self.register_buffer("img_std",  std,  persistent=False)

        # 3) 懒构建的 1×1 投影 (保持 FP32)
        self._proj_ready: bool = False
        self.proj_s8_context: Optional[nn.Conv2d]  = None
        self.proj_s8_feature: Optional[nn.Conv2d]  = None
        self.proj_s16_context: Optional[nn.Conv2d] = None
        self.proj_s16_feat_m2: Optional[nn.Conv2d] = None

        # 4) 位置编码缓存（只缓存基础网格 [1,2,h16,w16]；batch 在 forward 扩充）
        self._pos_base: Optional[torch.Tensor] = None     # [1,2,h16,w16] (float32)
        self._pos_hw:   Tuple[int, int] = (-1, -1)        # (h16,w16)

    # -------------------------
    # 运行期尺寸初始化（向后兼容）
    # -------------------------
    @torch.no_grad()
    def init_bhwd(self, batch_size: int, H: int, W: int, device, amp: bool = True):
        """
        仅做“友好缓存”，不依赖该函数保证正确性。
        外界有时传入 H/16,W/16，有时传入 H,W；这里用 ceil(H/16),ceil(W/16) 做一个基础网格。
        真正使用时 forward 里会按当前特征的 H16,W16 动态对齐。
        """
        h16 = math.ceil(H / 16)
        w16 = math.ceil(W / 16)
        self._make_or_update_pos_base(h16, w16, device=device)

    @torch.no_grad()
    def _make_or_update_pos_base(self, h16: int, w16: int, device):
        """创建/更新基础网格 [1,2,h16,w16]（float32），batch 在 forward 时再 repeat。"""
        if self._pos_base is not None and self._pos_hw == (h16, w16):
            # 只移动 device
            if self._pos_base.device != torch.device(device):
                self._pos_base = self._pos_base.to(device=device)
            return

        ys, xs = torch.meshgrid(
            torch.arange(h16, dtype=torch.float32, device=device),
            torch.arange(w16, dtype=torch.float32, device=device),
            indexing="ij",
        )
        ys = ys - h16 / 2.0
        xs = xs - w16 / 2.0
        pos = torch.stack([ys, xs], dim=0).unsqueeze(0)  # [1,2,h16,w16]
        self._pos_base = pos
        self._pos_hw = (h16, w16)

    def _ensure_proj(self, c8_in: int, c16_in: int, device, dtype=torch.float32):
        """懒构建 1×1 投影层，并固定为 FP32；必要时确保 device/dtype 对齐。"""
        if not self._proj_ready:
            self.proj_s8_context  = nn.Conv2d(c8_in,  self.context_dim_s8,  kernel_size=1, bias=False)
            self.proj_s8_feature  = nn.Conv2d(c8_in,  self.feature_dim_s8,  kernel_size=1, bias=False)
            self.proj_s16_context = nn.Conv2d(c16_in, self.context_dim_s16, kernel_size=1, bias=False)
            self.proj_s16_feat_m2 = nn.Conv2d(c16_in, self.feature_dim_s16 - 2, kernel_size=1, bias=False)

            for m in (self.proj_s8_context, self.proj_s8_feature,
                      self.proj_s16_context, self.proj_s16_feat_m2):
                m.to(device=device, dtype=dtype)

            self._proj_ready = True
            return

        # 已构建过，确保 device/dtype 正确
        for m in (self.proj_s8_context, self.proj_s8_feature,
                  self.proj_s16_context, self.proj_s16_feat_m2):
            if m is not None and (m.weight.device != torch.device(device) or m.weight.dtype != dtype):
                m.to(device=device, dtype=dtype)

    # -------------------------
    # 前向
    # -------------------------
    def forward(self, x_cat: torch.Tensor):
        """
        x_cat: [B*2, 3, H, W], 可能是 FP16 (AMP) 或 FP32
        返回:  (x_16_with_pos, x_8)
        """
        # (1) 可选 ImageNet 归一化（按输入 dtype 运算，兼容 AMP）
        if self.imagenet_norm:
            mean = self.img_mean.to(dtype=x_cat.dtype, device=x_cat.device)
            std  = self.img_std.to(dtype=x_cat.dtype, device=x_cat.device)
            x_bn = (x_cat - mean) / std
        else:
            x_bn = x_cat

        # (2) 主干特征（要求含 F8 & F16）
        feats = self.backbone(x_bn)
        if not isinstance(feats, (list, tuple)) or len(feats) < 3:
            raise RuntimeError("Backbone should return [F4, F8, F16, ...]. Got invalid outputs.")

        f8_raw  = feats[1]
        f16_raw = feats[2]
        if f8_raw is None or f16_raw is None:
            raise RuntimeError("Backbone did not produce s8/s16 features. Please check input size or backbone.")

        B2, C8,  H8,  W8  = f8_raw.shape
        _,  C16, H16, W16 = f16_raw.shape

        # (3) 懒构建投影层（保持 FP32），并把特征转换到投影权重 dtype
        self._ensure_proj(C8, C16, device=f8_raw.device, dtype=torch.float32)
        proj_dtype = self.proj_s8_context.weight.dtype  # 预期 float32

        f8  = f8_raw.to(dtype=proj_dtype)
        f16 = f16_raw.to(dtype=proj_dtype)

        # (4) s8: [context, feature]
        ctx8  = self.proj_s8_context(f8)
        feat8 = self.proj_s8_feature(f8)
        x_8   = torch.cat([ctx8, feat8], dim=1)

        # (5) s16: [context, feature(-2), pos(2)]
        ctx16    = self.proj_s16_context(f16)
        feat16m2 = self.proj_s16_feat_m2(f16)

        # 位置编码（自动与当前 B*2, H16, W16 对齐；不依赖外部 batch）
        self._make_or_update_pos_base(H16, W16, device=f16.device)  # _pos_base: [1,2,H16,W16] (float32)

        # 将 pos 转到与 ctx16 相同的 dtype/device，并扩展 batch
        pos = self._pos_base.to(dtype=ctx16.dtype, device=ctx16.device)
        if pos.shape[0] != B2:
            pos = pos.repeat(B2, 1, 1, 1)  # [B*2, 2, H16, W16]

        x_16 = torch.cat([ctx16, feat16m2, pos], dim=1)

        return x_16, x_8
