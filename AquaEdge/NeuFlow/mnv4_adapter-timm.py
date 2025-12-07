#Timm MobileNetV4

import torch
import torch.nn as nn
import timm


class MobileNetV4Encoder(nn.Module):
    """
    作为 CNNEncoder 的替代：
      forward(x_cat) -> (x_16_with_pos, x_8)

    - x_16: [context_s16 , feature_s16_minus2 , pos(2)]
    - x_8 : [context_s8  , feature_s8]
    """
    def __init__(
        self,
        feature_dim_s16, context_dim_s16,
        feature_dim_s8,  context_dim_s8,
        timm_name: str = "mobilenetv4_conv_small.e3600_r256_in1k",
        pretrained: bool = True,
        imagenet_norm: bool = True,
    ):
        super().__init__()
        self.feature_dim_s16 = feature_dim_s16
        self.context_dim_s16 = context_dim_s16
        self.feature_dim_s8  = feature_dim_s8
        self.context_dim_s8  = context_dim_s8
        self.imagenet_norm   = imagenet_norm

        # 1) timm 特征骨干：一次性导出所有 stage（0..4），后面按 reduction 选 s8/s16
        self.backbone = timm.create_model(
            timm_name, features_only=True, pretrained=pretrained, out_indices=(0, 1, 2, 3, 4)
        )
        # reductions 例如 [2, 4, 8, 16, 32]
        self.reductions = list(self.backbone.feature_info.reduction())

        # 找到最接近 8/16 下采样的索引
        self.idx_s8  = self._find_reduction_index(target=8)
        self.idx_s16 = self._find_reduction_index(target=16)

        # 2) ImageNet 归一化 buffer
        if self.imagenet_norm:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer("img_mean", mean, persistent=False)
            self.register_buffer("img_std",  std,  persistent=False)

        # 3) s16 位置编码（由 init_bhwd 填充）
        self.register_buffer("pos_s16", None, persistent=False)

        # 4) 懒构建 1×1 投影
        self._proj_ready = False
        self.proj_s8_context  = None
        self.proj_s8_feature  = None
        self.proj_s16_context = None
        self.proj_s16_feat_m2 = None

    def _find_reduction_index(self, target: int) -> int:
        # 找到 reduction 最接近 target(8/16) 的索引
        diffs = [abs(r - target) for r in self.reductions]
        return int(min(range(len(diffs)), key=lambda i: diffs[i]))

    @torch.no_grad()
    def init_bhwd(self, batch_size, h16, w16, device, amp=True):
        dtype = torch.half if amp else torch.float
        ys, xs = torch.meshgrid(
            torch.arange(h16, dtype=dtype, device=device),
            torch.arange(w16, dtype=dtype, device=device),
            indexing="ij",
        )
        ys = ys - h16 / 2
        xs = xs - w16 / 2
        pos = torch.stack([ys, xs], dim=0)  # [2, H16, W16]
        self.pos_s16 = pos.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [B*2, 2, H16, W16]

    def _build_projs_if_needed(self, c8_in: int, c16_in: int, device, dtype=torch.float32):
        if self._proj_ready:
            # 兜底：确保都在正确 device/dtype
            for m in (self.proj_s8_context, self.proj_s8_feature,
                      self.proj_s16_context, self.proj_s16_feat_m2):
                if m is not None and (m.weight.device != device or m.weight.dtype != dtype):
                    m.to(device=device, dtype=dtype)
            return

        self.proj_s8_context  = nn.Conv2d(c8_in,  self.context_dim_s8,  kernel_size=1, bias=False)
        self.proj_s8_feature  = nn.Conv2d(c8_in,  self.feature_dim_s8,  kernel_size=1, bias=False)
        self.proj_s16_context = nn.Conv2d(c16_in, self.context_dim_s16, kernel_size=1, bias=False)
        self.proj_s16_feat_m2 = nn.Conv2d(c16_in, self.feature_dim_s16 - 2, kernel_size=1, bias=False)

        for m in (self.proj_s8_context, self.proj_s8_feature,
                  self.proj_s16_context, self.proj_s16_feat_m2):
            m.to(device=device, dtype=dtype)  # 保持 FP32 权重更稳

        self._proj_ready = True

    def forward(self, x_cat: torch.Tensor):
        # (1) 可选 ImageNet 归一化
        if self.imagenet_norm:
            mean = self.img_mean.to(dtype=x_cat.dtype, device=x_cat.device)
            std  = self.img_std.to(dtype=x_cat.dtype, device=x_cat.device)
            x_bn = (x_cat - mean) / std
        else:
            x_bn = x_cat

        # (2) 主干前向，得到多尺度特征
        feats = self.backbone(x_bn)  # list[T]，与 out_indices 对齐
        f8_raw  = feats[self.idx_s8]
        f16_raw = feats[self.idx_s16]

        # (3) 懒构建 1×1 投影（放到与特征相同的 device，权重保持 FP32）
        self._build_projs_if_needed(
            c8_in=f8_raw.shape[1], c16_in=f16_raw.shape[1],
            device=f8_raw.device, dtype=torch.float32
        )

        # (4) 将输入特征转到投影层权重 dtype，避免 Half/Float 冲突
        proj_dtype = self.proj_s8_context.weight.dtype  # 预期是 float32
        f8  = f8_raw.to(dtype=proj_dtype)
        f16 = f16_raw.to(dtype=proj_dtype)

        # (5) s8 输出：[context, feature]
        ctx8  = self.proj_s8_context(f8)
        feat8 = self.proj_s8_feature(f8)
        x_8   = torch.cat([ctx8, feat8], dim=1)

        # (6) s16 输出：[context, feature(-2), pos(2)]
        ctx16    = self.proj_s16_context(f16)
        feat16m2 = self.proj_s16_feat_m2(f16)

        if self.pos_s16 is None:
            raise RuntimeError("pos_s16 is None，请先调用 init_bhwd(batch, H/16, W/16, ...)")

        pos = self.pos_s16.to(dtype=ctx16.dtype, device=ctx16.device)
        x_16 = torch.cat([ctx16, feat16m2, pos], dim=1)

        return x_16, x_8
