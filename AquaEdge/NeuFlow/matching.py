# NeuFlow/matching.py
import torch
import torch.nn.functional as F
from NeuFlow import utils


class Matching:
    """
    计算全局相关并用 softmax 得到对应点期望坐标：
      - 输入 feature0/feature1: [B, C, H, W]
      - 内部将其展平为 [B, HW, C] 分别作为 Q / K
      - 将预先构建的坐标网格 flatten 为 [B, HW, 2] 作为 V
      - 使用 scaled_dot_product_attention(Q, K, V) 得到 [B, HW, 2] 的坐标期望
      - 减去原始网格得到 flow: [B, 2, H, W]
    重点修复：在调用 SDPA 前，强制将 K 与 V 的 dtype / device 与 Q 对齐，避免 AMP 下的半精/单精混用报错。
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = float(temperature)
        self.grid = None          # [B, 2, H, W]
        self.flatten_grid = None  # [B, HW, 2]
        self._hw = None           # (H, W)

    @torch.no_grad()
    def init_bhwd(self, batch_size: int, height: int, width: int, device, amp: bool):
        """
        构建坐标网格：
          - amp=True 时用 FP16，否则 FP32（与训练/推理路径保持一致）
          - 同时缓存扁平化后的网格，便于当做 V 使用
        """
        self._hw = (int(height), int(width))

        self.grid = utils.coords_grid(batch_size, height, width, device, amp)  # [B, 2, H, W]
        # 保证内存布局 & 连续性
        self.grid = self.grid.contiguous()
        # 展平为 [B, HW, 2]
        self.flatten_grid = (
            self.grid.view(batch_size, 2, -1)
                .permute(0, 2, 1)
                .contiguous()
        )

    def _maybe_rebuild_grid(self, b: int, h: int, w: int, device, dtype):
        """
        当分辨率或 batch 大小变化时，自动按当前 dtype/device 重建网格，避免后续注意力里 dtype 不匹配。
        """
        need_rebuild = (
            self.grid is None or
            self.flatten_grid is None or
            self._hw is None or
            self._hw != (h, w) or
            self.grid.device != device or
            self.grid.dtype != dtype or
            self.grid.shape[0] != b
        )
        if need_rebuild:
            # 根据 dtype 判断 amp 布尔（半精认为 True）
            amp = (dtype in (torch.float16, torch.bfloat16))
            self.init_bhwd(b, h, w, device, amp)

    def global_correlation_softmax(self, feature0: torch.Tensor, feature1: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature0: [B, C, H, W]  -> 作为 Query
            feature1: [B, C, H, W]  -> 作为 Key
        Returns:
            flow: [B, 2, H, W]
        """
        assert feature0.ndim == 4 and feature1.ndim == 4, "Expect 4D tensors [B,C,H,W]"
        b, c, h, w = feature0.shape
        assert feature1.shape[:3] == (b, c, h), "feature0/feature1 must match shape except W"
        assert feature1.shape[3] == w, "feature0/feature1 must have same W"

        # 若分辨率或 dtype/device 发生变化，重建网格（grid / flatten_grid）
        self._maybe_rebuild_grid(b, h, w, feature0.device, feature0.dtype)

        # 展平成 [B, HW, C]
        # 注意：保持连续、确保 dtype/device 与后续一致
        q = feature0.flatten(-2).permute(0, 2, 1).contiguous()
        k = feature1.flatten(-2).permute(0, 2, 1).contiguous()
        v = self.flatten_grid

        # ⭐ 关键修复：K 与 V 的 dtype / device 与 Q 完全对齐（以 Q 为准）
        if k.dtype != q.dtype or k.device != q.device:
            k = k.to(dtype=q.dtype, device=q.device)
        if v.dtype != q.dtype or v.device != q.device:
            v = v.to(dtype=q.dtype, device=q.device)

        # 可选温度缩放
        if self.temperature != 1.0:
            q = q / self.temperature

        # PyTorch 2.x 的 SDPA：返回 [B, HW, Dv]，这里 Dv=2（来自 flatten_grid 的最后维度）
        # 如果你的 PyTorch 版本强制要求 Q/K/V 的最后维度一致，请升级到 2.1+；
        # 或者自行实现 attn = softmax(QK^T/sqrt(d_k)) @ V 以支持 Dv != C 的情形。
        attn_out = F.scaled_dot_product_attention(q, k, v)  # [B, HW, 2]
        attn_out = attn_out.view(b, h, w, 2).permute(0, 3, 1, 2).contiguous()  # [B, 2, H, W]

        # flow = E[coord] - coord
        flow = attn_out - self.grid  # [B, 2, H, W]，dtype/device 与 Q 一致
        return flow
