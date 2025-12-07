import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CSPPC"]

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    default_act = nn.SiLU()
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=bias)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    def forward_fuse(self, x):
        return self.act(self.conv(x))

class ECA(nn.Module):
    """Efficient Channel Attention：全局池化 + 1D conv，极轻量"""
    def __init__(self, c, k_size=None):
        super().__init__()
        if k_size is None:
            t = int(abs((math.log2(c) + 1) / 2))
            k_size = max(3, t * 2 + 1)  # 保证奇数
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
    def forward(self, x):
        y = F.adaptive_avg_pool2d(x, 1)                 # (B,C,1,1)
        y = self.conv1d(y.squeeze(-1).transpose(1, 2))  # (B,1,C)->1D conv
        y = torch.sigmoid(y.transpose(1, 2).unsqueeze(-1))
        return x * y

class Partial_Conv3(nn.Module):
    """
    在 dim//n_div 的“部分通道”上做：depthwise 3x3 (d=1) 与 (d=2) 并行，再 1x1 融合；
    其余通道旁路不变。仅引入“多空洞深度卷积”，不改变其它策略。
    """
    def __init__(self, dim, n_div=4):
        super().__init__()
        self.dim_conv = dim // n_div
        self.dim_keep = dim - self.dim_conv
        # 两条并行的深度卷积分支
        self.dw_d1 = nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, autopad(3, d=1),
                               groups=self.dim_conv, dilation=1, bias=False)
        self.dw_d2 = nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, autopad(3, d=2),
                               groups=self.dim_conv, dilation=2, bias=False)
        self.bn = nn.BatchNorm2d(self.dim_conv)
        # 1x1 点卷积轻量融合（提供通道间线性混合）
        self.pw_fuse = Conv(self.dim_conv, self.dim_conv, k=1, act=True)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_keep], dim=1)
        y = self.dw_d1(x1) + self.dw_d2(x1)
        y = F.silu(self.bn(y), inplace=False)
        y = self.pw_fuse(y)
        return torch.cat((y, x2), dim=1)

class CSPPC_Bottleneck(nn.Module):
    """保持与原逻辑一致：两次“部分卷积”（此处为多空洞深度卷积版）串联"""
    def __init__(self, dim, n_div=4):
        super().__init__()
        self.p1 = Partial_Conv3(dim, n_div=n_div)
        self.p2 = Partial_Conv3(dim, n_div=n_div)
    def forward(self, x):
        return self.p2(self.p1(x))

class CSPPC(nn.Module):
    """
    CSPPC-MDCA：
    - 仍采用 CSP 的分流-多级聚合；
    - 将 Partial_conv3 换成“多空洞深度卷积”版本；
    - 在最终拼接 ((2+n)*c) 后引入 ECA 注意力再做 1x1 融合。
    """
    def __init__(self, c1, c2, n=1, e=0.5, n_div=4):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.m = nn.ModuleList(CSPPC_Bottleneck(self.c, n_div=n_div) for _ in range(n))
        self.attn = ECA((2 + n) * self.c)   # 仅此处加入注意力
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

    def forward(self, x):
        y0, y1 = self.cv1(x).split((self.c, self.c), dim=1)
        ys = [y0, y1]
        for block in self.m:
            y1 = block(y1)
            ys.append(y1)
        y = torch.cat(ys, dim=1)   # (B, (2+n)*c, H, W)
        y = self.attn(y)           # 注意力后再 1x1 融合
        return self.cv2(y)

if __name__ == "__main__":
    x = torch.randn(1, 64, 224, 224)
    model = CSPPC(64, 128, n=1, e=0.5, n_div=4)
    y = model(x)
    print(y.shape)  # torch.Size([1, 128, 224, 224])
