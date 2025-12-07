"""
StarNet with FFT Parallel Branch (Scheme A)
- 局部分支：DW7x7 → 1x1扩张(f1,f2) → ReLU6门控相乘 → 1x1压回+BN → DW7x7
- 频域分支：1x1压回+BN → rfft2 → 低频复数仿射 → irfft2 → BN
- 两分支相加（带可学习缩放 alpha），再做残差 + DropPath
"""

import torch
import torch.nn as nn

# ---- timm 依赖的安全导入（带简易回退） ----
try:
    from timm.models.layers import DropPath, trunc_normal_
except Exception:
    from torch.nn.init import trunc_normal_
    class DropPath(nn.Module):
        def __init__(self, drop_prob: float = 0.):
            super().__init__()
            self.drop_prob = float(drop_prob)
        def forward(self, x):
            if self.drop_prob == 0. or not self.training:
                return x
            keep_prob = 1.0 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = x.new_empty(shape).bernoulli_(keep_prob) / keep_prob
            return x * random_tensor

__all__ = ['starnet_s1', 'starnet_s2', 'starnet_s3', 'starnet_s4',
           'starnet_s050', 'starnet_s100', 'starnet_s150']

model_urls = {
    "starnet_s1": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s1.pth.tar",
    "starnet_s2": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s2.pth.tar",
    "starnet_s3": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s3.pth.tar",
    "starnet_s4": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s4.pth.tar",
}


# ----------------------------
# 基础模块：Conv + (可选) BN
# ----------------------------
class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes,
                 kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_planes, out_planes, kernel_size,
                                          stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', nn.BatchNorm2d(out_planes))
            nn.init.constant_(self.bn.weight, 1.0)
            nn.init.constant_(self.bn.bias, 0.0)


# ----------------------------
# 频域混合：对低频子块做复数仿射
# ----------------------------
class SpectralMix(nn.Module):
    def __init__(self, channels, modes=(16, 16), norm='ortho'):
        super().__init__()
        self.c = channels
        self.mh, self.mw = modes
        self.norm = norm
        self.wr = nn.Parameter(torch.ones(channels, self.mh, self.mw))
        self.wi = nn.Parameter(torch.zeros(channels, self.mh, self.mw))
        self.br = nn.Parameter(torch.zeros(channels, self.mh, self.mw))
        self.bi = nn.Parameter(torch.zeros(channels, self.mh, self.mw))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C == self.c, f"SpectralMix: channel mismatch C={C} vs {self.c}"

        # --- 关键：频域分支强制用 float32 计算，避开 fp16+非2幂 的 cuFFT 限制 ---
        orig_dtype = x.dtype
        x32 = x.float()  # 升精度到 fp32（不会破坏梯度）
        # rFFT2: (B,C,H,W) -> (B,C,H,W//2+1) complex64
        X = torch.fft.rfft2(x32, dim=(-2, -1), norm=self.norm)
        Wc = W // 2 + 1

        kh = min(H, self.mh)
        kw = min(Wc, self.mw)

        Wr = self.wr[:, :kh, :kw]
        Wi = self.wi[:, :kh, :kw]
        Br = self.br[:, :kh, :kw]
        Bi = self.bi[:, :kh, :kw]

        W_complex = torch.complex(Wr, Wi).unsqueeze(0)  # complex64
        B_complex = torch.complex(Br, Bi).unsqueeze(0)

        Y = X.clone()
        Y[:, :, :kh, :kw] = X[:, :, :kh, :kw] * W_complex + B_complex

        y32 = torch.fft.irfft2(Y, s=(H, W), dim=(-2, -1), norm=self.norm)  # fp32
        y = y32.to(orig_dtype)  # 转回与输入一致的 dtype（fp16/bf16/fp32）
        return y



# ----------------------------
# Star Block：并联频域分支
# ----------------------------
class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.,
                 use_fft=False, fft_modes=(16, 16)):
        super().__init__()
        # 主干的首个深度卷积
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        # 两个 1x1 扩张分支（无 BN）
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)

        # 局部分支（原路径）
        self.g_local = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)

        # 频域分支（并联）
        self.use_fft = bool(use_fft)
        if self.use_fft:
            self.g_fft = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
            self.fft_mix = SpectralMix(dim, modes=fft_modes)
            self.bn_fft = nn.BatchNorm2d(dim)
            self.alpha = nn.Parameter(torch.tensor(1e-3))  # 小值初始化，稳定训练

        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)                # (B,C,H,W)
        x1, x2 = self.f1(x), self.f2(x)   # (B,rC,H,W)
        z = self.act(x1) * x2             # 门控相乘

        # 局部分支
        local = self.dwconv2(self.g_local(z))

        if self.use_fft:
            # 频域分支
            global_ = self.g_fft(z)
            global_ = self.fft_mix(global_)   # rfft2→低频仿射→irfft2
            global_ = self.bn_fft(global_)
            out = local + self.alpha * global_
        else:
            out = local

        out = shortcut + self.drop_path(out)
        return out


# ----------------------------
# 工具函数
# ----------------------------
def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# ----------------------------
# StarNet 主体
# ----------------------------
class StarNet(nn.Module):
    """
    返回四尺度特征 (list of tensors)，适合检测/分割等任务。
    关键参数:
      - width:     通道缩放
      - base_dim:  第一层 stage 的基通道，后续按 2^i 递增
      - depths:    每个 stage 的 block 数量
      - use_fft_stages: 长度=4 的布尔元组，控制各 stage 是否启用 FFT 分支
      - fft_modes:  频域低频子块大小 (mh, mw)
    """
    def __init__(self, depth=0.25, width=0.5, base_dim=32,
                 depths=(3, 3, 12, 5), mlp_ratio=4, drop_path_rate=0.0,
                 use_fft_stages=(False, True, True, False), fft_modes=(16, 16),
                 num_classes=1000, **kwargs):
        super().__init__()
        self.num_classes = num_classes

        # 通道尺度
        base_dim = _make_divisible(int(base_dim * width), 8)
        self.in_channel = _make_divisible(int(32 * width), 8)

        # stem
        self.stem = nn.Sequential(
            ConvBN(3, self.in_channel, kernel_size=3, stride=2, padding=1),
            nn.ReLU6()
        )

        # DropPath 线性递增
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # 构建 stages
        self.stages = nn.ModuleList()
        cur = 0
        for i_layer, n_blocks in enumerate(depths):
            embed_dim = base_dim * (2 ** i_layer)
            down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
            self.in_channel = embed_dim

            blocks = []
            for i in range(n_blocks):
                blocks.append(
                    Block(self.in_channel, mlp_ratio, dpr[cur + i],
                          use_fft=bool(use_fft_stages[i_layer]),
                          fft_modes=fft_modes)
                )
            cur += n_blocks
            self.stages.append(nn.Sequential(down_sampler, *blocks))

        # 参数初始化修正：正确的 isinstance 写法
        self.apply(self._init_weights)

        # 直接解析得到每个 stage 的通道数，避免 forward 污染 BN
        self.width_list = [base_dim * (2 ** i) for i in range(len(depths))]

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias, 0.0)
            if getattr(m, "weight", None) is not None:
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        feats_by_size = {}
        for stage in self.stages:
            x = stage(x)
            H, W = x.shape[2], x.shape[3]
            feats_by_size[(H, W)] = x  # 用空间尺度做 key，基本不会冲突
        # 取最后 4 个尺度
        return list(feats_by_size.values())[-4:]


# ----------------------------
# 各种规格的工厂函数
# ----------------------------
def _load_pretrained(model, url: str):
    # 新增分支会导致严格匹配失败，这里用 strict=False 兼容加载可匹配的权重
    checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[StarNet] Partially loaded pretrained weights. "
              f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    return model

def starnet_s1(depth=0.5, width=0.25, pretrained=False, **kwargs):
    model = StarNet(depth=depth, width=width, base_dim=24, depths=(2, 2, 8, 3), **kwargs)
    if pretrained:
        url = model_urls['starnet_s1']
        model = _load_pretrained(model, url)
    return model

def starnet_s2(depth=0.5, width=0.25, pretrained=False, **kwargs):
    model = StarNet(depth=depth, width=width, base_dim=32, depths=(1, 2, 6, 2), **kwargs)
    if pretrained:
        url = model_urls['starnet_s2']
        model = _load_pretrained(model, url)
    return model

def starnet_s3(depth=0.5, width=0.25, pretrained=False, **kwargs):
    model = StarNet(depth=depth, width=width, base_dim=32, depths=(2, 2, 8, 4), **kwargs)
    if pretrained:
        url = model_urls['starnet_s3']
        model = _load_pretrained(model, url)
    return model

def starnet_s4(depth=0.5, width=0.25, pretrained=False, **kwargs):
    model = StarNet(depth=depth, width=width, base_dim=32, depths=(3, 3, 12, 5), **kwargs)
    if pretrained:
        url = model_urls['starnet_s4']
        model = _load_pretrained(model, url)
    return model

def starnet_s050(depth=0.5, width=0.25, pretrained=False, **kwargs):
    return StarNet(depth=depth, width=width, base_dim=16, depths=(1, 1, 3, 1), mlp_ratio=3, **kwargs)

def starnet_s100(depth=0.5, width=0.25, pretrained=False, **kwargs):
    return StarNet(depth=depth, width=width, base_dim=20, depths=(1, 2, 4, 1), mlp_ratio=4, **kwargs)

def starnet_s150(depth=0.5, width=0.25, pretrained=False, **kwargs):
    return StarNet(depth=depth, width=width, base_dim=24, depths=(1, 2, 4, 2), mlp_ratio=3, **kwargs)


# ----------------------------
# 简单自测
# ----------------------------
if __name__ == "__main__":
    # 默认在中间两个 stage 开启 FFT 分支
    model = starnet_s1(use_fft_stages=(False, True, True, False), fft_modes=(16,16))
    inputs = torch.randn(1, 3, 640, 640)
    outs = model(inputs)
    for i, t in enumerate(outs):
        print(f"Stage {i}: {tuple(t.shape)}")
    print("width_list:", getattr(model, "width_list", None))
