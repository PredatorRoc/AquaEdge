# from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
#
# mnv4.py
from typing import Optional, List, Tuple, Dict
import torch
import torch.nn as nn

__all__ = [
    'MobileNetV4ConvSmall',
    'MobileNetV4ConvMedium',
    'MobileNetV4ConvLarge',
    'MobileNetV4HybridMedium',
    'MobileNetV4HybridLarge'
]

# -------------------------
# 规格（来自你的给定表）
# -------------------------
MNV4ConvSmall_BLOCK_SPECS = {
    "conv0": {"block_name": "convbn", "num_blocks": 1, "block_specs": [[3, 32, 3, 2]]},
    "layer1": {"block_name": "convbn", "num_blocks": 2, "block_specs": [[32, 32, 3, 2], [32, 32, 1, 1]]},
    "layer2": {"block_name": "convbn", "num_blocks": 2, "block_specs": [[32, 96, 3, 2], [96, 64, 1, 1]]},
    "layer3": {
        "block_name": "uib", "num_blocks": 6,
        "block_specs": [
            [64, 96, 5, 5, True, 2, 3],
            [96, 96, 0, 3, True, 1, 2],
            [96, 96, 0, 3, True, 1, 2],
            [96, 96, 0, 3, True, 1, 2],
            [96, 96, 0, 3, True, 1, 2],
            [96, 96, 3, 0, True, 1, 4],
        ]
    },
    "layer4": {
        "block_name": "uib", "num_blocks": 6,
        "block_specs": [
            [96, 128, 3, 3, True, 2, 6],
            [128, 128, 5, 5, True, 1, 4],
            [128, 128, 0, 5, True, 1, 4],
            [128, 128, 0, 5, True, 1, 3],
            [128, 128, 0, 3, True, 1, 4],
            [128, 128, 0, 3, True, 1, 4],
        ]
    },
    "layer5": {"block_name": "convbn", "num_blocks": 2, "block_specs": [[128, 960, 1, 1], [960, 1280, 1, 1]]}
}

MNV4ConvMedium_BLOCK_SPECS = {
    "conv0": {"block_name": "convbn", "num_blocks": 1, "block_specs": [[3, 32, 3, 2]]},
    "layer1": {"block_name": "fused_ib", "num_blocks": 1, "block_specs": [[32, 48, 2, 4.0, True]]},
    "layer2": {
        "block_name": "uib", "num_blocks": 2,
        "block_specs": [[48, 80, 3, 5, True, 2, 4], [80, 80, 3, 3, True, 1, 2]]
    },
    "layer3": {
        "block_name": "uib", "num_blocks": 8,
        "block_specs": [
            [80, 160, 3, 5, True, 2, 6],
            [160, 160, 3, 3, True, 1, 4],
            [160, 160, 3, 3, True, 1, 4],
            [160, 160, 3, 5, True, 1, 4],
            [160, 160, 3, 3, True, 1, 4],
            [160, 160, 3, 0, True, 1, 4],
            [160, 160, 0, 0, True, 1, 2],
            [160, 160, 3, 0, True, 1, 4],
        ]
    },
    "layer4": {
        "block_name": "uib", "num_blocks": 11,
        "block_specs": [
            [160, 256, 5, 5, True, 2, 6],
            [256, 256, 5, 5, True, 1, 4],
            [256, 256, 3, 5, True, 1, 4],
            [256, 256, 3, 5, True, 1, 4],
            [256, 256, 0, 0, True, 1, 4],
            [256, 256, 3, 0, True, 1, 4],
            [256, 256, 3, 5, True, 1, 2],
            [256, 256, 5, 5, True, 1, 4],
            [256, 256, 0, 0, True, 1, 4],
            [256, 256, 0, 0, True, 1, 4],
            [256, 256, 5, 0, True, 1, 2],
        ]
    },
    "layer5": {"block_name": "convbn", "num_blocks": 2, "block_specs": [[256, 960, 1, 1], [960, 1280, 1, 1]]}
}

MNV4ConvLarge_BLOCK_SPECS = {
    "conv0": {"block_name": "convbn", "num_blocks": 1, "block_specs": [[3, 24, 3, 2]]},
    "layer1": {"block_name": "fused_ib", "num_blocks": 1, "block_specs": [[24, 48, 2, 4.0, True]]},
    "layer2": {
        "block_name": "uib", "num_blocks": 2,
        "block_specs": [[48, 96, 3, 5, True, 2, 4], [96, 96, 3, 3, True, 1, 4]]
    },
    "layer3": {
        "block_name": "uib", "num_blocks": 11,
        "block_specs": [
            [96, 192, 3, 5, True, 2, 4],
            [192, 192, 3, 3, True, 1, 4],
            [192, 192, 3, 3, True, 1, 4],
            [192, 192, 3, 3, True, 1, 4],
            [192, 192, 3, 5, True, 1, 4],
            [192, 192, 5, 3, True, 1, 4],
            [192, 192, 5, 3, True, 1, 4],
            [192, 192, 5, 3, True, 1, 4],
            [192, 192, 5, 3, True, 1, 4],
            [192, 192, 5, 3, True, 1, 4],
            [192, 192, 3, 0, True, 1, 4],
        ]
    },
    "layer4": {
        "block_name": "uib", "num_blocks": 13,
        "block_specs": [
            [192, 512, 5, 5, True, 2, 4],
            [512, 512, 5, 5, True, 1, 4],
            [512, 512, 5, 5, True, 1, 4],
            [512, 512, 5, 5, True, 1, 4],
            [512, 512, 5, 0, True, 1, 4],
            [512, 512, 5, 3, True, 1, 4],
            [512, 512, 5, 0, True, 1, 4],
            [512, 512, 5, 0, True, 1, 4],
            [512, 512, 5, 3, True, 1, 4],
            [512, 512, 5, 5, True, 1, 4],
            [512, 512, 5, 0, True, 1, 4],
            [512, 512, 5, 0, True, 1, 4],
            [512, 512, 5, 0, True, 1, 4],
        ]
    },
    "layer5": {"block_name": "convbn", "num_blocks": 2, "block_specs": [[512, 960, 1, 1], [960, 1280, 1, 1]]}
}

MNV4HybridConvMedium_BLOCK_SPECS: Dict = {}
MNV4HybridConvLarge_BLOCK_SPECS: Dict = {}

MODEL_SPECS = {
    "MobileNetV4ConvSmall": MNV4ConvSmall_BLOCK_SPECS,
    "MobileNetV4ConvMedium": MNV4ConvMedium_BLOCK_SPECS,
    "MobileNetV4ConvLarge": MNV4ConvLarge_BLOCK_SPECS,
    "MobileNetV4HybridMedium": MNV4HybridConvMedium_BLOCK_SPECS,
    "MobileNetV4HybridLarge": MNV4HybridConvLarge_BLOCK_SPECS,
}

# -------------------------
# 基础组件
# -------------------------
def make_divisible(
    value: float, divisor: int, min_value: Optional[float] = None, round_down_protect: bool = True
) -> int:
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if round_down_protect and new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)

def conv_2d(
    inp: int,
    oup: int,
    kernel_size: int = 3,
    stride: int = 1,
    groups: int = 1,
    bias: bool = False,
    norm: bool = True,
    act: bool = True,
    norm_layer=nn.BatchNorm2d,
    act_layer=nn.ReLU6,
) -> nn.Sequential:
    m = nn.Sequential()
    padding = (kernel_size - 1) // 2
    m.add_module('conv', nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=bias, groups=groups))
    if norm:
        m.add_module('bn', norm_layer(oup))
    if act:
        m.add_module('act', act_layer(inplace=True))
    return m

class InvertedResidual(nn.Module):
    """标准 MBConv（用于 fused_ib 配置项）"""
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: float,
        act: bool = False,
        norm_layer=nn.BatchNorm2d,
        act_layer=nn.ReLU6,
    ):
        super().__init__()
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        layers = []
        if expand_ratio != 1:
            layers.append(conv_2d(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, act_layer=act_layer))
        layers.append(conv_2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim,
                              norm_layer=norm_layer, act_layer=act_layer))
        # 末端 1x1 常规不激活（若传入 act=True 则加激活）
        layers.append(conv_2d(hidden_dim, oup, kernel_size=1, act=act,
                              norm_layer=norm_layer, act_layer=act_layer))
        self.block = nn.Sequential(*layers)
        self.use_res_connect = (stride == 1 and inp == oup)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_res_connect:
            out = x + out
        return out

class UniversalInvertedBottleneckBlock(nn.Module):
    """UIB：支持起始/中间深度卷积，可下采样"""
    def __init__(
        self,
        inp: int,
        oup: int,
        start_dw_kernel_size: int,
        middle_dw_kernel_size: int,
        middle_dw_downsample: bool,
        stride: int,
        expand_ratio: float,
        norm_layer=nn.BatchNorm2d,
        act_layer=nn.ReLU6,
    ):
        super().__init__()
        self.has_start_dw = start_dw_kernel_size > 0
        self.has_middle_dw = middle_dw_kernel_size > 0

        # 起始 DW
        if self.has_start_dw:
            stride_ = stride if not middle_dw_downsample else 1
            self.start_dw = conv_2d(inp, inp, kernel_size=start_dw_kernel_size, stride=stride_,
                                     groups=inp, act=False, norm_layer=norm_layer, act_layer=act_layer)

        # 1x1 expand
        expand_filters = make_divisible(inp * expand_ratio, 8)
        self.expand = conv_2d(inp, expand_filters, kernel_size=1,
                              norm_layer=norm_layer, act_layer=act_layer)

        # 中间 DW
        if self.has_middle_dw:
            stride_ = stride if middle_dw_downsample else 1
            self.middle_dw = conv_2d(expand_filters, expand_filters, kernel_size=middle_dw_kernel_size,
                                     stride=stride_, groups=expand_filters,
                                     norm_layer=norm_layer, act_layer=act_layer)

        # 投影 1x1（不激活）
        self.project = conv_2d(expand_filters, oup, kernel_size=1, act=False,
                               norm_layer=norm_layer, act_layer=act_layer)

        self.use_res_connect = (stride == 1 and inp == oup)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if self.has_start_dw:
            x = self.start_dw(x)
        x = self.expand(x)
        if self.has_middle_dw:
            x = self.middle_dw(x)
        x = self.project(x)
        if self.use_res_connect:
            x = x + identity
        return x

# -------------------------
# 组网：根据规格表堆叠模块
# -------------------------
def build_blocks(layer_spec: Dict, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU6) -> nn.Sequential:
    if not layer_spec or not layer_spec.get('block_name'):
        return nn.Sequential()

    name = layer_spec['block_name']
    layers = []
    if name == "convbn":
        schema = ['inp', 'oup', 'kernel_size', 'stride']
        for i in range(layer_spec['num_blocks']):
            args = dict(zip(schema, layer_spec['block_specs'][i]))
            layers.append(conv_2d(**args, norm_layer=norm_layer, act_layer=act_layer))
    elif name == "uib":
        schema = ['inp', 'oup', 'start_dw_kernel_size', 'middle_dw_kernel_size', 'middle_dw_downsample', 'stride', 'expand_ratio']
        for i in range(layer_spec['num_blocks']):
            args = dict(zip(schema, layer_spec['block_specs'][i]))
            layers.append(UniversalInvertedBottleneckBlock(**args, norm_layer=norm_layer, act_layer=act_layer))
    elif name == "fused_ib":
        schema = ['inp', 'oup', 'stride', 'expand_ratio', 'act']
        for i in range(layer_spec['num_blocks']):
            args = dict(zip(schema, layer_spec['block_specs'][i]))
            layers.append(InvertedResidual(**args))
    else:
        raise NotImplementedError(f"Unknown block_name: {name}")

    return nn.Sequential(*layers)

# -------------------------
# 主干
# -------------------------
class MobileNetV4(nn.Module):
    """
    返回 4 个尺度特征: 下采样 4/8/16/32（若某阶段未命中则置为 None）。
    """
    def __init__(
        self,
        model: str,
        in_chans: int = 3,
        norm_layer=nn.BatchNorm2d,
        act_layer=nn.ReLU6,
    ):
        super().__init__()
        if model not in MODEL_SPECS:
            raise ValueError(f"Unsupported model {model}.")
        if "Hybrid" in model and len(MODEL_SPECS[model]) == 0:
            raise NotImplementedError(f"{model} spec is empty. Please fill Hybrid specs first.")

        spec = MODEL_SPECS[model]
        # 构建各层
        self.conv0  = build_blocks(spec.get('conv0'),  norm_layer=norm_layer, act_layer=act_layer)
        self.layer1 = build_blocks(spec.get('layer1'), norm_layer=norm_layer, act_layer=act_layer)
        self.layer2 = build_blocks(spec.get('layer2'), norm_layer=norm_layer, act_layer=act_layer)
        self.layer3 = build_blocks(spec.get('layer3'), norm_layer=norm_layer, act_layer=act_layer)
        self.layer4 = build_blocks(spec.get('layer4'), norm_layer=norm_layer, act_layer=act_layer)
        self.layer5 = build_blocks(spec.get('layer5'), norm_layer=norm_layer, act_layer=act_layer)

        self.features = nn.ModuleList([self.conv0, self.layer1, self.layer2, self.layer3, self.layer4, self.layer5])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming 正态更稳定
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def feature_info(self, x_size: int = 640) -> List[Tuple[int, int]]:
        """
        粗略返回（reduction, channels）信息，使用一次性前向推断统计。
        """
        with torch.no_grad():
            xs = torch.zeros(1, 3, x_size, x_size)
            feats = self.forward(xs)
        info = []
        reductions = [4, 8, 16, 32]
        for r, f in zip(reductions, feats):
            ch = 0 if f is None else f.size(1)
            info.append((r, ch))
        return info

    def forward(self, x: torch.Tensor) -> List[Optional[torch.Tensor]]:
        input_h = x.size(2)
        # 关注 4/8/16/32 下采样尺度
        target_scales = [4, 8, 16, 32]
        feats: List[Optional[torch.Tensor]] = [None, None, None, None]

        for f in self.features:
            x = f(x)
            ratio = input_h // x.size(2)
            if ratio in target_scales:
                feats[target_scales.index(ratio)] = x
        return feats

# -------------------------
# 便捷构造函数
# -------------------------
def MobileNetV4ConvSmall() -> MobileNetV4:
    return MobileNetV4('MobileNetV4ConvSmall')

def MobileNetV4ConvMedium() -> MobileNetV4:
    return MobileNetV4('MobileNetV4ConvMedium')

def MobileNetV4ConvLarge() -> MobileNetV4:
    return MobileNetV4('MobileNetV4ConvLarge')

def MobileNetV4HybridMedium() -> MobileNetV4:
    # 规格为空，按需补齐后可启用
    return MobileNetV4('MobileNetV4HybridMedium')

def MobileNetV4HybridLarge() -> MobileNetV4:
    # 规格为空，按需补齐后可启用
    return MobileNetV4('MobileNetV4HybridLarge')

# -------------------------
# 简单自测
# -------------------------
if __name__ == '__main__':
    model = MobileNetV4ConvSmall()
    model.eval()
    inputs = torch.randn(1, 3, 640, 640)
    outs = model(inputs)
    for i, o in enumerate(outs, start=1):
        if o is None:
            print(f'F{i}: None')
        else:
            print(f'F{i}: {tuple(o.shape)}')

















# import torch
# import torch.nn as nn
#
# __all__ = ['MobileNetV4ConvSmall', 'MobileNetV4ConvMedium', 'MobileNetV4ConvLarge', 'MobileNetV4HybridMedium',
#            'MobileNetV4HybridLarge']
#
# MNV4ConvSmall_BLOCK_SPECS = {
#     "conv0": {
#         "block_name": "convbn",
#         "num_blocks": 1,
#         "block_specs": [
#             [3, 32, 3, 2]
#         ]
#     },
#     "layer1": {
#         "block_name": "convbn",
#         "num_blocks": 2,
#         "block_specs": [
#             [32, 32, 3, 2],
#             [32, 32, 1, 1]
#         ]
#     },
#     "layer2": {
#         "block_name": "convbn",
#         "num_blocks": 2,
#         "block_specs": [
#             [32, 96, 3, 2],
#             [96, 64, 1, 1]
#         ]
#     },
#     "layer3": {
#         "block_name": "uib",
#         "num_blocks": 6,
#         "block_specs": [
#             [64, 96, 5, 5, True, 2, 3],
#             [96, 96, 0, 3, True, 1, 2],
#             [96, 96, 0, 3, True, 1, 2],
#             [96, 96, 0, 3, True, 1, 2],
#             [96, 96, 0, 3, True, 1, 2],
#             [96, 96, 3, 0, True, 1, 4],
#         ]
#     },
#     "layer4": {
#         "block_name": "uib",
#         "num_blocks": 6,
#         "block_specs": [
#             [96, 128, 3, 3, True, 2, 6],
#             [128, 128, 5, 5, True, 1, 4],
#             [128, 128, 0, 5, True, 1, 4],
#             [128, 128, 0, 5, True, 1, 3],
#             [128, 128, 0, 3, True, 1, 4],
#             [128, 128, 0, 3, True, 1, 4],
#         ]
#     },
#     "layer5": {
#         "block_name": "convbn",
#         "num_blocks": 2,
#         "block_specs": [
#             [128, 960, 1, 1],
#             [960, 1280, 1, 1]
#         ]
#     }
# }
#
# MNV4ConvMedium_BLOCK_SPECS = {
#     "conv0": {
#         "block_name": "convbn",
#         "num_blocks": 1,
#         "block_specs": [
#             [3, 32, 3, 2]
#         ]
#     },
#     "layer1": {
#         "block_name": "fused_ib",
#         "num_blocks": 1,
#         "block_specs": [
#             [32, 48, 2, 4.0, True]
#         ]
#     },
#     "layer2": {
#         "block_name": "uib",
#         "num_blocks": 2,
#         "block_specs": [
#             [48, 80, 3, 5, True, 2, 4],
#             [80, 80, 3, 3, True, 1, 2]
#         ]
#     },
#     "layer3": {
#         "block_name": "uib",
#         "num_blocks": 8,
#         "block_specs": [
#             [80, 160, 3, 5, True, 2, 6],
#             [160, 160, 3, 3, True, 1, 4],
#             [160, 160, 3, 3, True, 1, 4],
#             [160, 160, 3, 5, True, 1, 4],
#             [160, 160, 3, 3, True, 1, 4],
#             [160, 160, 3, 0, True, 1, 4],
#             [160, 160, 0, 0, True, 1, 2],
#             [160, 160, 3, 0, True, 1, 4]
#         ]
#     },
#     "layer4": {
#         "block_name": "uib",
#         "num_blocks": 11,
#         "block_specs": [
#             [160, 256, 5, 5, True, 2, 6],
#             [256, 256, 5, 5, True, 1, 4],
#             [256, 256, 3, 5, True, 1, 4],
#             [256, 256, 3, 5, True, 1, 4],
#             [256, 256, 0, 0, True, 1, 4],
#             [256, 256, 3, 0, True, 1, 4],
#             [256, 256, 3, 5, True, 1, 2],
#             [256, 256, 5, 5, True, 1, 4],
#             [256, 256, 0, 0, True, 1, 4],
#             [256, 256, 0, 0, True, 1, 4],
#             [256, 256, 5, 0, True, 1, 2]
#         ]
#     },
#     "layer5": {
#         "block_name": "convbn",
#         "num_blocks": 2,
#         "block_specs": [
#             [256, 960, 1, 1],
#             [960, 1280, 1, 1]
#         ]
#     }
# }
#
# MNV4ConvLarge_BLOCK_SPECS = {
#     "conv0": {
#         "block_name": "convbn",
#         "num_blocks": 1,
#         "block_specs": [
#             [3, 24, 3, 2]
#         ]
#     },
#     "layer1": {
#         "block_name": "fused_ib",
#         "num_blocks": 1,
#         "block_specs": [
#             [24, 48, 2, 4.0, True]
#         ]
#     },
#     "layer2": {
#         "block_name": "uib",
#         "num_blocks": 2,
#         "block_specs": [
#             [48, 96, 3, 5, True, 2, 4],
#             [96, 96, 3, 3, True, 1, 4]
#         ]
#     },
#     "layer3": {
#         "block_name": "uib",
#         "num_blocks": 11,
#         "block_specs": [
#             [96, 192, 3, 5, True, 2, 4],
#             [192, 192, 3, 3, True, 1, 4],
#             [192, 192, 3, 3, True, 1, 4],
#             [192, 192, 3, 3, True, 1, 4],
#             [192, 192, 3, 5, True, 1, 4],
#             [192, 192, 5, 3, True, 1, 4],
#             [192, 192, 5, 3, True, 1, 4],
#             [192, 192, 5, 3, True, 1, 4],
#             [192, 192, 5, 3, True, 1, 4],
#             [192, 192, 5, 3, True, 1, 4],
#             [192, 192, 3, 0, True, 1, 4]
#         ]
#     },
#     "layer4": {
#         "block_name": "uib",
#         "num_blocks": 13,
#         "block_specs": [
#             [192, 512, 5, 5, True, 2, 4],
#             [512, 512, 5, 5, True, 1, 4],
#             [512, 512, 5, 5, True, 1, 4],
#             [512, 512, 5, 5, True, 1, 4],
#             [512, 512, 5, 0, True, 1, 4],
#             [512, 512, 5, 3, True, 1, 4],
#             [512, 512, 5, 0, True, 1, 4],
#             [512, 512, 5, 0, True, 1, 4],
#             [512, 512, 5, 3, True, 1, 4],
#             [512, 512, 5, 5, True, 1, 4],
#             [512, 512, 5, 0, True, 1, 4],
#             [512, 512, 5, 0, True, 1, 4],
#             [512, 512, 5, 0, True, 1, 4]
#         ]
#     },
#     "layer5": {
#         "block_name": "convbn",
#         "num_blocks": 2,
#         "block_specs": [
#             [512, 960, 1, 1],
#             [960, 1280, 1, 1]
#         ]
#     }
# }
#
# MNV4HybridConvMedium_BLOCK_SPECS = {
#
# }
#
# MNV4HybridConvLarge_BLOCK_SPECS = {
#
# }
#
# MODEL_SPECS = {
#     "MobileNetV4ConvSmall": MNV4ConvSmall_BLOCK_SPECS,
#     "MobileNetV4ConvMedium": MNV4ConvMedium_BLOCK_SPECS,
#     "MobileNetV4ConvLarge": MNV4ConvLarge_BLOCK_SPECS,
#     "MobileNetV4HybridMedium": MNV4HybridConvMedium_BLOCK_SPECS,
#     "MobileNetV4HybridLarge": MNV4HybridConvLarge_BLOCK_SPECS,
# }
#
#
# def make_divisible(
#         value: float,
#         divisor: int,
#         min_value: Optional[float] = None,
#         round_down_protect: bool = True,
# ) -> int:
#     """
#     This function is copied from here
#     "https://github.com/tensorflow/models/blob/master/official/vision/modeling/layers/nn_layers.py"
#
#     This is to ensure that all layers have channels that are divisible by 8.
#
#     Args:
#         value: A `float` of original value.
#         divisor: An `int` of the divisor that need to be checked upon.
#         min_value: A `float` of  minimum value threshold.
#         round_down_protect: A `bool` indicating whether round down more than 10%
#         will be allowed.
#
#     Returns:
#         The adjusted value in `int` that is divisible against divisor.
#     """
#     if min_value is None:
#         min_value = divisor
#     new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
#     # Make sure that round down does not go down by more than 10%.
#     if round_down_protect and new_value < 0.9 * value:
#         new_value += divisor
#     return int(new_value)
#
#
# def conv_2d(inp, oup, kernel_size=3, stride=1, groups=1, bias=False, norm=True, act=True):
#     conv = nn.Sequential()
#     padding = (kernel_size - 1) // 2
#     conv.add_module('conv', nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=bias, groups=groups))
#     if norm:
#         conv.add_module('BatchNorm2d', nn.BatchNorm2d(oup))
#     if act:
#         conv.add_module('Activation', nn.ReLU6())
#     return conv
#
#
# class InvertedResidual(nn.Module):
#     def __init__(self, inp, oup, stride, expand_ratio, act=False):
#         super(InvertedResidual, self).__init__()
#         self.stride = stride
#         assert stride in [1, 2]
#         hidden_dim = int(round(inp * expand_ratio))
#         self.block = nn.Sequential()
#         if expand_ratio != 1:
#             self.block.add_module('exp_1x1', conv_2d(inp, hidden_dim, kernel_size=1, stride=1))
#         self.block.add_module('conv_3x3',
#                               conv_2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim))
#         self.block.add_module('red_1x1', conv_2d(hidden_dim, oup, kernel_size=1, stride=1, act=act))
#         self.use_res_connect = self.stride == 1 and inp == oup
#
#     def forward(self, x):
#         if self.use_res_connect:
#             return x + self.block(x)
#         else:
#             return self.block(x)
#
#
# class UniversalInvertedBottleneckBlock(nn.Module):
#     def __init__(self,
#                  inp,
#                  oup,
#                  start_dw_kernel_size,
#                  middle_dw_kernel_size,
#                  middle_dw_downsample,
#                  stride,
#                  expand_ratio
#                  ):
#         super().__init__()
#         # Starting depthwise conv.
#         self.start_dw_kernel_size = start_dw_kernel_size
#         if self.start_dw_kernel_size:
#             stride_ = stride if not middle_dw_downsample else 1
#             self._start_dw_ = conv_2d(inp, inp, kernel_size=start_dw_kernel_size, stride=stride_, groups=inp, act=False)
#         # Expansion with 1x1 convs.
#         expand_filters = make_divisible(inp * expand_ratio, 8)
#         self._expand_conv = conv_2d(inp, expand_filters, kernel_size=1)
#         # Middle depthwise conv.
#         self.middle_dw_kernel_size = middle_dw_kernel_size
#         if self.middle_dw_kernel_size:
#             stride_ = stride if middle_dw_downsample else 1
#             self._middle_dw = conv_2d(expand_filters, expand_filters, kernel_size=middle_dw_kernel_size, stride=stride_,
#                                       groups=expand_filters)
#         # Projection with 1x1 convs.
#         self._proj_conv = conv_2d(expand_filters, oup, kernel_size=1, stride=1, act=False)
#
#         # Ending depthwise conv.
#         # this not used
#         # _end_dw_kernel_size = 0
#         # self._end_dw = conv_2d(oup, oup, kernel_size=_end_dw_kernel_size, stride=stride, groups=inp, act=False)
#
#     def forward(self, x):
#         if self.start_dw_kernel_size:
#             x = self._start_dw_(x)
#             # print("_start_dw_", x.shape)
#         x = self._expand_conv(x)
#         # print("_expand_conv", x.shape)
#         if self.middle_dw_kernel_size:
#             x = self._middle_dw(x)
#             # print("_middle_dw", x.shape)
#         x = self._proj_conv(x)
#         # print("_proj_conv", x.shape)
#         return x
#
#
# def build_blocks(layer_spec):
#     if not layer_spec.get('block_name'):
#         return nn.Sequential()
#     block_names = layer_spec['block_name']
#     layers = nn.Sequential()
#     if block_names == "convbn":
#         schema_ = ['inp', 'oup', 'kernel_size', 'stride']
#         args = {}
#         for i in range(layer_spec['num_blocks']):
#             args = dict(zip(schema_, layer_spec['block_specs'][i]))
#             layers.add_module(f"convbn_{i}", conv_2d(**args))
#     elif block_names == "uib":
#         schema_ = ['inp', 'oup', 'start_dw_kernel_size', 'middle_dw_kernel_size', 'middle_dw_downsample', 'stride',
#                    'expand_ratio']
#         args = {}
#         for i in range(layer_spec['num_blocks']):
#             args = dict(zip(schema_, layer_spec['block_specs'][i]))
#             layers.add_module(f"uib_{i}", UniversalInvertedBottleneckBlock(**args))
#     elif block_names == "fused_ib":
#         schema_ = ['inp', 'oup', 'stride', 'expand_ratio', 'act']
#         args = {}
#         for i in range(layer_spec['num_blocks']):
#             args = dict(zip(schema_, layer_spec['block_specs'][i]))
#             layers.add_module(f"fused_ib_{i}", InvertedResidual(**args))
#     else:
#         raise NotImplementedError
#     return layers
#
#
# class MobileNetV4(nn.Module):
#     def __init__(self, model):
#         # MobileNetV4ConvSmall  MobileNetV4ConvMedium  MobileNetV4ConvLarge
#         # MobileNetV4HybridMedium  MobileNetV4HybridLarge
#         """Params to initiate MobilenNetV4
#         Args:
#             model : support 5 types of models as indicated in
#             "https://github.com/tensorflow/models/blob/master/official/vision/modeling/backbones/mobilenet.py"
#         """
#         super().__init__()
#         assert model in MODEL_SPECS.keys()
#         self.model = model
#         self.spec = MODEL_SPECS[self.model]
#
#         # conv0
#         self.conv0 = build_blocks(self.spec['conv0'])
#         # layer1
#         self.layer1 = build_blocks(self.spec['layer1'])
#         # layer2
#         self.layer2 = build_blocks(self.spec['layer2'])
#         # layer3
#         self.layer3 = build_blocks(self.spec['layer3'])
#         # layer4
#         self.layer4 = build_blocks(self.spec['layer4'])
#         # layer5
#         self.layer5 = build_blocks(self.spec['layer5'])
#         self.features = nn.ModuleList([self.conv0, self.layer1, self.layer2, self.layer3, self.layer4, self.layer5])
#         self.channel = [i.size(1) for i in self.forward(torch.randn(1, 3, 640, 640))]
#
#     def forward(self, x):
#         input_size = x.size(2)
#         scale = [4, 8, 16, 32]
#         features = [None, None, None, None]
#         for f in self.features:
#             x = f(x)
#             if input_size // x.size(2) in scale:
#                 features[scale.index(input_size // x.size(2))] = x
#         return features
#
#
# def MobileNetV4ConvSmall():
#     model = MobileNetV4('MobileNetV4ConvSmall')
#     return model
#
#
# def MobileNetV4ConvMedium():
#     model = MobileNetV4('MobileNetV4ConvMedium')
#     return model
#
#
# def MobileNetV4ConvLarge():
#     model = MobileNetV4('MobileNetV4ConvLarge')
#     return model
#
#
# def MobileNetV4HybridMedium():
#     model = MobileNetV4('MobileNetV4HybridMedium')
#     return model
#
#
# def MobileNetV4HybridLarge():
#     model = MobileNetV4('MobileNetV4HybridLarge')
#     return model
#
#
# if __name__ == '__main__':
#     model = MobileNetV4ConvSmall()
#     inputs = torch.randn((1, 3, 640, 640))
#     res = model(inputs)
#     for i in res:
#         print(i.size())