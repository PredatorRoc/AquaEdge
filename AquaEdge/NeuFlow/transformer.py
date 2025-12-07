import torch
import torch.nn.functional as F

class TransformerLayer(torch.nn.Module):
    def __init__(self,
                 feature_dim,
                 ffn=True,
                 ffn_dim_expansion=1
                 ):
        super(TransformerLayer, self).__init__()

        # multi-head attention
        self.q_proj = torch.nn.Linear(feature_dim, feature_dim)
        self.k_proj = torch.nn.Linear(feature_dim, feature_dim)
        self.v_proj = torch.nn.Linear(feature_dim, feature_dim)

        self.merge = torch.nn.Linear(feature_dim, feature_dim)

        # self.multi_head_attn = torch.nn.MultiheadAttention(feature_dim, 2, batch_first=True, device='cuda')

        self.norm1 = torch.nn.LayerNorm(feature_dim)

        self.ffn = ffn

        if self.ffn:
            in_channels = feature_dim * 2
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(in_channels, in_channels * ffn_dim_expansion, bias=False),
                torch.nn.GELU(),
                torch.nn.Linear(in_channels * ffn_dim_expansion, feature_dim, bias=False),
            )

            self.norm2 = torch.nn.LayerNorm(feature_dim)

    def forward(self, source, target):
        # source, target: [B, L, C]
        query, key, value = source, target, target

        # single-head attention
        query = self.q_proj(query)  # [B, L, C]
        key = self.k_proj(key)  # [B, L, C]
        value = self.v_proj(value)  # [B, L, C]

        message = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)

        message = self.merge(message)

        # message, _ = self.multi_head_attn(query, key, value, need_weights=False)
        message = self.norm1(message)

        if self.ffn:
            message = self.mlp(torch.cat([source, message], dim=-1))
            message = self.norm2(message)

        return source + message

class FeatureAttention(torch.nn.Module):
    #FeatureAttention 继承自 torch.nn.Module，这意味着它是一个 PyTorch 模块。
    # 构造函数 __init__ 定义了几个参数：
    # feature_dim：输入特征的通道数（C）。
    # num_layers：Transformer 层的数量，决定了注意力机制的层数。
    # ffn：是否使用前馈神经网络（Feed-Forward Network，FFN）。默认为 True，表示使用 FFN。
    # ffn_dim_expansion：前馈网络中扩展的维度倍数。默认为 1，即没有扩展。
    # post_norm：是否在最后应用批量归一化（Batch Normalization）。默认为 False，表示不应用。
    def __init__(self, feature_dim, num_layers, ffn=True, ffn_dim_expansion=1, post_norm=False):
        super(FeatureAttention, self).__init__()

        self.layers = torch.nn.ModuleList([
            TransformerLayer(feature_dim, ffn=ffn, ffn_dim_expansion=ffn_dim_expansion
                             )
            for i in range(num_layers)])

        self.post_norm = post_norm

        if self.post_norm:
            self.norm = torch.nn.BatchNorm2d(feature_dim)

    def forward(self, concat_features0):
        #forward 方法接受一个输入 concat_features0，这是一个四维的张量，表示批量中的图像特征。
    # b：批量大小（batch size）。
    # c：特征的通道数（channel）。
    # h：图像的高度（height）。
    # w：图像的宽度（width）。

        b, c, h, w = concat_features0.shape
        #将输入的特征图 concat_features0 展平为一个二维张量：[B, H*W, C]，即将每个特征图的空间维度（h 和 w）展开成一个长向量（H*W）。
        # 使用 permute(0, 2, 1) 将张量的维度从 [B, C, H, W] 转换为 [B, H*W, C]，其中 B 是批量大小，C 是特征通道数，H*W 是每张图的总像素数
        concat_features0 = concat_features0.flatten(-2).permute(0, 2, 1)  # [B, H*W, C]
        concat_features1 = torch.cat(concat_features0.chunk(chunks=2, dim=0)[::-1], dim=0)
        #将 concat_features0 沿批量维度 (dim=0) 分割成两部分 (chunks=2)，然后反转两部分的位置（[::-1]），并通过 torch.cat 将它们拼接回一个张量 concat_features1。
        # 这个操作的目的是将图像特征的批量顺序调换，这有助于计算注意力机制中基于对比的操作

        #遍历 self.layers 中的每一层，依次应用 TransformerLayer。每一层都会使用 concat_features0 和 concat_features1 来计算特征的增强（注意力）
        #concat_features0 和 concat_features1 的交替计算在每一层中更新它们，以实现注意力机制
        for layer in self.layers:
            concat_features0 = layer(concat_features0, concat_features1)
            concat_features1 = torch.cat(concat_features0.chunk(chunks=2, dim=0)[::-1], dim=0)

        # reshape back
        concat_features0 = concat_features0.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        
        if self.post_norm:
            concat_features0 = self.norm(concat_features0)

        return concat_features0


class FlowAttention(torch.nn.Module):
    """
    flow propagation with self-attention on feature
    query: feature0, key: feature0, value: flow
    """

    def __init__(self, feature_dim):
        super(FlowAttention, self).__init__()

        self.q_proj = torch.nn.Linear(feature_dim, feature_dim)
        self.k_proj = torch.nn.Linear(feature_dim, feature_dim)

    def forward(self, feature, flow):
        # q, k: feature [B, C, H, W], v: flow [B, 2, H, W]
        b, c, h, w = feature.size()

        feature = feature.flatten(-2).permute(0, 2, 1)  # [B, H*W, C]

        flow = flow.flatten(-2).permute(0, 2, 1)

        query = self.q_proj(feature)  # [B, H*W, C]
        key = self.k_proj(feature)  # [B, H*W, C]

        flow = F.scaled_dot_product_attention(query, key, flow)

        flow = flow.view(b, h, w, 2).permute(0, 3, 1, 2)

        return flow
