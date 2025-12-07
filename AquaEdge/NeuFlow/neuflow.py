import torch
import torch.nn.functional as F


#导入 NeuFlow 模块中定义的相关功能，
# 如 backbone_v7（特征提取网络）、
# transformer（注意力机制）、
# matching（匹配模块）、
# corr（计算相关性）、
# refine（精细化模块）、
# upsample（上采样模块）和 config（配置文件）
from NeuFlow import backbone_v7
from NeuFlow import transformer
from NeuFlow import matching
from NeuFlow import corr
from NeuFlow import refine
from NeuFlow import upsample
from NeuFlow import config

from NeuFlow.mnv4_adapter import MobileNetV4Encoder

from huggingface_hub import PyTorchModelHubMixin

#NeuFlow 继承自 torch.nn.Module，表明这是一个 PyTorch 模型。PyTorchModelHubMixin：使模型能够与 Hugging Face 模型库进行集成。通过 repo_url、license 和 pipeline_tag 来指定模型的元数据
class NeuFlow(torch.nn.Module,
              PyTorchModelHubMixin,
              repo_url="https://github.com/neufieldrobotics/NeuFlow_v2", license="apache-2.0", pipeline_tag="image-to-image"):
    def __init__(self):#调用父类的初始化方法
        super(NeuFlow, self).__init__()

        #self.backbone 是一个基于卷积神经网络的特征提取模块，使用 CNNEncoder 来从输入图像中提取特征。
        # self.cross_attn_s16 是一个特征注意力模块，应用在 16x16 尺度的特征图上。注意力机制用于强调图像中重要的区域，增强模型的鲁棒性

        #backbone：使用 CNNEncoder 模块进行图像特征的提取。
        # feature_dim_s16 和 context_dim_s16 等是从 config 中读取的配置，分别代表不同尺度下的特征维度。
        #self.backbone = backbone_v7.CNNEncoder(config.feature_dim_s16, config.context_dim_s16, config.feature_dim_s8, config.context_dim_s8)
        #MobileNetV4
        self.backbone = MobileNetV4Encoder(
            feature_dim_s16=config.feature_dim_s16,
            context_dim_s16=config.context_dim_s16,
            feature_dim_s8=config.feature_dim_s8,
            context_dim_s8=config.context_dim_s8,
            #timm_name="mobilenetv4_conv_small.e3600_r256_in1k",  # 或 conv_medium
            
            model_name="MobileNetV4ConvSmall",
            #pretrained=False,
            imagenet_norm=True,
        )

        #cross_attn_s16：使用 FeatureAttention 模块进行跨通道特征注意力机制的应用。
        # num_layers=2 表示注意力机制的层数，ffn 表示是否使用前馈网络，post_norm=True 表示在注意力计算后进行归一化
        self.cross_attn_s16 = transformer.FeatureAttention(config.feature_dim_s16+config.context_dim_s16, num_layers=2, ffn=True, ffn_dim_expansion=1, post_norm=True)

        #matching_s16：使用 Matching 模块进行光流的匹配
        self.matching_s16 = matching.Matching()

        # self.flow_attn_s16 = transformer.FlowAttention(config.feature_dim_s16)
        #corr_block_s16 和 corr_block_s8：分别用于计算16x16尺度和8x8尺度的图像特征之间的相关性
        self.corr_block_s16 = corr.CorrBlock(radius=4, levels=1)
        self.corr_block_s8 = corr.CorrBlock(radius=4, levels=1)

        #merge_s8：一个卷积模块，用于将16x16尺度和8x8尺度的特征图合并，并通过卷积、GELU 激活函数和批归一化进行处理
        self.merge_s8 = torch.nn.Sequential(torch.nn.Conv2d(config.feature_dim_s16 + config.feature_dim_s8, config.feature_dim_s8, kernel_size=3, stride=1, padding=1, bias=False),
                                              torch.nn.GELU(),
                                              torch.nn.Conv2d(config.feature_dim_s8, config.feature_dim_s8, kernel_size=3, stride=1, padding=1, bias=False),
                                              torch.nn.BatchNorm2d(config.feature_dim_s8))

        #context_merge_s8：类似于 merge_s8，但是用于处理上下文特征（即包含上下文信息的特征图）
        self.context_merge_s8 = torch.nn.Sequential(torch.nn.Conv2d(config.context_dim_s16 + config.context_dim_s8, config.context_dim_s8, kernel_size=3, stride=1, padding=1, bias=False),
                                           torch.nn.GELU(),
                                           torch.nn.Conv2d(config.context_dim_s8, config.context_dim_s8, kernel_size=3, stride=1, padding=1, bias=False),
                                           torch.nn.BatchNorm2d(config.context_dim_s8))

        #refine_s16 和 refine_s8：使用 Refine 模块进行光流的精细化处理
        self.refine_s16 = refine.Refine(config.context_dim_s16, config.iter_context_dim_s16, num_layers=5, levels=1, radius=4, inter_dim=128)
        self.refine_s8 = refine.Refine(config.context_dim_s8, config.iter_context_dim_s8, num_layers=5, levels=1, radius=4, inter_dim=96)

        #conv_s8：用于将输入图像通过卷积层进行下采样。
        # upsample_s8：用于通过上采样将特征图恢复到较高分辨率。
        self.conv_s8 = backbone_v7.ConvBlock(3, config.feature_dim_s1, kernel_size=8, stride=8, padding=0)
        self.upsample_s8 = upsample.UpSample(config.feature_dim_s1, upsample_factor=8)

        #使用 Xavier 均匀分布初始化模型的权
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    #init_bhwd 方法初始化了不同模块的尺寸，确保每个模块的输入输出尺寸一致，并且为每个模块分配了合适的设备和数据类型（如半精度浮点数）
    def init_bhwd(self, batch_size, height, width, device, amp=True):

        self.backbone.init_bhwd(batch_size*2, height//16, width//16, device, amp)

        self.matching_s16.init_bhwd(batch_size, height//16, width//16, device, amp)

        self.corr_block_s16.init_bhwd(batch_size, height//16, width//16, device, amp)
        self.corr_block_s8.init_bhwd(batch_size, height//8, width//8, device, amp)

        self.refine_s16.init_bhwd(batch_size, height//16, width//16, device, amp)
        self.refine_s8.init_bhwd(batch_size, height//8, width//8, device, amp)

        self.init_iter_context_s16 = torch.zeros(batch_size, config.iter_context_dim_s16, height//16, width//16, device=device, dtype=torch.half if amp else torch.float)
        self.init_iter_context_s8 = torch.zeros(batch_size, config.iter_context_dim_s8, height//8, width//8, device=device, dtype=torch.half if amp else torch.float)

    #split_features 将输入的特征图分割为上下文信息和主要特征部分
    def split_features(self, features, context_dim, feature_dim):

        context, features = torch.split(features, [context_dim, feature_dim], dim=1)

        context, _ = context.chunk(chunks=2, dim=0)
        feature0, feature1 = features.chunk(chunks=2, dim=0)

        return features, torch.relu(context)

    #输入两张图像 img0 和 img1，对其进行归一化。
    # 提取特征并通过注意力机制 (cross_attn_s16) 处理
    def forward(self, img0, img1, iters_s16=1, iters_s8=8):
    #forward 方法是模型的前向传播函数。它接受两张图像 (img0 和 img1) 作为输入，并计算它们之间的光流。具体步骤包括：
    # 对图像进行归一化。
    # 使用 backbone 提取特征图。
    # 使用 cross_attn_s16 进行跨通道特征注意力机制的应用。
    # 计算 16x16 和 8x8 尺度的光流。
    # 使用 refine_s16 和 refine_s8 模块进行光流的精细化。
    # 使用 upsample_s8 将光流恢复到较高的分辨率。

        flow_list = []

        img0 /= 255.
        img1 /= 255.

        features_s16, features_s8 = self.backbone(torch.cat([img0, img1], dim=0))

        features_s16 = self.cross_attn_s16(features_s16)

        #对16x16尺度和8x8尺度的特征图进行分割
        features_s16, context_s16 = self.split_features(features_s16, config.context_dim_s16, config.feature_dim_s16)
        features_s8, context_s8 = self.split_features(features_s8, config.context_dim_s8, config.feature_dim_s8)

        feature0_s16, feature1_s16 = features_s16.chunk(chunks=2, dim=0)

        #计算初始的光流 flow0
        flow0 = self.matching_s16.global_correlation_softmax(feature0_s16, feature1_s16)

        # flow0 = self.flow_attn_s16(feature0_s16, flow0)

        corr_pyr_s16 = self.corr_block_s16.init_corr_pyr(feature0_s16, feature1_s16)

        iter_context_s16 = self.init_iter_context_s16

        #使用 corr_block_s16 和 refine_s16 模块进行光流的精细化。每次迭代都会更新光流
        for i in range(iters_s16):

            if self.training and i > 0:
                flow0 = flow0.detach()
                # iter_context_s16 = iter_context_s16.detach()

            corrs = self.corr_block_s16(corr_pyr_s16, flow0)

            iter_context_s16, delta_flow = self.refine_s16(corrs, context_s16, iter_context_s16, flow0)

            flow0 = flow0 + delta_flow

            if self.training:
                up_flow0 = F.interpolate(flow0, scale_factor=16, mode='bilinear') * 16
                flow_list.append(up_flow0)

        #对光流进行上采样，将其恢复到较高的分辨率
        flow0 = F.interpolate(flow0, scale_factor=2, mode='nearest') * 2

        features_s16 = F.interpolate(features_s16, scale_factor=2, mode='nearest')

        #将16x16尺度的特征与8x8尺度的特征进行合并
        features_s8 = self.merge_s8(torch.cat([features_s8, features_s16], dim=1))

        feature0_s8, feature1_s8 = features_s8.chunk(chunks=2, dim=0)

        corr_pyr_s8 = self.corr_block_s8.init_corr_pyr(feature0_s8, feature1_s8)

        context_s16 = F.interpolate(context_s16, scale_factor=2, mode='nearest')

        context_s8 = self.context_merge_s8(torch.cat([context_s8, context_s16], dim=1))

        iter_context_s8 = self.init_iter_context_s8

        for i in range(iters_s8):

            if self.training and i > 0:
                flow0 = flow0.detach()
                # iter_context_s8 = iter_context_s8.detach()

            corrs = self.corr_block_s8(corr_pyr_s8, flow0)

            iter_context_s8, delta_flow = self.refine_s8(corrs, context_s8, iter_context_s8, flow0)

            flow0 = flow0 + delta_flow

            if self.training or i == iters_s8 - 1:

                feature0_s1 = self.conv_s8(img0)
                up_flow0 = self.upsample_s8(feature0_s1, flow0) * 8
                flow_list.append(up_flow0)

        return flow_list
