import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import spectral_norm
from torch.nn import Parameter
import shutil
import os
import webbrowser
from loss import smooth_mask_generate

def l2normalize(v, eps = 1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)
    
class LayerNorm(nn.Module):
    def __init__(self, num_features, eps = 1e-8, affine = True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = Parameter(torch.Tensor(num_features).uniform_())
            self.beta = Parameter(torch.zeros(num_features))

    def forward(self, x):
        # layer norm
        shape = [-1] + [1] * (x.dim() - 1)                                  # for 4d input: [-1, 1, 1, 1]
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)
        # if it is learnable
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)                          # for 4d input: [1, -1, 1, 1]
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'reflect', activation = 'lrelu', norm = 'none', sn = False):
        super(GatedConv2d, self).__init__()
        # Initialize the padding scheme
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)
        
        # Initialize the normalization type
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'ln':
            self.norm = LayerNorm(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)
        
        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU(inplace = True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace = True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace = True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sn:
            self.conv2d = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation))
            self.mask_conv2d = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation))
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation)
            self.mask_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        x = self.pad(x)
        conv = self.conv2d(x)
        mask = self.mask_conv2d(x)
        gated_mask = self.sigmoid(mask)
        x = conv * gated_mask
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x
    
class SdeGatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'reflect', activation = 'lrelu', norm = 'none', sn = False):
        super(GatedConv2d, self).__init__()
        # Initialize the padding scheme
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)
        
        # Initialize the normalization type
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'ln':
            self.norm = LayerNorm(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)
        
        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU(inplace = True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace = True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace = True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sn:
            self.conv2d = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation))
            self.mask_conv2d = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation))
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation)
            self.mask_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x, global_mask):
        x = self.pad(x)
        conv = self.conv2d(x)
        mask = self.mask_conv2d(x)
        gated_mask = self.sigmoid(mask) * global_mask
        x = conv * gated_mask
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class TransposeGatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', sn = True, scale_factor = 2):
        super(TransposeGatedConv2d, self).__init__()
        # Initialize the conv scheme
        self.scale_factor = scale_factor
        self.gated_conv2d = GatedConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, sn)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor = self.scale_factor, mode = 'nearest')
        x = self.gated_conv2d(x)
        return x

class EarlyAttentionModule(nn.Module):
    """适用于输入层的注意力模块"""
    def __init__(self, in_channels):
        super(EarlyAttentionModule, self).__init__()
        self.conv = nn.Conv2d(in_channels + 1, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, mask):
        # 确保掩码尺寸与输入特征匹配
        mask_resized = F.interpolate(mask, size=x.shape[2:], mode='bilinear', align_corners=True)
        # 连接特征和掩码
        combined = torch.cat([x, mask_resized], dim=1)
        # 生成注意力权重
        attention = self.sigmoid(self.conv(combined))
        # 应用注意力
        return x * attention

class MaskAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(MaskAttentionModule, self).__init__()
        self.conv = nn.Conv2d(in_channels + 1, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, mask):
        # 将掩码上采样到特征图尺寸
        mask_resized = F.interpolate(mask, size=x.shape[2:], mode='bilinear', align_corners=True)
        # 连接特征和掩码
        combined = torch.cat([x, mask_resized], dim=1)
        # 生成注意力权重
        attention = self.sigmoid(self.conv(combined))
        # 应用注意力
        return x * attention

# 添加这个新模块到CompletionNetwork类定义之前
class BoundaryAwareModule(nn.Module):
    """边界感知模块，专注于改善边界连续性"""
    def __init__(self, in_channels):
        super(BoundaryAwareModule, self).__init__()
        
        # 边界特征提取
        self.boundary_conv = nn.Sequential(
            nn.Conv2d(in_channels + 1, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # 特征融合
        self.fusion = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        
    def forward(self, features, mask):
        # 获取边界掩码
        boundary_mask = self._get_boundary_mask(mask)
        
        # 将特征与边界掩码连接
        boundary_input = torch.cat([features, boundary_mask], dim=1)
        
        # 提取边界特征
        boundary_features = self.boundary_conv(boundary_input)
        
        # 融合原始特征和边界特征
        combined = torch.cat([features, boundary_features], dim=1)
        enhanced_features = self.fusion(combined)
        
        return enhanced_features
    
    def _get_boundary_mask(self, mask):
        """简化的边界掩码生成"""
        # 使用最大池化和最小池化找出边界
        dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
        eroded = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
        
        # 边界是膨胀和腐蚀的差异
        boundary = torch.abs(dilated - eroded)
        
        return boundary
    
'''class GatedConv2d(nn.Module):
    """
    带门控机制的卷积层，可以更好地处理有掩码的输入
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(GatedConv2d, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        
    def forward(self, x, mask=None):
        # 特征通道
        features = nn.ReLU()(self.conv2d(x))
            
        # 生成门控通道
        gates = torch.sigmoid(self.mask_conv2d(x))
        
        # 如果提供了外部掩码，将它与门控机制结合
        if mask is not None:
            if mask.shape[2:] != gates.shape[2:]:
                mask = F.interpolate(mask, size=gates.shape[2:], mode='nearest')
            gates = gates * mask
            
        # 应用门控机制
        return features * gates'''


class CompletionNetwork(nn.Module):
    def __init__(self, input_channels=1):
        super(CompletionNetwork, self).__init__()
        
        # 显示调试信息的标志
        self.debug = False
        
        # 为局部和全局输入添加早期注意力模块
        self.early_local_attention = EarlyAttentionModule(input_channels)
        self.early_global_attention = EarlyAttentionModule(input_channels)

        # Local encoder branch with intermediate outputs saved for skip connections
        # 简化编码器结构，保持原有的跳跃连接设计
        self.local_enc1 = nn.Sequential(
            nn.Conv2d(input_channels * 2, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.local_enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.local_enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Global encoder branch - 使用GatedConv2d替换标准Conv2d
        self.global_enc1 = nn.Sequential(
            GatedConv2d(input_channels * 2, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.global_enc2 = nn.Sequential(
            GatedConv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.global_enc3 = nn.Sequential(
            GatedConv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            GatedConv2d(256, 256, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # 高级特征的全局注意力模块
        self.global_attention = MaskAttentionModule(256)
        
        # 特征融合的注意力模块
        self.fusion_attention = MaskAttentionModule(256)

        # Feature fusion - 更复杂的融合模块
        self.fusion = nn.Sequential(
            nn.Conv2d(512, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.boundary_aware = BoundaryAwareModule(256)  # 在深层特征处使用

        # 解码器增加更多层次和跳跃连接
        # 解码器第一层
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # 解码器第一层跳跃连接融合
        self.skip_fusion1 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # 解码器第二层
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # 解码器第二层跳跃连接融合
        self.skip_fusion2 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # 新增一个跳跃连接到编码器的第一个输出
        self.decoder3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, input_channels, kernel_size=3, stride=1, padding=1)
        )
        
        # 平滑过渡层
        self.transition_conv = nn.Conv2d(input_channels + 1, input_channels, kernel_size=3, stride=1, padding=1)
        # 添加偏移预测层
        # 优化全局偏移预测模块
        self.offset_module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Flatten(),
            nn.Linear(input_channels * 2, 64),  # 增加中间层
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),  # 最终输出一个偏移值
            nn.Tanh()  # 使用Tanh限制偏移范围在[-1, 1]
        )
        
        # 添加全局偏移幅度因子 - 控制偏移的强度
        self.offset_scale = nn.Parameter(torch.tensor(20.0))
        
        # 边界感知的偏移过渡模块
        self.offset_transition = nn.Sequential(
            nn.Conv2d(input_channels + 1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, input_channels, kernel_size=3, padding=1)
        )
        '''self.transition_conv = nn.Sequential(
        nn.Conv2d(input_channels + 1, 32, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=2, dilation=2),  # 扩张卷积扩大感受野
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, input_channels, kernel_size=3, stride=1, padding=1)
    )'''

    def print_sizes(self, tensor, name):
        """打印张量尺寸，用于调试"""
        if self.debug:
            print(f"{name} size: {tensor.size()}")

    def forward(self, local_input, local_mask, global_input, global_mask):

        local_mask = local_mask.float()
        global_mask = global_mask.float()

        
        # 对输入应用早期注意力
        local_input_attended = self.early_local_attention(local_input, local_mask)
        global_input_attended = self.early_global_attention(global_input, global_mask)
        
        # 将输入图像和掩码连接起来
        local_x = torch.cat([local_input_attended, local_mask], dim=1)
        # local_x = torch.cat([local_input, local_mask], dim=1)
        global_x = torch.cat([global_input_attended, global_mask], dim=1)
        
        # 本地编码器的前向传播
        local_feat1 = self.local_enc1(local_x)
        # self.print_sizes(local_feat1, "local_feat1")
        
        local_feat2 = self.local_enc2(local_feat1)
        # self.print_sizes(local_feat2, "local_feat2")
        
        local_feat3 = self.local_enc3(local_feat2)
        # self.print_sizes(local_feat3, "local_feat3")
        
        # 全局编码器的前向传播
        global_feat1 = self.global_enc1(global_x)
        # self.print_sizes(global_feat1, "global_feat1")
        
        global_feat2 = self.global_enc2(global_feat1)
        # self.print_sizes(global_feat2, "global_feat2")
        
        global_feat3 = self.global_enc3(global_feat2)
        # self.print_sizes(global_feat3, "global_feat3")
        
        # 将全局特征调整为与本地特征相同的大小
        global_feat3_resized = F.interpolate(
            global_feat3,
            size=local_feat3.size()[2:],
            mode="bilinear",
            align_corners=True
        )
        self.print_sizes(global_feat3_resized, "global_feat3_resized")
        
        # 融合本地和全局特征
        combined_features = torch.cat([local_feat3, global_feat3_resized], dim=1)
        self.print_sizes(combined_features, "combined_features")
        
        fused_features = self.fusion(combined_features)
        '''# 添加边界感知增强
        fused_features = self.boundary_aware(fused_features, F.interpolate(1-local_mask, 
                                                                 size=fused_features.shape[2:], 
                                                                 mode='bilinear', 
                                                                 align_corners=True))'''
        self.print_sizes(fused_features, "fused_features")
        
        # 解码器第一层 - 带跳跃连接
        dec1 = self.decoder1(fused_features)
        self.print_sizes(dec1, "dec1")
        
        # 确保解码器特征与编码器特征尺寸匹配
        if dec1.size()[2:] != local_feat2.size()[2:]:
            dec1 = F.interpolate(dec1, size=local_feat2.size()[2:], mode='bilinear', align_corners=True)
            self.print_sizes(dec1, "dec1_resized")
        
        # 第一个跳跃连接
        skip1 = torch.cat([dec1, local_feat2], dim=1)
        self.print_sizes(skip1, "skip1")
        
        skip1_fused = self.skip_fusion1(skip1)
        self.print_sizes(skip1_fused, "skip1_fused")
        
        # 解码器第二层
        dec2 = self.decoder2(skip1_fused)
        self.print_sizes(dec2, "dec2")
        
        # 确保解码器特征与编码器特征尺寸匹配
        if dec2.size()[2:] != local_feat1.size()[2:]:
            dec2 = F.interpolate(dec2, size=local_feat1.size()[2:], mode='bilinear', align_corners=True)
            self.print_sizes(dec2, "dec2_resized")
        
        # 第二个跳跃连接
        skip2 = torch.cat([dec2, local_feat1], dim=1)
        self.print_sizes(skip2, "skip2")
        
        skip2_fused = self.skip_fusion2(skip2)
        self.print_sizes(skip2_fused, "skip2_fused")
        
        # 最终输出
        output = self.decoder3(skip2_fused)
        self.print_sizes(output, "output")
        
        # 创建平滑过渡
        # 使用平均池化创建模糊的掩码边缘
        # blurred_mask = F.avg_pool2d(local_mask, kernel_size=3, stride=1, padding=1)
        blurred_mask = smooth_mask_generate(local_mask, 7.0, False)
        fused_mask = torch.max(blurred_mask, 1-local_mask)
        # blurred_mask = blurred_mask*local_mask + (1-local_mask)
        
        # 计算全局偏移
        # 1. 提取已知区域和生成区域的特征
        known_region = local_input * local_mask
        generated_region = output * (1 - local_mask)
        
        # 2. 连接特征用于偏移预测
        offset_features = torch.cat([
            known_region.mean(dim=(2, 3), keepdim=True),
            generated_region.mean(dim=(2, 3), keepdim=True)
        ], dim=1)
        
        # 3. 预测全局偏移值
        offset = self.offset_module(offset_features)
        offset = offset.view(-1, 1, 1, 1) # * self.offset_scale
        
        # 4. 根据掩码计算自适应偏移
        # 创建距离掩码 - 距离边界越远，偏移越大
        # distance_mask = self._compute_distance_mask(1 - local_mask)
        # adaptive_offset = offset * distance_mask
        
        # 5. 应用偏移到生成的输出
        offset_output = output + offset
        
        # 6. 使用边界感知的过渡模块融合结果
        transition_input = torch.cat([offset_output, fused_mask], dim=1)
        smoothed_output = self.offset_transition(transition_input)
        
        # output_with_mask = torch.cat([output, blurred_mask], dim=1)
        # output_with_mask = torch.cat([output, fused_mask], dim=1)
        # smoothed_output = self.transition_conv(output_with_mask)

        
        # 仅在掩码区域应用生成结果 - 平滑过渡
        final_output = (local_input * local_mask + smoothed_output * (1 - local_mask))
        # final_output = local_input * (1-blurred_mask) + smoothed_output * blurred_mask
        
        return final_output

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class EnhancedTerrainFeatureExtractor(nn.Module):
    """简化版地形特征提取器：仅包含坡度和原始高程"""
    def __init__(self, in_channels):
        super(EnhancedTerrainFeatureExtractor, self).__init__()
        
        # 坡度提取（使用Sobel算子）
        self.sobel_x = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        
        # 初始化滤波器
        with torch.no_grad():
            # Sobel滤波器
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
            
            # 扩展为卷积核
            sobel_x = sobel_x.reshape(1, 1, 3, 3).repeat(in_channels, in_channels, 1, 1)
            sobel_y = sobel_y.reshape(1, 1, 3, 3).repeat(in_channels, in_channels, 1, 1)
            
            # 设置卷积权重
            self.sobel_x.weight.data = sobel_x
            self.sobel_y.weight.data = sobel_y
        
    def forward(self, x):
        # 检查输入是否包含NaN
        if torch.isnan(x).any():
            print("警告: TerrainFeatureExtractor输入包含NaN!")
            x = torch.nan_to_num(x, nan=0.0)
            
        # 计算一阶导数
        grad_x = self.sobel_x(x)
        grad_y = self.sobel_y(x)
        
        # 计算坡度
        epsilon = 1e-6
        slope = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + epsilon)
        
        # 限制值范围，防止极值
        slope = torch.clamp(slope, min=0.0, max=10.0)
        
        # 返回简化的地形特征：原始高程 + 坡度
        features = torch.cat([x, slope], dim=1)
        features = F.dropout(features, p=0.1, training=self.training)
        
        return features


class StabilizedSelfAttention(nn.Module):
    """稳定版自注意力模块"""
    def __init__(self, in_channels):
        super(StabilizedSelfAttention, self).__init__()
        # 使用谱归一化提高稳定性
        self.query_conv = spectral_norm(nn.Conv2d(in_channels, in_channels // 8, kernel_size=1))
        self.key_conv = spectral_norm(nn.Conv2d(in_channels, in_channels // 8, kernel_size=1))
        self.value_conv = spectral_norm(nn.Conv2d(in_channels, in_channels, kernel_size=1))
        
        # 从较小值开始
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # 缩放因子
        self.scale_factor = torch.sqrt(torch.tensor(in_channels // 8, dtype=torch.float32))
        
    def forward(self, x):
        batch_size, C, height, width = x.size()
        
        # 检查输入是否包含NaN
        if torch.isnan(x).any():
            print("警告: SelfAttention输入包含NaN!")
            x = torch.nan_to_num(x, nan=0.0)
        
        # 压缩特征以减少内存使用
        proj_query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, height * width)
        
        # 计算注意力图，添加缩放因子提高数值稳定性
        energy = torch.bmm(proj_query, proj_key) / self.scale_factor
        
        # 使用更安全的softmax
        attention = F.softmax(energy, dim=-1)
        
        # 检查注意力权重是否包含NaN
        if torch.isnan(attention).any():
            print("警告: 注意力权重包含NaN!")
            attention = torch.nan_to_num(attention, nan=1.0/attention.size(-1))
        
        # 应用注意力
        proj_value = self.value_conv(x).view(batch_size, -1, height * width)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, height, width)
        
        # 使用更小的初始gamma值，并限制其最大影响
        gamma_clamped = torch.clamp(self.gamma, -1.0, 1.0)
        return gamma_clamped * out + x


class DisGatedConv2d(nn.Module):
    """稳定版门控卷积层"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(DisGatedConv2d, self).__init__()
        self.conv2d = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias))
        self.mask_conv2d = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias))
        
        # 初始化门控卷积的偏置，使初始阶段大部分门控值接近1
        if bias:
            self.mask_conv2d.bias.data.fill_(1.0)
    
    def forward(self, x, mask=None):
        # 特征通道
        features = self.conv2d(x)
            
        # 生成门控通道
        gates = torch.sigmoid(self.mask_conv2d(x))
        
        # 如果提供了外部掩码，将它与门控机制结合
        if mask is not None:
            if mask.shape[2:] != gates.shape[2:]:
                mask = F.interpolate(mask, size=gates.shape[2:], mode='nearest')
            gates = gates * mask
            
        # 应用门控机制
        return features * gates


class GlobalDiscriminator(nn.Module):
    """简化的全局判别器，减少层数以减轻过拟合"""
    def __init__(self, input_channels=1, input_size=600):
        super(GlobalDiscriminator, self).__init__()
        self.input_shape = (input_channels, input_size, input_size)
        
        # 使用简化版地形特征提取器
        self.terrain_extractor = EnhancedTerrainFeatureExtractor(input_channels)
        # 计算通道数：原始 + 坡度 + 掩码
        enhanced_channels = input_channels * 2 + 1
        
        # 创建空间注意力模块
        self.spatial_attention = nn.Sequential(
            spectral_norm(nn.Conv2d(enhanced_channels, 16, kernel_size=7, padding=3)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(16, 1, kernel_size=7, padding=3)),
            nn.Sigmoid()
        )
        
        # 第一层卷积 - 使用门控卷积处理掩码
        self.conv1 = DisGatedConv2d(enhanced_channels, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        # 第二层卷积
        self.conv2 = DisGatedConv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

        # 第三层卷积
        self.conv3 = DisGatedConv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.act3 = nn.LeakyReLU(0.2, inplace=True)
        
        # 添加自注意力机制
        self.self_attention = StabilizedSelfAttention(256)

        # 第四层卷积
        self.conv4 = DisGatedConv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.act4 = nn.LeakyReLU(0.2, inplace=True)

        # 第五层卷积 - 保留一个额外层以处理大的输入尺寸
        self.conv5 = DisGatedConv2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.act5 = nn.LeakyReLU(0.2, inplace=True)
        
        # 全连接层 - 添加强Dropout
        self.flatten = nn.Flatten()
        
        # 动态计算特征尺寸
        # 计算经过5次下采样后的特征图大小
        h = w = input_size
        for _ in range(5):  # 5层卷积，每层stride=2
            h = (h + 2 * 1 - 4) // 2 + 1
            w = (w + 2 * 1 - 4) // 2 + 1
        
        self.feature_size = 512 * h * w
        
        # 动态创建线性层
        self.linear = nn.Sequential(
            spectral_norm(nn.Linear(self.feature_size, 1024)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5)  # 增加Dropout
        )

    def forward(self, globalin, mask):
        # 检查输入是否包含NaN
        if torch.isnan(globalin).any():
            print("警告: GlobalDiscriminator输入包含NaN!")
            globalin = torch.nan_to_num(globalin, nan=0.0)
            
        mask = mask.float()
        
        # 提取地形特征
        terrain_features = self.terrain_extractor(globalin)
        
        # 拼接特征和掩码
        x = torch.cat([terrain_features, mask], dim=1)
        
        # 应用空间注意力，重点关注有效区域
        attention = self.spatial_attention(x)
        x = x * attention
        
        # 创建下采样掩码进行跟踪
        current_mask = mask
        
        # 第一层卷积，应用门控卷积
        x = self.conv1(x, current_mask)
        x = self.bn1(x)
        x = self.act1(x)
        # 更新掩码尺寸
        current_mask = F.interpolate(current_mask, scale_factor=0.5, mode='nearest')
        
        # 第二层卷积
        x = self.conv2(x, current_mask)
        x = self.bn2(x)
        x = self.act2(x)
        # 更新掩码尺寸
        current_mask = F.interpolate(current_mask, scale_factor=0.5, mode='nearest')
        
        # 第三层卷积
        x = self.conv3(x, current_mask)
        x = self.bn3(x)
        x = self.act3(x)
        
        # 应用自注意力
        x = self.self_attention(x)
        
        # 更新掩码尺寸
        current_mask = F.interpolate(current_mask, scale_factor=0.5, mode='nearest')
        
        # 第四层卷积
        x = self.conv4(x, current_mask)
        x = self.bn4(x)
        x = self.act4(x)
        # 更新掩码尺寸
        current_mask = F.interpolate(current_mask, scale_factor=0.5, mode='nearest')
        
        # 第五层卷积
        x = self.conv5(x, current_mask)
        x = self.bn5(x)
        x = self.act5(x)

        # 扁平化并检查尺寸
        x = self.flatten(x)
        
        # 检查实际特征尺寸与预期尺寸是否匹配
        if x.size(1) != self.feature_size:
            print(f"警告: 特征尺寸不匹配! 预期 {self.feature_size}，实际 {x.size(1)}")
            # 对于add_graph调用时，调整线性层以匹配实际尺寸
            if not hasattr(self, '_feature_size_adjusted') and not self.training:
                self._feature_size_adjusted = True
                self.linear = nn.Sequential(
                    spectral_norm(nn.Linear(x.size(1), 1024)),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(0.5)
                )

        # 全连接层
        x = self.linear(x)
        
        # 检查输出是否包含NaN
        if torch.isnan(x).any():
            print("警告: GlobalDiscriminator输出包含NaN!")
            x = torch.nan_to_num(x, nan=0.0)

        return x


class LocalDiscriminator(nn.Module):
    """简化的局部判别器，减少层数和复杂度"""
    def __init__(self, input_channels=1, input_size=33):
        super(LocalDiscriminator, self).__init__()
        self.input_shape = (input_channels, input_size, input_size)
        
        # 使用简化版地形特征提取器
        self.terrain_extractor = EnhancedTerrainFeatureExtractor(input_channels)
        # 计算输出通道数：原始 + 坡度
        terrain_channels = input_channels * 2
        
        # 第一层卷积 - 使用谱归一化
        self.conv1 = spectral_norm(nn.Conv2d(terrain_channels, 64, kernel_size=4, stride=2, padding=1))
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        # 第二层卷积
        self.conv2 = spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1))
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        
        # 第三层卷积
        self.conv3 = spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1))
        self.bn3 = nn.BatchNorm2d(256)
        self.act3 = nn.LeakyReLU(0.2, inplace=True)

        # 第四层卷积 - 最后一层卷积
        self.conv4 = spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1))
        self.bn4 = nn.BatchNorm2d(512)
        self.act4 = nn.LeakyReLU(0.2, inplace=True)
        
        # 全连接层
        self.flatten = nn.Flatten()
        
        # 动态计算特征尺寸
        h = w = input_size
        for _ in range(4):  # 4层卷积，每层stride=2
            h = (h + 2 * 1 - 4) // 2 + 1
            w = (w + 2 * 1 - 4) // 2 + 1
        
        self.feature_size = 512 * h * w
        
        # 添加更强的Dropout
        self.linear = nn.Sequential(
            spectral_norm(nn.Linear(self.feature_size, 1024)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5)  # 增加Dropout
        )

    def forward(self, local, mask):
        # 检查输入是否包含NaN
        if torch.isnan(local).any():
            print("警告: LocalDiscriminator输入包含NaN!")
            local = torch.nan_to_num(local, nan=0.0)
            
        # 提取地形特征
        x = self.terrain_extractor(local)
        
        # 第一层卷积
        x = self.act1(self.bn1(self.conv1(x)))
        
        # 第二层卷积
        x = self.act2(self.bn2(self.conv2(x)))
        
        # 第三层卷积
        x = self.act3(self.bn3(self.conv3(x)))
        
        # 第四层卷积
        x = self.act4(self.bn4(self.conv4(x)))

        # 特征展平
        x = self.flatten(x)
        
        # 检查实际特征尺寸与预期尺寸是否匹配
        if x.size(1) != self.feature_size:
            print(f"警告: 特征尺寸不匹配! 预期 {self.feature_size}，实际 {x.size(1)}")
            # 对于add_graph调用时，调整线性层以匹配实际尺寸
            if not hasattr(self, '_feature_size_adjusted') and not self.training:
                self._feature_size_adjusted = True
                self.linear = nn.Sequential(
                    spectral_norm(nn.Linear(x.size(1), 1024)),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(0.5)
                )
        
        # 全连接层
        x = self.linear(x)
        
        # 检查输出是否包含NaN
        if torch.isnan(x).any():
            print("警告: LocalDiscriminator输出包含NaN!")
            x = torch.nan_to_num(x, nan=0.0)

        return x

class ContextDiscriminator(nn.Module):
    """简化版上下文判别器，移除了权重共享，增加正则化"""
    def __init__(
        self,
        local_input_channels=1,
        local_input_size=33,
        global_input_channels=1,
        global_input_size=600,
    ):
        super(ContextDiscriminator, self).__init__()
        
        # 创建简化版局部判别器和全局判别器
        self.model_ld = LocalDiscriminator(
            input_channels=local_input_channels, input_size=local_input_size
        )
        self.model_gd = GlobalDiscriminator(
            input_channels=global_input_channels, input_size=global_input_size
        )
        
        # 移除权重共享，避免接口不兼容问题
        # 不再使用: self.model_ld.conv4 = shared_conv

        # 扁平化层
        self.flatten_ld = nn.Flatten()
        self.flatten_gd = nn.Flatten()

        # 特征融合层 - 增加Dropout和更强的正则化
        self.fusion = nn.Sequential(
            spectral_norm(nn.Linear(1024 + 1024, 1024)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.6)  # 增加dropout以提高正则化
        )

        # 分类器 - 更简单的架构
        self.classifier = nn.Sequential(
            spectral_norm(nn.Linear(1024, 512)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            spectral_norm(nn.Linear(512, 1))  # 输出logits，适配BCEWithLogitsLoss
        )

    def forward(self, x_ld, x_lm, x_gd, x_gm):
        # 通过局部和全局判别器
        x_ld = self.model_ld(x_ld, x_lm)  # 1024维特征
        x_gd = self.model_gd(x_gd, x_gm)  # 1024维特征

        # 确保形状正确
        if len(x_ld.shape) > 2:
            x_ld = self.flatten_ld(x_ld)
        if len(x_gd.shape) > 2:
            x_gd = self.flatten_gd(x_gd)

        # 连接特征
        combined = torch.cat([x_ld, x_gd], dim=1)
        
        # 融合特征
        fused = self.fusion(combined)

        # 输出logits，不使用sigmoid
        logits = self.classifier(fused)
        
        # 检查logits是否包含NaN
        if torch.isnan(logits).any():
            print("警告: ContextDiscriminator输出包含NaN!")
            logits = torch.zeros_like(logits, device=logits.device)

        # 为了与原代码兼容，返回logits和特征
        return logits, x_ld, x_gd
    
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test = 2
    # Remove the "logs" folder if it exists
    logs_path = "logsgd"
    if os.path.exists(logs_path):
        shutil.rmtree(logs_path)

    match test:
        case 0:  # Generator
            cn = CompletionNetwork().to(device)
            input1 = torch.ones(1, 1, 33, 33)
            input2 = torch.ones(1, 1, 600, 600)

            input1 = input1.to(device)
            input2 = input2.to(device)
            writer = SummaryWriter("logs")
            writer.add_graph(cn, [input1, input1, input2, input2])
            writer.close()
        case 1:  # Local Discriminator
            ld = LocalDiscriminator().to(device)
            input1 = torch.ones(1, 1, 33, 33)

            input1 = input1.to(device)
            writer = SummaryWriter("logsld")
            writer.add_graph(ld, [input1, input1])
            writer.close()
        case 2:  # Global Discriminator
            gd = GlobalDiscriminator().to(device)
            input1 = torch.ones(1, 1, 600, 600)

            input1 = input1.to(device)
            writer = SummaryWriter("logsgd")
            writer.add_graph(gd, [input1, input1])
            writer.close()

    # Run the tensorboard command
    #os.system("tensorboard --logdir=logs")
    # Open the specified URL directly in the default web browser
    # webbrowser.open("http://localhost:6006/?darkMode=true#graphs&run=.")
