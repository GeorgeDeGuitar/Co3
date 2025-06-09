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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Parameter

def l2normalize(v, eps=1e-12):
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
    def __init__(self, num_features, eps=1e-8, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = Parameter(torch.Tensor(num_features).uniform_())
            self.beta = Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        if x.size(0) == 1:
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 pad_type='reflect', activation='lrelu', norm='none', sn=False):
        super(GatedConv2d, self).__init__()
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            raise ValueError(f"Unsupported padding type: {pad_type}")

        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'ln':
            self.norm = LayerNorm(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            raise ValueError(f"Unsupported normalization: {norm}")

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        if sn:
            self.conv2d = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation))
            self.mask_conv2d = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation))
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation)
            self.mask_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pad(x)
        conv = self.conv2d(x)
        mask_conv = self.mask_conv2d(x)
        gated_mask = self.sigmoid(mask_conv)
        x = conv * gated_mask
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class PartialGatedConv2d(nn.Module):
    """
    结合Partial Convolution和Gated Convolution的模块
    流程：Gated Conv → Partial Conv Mask → 相乘输出
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 pad_type='reflect', activation='lrelu', norm='none', sn=False, multi_channel=False):
        super(PartialGatedConv2d, self).__init__()

        # 存储参数
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.multi_channel = multi_channel
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)

        # 填充层
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            raise ValueError(f"Unsupported padding type: {pad_type}")

        # Gated Convolution的两个分支
        if sn:
            self.conv_feature = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=self.padding, dilation=dilation))
            self.conv_gating = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=self.padding, dilation=dilation))
        else:
            self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=self.padding, dilation=dilation)
            self.conv_gating = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=self.padding, dilation=dilation)
        
        self.sigmoid = nn.Sigmoid()

        # Partial Convolution的mask更新机制（参考官方实现）
        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])
        
        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * self.weight_maskUpdater.shape[3]
        self.last_size = (None, None, None, None)
        self.update_mask = None

        # 归一化层
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'ln':
            self.norm = LayerNorm(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            raise ValueError(f"Unsupported normalization: {norm}")

        # 激活函数
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        
        # 1. 应用填充
        # input_padded = self.pad(input)
        
        # 2. Gated Convolution：计算feature和gating
        feature = self.conv_feature(input)
        gating = self.sigmoid(self.conv_gating(input))
        
        # 3. Partial Convolution的mask更新机制
        with torch.no_grad():
            if self.weight_maskUpdater.type() != input.type():
                self.weight_maskUpdater = self.weight_maskUpdater.to(input)
            
            # 关键修正：确保mask与当前输入尺寸匹配
            if mask_in is None:
                if self.multi_channel:
                    current_mask = torch.ones_like(input)
                else:
                    current_mask = torch.ones(input.shape[0], 1, input.shape[2], input.shape[3]).to(input)
            else:
                '''# 如果传入的mask与当前输入尺寸不匹配，调整到输入尺寸
                if mask_in.shape[2:] != input.shape[2:]:

                    current_mask = F.interpolate(mask_in, size=input.shape[2:], mode='nearest')
                else:'''
                current_mask = mask_in
            
            # 使用与卷积相同的参数计算mask更新
            self.update_mask = F.conv2d(current_mask, self.weight_maskUpdater, bias=None, 
                                      stride=self.stride, padding=self.padding, 
                                      dilation=self.dilation, groups=1)
            self.update_mask = torch.clamp(self.update_mask, 0, 1)
        
        # 4. 验证空间尺寸
        # print(f"Input: {input.shape}, Feature: {feature.shape}, Gating: {gating.shape}, Mask: {self.update_mask.shape}")
        
        # 检查空间尺寸是否匹配
        assert feature.shape[2:] == gating.shape[2:] == self.update_mask.shape[2:], \
            f"空间尺寸不匹配: feature={feature.shape[2:]}, gating={gating.shape[2:]}, mask={self.update_mask.shape[2:]}"
        
        # 5. 三者相乘（PyTorch会自动处理广播）
        output = feature * gating * self.update_mask
        
        # 6. 归一化和激活
        if self.norm:
            output = self.norm(output)
        if self.activation:
            output = self.activation(output)

        return output, self.update_mask


    
class PartialMaxPool2d(nn.Module):
    """
    更灵活的Partial MaxPool，允许部分有效窗口
    """
    def __init__(self, kernel_size, stride=None, padding=0, valid_threshold=0.5):
        super(PartialMaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.valid_threshold = valid_threshold  # 有效性阈值
        
        self.register_buffer('weight_maskUpdater', 
                           torch.ones(1, 1, kernel_size, kernel_size))
        self.slide_winsize = kernel_size * kernel_size
        
    def forward(self, x, mask):
        # 特征池化
        pooled_x = F.max_pool2d(x, kernel_size=self.kernel_size, 
                               stride=self.stride, padding=self.padding)
        
        # Mask更新
        with torch.no_grad():
            mask_sum = F.conv2d(mask, self.weight_maskUpdater, 
                               stride=self.stride, padding=self.padding)
            
            # 使用阈值策略：窗口内有效像素比例超过阈值才认为有效
            mask_ratio = mask_sum / self.slide_winsize
            updated_mask = (mask_ratio >= self.valid_threshold).float()
        
        return pooled_x, updated_mask
    
class TransposeGatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 pad_type='zero', activation='lrelu', norm='none', sn=True, scale_factor=2):
        super(TransposeGatedConv2d, self).__init__()
        self.scale_factor = scale_factor
        self.gated_conv2d = GatedConv2d(in_channels, out_channels, kernel_size, stride, padding,
                                      dilation, pad_type, activation, norm, sn)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        x = self.gated_conv2d(x)
        return x

class EarlyAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(EarlyAttentionModule, self).__init__()
        self.conv = nn.Conv2d(in_channels + 1, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask):
        # mask_resized = F.interpolate(mask, size=x.shape[2:], mode='bilinear', align_corners=True)
        combined = torch.cat([x, mask], dim=1)
        attention = self.sigmoid(self.conv(combined))
        return x * attention
'''# 方案3：针对DEM任务优化的版本
class EarlyAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super(EarlyAttentionModule, self).__init__()
        
        # 使用更小的中间层，减少参数
        mid_channels = max(1, in_channels // reduction_ratio)
        
        # 第一层：降维 + gated convolution
        self.conv1 = GatedConv2d(
            in_channels + 1, 
            mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            pad_type='zero',
            activation='relu',    # 中间层使用ReLU
            norm='bn',           # 使用批归一化稳定训练
            sn=False
        )
        
        # 第二层：升维 + 输出attention
        self.conv2 = GatedConv2d(
            mid_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            pad_type='zero',
            activation='none',   # 输出层不使用激活
            norm='none',
            sn=False
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask):
        # 输入特征和mask的融合
        combined = torch.cat([x, mask], dim=1)
        
        # 两层gated convolution
        attention = self.conv1(combined)
        attention = self.conv2(attention)
        
        # 生成attention权重
        attention = self.sigmoid(attention)
        
        # 应用attention
        return x * attention'''

class MaskAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(MaskAttentionModule, self).__init__()
        self.conv = nn.Conv2d(in_channels + 1, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask):
        mask_resized = F.interpolate(mask, size=x.shape[2:], mode='bilinear', align_corners=True)
        combined = torch.cat([x, mask_resized], dim=1)
        attention = self.sigmoid(self.conv(combined))
        return x * attention

class BoundaryAwareModule(nn.Module):
    def __init__(self, in_channels):
        super(BoundaryAwareModule, self).__init__()
        self.boundary_conv = nn.Sequential(
            nn.Conv2d(in_channels + 1, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.fusion = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, features, mask):
        boundary_mask = self._get_boundary_mask(mask)
        boundary_input = torch.cat([features, boundary_mask], dim=1)
        boundary_features = self.boundary_conv(boundary_input)
        combined = torch.cat([features, boundary_features], dim=1)
        enhanced_features = self.fusion(combined)
        return enhanced_features

    def _get_boundary_mask(self, mask):
        dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
        eroded = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
        boundary = torch.abs(dilated - eroded)
        return boundary

class CompletionNetwork(nn.Module):
    def __init__(self, input_channels=1):
        super(CompletionNetwork, self).__init__()
        self.debug = False

        # Early attention modules for local and global inputs
        self.early_local_attention = EarlyAttentionModule(input_channels)
        self.early_global_attention = EarlyAttentionModule(input_channels)

        # Local encoder with GatedConv2d
        '''self.local_enc1 = nn.Sequential(
            GatedConv2d(input_channels * 2, 64, kernel_size=5, stride=1, padding=2,
                        pad_type='reflect', activation='relu', norm='bn'),
        )'''
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
        '''self.local_enc2 = nn.Sequential(
            GatedConv2d(64, 128, kernel_size=3, stride=2, padding=1,
                        pad_type='reflect', activation='relu', norm='bn'),
            GatedConv2d(128, 128, kernel_size=3, stride=1, padding=1,
                        pad_type='reflect', activation='relu', norm='bn'),
        )
        self.local_enc3 = nn.Sequential(
            GatedConv2d(128, 256, kernel_size=3, stride=2, padding=1,
                        pad_type='reflect', activation='relu', norm='bn'),
            GatedConv2d(256, 256, kernel_size=3, stride=1, padding=1,
                        pad_type='reflect', activation='relu', norm='bn'),
            GatedConv2d(256, 256, kernel_size=3, stride=1, padding=1,
                        pad_type='reflect', activation='relu', norm='bn'),
        )'''

        # Global encoder with PartialGatedConv2d
        self.global_enc1_conv = PartialGatedConv2d(input_channels * 2, 64, kernel_size=5, stride=2, padding=2,
                                                  pad_type='reflect', activation='relu', norm='bn')
        self.global_enc1_pool = PartialMaxPool2d(kernel_size=2, stride=2)
        
        '''self.global_enc2_conv = PartialGatedConv2d(64, 128, kernel_size=3, stride=2, padding=1,
                                                  pad_type='reflect', activation='relu', norm='bn')
        self.global_enc2_pool = PartialMaxPool2d(kernel_size=2, stride=2)
        
        self.global_enc3_conv1 = PartialGatedConv2d(128, 256, kernel_size=3, stride=2, padding=1,
                                                   pad_type='reflect', activation='relu', norm='bn')
        self.global_enc3_conv2 = PartialGatedConv2d(256, 256, kernel_size=3, stride=2, padding=0,
                                                   pad_type='reflect', activation='relu', norm='bn')'''
        self.global_enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.global_enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Global attention module for high-level features
        self.global_attention = MaskAttentionModule(256)
        self.fusion_attention = MaskAttentionModule(256)

        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(512, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.boundary_aware = BoundaryAwareModule(256)

        # Decoder
        '''self.decoder1 = TransposeGatedConv2d(256, 128, kernel_size=4, stride=2, padding=1,
                                            pad_type='reflect', activation='relu', norm='bn', scale_factor=2)'''
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.skip_fusion1 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        '''self.decoder2 = TransposeGatedConv2d(128, 64, kernel_size=4, stride=2, padding=1,
                                            pad_type='reflect', activation='relu', norm='bn', scale_factor=2)'''
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
                                            
        self.skip_fusion2 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        '''self.decoder3 = nn.Sequential(
            GatedConv2d(64, 32, kernel_size=3, stride=1, padding=1,
                        pad_type='reflect', activation='relu', norm='bn'),
            GatedConv2d(32, input_channels, kernel_size=3, stride=1, padding=1,
                        pad_type='reflect', activation='none', norm='none'),
        )'''
        self.decoder3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, input_channels, kernel_size=3, stride=1, padding=1)
        )

        # Transition layers
        self.transition_conv = nn.Conv2d(input_channels + 1, input_channels, kernel_size=3, stride=1, padding=1)

        self.offset_module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(input_channels * 2, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
            nn.Tanh()
        )
        self.offset_scale = nn.Parameter(torch.tensor(20.0))
        self.offset_transition = nn.Sequential(
            nn.Conv2d(input_channels + 1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, input_channels, kernel_size=3, padding=1)
        )
        '''self.offset_transition = nn.Sequential(
            GatedConv2d(input_channels + 1, 16, kernel_size=3, stride=1, padding=1,
                        pad_type='reflect', activation='relu', norm='bn'),
            GatedConv2d(16, input_channels, kernel_size=3, stride=1, padding=1,
                        pad_type='reflect', activation='none', norm='none')
        )
'''
    def print_sizes(self, tensor, name):
        if self.debug:
            print(f"{name} size: {tensor.size()}")

    def forward(self, local_input, local_mask, global_input, global_mask):
        local_mask = local_mask.float()
        global_mask = global_mask.float()

        # Apply early attention
        local_input_attended = self.early_local_attention(local_input, local_mask)
        global_input_attended = self.early_global_attention(global_input, global_mask)

        # Concatenate input and mask
        local_x = torch.cat([local_input_attended, local_mask], dim=1)
        global_x = torch.cat([global_input_attended, global_mask], dim=1)

        # Local encoder forward pass with GatedConv2d (no mask input)
        local_feat1 = self.local_enc1(local_x)
        # self.print_sizes(local_feat1, "local_feat1")

        local_feat2 = self.local_enc2(local_feat1)
        # self.print_sizes(local_feat2, "local_feat2")

        local_feat3 = self.local_enc3(local_feat2)
        # self.print_sizes(local_feat3, "local_feat3")

        # Global encoder forward pass with PartialGatedConv2d - 修正版本
        global_mask_current = global_mask
        
        global_feat1, global_mask_current = self.global_enc1_conv(global_x, global_mask_current)
        global_feat1, global_mask_current = self.global_enc1_pool(global_feat1, global_mask_current)
        # self.print_sizes(global_feat1, "global_feat1")

        # global_feat2, global_mask_current = self.global_enc2_conv(global_feat1, global_mask_current)
        # global_feat2, global_mask_current = self.global_enc2_pool(global_feat2, global_mask_current)
        # self.print_sizes(global_feat2, "global_feat2")

        # global_feat3, global_mask_current = self.global_enc3_conv1(global_feat2, global_mask_current)
        # self.print_sizes(global_feat3, "global_feat3_conv1")
        
        # global_feat3, global_mask_current = self.global_enc3_conv2(global_feat3, global_mask_current)
        # self.print_sizes(global_feat3, "global_feat3_conv2")
        global_feat2 = self.global_enc2(global_feat1)
        
        global_feat3 = self.global_enc3(global_feat2)

        # Resize global features to match local features
        '''global_feat3_resized = F.interpolate(
            global_feat3, size=local_feat3.size()[2:], mode="bilinear", align_corners=True
        )
        global_mask_resized = F.interpolate(
            global_mask_current, size=local_feat3.size()[2:], mode="bilinear", align_corners=True
        )'''
        # self.print_sizes(global_feat3, "global_feat3_resized")

        # Feature fusion
        combined_features = torch.cat([local_feat3, global_feat3], dim=1)
        # self.print_sizes(combined_features, "combined_features")

        fused_features = self.fusion(combined_features)
        # self.print_sizes(fused_features, "fused_features")

        # Decoder with TransposeGatedConv2d (no mask input)
        dec1 = self.decoder1(fused_features)
        # self.print_sizes(dec1, "dec1")

        '''if dec1.size()[2:] != local_feat2.size()[2:]:
            dec1 = F.interpolate(dec1, size=local_feat2.size()[2:], mode='bilinear', align_corners=True)'''
            # self.print_sizes(dec1, "dec1_resized")

        skip1 = torch.cat([dec1, local_feat2], dim=1)
        # self.print_sizes(skip1, "skip1")

        skip1_fused = self.skip_fusion1(skip1)
        # self.print_sizes(skip1_fused, "skip1_fused")

        dec2 = self.decoder2(skip1_fused)
        # self.print_sizes(dec2, "dec2")

        '''if dec2.size()[2:] != local_feat1.size()[2:]:
            dec2 = F.interpolate(dec2, size=local_feat1.size()[2:], mode='bilinear', align_corners=True)'''
            # self.print_sizes(dec2, "dec2_resized")

        skip2 = torch.cat([dec2, local_feat1], dim=1)
        # self.print_sizes(skip2, "skip2")

        skip2_fused = self.skip_fusion2(skip2)
        # self.print_sizes(skip2_fused, "skip2_fused")

        # Final output with GatedConv2d (no mask input)
        output = self.decoder3(skip2_fused)
        # self.print_sizes(output, "output")

        # Smooth transition
        # blurred_mask = F.avg_pool2d(local_mask, kernel_size=3, stride=1, padding=1)
        blurred_mask = smooth_mask_generate(local_mask, 7.0, False)
        fused_mask = torch.max(blurred_mask, 1 - local_mask)

        '''known_region = local_input * local_mask
        generated_region = output * (1 - local_mask)

        offset_features = torch.cat([
            known_region.mean(dim=(2, 3), keepdim=True),
            generated_region.mean(dim=(2, 3), keepdim=True)
        ], dim=1)

        offset = self.offset_module(offset_features)
        offset = offset.view(-1, 1, 1, 1)
        # print("offset value:", offset)

        offset_output = output + offset'''
        transition_input = torch.cat([output, fused_mask], dim=1)
        smoothed_output = self.offset_transition(transition_input)

        final_output = (local_input * local_mask + smoothed_output * (1 - local_mask))
        return final_output

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
    """使用PartialGatedConv2d的全局判别器"""
    def __init__(self, input_channels=1, input_size=600):
        super(GlobalDiscriminator, self).__init__()
        self.input_shape = (input_channels, input_size, input_size)
        
        # 使用简化版地形特征提取器
        self.terrain_extractor = EnhancedTerrainFeatureExtractor(input_channels)
        # 计算通道数：原始 + 坡度 + 掩码
        enhanced_channels = input_channels * 2 + 1
        
        # 创建空间注意力模块 - 替换为PartialGatedConv2d
        self.spatial_attention_conv1 = PartialGatedConv2d(enhanced_channels, 16, kernel_size=7, 
                                                         stride=1, padding=3, activation='lrelu', 
                                                         norm='bn', sn=True)
        self.spatial_attention_conv2 = PartialGatedConv2d(16, 1, kernel_size=7, 
                                                         stride=1, padding=3, activation='sigmoid', 
                                                         norm='none', sn=True)
        
        # 第一层卷积 - 替换为PartialGatedConv2d
        self.conv1 = PartialGatedConv2d(enhanced_channels, 64, kernel_size=4, stride=2, padding=1,
                                       activation='lrelu', norm='bn', sn=True)

        # 第二层卷积
        self.conv2 = PartialGatedConv2d(64, 128, kernel_size=4, stride=2, padding=1,
                                       activation='lrelu', norm='bn', sn=True)

        # 第三层卷积
        self.conv3 = PartialGatedConv2d(128, 256, kernel_size=4, stride=2, padding=1,
                                       activation='lrelu', norm='bn', sn=True)
        
        # 添加自注意力机制
        self.self_attention = StabilizedSelfAttention(256)

        # 第四层卷积
        self.conv4 = PartialGatedConv2d(256, 512, kernel_size=4, stride=2, padding=1,
                                       activation='lrelu', norm='bn', sn=True)

        # 第五层卷积 - 保留一个额外层以处理大的输入尺寸
        self.conv5 = PartialGatedConv2d(512, 512, kernel_size=4, stride=2, padding=1,
                                       activation='lrelu', norm='bn', sn=True)
        
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
        
        # 应用空间注意力，使用PartialGatedConv2d
        attention, attention_mask = self.spatial_attention_conv1(x, mask)
        attention, attention_mask = self.spatial_attention_conv2(attention, attention_mask)
        
        # 应用注意力权重
        x = x * attention
        
        # 创建当前掩码进行跟踪
        current_mask = mask
        
        # 第一层卷积，使用PartialGatedConv2d（集成了归一化和激活）
        x, current_mask = self.conv1(x, current_mask)
        
        # 第二层卷积
        x, current_mask = self.conv2(x, current_mask)
        
        # 第三层卷积
        x, current_mask = self.conv3(x, current_mask)
        
        # 应用自注意力
        x = self.self_attention(x)
        
        # 第四层卷积
        x, current_mask = self.conv4(x, current_mask)
        
        # 第五层卷积
        x, current_mask = self.conv5(x, current_mask)

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
    """使用GatedConv2d的局部判别器，减少层数和复杂度"""
    def __init__(self, input_channels=1, input_size=33):
        super(LocalDiscriminator, self).__init__()
        self.input_shape = (input_channels, input_size, input_size)
        
        # 使用简化版地形特征提取器
        self.terrain_extractor = EnhancedTerrainFeatureExtractor(input_channels)
        # 计算输出通道数：原始 + 坡度
        terrain_channels = input_channels * 2
        
        # 第一层卷积 - 替换为GatedConv2d
        self.conv1 = GatedConv2d(terrain_channels, 64, kernel_size=4, stride=2, padding=1,
                                activation='lrelu', norm='bn', sn=True)
        
        # 第二层卷积
        self.conv2 = GatedConv2d(64, 128, kernel_size=4, stride=2, padding=1,
                                activation='lrelu', norm='bn', sn=True)
        
        # 第三层卷积
        self.conv3 = GatedConv2d(128, 256, kernel_size=4, stride=2, padding=1,
                                activation='lrelu', norm='bn', sn=True)
        
        # 第四层卷积 - 最后一层卷积
        self.conv4 = GatedConv2d(256, 512, kernel_size=4, stride=2, padding=1,
                                activation='lrelu', norm='bn', sn=True)
        
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
        
        # 第一层卷积 - GatedConv2d（集成了归一化和激活）
        x = self.conv1(x)
        
        # 第二层卷积
        x = self.conv2(x)
        
        # 第三层卷积
        x = self.conv3(x)
        
        # 第四层卷积
        x = self.conv4(x)

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
    """上下文判别器，融合Local和Global的1024维特征"""
    def __init__(
        self,
        local_input_channels=1,
        local_input_size=33,
        global_input_channels=1,
        global_input_size=600,
    ):
        super(ContextDiscriminator, self).__init__()
        
        # 创建局部判别器和全局判别器
        self.model_ld = LocalDiscriminator(
            input_channels=local_input_channels, input_size=local_input_size
        )
        self.model_gd = GlobalDiscriminator(
            input_channels=global_input_channels, input_size=global_input_size
        )

        # 特征融合层 - 1024 + 1024 = 2048 -> 1024
        self.fusion = nn.Sequential(
            spectral_norm(nn.Linear(1024 + 1024, 1024)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.6)
        )

        # 分类器
        self.classifier = nn.Sequential(
            spectral_norm(nn.Linear(1024, 512)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            spectral_norm(nn.Linear(512, 1))
        )

    def forward(self, x_ld, x_lm, x_gd, x_gm):
        # 通过局部和全局判别器，都输出1024维特征
        x_ld = self.model_ld(x_ld, x_lm)  # 1024维特征
        x_gd = self.model_gd(x_gd, x_gm)  # 1024维特征

        # 连接特征 -> 2048维
        combined = torch.cat([x_ld, x_gd], dim=1)
        
        # 融合特征 -> 1024维
        fused = self.fusion(combined)

        # 分类 -> 1维输出
        logits = self.classifier(fused)
        
        # 检查logits是否包含NaN
        if torch.isnan(logits).any():
            print("警告: ContextDiscriminator输出包含NaN!")
            logits = torch.zeros_like(logits, device=logits.device)

        # 返回logits和特征
        return logits, x_ld, x_gd
    
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test = 0
    # Remove the "logs" folder if it exists
    logs_path = "logscn"
    if os.path.exists(logs_path):
        shutil.rmtree(logs_path)

    match test:
        case 0:  # Generator
            cn = CompletionNetwork().to(device)
            input1 = torch.ones(1, 1, 33, 33)
            input2 = torch.ones(1, 1, 600, 600)

            input1 = input1.to(device)
            input2 = input2.to(device)
            writer = SummaryWriter("logscn")
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
