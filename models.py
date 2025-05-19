import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import shutil
import os
import webbrowser
from loss import smooth_mask_generate

'''class EarlyAttentionModule(nn.Module):
    """适用于输入层的注意力模块"""
    def __init__(self, in_channels):
        super(EarlyAttentionModule, self).__init__()
        self.conv = nn.Conv2d(in_channels + 1, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, mask):
        # 确保掩码尺寸与输入特征匹配
        mask_resized = F.interpolate(mask, size=x.shape[2:], mode='nearest')
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
        # mask_resized = F.interpolate(mask, size=x.shape[2:], mode='nearest')
        # 连接特征和掩码
        combined = torch.cat([x, mask], dim=1)
        # 生成注意力权重
        attention = self.sigmoid(self.conv(combined))
        # 应用注意力
        return x * attention
    

def standardize_input(global_input, local_input):
    global_mean = global_input.mean()
    global_std = global_input.std()
    global_input_std = (global_input - global_mean) / (global_std + 1e-8)
    local_input_std = (local_input - global_mean) / (global_std + 1e-8)
    return global_input_std, local_input_std, global_mean, global_std

def unstandardize_output(output, mean, std):
    return output * std + mean

class CompletionNetwork(nn.Module):
    def __init__(self, input_channels=1):
        super(CompletionNetwork, self).__init__()
        
        # 为局部和全局输入添加早期注意力模块
        self.early_local_attention = EarlyAttentionModule(input_channels)
        self.early_global_attention = EarlyAttentionModule(input_channels)

        # Local encoder branch with intermediate outputs saved for skip connections
        self.local_enc1 = nn.Sequential(
            nn.Conv2d(input_channels * 2, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.1)
        )
        self.local_enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.1)
        )
        self.local_enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.1)
        )

        # Global encoder branch
        self.global_enc1 = nn.Sequential(
            nn.Conv2d(input_channels * 2, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
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
        
        # 高级特征的全局注意力模块
        self.global_attention = MaskAttentionModule(256)
        
        # 特征融合的注意力模块
        self.fusion_attention = MaskAttentionModule(256)

        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Decoder with skip connections
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 在解码器第一层后合并跳跃连接的特征
        self.skip_fusion1 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 在解码器第二层后合并跳跃连接的特征
        self.skip_fusion2 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, input_channels, kernel_size=3, stride=1, padding=1),
            # nn.Sigmoid()
        )
        # 平滑过渡层
        self.transition_conv = nn.Conv2d(input_channels + 1, input_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, local_input, local_mask, global_input, global_mask):
        # 统一标准化
        #global_input_std, local_input_std, global_mean, global_std = standardize_input(global_input, local_input)
        input_size = local_input.size()[2:]
        
        # 对全局输入应用早期注意力
        local_input_attended = self.early_local_attention(local_input, local_mask)
        global_input_attended = self.early_global_attention(global_input, global_mask)
        
        # 将输入图像和掩码连接起来
        local_x = torch.cat([local_input_attended, local_mask], dim=1)
        global_x = torch.cat([global_input_attended, global_mask], dim=1)
        
        # 本地编码器的前向传播，保存中间特征用于跳跃连接
        local_feat1 = self.local_enc1(local_x)        # 第一层特征 (64通道)
        local_feat2 = self.local_enc2(local_feat1)    # 第二层特征 (128通道)
        local_feat3 = self.local_enc3(local_feat2)    # 第三层特征 (256通道)
        
        # 全局编码器的前向传播
        global_feat1 = self.global_enc1(global_x)
        global_feat2 = self.global_enc2(global_feat1)
        global_feat3 = self.global_enc3(global_feat2)
        
        """# 将全局特征调整为与本地特征相同的大小
        global_feat3_resized = F.interpolate(
            global_feat3,
            size=local_feat3.size()[2:],
            mode="bilinear",
            align_corners=True
        )"""
        
        # 对高级全局特征应用注意力
        # global_attended = self.global_attention(global_feat3_resized, global_mask)
        
        # 融合本地和全局特征
        #combined_features = torch.cat([local_feat3, global_feat3_resized], dim=1)
        combined_features = torch.cat([local_feat3, global_feat3], dim=1)
        fused_features = self.fusion(combined_features)
        
        # 对融合特征应用注意力
        # resized_local_mask = F.interpolate(local_mask, size=fused_features.shape[2:], mode="bilinear", align_corners=True)
        #fused_attended = self.fusion_attention(fused_features, resized_local_mask)
        
        # 解码器第一层
        #dec1 = self.decoder1(fused_features)
        dec1 = self.decoder1(fused_features)
        
        # 确保dec1的空间尺寸与local_feat2匹配
        # dec1 = F.interpolate(dec1, size=local_feat2.size()[2:], mode="bilinear", align_corners=True)
        
        # 将解码器第一层输出与编码器第二层特征连接（跳跃连接）
        skip1 = torch.cat([dec1, local_feat2], dim=1)
        skip1_fused = self.skip_fusion1(skip1)
        
        # 解码器第二层
        dec2 = self.decoder2(skip1_fused)
        #dec2 = self.decoder2(dec1)
        
        # 确保dec2的空间尺寸与local_feat1匹配
        #dec2 = F.interpolate(dec2, size=local_feat1.size()[2:], mode="bilinear", align_corners=True)
        
        # 将解码器第二层输出与编码器第一层特征连接（跳跃连接）
        skip2 = torch.cat([dec2, local_feat1], dim=1)
        skip2_fused = self.skip_fusion2(skip2)
        
        # 最终输出层
        output = self.final_conv(skip2_fused)
        #output = self.final_conv(dec2)
        
        # 调整输出大小并应用缩放
        # output = F.interpolate(output, size=input_size, mode="bilinear", align_corners=True)
        #output_unstd = unstandardize_output(output, global_mean, global_std)
        
        # 仅在掩码区域应用生成结果（修正的逻辑）
        # local_mask为1的区域保留原始输入，为0的区域应用生成结果
        # final_output = local_input * local_mask + (output) * (1 - local_mask)
        # 创建平滑过渡
        # 使用平均池化创建模糊的掩码边缘
        # blurred_mask = F.avg_pool2d(local_mask, kernel_size=3, stride=1, padding=1)
        blurred_mask = smooth_mask_generate(local_mask, 5.0, False)
        # blurred_mask = blurred_mask*local_mask + (1-local_mask)
        output_with_mask = torch.cat([output, blurred_mask], dim=1)
        smoothed_output = self.transition_conv(output_with_mask)
        
        # 仅在掩码区域应用生成结果 - 平滑过渡
        final_output = local_input * local_mask + smoothed_output * (1 - local_mask)
        # final_output = local_input * (1-blurred_mask) + smoothed_output * blurred_mask
        
        return final_output
        
        return final_output'''

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
    
class GatedConv2d(nn.Module):
    """
    带门控机制的卷积层，可以更好地处理有掩码的输入
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(GatedConv2d, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        
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

class LocalDiscriminator(nn.Module):
    def __init__(self, input_channels=1, input_size=33):
        super(LocalDiscriminator, self).__init__()
        self.input_shape = (input_channels, input_size, input_size)
        self.img_c = input_channels
        self.img_h = input_size
        self.img_w = input_size

        # 计算每一层的输出尺寸
        # 步长为2的卷积: 输出尺寸 = (输入尺寸 + 2*padding - kernel_size)/stride + 1
        h1 = (input_size + 2 * 2 - 5) // 2 + 1  # 第一层后的高度
        w1 = (input_size + 2 * 2 - 5) // 2 + 1  # 第一层后的宽度

        h2 = (h1 + 2 * 2 - 5) // 2 + 1  # 第二层后的高度
        w2 = (w1 + 2 * 2 - 5) // 2 + 1  # 第二层后的宽度

        h3 = (h2 + 2 * 2 - 5) // 2 + 1  # 第三层后的高度
        w3 = (w2 + 2 * 2 - 5) // 2 + 1  # 第三层后的宽度

        h4 = (h3 + 2 * 2 - 5) // 2 + 1  # 第四层后的高度
        w4 = (w3 + 2 * 2 - 5) // 2 + 1  # 第四层后的宽度

        h5 = (h4 + 2 * 2 - 5) // 2 + 1  # 第五层后的高度
        w5 = (w4 + 2 * 2 - 5) // 2 + 1  # 第五层后的宽度

        # 最终特征数量
        self.final_features = 512 * h5 * w5

        # 第一层卷积
        self.conv1 = nn.Conv2d(self.img_c, 64, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU(inplace=True)
        # 尺寸: (None, 64, img_h//2, img_w//2)

        # 第二层卷积
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.ReLU(inplace=True)
        # 尺寸: (None, 128, img_h//4, img_w//4)

        # 第三层卷积
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(256)
        self.act3 = nn.ReLU(inplace=True)
        # 尺寸: (None, 256, img_h//8, img_w//8)

        # 第四层卷积
        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2)
        self.bn4 = nn.BatchNorm2d(512)
        self.act4 = nn.ReLU(inplace=True)
        # 尺寸: (None, 512, img_h//16, img_w//16)

        # 第五层卷积（与原始代码相同）
        self.conv5 = nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=2)
        self.bn5 = nn.BatchNorm2d(512)
        self.act5 = nn.ReLU(inplace=True)
        # 尺寸: (None, 512, img_h//32, img_w//32)

        # 计算特征数量（适应不同输入尺寸）
        # 对于33x33的输入，最后特征图大小为 (512, 2, 2)
        features_h = max(1, self.img_h // 32)
        features_w = max(1, self.img_w // 32)
        in_features = 512 * features_h * features_w

        # 全连接层
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(self.final_features, 1024)
        self.output_act = nn.ReLU(inplace=True)

        # 最后输出层，用于分类
        # self.classifier = nn.Linear(1024, 1)

    def forward(self, local, mask):
        # 卷积层
        # x = torch.cat([local, mask], dim=1)
        x = local
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.act4(self.bn4(self.conv4(x)))
        x = self.act5(self.bn5(self.conv5(x)))

        # 特征展平和全连接层
        x = self.flatten(x)
        x = self.output_act(self.linear(x))

        # 分类
        # output = self.classifier(x)

        return x


class GlobalDiscriminator(nn.Module):
    def __init__(self, input_channels=1, input_size=600):
        super(GlobalDiscriminator, self).__init__()
        self.input_shape = (input_channels, input_size, input_size)
        self.output_shape = (1024,)
        self.img_c = input_channels
        self.img_h = input_size
        self.img_w = input_size

        # 创建注意力模块帮助聚焦有效区域
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(self.img_c * 2, 16, kernel_size=7, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        # 第一层卷积 - 使用GatedConv2d替换普通Conv2d
        self.conv1 = GatedConv2d(self.img_c * 2, 64, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU(inplace=True)
        # 输出尺寸: (batch_size, 64, 300, 300)

        # 第二层卷积 - 使用GatedConv2d替换普通Conv2d
        self.conv2 = GatedConv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.ReLU(inplace=True)
        # 输出尺寸: (batch_size, 128, 150, 150)

        # 第三层卷积 - 使用GatedConv2d替换普通Conv2d
        self.conv3 = GatedConv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(256)
        self.act3 = nn.ReLU(inplace=True)
        # 输出尺寸: (batch_size, 256, 75, 75)

        # 第四层卷积 - 使用GatedConv2d替换普通Conv2d
        self.conv4 = GatedConv2d(256, 512, kernel_size=5, stride=2, padding=2)
        self.bn4 = nn.BatchNorm2d(512)
        self.act4 = nn.ReLU(inplace=True)
        # 输出尺寸: (batch_size, 512, 38, 38)

        # 第五层卷积 - 使用GatedConv2d替换普通Conv2d
        self.conv5 = GatedConv2d(512, 512, kernel_size=5, stride=2, padding=2)
        self.bn5 = nn.BatchNorm2d(512)
        self.act5 = nn.ReLU(inplace=True)
        # 输出尺寸: (batch_size, 512, 19, 19)

        # 第六层卷积 - 使用GatedConv2d替换普通Conv2d
        self.conv6 = GatedConv2d(512, 512, kernel_size=5, stride=2, padding=2)
        self.bn6 = nn.BatchNorm2d(512)
        self.act6 = nn.ReLU(inplace=True)
        # 输出尺寸: (batch_size, 512, 10, 10)

        # 第七层卷积 - 使用GatedConv2d替换普通Conv2d
        self.conv7 = GatedConv2d(512, 512, kernel_size=5, stride=2, padding=2)
        self.bn7 = nn.BatchNorm2d(512)
        self.act7 = nn.ReLU(inplace=True)
        # 输出尺寸: (batch_size, 512, 5, 5)

        # 扁平化
        self.flatten = nn.Flatten()
        # 输出尺寸: (batch_size, 512*5*5)

        # 全连接层
        self.linear = nn.Linear(512 * 5 * 5, 1024)
        self.final_act = nn.ReLU(inplace=True)
        # 输出尺寸: (batch_size, 1024)

    def forward(self, globalin, mask):
        mask = mask.float()
        # 拼接输入和掩码
        x = torch.cat([globalin, mask], dim=1)
        attention = self.spatial_attention(x)
        x=x*attention
        
        # 创建下采样掩码进行跟踪
        current_mask = mask
        
        # 第一层卷积，应用GatedConv
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
        # 更新掩码尺寸
        current_mask = F.interpolate(current_mask, scale_factor=0.5, mode='nearest')
        
        # 第六层卷积
        x = self.conv6(x, current_mask)
        x = self.bn6(x)
        x = self.act6(x)
        # 更新掩码尺寸
        current_mask = F.interpolate(current_mask, scale_factor=0.5, mode='nearest')
        
        # 第七层卷积
        x = self.conv7(x, current_mask)
        x = self.bn7(x)
        x = self.act7(x)

        # 扁平化
        x = self.flatten(x)

        # 全连接层
        x = self.linear(x)
        x = self.final_act(x)

        return x


class ContextDiscriminator(nn.Module):
    def __init__(
        self,
        local_input_channels=1,
        local_input_size=33,
        global_input_channels=1,
        global_input_size=600,
    ):
        super(ContextDiscriminator, self).__init__()

        # 定义输入形状
        local_input_shape = (local_input_channels, local_input_size, local_input_size)
        global_input_shape = (
            global_input_channels,
            global_input_size,
            global_input_size,
        )

        self.input_shape = [local_input_shape, global_input_shape]
        self.output_shape = (1,)

        # 创建局部判别器和全局判别器
        self.model_ld = LocalDiscriminator(
            input_channels=local_input_channels, input_size=local_input_size
        )
        self.model_gd = GlobalDiscriminator(
            input_channels=global_input_channels, input_size=global_input_size
        )

        # 连接层
        self.flatten_ld = nn.Flatten()
        self.flatten_gd = nn.Flatten()

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(1024 + 1024, 512),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            # nn.Sigmoid(),
        )

    def forward(self, x_ld, x_lm, x_gd, x_gm):

        # 通过局部和全局判别器
        x_ld = self.model_ld(x_ld, x_lm)  # 应该已经是1024维特征
        x_gd = self.model_gd(x_gd, x_gm)  # 应该已经是1024维特征

        # 确保形状正确
        if len(x_ld.shape) > 2:
            x_ld = self.flatten_ld(x_ld)
        if len(x_gd.shape) > 2:
            x_gd = self.flatten_gd(x_gd)

        # 连接特征
        combined = torch.cat([x_ld, x_gd], dim=1)

        # 分类
        out = self.classifier(combined)

        return out, x_ld, x_gd


"""class GlobalDiscriminator(nn.Module):
    def __init__(self, input_channels=1):
        super(GlobalDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),
        )

    def forward(self, x):
        return self.model(x)"""


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test = 1
    # Remove the "logs" folder if it exists
    logs_path = "logsld"
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
            writer = SummaryWriter("logs")
            writer.add_graph(gd, [input1, input1])
            writer.close()

    # Run the tensorboard command
    #os.system("tensorboard --logdir=logs")
    # Open the specified URL directly in the default web browser
    # webbrowser.open("http://localhost:6006/?darkMode=true#graphs&run=.")
