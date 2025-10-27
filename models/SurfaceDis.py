import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import spectral_norm
import matplotlib.pyplot as plt
import os

save_path = r"e:\KingCrimson Dataset\Comparation\BQDvis\BQD65"
global cnt
cnt=0

def _tensor_to_numpy(tensor):
    """将张量转换为numpy数组用于保存图片"""
    if tensor.dim() == 4:  # [B, C, H, W]
        # 取第一个batch和第一个channel
        tensor = tensor[0, 0]
    elif tensor.dim() == 3:  # [C, H, W]
        tensor = tensor[0]
    elif tensor.dim() == 2:  # [H, W]
        pass
    else:
        raise ValueError(f"Unexpected tensor dimension: {tensor.dim()}")
    
    # 转换为numpy并移动到CPU
    return tensor.detach().cpu().numpy()

def _save_terrain_image(data, filename, binary=False, inc=False):
    """
    保存地形图或二值图样式的图片
    
    Args:
        data: 输入数据（tensor或numpy数组）
        filename: 文件名
        save_path: 保存路径
        binary: 是否绘制二值图（黑白），False为地形图
    """
    if inc:
        global cnt
        cnt+=1
    # 转换数据
    if torch.is_tensor(data):
        data = _tensor_to_numpy(data)
    
    # 创建图形，去掉所有边框和空白
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 选择颜色映射
    if binary:
        # 二值图：黑白
        cmap = 'gray'
        # 确保数据在[0,1]范围内用于二值显示
        vmin, vmax = 0, 1
    else:
        # 地形图
        cmap = 'terrain'
        vmin, vmax = None, None
    
    # 显示图像
    ax.imshow(data, cmap=cmap, aspect='equal', vmin=vmin, vmax=vmax)
    
    # 移除所有装饰元素
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    
    # 移除图形周围的空白
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # 保存图片
    filepath = os.path.join(save_path, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()  # 释放内存
    print(f"Saved: {filepath}")

class BoundaryQualityDiscriminator(nn.Module):
    """
    DEM补全区域边界质量判别器
    用于评估DEM补全结果在边界处的连续性和质量
    支持多种输入尺寸：33x33, 65x65, 129x129, 257x257
    优化版本：固定5层或根据尺寸增加深度，使用膨胀卷积增大感受野，保持分辨率以精确提取2像素宽条带
    """
    def __init__(self, input_channels=1, input_size=33):
        super().__init__()
        self.input_size = input_size
        # 边界检测器（假设已定义）
        self.true_extractor = BoundaryDetector(input_size)
        print(f"input_size: {input_size}, input_channels: {input_channels}")
        
        # 根据输入尺寸调整基础通道数和层数
        if input_size <= 33:
            base_channels = 32
            num_layers = 5
            dilation_pattern = [1, 1, 1, 1, 1]  # 小尺寸无需膨胀
        elif input_size <= 65:
            base_channels = 48
            num_layers = 5
            dilation_pattern = [1, 1, 2, 1, 1]  # 中间层轻微膨胀
        elif input_size <= 129:
            base_channels = 64
            num_layers = 6  # 增加1层以适应更大输入
            dilation_pattern = [1, 1, 2, 2, 1, 1]  # 中间层适度膨胀
        else:  # 257
            base_channels = 64
            num_layers = 5  # 增加到7层以增强容量
            dilation_pattern = [1, 1, 1, 1, 1]  # 中间层更大膨胀
        
        # 边界特征提取网络
        layers = []
        in_ch = input_channels
        
        for i in range(num_layers):
            out_ch = base_channels * (2 ** min(i, 3))  # 限制最大通道数
            dilation = dilation_pattern[i]
            padding = dilation  # padding等于dilation以保持尺寸
            
            layers.extend([
                nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=padding, dilation=dilation),
                nn.ReLU(inplace=True),  # ReLU保持边缘锐利
            ])
            in_ch = out_ch
        
        self.boundary_extractor = nn.Sequential(*layers)
        
        # 添加最终输出层，保持3x3卷积核，无膨胀
        self.final_conv = nn.Conv2d(in_ch, 1, kernel_size=3, padding=1)
        
        # 添加sigmoid激活，确保输出在[0,1]范围内
        self.sigmoid = nn.Sigmoid()
        self.loss = F.binary_cross_entropy

    def forward(self, generated_data, mask):
        """
        前向传播
        Args:
            generated_data: [B, C, H, W] 生成器输出的数据
            mask: [B, C, H, W] mask
        Returns:
            extracted_boundary: [B, C, H, W] 提取的边界
            loss_full: 完整损失
            loss_mask: mask损失
        """
        # 提取边界特征
        boundary_features = self.boundary_extractor(generated_data)
        
        # 映射到单通道，无需上采样（分辨率已保持）
        extracted_boundary = self.final_conv(boundary_features)
        
        # 应用sigmoid激活
        extracted_boundary = self.sigmoid(extracted_boundary)
        # _save_terrain_image(extracted_boundary, f"extracted_boundary_{self.input_size}_{cnt}.png", True)

        # 获取真实边界
        true_boundary = self.true_extractor(mask)
        true_boundary = true_boundary.detach()
        # _save_terrain_image(true_boundary, f"true_boundary_{self.input_size}_{cnt}.png", True)
        # _save_terrain_image(mask, f"input_mask_{self.input_size}_{cnt}.png", True)
        # _save_terrain_image(generated_data, f"generated_data_{self.input_size}_{cnt}.png", False, True)

        # 计算损失
        loss_full = self.loss(extracted_boundary, true_boundary)
        loss_full = loss_full * 10

        loss_mask = self.loss(
            extracted_boundary * true_boundary, torch.zeros_like(true_boundary)
        )

        # 除以mask=1的数量，再乘以mask整体值数量
        mask_sum = true_boundary.sum()
        total_sum = true_boundary.numel()
        if mask_sum > 0:
            loss_mask = loss_mask / mask_sum * total_sum
        else:
            print("Warning: mask_sum is zero, loss_mask will be zero.")
            loss_mask = loss_mask
        loss_mask = loss_mask * 0.1

        return extracted_boundary, loss_full, loss_mask


class BoundaryDetector(nn.Module):
    """
    边界检测模块

    检测mask边界（补全区域与非补全区域的交界）
    """

    def __init__(self, input_size=33):
        super().__init__()
        self.input_size = input_size

    def forward(self, mask):
        """
        检测mask边界

        Args:
            mask: [B, 1, H, W] 掩码（1表示有效数据，0表示需要补全的区域）

        Returns:
            boundary: [B, 1, H, W] 边界图
        """
        return self._detect_mask_boundary_conv(mask)

    def _detect_mask_boundary_conv(self, mask):
        """使用卷积操作检测mask边界"""
        # 确保mask是二值的
        mask_binary = (mask > 0.5).float()

        # 定义边缘检测卷积核
        edge_kernel = (
            torch.tensor(
                [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                dtype=torch.float32,
                device=mask.device,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

        # 使用卷积检测边界
        edge_response = F.conv2d(mask_binary, edge_kernel, padding=1)
        boundary = (edge_response.abs() > 0.1).float()

        # 排除最外圈像素
        boundary[:, :, 0, :] = 0
        boundary[:, :, -1, :] = 0
        boundary[:, :, :, 0] = 0
        boundary[:, :, :, -1] = 0

        # 检测边缘的0-1变化
        edge_boundaries = self._detect_edge_transitions(mask_binary)

        # 合并内部边界和边缘边界
        boundary = torch.clamp(boundary + edge_boundaries, 0, 1)

        return boundary

    def _detect_edge_transitions(self, mask):
        """检测图像边缘的0-1变化"""
        _, _, H, W = mask.shape
        edge_boundaries = torch.zeros_like(mask)

        # 顶边和底边
        if W > 1:
            # 顶边
            top_diff = torch.abs(mask[:, :, 0, 1:] - mask[:, :, 0, :-1])
            top_changes = (top_diff > 0.5).float()
            edge_boundaries[:, :, 0, :-1] += top_changes
            edge_boundaries[:, :, 0, 1:] += top_changes

            # 底边
            bottom_diff = torch.abs(mask[:, :, -1, 1:] - mask[:, :, -1, :-1])
            bottom_changes = (bottom_diff > 0.5).float()
            edge_boundaries[:, :, -1, :-1] += bottom_changes
            edge_boundaries[:, :, -1, 1:] += bottom_changes

        # 左边和右边
        if H > 1:
            # 左边
            left_diff = torch.abs(mask[:, :, 1:, 0] - mask[:, :, :-1, 0])
            left_changes = (left_diff > 0.5).float()
            edge_boundaries[:, :, :-1, 0] += left_changes
            edge_boundaries[:, :, 1:, 0] += left_changes

            # 右边
            right_diff = torch.abs(mask[:, :, 1:, -1] - mask[:, :, :-1, -1])
            right_changes = (right_diff > 0.5).float()
            edge_boundaries[:, :, :-1, -1] += right_changes
            edge_boundaries[:, :, 1:, -1] += right_changes

        return torch.clamp(edge_boundaries, 0, 1)


# 使用示例
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 测试不同尺寸
    for size in [33, 65, 129, 257]:
        print(f"\n测试 {size}x{size} 尺寸:")

        # 创建判别器
        discriminator = BoundaryQualityDiscriminator(
            input_channels=1, input_size=size
        ).to(device)

        # 创建测试数据
        batch_size = 2
        local_input = torch.randn(batch_size, 1, size, size).to(device)
        local_mask = torch.bernoulli(torch.ones(batch_size, 1, size, size) * 0.7).to(
            device
        )

        # 前向传播
        quality_score, boundary_map, continuity_score = discriminator(
            local_input, local_mask
        )

        print(f"  质量分数形状: {quality_score.shape}")
        print(f"  边界图形状: {boundary_map.shape}")
        print(f"  连续性分数形状: {continuity_score.shape}")
        print(f"  质量分数范围: [{quality_score.min():.3f}, {quality_score.max():.3f}]")
        print(
            f"  连续性分数范围: [{continuity_score.min():.3f}, {continuity_score.max():.3f}]"
        )

