import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from scipy.ndimage import gaussian_filter
from fanaly import ElevationBoundaryContinuityLoss
import numpy as np

import matplotlib.pyplot as plt

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):
        # Focal loss for elevation data (focusing on areas with high error)
        loss = F.mse_loss(pred, target, reduction="none")
        # Apply focal weighting
        pt = torch.exp(-loss)  # Similar to how focal loss works for classification
        focal_weight = (1 - pt) ** self.gamma
        # Apply weight
        return (focal_weight * loss).mean()
    
class SpatialConsistencyLoss(nn.Module):
    """
    空间一致性损失，确保生成图像在空间上与原始图像保持一致性
    实现了两种版本：
    1. 基于梯度差异的版本
    2. 基于局部窗口像素差异的版本
    """
    def __init__(self, loss_type='gradient', window_size=3):
        """
        初始化
        Args:
            loss_type: 'gradient' 使用梯度差异计算损失，'window' 使用局部窗口计算损失
            window_size: 局部窗口大小，用于第二种计算方式
        """
        super(SpatialConsistencyLoss, self).__init__()
        self.loss_type = loss_type
        self.window_size = window_size
    
    def forward(self, output, target, mask=None):
        """
        计算空间一致性损失
        Args:
            output: 生成图像，形状为 [B, C, H, W]
            target: 原始图像，形状为 [B, C, H, W]
            mask: 可选的掩码，用于只关注特定区域，形状为 [B, 1, H, W]
        Returns:
            loss: 计算得到的损失值
        """
        if self.loss_type == 'gradient':
            return self.gradient_based_loss(output, target, mask)
        else:
            return self.window_based_loss(output, target, mask)
    
    def gradient_based_loss(self, output, target, mask=None):
        """
        基于梯度差异的空间一致性损失
        公式: L_SC = Σ ||∇I_p - ∇I_p'||_1
        """
        # 计算水平和垂直方向的梯度
        output_dx = output[:, :, :, 1:] - output[:, :, :, :-1]  # 水平梯度
        output_dy = output[:, :, 1:, :] - output[:, :, :-1, :]  # 垂直梯度
        
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]  # 水平梯度
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]  # 垂直梯度
        
        # 如果提供了掩码，则应用掩码
        if mask is not None:
            mask=1-mask
            # 调整掩码大小以匹配梯度
            mask_dx = mask[:, :, :, 1:]
            mask_dy = mask[:, :, 1:, :]
            
            # 加权损失计算
            diff_dx = torch.abs(output_dx - target_dx) * mask_dx
            diff_dy = torch.abs(output_dy - target_dy) * mask_dy
            
            # 计算掩码区域的平均损失
            loss_dx = diff_dx.sum() / (mask_dx.sum() + 1e-8)
            loss_dy = diff_dy.sum() / (mask_dy.sum() + 1e-8)
        else:
            # 不使用掩码的标准计算
            loss_dx = F.l1_loss(output_dx, target_dx)
            loss_dy = F.l1_loss(output_dy, target_dy)
        
        return loss_dx + loss_dy
    
    def window_based_loss(self, output, target, mask=None):
        """
        基于局部窗口的空间一致性损失
        公式: L_SC = Σ 1/|Ω_p| Σ (I_q - I_q')²
                    p      q∈Ω_p
        """
        batch_size, channels, height, width = output.shape
        total_loss = 0.0
        
        # 生成卷积核用于局部窗口求和
        kernel_size = self.window_size
        pad_size = kernel_size // 2
        window_filter = torch.ones(1, 1, kernel_size, kernel_size, device=output.device)
        
        for c in range(channels):
            for b in range(batch_size):
                # 提取当前通道
                output_channel = output[b:b+1, c:c+1, :, :]
                target_channel = target[b:b+1, c:c+1, :, :]
                
                # 计算像素差异的平方
                pixel_diff_squared = (output_channel - target_channel) ** 2
                
                # 对于每个位置p，计算局部窗口Ω_p内的均方差异
                # 使用卷积操作高效计算窗口内的和
                padded_diff = F.pad(pixel_diff_squared, (pad_size, pad_size, pad_size, pad_size))
                window_sum = F.conv2d(padded_diff, window_filter)
                
                # 窗口内像素数
                window_count = kernel_size * kernel_size
                
                # 计算每个窗口的平均损失
                window_loss = window_sum / window_count
                
                # 如果提供了掩码，则只考虑掩码区域
                if mask is not None:
                    current_mask = mask[b:b+1, 0:1, :, :]
                    window_loss = window_loss * current_mask
                    total_loss += window_loss.sum() / (current_mask.sum() + 1e-8)
                else:
                    total_loss += window_loss.mean()
        
        return total_loss / (channels * batch_size)


# 简化版本的空间一致性损失，直接基于梯度差异
def spatial_consistency_loss(output, target, mask=None):
    """
    简化版本的基于梯度差异的空间一致性损失
    Args:
        output: 生成图像，形状为 [B, C, H, W]
        target: 原始图像，形状为 [B, C, H, W]
        mask: 可选的掩码，形状为 [B, 1, H, W]
    Returns:
        loss: 空间一致性损失
    """
    # 计算水平和垂直方向的梯度
    output_dx = output[:, :, :, 1:] - output[:, :, :, :-1]
    output_dy = output[:, :, 1:, :] - output[:, :, :-1, :]
    
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    
    if mask is not None:
        # 调整掩码大小
        mask_dx = mask[:, :, :, 1:]
        mask_dy = mask[:, :, 1:, :]
        
        # 计算差异并应用掩码
        diff_dx = torch.abs(output_dx - target_dx) * mask_dx
        diff_dy = torch.abs(output_dy - target_dy) * mask_dy
        
        # 计算损失
        loss = (diff_dx.sum() + diff_dy.sum()) / (mask_dx.sum() + mask_dy.sum() + 1e-8)
    else:
        loss_dx = F.l1_loss(output_dx, target_dx)
        loss_dy = F.l1_loss(output_dy, target_dy)
        loss = loss_dx + loss_dy
    
    return loss


def masked_tv_loss(completed, completed_mask):
    """
    计算只在补全区域 (completed_mask == 1) 的 TV 损失
    Args:
        completed: 补全输出，[B, C, H, W]
        completed_mask: 补全掩码，[B, 1, H, W]，1 表示补全区域
    Returns:
        loss: 掩码加权的 TV 损失
    """
    diff_h = torch.abs(completed[:, :, 1:, :] - completed[:, :, :-1, :])
    diff_w = torch.abs(completed[:, :, :, 1:] - completed[:, :, :, :-1])

    # 调整掩码尺寸以匹配差异图
    mask_h = completed_mask[:, :, 1:, :]
    mask_w = completed_mask[:, :, :, 1:]

    # 只在补全区域 (mask == 1) 计算损失
    loss_h = (diff_h * mask_h).sum()
    loss_w = (diff_w * mask_w).sum()

    return (loss_h + loss_w) / (mask_h.sum() + mask_w.sum() + 1e-8)


def edge_continuity_loss(completed, global_target, pos, edge_width, completed_mask):
    """
    计算局部补全区域与全局目标的连续性损失，包括0阶(像素)、1阶(梯度)和2阶(曲率)误差
    对整个区域应用权重，距离边缘越近权重越大
    使用向量化操作提高计算效率
    
    Args:
        completed: 局部补全输出，[B, C, H_local, W_local]
        global_target: 全局目标图像，[B, C, H_global, W_global]
        pos: 局部区域位置 [xmin, xmax, ymin, ymax]
        edge_width: 边缘区域的宽度（像素）
    
    Returns:
        loss: 加权边缘连续性损失
    """
    batch_size, channels, h_local, w_local = completed.shape
    zero_loss = 0.0
    one_loss = 0.0
    two_loss = 0.0

    for b in range(batch_size):
        # 提取位置信息
        xmin, xmax, ymin, ymax = pos[b]
        
        # 计算内部区域大小
        inner_h, inner_w = xmax - xmin, ymax - ymin
        
        # 提取包含内部区域和周围edge_width区域的完整区域
        extended_xmin, extended_xmax = xmin - edge_width, xmax + edge_width
        extended_ymin, extended_ymax = ymin - edge_width, ymax + edge_width
        
        # 提取全局目标和补全区域的扩展区域
        global_region = global_target[b:b+1, :, extended_xmin:extended_xmax, extended_ymin:extended_ymax]
        completed_region = completed[b:b+1, :, extended_xmin:extended_xmax, extended_ymin:extended_ymax]
        comtarget_region = completed_mask[b:b+1, :, extended_xmin:extended_xmax, extended_ymin:extended_ymax] #是否应该考虑？
        comtarget_region = torch.ones_like(comtarget_region)
        
        # 创建权重掩码，使用向量化操作替代循环
        ext_h, ext_w = extended_xmax - extended_xmin, extended_ymax - extended_ymin

        
        # 生成坐标网格
        y_coords = torch.arange(ext_h, device=completed.device).view(-1, 1).repeat(1, ext_w)
        x_coords = torch.arange(ext_w, device=completed.device).view(1, -1).repeat(ext_h, 1)
        
        # 转换为相对内部区域的坐标
        rel_y = y_coords - edge_width
        rel_x = x_coords - edge_width
        
        # 计算到补全区域边界的距离
        dist_to_top = rel_y  # 到上边缘
        dist_to_bottom = inner_h - rel_y  # 到下边缘
        dist_to_left = rel_x  # 到左边缘
        dist_to_right = inner_w - rel_x  # 到右边缘
        
        # 计算到最近边界的距离（曼哈顿距离）
        dist_to_edge = abs(torch.min(
            torch.min(dist_to_top, dist_to_bottom),
            torch.min(dist_to_left, dist_to_right)
        ))
        
        
        # 指数衰减权重
        decay_factor = 0.4  # 可调衰减因子，替换原来的 10
        weight_mask = torch.exp(-dist_to_edge * decay_factor)
        
        # 扩展权重掩码的维度
        weight_mask = weight_mask.view(1, 1, ext_h, ext_w).expand(1, channels, ext_h, ext_w)
        
        # 1. 计算像素差异 (0阶导数误差)
        pixel_diff = torch.abs(completed_region - global_region) * weight_mask*comtarget_region
        zero_loss += pixel_diff.sum() / ((weight_mask*comtarget_region).sum() + 1e-8)

        # 2. 计算梯度差异 (1阶导数误差)
        # 水平梯度
        out_grad_h = completed_region[:, :, 1:, :] - completed_region[:, :, :-1, :]
        global_grad_h = global_region[:, :, 1:, :] - global_region[:, :, :-1, :]
        weight_mask_h = weight_mask[:, :, 1:, :]
        comtarget_region_h = comtarget_region[:, :, 1:, :]
        
        # 垂直梯度
        out_grad_w = completed_region[:, :, :, 1:] - completed_region[:, :, :, :-1]
        global_grad_w = global_region[:, :, :, 1:] - global_region[:, :, :, :-1]
        weight_mask_w = weight_mask[:, :, :, 1:]
        comtarget_region_w = comtarget_region[:, :, :, 1:]

        grad_diff_h = torch.abs(out_grad_h - global_grad_h) * weight_mask_h*comtarget_region_h
        grad_diff_w = torch.abs(out_grad_w - global_grad_w) * weight_mask_w*comtarget_region_w

        one_loss += (grad_diff_h.sum() + grad_diff_w.sum()) / (
            (weight_mask_h*comtarget_region_h).sum() + (weight_mask_w*comtarget_region_w).sum() + 1e-8
        )

        # 3. 计算二阶导数差异 (2阶导数误差/曲率误差) - 并行计算所有方向
        # 水平二阶导数
        out_grad2_h = (
            completed_region[:, :, 2:, :]
            - 2 * completed_region[:, :, 1:-1, :]
            + completed_region[:, :, :-2, :]
        )
        global_grad2_h = (
            global_region[:, :, 2:, :]
            - 2 * global_region[:, :, 1:-1, :]
            + global_region[:, :, :-2, :]
        )
        weight_mask_h2 = weight_mask[:, :, 2:, :]
        comtarget_region_h2 = comtarget_region[:, :, 2:, :]

        # 垂直二阶导数
        out_grad2_w = (
            completed_region[:, :, :, 2:]
            - 2 * completed_region[:, :, :, 1:-1]
            + completed_region[:, :, :, :-2]
        )
        global_grad2_w = (
            global_region[:, :, :, 2:]
            - 2 * global_region[:, :, :, 1:-1]
            + global_region[:, :, :, :-2]
        )
        weight_mask_w2 = weight_mask[:, :, :, 2:]
        comtarget_region_w2 = comtarget_region[:, :, :, 2:]

        # 对角线二阶导数
        out_grad2_diag = (
            completed_region[:, :, 2:, 2:]
            - 2 * completed_region[:, :, 1:-1, 1:-1]
            + completed_region[:, :, :-2, :-2]
        )
        global_grad2_diag = (
            global_region[:, :, 2:, 2:]
            - 2 * global_region[:, :, 1:-1, 1:-1]
            + global_region[:, :, :-2, :-2]
        )
        weight_mask_diag = weight_mask[:, :, 2:, 2:]
        comtarget_region_diag = comtarget_region[:, :, 2:, 2:]

        # 并行计算所有二阶导数差异
        grad2_diff_h = torch.abs(out_grad2_h - global_grad2_h) * weight_mask_h2 * comtarget_region_h2
        grad2_diff_w = torch.abs(out_grad2_w - global_grad2_w) * weight_mask_w2 * comtarget_region_w2
        grad2_diff_diag = torch.abs(out_grad2_diag - global_grad2_diag) * weight_mask_diag * comtarget_region_diag

        two_loss += (
            grad2_diff_h.sum() + grad2_diff_w.sum() + grad2_diff_diag.sum()
        ) / ((weight_mask_h2*comtarget_region_h2).sum() + (weight_mask_w2*comtarget_region_w2).sum() + (weight_mask_diag*comtarget_region_diag).sum() + 1e-8)

    # 加权合并损失项
    f1 = 0.6  # 像素差异权重
    f2 = 0.2  # 一阶导数权重
    f3 = 0.2  # 二阶导数权重
    total_loss = f1 * zero_loss + f2 * one_loss + f3 * two_loss

    return total_loss / batch_size

def smooth_mask_generate(mask, sigma=1.0,visualize=False):
    
    # 平滑掩码以生成边界过渡区域
    mask = mask.float()
    mask_np = mask.cpu().numpy()
    smoothed_mask = np.zeros_like(mask_np)
    for b in range(mask_np.shape[0]):
        for c in range(mask_np.shape[1]):
            smoothed_mask[b, c] = gaussian_filter(mask_np[b, c], sigma=sigma)
    smoothed_mask = torch.tensor(smoothed_mask, device=mask.device, dtype=mask.dtype)
    smoothed_mask = smoothed_mask.clamp(0, 1)

    # 计算到边界的距离作为权重
    # 0.5在边界中心，越接近0或1则越远离边界
    boundary_distance = torch.abs(smoothed_mask - 0.5)
    # 将距离转换为权重：距离越小，权重越大
    boundary_weights = 1.0 - (boundary_distance / 0.5)
    # 限制权重范围
    boundary_weights = boundary_weights.clamp(0, 1)
    boundary_mask = boundary_weights
    
    if visualize:
        # 可视化原始掩码和平滑后的掩码
        for b in range(mask_np.shape[0]):
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title("Original Mask")
            plt.imshow(mask_np[b, 0], cmap="gray")
            plt.subplot(1, 2, 2)
            plt.title("Smoothed Mask")
            plt.imshow(boundary_mask[b, 0].cpu().numpy(), cmap="gray")
            plt.show()
    return boundary_mask

def boundary_continuity_loss(output, target, mask, sigma=1.0):
    """
    计算掩码边界区域的连续性损失，强调补全区域与非补全区域的平滑过渡。

    Args:
        completed: 补全输出，[B, C, H, W]
        global_target: 全局目标图像，[B, C, H, W]
        mask: 局部掩码，[B, 1, H, W]，0 表示补全区域，1 表示保留区域
        boundary_width: 边界区域宽度（像素）
        sigma: 高斯模糊的标准差，用于平滑掩码

    Returns:
        loss: 边界连续性损失
    """

    boundary_mask = smooth_mask_generate(mask, sigma=sigma)

    # 像素差异（0 阶）
    pixel_diff = torch.abs(output - target) * boundary_mask
    pixel_loss = pixel_diff.sum() / (boundary_mask.sum() + 1e-8)

    # 梯度差异（1 阶）
    completed_grad_h = output[:, :, 1:, :] - output[:, :, :-1, :]
    completed_grad_w = output[:, :, :, 1:] - output[:, :, :, :-1]
    target_grad_h = target[:, :, 1:, :] - target[:, :, :-1, :]
    target_grad_w = target[:, :, :, 1:] - target[:, :, :, :-1]

    boundary_mask_h = boundary_mask[:, :, 1:, :]
    boundary_mask_w = boundary_mask[:, :, :, 1:]

    grad_diff_h = torch.abs(completed_grad_h - target_grad_h) * boundary_mask_h
    grad_diff_w = torch.abs(completed_grad_w - target_grad_w) * boundary_mask_w

    grad_loss = (grad_diff_h.sum() + grad_diff_w.sum()) / (
        boundary_mask_h.sum() + boundary_mask_w.sum() + 1e-8
    )

    # 组合损失
    pixel_weight = 0.7  # 像素差异权重
    grad_weight = 0.3  # 梯度差异权重
    return pixel_weight * pixel_loss + grad_weight * grad_loss

def improved_spatial_consistency_loss(output, target, mask=None):
    """改进的空间一致性损失，可选地使用mask"""
    # 计算水平和垂直方向的差分
    output_dx = output[:, :, :, 1:] - output[:, :, :, :-1]
    output_dy = output[:, :, 1:, :] - output[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    
    # 如果提供了mask，则创建边界权重
    if mask is not None:
        # 平滑掩码以得到边界区域
        boundary_weights = smooth_mask_generate(mask, sigma=1.0)
        
        # 边界权重的对应版本
        weights_x = boundary_weights[:, :, :, 1:] * boundary_weights[:, :, :, :-1]
        weights_y = boundary_weights[:, :, 1:, :] * boundary_weights[:, :, :-1, :]
        
        # 加权损失计算
        loss_dx = torch.abs(output_dx - target_dx) * weights_x
        loss_dy = torch.abs(output_dy - target_dy) * weights_y
        return loss_dx.mean() + loss_dy.mean()
    else:
        # 标准LSC
        loss_dx = F.l1_loss(output_dx, target_dx)
        loss_dy = F.l1_loss(output_dy, target_dy)
        return loss_dx + loss_dy

class EnhancedGradientConsistencyLoss(nn.Module):
    def __init__(self):
        super(EnhancedGradientConsistencyLoss, self).__init__()
        # 不直接创建张量，而是使用register_buffer
        # 这样张量会自动移动到模型的设备上
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        
    def compute_gradient_magnitude_direction(self, x):
        # 计算图像x的梯度幅值和方向
        batch_size, channels = x.shape[:2]
        
        # 扩展卷积核以支持多通道
        extended_sobel_x = self.sobel_x.repeat(channels, 1, 1, 1)
        extended_sobel_y = self.sobel_y.repeat(channels, 1, 1, 1)
        
        # 确保卷积核与输入在同一设备上
        # 注意：这一步在使用register_buffer后应该不需要了，但为安全起见仍保留
        extended_sobel_x = extended_sobel_x.to(x.device)
        extended_sobel_y = extended_sobel_y.to(x.device)
        
        # 计算梯度
        grad_x = F.conv2d(x, extended_sobel_x, padding=1, groups=channels)
        grad_y = F.conv2d(x, extended_sobel_y, padding=1, groups=channels)
        
        # 计算梯度幅值
        magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        
        # 计算梯度方向（弧度）
        direction = torch.atan2(grad_y, grad_x)
        
        return magnitude, direction
    
    def forward(self, output, target, mask):
        mask = mask.float()
        # 计算输出和目标的梯度信息
        output_mag, output_dir = self.compute_gradient_magnitude_direction(output)
        target_mag, target_dir = self.compute_gradient_magnitude_direction(target)
        
        # 计算过渡掩码（边界权重）- 使用平滑函数替代手动CPU计算
        boundary_weight = smooth_mask_generate(mask)
        
        # 计算梯度幅值差异损失
        magnitude_loss = F.l1_loss(output_mag * boundary_weight, 
                                  target_mag * boundary_weight) 
        
        # 考虑到boundary_weight可能有很多零值，使用加权平均
        if boundary_weight.sum() > 0:
            magnitude_loss = magnitude_loss / (boundary_weight.mean() + 1e-8)
        
        # 计算梯度方向差异损失（考虑周期性）
        # 方向差异为[-π, π]范围
        direction_diff = torch.abs(output_dir - target_dir)
        # 考虑方向的周期性，确保差异在[0, π]范围内
        direction_diff = torch.min(direction_diff, 2*math.pi - direction_diff)
        # 加权方向损失
        direction_loss = (direction_diff * boundary_weight).sum()
        if boundary_weight.sum() > 0:
            direction_loss = direction_loss / (boundary_weight.sum() + 1e-8)
        
        # 加权组合
        return magnitude_loss + direction_loss
        
    '''def smooth_mask_generate(self, mask, sigma=1.0, edge_width=10, edge_weight=1.0, visualize=False):
        """
        生成平滑的过渡权重掩码，特别关注图像边缘区域和掩码交界处
        
        参数:
            mask: 输入掩码张量，形状为 [B, C, H, W]，1表示保留区域，0表示补全区域
            sigma: 高斯滤波的标准差，控制平滑程度
            edge_width: 图像边缘区域宽度（像素）
            edge_weight: 边缘区域的最大权重值
            visualize: 是否可视化结果
            
        返回:
            final_weights: 最终权重掩码
        """
        # 将掩码转换为浮点类型
        mask = mask.float()
        batch_size, channels, height, width = mask.shape
        device = mask.device
        
        # 1. 创建两个独立的权重掩码：一个用于0-1交界处，一个用于图像边缘
        
        # 对于0-1交界处的权重计算
        mask_np = mask.cpu().numpy()
        smoothed_mask = np.zeros_like(mask_np)
        
        for b in range(mask_np.shape[0]):
            for c in range(mask_np.shape[1]):
                smoothed_mask[b, c] = gaussian_filter(mask_np[b, c], sigma=sigma)
        
        smoothed_mask = torch.tensor(smoothed_mask, device=device, dtype=mask.dtype)
        smoothed_mask = smoothed_mask.clamp(0, 1)
        
        # 计算交界处权重
        boundary_distance = torch.abs(smoothed_mask - 0.5)
        boundary_weights = (1.0 - (boundary_distance / 0.5)) * edge_weight
        boundary_weights = boundary_weights.clamp(0, edge_weight)
        
        # 2. 创建专门用于图像边缘的权重掩码
        # 首先创建一个全零掩码
        edge_mask = torch.zeros_like(mask)
        
        # 获取补全区域掩码(mask=0的区域)
        completion_mask = (1.0 - mask)
        
        # 为每个边缘创建递减权重
        for b in range(batch_size):
            for c in range(channels):
                # 只处理补全区域(mask=0)
                current_mask = completion_mask[b, c]
                
                # 上边缘 - 从边缘向内递减
                for i in range(min(edge_width, height)):
                    weight = edge_weight * (1.0 - i/edge_width)
                    edge_mask[b, c, i, :] = torch.maximum(edge_mask[b, c, i, :], 
                                                        weight * current_mask[i, :])
                
                # 下边缘 - 从边缘向内递减
                for i in range(min(edge_width, height)):
                    weight = edge_weight * (1.0 - i/edge_width)
                    edge_mask[b, c, height-i-1, :] = torch.maximum(edge_mask[b, c, height-i-1, :], 
                                                                weight * current_mask[height-i-1, :])
                
                # 左边缘 - 从边缘向内递减
                for j in range(min(edge_width, width)):
                    weight = edge_weight * (1.0 - j/edge_width)
                    edge_mask[b, c, :, j] = torch.maximum(edge_mask[b, c, :, j], 
                                                        weight * current_mask[:, j])
                
                # 右边缘 - 从边缘向内递减
                for j in range(min(edge_width, width)):
                    weight = edge_weight * (1.0 - j/edge_width)
                    edge_mask[b, c, :, width-j-1] = torch.maximum(edge_mask[b, c, :, width-j-1], 
                                                                weight * current_mask[:, width-j-1])
        
        # 3. 合并两种权重掩码，取较大值
        final_weights = torch.maximum(boundary_weights, edge_mask)
        
        # 4. 确保权重只应用于补全区域(mask=0)
        final_weights = final_weights * completion_mask
        
        # 可视化结果（如果需要）
        if visualize:
            for b in range(mask_np.shape[0]):
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 3, 1)
                plt.title("原始掩码")
                plt.imshow(mask_np[b, 0], cmap="gray")
                
                plt.subplot(1, 3, 2)
                plt.title("边界交界处权重")
                plt.imshow(edge_mask[b, 0].cpu().numpy(), cmap="jet")
                
                plt.subplot(1, 3, 3)
                plt.title("最终权重掩码")
                plt.imshow(final_weights[b, 0].cpu().numpy(), cmap="jet")
                
                plt.colorbar()
                plt.show()
        
        return final_weights
'''
def total_variation_loss(pred, mask=None, dilation_kernel_size=3, visualize=False):
    """
    计算总变差损失，鼓励图像整体平滑。
    
    参数：
        pred (torch.Tensor): 预测图像，形状为 [B, C, H, W]，单通道时 C=1
        mask (torch.Tensor, optional): mask 图像，形状为 [B, C, H, W]，值为 0 或 1
        dilation_kernel_size (int): 边界膨胀的卷积核大小，控制边界区域范围
        visualize (bool): 是否可视化边界 mask
    
    返回：
        loss (torch.Tensor): 总变差损失
    """
    # 计算水平和垂直方向的像素差值
    diff_x = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
    diff_y = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])

    # 如果指定了 mask，则仅在 mask 边界附近计算
    if mask is not None:
        # 将 mask 转换为浮点类型以匹配卷积操作
        mask_float = mask.float()
        # 使用卷积核检测边界
        kernel = torch.ones(1, 1, 3, 3, device=mask_float.device, dtype=mask_float.dtype)  # 3x3 卷积核
        mask_padded = F.conv2d(mask_float, kernel, padding=1)  # 卷积后形状仍为 [B, C, H, W]
        # 边界定义为 mask 值变化的区域
        boundary_mask = (mask_padded > 0) & (mask_padded < 9)  # 9 是 3x3 卷积核的最大值（全 1 时）
        boundary_mask = boundary_mask.float()

        # 边界膨胀，扩大约束区域
        if dilation_kernel_size > 1:
            kernel = torch.ones(1, 1, dilation_kernel_size, dilation_kernel_size, device=mask_float.device, dtype=mask_float.dtype)
            boundary_mask = F.conv2d(boundary_mask, kernel, padding=dilation_kernel_size//2) > 0

        # 调整 boundary_mask 的形状以匹配 diff_x 和 diff_y
        boundary_mask_x = boundary_mask[:, :, :, :-1]  # [B, C, H, W-1]
        boundary_mask_y = boundary_mask[:, :, :-1, :]  # [B, C, H-1, W]

        # 应用边界 mask
        diff_x = diff_x * boundary_mask_x
        diff_y = diff_y * boundary_mask_y

        # 可视化边界 mask
        if visualize:
            import matplotlib.pyplot as plt
            for b in range(boundary_mask.shape[0]):
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.title("Boundary Mask X")
                plt.imshow(boundary_mask_x[b, 0].cpu().numpy(), cmap="gray")
                plt.subplot(1, 2, 2)
                plt.title("Boundary Mask Y")
                plt.imshow(boundary_mask_y[b, 0].cpu().numpy(), cmap="gray")
                plt.show()

    # 计算损失（L1 范数）
    loss = torch.mean(diff_x) + torch.mean(diff_y)
    return loss

def gradient_similarity_loss(pred, gt, mask=None, dilation_kernel_size=3, visualize=False):
    """
    计算梯度相似性损失（Gradient Similarity Loss, GSL），鼓励预测图像和真实图像的梯度方向一致。
    
    参数：
        pred (torch.Tensor): 预测图像，形状为 [B, C, H, W]，单通道时 C=1
        gt (torch.Tensor): 真实图像，形状为 [B, C, H, W]
        mask (torch.Tensor, optional): mask 图像，形状为 [B, C, H, W]，值为 0 或 1，1 表示需要计算的区域
        dilation_kernel_size (int): 边界膨胀的卷积核大小，控制边界区域范围
        visualize (bool): 是否可视化边界 mask
    
    返回：
        loss (torch.Tensor): 梯度相似性损失
    """
    # 计算水平和垂直梯度
    pred_grad_x = pred[:, :, :, :-1] - pred[:, :, :, 1:]  # 预测图像水平梯度，[B, C, H, W-1]
    pred_grad_y = pred[:, :, :-1, :] - pred[:, :, 1:, :]  # 预测图像垂直梯度，[B, C, H-1, W]
    gt_grad_x = gt[:, :, :, :-1] - gt[:, :, :, 1:]       # 真实图像水平梯度，[B, C, H, W-1]
    gt_grad_y = gt[:, :, :-1, :] - gt[:, :, 1:, :]       # 真实图像垂直梯度，[B, C, H-1, W]

    # 如果指定了 mask，则仅在 mask 边界附近计算
    if mask is not None:
        # 将 mask 转换为浮点类型以匹配 kernel
        mask_float = mask.float()
        # 使用卷积核检测边界
        kernel = torch.ones(1, 1, 3, 3, device=mask_float.device, dtype=mask_float.dtype)  # 3x3 卷积核
        mask_padded = F.conv2d(mask_float, kernel, padding=1)  # 卷积后形状仍为 [B, C, H, W]
        # 边界定义为 mask 值变化的区域
        boundary_mask = (mask_padded > 0) & (mask_padded < 9)  # 9 是 3x3 卷积核的最大值（全 1 时）
        boundary_mask = boundary_mask.float()

        # 边界膨胀，扩大约束区域
        if dilation_kernel_size > 1:
            kernel = torch.ones(1, 1, dilation_kernel_size, dilation_kernel_size, device=mask_float.device, dtype=mask_float.dtype)
            boundary_mask = F.conv2d(boundary_mask, kernel, padding=dilation_kernel_size//2) > 0

        # 调整 boundary_mask 的形状以匹配梯度
        boundary_mask_x = boundary_mask[:, :, :, :-1]  # [B, C, H, W-1]
        boundary_mask_y = boundary_mask[:, :, :-1, :]  # [B, C, H-1, W]
    else:
        # 如果没有 mask，则全图计算
        boundary_mask_x = torch.ones_like(pred_grad_x)
        boundary_mask_y = torch.ones_like(pred_grad_y)

    # 可视化边界 mask
    if visualize and mask is not None:
        import matplotlib.pyplot as plt
        for b in range(boundary_mask.shape[0]):
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title("Boundary Mask X")
            plt.imshow(boundary_mask_x[b, 0].cpu().numpy(), cmap="gray")
            plt.subplot(1, 2, 2)
            plt.title("Boundary Mask Y")
            plt.imshow(boundary_mask_y[b, 0].cpu().numpy(), cmap="gray")
            plt.show()

    # 计算梯度点积（逐元素对应）
    dot_product_x = torch.abs(pred_grad_x * gt_grad_x)  # [B, C, H, W-1]
    dot_product_y = torch.abs(pred_grad_y * gt_grad_y)  # [B, C, H-1, W]

    # 计算梯度 L2 范数
    pred_norm_x = torch.sqrt(pred_grad_x**2 + 1e-8)  # [B, C, H, W-1]
    gt_norm_x = torch.sqrt(gt_grad_x**2 + 1e-8)
    pred_norm_y = torch.sqrt(pred_grad_y**2 + 1e-8)  # [B, C, H-1, W]
    gt_norm_y = torch.sqrt(gt_grad_y**2 + 1e-8)

    # 计算相似性分数
    similarity_x = dot_product_x / (pred_norm_x * gt_norm_x + 1e-8)
    similarity_y = dot_product_y / (pred_norm_y * gt_norm_y + 1e-8)

    # 应用 mask
    similarity_x = similarity_x * boundary_mask_x
    similarity_y = similarity_y * boundary_mask_y

    # 计算损失（1 - 相似性，目标是最大化相似性）
    total_similarity = (torch.sum(similarity_x) + torch.sum(similarity_y))
    total_mask = (torch.sum(boundary_mask_x) + torch.sum(boundary_mask_y) + 1e-8)
    loss = 1.0 - total_similarity / total_mask

    return loss


def gaussian_window(size, sigma):
    """
    生成高斯窗，用于计算局部均值和方差。
    
    参数：
        size (int): 窗大小（必须为奇数）
        sigma (float): 高斯分布的标准差
    
    返回：
        window (torch.Tensor): 高斯窗，形状为 [1, 1, size, size]
    """
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    coords = coords.view(1, -1).repeat(size, 1)
    window = torch.exp(-(coords**2 + coords.T**2) / (2 * sigma**2))
    window = window / window.sum()
    return window.view(1, 1, size, size)

def ssim_loss(pred, gt, mask=None, window_size=5, sigma=1.5, size_average=True):
    """
    计算 SSIM 损失，适用于单通道图像，支持 mask。
    
    参数：
        pred (torch.Tensor): 预测图像，形状为 [B, C, H, W]，单通道时 C=1
        gt (torch.Tensor): 真实图像，形状为 [B, C, H, W]
        mask (torch.Tensor, optional): mask 图像，形状为 [B, C, H, W]，值为 0 或 1，1 表示需要计算的区域
        window_size (int): 高斯窗大小（奇数）
        sigma (float): 高斯核标准差
        size_average (bool): 是否对损失取平均
    
    返回：
        loss (torch.Tensor): SSIM 损失
    """
    # 确保输入是单通道
    assert pred.shape[1] == 1, "SSIM loss expects single-channel images"
    
    # 生成高斯窗
    window = gaussian_window(window_size, sigma).to(pred.device)
    
    # 计算均值（亮度）
    mu_pred = F.conv2d(pred, window, padding=window_size//2, groups=1)
    mu_gt = F.conv2d(gt, window, padding=window_size//2, groups=1)
    
    # 计算方差和协方差
    mu_pred_sq = mu_pred**2
    mu_gt_sq = mu_gt**2
    mu_pred_gt = mu_pred * mu_gt
    
    sigma_pred_sq = F.conv2d(pred * pred, window, padding=window_size//2, groups=1) - mu_pred_sq
    sigma_gt_sq = F.conv2d(gt * gt, window, padding=window_size//2, groups=1) - mu_gt_sq
    sigma_pred_gt = F.conv2d(pred * gt, window, padding=window_size//2, groups=1) - mu_pred_gt
    
    # 常数，用于数值稳定性
    C1 = 0.01**2  # 对应动态范围 L=1（假设图像归一化到 [0,1]）
    C2 = 0.03**2
    
    # 计算 SSIM
    numerator1 = 2 * mu_pred_gt + C1
    numerator2 = 2 * sigma_pred_gt + C2
    denominator1 = mu_pred_sq + mu_gt_sq + C1
    denominator2 = sigma_pred_sq + sigma_gt_sq + C2
    
    ssim_map = (numerator1 * numerator2) / (denominator1 * denominator2)
    
    # 应用 mask
    if mask is not None:
        ssim_map = ssim_map * mask
        if size_average:
            loss = 1.0 - ssim_map.sum() / (mask.sum() + 1e-8)
        else:
            loss = 1.0 - ssim_map.sum()
    else:
        if size_average:
            loss = 1.0 - ssim_map.mean()
        else:
            loss = 1.0 - ssim_map.sum()
    
    return loss

def mean_shift_loss(pred, mask):
    """
    计算补全部分 (mask=0) 和已知部分 (mask=1) 均值差的损失。
    
    参数：
        pred (torch.Tensor): 预测图像，形状为 [B, C, H, W]，单通道时 C=1
        mask (torch.Tensor): mask 图像，形状为 [B, C, H, W]，值为 0 或 1，1 表示已知区域
    
    返回：
        loss (torch.Tensor): 均值差损失
    """
    # 确保 mask 为浮点类型
    mask = mask.float()
    
    # 计算 mask=1 (已知部分) 的均值
    known_mean = torch.sum(pred * mask) / (torch.sum(mask) + 1e-8)
    
    # 计算 mask=0 (补全部分) 的均值
    unknown_mean = torch.sum(pred * (1 - mask)) / (torch.sum(1 - mask) + 1e-8)
    
    # 计算均值差的绝对值作为损失
    loss = torch.abs(known_mean - unknown_mean)
    return loss

def input_smoothness_loss(output, mask, sigma=0.5):
    """
    只关注生成输出自身在边界区域的平滑性
    """
    # 生成边界mask
    boundary_mask = smooth_mask_generate(mask, sigma=sigma)
    
    # 方法1：二阶导数（拉普拉斯算子）- 检测突变
    laplacian_kernel = torch.tensor([[0, 1, 0], 
                                   [1, -4, 1], 
                                   [0, 1, 0]], 
                                  dtype=output.dtype, device=output.device)
    laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3)
    
    # 计算拉普拉斯（二阶导数）
    laplacian = F.conv2d(output, laplacian_kernel, padding=1)
    
    # 只在边界区域惩罚高曲率（突变）
    smoothness_loss = (laplacian ** 2) * boundary_mask
    smoothness_loss = smoothness_loss.sum() / (boundary_mask.sum() + 1e-8)
    
    return smoothness_loss

def completion_network_loss(
    output, local_target, mask, global_target, completed, completed_mask, pos
):
    """Calculate the completion network loss."""
    nw = 0.2# 0.3
    bw = 0# 0.2# 0.5 #completed，大部分在无法改变的global
    ew = 0# 0.2
    cw = 1
    cg=0
    vw = 0# .05 # 加mask
    gsw = 0 # 加mask
    ssimw = 0
    msw = 0
    faw = 0# .4
    totalw = nw+bw+ew+cw+cg+vw+gsw+ssimw+msw
    
    criterion = nn.L1Loss()
    mse_loss = nn.MSELoss(reduction="mean")
    
    null_loss = mse_loss(local_target * (1-mask), output * (1-mask))  # target 是目标补全图像
    '''valid_loss = mse_loss(
        local_target * mask, output * mask
    )  # target 是目标补全图像'''
    # smooth_loss = masked_tv_loss(completed, completed_mask) * 100
    # 边界连续性损失（掩码边界过渡）
    boundary_loss = boundary_continuity_loss(output, local_target, mask, sigma=0.5)*5  # 边界连续性损失，强调补全区域与非补全区域的平滑过渡
    # 各阶梯度损失
    # edge_loss = edge_continuity_loss(completed, global_target, pos, 2, completed_mask)*10
    # 空间一致性损失
    '''consistency_loss= improved_spatial_consistency_loss(
        output, local_target, mask
    )'''
    # mean_shift = mean_shift_loss(output, mask)  # 均值漂移损失，补全区域和已知区域的均值差异
    spatial_consistency = SpatialConsistencyLoss(loss_type='gradient')
    consistency_loss = spatial_consistency.gradient_based_loss(output, local_target, mask)*10 # 空间一致性损失，可选择输入mask
    # enhanced_gradient_loss = EnhancedGradientConsistencyLoss()
    # gradient_loss = enhanced_gradient_loss(output, local_target, mask) # 增强损失，加入mask作用
    # blurred_mask = enhanced_gradient_loss.smooth_mask_generate(mask)
    # variation_loss = total_variation_loss(output, mask)*100 #全变差损失，相邻像素差异
    
    fa_boundary_loss = ElevationBoundaryContinuityLoss().to(output.device)
    all_fa_loss = fa_boundary_loss(output, local_target, mask)
    fa_loss = all_fa_loss['total']*1e-4
    insm_loss = input_smoothness_loss(output, mask, sigma=0.5) * 0.02 # 输入平滑性损失，鼓励生成输出在边界区域的平滑性
    # gradient_similarity = gradient_similarity_loss(output,local_target)*1000 # 梯度相似性损失，梯度方向的一致性
    # ssim = ssim_loss(output,local_target)*10
    # print(f"Consistency_loss: {consistency_loss}, SSIM loss: {ssim}, Gradient similarity loss: {gradient_similarity}, Variation loss: {variation_loss}, Mean shift loss: {mean_shift}")
    '''print(
        f"Null loss: {null_loss*nw}, Consistency loss: {consistency_loss*cw}, Boundary loss: {insm_loss*bw}"
    )'''
    # return (null_loss * nw + bw * boundary_loss + edge_loss * ew + consistency_loss * cw + gradient_loss * cg + variation_loss*vw + gradient_similarity*gsw)/totalw
    return 10*(null_loss * nw + consistency_loss * cw + fa_loss * faw + insm_loss * bw) / totalw

