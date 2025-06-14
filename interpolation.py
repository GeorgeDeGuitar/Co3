"""
interpolation.py - 插值处理模块
用于在网络forward过程中对mask区域进行插值预处理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TorchInterpolationModule(nn.Module):
    """
    基于PyTorch的快速插值模块，适合在GPU上运行
    """
    def __init__(self, method='bicubic', enabled=True, mixed_training=True, interpolation_prob=0.7):
        """
        Args:
            method: 插值方法 ['bicubic', 'bilinear', 'nearest', 'adaptive']
            enabled: 是否启用插值
            mixed_training: 训练时是否随机选择使用插值
            interpolation_prob: 训练时使用插值的概率
        """
        super().__init__()
        self.method = method
        self.enabled = enabled
        self.mixed_training = mixed_training
        self.interpolation_prob = interpolation_prob
        
        # 注册为buffer，不参与梯度计算但会随模型移动到GPU
        self.register_buffer('_dummy', torch.tensor(0.0))
        
    def forward(self, data, mask):
        """
        对输入数据进行插值处理
        
        Args:
            data: [B, C, H, W] 输入数据
            mask: [B, C, H, W] 或 [B, 1, H, W] 掩码，1=有效，0=无效
            
        Returns:
            interpolated_data: [B, C, H, W] 插值后的数据
        """
        if not self.enabled:
            return data
            
        # 训练时的随机策略
        if self.training and self.mixed_training:
            if torch.rand(1, device=data.device).item() > self.interpolation_prob:
                return data
                
        # 确保mask维度匹配
        if mask.shape[1] == 1 and data.shape[1] > 1:
            mask = mask.expand_as(data)
        elif mask.shape[1] != data.shape[1]:
            # 取第一个通道的mask应用到所有通道
            mask = mask[:, 0:1].expand_as(data)
            
        return self._interpolate_batch(data, mask)
    
    def _interpolate_batch(self, data, mask):
        """对整个batch进行插值"""
        if self.method == 'bicubic':
            return self._bicubic_interpolation(data, mask)
        elif self.method == 'bilinear':
            return self._bilinear_interpolation(data, mask)
        elif self.method == 'nearest':
            return self._nearest_interpolation(data, mask)
        elif self.method == 'adaptive':
            return self._adaptive_interpolation(data, mask)
        else:
            raise ValueError(f"Unsupported interpolation method: {self.method}")
    
    def _bicubic_interpolation(self, data, mask):
        """双三次插值（基于卷积的近似实现）"""
        batch_size, channels, height, width = data.shape
        
        # 创建距离权重
        invalid_mask = (mask == 0).float()
        
        # 使用多尺度平均来近似双三次插值
        result = data.clone()
        
        for b in range(batch_size):
            for c in range(channels):
                current_data = data[b, c:c+1]
                current_mask = mask[b, c:c+1]
                current_invalid = invalid_mask[b, c:c+1]
                
                if current_invalid.sum() == 0:
                    continue
                    
                # 多尺度插值
                interpolated = self._multiscale_interpolation(current_data, current_mask)
                result[b, c:c+1] = torch.where(current_mask > 0, current_data, interpolated)
                
        return result
    
    def _bilinear_interpolation(self, data, mask):
        """双线性插值"""
        batch_size, channels, height, width = data.shape
        result = data.clone()
        
        # 创建膨胀的掩码来扩展有效区域
        kernel = torch.ones(1, 1, 3, 3, device=data.device) / 9.0
        
        for b in range(batch_size):
            for c in range(channels):
                current_data = data[b, c:c+1]
                current_mask = mask[b, c:c+1]
                
                if (current_mask == 0).sum() == 0:
                    continue
                
                # 膨胀有效区域
                dilated_data = current_data * current_mask
                dilated_mask = current_mask
                
                # 多次膨胀操作
                for _ in range(3):
                    # 对数据和掩码同时进行卷积
                    conv_data = F.conv2d(dilated_data, kernel, padding=1)
                    conv_mask = F.conv2d(dilated_mask, kernel, padding=1)
                    
                    # 归一化
                    dilated_data = torch.where(conv_mask > 0, conv_data / torch.clamp(conv_mask, min=1e-8), dilated_data)
                    dilated_mask = torch.clamp(conv_mask, max=1.0)
                
                # 只在原始无效区域使用插值结果
                result[b, c:c+1] = torch.where(current_mask > 0, current_data, dilated_data)
                
        return result
    
    def _nearest_interpolation(self, data, mask):
        """最近邻插值（基于膨胀操作）"""
        batch_size, channels, height, width = data.shape
        result = data.clone()
        
        # 定义膨胀核
        kernel_size = 3
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=data.device)
        
        for b in range(batch_size):
            for c in range(channels):
                current_data = data[b, c:c+1]
                current_mask = mask[b, c:c+1]
                
                if (current_mask == 0).sum() == 0:
                    continue
                
                # 逐步膨胀有效区域
                filled_data = current_data.clone()
                filled_mask = current_mask.clone()
                
                max_iterations = max(height, width) // 2
                for iteration in range(max_iterations):
                    if filled_mask.min() > 0:
                        break
                        
                    # 膨胀操作
                    dilated_data = F.max_pool2d(-(-filled_data * filled_mask), 
                                              kernel_size, stride=1, padding=kernel_size//2)
                    dilated_mask = F.max_pool2d(filled_mask, 
                                              kernel_size, stride=1, padding=kernel_size//2)
                    
                    # 更新未填充区域
                    unfilled = (filled_mask == 0).float()
                    filled_data = filled_data + dilated_data * unfilled
                    filled_mask = torch.clamp(filled_mask + dilated_mask * unfilled, max=1.0)
                
                result[b, c:c+1] = filled_data
                
        return result
    
    def _adaptive_interpolation(self, data, mask):
        """自适应插值：根据无效区域大小选择方法"""
        batch_size, channels, height, width = data.shape
        result = data.clone()
        
        for b in range(batch_size):
            for c in range(channels):
                current_mask = mask[b, c]
                invalid_ratio = (current_mask == 0).float().mean()
                
                # 根据无效区域比例选择插值方法
                if invalid_ratio < 0.1:  # 小洞用双三次
                    result[b, c:c+1] = self._bicubic_interpolation(
                        data[b:b+1, c:c+1], mask[b:b+1, c:c+1]
                    )[0]
                elif invalid_ratio < 0.4:  # 中等区域用双线性
                    result[b, c:c+1] = self._bilinear_interpolation(
                        data[b:b+1, c:c+1], mask[b:b+1, c:c+1]
                    )[0]
                else:  # 大区域用最近邻
                    result[b, c:c+1] = self._nearest_interpolation(
                        data[b:b+1, c:c+1], mask[b:b+1, c:c+1]
                    )[0]
                    
        return result
    
    def _multiscale_interpolation(self, data, mask):
        """多尺度插值"""
        # 不同尺度的卷积核
        kernels = [
            torch.ones(1, 1, 3, 3, device=data.device) / 9.0,    # 3x3
            torch.ones(1, 1, 5, 5, device=data.device) / 25.0,   # 5x5
            torch.ones(1, 1, 7, 7, device=data.device) / 49.0,   # 7x7
        ]
        
        results = []
        weights = []
        
        for i, kernel in enumerate(kernels):
            padding = kernel.shape[-1] // 2
            
            # 计算加权平均
            weighted_data = data * mask
            conv_data = F.conv2d(weighted_data, kernel, padding=padding)
            conv_weight = F.conv2d(mask, kernel, padding=padding)
            
            # 归一化
            interpolated = torch.where(conv_weight > 1e-8, 
                                     conv_data / conv_weight, 
                                     torch.zeros_like(conv_data))
            
            # 计算置信度权重（有效邻域越多权重越大）
            confidence = torch.clamp(conv_weight, 0, 1)
            
            results.append(interpolated)
            weights.append(confidence)
        
        # 加权融合多个尺度的结果
        total_weight = sum(weights) + 1e-8
        final_result = sum(r * w for r, w in zip(results, weights)) / total_weight
        
        return final_result


class EdgeAwareInterpolation(nn.Module):
    """
    边缘感知插值模块，保持地形特征
    """
    def __init__(self, enabled=True, edge_threshold=0.1):
        super().__init__()
        self.enabled = enabled
        self.edge_threshold = edge_threshold
        
        # Sobel算子用于边缘检测
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
        
    def forward(self, data, mask):
        """
        边缘感知插值
        """
        if not self.enabled:
            return data
            
        # 检测边缘
        edges = self._detect_edges(data)
        
        # 在边缘附近使用更保守的插值
        return self._edge_aware_interpolate(data, mask, edges)
    
    def _detect_edges(self, data):
        """检测地形边缘"""
        # 计算梯度
        grad_x = F.conv2d(data, self.sobel_x.expand(data.shape[1], -1, -1, -1), 
                         padding=1, groups=data.shape[1])
        grad_y = F.conv2d(data, self.sobel_y.expand(data.shape[1], -1, -1, -1), 
                         padding=1, groups=data.shape[1])
        
        # 计算梯度幅值
        edge_magnitude = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-8)
        
        # 二值化边缘
        edges = (edge_magnitude > self.edge_threshold).float()
        
        return edges
    
    def _edge_aware_interpolate(self, data, mask, edges):
        """基于边缘信息的插值"""
        # 在边缘区域使用最近邻插值
        # 在平滑区域使用双线性插值
        
        basic_interpolator = TorchInterpolationModule(method='bilinear', enabled=True)
        edge_interpolator = TorchInterpolationModule(method='nearest', enabled=True)
        
        basic_result = basic_interpolator(data, mask)
        edge_result = edge_interpolator(data, mask)
        
        # 根据边缘权重融合结果
        edge_weight = F.max_pool2d(edges, kernel_size=3, stride=1, padding=1)
        result = edge_weight * edge_result + (1 - edge_weight) * basic_result
        
        # 保持原有有效区域不变
        return torch.where(mask > 0, data, result)


# ========================================
# 便捷的创建函数
# ========================================

def create_interpolation_module(method='bicubic', enabled=True, **kwargs):
    """
    创建插值模块的便捷函数
    
    Args:
        method: 插值方法
        enabled: 是否启用
        **kwargs: 其他参数
    
    Returns:
        插值模块实例
    """
    if method == 'edge_aware':
        return EdgeAwareInterpolation(enabled=enabled, **kwargs)
    else:
        return TorchInterpolationModule(method=method, enabled=enabled, **kwargs)


# ========================================
# 使用示例
# ========================================

if __name__ == "__main__":
    # 创建测试数据
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 模拟DEM数据
    data = torch.randn(2, 1, 33, 33, device=device) * 10 - 30  # 模拟高程数据
    mask = torch.ones(2, 1, 33, 33, device=device)
    
    # 创建一些无效区域
    mask[:, :, 10:20, 10:20] = 0  # 中心区域无效
    mask[:, :, 5:8, 25:30] = 0   # 边缘小区域无效
    
    # 将无效区域设为特殊值
    data = data * mask + 100.0 * (1 - mask)
    
    print("原始数据统计:")
    print(f"有效区域: {(mask > 0).sum().item()}/{mask.numel()}")
    print(f"数据范围: [{data[mask > 0].min():.2f}, {data[mask > 0].max():.2f}]")
    
    # 测试不同插值方法
    methods = ['bicubic', 'bilinear', 'nearest', 'adaptive']
    
    for method in methods:
        print(f"\n测试 {method} 插值:")
        
        interpolator = TorchInterpolationModule(
            method=method, 
            enabled=True, 
            mixed_training=False  # 测试时总是插值
        )
        interpolator.to(device)
        interpolator.eval()
        
        with torch.no_grad():
            interpolated = interpolator(data, mask)
            
        print(f"插值后数据范围: [{interpolated.min():.2f}, {interpolated.max():.2f}]")
        print(f"无效区域插值值范围: [{interpolated[mask == 0].min():.2f}, {interpolated[mask == 0].max():.2f}]")
    
    # 测试边缘感知插值
    print(f"\n测试边缘感知插值:")
    edge_interpolator = EdgeAwareInterpolation(enabled=True)
    edge_interpolator.to(device)
    edge_interpolator.eval()
    
    with torch.no_grad():
        edge_result = edge_interpolator(data, mask)
        
    print(f"边缘感知插值后数据范围: [{edge_result.min():.2f}, {edge_result.max():.2f}]")
    print(f"无效区域插值值范围: [{edge_result[mask == 0].min():.2f}, {edge_result[mask == 0].max():.2f}]")