import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

class ElevationBoundaryContinuityLoss(nn.Module):
    """专门针对高程数据的边界连续性损失函数"""
    def __init__(self, lambda_grad=2.0, lambda_freq=1.5, lambda_smooth=3.0, lambda_slope=2.0):
        super(ElevationBoundaryContinuityLoss, self).__init__()
        self.lambda_grad = lambda_grad      # 梯度连续性
        self.lambda_freq = lambda_freq      # 频域连续性  
        self.lambda_smooth = lambda_smooth  # 平滑度约束
        self.lambda_slope = lambda_slope    # 坡度连续性
        
        # 预定义梯度算子
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                                    dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                                    dtype=torch.float32).view(1, 1, 3, 3))
        
        # 拉普拉斯算子用于检测二阶导数不连续
        self.register_buffer('laplacian', torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], 
                                                      dtype=torch.float32).view(1, 1, 3, 3))
    
    def extract_boundary_mask(self, mask, width=2):
        """提取边界mask - 多层边界"""
        boundary_masks = {}
        
        for w in range(1, width + 1):
            kernel_size = w * 2 + 1
            kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device)
            
            dilated = F.conv2d(mask.float(), kernel, padding=w) > 0
            eroded = F.conv2d(mask.float(), kernel, padding=w) >= kernel.numel()
            boundary = (dilated.float() - eroded.float()).clamp(0, 1)
            boundary_masks[f'width_{w}'] = boundary
            
        return boundary_masks
    
    def compute_elevation_gradients(self, elevation):
        """计算高程梯度（坡度和坡向）"""
        # 一阶梯度
        grad_x = F.conv2d(elevation, self.sobel_x, padding=1)
        grad_y = F.conv2d(elevation, self.sobel_y, padding=1)
        
        # 坡度幅度
        slope_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        
        # 坡向
        slope_direction = torch.atan2(grad_y, grad_x + 1e-8)
        
        # 二阶导数（曲率）
        curvature = F.conv2d(elevation, self.laplacian, padding=1)
        
        return {
            'grad_x': grad_x,
            'grad_y': grad_y, 
            'slope_magnitude': slope_magnitude,
            'slope_direction': slope_direction,
            'curvature': curvature
        }
    
    def gradient_continuity_loss(self, pred_elevation, target_elevation, boundary_masks):
        """梯度连续性损失 - 关键用于高程数据"""
        pred_grads = self.compute_elevation_gradients(pred_elevation)
        target_grads = self.compute_elevation_gradients(target_elevation)
        
        total_loss = 0.0
        
        # 1. 坡度幅度连续性
        slope_diff = torch.abs(pred_grads['slope_magnitude'] - target_grads['slope_magnitude'])
        for width, boundary_mask in boundary_masks.items():
            weight = 1.0 if width == 'width_1' else 0.5  # 最近边界权重更高
            total_loss += weight * (slope_diff * boundary_mask).mean()
        
        # 2. 坡向连续性（处理角度周期性）
        direction_diff = torch.abs(pred_grads['slope_direction'] - target_grads['slope_direction'])
        direction_diff = torch.minimum(direction_diff, 2*np.pi - direction_diff)  # 处理角度周期性
        for width, boundary_mask in boundary_masks.items():
            weight = 1.0 if width == 'width_1' else 0.5
            total_loss += weight * (direction_diff * boundary_mask).mean()
        
        # 3. 曲率连续性（二阶导数）
        curvature_diff = torch.abs(pred_grads['curvature'] - target_grads['curvature'])
        for width, boundary_mask in boundary_masks.items():
            weight = 2.0 if width == 'width_1' else 1.0  # 曲率连续性更重要
            total_loss += weight * (curvature_diff * boundary_mask).mean()
            
        return total_loss / len(boundary_masks)
    
    def frequency_continuity_loss(self, pred_elevation, target_elevation, mask):
        """频域连续性损失 - 检测高频不连续"""
        # FFT变换
        pred_fft = torch.fft.fft2(pred_elevation)
        target_fft = torch.fft.fft2(target_elevation)
        
        # 幅度谱
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        # 相位谱
        pred_phase = torch.angle(pred_fft)
        target_phase = torch.angle(target_fft)
        
        # 高频区域mask（高程突变通常表现为高频）
        h, w = pred_elevation.shape[-2:]
        high_freq_mask = self.create_high_freq_mask(h, w, pred_elevation.device)
        
        # 高频幅度损失
        high_freq_mag_loss = F.mse_loss(
            pred_mag * high_freq_mask, 
            target_mag * high_freq_mask
        )
        
        # 相位连续性损失
        phase_diff = torch.angle(torch.exp(1j * (pred_phase - target_phase)))
        phase_loss = torch.mean((phase_diff * high_freq_mask)**2)
        
        return high_freq_mag_loss + 2.0 * phase_loss
    
    def create_high_freq_mask(self, h, w, device):
        """创建高频mask"""
        mask = torch.zeros(1, 1, h, w, device=device)
        center_h, center_w = h // 2, w // 2
        
        # 高频区域（远离DC分量）
        for i in range(h):
            for j in range(w):
                freq_dist = np.sqrt((i - center_h)**2 + (j - center_w)**2)
                if freq_dist > min(h, w) * 0.3:  # 高频阈值
                    mask[0, 0, i, j] = 1.0
                    
        return mask
    
    def smoothness_constraint_loss(self, pred_elevation, boundary_masks):
        """平滑度约束损失 - 修复NaN问题"""
        smoothness_loss = 0.0
        
        for width, boundary_mask in boundary_masks.items():
            if width == 'width_1':
                # 3x3邻域均值
                kernel = torch.ones(1, 1, 3, 3, device=pred_elevation.device) / 9.0
                local_mean = F.conv2d(pred_elevation, kernel, padding=1)
                local_mean_sq = F.conv2d(pred_elevation**2, kernel, padding=1)
                
                # 修复方差计算，防止负数
                local_var = torch.clamp(local_mean_sq - local_mean**2, min=1e-8)
                local_std = torch.sqrt(local_var)
                
                # 检查是否有NaN
                if torch.isnan(local_std).any():
                    print("Warning: NaN in local_std, skipping smoothness loss")
                    return torch.tensor(0.0, device=pred_elevation.device)
                
                smoothness_loss += (local_std * boundary_mask).mean()
                
        return smoothness_loss
    
    def slope_continuity_loss(self, pred_elevation, target_elevation, boundary_masks):
        """坡度连续性损失 - 专门针对地形数据"""
        pred_grads = self.compute_elevation_gradients(pred_elevation)
        target_grads = self.compute_elevation_gradients(target_elevation)
        
        slope_loss = 0.0
        
        # 检查相邻像素间的坡度变化
        for width, boundary_mask in boundary_masks.items():
            if width == 'width_1':  # 重点关注直接边界
                # 计算坡度的空间变化率
                pred_slope_grad_x = F.conv2d(pred_grads['slope_magnitude'], self.sobel_x, padding=1)
                pred_slope_grad_y = F.conv2d(pred_grads['slope_magnitude'], self.sobel_y, padding=1)
                pred_slope_change = torch.sqrt(pred_slope_grad_x**2 + pred_slope_grad_y**2 + 1e-8)
                
                target_slope_grad_x = F.conv2d(target_grads['slope_magnitude'], self.sobel_x, padding=1)
                target_slope_grad_y = F.conv2d(target_grads['slope_magnitude'], self.sobel_y, padding=1)
                target_slope_change = torch.sqrt(target_slope_grad_x**2 + target_slope_grad_y**2 + 1e-8)
                
                # 坡度变化的差异
                slope_change_diff = torch.abs(pred_slope_change - target_slope_change)
                slope_loss += (slope_change_diff * boundary_mask).mean()
                
        return slope_loss
    
    def forward(self, pred_elevation, target_elevation, mask):
        """
        Args:
            pred_elevation: 预测的高程图 [B, 1, H, W]
            target_elevation: 目标高程图 [B, 1, H, W]
            mask: inpainting mask [B, 1, H, W]
        """
        boundary_masks = self.extract_boundary_mask(mask, width=3)
        
        # 各项损失
        grad_loss = self.gradient_continuity_loss(pred_elevation, target_elevation, boundary_masks)
        freq_loss = self.frequency_continuity_loss(pred_elevation, target_elevation, mask)
        smooth_loss = self.smoothness_constraint_loss(pred_elevation, boundary_masks)
        slope_loss = self.slope_continuity_loss(pred_elevation, target_elevation, boundary_masks)
        
        total_loss = (self.lambda_grad * grad_loss + 
                     self.lambda_freq * freq_loss + 
                     self.lambda_smooth * smooth_loss +
                     self.lambda_slope * slope_loss)

        return {
            'total': total_loss,
            'gradient': grad_loss,
            'frequency': freq_loss,
            'smoothness': smooth_loss,
            'slope': slope_loss
        }

class ElevationFrequencyPatchGAN(nn.Module):
    """专门针对单通道高程数据的频域PatchGAN判别器"""
    def __init__(self, input_channels=1):
        super(ElevationFrequencyPatchGAN, self).__init__()
        
        # 空间域判别器 - 适配单通道
        self.spatial_disc = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, 1, 1)
        )
        
        # 梯度域判别器 - 专门判别坡度连续性
        self.gradient_disc = nn.Sequential(
            nn.Conv2d(2, 32, 3, 1, 1),  # grad_x + grad_y
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, 1, 1)
        )
        
        # 频域判别器
        self.freq_disc = nn.Sequential(
            nn.Conv2d(input_channels * 2, 32, 3, 1, 1),  # 实部+虚部
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, 1, 1)
        )
        
        # 边界专用判别器
        self.boundary_disc = nn.Sequential(
            nn.Conv2d(input_channels, 32, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 1, 1, 0)
        )
        
        # 梯度算子
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                                    dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                                    dtype=torch.float32).view(1, 1, 3, 3))
    
    def extract_boundary_mask(self, mask, width=2):
        """提取边界mask"""
        kernel_size = width * 2 + 1
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device)
        
        dilated = F.conv2d(mask.float(), kernel, padding=width) > 0
        eroded = F.conv2d(mask.float(), kernel, padding=width) >= kernel.numel()
        boundary = (dilated.float() - eroded.float()).clamp(0, 1)
        return boundary
    
    def forward(self, elevation, mask=None):
        """
        Args:
            elevation: 高程数据 [B, 1, H, W]
            mask: inpainting mask [B, 1, H, W]
        """
        # 空间域判别
        spatial_out = self.spatial_disc(elevation)
        
        # 梯度域判别
        grad_x = F.conv2d(elevation, self.sobel_x, padding=1)
        grad_y = F.conv2d(elevation, self.sobel_y, padding=1)
        gradient_input = torch.cat([grad_x, grad_y], dim=1)
        gradient_out = self.gradient_disc(gradient_input)
        
        # 频域判别
        fft_elevation = torch.fft.fft2(elevation)
        freq_input = torch.cat([fft_elevation.real, fft_elevation.imag], dim=1)
        freq_out = self.freq_disc(freq_input)
        
        # 边界判别
        boundary_out = self.boundary_disc(elevation)
        if mask is not None:
            boundary_mask = self.extract_boundary_mask(mask)
            boundary_out = boundary_out * boundary_mask
        
        
        return {
            'spatial': spatial_out,
            'gradient': gradient_out,
            'frequency': freq_out,
            'boundary': boundary_out
        }

def create_test_elevation_data():
    """创建测试用的高程数据"""
    # 创建真实高程 - 平滑的山丘
    real_elevation = np.zeros((33, 33))
    center = 16
    
    for i in range(33):
        for j in range(33):
            # 创建径向距离
            dist = np.sqrt((i - center)**2 + (j - center)**2)
            # 高斯山丘
            elevation = 100 * np.exp(-dist**2 / (2 * 8**2))
            # 添加一些噪声使其更真实
            elevation += 5 * np.sin(i * 0.3) * np.cos(j * 0.3)
            real_elevation[i, j] = elevation
    
    # 创建假高程 - 在边界处添加不连续
    fake_elevation = real_elevation.copy()
    
    # 在mask边界创建高程突变
    mask_region = slice(center-8, center+8)
    
    # 边界处添加高程跳跃
    for i in range(center-8, center+8):
        for j in range(center-8, center+8):
            if abs(i - (center-8)) < 2 or abs(i - (center+7)) < 2 or \
               abs(j - (center-8)) < 2 or abs(j - (center+7)) < 2:
                # 添加高程突变（模拟inpainting边界问题）
                fake_elevation[i, j] += np.random.normal(0, 15)  # 15m的突变
    
    # 创建mask
    mask = np.zeros((33, 33))
    mask[mask_region, mask_region] = 1
    
    return real_elevation, fake_elevation, mask

def visualize_elevation_analysis(real_elevation, fake_elevation, mask, model):
    """可视化高程数据分析"""
    # 转换为tensor
    real_tensor = torch.tensor(real_elevation).unsqueeze(0).unsqueeze(0).float()
    fake_tensor = torch.tensor(fake_elevation).unsqueeze(0).unsqueeze(0).float()
    mask_tensor = torch.tensor(mask).unsqueeze(0).unsqueeze(0).float()
    
    # 获取判别结果
    with torch.no_grad():
        real_results = model(real_tensor, mask_tensor)
        fake_results = model(fake_tensor, mask_tensor)
    
    # 计算梯度
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    
    real_grad_x = F.conv2d(real_tensor, sobel_x, padding=1).squeeze().numpy()
    real_grad_y = F.conv2d(real_tensor, sobel_y, padding=1).squeeze().numpy()
    real_slope = np.sqrt(real_grad_x**2 + real_grad_y**2)
    
    fake_grad_x = F.conv2d(fake_tensor, sobel_x, padding=1).squeeze().numpy()
    fake_grad_y = F.conv2d(fake_tensor, sobel_y, padding=1).squeeze().numpy()
    fake_slope = np.sqrt(fake_grad_x**2 + fake_grad_y**2)
    
    # 频域分析
    real_fft = torch.fft.fft2(real_tensor)
    fake_fft = torch.fft.fft2(fake_tensor)
    real_magnitude = torch.abs(real_fft).squeeze().numpy()
    fake_magnitude = torch.abs(fake_fft).squeeze().numpy()
    
    # 创建可视化
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 6, figure=fig, hspace=0.3, wspace=0.3)
    
    # 第一行：高程数据对比
    ax1 = fig.add_subplot(gs[0, 0:2])
    im1 = ax1.imshow(real_elevation, cmap='terrain')
    ax1.set_title('Real Elevation (Ground Truth)', fontsize=12, fontweight='bold')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    ax2 = fig.add_subplot(gs[0, 2:4])
    im2 = ax2.imshow(fake_elevation, cmap='terrain')
    ax2.set_title('Fake Elevation (Inpainting Result)', fontsize=12, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    ax3 = fig.add_subplot(gs[0, 4:6])
    ax3.imshow(mask, cmap='gray')
    ax3.set_title('Inpainting Mask', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # 第二行：坡度对比
    ax4 = fig.add_subplot(gs[1, 0:2])
    im4 = ax4.imshow(real_slope, cmap='plasma')
    ax4.set_title('Real Slope Magnitude', fontsize=12)
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046)
    
    ax5 = fig.add_subplot(gs[1, 2:4])
    im5 = ax5.imshow(fake_slope, cmap='plasma')
    ax5.set_title('Fake Slope Magnitude', fontsize=12)
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046)
    
    ax6 = fig.add_subplot(gs[1, 4:6])
    slope_diff = np.abs(real_slope - fake_slope)
    im6 = ax6.imshow(slope_diff, cmap='hot')
    ax6.set_title('Slope Difference', fontsize=12)
    ax6.axis('off')
    plt.colorbar(im6, ax=ax6, fraction=0.046)
    
    # 第三行：频域对比
    ax7 = fig.add_subplot(gs[2, 0:2])
    im7 = ax7.imshow(np.log(real_magnitude + 1e-8), cmap='viridis')
    ax7.set_title('Real Frequency Magnitude (Log)', fontsize=12)
    ax7.axis('off')
    plt.colorbar(im7, ax=ax7, fraction=0.046)
    
    ax8 = fig.add_subplot(gs[2, 2:4])
    im8 = ax8.imshow(np.log(fake_magnitude + 1e-8), cmap='viridis')
    ax8.set_title('Fake Frequency Magnitude (Log)', fontsize=12)
    ax8.axis('off')
    plt.colorbar(im8, ax=ax8, fraction=0.046)
    
    ax9 = fig.add_subplot(gs[2, 4:6])
    freq_diff = np.abs(real_magnitude - fake_magnitude)
    im9 = ax9.imshow(freq_diff, cmap='hot')
    ax9.set_title('Frequency Magnitude Difference', fontsize=12)
    ax9.axis('off')
    plt.colorbar(im9, ax=ax9, fraction=0.046)
    
    # 第四行：判别器输出
    discriminator_outputs = ['spatial', 'gradient', 'frequency']
    titles = ['Spatial Disc', 'Gradient Disc', 'Frequency Disc']
    
    for i, (output_key, title) in enumerate(zip(discriminator_outputs, titles)):
        ax = fig.add_subplot(gs[3, i*2:(i+1)*2])
        
        real_score = real_results[output_key].squeeze().numpy()
        fake_score = fake_results[output_key].squeeze().numpy()
        
        combined = np.concatenate([real_score, fake_score], axis=1)
        im = ax.imshow(combined, cmap='RdYlBu')
        ax.set_title(f'{title}\nLeft:Real Right:Fake', fontsize=10)
        ax.axvline(x=real_score.shape[1]-0.5, color='white', linewidth=2)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.suptitle('Elevation Data - Frequency Domain PatchGAN Analysis', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # 打印统计信息
    print("=== 高程数据判别器得分统计 ===")
    for key in ['spatial', 'gradient', 'frequency', 'boundary']:
        real_mean = real_results[key].mean().item()
        fake_mean = fake_results[key].mean().item()
        print(f"{key:12}: Real={real_mean:.4f}, Fake={fake_mean:.4f}, Diff={real_mean-fake_mean:.4f}")
    
    # 高程数据特有的分析
    print(f"\n=== 高程数据边界分析 ===")
    print(f"边界区域坡度差异: {slope_diff.mean():.4f}")
    print(f"最大坡度差异: {slope_diff.max():.4f}")
    print(f"Real边界平均坡度: {real_slope.mean():.4f}")
    print(f"Fake边界平均坡度: {fake_slope.mean():.4f}")
    
    return real_results, fake_results

# 主执行代码
if __name__ == "__main__":
    # 创建模型
    model = ElevationFrequencyPatchGAN(input_channels=1)
    model.eval()
    
    # 创建损失函数
    boundary_loss = ElevationBoundaryContinuityLoss()
    
    # 创建测试高程数据
    print("创建测试高程数据...")
    real_elevation, fake_elevation, mask = create_test_elevation_data()
    
    # 测试损失函数
    real_tensor = torch.tensor(real_elevation).unsqueeze(0).unsqueeze(0).float()
    fake_tensor = torch.tensor(fake_elevation).unsqueeze(0).unsqueeze(0).float()
    mask_tensor = torch.tensor(mask).unsqueeze(0).unsqueeze(0).float()
    
    losses = boundary_loss(fake_tensor, real_tensor, mask_tensor)
    print("\n=== 高程边界连续性损失 ===")
    for key, value in losses.items():
        print(f"{key:12}: {value.item():.4f}")
    
    # 可视化分析
    print("\n生成高程数据可视化结果...")
    # real_results, fake_results = visualize_elevation_analysis(real_elevation, fake_elevation, mask, model)
    
    print("\n高程数据分析完成！")
