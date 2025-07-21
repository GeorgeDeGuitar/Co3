import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from DemDataset2 import DemDataset

# from train import simple_dataset_split, fast_collate_fn
import matplotlib.pyplot as plt


class BoundaryQualityDiscriminator(nn.Module):
    """
    边界质量判别器

    核心思想：
    - 不判断真假，而是评估生成质量
    - 通过边界提取的准确性来评判生成器性能
    """

    def __init__(self, input_channels=1, input_size=33):
        super().__init__()

        self.ture_extractor = BoundaryDetector(input_size)
        # 边界提取网络
        self.boundary_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Sigmoid(),
        )

        # 质量评估器（可选）
        self.quality_evaluator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # 输出质量分数
        )
        self.loss = AdaptiveBoundaryLoss(training_mode="pretrain")  # 初始为预训练模式

    def forward(self, generated_data, mask):
        """
        前向传播

        Args:
            generated_data: [B, C, H, W] 生成器输出的数据
            true_boundary: [B, C, H, W] 真实边界

        Returns:
            extracted_boundary: [B, C, H, W] 提取的边界
            quality_score: [B, 1] 边界提取质量分数（可选）
        """
        # 提取边界
        extracted_boundary = self.boundary_extractor(generated_data)
        # 二值化处理 ？？？？？？？？？？？？是否要二值化？？？？？？？？？？？？？？？？
        # extracted_boundary = (extracted_boundary >= 0.5).float()
        
        # with torch.no_grad():
        true_boundary = self.ture_extractor(mask)  # 使用真实mask提取边界
        true_boundary = true_boundary.detach()

        # 评估提取质量（可选）
        """boundary_error = F.mse_loss(extracted_boundary, true_boundary, reduction="none")
        boundary_error_avg = boundary_error.mean(dim=(1, 2, 3), keepdim=True)
        quality_score = self.quality_evaluator(boundary_error_avg)"""

        loss_full, _ = self.loss(extracted_boundary, true_boundary)
        loss_full = loss_full * 100
        # loss_full = (1 - torch.exp(-loss_full * 50)) * 2

        # loss_mask, _ = self.loss(extracted_boundary * true_boundary, true_boundary)
        # loss_mask = torch.exp(-loss_mask * 50) * 0.5 * 0.2
        loss_mask, _ = self.loss(
            extracted_boundary, torch.zeros_like(true_boundary)
        )
        # 除以mask=1的数量，再乘以mask整体值数量
        '''mask_sum = true_boundary.sum()
        total_sum = true_boundary.numel()
        if mask_sum > 0:
            loss_mask = loss_mask / mask_sum * total_sum
        else:
            print("Warning: mask_sum is zero, loss_mask will be zero.")
            loss_mask = loss_mask'''
        loss_mask = loss_mask*0.1
        # print(f"loss_full: {loss_full.item()}, loss_mask: {loss_mask.item()}")

        return extracted_boundary, loss_full, loss_mask*0.1


class AdaptiveBoundaryLoss(nn.Module):
    """
    自适应边界损失：根据训练阶段调整损失方向
    """

    def __init__(self, training_mode="pretrain"):
        super().__init__()
        self.training_mode = training_mode  # 'pretrain' or 'adversarial'
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.bce_loss = F.binary_cross_entropy

    def forward(self, extracted_boundary, true_boundary):
        """
        计算自适应损失

        Args:
            extracted_boundary: [B, C, H, W] 提取的边界
            true_boundary: [B, C, H, W] 真实边界

        Returns:
            loss: 损失值
            loss_info: 详细信息
        """
        # 基础边界提取误差
        boundary_mse = self.mse_loss(extracted_boundary, true_boundary)
        boundary_l1 = self.l1_loss(extracted_boundary, true_boundary)
        boundary_bce = self.bce_loss(extracted_boundary,true_boundary)

        if self.training_mode == "pretrain":
            # 预训练阶段：希望提取准确，minimize error
            discriminator_loss = boundary_bce
            loss_direction = "minimize_error"

        elif self.training_mode == "adversarial":
            # 对抗训练阶段：希望提取失败，maximize error
            discriminator_loss = -(boundary_mse * 0.5 + 0.5 * boundary_l1)
            loss_direction = "maximize_error"

        else:
            raise ValueError(f"Unknown training mode: {self.training_mode}")

        loss_info = {
            "boundary_mse": boundary_mse.item(),
            "boundary_l1": boundary_l1.item(),
            "discriminator_loss": discriminator_loss.item(),
            "mode": self.training_mode,
            "direction": loss_direction,
        }

        return discriminator_loss, loss_info

    def set_training_mode(self, mode):
        """切换训练模式"""
        assert mode in ["pretrain", "adversarial"]
        self.training_mode = mode
        print(f"边界质量判别器模式切换到: {mode}")


class BoundaryAwareDiscriminator(nn.Module):
    """
    融合式边界感知判别器

    核心思想：
    1. 提取原始特征
    2. 检测边界不连续性
    3. 融合两种信息进行判别
    """

    def __init__(
        self,
        local_input_channels=1,
        local_input_size=33,
        global_input_channels=1,
        global_input_size=600,
    ):
        super().__init__()

        # === 原始CNN特征提取器 ===
        self.local_cnn = self._build_local_cnn(local_input_channels, local_input_size)
        self.global_cnn = self._build_global_cnn(
            global_input_channels, global_input_size
        )

        # === 边界检测模块 ===
        self.boundary_detector = BoundaryDetector(local_input_size)

        # === 特征融合层 ===
        local_feat_dim = 512  # 来自local_cnn
        global_feat_dim = 1024  # 来自global_cnn
        boundary_feat_dim = 128  # 来自boundary_detector

        self.fusion_layer = nn.Sequential(
            nn.Linear(local_feat_dim + global_feat_dim + boundary_feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),  # 最终判别输出
        )

        # === 边界预测头（辅助任务） ===
        self.boundary_predictor = nn.Sequential(
            nn.Linear(local_feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, local_input_size * local_input_size),
            nn.Sigmoid(),
        )

    def _build_local_cnn(self, input_channels, input_size):
        """构建局部CNN特征提取器"""
        return nn.Sequential(
            # 第一层
            nn.Conv2d(input_channels, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 33x33 -> 16x16
            # 第二层
            nn.Conv2d(64, 128, 5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16 -> 8x8
            # 第三层
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),  # -> 4x4
            # 展平
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def _build_global_cnn(self, input_channels, input_size):
        """构建全局CNN特征提取器"""
        return nn.Sequential(
            # 第一层
            nn.Conv2d(input_channels, 32, 7, stride=2, padding=3),  # 600x600 -> 300x300
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 300x300 -> 150x150
            # 第二层
            nn.Conv2d(32, 64, 5, stride=2, padding=2),  # 150x150 -> 75x75
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 75x75 -> 37x37
            # 第三层
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 37x37 -> 19x19
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 19x19 -> 9x9
            # 第四层
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),  # -> 4x4
            # 展平
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, local_data, local_mask, global_data, global_mask):
        """
        前向传播

        Returns:
            discriminator_output: 真假判别结果
            boundary_prediction: 预测的边界图
            boundary_score: 边界连续性得分
        """
        batch_size = local_data.size(0)

        # === 1. 提取原始特征 ===
        local_features = self.local_cnn(local_data)  # [B, 512]
        global_features = self.global_cnn(global_data)  # [B, 1024]

        # === 2. 边界检测 ===
        boundary_features, boundary_score = self.boundary_detector(
            local_data, local_mask
        )  # [B, 128], [B, 1]

        # === 3. 预测边界图（辅助任务） ===
        predicted_boundary = self.boundary_predictor(local_features)  # [B, 33*33]
        predicted_boundary = predicted_boundary.view(batch_size, 1, 33, 33)

        # === 4. 特征融合 ===
        combined_features = torch.cat(
            [local_features, global_features, boundary_features], dim=1
        )
        discriminator_output = self.fusion_layer(combined_features)  # [B, 1]

        return discriminator_output, predicted_boundary, boundary_score


class BoundaryDetector(nn.Module):
    """
    边界检测模块

    检测mask边界处的数据连续性
    """

    def __init__(self, input_size=33):
        super().__init__()
        self.input_size = input_size

        # Sobel算子用于边缘检测
        self.register_buffer(
            "sobel_x",
            torch.tensor(
                [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32
            ).unsqueeze(0),
        )

        self.register_buffer(
            "sobel_y",
            torch.tensor(
                [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32
            ).unsqueeze(0),
        )

        # 边界特征提取网络
        self.boundary_cnn = nn.Sequential(
            nn.Conv2d(
                3, 32, 3, padding=1
            ),  # 输入: [gradient_x, gradient_y, mask_boundary]
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),  # -> 4x4
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
        )

        # 边界连续性评分器
        self.boundary_scorer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # 输出0-1的连续性得分
        )

    def forward(self, mask):
        """
        检测边界连续性

        Args:
            data: [B, C, H, W] 输入数据
            mask: [B, C, H, W] 对应的mask

        Returns:
            boundary_features: [B, 128] 边界特征
            boundary_score: [B, 1] 边界连续性得分（越高越连续）
        """
        """batch_size = data.size(0)

        # === 1. 计算数据梯度 ===
        # 对每个通道分别计算梯度
        grad_x_list = []
        grad_y_list = []

        for i in range(data.size(1)):
            data_channel = data[:, i : i + 1, :, :]  # [B, 1, H, W]

            # 计算x方向梯度
            grad_x = F.conv2d(data_channel, self.sobel_x, padding=1)
            grad_x_list.append(grad_x)

            # 计算y方向梯度
            grad_y = F.conv2d(data_channel, self.sobel_y, padding=1)
            grad_y_list.append(grad_y)

        # 合并所有通道的梯度
        grad_x = torch.cat(grad_x_list, dim=1).mean(dim=1, keepdim=True)  # [B, 1, H, W]
        grad_y = torch.cat(grad_y_list, dim=1).mean(dim=1, keepdim=True)  # [B, 1, H, W]"""

        # === 2. 检测mask边界 ===
        mask_float = mask.float().mean(dim=1, keepdim=True)  # [B, 1, H, W]
        mask_boundary = self._detect_mask_boundary_conv(mask_float)  # [B, 1, H, W]

        """# === 3. 组合边界信息 ===
        boundary_input = torch.cat(
            [grad_x, grad_y, mask_boundary], dim=1
        )  # [B, 3, H, W]

        # === 4. 提取边界特征 ===
        boundary_features = self.boundary_cnn(boundary_input)  # [B, 128]

        # === 5. 计算边界连续性得分 ===
        boundary_score = self.boundary_scorer(boundary_features)  # [B, 1]"""

        return mask_boundary  # boundary_features, boundary_score

    def _detect_mask_boundary_conv(self, mask, visualize=False):
        """
        使用卷积操作检测mask边界
        Args:
            mask: [B, C, H, W] 输入mask，值为0或1
        Returns:
            boundary: [B, C, H, W] 边界图，边界处为1，其他为0
        """
        # 确保mask是0或1的二值图像
        mask_binary = (mask > 0.5).float()
        batch_size, channels, height, width = mask_binary.shape
        
        # === 早期检测：全0或全1的mask ===
        boundary_list = []
        for c in range(channels):
            mask_channel = mask_binary[:, c : c + 1, :, :]  # [B, 1, H, W]
            
            # 检查是否为全0或全1
            mask_min = mask_channel.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]  # [B, 1, 1, 1]
            mask_max = mask_channel.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]  # [B, 1, 1, 1]
            
            # 如果min==max，说明整个mask都是同一个值（全0或全1）
            is_uniform = (mask_min == mask_max)  # [B, 1, 1, 1]
            
            if is_uniform.all():
                # 全0或全1的情况，直接返回全0边界
                boundary_channel = torch.zeros_like(mask_channel)
            else:
                # 正常的边界检测流程
                # 定义边缘检测卷积核 - 检测上下左右邻居
                edge_kernel = (
                    torch.tensor(
                        [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                        dtype=torch.float32,
                        device=mask.device,
                    )
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
                
                # === 第一步：使用卷积检测内部边界 ===
                edge_response = F.conv2d(mask_channel, edge_kernel, padding=1)
                boundary_channel = (edge_response != 0).float()
                
                # 排除图片最外圈的像素（避免卷积边界效应）
                outer_edge_mask = torch.zeros_like(mask_channel)
                outer_edge_mask[:, :, 0, :] = 1  # 顶边
                outer_edge_mask[:, :, -1, :] = 1  # 底边
                outer_edge_mask[:, :, :, 0] = 1  # 左边
                outer_edge_mask[:, :, :, -1] = 1  # 右边
                boundary_channel = boundary_channel * (1 - outer_edge_mask)
                
                # === 第二步：主动检测图片边缘的0-1变化 ===
                edge_boundaries = self._detect_edge_transitions_optimized(mask_channel)
                
                # === 第三步：合并内部边界和边缘边界 ===
                boundary_channel = torch.clamp(boundary_channel + edge_boundaries, 0, 1)
            
            boundary_list.append(boundary_channel)
        
        boundary = torch.cat(boundary_list, dim=1)
        
        if visualize:
            self._visualize_boundary_detection(mask_binary, boundary)
        
        return boundary

    def _detect_edge_transitions_optimized(self, mask_channel):
        """
        优化版本：使用向量化操作检测边缘0-1变化

        更高效的实现，避免嵌套循环
        """
        batch_size, _, height, width = mask_channel.shape
        device = mask_channel.device

        # 初始化边缘边界图
        edge_boundaries = torch.zeros_like(mask_channel)

        # === 检测水平变化（顶边和底边） ===
        if width > 1:
            # 顶边水平变化
            top_row = mask_channel[:, :, 0, :]  # [B, 1, W]
            top_diff = torch.abs(top_row[:, :, 1:] - top_row[:, :, :-1])  # [B, 1, W-1]
            top_changes = (top_diff > 0.5).float()  # 检测0-1变化

            # 标记变化位置（相邻的两个像素都标记）
            edge_boundaries[:, :, 0, :-1] += top_changes
            edge_boundaries[:, :, 0, 1:] += top_changes

            # 底边水平变化
            bottom_row = mask_channel[:, :, -1, :]  # [B, 1, W]
            bottom_diff = torch.abs(
                bottom_row[:, :, 1:] - bottom_row[:, :, :-1]
            )  # [B, 1, W-1]
            bottom_changes = (bottom_diff > 0.5).float()

            edge_boundaries[:, :, -1, :-1] += bottom_changes
            edge_boundaries[:, :, -1, 1:] += bottom_changes

        # === 检测垂直变化（左边和右边） ===
        if height > 1:
            # 左边垂直变化
            left_col = mask_channel[:, :, :, 0]  # [B, 1, H]
            left_diff = torch.abs(
                left_col[:, :, 1:] - left_col[:, :, :-1]
            )  # [B, 1, H-1]
            left_changes = (left_diff > 0.5).float()

            edge_boundaries[:, :, :-1, 0] += left_changes
            edge_boundaries[:, :, 1:, 0] += left_changes

            # 右边垂直变化
            right_col = mask_channel[:, :, :, -1]  # [B, 1, H]
            right_diff = torch.abs(
                right_col[:, :, 1:] - right_col[:, :, :-1]
            )  # [B, 1, H-1]
            right_changes = (right_diff > 0.5).float()

            edge_boundaries[:, :, :-1, -1] += right_changes
            edge_boundaries[:, :, 1:, -1] += right_changes

        # 确保边界值为0或1
        edge_boundaries = torch.clamp(edge_boundaries, 0, 1)

        return edge_boundaries

    def _visualize_boundary_detection(self, mask, boundary):
        """可视化边界检测结果"""
        mask_np = mask.detach().cpu().numpy()
        boundary_np = boundary.detach().cpu().numpy()

        # 只显示前2个样本
        num_samples = min(2, mask.size(0))

        for i in range(num_samples):
            plt.figure(figsize=(12, 4))

            # 原始mask
            plt.subplot(1, 3, 1)
            plt.title("Input Mask")
            plt.imshow(mask_np[i, 0], cmap="gray", vmin=0, vmax=1)
            plt.axis("off")

            # 检测到的边界
            plt.subplot(1, 3, 2)
            plt.title("Detected Boundary")
            plt.imshow(boundary_np[i, 0], cmap="hot", vmin=0, vmax=1)
            plt.axis("off")

            # 叠加显示
            plt.subplot(1, 3, 3)
            plt.title("Mask + Boundary Overlay")
            plt.imshow(mask_np[i, 0], cmap="gray", alpha=0.7, vmin=0, vmax=1)
            plt.imshow(boundary_np[i, 0], cmap="Reds", alpha=0.8, vmin=0, vmax=1)
            plt.axis("off")

            plt.tight_layout()
            plt.show()


# ============================================================================
# 训练损失函数
# ============================================================================
class BoundaryAwareLoss(nn.Module):
    """
    边界感知损失函数

    结合三种损失：
    1. 对抗损失
    2. 边界预测损失
    3. 边界连续性损失
    """

    def __init__(
        self, adv_weight=1.0, boundary_pred_weight=0.5, boundary_cont_weight=0.3
    ):
        super().__init__()
        self.adv_weight = adv_weight
        self.boundary_pred_weight = boundary_pred_weight
        self.boundary_cont_weight = boundary_cont_weight

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        discriminator_output,
        predicted_boundary,
        boundary_score,
        real_labels,
        true_boundary=None,
        is_real_data=None,
    ):
        """
        计算总损失

        Args:
            discriminator_output: [B, 1] 判别器输出
            predicted_boundary: [B, 1, H, W] 预测的边界图
            boundary_score: [B, 1] 边界连续性得分
            real_labels: [B, 1] 真假标签
            true_boundary: [B, 1, H, W] 真实边界图（可选）
            is_real_data: [B] 是否为真实数据的标记
        """
        # 1. 对抗损失
        adv_loss = self.bce_loss(discriminator_output, real_labels)

        total_loss = self.adv_weight * adv_loss
        loss_dict = {"adv_loss": adv_loss.item()}

        # 2. 边界预测损失（如果提供了真实边界）
        if true_boundary is not None:
            boundary_pred_loss = self.mse_loss(predicted_boundary, true_boundary)
            total_loss += self.boundary_pred_weight * boundary_pred_loss
            loss_dict["boundary_pred_loss"] = boundary_pred_loss.item()

        # 3. 边界连续性损失
        if is_real_data is not None:
            # 真实数据应该有高连续性得分，生成数据应该有低连续性得分
            target_continuity = is_real_data.float().unsqueeze(1)  # [B, 1]
            boundary_cont_loss = self.mse_loss(boundary_score, target_continuity)
            total_loss += self.boundary_cont_weight * boundary_cont_loss
            loss_dict["boundary_cont_loss"] = boundary_cont_loss.item()

        loss_dict["total_loss"] = total_loss.item()
        return total_loss, loss_dict


# ============================================================================
# 训练函数示例
# ============================================================================


def train_boundary_aware_discriminator(model, dataloader, optimizer, device, epoch):
    """
    边界感知判别器训练函数
    """
    model.train()
    criterion = BoundaryAwareLoss()

    total_losses = {
        "adv_loss": 0,
        "boundary_pred_loss": 0,
        "boundary_cont_loss": 0,
        "total_loss": 0,
    }
    total_real_acc = 0
    total_fake_acc = 0
    num_batches = 0

    for batch_idx, batch_data in enumerate(dataloader):
        # 准备数据
        (
            batch_local_inputs,
            batch_local_masks,
            batch_local_targets,
            batch_global_inputs,
            batch_global_masks,
            batch_global_targets,
            metadata,
        ) = batch_data

        # 移动到设备
        batch_local_inputs = batch_local_inputs.to(device)
        batch_local_masks = batch_local_masks.to(device)
        batch_local_targets = batch_local_targets.to(device)
        batch_global_inputs = batch_global_inputs.to(device)
        batch_global_masks = batch_global_masks.to(device)

        batch_size = batch_local_inputs.size(0)

        # === 训练真实数据 ===
        real_labels = torch.ones(batch_size, 1, device=device)
        real_is_real = torch.ones(batch_size, device=device)

        # 生成真实边界图（可选）
        real_boundary = generate_true_boundary_map(
            batch_local_targets, batch_local_masks
        )

        # 前向传播
        real_output, real_pred_boundary, real_boundary_score = model(
            batch_local_targets,
            batch_local_masks,
            batch_global_inputs,
            batch_global_masks,
        )

        # 计算损失
        real_loss, real_loss_dict = criterion(
            real_output,
            real_pred_boundary,
            real_boundary_score,
            real_labels,
            real_boundary,
            real_is_real,
        )

        # === 训练生成数据 ===
        # 这里需要你的生成器生成假数据
        with torch.no_grad():
            fake_local_data = generate_fake_data(
                batch_local_inputs, batch_local_masks
            )  # 你的生成器

        fake_labels = torch.zeros(batch_size, 1, device=device)
        fake_is_real = torch.zeros(batch_size, device=device)

        fake_output, fake_pred_boundary, fake_boundary_score = model(
            fake_local_data, batch_local_masks, batch_global_inputs, batch_global_masks
        )

        fake_loss, fake_loss_dict = criterion(
            fake_output,
            fake_pred_boundary,
            fake_boundary_score,
            fake_labels,
            None,
            fake_is_real,  # 不需要真实边界图
        )

        # === 总损失和优化 ===
        total_loss = (real_loss + fake_loss) / 2

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # === 计算准确率 ===
        with torch.no_grad():
            real_acc = (torch.sigmoid(real_output) >= 0.5).float().mean().item()
            fake_acc = (torch.sigmoid(fake_output) < 0.5).float().mean().item()

        # 累积统计
        for key in total_losses:
            if key in real_loss_dict:
                total_losses[key] += real_loss_dict[key]
            if key in fake_loss_dict:
                total_losses[key] += fake_loss_dict[key]

        total_real_acc += real_acc
        total_fake_acc += fake_acc
        num_batches += 1

        # 打印进度
        if batch_idx % 50 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}:")
            print(f"  Real Acc: {real_acc:.3f}, Fake Acc: {fake_acc:.3f}")
            print(f"  Real Boundary Score: {real_boundary_score.mean().item():.3f}")
            print(f"  Fake Boundary Score: {fake_boundary_score.mean().item():.3f}")

    # 返回平均统计
    avg_losses = {k: v / (num_batches * 2) for k, v in total_losses.items()}
    avg_real_acc = total_real_acc / num_batches
    avg_fake_acc = total_fake_acc / num_batches

    return avg_losses, avg_real_acc, avg_fake_acc


def generate_true_boundary_map(data, mask):
    """
    生成真实数据的边界图（用于监督学习）
    这里可以返回None，表示不使用边界预测的监督学习
    """
    return None  # 或者实现具体的边界生成逻辑


def generate_fake_data(inputs, masks):
    """
    生成假数据的占位函数
    在实际使用中，这里应该调用你的生成器
    """
    # 这里应该是你的生成器的调用
    # return your_generator(inputs, masks)
    return inputs  # 占位符


# ============================================================================
# 使用示例
# ============================================================================

"""
使用方法：

1. 创建边界感知判别器：
discriminator = BoundaryAwareDiscriminator(
    local_input_channels=1,
    local_input_size=33,
    global_input_channels=1,
    global_input_size=600
).to(device)

2. 在你的训练循环中替换原有的判别器训练：
# 原来的代码：
# fake_predictions, fake_lf, fake_gf = model_cd(...)

# 新的代码：
fake_output, fake_pred_boundary, fake_boundary_score = discriminator(
    fake_data, local_masks, global_data, global_masks
)

3. 使用新的损失函数：
criterion = BoundaryAwareLoss()
loss, loss_dict = criterion(
    discriminator_output, predicted_boundary, boundary_score,
    labels, true_boundary, is_real_data
)

这样你就能利用边界不连续性来增强判别器的判别能力！
"""
if __name__ == "__main__":
    json_dir = r"F:\Dataset\Simulate\data0\json"  # 局部json
    array_dir = r"F:\Dataset\Simulate\data0\arraynmask\array"  # 全局array
    mask_dir = r"F:\Dataset\Simulate\data0\arraynmask\mask"  # 全局mask
    target_dir = (
        r"F:\Dataset\Simulate\data0\groundtruthstatus\statusarray"  # 全局target
    )
    try:
        dataset = DemDataset(
            json_dir,
            array_dir,
            mask_dir,
            target_dir,
            min_valid_pixels=20,
            max_valid_pixels=900,
            enable_synthetic_masks=True,  # 启用生成式mask
            synthetic_ratio=1.0,  # 每个原始数据生成1个合成数据
            analyze_data=False,  # 分析数据质量
        )
        full_dataset = dataset
        print(f"数据集加载成功，总数据量: {len(full_dataset)}")
        print(f"  - 原始数据: {dataset.original_length}")
        if dataset.enable_synthetic_masks:
            print(f"  - 生成式数据: {dataset.synthetic_length}")
    except Exception as e:
        print(f"加载数据集失败: {e}")
        raise

    # ==================== 简洁的数据分割（仅需这几行）====================
    train_dataset, val_dataset, test_dataset = simple_dataset_split(
        full_dataset,
        test_ratio=0.1,  # 10% 测试集
        val_ratio=0.1,  # 25% 验证集（从剩余90%中取）
        seed=42,  # 固定种子确保可重现
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=fast_collate_fn,  # 使用优化的collate函数
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    train_boundary_aware_discriminator(
        model=BoundaryAwareDiscriminator().to("cuda"),
        dataloader=train_loader,
        optimizer=torch.optim.Adam(BoundaryAwareDiscriminator().parameters(), lr=0.001),
        device="cuda",
        epoch=1,
    )
