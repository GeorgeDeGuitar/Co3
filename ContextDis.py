import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import spectral_norm
from torch.nn import Parameter
import shutil
import os
import webbrowser
import traceback


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


class MaskAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(MaskAttentionModule, self).__init__()
        self.conv = nn.Conv2d(in_channels + 1, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask):
        mask_resized = F.interpolate(
            mask, size=x.shape[2:], mode="bilinear", align_corners=True
        )
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
            nn.ReLU(inplace=True),
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


class ResidualCorrectionModule(nn.Module):
    """
    残差修正模块 - 专门用于修正边界不连续性
    """

    def __init__(self, input_channels=1):
        super(ResidualCorrectionModule, self).__init__()

        # 坡度计算模块
        self.slope_calculator = SlopeCalculator()

        # 残差学习网络 - 适配33x33小尺寸
        self.residual_net = nn.Sequential(
            # 输入: [output, mask, slope_x, slope_y] = input_channels + 1 + 2
            nn.Conv2d(
                input_channels + 3, 32, kernel_size=3, padding=1
            ),  # 减小kernel_size和通道数
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # 输出残差修正
            nn.Conv2d(16, input_channels, kernel_size=3, padding=1),
            nn.Tanh(),  # 限制残差幅度
        )

        # 残差权重网络 - 学习在哪里应用残差
        self.weight_net = nn.Sequential(
            nn.Conv2d(input_channels + 3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),  # 权重在0-1之间
        )

        # 残差缩放因子
        self.residual_scale = nn.Parameter(torch.tensor(0.08))

    def forward(self, network_output, mask):
        """
        Args:
            network_output: 当前网络的输出 [B, C, H, W]
            mask: 对应的mask [B, 1, H, W]
        Returns:
            corrected_output: 残差修正后的输出
        """
        # 计算坡度
        slope_x, slope_y = self.slope_calculator(network_output)

        # 构建残差模块输入
        residual_input = torch.cat([network_output, mask, slope_x, slope_y], dim=1)

        # 计算残差修正
        residual_correction = self.residual_net(residual_input)

        # 计算残差权重（在哪里应用残差）
        residual_weights = self.weight_net(residual_input)

        # 应用残差修正
        '''corrected_output = (
            network_output
            + self.residual_scale * residual_correction * residual_weights
        )

        return corrected_output'''
        res = (self.residual_scale * residual_correction * residual_weights)
        return res


class SlopeCalculator(nn.Module):
    """
    计算DEM的坡度信息
    """

    def __init__(self):
        super(SlopeCalculator, self).__init__()
        # Sobel算子
        self.register_buffer(
            "sobel_x",
            torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]).float().unsqueeze(0),
        )

        self.register_buffer(
            "sobel_y",
            torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]).float().unsqueeze(0),
        )

    def forward(self, dem):
        """
        计算DEM的坡度
        Args:
            dem: DEM数据 [B, C, H, W]
        Returns:
            slope_x, slope_y: x和y方向的坡度
        """
        B, C, H, W = dem.shape

        # 对每个通道分别计算坡度
        slope_x_list = []
        slope_y_list = []

        for c in range(C):
            dem_c = dem[:, c : c + 1, :, :]  # [B, 1, H, W]

            # 计算梯度
            grad_x = F.conv2d(dem_c, self.sobel_x, padding=1)
            grad_y = F.conv2d(dem_c, self.sobel_y, padding=1)

            slope_x_list.append(grad_x)
            slope_y_list.append(grad_y)

        slope_x = torch.cat(slope_x_list, dim=1)
        slope_y = torch.cat(slope_y_list, dim=1)

        return slope_x, slope_y


class CompletionNetwork(nn.Module):
    def __init__(self, input_channels=1, local_size=33):
        super(CompletionNetwork, self).__init__()
        self.debug = False  # 设置为True以便调试
        self.local_size = local_size

        # 计算需要的上采样次数
        if local_size == 33:
            self.num_downsamples = 2
        elif local_size == 65:
            self.num_downsamples = 3
        elif local_size == 129:
            self.num_downsamples = 4
        elif local_size == 257:
            self.num_downsamples = 5
        else:
            raise ValueError(f"Unsupported local size: {local_size}")

        # Early attention modules
        self.early_local_attention = EarlyAttentionModule(input_channels)
        self.early_global_attention = EarlyAttentionModule(input_channels)

        # Local encoder
        self.local_enc1 = nn.Sequential(
            nn.Conv2d(input_channels * 2, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.local_enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
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
            nn.ReLU(inplace=True),
        )

        # 根据local_size添加额外的编码器层
        self.extra_encoders = nn.ModuleList()
        in_ch = 256
        for i in range(self.num_downsamples - 2):
            self.extra_encoders.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(in_ch),
                    nn.ReLU(inplace=True),
                )
            )

        # Global encoder
        self.global_enc1 = nn.Sequential(
            nn.Conv2d(input_channels * 2, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.global_enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.global_enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Attention and fusion layers
        self.global_attention = MaskAttentionModule(256)
        self.fusion_attention = MaskAttentionModule(256)

        self.fusion = nn.Sequential(
            nn.Conv2d(512, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        self.boundary_aware = BoundaryAwareModule(256)

        # Decoder with skip connections
        self.extra_decoders = nn.ModuleList()
        self.extra_skip_fusions = nn.ModuleList()

        for i in range(self.num_downsamples - 2):
            self.extra_decoders.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        256, 256, kernel_size=3, stride=2, padding=1, output_padding=1
                    ),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                )
            )
            # Skip connection融合
            self.extra_skip_fusions.append(
                nn.Sequential(
                    nn.Conv2d(256 + 256, 256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                )
            )

        # Decoder layers
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(
                256, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.skip_fusion1 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.skip_fusion2 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.decoder3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, input_channels, kernel_size=3, padding=1),
        )

        # Other modules
        self.transition_conv = nn.Conv2d(
            input_channels + 1, input_channels, kernel_size=3, stride=1, padding=1
        )
        self.offset_module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(input_channels * 2, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),
        )
        self.offset_scale = nn.Parameter(torch.tensor(20.0))
        self.offset_transition = nn.Sequential(
            nn.Conv2d(input_channels + 1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, input_channels, kernel_size=3, padding=1),
        )
        self.residual_module = ResidualCorrectionModule(input_channels)

    def forward(self, local_input, local_mask, global_input, global_mask):
        # print(f"local_input size: {local_input.size()}, local_mask size: {local_mask.size()}, global_input size: {global_input.size()}, global_mask size: {global_mask.size()}")
        local_mask = local_mask.float()
        global_mask = global_mask.float()

        # 归一化
        global_min = global_input.min()
        global_max = global_input.max()

        if global_max - global_min > 1e-8:
            global_input = (global_input - global_min) / (global_max - global_min)
            local_input = (local_input - global_min) / (global_max - global_min)
        else:
            global_input = torch.full_like(global_input, 0.5)
            local_input = torch.full_like(local_input, 0.5)

        # Early attention
        local_input_attended = self.early_local_attention(local_input, local_mask)
        global_input_attended = self.early_global_attention(global_input, global_mask)

        local_x = torch.cat([local_input_attended, local_mask], dim=1)
        global_x = torch.cat([global_input_attended, global_mask], dim=1)

        # Local encoder
        local_feat1 = self.local_enc1(local_x)
        if self.debug:
            print(f"local_feat1 size: {local_feat1.size()}")
        local_feat2 = self.local_enc2(local_feat1)
        if self.debug:
            print(f"local_feat2 size: {local_feat2.size()}")
        local_feat3 = self.local_enc3(local_feat2)
        if self.debug:
            print(f"local_feat3 size: {local_feat3.size()}")

        # 保存所有特征用于skip connections
        # 注意：我们分开保存基础特征和额外特征
        base_local_feats = [local_feat1, local_feat2, local_feat3]
        extra_local_feats = []

        # 额外的编码器层
        x = local_feat3
        for encoder in self.extra_encoders:
            x = encoder(x)
            extra_local_feats.append(x)
            if self.debug:
                print(f"extra_local_feats[-1] size: {x.size()}")

        # Global encoder
        global_feat1 = self.global_enc1(global_x)
        if self.debug:
            print(f"global_feat1 size: {global_feat1.size()}")
        global_feat2 = self.global_enc2(global_feat1)
        if self.debug:
            print(f"global_feat2 size: {global_feat2.size()}")
        global_feat3 = self.global_enc3(global_feat2)
        if self.debug:
            print(f"global_feat3 size: {global_feat3.size()}")

        # Feature fusion - 使用最深层的local特征
        if extra_local_feats:
            deepest_local = extra_local_feats[-1]
        else:
            deepest_local = local_feat3

        combined_features = torch.cat([deepest_local, global_feat3], dim=1)
        if self.debug:
            print(f"combined_features size: {combined_features.size()}")
        fused_features = self.fusion(combined_features)
        if self.debug:
            print(f"fused_features size: {fused_features.size()}")

        # Decoder with skip connections
        x = fused_features

        # 额外的解码器层（倒序）- 与额外的编码器层对应
        for i in range(len(self.extra_decoders) - 1, -1, -1):
            x = self.extra_decoders[i](x)
            if self.debug:
                print(f"extra_decoder[{i}] output size: {x.size()}")

            # 对应的skip connection - 从extra_local_feats获取
            # 注意：extra_decoders是倒序的，所以需要调整索引
            # 最后一个decoder对应倒数第二个encoder特征（因为最后一个已经用于fusion）
            if i < len(extra_local_feats) - 1:
                skip_feat = extra_local_feats[len(extra_local_feats) - 2 - i]
                if self.debug:
                    print(
                        f"skip_feat from extra_local_feats[{len(extra_local_feats) - 2 - i}] size: {skip_feat.size()}"
                    )

                # 检查尺寸是否匹配，如果不匹配则调整
                if x.shape[2:] != skip_feat.shape[2:]:
                    if self.debug:
                        print(
                            f"Size mismatch: x-{x.size()}, skip_feat-{skip_feat.size()}"
                        )
                    # 调整x的大小以匹配skip_feat
                    x = F.interpolate(
                        x, size=skip_feat.shape[2:], mode="bilinear", align_corners=True
                    )
                    if self.debug:
                        print(f"x resized to: {x.size()}")

                skip = torch.cat([x, skip_feat], dim=1)
                x = self.extra_skip_fusions[i](skip)
                if self.debug:
                    print(f"extra_skip_fusions[{i}] output size: {x.size()}")

        # 原有的解码器层
        # 第一个标准解码器
        dec1 = self.decoder1(x)
        if self.debug:
            print(f"dec1 size: {dec1.size()}")

        # 调整skip connection大小 - 与local_feat2对应
        if dec1.shape[2:] != local_feat2.shape[2:]:
            if self.debug:
                print(
                    f"different size: dec1-{dec1.size()}, local_feat2-{local_feat2.size()}"
                )
            # 使用插值进行大小调整
            dec1_resized = F.interpolate(
                dec1, size=local_feat2.shape[2:], mode="bilinear", align_corners=True
            )
            if self.debug:
                print(f"dec1_resized size: {dec1_resized.size()}")
            skip1 = torch.cat([dec1_resized, local_feat2], dim=1)
        else:
            skip1 = torch.cat([dec1, local_feat2], dim=1)
        skip1_fused = self.skip_fusion1(skip1)
        if self.debug:
            print(f"skip1_fused size: {skip1_fused.size()}")

        # 第二个标准解码器
        dec2 = self.decoder2(skip1_fused)
        if self.debug:
            print(f"dec2 size: {dec2.size()}")

        # 调整skip connection大小 - 与local_feat1对应
        if dec2.shape[2:] != local_feat1.shape[2:]:
            if self.debug:
                print(
                    f"different size: dec2-{dec2.size()}, local_feat1-{local_feat1.size()}"
                )
            # 使用插值进行大小调整
            dec2_resized = F.interpolate(
                dec2, size=local_feat1.shape[2:], mode="bilinear", align_corners=True
            )
            if self.debug:
                print(f"dec2_resized size: {dec2_resized.size()}")
            skip2 = torch.cat([dec2_resized, local_feat1], dim=1)
        else:
            skip2 = torch.cat([dec2, local_feat1], dim=1)
        skip2_fused = self.skip_fusion2(skip2)
        if self.debug:
            print(f"skip2_fused size: {skip2_fused.size()}")

        # 最终输出
        output = self.decoder3(skip2_fused)
        if self.debug:
            print(f"output size: {output.size()}")

        # 确保输出与输入尺寸一致
        if output.shape[2:] != local_input.shape[2:]:
            output = F.interpolate(
                output, size=local_input.shape[2:], mode="bilinear", align_corners=True
            )
            if self.debug:
                print(f"output resized to: {output.size()}")

        output = local_input * local_mask + output * (1 - local_mask)

        # 残差修正
        res = self.residual_module(output, local_mask)
        final_output = output + res * (1 - local_mask)

        # 逆归一化
        if global_max - global_min > 1e-8:
            output = final_output * (global_max - global_min) + global_min
        else:
            output = final_output

        return output


class EnhancedTerrainFeatureExtractor(nn.Module):
    """简化版地形特征提取器：仅包含坡度和原始高程"""

    def __init__(self, in_channels):
        super(EnhancedTerrainFeatureExtractor, self).__init__()

        # 坡度提取（使用Sobel算子）
        self.sobel_x = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, bias=False
        )
        self.sobel_y = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, bias=False
        )

        # 初始化滤波器
        with torch.no_grad():
            # Sobel滤波器
            sobel_x = torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
            )
            sobel_y = torch.tensor(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
            )

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

        # 返回简化的地形特征：高程 + 坡度
        features = torch.cat([x, slope], dim=1)
        features = F.dropout(features, p=0.1, training=self.training)

        return features


class StabilizedSelfAttention(nn.Module):
    """稳定版自注意力模块"""

    def __init__(self, in_channels):
        super(StabilizedSelfAttention, self).__init__()
        # 使用谱归一化提高稳定性
        self.query_conv = spectral_norm(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        )
        self.key_conv = spectral_norm(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        )
        self.value_conv = spectral_norm(
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )

        # 从较小值开始
        self.gamma = nn.Parameter(torch.zeros(1))

        # 缩放因子
        self.scale_factor = torch.sqrt(
            torch.tensor(in_channels // 8, dtype=torch.float32)
        )

    def forward(self, x):
        batch_size, C, height, width = x.size()

        # 检查输入是否包含NaN
        if torch.isnan(x).any():
            print("警告: SelfAttention输入包含NaN!")
            x = torch.nan_to_num(x, nan=0.0)

        # 压缩特征以减少内存使用
        proj_query = (
            self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        )
        proj_key = self.key_conv(x).view(batch_size, -1, height * width)

        # 计算注意力图，添加缩放因子提高数值稳定性
        energy = torch.bmm(proj_query, proj_key) / self.scale_factor

        # 使用更安全的softmax
        attention = F.softmax(energy, dim=-1)

        # 检查注意力权重是否包含NaN
        if torch.isnan(attention).any():
            print("警告: 注意力权重包含NaN!")
            attention = torch.nan_to_num(attention, nan=1.0 / attention.size(-1))

        # 应用注意力
        proj_value = self.value_conv(x).view(batch_size, -1, height * width)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, height, width)

        # 使用更小的初始gamma值，并限制其最大影响
        gamma_clamped = torch.clamp(self.gamma, -1.0, 1.0)
        return gamma_clamped * out + x


class DisGatedConv2d(nn.Module):
    """稳定版门控卷积层"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(DisGatedConv2d, self).__init__()
        self.conv2d = spectral_norm(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias,
            )
        )
        self.mask_conv2d = spectral_norm(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias,
            )
        )

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
                mask = F.interpolate(mask, size=gates.shape[2:], mode="nearest")
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
            nn.Sigmoid(),
        )

        # 第一层卷积 - 使用门控卷积处理掩码
        self.conv1 = nn.Conv2d(
            enhanced_channels, 64, kernel_size=4, stride=2, padding=1
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        # 第二层卷积
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

        # 第三层卷积
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.act3 = nn.LeakyReLU(0.2, inplace=True)

        # 添加自注意力机制
        self.self_attention = StabilizedSelfAttention(256)

        # 第四层卷积
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.act4 = nn.LeakyReLU(0.2, inplace=True)

        # 第五层卷积 - 保留一个额外层以处理大的输入尺寸
        self.conv5 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
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
            nn.Dropout(0.5),  # 增加Dropout
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
        # current_mask = mask

        # 第一层卷积，应用门控卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        # 更新掩码尺寸
        # current_mask = F.interpolate(current_mask, scale_factor=0.5, mode="nearest")

        # 第二层卷积
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        # 更新掩码尺寸
        # current_mask = F.interpolate(current_mask, scale_factor=0.5, mode="nearest")

        # 第三层卷积
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)

        # 应用自注意力
        x = self.self_attention(x)

        # 更新掩码尺寸
        # current_mask = F.interpolate(current_mask, scale_factor=0.5, mode="nearest")

        # 第四层卷积
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.act4(x)
        # 更新掩码尺寸
        # current_mask = F.interpolate(current_mask, scale_factor=0.5, mode="nearest")

        # 第五层卷积
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.act5(x)

        # 扁平化并检查尺寸
        x = self.flatten(x)

        # 检查实际特征尺寸与预期尺寸是否匹配
        if x.size(1) != self.feature_size:
            print(f"警告: 特征尺寸不匹配! 预期 {self.feature_size}，实际 {x.size(1)}")
            # 对于add_graph调用时，调整线性层以匹配实际尺寸
            if not hasattr(self, "_feature_size_adjusted") and not self.training:
                self._feature_size_adjusted = True
                self.linear = nn.Sequential(
                    spectral_norm(nn.Linear(x.size(1), 1024)),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(0.5),
                )

        # 全连接层
        x = self.linear(x)

        # 检查输出是否包含NaN
        if torch.isnan(x).any():
            print("警告: GlobalDiscriminator输出包含NaN!")
            x = torch.nan_to_num(x, nan=0.0)

        return x


class LocalDiscriminator(nn.Module):
    def __init__(self, input_channels=1, input_size=33):
        super(LocalDiscriminator, self).__init__()
        self.input_shape = (input_channels, input_size, input_size)
        self.input_size = input_size

        # 根据输入尺寸调整网络深度
        if input_size == 33:
            self.num_layers = 4
        elif input_size == 65:
            self.num_layers = 5
        elif input_size == 129:
            self.num_layers = 6
        elif input_size == 257:
            self.num_layers = 7
        else:
            raise ValueError(f"Unsupported input size: {input_size}")

        self.terrain_extractor = EnhancedTerrainFeatureExtractor(input_channels)
        terrain_channels = input_channels * 2

        # 构建卷积层
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.acts = nn.ModuleList()

        channels = [terrain_channels, 64, 128, 256, 512, 512, 512, 512]

        for i in range(self.num_layers):
            in_ch = channels[i]
            out_ch = channels[i + 1]

            self.convs.append(
                spectral_norm(
                    nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
                )
            )
            self.bns.append(nn.BatchNorm2d(out_ch))
            self.acts.append(nn.LeakyReLU(0.2, inplace=True))

        # 计算特征尺寸
        h = w = input_size
        for _ in range(self.num_layers):
            h = (h + 2 * 1 - 4) // 2 + 1
            w = (w + 2 * 1 - 4) // 2 + 1

        self.feature_size = channels[self.num_layers] * h * w

        self.flatten = nn.Flatten()
        self.linear = nn.Sequential(
            spectral_norm(nn.Linear(self.feature_size, 1024)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, local, mask):
        if torch.isnan(local).any():
            print("警告: LocalDiscriminator输入包含NaN!")
            local = torch.nan_to_num(local, nan=0.0)

        x = self.terrain_extractor(local)

        # 通过所有卷积层
        for i in range(self.num_layers):
            x = self.convs[i](x)
            x = self.bns[i](x)
            x = self.acts[i](x)

        x = self.flatten(x)

        if x.size(1) != self.feature_size:
            print(f"警告: 特征尺寸不匹配! 预期 {self.feature_size}，实际 {x.size(1)}")

        x = self.linear(x)

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
            nn.Dropout(0.6),  # 增加dropout以提高正则化
        )

        # 分类器 - 更简单的架构
        self.classifier = nn.Sequential(
            spectral_norm(nn.Linear(1024, 512)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            spectral_norm(nn.Linear(512, 1)),  # 输出logits，适配BCEWithLogitsLoss
        )
    
    def ensure_4d(self, x):
        # 如果是5D，例如 [B, 1, 1, H, W]，就 squeeze 掉第二个维度
        if x.dim() == 5 and x.size(1) == 1:
            x = x.squeeze(1)
        # 如果是3D，例如 [B, H, W]，就 unsqueeze 成 [B, 1, H, W]
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        # 如果是4D就不处理
        return x


    def forward(self, x_ld, x_lm, x_gd, x_gm):
        x_ld = self.ensure_4d(x_ld)
        x_lm = self.ensure_4d(x_lm)
        x_gd = self.ensure_4d(x_gd)
        x_gm = self.ensure_4d(x_gm)
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


# 修改main函数
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 测试不同尺寸
    for local_size in [33, 65, 129, 257]:
        print(f"\n测试 {local_size}x{local_size} 尺寸:")

        # 测试生成器
        try:
            cn = CompletionNetwork(local_size=local_size).to(device)
            local_input = torch.randn(1, 1, local_size, local_size).to(device)
            local_mask = torch.ones(1, 1, local_size, local_size).to(device)
            global_input = torch.randn(1, 1, 600, 600).to(device)
            global_mask = torch.ones(1, 1, 600, 600).to(device)

            output = cn(local_input, local_mask, global_input, global_mask)
            print(f"  生成器输出形状: {output.shape} ✓")
        except Exception as e:
            print(f"  生成器错误: {e}")
            traceback.print_exc()

        # 测试局部判别器
        try:
            ld = LocalDiscriminator(input_size=local_size).to(device)
            local_input = torch.randn(1, 1, local_size, local_size).to(device)
            local_mask = torch.ones(1, 1, local_size, local_size).to(device)

            output = ld(local_input, local_mask)
            print(f"  局部判别器输出形状: {output.shape} ✓")
        except Exception as e:
            print(f"  局部判别器错误: {e}")

    # 测试上下文判别器
    print("\n测试上下文判别器:")
    for local_size in [33, 65, 129, 257]:
        try:
            cd = ContextDiscriminator(local_input_size=local_size).to(device)
            local_input = torch.randn(1, 1, local_size, local_size).to(device)
            local_mask = torch.ones(1, 1, local_size, local_size).to(device)
            global_input = torch.randn(1, 1, 600, 600).to(device)
            global_mask = torch.ones(1, 1, 600, 600).to(device)

            output, _, _ = cd(local_input, local_mask, global_input, global_mask)
            print(f"  {local_size}x{local_size}: 输出形状 {output.shape} ✓")
        except Exception as e:
            print(f"  {local_size}x{local_size}: 错误 {e}")
