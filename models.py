import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import shutil
import os
import webbrowser

class EarlyAttentionModule(nn.Module):
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
        
        # 为全局输入添加早期注意力模块
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
            nn.Conv2d(input_channels * 2, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.global_enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.global_enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
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
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 在解码器第一层后合并跳跃连接的特征
        self.skip_fusion1 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
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

    def forward(self, local_input, local_mask, global_input, global_mask):
        # 统一标准化
        global_input_std, local_input_std, global_mean, global_std = standardize_input(global_input, local_input)
        input_size = local_input.size()[2:]
        
        # 对全局输入应用早期注意力
        global_input_attended = self.early_global_attention(global_input, global_mask)
        
        # 将输入图像和掩码连接起来
        local_x = torch.cat([local_input, local_mask], dim=1)
        global_x = torch.cat([global_input_attended, global_mask], dim=1)
        
        # 本地编码器的前向传播，保存中间特征用于跳跃连接
        local_feat1 = self.local_enc1(local_x)        # 第一层特征 (64通道)
        local_feat2 = self.local_enc2(local_feat1)    # 第二层特征 (128通道)
        local_feat3 = self.local_enc3(local_feat2)    # 第三层特征 (256通道)
        
        # 全局编码器的前向传播
        global_feat1 = self.global_enc1(global_x)
        global_feat2 = self.global_enc2(global_feat1)
        global_feat3 = self.global_enc3(global_feat2)
        
        # 将全局特征调整为与本地特征相同的大小
        global_feat3_resized = F.interpolate(
            global_feat3,
            size=local_feat3.size()[2:],
            mode="bilinear",
            align_corners=True
        )
        
        # 对高级全局特征应用注意力
        # global_attended = self.global_attention(global_feat3_resized, global_mask)
        
        # 融合本地和全局特征
        combined_features = torch.cat([local_feat3, global_feat3_resized], dim=1)
        fused_features = self.fusion(combined_features)
        
        # 对融合特征应用注意力
        # resized_local_mask = F.interpolate(local_mask, size=fused_features.shape[2:], mode="bilinear", align_corners=True)
        #fused_attended = self.fusion_attention(fused_features, resized_local_mask)
        
        # 解码器第一层
        #dec1 = self.decoder1(fused_features)
        dec1 = self.decoder1(fused_features)
        
        # 确保dec1的空间尺寸与local_feat2匹配
        dec1 = F.interpolate(dec1, size=local_feat2.size()[2:], mode="bilinear", align_corners=True)
        
        # 将解码器第一层输出与编码器第二层特征连接（跳跃连接）
        skip1 = torch.cat([dec1, local_feat2], dim=1)
        skip1_fused = self.skip_fusion1(skip1)
        
        # 解码器第二层
        dec2 = self.decoder2(skip1_fused)
        
        # 确保dec2的空间尺寸与local_feat1匹配
        dec2 = F.interpolate(dec2, size=local_feat1.size()[2:], mode="bilinear", align_corners=True)
        
        # 将解码器第二层输出与编码器第一层特征连接（跳跃连接）
        skip2 = torch.cat([dec2, local_feat1], dim=1)
        skip2_fused = self.skip_fusion2(skip2)
        
        # 最终输出层
        output = self.final_conv(skip2_fused)
        
        # 调整输出大小并应用缩放
        output = F.interpolate(output, size=input_size, mode="bilinear", align_corners=True)
        output_unstd = unstandardize_output(output, global_mean, global_std)
        
        # 仅在掩码区域应用生成结果（修正的逻辑）
        # local_mask为1的区域保留原始输入，为0的区域应用生成结果
        final_output = local_input * local_mask + (output_unstd) * (1 - local_mask)
        
        return final_output

'''class CompletionNetwork(nn.Module):
    def __init__(self, input_channels=1):
        super(CompletionNetwork, self).__init__()
        
        # 为全局输入添加早期注意力模块
        self.early_global_attention = EarlyAttentionModule(input_channels)

        # Local encoder branch (processes 33x33 patches)
        self.local_encoder = nn.Sequential(
            nn.Conv2d(input_channels * 2, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        

        # Global encoder branch (processes downsampled global data)
        self.global_encoder = nn.Sequential(
            nn.Conv2d(input_channels * 2, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        # Decoder (to generate completed local patch)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, input_channels, kernel_size=3, stride=1, padding=1),
            # nn.Tanh(),
        )

    def forward(self, local_input, local_mask, global_input, global_mask):
        # Concatenate input and mask for both local and global streams
        input_size = local_input.size()[2:]
        local_x = torch.cat([-local_input, local_mask], dim=1)  # 局部输入为负

        local_features = self.local_encoder(local_x)
        
        # 对全局输入应用早期注意力
        global_input_attended = self.early_global_attention(global_input, global_mask)

        
        # Process global input (will need to be adjusted in shape)
        global_x = torch.cat([-global_input_attended, global_mask], dim=1)  # 全局输入为负
        global_features = self.global_encoder(global_x)

        # Resize global features to match local features size
        global_features = F.interpolate(
            global_features,
            size=local_features.size()[2:],
            mode="bilinear",
            align_corners=False,
        )

        # Combine features
        combined_features = torch.cat([local_features, global_features], dim=1)
        fused_features = self.fusion(combined_features)
        

        # Decode to generate completed output
        output = self.decoder(fused_features)
        output = F.interpolate(
            output, size=input_size, mode="bilinear", align_corners=False
        )
        # print(f"meano: {torch.mean(-output*30)}")
        final_output = local_input * local_mask + (-output) * (1 - local_mask)
        return final_output  # 输出为负'''


"""class LocalDiscriminator(nn.Module):
    def __init__(self, input_channels=1):
        super(LocalDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),
        )

    def forward(self, x):
        return self.model(x)"""


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
        self.conv1 = nn.Conv2d(self.img_c * 2, 64, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        # 尺寸: (None, 64, img_h//2, img_w//2)

        # 第二层卷积
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        # 尺寸: (None, 128, img_h//4, img_w//4)

        # 第三层卷积
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(256)
        self.act3 = nn.LeakyReLU(0.2, inplace=True)
        # 尺寸: (None, 256, img_h//8, img_w//8)

        # 第四层卷积
        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2)
        self.bn4 = nn.BatchNorm2d(512)
        self.act4 = nn.LeakyReLU(0.2, inplace=True)
        # 尺寸: (None, 512, img_h//16, img_w//16)

        # 第五层卷积（与原始代码相同）
        self.conv5 = nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=2)
        self.bn5 = nn.BatchNorm2d(512)
        self.act5 = nn.LeakyReLU(0.2, inplace=True)
        # 尺寸: (None, 512, img_h//32, img_w//32)

        # 计算特征数量（适应不同输入尺寸）
        # 对于33x33的输入，最后特征图大小为 (512, 2, 2)
        features_h = max(1, self.img_h // 32)
        features_w = max(1, self.img_w // 32)
        in_features = 512 * features_h * features_w

        # 全连接层
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(self.final_features, 1024)
        self.output_act = nn.LeakyReLU(0.2, inplace=True)

        # 最后输出层，用于分类
        # self.classifier = nn.Linear(1024, 1)

    def forward(self, local, mask):
        # 卷积层
        x = torch.cat([-local, mask], dim=1)
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

        # 第一层卷积
        self.conv1 = nn.Conv2d(self.img_c * 2, 64, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        # 输出尺寸: (batch_size, 64, 300, 300)

        # 第二层卷积
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        # 输出尺寸: (batch_size, 128, 150, 150)

        # 第三层卷积
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(256)
        self.act3 = nn.LeakyReLU(0.2, inplace=True)
        # 输出尺寸: (batch_size, 256, 75, 75)

        # 第四层卷积
        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2)
        self.bn4 = nn.BatchNorm2d(512)
        self.act4 = nn.LeakyReLU(0.2, inplace=True)
        # 输出尺寸: (batch_size, 512, 38, 38)

        # 第五层卷积
        self.conv5 = nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=2)
        self.bn5 = nn.BatchNorm2d(512)
        self.act5 = nn.LeakyReLU(0.2, inplace=True)
        # 输出尺寸: (batch_size, 512, 19, 19)

        # 第六层卷积 - 添加额外卷积层以进一步降低尺寸
        self.conv6 = nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=2)
        self.bn6 = nn.BatchNorm2d(512)
        self.act6 = nn.LeakyReLU(0.2, inplace=True)
        # 输出尺寸: (batch_size, 512, 10, 10)

        # 第七层卷积 - 继续降低尺寸
        self.conv7 = nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=2)
        self.bn7 = nn.BatchNorm2d(512)
        self.act7 = nn.LeakyReLU(0.2, inplace=True)
        # 输出尺寸: (batch_size, 512, 5, 5)

        # 扁平化
        self.flatten = nn.Flatten()
        # 输出尺寸: (batch_size, 512*5*5)

        # 全连接层
        self.linear = nn.Linear(512 * 5 * 5, 1024)
        self.final_act = nn.LeakyReLU(0.2, inplace=True)
        # 输出尺寸: (batch_size, 1024)

    def forward(self, globalin, mask):
        # 卷积层
        x = torch.cat([-globalin, mask], dim=1)
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.act4(self.bn4(self.conv4(x)))
        x = self.act5(self.bn5(self.conv5(x)))
        x = self.act6(self.bn6(self.conv6(x)))
        x = self.act7(self.bn7(self.conv7(x)))

        # 扁平化
        x = self.flatten(x)

        # 全连接层
        x = self.final_act(self.linear(x))

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
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid(),
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

        return out


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
    test = 0
    # Remove the "logs" folder if it exists
    logs_path = "logs"
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
            writer = SummaryWriter("logs")
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
    webbrowser.open("http://localhost:6006/?darkMode=true#graphs&run=.")
