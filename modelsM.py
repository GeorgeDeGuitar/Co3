import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Concatenate(nn.Module):
    def __init__(self, dim=-1):
        super(Concatenate, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat(x, dim=self.dim)


class CompletionNetwork(nn.Module):
    """DEM数据补全网络 - 单通道输入输出"""

    def __init__(self):
        super(CompletionNetwork, self).__init__()
        # 输入: (None, 2, 600, 600) - DEM + mask
        self.conv1 = nn.Conv2d(2, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU()

        # 下采样路径
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # 300x300
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # 150x150
        self.bn4 = nn.BatchNorm2d(256)
        self.act4 = nn.ReLU()

        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.act5 = nn.ReLU()

        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.act6 = nn.ReLU()

        # 空洞卷积增加感受野 - 适合地形数据的大范围依赖
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=2, padding=2)
        self.bn7 = nn.BatchNorm2d(256)
        self.act7 = nn.ReLU()

        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=4, padding=4)
        self.bn8 = nn.BatchNorm2d(256)
        self.act8 = nn.ReLU()

        self.conv9 = nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=8, padding=8)
        self.bn9 = nn.BatchNorm2d(256)
        self.act9 = nn.ReLU()

        self.conv10 = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, dilation=16, padding=16
        )
        self.bn10 = nn.BatchNorm2d(256)
        self.act10 = nn.ReLU()

        # 继续卷积
        self.conv11 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn11 = nn.BatchNorm2d(256)
        self.act11 = nn.ReLU()

        self.conv12 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn12 = nn.BatchNorm2d(256)
        self.act12 = nn.ReLU()

        # 上采样路径
        self.deconv13 = nn.ConvTranspose2d(
            256, 128, kernel_size=4, stride=2, padding=1
        )  # 300x300
        self.bn13 = nn.BatchNorm2d(128)
        self.act13 = nn.ReLU()

        self.conv14 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn14 = nn.BatchNorm2d(128)
        self.act14 = nn.ReLU()

        self.deconv15 = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1
        )  # 600x600
        self.bn15 = nn.BatchNorm2d(64)
        self.act15 = nn.ReLU()

        self.conv16 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn16 = nn.BatchNorm2d(32)
        self.act16 = nn.ReLU()

        # 输出单通道DEM
        self.conv17 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        # 不使用Sigmoid，因为DEM值可能为负

    def forward(self, x):
        x = self.bn1(self.act1(self.conv1(x)))
        x = self.bn2(self.act2(self.conv2(x)))
        x = self.bn3(self.act3(self.conv3(x)))
        x = self.bn4(self.act4(self.conv4(x)))
        x = self.bn5(self.act5(self.conv5(x)))
        x = self.bn6(self.act6(self.conv6(x)))
        x = self.bn7(self.act7(self.conv7(x)))
        x = self.bn8(self.act8(self.conv8(x)))
        x = self.bn9(self.act9(self.conv9(x)))
        x = self.bn10(self.act10(self.conv10(x)))
        x = self.bn11(self.act11(self.conv11(x)))
        x = self.bn12(self.act12(self.conv12(x)))
        x = self.bn13(self.act13(self.deconv13(x)))
        x = self.bn14(self.act14(self.conv14(x)))
        x = self.bn15(self.act15(self.deconv15(x)))
        x = self.bn16(self.act16(self.conv16(x)))
        x = self.conv17(x)
        return x


class LocalDiscriminator(nn.Module):
    """局部判别器 - 适配33x33单通道输入"""

    def __init__(self, input_shape=(1, 33, 33)):
        super(LocalDiscriminator, self).__init__()
        self.input_shape = input_shape
        self.output_shape = (1024,)
        self.img_c = input_shape[0]  # 1
        self.img_h = input_shape[1]  # 33
        self.img_w = input_shape[2]  # 33

        # 33x33 -> 16x16 (stride=2, kernel=5, padding=2)
        self.conv1 = nn.Conv2d(self.img_c, 64, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU()

        # 16x16 -> 8x8
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.ReLU()

        # 8x8 -> 4x4
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(256)
        self.act3 = nn.ReLU()

        # 4x4 -> 2x2
        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2)
        self.bn4 = nn.BatchNorm2d(512)
        self.act4 = nn.ReLU()

        # 2x2 -> 1x1
        self.conv5 = nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=2)
        self.bn5 = nn.BatchNorm2d(512)
        self.act5 = nn.ReLU()

        # 计算特征维度
        # 经过5次stride=2的卷积，33 -> 16 -> 8 -> 4 -> 2 -> 1
        in_features = 512 * 2 * 2
        self.flatten6 = Flatten()
        self.linear6 = nn.Linear(in_features, 1024)
        self.act6 = nn.ReLU()

    def forward(self, x):
        x = self.bn1(self.act1(self.conv1(x)))
        x = self.bn2(self.act2(self.conv2(x)))
        x = self.bn3(self.act3(self.conv3(x)))
        x = self.bn4(self.act4(self.conv4(x)))
        x = self.bn5(self.act5(self.conv5(x)))
        x = self.act6(self.linear6(self.flatten6(x)))
        return x


class GlobalDiscriminator(nn.Module):
    """全局判别器 - 适配600x600单通道输入"""

    def __init__(self, input_shape=(1, 600, 600)):
        super(GlobalDiscriminator, self).__init__()
        self.input_shape = input_shape
        self.output_shape = (1024,)
        self.img_c = input_shape[0]  # 1
        self.img_h = input_shape[1]  # 600
        self.img_w = input_shape[2]  # 600

        # 600x600 -> 300x300
        self.conv1 = nn.Conv2d(self.img_c, 64, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU()

        # 300x300 -> 150x150
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.ReLU()

        # 150x150 -> 75x75
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(256)
        self.act3 = nn.ReLU()

        # 75x75 -> 38x38 (向上取整)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2)
        self.bn4 = nn.BatchNorm2d(512)
        self.act4 = nn.ReLU()

        # 38x38 -> 19x19
        self.conv5 = nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=2)
        self.bn5 = nn.BatchNorm2d(512)
        self.act5 = nn.ReLU()

        # 19x19 -> 10x10 (向上取整)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=2)
        self.bn6 = nn.BatchNorm2d(512)
        self.act6 = nn.ReLU()

        # 计算最终特征图大小
        # 600 -> 300 -> 150 -> 75 -> 38 -> 19 -> 10
        final_h = 10
        final_w = 10
        in_features = 512 * final_h * final_w

        self.flatten7 = Flatten()
        self.linear7 = nn.Linear(in_features, 1024)
        self.act7 = nn.ReLU()

    def forward(self, x):
        x = self.bn1(self.act1(self.conv1(x)))
        x = self.bn2(self.act2(self.conv2(x)))
        x = self.bn3(self.act3(self.conv3(x)))
        x = self.bn4(self.act4(self.conv4(x)))
        x = self.bn5(self.act5(self.conv5(x)))
        x = self.bn6(self.act6(self.conv6(x)))
        x = self.act7(self.linear7(self.flatten7(x)))
        return x


class ContextDiscriminator(nn.Module):
    """上下文判别器 - 结合局部和全局信息"""

    def __init__(self, local_input_shape=(1, 33, 33), global_input_shape=(1, 600, 600)):
        super(ContextDiscriminator, self).__init__()
        self.input_shape = [local_input_shape, global_input_shape]
        self.output_shape = (1,)

        self.model_ld = LocalDiscriminator(local_input_shape)
        self.model_gd = GlobalDiscriminator(global_input_shape)

        # 融合局部和全局特征
        in_features = (
            self.model_ld.output_shape[-1] + self.model_gd.output_shape[-1]
        )  # 2048
        self.concat1 = Concatenate(dim=-1)
        self.linear1 = nn.Linear(in_features, 1)
        self.act1 = nn.Sigmoid()

    def forward(self, x):
        x_ld, x_gd = x
        x_ld = self.model_ld(x_ld)
        x_gd = self.model_gd(x_gd)
        out = self.act1(self.linear1(self.concat1([x_ld, x_gd])))
        return out


if __name__ == "__main__":
    # 测试模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 测试CompletionNetwork
    print("Testing CompletionNetwork...")
    cn = CompletionNetwork().to(device)
    test_input = torch.randn(2, 2, 600, 600).to(device)  # DEM + mask
    output = cn(test_input)
    print(f"CN Input shape: {test_input.shape}")
    print(f"CN Output shape: {output.shape}")

    # 测试ContextDiscriminator
    print("\nTesting ContextDiscriminator...")
    cd = ContextDiscriminator().to(device)
    test_local = torch.randn(2, 1, 33, 33).to(device)
    test_global = torch.randn(2, 1, 600, 600).to(device)
    output = cd((test_local, test_global))
    print(f"CD Local input shape: {test_local.shape}")
    print(f"CD Global input shape: {test_global.shape}")
    print(f"CD Output shape: {output.shape}")

    print("\nAll tests passed!")
