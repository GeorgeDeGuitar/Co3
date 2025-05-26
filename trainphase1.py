import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.nn.utils import spectral_norm
from torch.optim import Adadelta, Adam
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw
import visdom
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from loss import  completion_network_loss
import torch.nn.functional as F
# Import your models
from modelcn import CompletionNetwork, ContextDiscriminator

# Import your custom dataset
from DemDataset import DemDataset
# from DEMData import DEMData

# Import the custom_collate_fn function from your dataset module
from DemDataset import custom_collate_fn
# from DEMData import custom_collate_fn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau

import gc
import copy
import glob


# setx PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:98
def save_checkpoint(model_cn, model_cd, opt_cn, opt_cd, step, phase, result_dir, best_val_loss=None, best_acc=None):
    """
    保存训练检查点，包括模型权重、优化器状态和训练进度
    """
    checkpoint = {
        'model_cn_state_dict': model_cn.state_dict(),
        'model_cd_state_dict': model_cd.state_dict() if model_cd is not None else None,
        'opt_cn_state_dict': opt_cn.state_dict(),
        'opt_cd_state_dict': opt_cd.state_dict() if opt_cd is not None else None,
        'step': step,
        'phase': phase,
        'best_val_loss': best_val_loss,
        'best_acc': best_acc
    }
    # 删除旧的模型文件
    old_files = glob.glob(os.path.join(result_dir, f"checkpoint_phase{phase}_step*"))
    for file in old_files:
        try:
            os.remove(file)
            print(f"删除旧文件: {os.path.basename(file)}")
        except Exception as e:
            print(f"删除文件失败: {e}")
    checkpoint_path = os.path.join(result_dir, f"checkpoint_phase{phase}_step{step}.pth")
    torch.save(checkpoint, checkpoint_path)
    print(f"保存检查点到 {checkpoint_path}")
    
    '''# 保存一个最新的检查点文件，方便恢复
    latest_checkpoint_path = os.path.join(result_dir, f"latest_checkpoint.pth")
    torch.save(checkpoint, latest_checkpoint_path)
    print(f"保存最新检查点到 {latest_checkpoint_path}")'''
    
    return checkpoint_path

def load_checkpoint(checkpoint_path, model_cn, model_cd, opt_cn, opt_cd, device):
    """
    加载训练检查点，恢复模型权重、优化器状态和训练进度
    """
    print(f"从 {checkpoint_path} 加载检查点")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    model_cn.load_state_dict(checkpoint['model_cn_state_dict'])
    
    if model_cd is not None and checkpoint['model_cd_state_dict'] is not None:
        model_cd.load_state_dict(checkpoint['model_cd_state_dict'])
    
    opt_cn.load_state_dict(checkpoint['opt_cn_state_dict'])
    
    if opt_cd is not None and checkpoint['opt_cd_state_dict'] is not None:
        opt_cd.load_state_dict(checkpoint['opt_cd_state_dict'])
    
    step = checkpoint['step']
    phase = checkpoint['phase']
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    best_acc = checkpoint.get('best_acc', float('-inf'))
    
    print(f"成功加载检查点：阶段 {phase}，步骤 {step}")
    
    return step, phase, best_val_loss, best_acc

def prepare_batch_data(batch, device):
    """
    Prepare batch data by moving tensors to the specified device.

    Args:
        batch: A tuple containing batched data
        device: The device to move tensors to

    Returns:
        Tuple of tensors moved to the specified device
    """
    (
        local_inputs,
        local_masks,
        local_targets,
        global_inputs,
        global_masks,
        global_targets,
        metadata,
    ) = batch

    # Move tensors to device with highest precision
    local_inputs = local_inputs.to(device, dtype=torch.float32)
    local_masks = local_masks.to(device, dtype=torch.int)
    local_targets = local_targets.to(device, dtype=torch.float32)
    global_inputs = global_inputs.to(device, dtype=torch.float32)
    global_masks = global_masks.to(device, dtype=torch.int)
    global_targets = global_targets.to(device, dtype=torch.float32)

    return (
        local_inputs,
        local_masks,
        local_targets,
        global_inputs,
        global_masks,
        global_targets,
        metadata,
    )


def train(resume_from=None):
    # =================================================
    # Configuration settings (replace with your settings)
    # =================================================
    '''json_dir = r"F:\Dataset\Simulate\data0\json"  # 局部json
    array_dir = r"F:\Dataset\Simulate\data0\arraynmask\array"  # 全局array
    mask_dir = r"F:\Dataset\Simulate\data0\arraynmask\mask"  # 全局mask
    target_dir = (
        r"F:\Dataset\Simulate\data0\groundtruthstatus\statusarray"  # 全局target
    )
    result_dir = r"F:\Dataset\Simulate\data0\result"  # 结果保存路径'''

    json_dir = r"E:\KingCrimson Dataset\Simulate\data0\json"  # 局部json
    array_dir = r"E:\KingCrimson Dataset\Simulate\data0\arraynmask\array"  # 全局array
    mask_dir = r"E:\KingCrimson Dataset\Simulate\data0\arraynmask\mask"  # 全局mask
    target_dir = (
        r"E:\KingCrimson Dataset\Simulate\data0\groundtruthstatus\statusarray" # 全局target
    )
    result_dir = r"E:\KingCrimson Dataset\Simulate\data0\resultsphase2"  # 结果保存路径
    cache_dir = r"E:\KingCrimson Dataset\Simulate\data0\traindata"  # 缓存目录

    # Training parameters
    steps_1 = 100000  # 90000  # Phase 1 training steps
    steps_2 = 50000  # 10000  # Phase 2 training steps
    steps_3 = 200000  # 400000  # Phase 3 training steps

    snaperiod_1 =2000   # 10000  # How often to save snapshots in phase 1
    snaperiod_2 = 500 # 2000  # How often to save snapshots in phase 2
    snaperiod_3 = 500   # 10000  # How often to save snapshots in phase 3

    batch_size = 8  # Batch size
    alpha = 4e-1  # Alpha parameter for loss weighting
    validation_split = 0.1  # Percentage of data to use for validation
    
    # 清空 PyTorch 的 CUDA 缓存
    torch.cuda.empty_cache()

    # Hardware setup
    if not torch.cuda.is_available():
        raise Exception("At least one GPU must be available.")
    device = torch.device("cuda:1")
    #device = torch.device("cpu")

    # Initialize Visdom
    viz = visdom.Visdom(env="dem_completion_phase2")
    viz.close(env="dem_completion_phase2")

    # Create windows for the different phases and metrics
    loss_windows = {
        "phase1_train": viz.line(
            Y=torch.zeros(1), X=torch.zeros(1), opts=dict(title="Phase 1 Training Loss")
        ),
        "phase1_val": viz.line(
            Y=torch.zeros(1),
            X=torch.zeros(1),
            opts=dict(title="Phase 1 Validation Loss"),
        ),
        "phase2_disc": viz.line(
            Y=torch.zeros(1),
            X=torch.zeros(1),
            opts=dict(title="Phase 2 Discriminator Loss"),
        ),
        "phase2_acc": viz.line(
            Y=torch.zeros(1),
            X=torch.zeros(1),
            opts=dict(title="Phase 2 Discriminator Accuracy"),
        ),
        "phase3_gen": viz.line(
            Y=torch.zeros(1),
            X=torch.zeros(1),
            opts=dict(title="Phase 3 Generator Loss"),
        ),
        "phase3_disc": viz.line(
            Y=torch.zeros(1),
            X=torch.zeros(1),
            opts=dict(title="Phase 3 Discriminator Loss"),
        ),
        "phase3_val": viz.line(
            Y=torch.zeros((1, 5)),
            X=torch.zeros(1),
            opts=dict(title="Phase 3 Validation Loss",
                      legend=['Recon Loss', 'Adv Loss', 'Real Acc', 'Fake Acc', 'Disc Acc'],
                      linecolor=np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255]]),
                      ),
        ),
        "phase3_fake_acc": viz.line(
            Y=torch.zeros((1, 2)),
            X=torch.zeros(1),
            opts=dict(
                title="Phase 3 Discriminator Accuracy",
                legend=['Fake Acc', 'Real Acc'],
                linecolor=np.array([[255, 0, 0], [0, 0, 255]]),
            )
        ),
    }

    # =================================================
    # Preparation
    # =================================================
    # Create result directories
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    for phase in ["phase_1", "phase_2", "phase_3", "validation"]:
        if not os.path.exists(os.path.join(result_dir, phase)):
            os.makedirs(os.path.join(result_dir, phase))

    gc.collect()
    torch.cuda.empty_cache()  # 如果使用GPU
    # Load dataset
    dataset = DemDataset(json_dir, array_dir, mask_dir, target_dir)
    # dataset = DEMData(cache_dir)
    # 过滤掉 __getitem__ 返回 None 的数据
    # full_dataset = [data for data in dataset if data is not None]
    full_dataset = dataset  # 直接使用完整数据集


    # Split dataset into train and validation
    dataset_size = len(full_dataset)
    val_size = int(validation_split * dataset_size)
    train_size = dataset_size - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),  # For reproducibility
    )

    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn,num_workers=8,pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn,num_workers=8,pin_memory=True
    )
    val_subset = torch.utils.data.Subset(val_loader.dataset, range(0, 30))
    val_subset_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn,num_workers=8,pin_memory=True)

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    alpha = torch.tensor(alpha, dtype=torch.float32).to(device)

    # =================================================
    # Setup Models
    # =================================================
    # Initialize the Completion Network - adjust input channels as needed
    model_cn = CompletionNetwork(input_channels=1).to(
        device
    )  # 1 for elevation + 1 for mask

    # Initialize the Context Discriminator
    model_cd = ContextDiscriminator(
        local_input_channels=1,  # For elevation data
        local_input_size=33,  # The local patch size from your dataset
        global_input_channels=1,  # For elevation data
        global_input_size=600,  # The global image size from your dataset
    ).to(device)

    # Setup optimizers
    opt_cn = Adadelta(model_cn.parameters())
    opt_cd = Adadelta(model_cd.parameters())
    
    # opt_cn = Adam(model_cn.parameters(), lr=1e-4)
    # opt_cd = Adam(model_cd.parameters(), lr=1e-4)
    
     # 初始化训练状态变量
    phase = 1  # 从第一阶段开始
    step_phase1 = 0
    step_phase2 = 0
    step_phase3 = 0
    best_val_loss = float('inf')
    best_acc = float('-inf')
    best_val_loss_joint = float('inf')
    
    # 如果指定了检查点文件，则加载它恢复训练
    if resume_from:
        step, phase, best_val_loss, best_acc = load_checkpoint(resume_from, model_cn, model_cd, opt_cn, opt_cd, device)
        
        if phase == 1:
            step_phase1 = step
            best_val_loss = best_val_loss
        elif phase == 2:
            step_phase1 = steps_1  # 第一阶段已完成
            step_phase2 = step
            best_acc = best_acc
        elif phase == 3:
            step_phase1 = steps_1  # 第一阶段已完成
            step_phase2 = steps_2  # 第二阶段已完成
            step_phase3 = step
            best_val_loss_joint = best_val_loss  # 在第三阶段使用
    else:
        if phase==2 or phase==3:
            # Load pre-trained weights if available
            pretrained_weights_path = r"E:\KingCrimson Dataset\Simulate\data0\results\phase_1\model_cn_best"
            if os.path.exists(pretrained_weights_path):
                model_cn.load_state_dict(torch.load(pretrained_weights_path, map_location=device))
                print(f"Phase2: Loaded pre-trained weights from {pretrained_weights_path}")
        if phase==3:
            # Load pre-trained weights if available
            pretrained_weights_path_cn= r"E:\KingCrimson Dataset\Simulate\data0\results\phase_1\model_cn_best"
            if os.path.exists(pretrained_weights_path_cn):
                model_cn.load_state_dict(torch.load(pretrained_weights_path_cn, map_location=device))
                print(f"Phase2: Loaded pre-trained weights from {pretrained_weights_path_cn}")
                
            pretrained_weights_path_cd = r"E:\KingCrimson Dataset\Simulate\data0\results\phase_2\model_cd_best0"
            if os.path.exists(pretrained_weights_path_cd):
                model_cd.load_state_dict(torch.load(pretrained_weights_path_cd, map_location=device))
                print(f"Phase3: Loaded pre-trained weights from {pretrained_weights_path_cd}")

    # Loss functions
    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()

    def visualize_results(
        local_targets,
        local_inputs,
        outputs,
        completed=None,
        step=0,
        phase="phase_1",
        num_test_completions=4,
    ):
        """
        在Visdom中可视化输入、目标、输出和完成的图像，使用高程图着色方案。

        Args:
            local_targets: 目标小块的真实值
            local_inputs: 输入小块
            outputs: 模型输出小块
            completed: 合并的全局输出 (可选)
            step: 当前训练步骤
            phase: 当前训练阶段
            num_test_completions: 要可视化的测试完成图像数量
        """

        # 定义高程图着色函数

        def apply_colormap(tensor):
            """将张量转换为带有高程图着色的PIL图像"""
            # 确保输入是2D张量(单通道图像)或取第一个通道
            if tensor.dim() > 2:
                if tensor.size(0) == 1:
                    tensor = tensor.squeeze(0)  # 单通道图像
                else:
                    # 如果是多通道，计算平均值或取第一个通道
                    tensor = tensor[0]  # 取第一个通道

            # 转换为numpy数组并归一化到0-1范围
            np_array = tensor.detach().numpy()#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            min_val = np_array.min()
            max_val = np_array.max()
            norm_array = (np_array - min_val) / (max_val - min_val + 1e-8)

            # 应用matplotlib的jet色彩映射
            colored_array = plt.cm.jet(norm_array)

            # 转换为PIL图像 (colored_array是RGBA格式)
            colored_array_rgb = (colored_array[:, :, :3] * 255).astype(np.uint8)
            colored_image = Image.fromarray(colored_array_rgb)

            return colored_image

        # 确保不超过可用的样本数量
        num_samples = min(num_test_completions, len(local_targets))

        # 处理并着色小块图像
        colored_patches = []
        labels = ["目标", "输入", "输出"]
        patch_tensors = [
            local_targets[:num_samples],
            local_inputs[:num_samples],
            outputs[:num_samples],
        ]

        for i, tensor_batch in enumerate(patch_tensors):
            for j in range(len(tensor_batch)):
                # 将每个张量转换为带有高程图着色的PIL图像
                colored_img = apply_colormap(tensor_batch[j].cpu())
                """# 添加标签
                if j == 0:
                    width, height = colored_img.size
                    label_img = Image.new("RGB", (width, 20), color=(255, 255, 255))
                    draw = ImageDraw.Draw(label_img)
                    # draw.text((5, 5), labels[i], fill=(0, 0, 0))
                    combined = Image.new("RGB", (width, height + 20))
                    combined.paste(label_img, (0, 0))
                    combined.paste(colored_img, (0, 20))
                    colored_patches.append(combined)
                else:"""
                colored_patches.append(colored_img)

        # 创建网格布局
        cols = num_samples
        rows = 3  # 目标、输入、输出
        grid_width = (colored_patches[0].width + 5) * cols
        grid_height = (colored_patches[0].height + 5) * rows
        grid_img = Image.new("RGB", (grid_width, grid_height))

        # 填充网格
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                if idx < len(colored_patches):
                    x = j * (colored_patches[idx].width + 5)
                    y = i * (colored_patches[idx].height + 5)
                    grid_img.paste(colored_patches[idx], (x, y))

        # 放大图像3倍
        grid_img = grid_img.resize(
            (grid_img.width * 3, grid_img.height * 3), Image.NEAREST
        )

        viz.image(
            np.array(grid_img).transpose(2, 0, 1),
            opts=dict(
            caption=f"{phase} 小块图像 步骤 {step}",
            title=f"{phase} 小块可视化 (高程图着色)",
            ),
            win=f"{phase}_patches",
        )

        # 如果有completed图像，单独处理并可视化它们
        if completed is not None:
            completed_samples = min(num_samples, len(completed))
            colored_completed = []

            for i in range(completed_samples):
                colored_img = apply_colormap(completed[i].cpu())
                colored_completed.append(colored_img)

            # 创建completed图像网格
            cols = min(2, completed_samples)
            rows = (completed_samples + cols - 1) // cols
            completed_grid_width = (colored_completed[0].width + 5) * cols
            completed_grid_height = (colored_completed[0].height + 5) * rows
            completed_grid_img = Image.new(
                "RGB", (completed_grid_width, completed_grid_height)
            )

            # 填充网格
            for i in range(completed_samples):
                x = (i % cols) * (colored_completed[i].width + 5)
                y = (i // cols) * (colored_completed[i].height + 5)
                completed_grid_img.paste(colored_completed[i], (x, y))
            
            # 缩小图像为原来的一半
            completed_grid_img = completed_grid_img.resize(
                (completed_grid_img.width // 2, completed_grid_img.height // 2), Image.Resampling.LANCZOS
            )

            # 在Visdom中显示completed图像网格
            viz.image(
                np.array(completed_grid_img).transpose(2, 0, 1),  # 转换为CHW格式
                opts=dict(
                    caption=f"{phase} 完成图像 步骤 {step}",
                    title=f"{phase} 完成可视化 (高程图着色)",
                ),
                win=f"{phase}_completed",
            )

        """ # 添加颜色条来显示高程图对应关系
        colorbar_width = 30
        colorbar_height = 200
        colorbar = Image.new("RGB", (colorbar_width, colorbar_height))

        for i in range(colorbar_height):
            # 从下到上渐变
            value = (colorbar_height - i - 1) / (colorbar_height - 1)
            # 修复这里：将颜色转换为numpy数组后再使用astype
            color_rgba = plt.cm.jet(value)  # 这是一个RGBA格式的颜色
            color_rgb = np.array(color_rgba[:3]) * 255  # 取前3个通道(RGB)并缩放到0-255
            color = tuple(color_rgb.astype(np.uint8))  # 转换为元组

            for j in range(colorbar_width):
                colorbar.putpixel((j, i), color)

        # 添加最大值和最小值标签
        colorbar_with_labels = Image.new(
            "RGB", (colorbar_width + 50, colorbar_height), color=(255, 255, 255)
        )
        colorbar_with_labels.paste(colorbar, (0, 0))

        draw = ImageDraw.Draw(colorbar_with_labels)
        # draw.text((colorbar_width + 5, 5), "最大值", fill=(0, 0, 0))
        # draw.text((colorbar_width + 5, colorbar_height - 15), "最小值", fill=(0, 0, 0))

        # 在Visdom中显示颜色条
        viz.image(
            np.array(colorbar_with_labels).transpose(2, 0, 1),
            opts=dict(caption="高程图颜色对应关系", title="高程图颜色图例"),
            win=f"{phase}_colorbar",
        )"""

    # Helper function to evaluate model on validation set
    def validate(model_cn, val_subset_loader, device):
        model_cn.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_subset_loader:
                # Prepare batch data
                batch_data = prepare_batch_data(batch, device)
                (
                    batch_local_inputs,
                    batch_local_masks,
                    batch_local_targets,
                    batch_global_inputs,
                    batch_global_masks,
                    batch_global_targets,
                    metadata,
                ) = batch_data

                # Forward pass
                outputs = model_cn(
                    batch_local_inputs,
                    batch_local_masks,
                    batch_global_inputs,
                    batch_global_masks,
                )

                completed, completed_mask, pos = merge_local_to_global(
                    outputs, batch_global_inputs, batch_global_masks, metadata
                )

                # Compute loss
                loss = completion_network_loss(
                    outputs,
                    batch_local_targets,   
                    batch_local_masks,
                    batch_global_targets,
                    completed,
                    completed_mask,
                    pos,
                )
                val_loss += loss.item()

        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_subset_loader)
        model_cn.train()

        return avg_val_loss
    
    def validateAll(model_cn, model_cd, val_subset_loader, device):
        """全面验证生成器和判别器性能"""
        model_cn.eval()
        model_cd.eval()
        
        # 初始化各种损失和指标
        recon_loss_sum = 0.0
        adv_loss_sum = 0.0
        real_acc_sum = 0.0
        fake_acc_sum = 0.0
        total_batches = 0
        
        with torch.no_grad():
            for batch in val_subset_loader:
                total_batches += 1
                
                # 准备批次数据
                batch_data = prepare_batch_data(batch, device)
                (
                    batch_local_inputs,
                    batch_local_masks,
                    batch_local_targets,
                    batch_global_inputs,
                    batch_global_masks,
                    batch_global_targets,
                    metadata,
                ) = batch_data
                
                # 1. 评估生成器重建性能
                outputs = model_cn(
                    batch_local_inputs,
                    batch_local_masks,
                    batch_global_inputs,
                    batch_global_masks,
                )
                
                # 计算全局完成结果
                completed, completed_mask, pos = merge_local_to_global(
                    outputs, batch_global_inputs, batch_global_masks, metadata
                )
                
                # 计算重建损失
                recon_loss = completion_network_loss(
                    outputs,
                    batch_local_targets,
                    batch_local_masks,
                    batch_global_targets,
                    completed,
                    completed_mask,
                    pos,
                )
                recon_loss_sum += recon_loss.item()
                
                # 2. 评估对抗性性能
                # 创建标签
                batch_size = batch_local_targets.size(0)
                real_labels = torch.ones(batch_size, 1, device=device)
                fake_labels = torch.zeros(batch_size, 1, device=device)
                
                # 将生成器输出传递给判别器
                fake_global_embedded, fake_global_mask_embedded, _ = merge_local_to_global(
                    outputs, batch_global_inputs, batch_global_masks, metadata
                )
                
                fake_predictions, _, _ = model_cd(
                    outputs,
                    batch_local_masks,
                    fake_global_embedded,
                    fake_global_mask_embedded,
                )
                
                # 将真实样本传递给判别器
                real_global_embedded, real_global_mask_embedded, _ = merge_local_to_global(
                    batch_local_targets, batch_global_inputs, batch_global_masks, metadata
                )
                
                real_predictions, _, _ = model_cd(
                    batch_local_targets,
                    batch_local_masks,
                    real_global_embedded,
                    real_global_mask_embedded,
                )
                
                # 计算对抗性损失 - 生成器视角（希望判别器将生成样本识别为真）
                adv_loss = F.binary_cross_entropy_with_logits(fake_predictions, real_labels)
                adv_loss_sum += adv_loss.item()
                
                # 计算判别器准确率
                real_acc = (torch.sigmoid(real_predictions) >= 0.5).float().mean().item()
                fake_acc = (torch.sigmoid(fake_predictions) < 0.5).float().mean().item()
                
                real_acc_sum += real_acc
                fake_acc_sum += fake_acc
        
        # 计算平均值
        avg_recon_loss = recon_loss_sum / total_batches
        avg_adv_loss = adv_loss_sum / total_batches
        avg_real_acc = real_acc_sum / total_batches
        avg_fake_acc = fake_acc_sum / total_batches
        avg_disc_acc = (avg_real_acc + avg_fake_acc) / 2
        
        # 将模型设回训练模式
        model_cn.train()
        model_cd.train()
        
        return {
            'recon_loss': avg_recon_loss,
            'adv_loss': avg_adv_loss,
            'real_acc': avg_real_acc,
            'fake_acc': avg_fake_acc,
            'disc_acc': avg_disc_acc
        }

    '''def completion_network_loss(input, output, mask):
        """Calculate the completion network loss."""
        return mse_loss(output * mask, input * mask)'''

    def merge_local_to_global(
        local_data, global_data, global_mask, metadata, kernel_size=33
    ):
        """
        Merge local patch data into global data based on metadata.

        Args:
            local_data: The local patch data
            global_data: The global data
            global_mask: The global mask
            metadata: Metadata containing position information
            kernel_size: The size of the local patch

        Returns:
            Tuple of (merged_data, merged_mask)
        """
        merged_data = global_data.clone()
        merged_mask = global_mask.clone()

        pos=[]
        for i in range(len(metadata)):
            centerx = metadata[i][0]
            centery = metadata[i][1]
            xbegain = metadata[i][2]
            ybegain = metadata[i][3]

            # Calculate the top-left corner of the kernel_size x kernel_size window
            minx = int(centerx - xbegain) - kernel_size // 2
            maxx = int(centerx - xbegain) + kernel_size // 2 + 1
            miny = int(centery - ybegain) - kernel_size // 2
            maxy = int(centery - ybegain) + kernel_size // 2 + 1

            # Embed the local data into the global data
            merged_data[i, :, minx:maxx, miny:maxy] = local_data[i]
            merged_mask[i, :, minx:maxx, miny:maxy] = 1
            pos.append([minx, maxx, miny, maxy])

        return merged_data, merged_mask, pos

    # =================================================
    # Training Phase 1: Train Completion Network only
    # =================================================
    '''print("Starting Phase 1 training...")
    model_cn.train()
    pbar = tqdm(total=steps_1)'''
    # best_val_loss = float("inf")
    
    if step_phase1 < steps_1 and phase == 1:
        print("开始/继续第1阶段训练...")
        model_cn.train()
        pbar = tqdm(total=steps_1, initial=step_phase1)
        step = step_phase1

    # step = 0
        while step < steps_1:
            for batch in train_loader:
                # Prepare batch data
                batch_data = prepare_batch_data(batch, device)
                (
                    batch_local_inputs,
                    batch_local_masks,
                    batch_local_targets,
                    batch_global_inputs,
                    batch_global_masks,
                    batch_global_targets,
                    metadata,
                ) = batch_data

                # Forward pass
                outputs = model_cn(
                    batch_local_inputs,
                    batch_local_masks,
                    batch_global_inputs,
                    batch_global_masks,
                )

                completed, completed_mask, pos = merge_local_to_global(
                    outputs, batch_global_inputs, batch_global_masks, metadata
                )

                # Calculate loss
                loss = completion_network_loss(
                    outputs,
                    batch_local_targets,             
                    batch_local_masks,
                    batch_global_targets,
                    completed,
                    completed_mask,
                    pos
                )

                # Backward and optimize
                loss.backward()
                opt_cn.step()
                opt_cn.zero_grad()

                # Update Visdom plot for training loss
                viz.line(
                    Y=torch.tensor([loss.item()]) if torch.tensor([loss.item()]) <1000 else torch.tensor([-1]),
                    X=torch.tensor([step]),
                    win=loss_windows["phase1_train"],
                    update="append",
                )
                # 在每个snaperiod_1步保存检查点
                if step % snaperiod_1 == 0:
                    # 运行验证和可视化代码...
                    
                    # 保存检查点
                    save_checkpoint(model_cn, None, opt_cn, opt_cd, step, 1, result_dir, best_val_loss=best_val_loss)

                step += 1
                pbar.set_description(f"Phase 1 | train loss: {loss.item():.5f}")
                pbar.update(1)

                # Testing and saving snapshots
                if step % snaperiod_1 == 0:
                    # Run validation
                    # Use a subset of val_loader for validation
                    
                    val_loss = validate(model_cn, val_subset_loader, device)
                    print(f"Step {step}, Validation Loss: {val_loss:.5f}")

                    # Update Visdom plot for validation loss
                    viz.line(
                        Y=torch.tensor([val_loss]),
                        X=torch.tensor([step]),
                        win=loss_windows["phase1_val"],
                        update="append",
                    )

                    # Save if best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_path = os.path.join(
                            result_dir, "phase_1", "model_cn_best"
                        )
                        torch.save(model_cn.state_dict(), best_model_path)

                    # Generate some test completions
                    model_cn.eval()
                    with torch.no_grad():
                        # Get a batch from validation set
                        val_batch = next(iter(val_subset_loader))

                        # Prepare validation batch
                        val_data = prepare_batch_data(val_batch, device)
                        (
                            val_local_inputs,
                            val_local_masks,
                            val_local_targets,
                            val_global_inputs,
                            val_global_masks,
                            val_global_targets,
                            metadata,
                        ) = val_data

                        # Generate completions
                        test_output = model_cn(
                            val_local_inputs,
                            val_local_masks,
                            val_global_inputs,
                            val_global_masks,
                        )

                        # Merge local output to global for visualization
                        completed, completed_mask, _ = merge_local_to_global(
                            test_output, val_global_inputs, val_global_masks, metadata
                        )

                        # Visualize results in Visdom
                        visualize_results(
                            val_local_targets,
                            val_local_inputs,
                            test_output,
                            completed,
                            step,
                            "phase_1",
                        )

                        # Save model
                        model_path = os.path.join(
                            result_dir, "phase_1", f"model_cn_step{step}"
                        )
                        torch.save(model_cn.state_dict(), model_path)

                    model_cn.train()

                if step >= steps_1:
                    break

        pbar.close()

    # =================================================
    # Training Phase 2: Train Context Discriminator only
    # =================================================
    
    
    def balanced_discriminator_loss(real_preds, fake_preds, real_labels, fake_labels):
        """
        平衡的判别器损失，特别处理不平衡情况
        """
        # 基础MSE损失
        real_loss = F.mse_loss(real_preds, real_labels)
        fake_loss = F.mse_loss(fake_preds, fake_labels)
        
        # 计算真假分类情况
        real_accuracy = (real_preds > 0.5).float().mean()
        fake_accuracy = (fake_preds < 0.5).float().mean()
        
        # 如果分类严重不平衡
        if fake_accuracy > 0.8 and real_accuracy < 0.2:
            # 大幅增加真实样本权重，减少假样本权重
            return real_loss * 5.0 + fake_loss * 0.1
        elif real_accuracy > 0.8 and fake_accuracy < 0.2:
            # 反之亦然
            return real_loss * 0.1 + fake_loss * 5.0
        else:
            # 相对平衡情况
            return (real_loss + fake_loss) / 2.0
    
    


    def weighted_bce_loss(predictions, targets, pos_weight=2.0, neg_weight=2.0):
        """改进的加权二元交叉熵损失，防止数值问题"""
        # 使用更安全的epsilon值
        eps = 1e-6
        
        # 使用更严格的clamp确保预测值在有效范围内
        predictions = torch.clamp(predictions, eps, 1.0 - eps)
        
        # 使用更稳定的形式计算损失
        pos_part = -targets * torch.log(predictions) * pos_weight
        neg_part = -(1.0 - targets) * torch.log(1.0 - predictions) * neg_weight
        
        # 检查无效值并替换
        pos_part = torch.where(torch.isnan(pos_part), torch.zeros_like(pos_part), pos_part)
        neg_part = torch.where(torch.isnan(neg_part), torch.zeros_like(neg_part), neg_part)
        
        # 计算均值前检查总损失
        loss = pos_part + neg_part
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        
        return loss.mean()
        
    def feature_contrastive_loss(real_local_feat, real_global_feat, fake_local_feat, fake_global_feat):
        """特征对比损失 - 鼓励真假样本特征分离"""
        
        # 计算真假特征均值
        real_local_mean = real_local_feat.mean(0)
        fake_local_mean = fake_local_feat.mean(0)
        real_global_mean = real_global_feat.mean(0)
        fake_global_mean = fake_global_feat.mean(0)
        
        # 计算欧氏距离
        local_distance = torch.sqrt(torch.sum((real_local_mean - fake_local_mean)**2) + 1e-6)
        global_distance = torch.sqrt(torch.sum((real_global_mean - fake_global_mean)**2) + 1e-6)
        
        # 或使用 pairwise_distance
        # local_distance = F.pairwise_distance(real_local_mean.unsqueeze(0), fake_local_mean.unsqueeze(0)).mean()
        # global_distance = F.pairwise_distance(real_global_mean.unsqueeze(0), fake_global_mean.unsqueeze(0)).mean()
        
        # 对比损失 - 我们希望最大化距离（所以使用负号）
        contrastive_loss = 1.0/(local_distance + global_distance)
        
        return contrastive_loss
    
    def gradient_penalty(discriminator, 
                    real_local_data, 
                    real_local_mask, 
                    real_global_data, 
                    real_global_mask, 
                    fake_local_data, 
                    fake_local_mask,
                    fake_global_data, 
                    fake_global_mask):
        """内存效率更高的梯度惩罚计算"""
        batch_size = real_local_data.size(0)
        
        # 只使用较小的批量计算梯度惩罚
        max_samples = min(batch_size, 4)  # 一次最多处理4个样本
        
        # 随机选择样本
        indices = torch.randperm(batch_size)[:max_samples]
        
        real_local_sample = real_local_data[indices]
        real_local_mask_sample = real_local_mask[indices]
        real_global_sample = real_global_data[indices]
        real_global_mask_sample = real_global_mask[indices]
        
        fake_local_sample = fake_local_data[indices]
        fake_local_mask_sample = fake_local_mask[indices]  
        fake_global_sample = fake_global_data[indices]
        fake_global_mask_sample = fake_global_mask[indices]
        
        # 以下计算与原始函数相同，但使用较小的样本
        alpha = torch.rand(max_samples, 1, 1, 1, device=real_local_data.device)
        
        interpolates_local = alpha * real_local_sample + (1 - alpha) * fake_local_sample
        interpolates_global = alpha * real_global_sample + (1 - alpha) * fake_global_sample
        
        interpolates_local_mask = (alpha * real_local_mask_sample + (1 - alpha) * fake_local_mask_sample).round()
        interpolates_global_mask = (alpha * real_global_mask_sample + (1 - alpha) * fake_global_mask_sample).round()
        
        interpolates_local.requires_grad_(True)
        interpolates_global.requires_grad_(True)
        
        d_interpolates, _, _ = discriminator(
            interpolates_local, 
            interpolates_local_mask, 
            interpolates_global, 
            interpolates_global_mask
        )
        
        fake = torch.ones(d_interpolates.size(), device=real_local_data.device)
        
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=[interpolates_local, interpolates_global],
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )
        
        gradients_local = gradients[0].view(max_samples, -1)
        gradients_global = gradients[1].view(max_samples, -1)
        
        gradients_all = torch.cat([gradients_local, gradients_global], dim=1)
        gradient_norm = gradients_all.norm(2, dim=1)
        
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        
        return gradient_penalty

    def wgan_gp_discriminator_loss(discriminator, 
                                   real_local_data, 
                                   real_local_mask, 
                                   real_global_data, 
                                   real_global_mask, 
                                   fake_local_data, 
                                   fake_local_mask,
                                   fake_global_data, 
                                   fake_global_mask,
                                   lambda_gp=10
                                   ):
        """
        计算完整的WGAN-GP判别器损失
        
        参数:
            discriminator: 判别器模型
            real_data: 真实数据样本
            fake_data: 生成的数据样本
            lambda_gp: 梯度惩罚系数，通常为10
            mask: 可选的掩码，用于条件GAN
            
        返回:
            loss: 带有梯度惩罚的WGAN判别器总损失
        """
        # 获取判别器对真实和生成样本的评分

        real_pred,_,_ = discriminator(real_local_data, real_local_mask, real_global_data, real_global_mask)
        fake_pred,_,_ = discriminator(fake_local_data, fake_local_mask, fake_global_data, fake_global_mask)
        
        # 计算Wasserstein损失
        wasserstein_loss = torch.mean(fake_pred) - torch.mean(real_pred)
        
        # 计算梯度惩罚
        gp = gradient_penalty(discriminator, 
                            real_local_data, 
                            real_local_mask, 
                            real_global_data, 
                            real_global_mask, 
                            fake_local_data, 
                            fake_local_mask,
                            fake_global_data, 
                            fake_global_mask)
        
        # 总损失 = Wasserstein损失 + lambda_gp * 梯度惩罚
        return wasserstein_loss + lambda_gp * gp
    
    def analyze_predictions_distribution(real_preds, fake_preds):
        """分析判别器的预测值分布"""
        # 使用detach()分离计算图
        real_vals = real_preds.detach().cpu().numpy().flatten()
        fake_vals = fake_preds.detach().cpu().numpy().flatten()
        
        # 统计基本信息
        print(f"真实样本预测值: 平均={real_vals.mean():.4f}, 中位数={np.median(real_vals):.4f}, 最小={real_vals.min():.4f}, 最大={real_vals.max():.4f}")
        print(f"虚假样本预测值: 平均={fake_vals.mean():.4f}, 中位数={np.median(fake_vals):.4f}, 最小={fake_vals.min():.4f}, 最大={fake_vals.max():.4f}")
        
        
        # 检查是否存在对称模式
        real_below = np.sum(real_vals < 0.5) / len(real_vals)
        real_above = np.sum(real_vals > 0.5) / len(real_vals)
        fake_below = np.sum(fake_vals < 0.5) / len(fake_vals)
        fake_above = np.sum(fake_vals > 0.5) / len(fake_vals)
        
        print(f"真实样本分布: <0.5占{real_below:.2%}, >0.5占{real_above:.2%}")
        print(f"虚假样本分布: <0.5占{fake_below:.2%}, >0.5占{fake_above:.2%}")
        
        # 检查是否刚好出现互补模式
        complement_ratio = (real_above + fake_below) / 2
        print(f"互补分布比例: {complement_ratio:.2%} (0表示完美互补)")
        
        return {
            'real_mean': real_vals.mean(),
            'fake_mean': fake_vals.mean(),
            'real_median': np.median(real_vals),
            'fake_median': np.median(fake_vals),
            'real_above_threshold': real_above,
            'fake_below_threshold': fake_below,
            'complement_ratio': complement_ratio
        }
    
    def log_scale_balance_loss(loss_real, loss_fake, lambda_balance=1.0):
        """
        对数比例平衡损失，专门处理数量级差异
        
        参数:
        - loss_real: 真实样本的损失
        - loss_fake: 虚假样本的损失
        - lambda_balance: 平衡损失权重
        
        返回:
        - 平衡损失项
        """
        # 添加小常数防止log(0)
        epsilon = 1e-8
        
        # 计算对数比例
        log_ratio = torch.log(loss_real + epsilon) - torch.log(loss_fake + epsilon)
        
        # 平衡损失是对数比例的平方，当比例为1(对数为0)时最小
        balance_loss = lambda_balance * log_ratio ** 2
        
        return balance_loss
    
    
    def r1_regularization(discriminator, real_data, real_mask, real_global, real_global_mask, weight=1e3):
        """R1正则化 - 在真实数据上的梯度惩罚"""
        # 需要梯度
        real_data.requires_grad_(True)
        
        # 前向传播
        real_pred, _, _ = discriminator(real_data, real_mask, real_global, real_global_mask)
        
        # 对所有样本的预测求和
        pred_sum = real_pred.sum()
        
        # 计算梯度
        gradients = torch.autograd.grad(outputs=pred_sum, inputs=real_data, 
                                    create_graph=True, retain_graph=True)[0]
        
        # 梯度平方范数
        grad_norm_squared = gradients.pow(2).reshape(gradients.shape[0], -1).sum(1)
        
        # R1正则化
        r1_penalty = weight * grad_norm_squared.mean() / 2
        
        return r1_penalty

    # 动态增加R1权重
    def get_r1_weight(step, max_steps, min_weight=1.0, max_weight=10.0):
        return min_weight + (max_weight - min_weight) * min(1.0, step / (0.5 * max_steps))
    
    '''print("Starting Phase 2 training...")
    pbar = tqdm(total=steps_2)
    step = 0'''
    # best_acc = float("-inf")
    

    if step_phase2 < steps_2 and phase == 2:
        print("开始/继续第2阶段训练...")
    
        # 使用Adam优化器
        opt_cd = torch.optim.Adam(model_cd.parameters(), lr=2e-5, betas=(0.5, 0.999), eps=1e-8)
        
        # 创建梯度缩放器，用于自动混合精度训练
        scaler = GradScaler(enabled=True)
        
        # 创建学习率调度器
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(opt_cd, T_max=steps_2, eta_min=1e-6)
        
        # 实例噪声初始值和最小值
        start_noise = 0.1
        min_noise = 0.001
        
        # 跟踪性能指标
        running_loss = 0.0
        running_real_acc = 0.0
        running_fake_acc = 0.0
        best_acc = float('-inf')
        patience_counter = 0
        max_patience = 10
        
        pbar = tqdm(total=steps_2, initial=step_phase2)
        step = step_phase2

        while step < steps_2:
            for batch in train_loader:
                # 准备批次数据
                batch_data = prepare_batch_data(batch, device)
                (
                    batch_local_inputs,
                    batch_local_masks,
                    batch_local_targets,
                    batch_global_inputs,
                    batch_global_masks,
                    batch_global_targets,
                    metadata,
                ) = batch_data
                
                # 使用自动混合精度
                with autocast(device_type=device.type, dtype=torch.float32, enabled=True):
                    # 生成假样本
                    with torch.no_grad():
                        fake_outputs = model_cn(
                            batch_local_inputs,
                            batch_local_masks,
                            batch_global_inputs,
                            batch_global_masks,
                        )              
                    
                    # 计算当前噪声水平 - 随着训练进展减少
                    noise_level = max(min_noise, start_noise * (1.0 - step / steps_2))
                    
                    # 应用实例噪声
                    batch_local_targets_noisy = batch_local_targets + torch.randn_like(batch_local_targets) * noise_level
                    fake_outputs_noisy = fake_outputs + torch.randn_like(fake_outputs) * noise_level
                    
                    
                    # 嵌入假输出到全局
                    fake_global_embedded, fake_global_mask_embedded, _ = merge_local_to_global(
                        fake_outputs_noisy, batch_global_inputs, batch_global_masks, metadata
                    )
                    
                    # 嵌入真实输入到全局
                    real_global_embedded, real_global_mask_embedded, _ = merge_local_to_global(
                        batch_local_targets_noisy, batch_global_inputs, batch_global_masks, metadata
                    )
                    
                    # 创建平滑标签
                    batch_size = batch_local_targets.size(0)
                    real_labels = torch.ones(batch_size, 1).to(device)*0.95   # 标签平滑
                    fake_labels = torch.zeros(batch_size, 1).to(device)*0.05   # 标签平滑
                    
                    # 训练判别器处理假样本
                    fake_predictions, fake_lf, fake_gf = model_cd(
                        fake_outputs_noisy,
                        batch_local_masks,
                        fake_global_embedded,
                        fake_global_mask_embedded,
                    )
                    
                    # 训练判别器处理真实样本
                    real_predictions, real_lf, real_gf = model_cd(
                        batch_local_targets_noisy,
                        batch_local_masks,
                        real_global_embedded,
                        real_global_mask_embedded,
                    )
                    
                    # 使用BCEWithLogitsLoss代替BCE或自定义损失
                    loss_real = F.binary_cross_entropy_with_logits(real_predictions, real_labels)
                    loss_fake = F.binary_cross_entropy_with_logits(fake_predictions, fake_labels)
                    
                    # 计算特征匹配损失
                    # 优化特征匹配，关注均值和方差
                    fm_weight = 4  # 使用较小的权重
                    

                    fm_loss = feature_contrastive_loss(
                        real_lf, real_gf, fake_lf, fake_gf)* fm_weight
                    lsb_loss = log_scale_balance_loss(loss_real, loss_fake)
                    r1_loss = r1_regularization(
                        model_cd,
                        batch_local_targets_noisy,
                        batch_local_masks,
                        real_global_embedded,
                        real_global_mask_embedded
                    )

                    '''gp_loss = gradient_penalty(model_cd, 
                        batch_local_targets_noisy, 
                        batch_local_masks, 
                        real_global_embedded, 
                        real_global_mask_embedded, 
                        fake_outputs_noisy, 
                        batch_local_masks,
                        fake_global_embedded,
                        fake_global_mask_embedded)'''
                    
                    # 使用sigmoid计算预测概率，用于准确率计算
                    with torch.no_grad():
                        real_probs = torch.sigmoid(real_predictions)
                        fake_probs = torch.sigmoid(fake_predictions)
                        
                        real_acc = (real_probs >= 0.5).float().mean().item()
                        fake_acc = (fake_probs < 0.5).float().mean().item()
                        avg_acc = (real_acc + fake_acc) / 2
        
                    '''# 动态调整损失权重，平衡训练
                    if real_acc < 0.1 and fake_acc > 0.9:
                        # 真实样本识别困难，增加真实样本权重
                        loss_real *= 1.5
                        loss_fake *= 0.66
                        print("loss real weight strengthened")
                    elif fake_acc < 0.1 and real_acc > 0.9:
                        # 假样本识别困难，增加假样本权重
                        loss_real *= 0.66
                        loss_fake *= 1.5
                        print("loss fake weight strengthened")
                    else:
                        0'''

                    loss = loss_real + loss_fake + fm_loss + lsb_loss + r1_loss # + gp_loss
                    print(f"loss_real: {loss_real}, loss_fake: {loss_fake}, fm_loss: {fm_loss}, r1_loss: {r1_loss}")
                
                # 使用梯度缩放器进行反向传播
                scaler.scale(loss).backward()
                
                # 梯度裁剪
                scaler.unscale_(opt_cd)
                # torch.nn.utils.clip_grad_norm_(model_cd.parameters(), 1.0)
                
                # 更新参数
                scaler.step(opt_cd)
                scaler.update()
                opt_cd.zero_grad(set_to_none=True)  # 更彻底地清理梯度
                
                # 更新学习率
                scheduler.step()
                
                # 更新运行指标
                running_loss += loss.item()
                running_real_acc += real_acc
                running_fake_acc += fake_acc
                
                # 定期打印详细统计信息
                if step % 100 == 0:
                    avg_loss = running_loss / min(100, step - step_phase2 + 1)
                    avg_real_acc = running_real_acc / min(100, step - step_phase2 + 1)
                    avg_fake_acc = running_fake_acc / min(100, step - step_phase2 + 1)
                    
                    print(f"\n统计信息(最近100步):")
                    print(f"平均损失: {avg_loss:.4f}")
                    print(f"真实样本平均准确率: {avg_real_acc:.4f}")
                    print(f"虚假样本平均准确率: {avg_fake_acc:.4f}")
                    print(f"当前学习率: {scheduler.get_last_lr()[0]:.6f}")
                    print(f"真实样本预测分布: 均值={real_probs.mean().item():.4f}, 最小={real_probs.min().item():.4f}, 最大={real_probs.max().item():.4f}")
                    print(f"虚假样本预测分布: 均值={fake_probs.mean().item():.4f}, 最小={fake_probs.min().item():.4f}, 最大={fake_probs.max().item():.4f}")
                    
                    # 重置运行指标
                    running_loss = 0.0
                    running_real_acc = 0.0
                    running_fake_acc = 0.0
                
                # 更新进度条
                current_lr = scheduler.get_last_lr()[0]
                pbar.set_description(f"Phase 2 | loss: {loss.item():.4f}, R_acc: {real_acc:.3f}, F_acc: {fake_acc:.3f}, lr: {current_lr:.1e}")
                pbar.update(1)
                
                # Visdom可视化
                viz.line(
                    Y=torch.tensor([loss.item()]),
                    X=torch.tensor([step]),
                    win=loss_windows["phase2_disc"],
                    update="append",
                )
                
                viz.line(
                    Y=torch.tensor([avg_acc]),
                    X=torch.tensor([step]),
                    win=loss_windows["phase2_acc"],
                    update="append",
                )
                
                step += 1
                
                # 在每个snaperiod_2步保存检查点和进行验证
                if step % snaperiod_2 == 0:
                    # 评估判别器
                    model_cd.eval()
                    val_correct = 0
                    val_total = 0

                    with torch.no_grad():
                        for val_batch in val_subset_loader:
                            # 准备验证批次
                            val_data = prepare_batch_data(val_batch, device)
                            (
                                val_local_inputs,
                                val_local_masks,
                                val_local_targets,
                                val_global_inputs,
                                val_global_masks,
                                val_global_targets,
                                val_metadata,
                            ) = val_data

                            # 生成假样本
                            val_fake_outputs = model_cn(
                                val_local_inputs,
                                val_local_masks,
                                val_global_inputs,
                                val_global_masks,
                            )
                            
                            # 嵌入
                            val_fake_global_embedded, val_fake_global_mask_embedded, _ = merge_local_to_global(
                                val_fake_outputs,
                                val_global_inputs,
                                val_global_masks,
                                val_metadata,
                            )
                            
                            val_real_global_embedded, val_real_global_mask_embedded, _ = merge_local_to_global(
                                val_local_targets,
                                val_global_inputs,
                                val_global_masks,
                                val_metadata,
                            )

                            # 获取判别器预测
                            val_fake_preds, _, _ = model_cd(
                                val_fake_outputs,
                                val_local_masks,
                                val_fake_global_embedded,
                                val_fake_global_mask_embedded,
                            )
                            
                            val_real_preds, _, _ = model_cd(
                                val_local_targets,
                                val_local_masks,
                                val_real_global_embedded,
                                val_real_global_mask_embedded,
                            )
                            val_fake_preds = torch.sigmoid(val_fake_preds)
                            val_real_preds = torch.sigmoid(val_real_preds)

                            # 计算准确率
                            val_total += val_fake_preds.size(0) * 2
                            val_correct += (
                                (val_fake_preds < 0.5).sum() + (val_real_preds >= 0.5).sum()
                            ).item()
                            print(f"validation: fake_preds: {(val_fake_preds<0.5).sum()}/{val_fake_preds.size(0)}, real_preds: {(val_real_preds>0.5).sum()}/{val_real_preds.size(0)}")
                    
                    val_accuracy = val_correct / val_total
                    print(f"\n验证准确率: {val_accuracy:.4f}")
                    
                    # 检查是否有改进
                    if val_accuracy > best_acc:
                        best_acc = val_accuracy
                        best_model_path = os.path.join(result_dir, "phase_2", "model_cd_best")
                        torch.save(model_cd.state_dict(), best_model_path)
                        print(f"发现新的最佳模型，准确率: {val_accuracy:.4f}")
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        print(f"未见改善。耐心计数: {patience_counter}/{max_patience}")
                        
                        # 如果长时间没有改善，降低学习率
                        if patience_counter >= max_patience:
                            for param_group in opt_cd.param_groups:
                                param_group['lr'] *= 0.5
                            print(f"性能停滞，将学习率降低到: {opt_cd.param_groups[0]['lr']:.6f}")
                            patience_counter = 0
                    
                    # 保存模型
                    folder= os.path.join(result_dir, "phase_2")
                    old_files = glob.glob(os.path.join(folder, "model_cd_step*"))
                    for file in old_files:
                        try:
                            os.remove(file)
                            print(f"删除旧文件: {os.path.basename(file)}")
                        except Exception as e:
                            print(f"删除文件失败: {e}")
                    model_path = os.path.join(result_dir, "phase_2", f"model_cd_step{step}")
                    torch.save(model_cd.state_dict(), model_path)
                    
                    
                    # 回到训练模式
                    model_cd.train()
                    
                    # 保存检查点
                    save_checkpoint(model_cn, model_cd, opt_cn, opt_cd, step, 2, result_dir, best_acc=best_acc)
                
                # 判断是否完成训练
                if step >= steps_2:
                    break
            
            # 如果数据集遍历完而步数未达到，重置数据加载器
            if step < steps_2:
                print("重置数据加载器...")
    
        pbar.close()
        print(f"Phase 2 训练完成! 最佳判别器准确率: {best_acc:.4f}")

    # =================================================
    # Training Phase 3: Joint training of both networks
    # =================================================
    '''print("Starting Phase 3 training...")
    torch.cuda.empty_cache()
    pbar = tqdm(total=steps_3)
    step = 0'''
    def random_label_flip(labels, flip_prob=0.1):
        """随机翻转一部分标签"""
        flipped_labels = labels.clone()
        flip_mask = (torch.rand_like(labels) < flip_prob).float()
        flipped_labels = labels * (1 - flip_mask) + (1 - labels) * flip_mask
        return flipped_labels
    def r1_regularization(discriminator, real_data, real_mask, real_global, real_global_mask, weight=1e3):
        """R1正则化 - 在真实数据上的梯度惩罚"""
        # 需要梯度
        real_data.requires_grad_(True)
        
        # 前向传播
        real_pred, _, _ = discriminator(real_data, real_mask, real_global, real_global_mask)
        
        # 对所有样本的预测求和
        pred_sum = real_pred.sum()
        
        # 计算梯度
        gradients = torch.autograd.grad(outputs=pred_sum, inputs=real_data, 
                                    create_graph=True, retain_graph=True)[0]
        
        # 梯度平方范数
        grad_norm_squared = gradients.pow(2).reshape(gradients.shape[0], -1).sum(1)
        
        # R1正则化
        r1_penalty = weight * grad_norm_squared.mean() / 2
        
        return r1_penalty

    # 动态增加R1权重
    def get_r1_weight(step, max_steps, min_weight=1.0, max_weight=10.0):
        return min_weight + (max_weight - min_weight) * min(1.0, step / (0.5 * max_steps))
    
    def check_discriminator_collapse(real_acc, fake_acc, threshold=0.2):
        """检查判别器是否崩溃"""
        # 如果真实样本准确率太低或假样本准确率太高
        return real_acc < threshold or (1 - fake_acc) < threshold

    # best_val_loss_joint = float("inf")
    if step_phase3 < steps_3 and phase == 3:
        print("开始/继续第3阶段训练...")
        torch.cuda.empty_cache()
        pbar = tqdm(total=steps_3, initial=step_phase3)
        step = step_phase3
        # 初始化阶段设置不同的权重
        alpha_d = 1.0  # 判别器权重
        alpha_g = 0.1  # 生成器权重，通常小于判别器权重
        fm_weight = 1.0  # 特征匹配权重
        lambda_gp = 10.0  # 梯度惩罚权重
        smooth_factor = 0.1  # 标签平滑因子
        
        # 修改判别器优化器，增加权重衰减
        opt_cd = torch.optim.Adam(model_cd.parameters(), lr=1e-5, betas=(0.5, 0.999), weight_decay=1e-4)
        '''def apply_spectral_norm(module):
            """递归应用谱归一化到所有卷积和线性层"""
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                return spectral_norm(module)
            for name, child in module.named_children():
                module._modules[name] = apply_spectral_norm(child)
            return module

        # 应用到判别器
        model_cd = apply_spectral_norm(model_cd)'''

        
        # 添加学习率调度器，在训练进行时逐渐降低学习率
        # ReduceLROnPlateau 学习率调度器参数说明（中文）：
        # mode: 'min' 表示当监控的指标停止下降时，降低学习率
        # factor: 学习率缩放因子，每次降低为原来的多少（如0.5表示减半）
        # patience: 容忍多少个epoch/step指标无改善后才降低学习率
        # verbose: 是否打印学习率调整信息
        # 这里 scheduler_d 监控判别器的对抗损失（val_adv_loss），scheduler_g 监控生成器的重建损失（val_recon_loss）
        scheduler_d = ReduceLROnPlateau(opt_cd, mode='min', factor=0.8, patience=10, verbose=True)
        scheduler_g = ReduceLROnPlateau(opt_cn, mode='min', factor=0.8, patience=10, verbose=True)
        good_state = copy.deepcopy(model_cd.state_dict())
        patience_counter = 0

        # 主训练循环
        while step < steps_3:
            for batch in train_loader:
                # 只在必要时清理缓存
                if step % 50 == 0:
                    torch.cuda.empty_cache()
                    
                # 准备数据
                batch_data = prepare_batch_data(batch, device)
                (batch_local_inputs, batch_local_masks, batch_local_targets,
                batch_global_inputs, batch_global_masks, batch_global_targets, metadata) = batch_data

                # --------------------------------------
                # 1. 训练判别器
                # --------------------------------------
                # 冻结生成器参数
                for param in model_cn.parameters():
                    param.requires_grad = False
                    
                # 启用判别器梯度
                for param in model_cd.parameters():
                    param.requires_grad = True
                    
                # 生成假样本
                with torch.no_grad():
                    fake_outputs = model_cn(batch_local_inputs, batch_local_masks, 
                                            batch_global_inputs, batch_global_masks)

                # 准备真假标签 - 使用标签平滑
                batch_size = batch_local_targets.size(0)
                real_labels = torch.ones(batch_size, 1, device=device) * (1.0 - smooth_factor)
                fake_labels = torch.zeros(batch_size, 1, device=device) + smooth_factor
                # 在训练判别器时使用
                real_labels_flipped = random_label_flip(real_labels, flip_prob=0.1)
                fake_labels_flipped = random_label_flip(fake_labels, flip_prob=0.1)
                
                # 真实样本前向传播
                real_global_embedded, real_global_mask_embedded, _ = merge_local_to_global(
                    batch_local_targets, batch_global_inputs, batch_global_masks, metadata)
                real_predictions, real_lf, real_gf = model_cd(
                    batch_local_targets, batch_local_masks, 
                    real_global_embedded, real_global_mask_embedded)
                loss_cd_real = F.binary_cross_entropy_with_logits(real_predictions, real_labels_flipped)
                
                # 假样本前向传播
                fake_global_embedded, fake_global_mask_embedded, _ = merge_local_to_global(
                    fake_outputs, batch_global_inputs, batch_global_masks, metadata)
                fake_predictions, fake_lf, fake_gf = model_cd(
                    fake_outputs.detach(),  # 分离梯度
                    batch_local_masks, 
                    fake_global_embedded.detach(),  # 分离梯度
                    fake_global_mask_embedded)
                loss_cd_fake = F.binary_cross_entropy_with_logits(fake_predictions, fake_labels_flipped)
                
                # 特征匹配损失 - 使用分离的生成器特征
                fm_loss_d = feature_contrastive_loss(
                    real_lf, real_gf, fake_lf, fake_gf) * fm_weight
                    
                # 损失平衡
                lsb_loss = log_scale_balance_loss(loss_cd_real, loss_cd_fake)
                r1_loss = r1_regularization(
                        model_cd,
                        batch_local_targets,
                        batch_local_masks,
                        real_global_embedded,
                        real_global_mask_embedded
                    )
                
                with torch.no_grad():
                    real_probs = torch.sigmoid(real_predictions)
                    fake_probs = torch.sigmoid(fake_predictions)
                    
                    real_acc = (real_probs >= 0.5).float().mean().item()
                    fake_acc = (fake_probs < 0.5).float().mean().item()
                    avg_acc = (real_acc + fake_acc) / 2
                # 动态调整损失权重，平衡训练
                if real_acc < 0.1 and fake_acc > 0.9:
                    # 真实样本识别困难，增加真实样本权重
                    loss_real *= 1.1
                    loss_fake *= 0.95
                    print("loss real weight strengthened")
                elif fake_acc < 0.1 and real_acc > 0.9:
                    # 假样本识别困难，增加假样本权重
                    loss_real *= 0.95 
                    loss_fake *= 1.1
                    print("loss fake weight strengthened")
                else:
                    0
                
                ''' # 梯度惩罚
                gp = gradient_penalty(model_cd, 
                                    batch_local_targets, batch_local_masks, 
                                    real_global_embedded, real_global_mask_embedded,
                                    fake_outputs.detach(), batch_local_masks, 
                                    fake_global_embedded.detach(), fake_global_mask_embedded)'''
                
                # 组合判别器损失
                loss_cd = ((loss_cd_fake + loss_cd_real) / 2.0 + 
                            fm_loss_d + lsb_loss) * alpha_d + r1_loss
            
                # Update Visdom plot for discriminator loss
                viz.line(
                    Y=torch.tensor([loss_cd.item()]),
                    X=torch.tensor([step]),
                    win=loss_windows["phase3_disc"],
                    update="append",
                )
                viz.line(
                    Y=torch.tensor([[fake_acc.item(), real_acc.item()]]),
                    X=torch.tensor([step]),
                    win=loss_windows["phase3_fake_acc"],
                    update="append",
                )
                
                # 反向传播和优化
                opt_cd.zero_grad()
                loss_cd.backward()
                torch.nn.utils.clip_grad_norm_(model_cd.parameters(), max_norm=1.0)
                opt_cd.step()
        
                
                # --------------------------------------
                # 2. 训练生成器
                # --------------------------------------
                # 解冻生成器参数
                for param in model_cn.parameters():
                    param.requires_grad = True
                    
                # 冻结判别器参数
                for param in model_cd.parameters():
                    param.requires_grad = False
                    
                # 生成假样本
                fake_outputs = model_cn(batch_local_inputs, batch_local_masks, 
                                        batch_global_inputs, batch_global_masks)
                fake_completed, fake_completed_mask, pos = merge_local_to_global(
                    fake_outputs, batch_global_inputs, batch_global_masks, metadata
                )
                
                # 计算重建损失
                loss_cn_recon = completion_network_loss(
                    fake_outputs, batch_local_targets, batch_local_masks,
                    batch_global_targets, fake_completed, fake_completed_mask, pos)
                
                # 计算对抗损失
                fake_global_embedded, fake_global_mask_embedded, _ = merge_local_to_global(
                    fake_outputs, batch_global_inputs, batch_global_masks, metadata)
                fake_predictions, fake_lf, fake_gf = model_cd(
                    fake_outputs, batch_local_masks, 
                    fake_global_embedded, fake_global_mask_embedded)
                loss_cn_adv = F.binary_cross_entropy_with_logits(fake_predictions, real_labels)
                
                # 特征匹配损失 - 使用分离的判别器特征
                fm_loss_g = feature_contrastive_loss(
                    real_lf.detach(), real_gf.detach(), fake_lf, fake_gf) * fm_weight
                
                # 组合生成器损失
                loss_cn = loss_cn_recon + alpha_g * loss_cn_adv + fm_loss_g
                
                # 反向传播和优化
                opt_cn.zero_grad()
                loss_cn.backward()
                # 在这里添加梯度裁剪，backward()之后，step()之前
                torch.nn.utils.clip_grad_norm_(model_cn .parameters(), max_norm=1.0)
                opt_cn.step()
                
                          
                # 解除参数冻结
                for param in model_cd.parameters():
                    param.requires_grad = True
                    
                # 更新可视化和进度条...
                # Update Visdom plot for generator loss
                viz.line(
                    Y=torch.tensor([loss_cn.item()]),
                    X=torch.tensor([step]),
                    win=loss_windows["phase3_gen"],
                    update="append",
                )
                # 在每个snaperiod_3步保存检查点
                if step % snaperiod_3 == 0:
                    # 运行验证和可视化代码...
                    
                    # 保存检查点
                    save_checkpoint(model_cn, model_cd, opt_cn, opt_cd, step, 3, result_dir, best_val_loss=best_val_loss_joint)
                
                step += 1
                pbar.set_description(
                    f"Phase 3 | D loss: {loss_cd.item():.5f}, G loss: {loss_cn.item():.5f}"
                )
                pbar.update(1)

                # Testing and saving snapshots
                if step % snaperiod_3 == 0:
                    # Run validation
                    model_cn.eval()
                    val_loss = 0.0

                    # avg_val_loss = validate(model_cn, val_subset_loader, device)
                    # 使用改进后的验证函数评估模型
                    val_metrics = validateAll(model_cn, model_cd, val_subset_loader, device)  
                    # 提取各项指标
                    val_recon_loss = val_metrics['recon_loss']
                    val_adv_loss = val_metrics['adv_loss']
                    val_real_acc = val_metrics['real_acc']
                    val_fake_acc = val_metrics['fake_acc']
                    val_disc_acc = val_metrics['disc_acc']  
                    # print(f"Step {step}, Validation Loss: {avg_val_loss:.5f}")
                    # 使用验证损失更新调度器
                    scheduler_d.step(val_adv_loss)
                    scheduler_g.step(val_recon_loss)
                    
                    
                    # 检查判别器崩溃
                    if (val_real_acc < 0.3 or val_fake_acc < 0.3) and patience_counter>10:
                        print("检测到判别器可能崩溃，采取措施...")
                        
                        # 如果有良好状态的保存，则重置
                        if 'good_state' in locals() and good_state is not None:
                            print("重置判别器到之前的良好状态")
                            model_cd.load_state_dict(good_state)
                            patience_counter += 1
                        # 否则减小学习率并增加正则化
                        else:
                            print("减小学习率并增加正则化")
                            for param_group in opt_cd.param_groups:
                                param_group['lr'] *= 0.8
                            # 增加权重衰减
                            opt_cd = torch.optim.Adam(model_cd.parameters(), 
                                                    lr=param_group['lr'], 
                                                    betas=(0.8, 0.999), 
                                                    weight_decay=5e-4)
                    
                    # 如果判别器表现良好，保存状态
                    elif 0.4 < val_real_acc  and 0.4 < val_fake_acc:
                        print("判别器表现平衡，保存良好状态")
                        good_state = copy.deepcopy(model_cd.state_dict())
                        patience_counter = 0

                    # Update Visdom plot for validation loss
                    viz.line(
                        Y=torch.tensor([[val_recon_loss, val_adv_loss, val_real_acc, val_fake_acc, val_disc_acc]]),
                        X=torch.tensor([step]),
                        win=loss_windows["phase3_val"],
                        update="append",
                    )

                    if val_recon_loss < best_val_loss_joint:
                        best_val_loss_joint = val_recon_loss
                        cn_best_path = os.path.join(result_dir, "phase_3", "model_cn_best")
                        cd_best_path = os.path.join(result_dir, "phase_3", "model_cd_best")
                        torch.save(model_cn.state_dict(), cn_best_path)
                        torch.save(model_cd.state_dict(), cd_best_path)
                        print(f"  Saved new best model with val_recon_loss = {val_recon_loss:.4f}")

                    # Generate test completions for visualization
                    test_batch = next(iter(val_subset_loader))

                    # Prepare test batch
                    test_data = prepare_batch_data(test_batch, device)
                    (
                        test_local_inputs,
                        test_local_masks,
                        test_local_targets,
                        test_global_inputs,
                        test_global_masks,
                        test_global_targets,
                        metadata,
                    ) = test_data

                    # Generate completions
                    test_output = model_cn(
                        test_local_inputs,
                        test_local_masks,
                        test_global_inputs,
                        test_global_masks,
                    )

                    # Merge local output to global for visualization
                    completed, completed_mask, _ = merge_local_to_global(
                        test_output, test_global_inputs, test_global_masks, metadata
                    )

                    # Visualize results in Visdom
                    visualize_results(
                        test_local_targets,
                        test_local_inputs,
                        test_output,
                        completed,
                        step,
                        "phase_3",
                    )

                    # Save both models
                    # 删除旧的 cn 和 cd 模型文件
                    for pattern in ["model_cn_step*", "model_cd_step*"]:
                        old_files = glob.glob(os.path.join(result_dir, pattern))
                        for file in old_files:
                            try:
                                os.remove(file)
                                print(f"删除旧文件: {os.path.basename(file)}")
                            except Exception as e:
                                print(f"删除文件失败: {e}")
                    cn_path = os.path.join(result_dir, "phase_3", f"model_cn_step{step}")
                    cd_path = os.path.join(result_dir, "phase_3", f"model_cd_step{step}")
                    torch.save(model_cn.state_dict(), cn_path)
                    torch.save(model_cd.state_dict(), cd_path)

                    model_cn.train()
                    model_cd.train()

                if step >= steps_3:
                    break

        pbar.close()




        # Final visualization: Compare best models from each phase
        test_batch = next(iter(val_subset_loader))
        test_data = prepare_batch_data(test_batch, device)
        (
            test_local_inputs,
            test_local_masks,
            test_local_targets,
            test_global_inputs,
            test_global_masks,
            test_global_targets,
            metadata,
        ) = test_data

        # Load best models from each phase
        best_models = {}

        # Best model from phase 1
        best_model_p1 = CompletionNetwork(input_channels=1).to(device)
        best_model_p1.load_state_dict(
            torch.load(os.path.join(result_dir, "phase_1", "model_cn_best"), weights_only=True)
        )
        best_model_p1.eval()

        # Best model from phase 3 (joint training)
        best_model_p3 = CompletionNetwork(input_channels=1).to(device)
        best_model_p3.load_state_dict(
            torch.load(os.path.join(result_dir, "phase_3", "model_cn_best"), weights_only=True)
        )
        best_model_p3.eval()

        with torch.no_grad():
            # Generate completions with best models
            output_p1 = best_model_p1(
                test_local_inputs,
                test_local_masks,
                test_global_inputs,
                test_global_masks,
            )

            output_p3 = best_model_p3(
                test_local_inputs,
                test_local_masks,
                test_global_inputs,
                test_global_masks,
            )

            # Merge outputs to global for visualization
            completed_p1, _, _ = merge_local_to_global(
                output_p1, test_global_inputs, test_global_masks, metadata
            )

            completed_p3, _, _ = merge_local_to_global(
                output_p3, test_global_inputs, test_global_masks, metadata
            )

            # Create comparison grid
            comparison_grid = [
                test_global_inputs.cpu(),  # Ground truth
                test_global_targets.cpu(),  # Input
                completed_p1.cpu(),  # Phase 1 completed
                completed_p3.cpu(),  # Phase 3 completed
            ]

            # Visualize comparison in Visdom
            '''grid = make_grid(torch.cat(comparison_grid, dim=0), nrow=600)
            viz.image(
                grid,
                opts=dict(
                    caption="Model Comparison (Ground Truth, Input, P1 Output, P3 Output, P1 Completed, P3 Completed)",
                    title="Best Models Comparison",
                ),
                win="best_models_comparison",
            )'''

        print("Training complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DEM completion network training") 
    parser.add_argument("--resume", type=str, default=r"E:\KingCrimson Dataset\Simulate\data0\resultsphase2\checkpoint_phase2_step28000.pth", help="resume from checkpoint path")
    args = parser.parse_args()
    
    train(resume_from=args.resume)
