import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
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

# Import your models
from models import CompletionNetwork, ContextDiscriminator

# Import your custom dataset
from DemDataset import DemDataset
# from DEMData import DEMData

# Import the custom_collate_fn function from your dataset module
from DemDataset import custom_collate_fn
# from DEMData import custom_collate_fn

import gc

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# setx PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:98

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


def train():
    # =================================================
    # Configuration settings (replace with your settings)
    # =================================================
    '''json_dir = r"F:\Dataset\Simulate\data0\testjson"  # 局部json
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
    result_dir = r"E:\KingCrimson Dataset\Simulate\data0\results"  # 结果保存路径
    cache_dir = r"E:\KingCrimson Dataset\Simulate\data0\traindata"  # 缓存目录

    # Training parameters
    steps_1 = 10000  # 90000  # Phase 1 training steps
    steps_2 = 10000  # 10000  # Phase 2 training steps
    steps_3 = 10000  # 400000  # Phase 3 training steps

    snaperiod_1 =500   # 10000  # How often to save snapshots in phase 1
    snaperiod_2 = 500 # 2000  # How often to save snapshots in phase 2
    snaperiod_3 = 500   # 10000  # How often to save snapshots in phase 3

    batch_size = 16  # Batch size
    alpha = 4e-4  # Alpha parameter for loss weighting
    validation_split = 0.1  # Percentage of data to use for validation

    # Hardware setup
    if not torch.cuda.is_available():
        raise Exception("At least one GPU must be available.")
    device = torch.device("cuda:1")

    # Initialize Visdom
    viz = visdom.Visdom(env="dem_completion")
    viz.close(env="dem_completion")

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
            Y=torch.zeros(1),
            X=torch.zeros(1),
            opts=dict(title="Phase 3 Validation Loss"),
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
    print("Starting Phase 1 training...")
    model_cn.train()
    pbar = tqdm(total=steps_1)

    step = 0
    best_val_loss = float("inf")

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
    print("Starting Phase 2 training...")
    pbar = tqdm(total=steps_2)
    step = 0
    best_acc = float("-inf")

    while step < steps_2:
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

            # Generate fake samples using completion network
            with torch.no_grad():
                fake_outputs = model_cn(
                    batch_local_inputs,
                    batch_local_masks,
                    batch_global_inputs,
                    batch_global_masks,
                )

            # Embed fake outputs into global inputs
            fake_global_embedded, fake_global_mask_embedded, _ = merge_local_to_global(
                fake_outputs, batch_global_inputs, batch_global_masks, metadata
            )

            # Train discriminator with fake samples
            fake_labels = torch.zeros(fake_outputs.size(0), 1).to(device)
            fake_predictions = model_cd(
                fake_outputs,
                batch_local_masks,
                batch_global_targets,
                fake_global_mask_embedded,
            )
            loss_fake = bce_loss(fake_predictions, fake_labels)

            # Embed real inputs into global inputs
            real_global_embedded, real_global_mask_embedded, _ = merge_local_to_global(
                batch_local_inputs, batch_global_inputs, batch_global_masks, metadata
            )

            # Train discriminator with real samples
            real_labels = torch.ones(batch_local_targets.size(0), 1).to(device)
            real_predictions = model_cd(
                batch_local_targets,
                batch_local_masks,
                batch_global_targets,
                real_global_mask_embedded,
            )
            loss_real = bce_loss(real_predictions, real_labels)

            # Combined loss
            loss = (loss_fake + loss_real) / 2.0

            # Backward and optimize
            loss.backward()
            opt_cd.step()
            opt_cd.zero_grad()

            # Update Visdom plot for discriminator loss
            viz.line(
                Y=torch.tensor([loss.item()]),
                X=torch.tensor([step]),
                win=loss_windows["phase2_disc"],
                update="append",
            )

            step += 1
            pbar.set_description(f"Phase 2 | train loss: {loss.item():.5f}")
            pbar.update(1)

            # Testing and saving snapshots
            if step % snaperiod_2 == 0:
                # Calculate discriminator accuracy
                model_cd.eval()
                correct = 0
                total = 0

                with torch.no_grad():
                    # Check on validation set
                    for val_batch in val_subset_loader:
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

                        # Generate fake samples
                        fake_outputs = model_cn(
                            val_local_inputs,
                            val_local_masks,
                            val_global_inputs,
                            val_global_masks,
                        )

                        # Embed fake outputs into global inputs
                        val_fake_global_embedded, val_fake_global_mask_embedded, _ = (
                            merge_local_to_global(
                                fake_outputs,
                                val_global_inputs,
                                val_global_masks,
                                metadata,
                            )
                        )

                        # Get discriminator predictions for fake samples
                        fake_preds = model_cd(
                            fake_outputs,
                            val_local_masks,
                            val_global_targets,
                            val_fake_global_mask_embedded,
                        )

                        # Embed real inputs into global inputs
                        val_real_global_embedded, val_real_global_mask_embedded, _ = (
                            merge_local_to_global(
                                val_local_inputs,
                                val_global_inputs,
                                val_global_masks,
                                metadata,
                            )
                        )

                        # Get discriminator predictions for real samples
                        real_preds = model_cd(
                            val_local_targets,
                            val_local_masks,
                            val_global_targets,
                            val_real_global_mask_embedded,
                        )

                        # Calculate accuracy
                        total += fake_preds.size(0) * 2  # both real and fake samples
                        correct += (
                            (fake_preds < 0.5).sum() + (real_preds >= 0.5).sum()
                        ).item()

                accuracy = correct / total if total > 0 else 0
                print(f"Step {step}, Discriminator Accuracy: {accuracy:.4f}")

                # Update Visdom plot for accuracy
                viz.line(
                    Y=torch.tensor([accuracy]),
                    X=torch.tensor([step]),
                    win=loss_windows["phase2_acc"],
                    update="append",
                )

                # Save model
                model_path = os.path.join(result_dir, "phase_2", f"model_cd_step{step}")
                torch.save(model_cd.state_dict(), model_path)

                # Visualize some samples
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

                    # Visualize results in Visdom (without completed global view for phase 2)
                    visualize_results(
                        val_local_targets,
                        val_local_inputs,
                        test_output,
                        step=step,
                        phase="phase_2",
                    )

                # Save best model if accuracy improved
                if (
                    accuracy > 0.75 and accuracy > best_acc
                ):  # Using accuracy threshold of 75%
                    best_acc = accuracy
                    best_model_path = os.path.join(
                        result_dir, "phase_2", "model_cd_best"
                    )
                    torch.save(model_cd.state_dict(), best_model_path)

                model_cd.train()

            if step >= steps_2:
                break

    pbar.close()

    # =================================================
    # Training Phase 3: Joint training of both networks
    # =================================================
    print("Starting Phase 3 training...")
    torch.cuda.empty_cache()
    pbar = tqdm(total=steps_3)
    step = 0
    best_val_loss_joint = float("inf")

    while step < steps_3:
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

            # --------------------------------------
            # 1. Train the discriminator first
            # --------------------------------------
            # Generate fake samples
            with torch.no_grad():
                fake_outputs = model_cn(
                    batch_local_inputs,
                    batch_local_masks,
                    batch_global_inputs,
                    batch_global_masks,
                )

            # Train discriminator with fake samples
            fake_labels = torch.zeros(fake_outputs.size(0), 1).to(device)
            fake_global_embedded, fake_global_mask_embedded, _ = merge_local_to_global(
                fake_outputs,
                batch_global_inputs,
                batch_global_masks,
                metadata,
            )
            fake_predictions = model_cd(
                fake_outputs,
                batch_local_masks,
                batch_global_targets,
                fake_global_mask_embedded,
            )
            loss_cd_fake = bce_loss(fake_predictions, fake_labels)

            # Train discriminator with real samples
            real_labels = torch.ones(batch_local_targets.size(0), 1).to(device)
            real_global_embedded, real_global_mask_embedded, _ = merge_local_to_global(
                batch_local_targets,
                batch_global_inputs,
                batch_global_masks,
                metadata,
            )
            real_predictions = model_cd(
                batch_local_targets,
                batch_local_masks,
                batch_global_targets,
                real_global_mask_embedded,
            )
            loss_cd_real = bce_loss(real_predictions, real_labels)

            # Combined discriminator loss
            loss_cd = (loss_cd_fake + loss_cd_real) * alpha / 2.0

            # Backward and optimize discriminator
            loss_cd.backward(retain_graph=True)
            opt_cd.step()
            opt_cd.zero_grad()

            # Update Visdom plot for discriminator loss
            viz.line(
                Y=torch.tensor([loss_cd.item()]),
                X=torch.tensor([step]),
                win=loss_windows["phase3_disc"],
                update="append",
            )

            # --------------------------------------
            # 2. Then train the generator (completion network)
            # --------------------------------------
            # Generate outputs and calculate reconstruction loss
            fake_outputs = model_cn(
                batch_local_inputs,
                batch_local_masks,
                batch_global_inputs,
                batch_global_masks,
            )
            
            completed, completed_mask, pos = merge_local_to_global(
                fake_outputs, batch_global_inputs, batch_global_masks, metadata
            )

             # Calculate loss
             # loss_cn_recon = mse_loss(fake_outputs, batch_local_targets)
            loss_cn_recon = completion_network_loss(               
                fake_outputs,
                batch_local_targets,
                batch_local_masks,
                batch_global_targets,
                completed,
                completed_mask,
                pos
            )


            # Calculate adversarial loss (fool the discriminator)
            fake_global_embedded, fake_global_mask_embedded, _ = merge_local_to_global(
                fake_outputs, batch_global_inputs, batch_global_masks, metadata
            )
            fake_predictions = model_cd(
                fake_outputs,
                batch_local_masks,
                batch_global_targets,
                fake_global_mask_embedded,
            )
            loss_cn_adv = bce_loss(
                fake_predictions, real_labels
            )  # Try to make discriminator think it's real

            # Combined generator loss
            loss_cn = loss_cn_recon + alpha * loss_cn_adv

            # Backward and optimize generator
            loss_cn.backward()
            opt_cn.step()
            opt_cn.zero_grad()

            # Update Visdom plot for generator loss
            viz.line(
                Y=torch.tensor([loss_cn.item()]),
                X=torch.tensor([step]),
                win=loss_windows["phase3_gen"],
                update="append",
            )

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

                with torch.no_grad():
                    for val_batch in val_subset_loader:
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
                        val_outputs = model_cn(
                            val_local_inputs,
                            val_local_masks,
                            val_global_inputs,
                            val_global_masks,
                        )

                        # Calculate loss
                        batch_loss = mse_loss(val_outputs, val_local_targets)
                        val_loss += batch_loss.item()

                # Calculate average validation loss
                avg_val_loss = val_loss / len(val_subset_loader)
                print(f"Step {step}, Validation Loss: {avg_val_loss:.5f}")

                # Update Visdom plot for validation loss
                viz.line(
                    Y=torch.tensor([avg_val_loss]),
                    X=torch.tensor([step]),
                    win=loss_windows["phase3_val"],
                    update="append",
                )

                # Save if best model
                if avg_val_loss < best_val_loss_joint:
                    best_val_loss_joint = avg_val_loss
                    cn_best_path = os.path.join(result_dir, "phase_3", "model_cn_best")
                    cd_best_path = os.path.join(result_dir, "phase_3", "model_cd_best")
                    torch.save(model_cn.state_dict(), cn_best_path)
                    torch.save(model_cd.state_dict(), cd_best_path)

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
    train()
