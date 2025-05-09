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
from loss import completion_network_loss
import torch.multiprocessing as mp

# Import your models
from models import CompletionNetwork, ContextDiscriminator

# Import your custom dataset
from DemDataset import DemDataset
# from DEMData import DEMData

# Import the custom_collate_fn function from your dataset module
from DemDataset import custom_collate_fn
# from DEMData import custom_collate_fn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import gc

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# setx PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:98

'''def setup(rank, world_size):
    # 初始化进程组
    # 设置环境变量明确禁用libuv
    os.environ["TORCH_DISTRIBUTED_USE_LIBUV"] = "0"
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    #dist.init_process_group("gloo", rank=rank, world_size=world_size)
    #torch.cuda.set_device(rank)
    #dist.init_process_group("gloo", rank=rank, world_size=world_size, init_method="env://")
    #torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        rank=rank,
        world_size=world_size,
        #timeout=timedelta(seconds=1800),
        use_libuv=False  # 禁用 libuv
    )
    torch.cuda.set_device(rank)'''
def setup(rank, world_size):
    import os
    from datetime import timedelta
    import getpass

    # 使用用户主目录
    user = getpass.getuser()
    tmp_dir = f"C:/Users/{user}/tmp"
    shared_file = f"file:///C:/Users/{user}/tmp/sharedfile"
    file_path = f"C:/Users/{user}/tmp/sharedfile"

    # 仅 rank 0 创建目录和清理旧文件
    if rank == 0:
        os.makedirs(tmp_dir, exist_ok=True)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except PermissionError:
                print(f"Warning: Could not delete {file_path}")
    
    # 使用 barrier 同步
    dist.init_process_group(
        backend="gloo",
        init_method=shared_file,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=1800)
    )
    if rank == 0:
        print(f"Initialized with shared file: {file_path}")
    dist.barrier()  # 确保所有进程同步
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

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

def train(rank, world_size):
    # =================================================
    # Configuration settings (replace with your settings)
    # =================================================
    json_dir = r"E:\KingCrimson Dataset\Simulate\data0\testjson"  # 局部json
    array_dir = r"E:\KingCrimson Dataset\Simulate\data0\arraynmask\array"  # 全局array
    mask_dir = r"E:\KingCrimson Dataset\Simulate\data0\arraynmask\mask"  # 全局mask
    target_dir = (
        r"E:\KingCrimson Dataset\Simulate\data0\groundtruthstatus\statusarray" # 全局target
    )
    result_dir = r"E:\KingCrimson Dataset\Simulate\data0\results"  # 结果保存路径
    cache_dir = r"E:\KingCrimson Dataset\Simulate\data0\traindata"  # 缓存目录

    # Training parameters
    steps_1 = 10000  # Phase 1 training steps
    steps_2 = 10000  # Phase 2 training steps
    steps_3 = 10000  # Phase 3 training steps

    snaperiod_1 = 500   # How often to save snapshots in phase 1
    snaperiod_2 = 500   # How often to save snapshots in phase 2
    snaperiod_3 = 500   # How often to save snapshots in phase 3

    # 全局批量大小需要除以GPU数量
    batch_size = 16 // world_size  # 每个GPU的批次大小
    alpha = 4e-4  # Alpha parameter for loss weighting
    validation_split = 0.1  # Percentage of data to use for validation

    # 初始化分布式环境
    setup(rank, world_size)
    
    # 将模型移到对应GPU
    device = torch.device(f"cuda:{rank}")

    # Initialize Visdom (只在主进程)
    if rank == 0:
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
    else:
        viz = None
        loss_windows = None

    # =================================================
    # Preparation
    # =================================================
    # Create result directories (只在主进程)
    if rank == 0:
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        for phase in ["phase_1", "phase_2", "phase_3", "validation"]:
            if not os.path.exists(os.path.join(result_dir, phase)):
                os.makedirs(os.path.join(result_dir, phase))

    # 确保所有进程同步
    dist.barrier()

    gc.collect()
    torch.cuda.empty_cache()  # 如果使用GPU
    
    # Load dataset
    dataset = DemDataset(json_dir, array_dir, mask_dir, target_dir)
    full_dataset = dataset  # 直接使用完整数据集

    # Split dataset into train and validation
    dataset_size = len(full_dataset)
    val_size = int(validation_split * dataset_size)
    train_size = dataset_size - val_size

    # 使用相同的种子以确保所有进程拥有相同的数据划分
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=generator,  # For reproducibility
    )

    # 创建分布式采样器
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=42
    )
    
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=42
    )

    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,  # 每个GPU的批次大小
        shuffle=False,  # 使用分布式采样器时不要设置shuffle
        sampler=train_sampler,
        collate_fn=custom_collate_fn,
        num_workers=4,  # 减少每个进程的workers数
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        sampler=val_sampler,
        collate_fn=custom_collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    # 验证子集仅在主进程创建
    val_subset_loader = None
    if rank == 0:
        val_subset = torch.utils.data.Subset(val_dataset, range(0, 30))
        val_subset_loader = DataLoader(
            val_subset, 
            batch_size=batch_size, 
            shuffle=True,  # 不使用分布式采样器
            collate_fn=custom_collate_fn,
            num_workers=4,
            pin_memory=True
        )

    if rank == 0:
        print(f"Training dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
        print(f"Per GPU batch size: {batch_size}")

    alpha = torch.tensor(alpha, dtype=torch.float32).to(device)

    # =================================================
    # Setup Models
    # =================================================
    # Initialize the Completion Network - adjust input channels as needed
    model_cn = CompletionNetwork(input_channels=1).to(device)
    
    # Initialize the Context Discriminator
    model_cd = ContextDiscriminator(
        local_input_channels=1,  # For elevation data
        local_input_size=33,  # The local patch size from your dataset
        global_input_channels=1,  # For elevation data
        global_input_size=600,  # The global image size from your dataset
    ).to(device)
    
    # 包装模型用于分布式训练
    model_cn = DDP(model_cn, device_ids=[rank])
    model_cd = DDP(model_cd, device_ids=[rank])

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
        注意：仅在主进程(rank=0)中调用此函数
        """
        if rank != 0 or viz is None:
            return  # 仅在主进程可视化

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
            np_array = tensor.detach().cpu().numpy()  # 确保是CPU张量
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
                colored_img = apply_colormap(tensor_batch[j])
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
                colored_img = apply_colormap(completed[i])
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

    # Helper function to evaluate model on validation set (只在主进程使用)
    def validate(model_cn, val_subset_loader, device):
        if rank != 0 or val_subset_loader is None:
            return 0.0  # 非主进程返回占位值
            
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
    if rank == 0:
        print("Starting Phase 1 training...")
    model_cn.train()
    
    # 创建进度条 (只在主进程)
    if rank == 0:
        pbar = tqdm(total=steps_1)
    else:
        pbar = None

    step = 0
    best_val_loss = float("inf")

    # 修改：使用 iterations 而不是 epochs，创建迭代器以便重复使用
    train_iterator = iter(train_loader)

    while step < steps_1:
        # 设置采样器epoch以改变数据顺序
        train_sampler.set_epoch(step // len(train_loader) + 1)
        
        try:
            batch = next(train_iterator)
        except StopIteration:
            # 重置迭代器
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
            
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

        # 在所有GPU上同步损失值
        loss_tensor = torch.tensor([loss.item()], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        loss_tensor /= world_size

        # Update Visdom plot for training loss (只在主进程)
        if rank == 0:
            viz.line(
                Y=torch.tensor([loss_tensor.item()]) if torch.tensor([loss_tensor.item()]) < 1000 else torch.tensor([-1]),
                X=torch.tensor([step]),
                win=loss_windows["phase1_train"],
                update="append",
            )
            pbar.set_description(f"Phase 1 | train loss: {loss_tensor.item():.5f}")
            pbar.update(1)

        step += 1

        # Testing and saving snapshots
        if step % snaperiod_1 == 0:
            # 确保所有进程同步
            dist.barrier()
            
            # Run validation (只在主进程)
            val_loss = validate(model_cn, val_subset_loader, device)
            
            # 同步验证损失到所有进程
            val_loss_tensor = torch.tensor([val_loss], device=device)
            dist.broadcast(val_loss_tensor, 0)
            val_loss = val_loss_tensor.item()
            
            if rank == 0:
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
                    # 保存未包装的模型
                    torch.save(model_cn.module.state_dict(), best_model_path)

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
                    torch.save(model_cn.module.state_dict(), model_path)

                model_cn.train()

        if step >= steps_1:
            break

    if rank == 0:
        pbar.close()

    # =================================================
    # Training Phase 2: Train Context Discriminator only
    # =================================================
    if rank == 0:
        print("Starting Phase 2 training...")
        pbar = tqdm(total=steps_2)
    
    step = 0
    best_acc = float("-inf")
    
    # 重置迭代器
    train_iterator = iter(train_loader)

    while step < steps_2:
        # 设置采样器epoch
        train_sampler.set_epoch(step // len(train_loader) + 1000)  # 使用不同的基数
        
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
            
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

        # 同步损失
        loss_tensor = torch.tensor([loss.item()], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        loss_tensor /= world_size

        # Update Visdom plot for discriminator loss
        if rank == 0:
            viz.line(
                Y=torch.tensor([loss_tensor.item()]),
                X=torch.tensor([step]),
                win=loss_windows["phase2_disc"],
                update="append",
            )
            pbar.set_description(f"Phase 2 | train loss: {loss_tensor.item():.5f}")
            pbar.update(1)

        step += 1

        # Testing and saving snapshots
        if step % snaperiod_2 == 0:
            # 确保所有进程同步
            dist.barrier()
            
            # Calculate discriminator accuracy (在主进程)
            if rank == 0:
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

                        val_real_global_embedded, val_real_global_mask_embedded, _ = (
                            merge_local_to_global(
                                val_local_targets,
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
                torch.save(model_cd.module.state_dict(), model_path)

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
                    torch.save(model_cd.module.state_dict(), best_model_path)

                model_cd.train()
                
            # 广播最佳精度到所有进程
            best_acc_tensor = torch.tensor([best_acc], device=device)
            dist.broadcast(best_acc_tensor, 0)
            best_acc = best_acc_tensor.item()

        if step >= steps_2:
            break

    if rank == 0:
        pbar.close()

    # =================================================
    # Training Phase 3: Joint training of both networks
    # =================================================
    if rank == 0:
        print("Starting Phase 3 training...")
        torch.cuda.empty_cache()
        pbar = tqdm(total=steps_3)
    
    step = 0
    best_val_loss_joint = float("inf")
    
    # 重置迭代器
    train_iterator = iter(train_loader)

    while step < steps_3:
        # 设置采样器epoch
        train_sampler.set_epoch(step // len(train_loader) + 2000)  # 使用不同的基数
        
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
            
        # Prepare batch data
        torch.cuda.empty_cache()
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

        # 同步判别器损失
        loss_cd_tensor = torch.tensor([loss_cd.item()], device=device)
        dist.all_reduce(loss_cd_tensor, op=dist.ReduceOp.SUM)
        loss_cd_tensor /= world_size

        # Update Visdom plot for discriminator loss (只在主进程)
        if rank == 0:
            viz.line(
                Y=torch.tensor([loss_cd_tensor.item()]),
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

        # 同步生成器损失
        loss_cn_tensor = torch.tensor([loss_cn.item()], device=device)
        dist.all_reduce(loss_cn_tensor, op=dist.ReduceOp.SUM)
        loss_cn_tensor /= world_size

        # Update Visdom plot for generator loss (只在主进程)
        if rank == 0:
            viz.line(
                Y=torch.tensor([loss_cn_tensor.item()]),
                X=torch.tensor([step]),
                win=loss_windows["phase3_gen"],
                update="append",
            )
            pbar.set_description(
                f"Phase 3 | D loss: {loss_cd_tensor.item():.5f}, G loss: {loss_cn_tensor.item():.5f}"
            )
            pbar.update(1)

        step += 1

        # Testing and saving snapshots
        if step % snaperiod_3 == 0:
            # 确保所有进程同步
            dist.barrier()
            
            # Run validation (只在主进程)
            if rank == 0:
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
                    torch.save(model_cn.module.state_dict(), cn_best_path)
                    torch.save(model_cd.module.state_dict(), cd_best_path)

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
                torch.save(model_cn.module.state_dict(), cn_path)
                torch.save(model_cd.module.state_dict(), cd_path)

                model_cn.train()
                model_cd.train()
                
            # 广播最佳验证损失到所有进程
            best_val_loss_joint_tensor = torch.tensor([best_val_loss_joint], device=device)
            dist.broadcast(best_val_loss_joint_tensor, 0)
            best_val_loss_joint = best_val_loss_joint_tensor.item()

        if step >= steps_3:
            break

    if rank == 0:
        pbar.close()

    # 最终可视化：比较每个阶段的最佳模型（仅在主进程）
    if rank == 0:
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

        # 加载每个阶段的最佳模型
        # 从阶段1加载最佳模型
        best_model_p1 = CompletionNetwork(input_channels=1).to(device)
        best_model_p1.load_state_dict(
            torch.load(os.path.join(result_dir, "phase_1", "model_cn_best"))
        )
        best_model_p1.eval()

        # 从阶段3加载最佳模型（联合训练）
        best_model_p3 = CompletionNetwork(input_channels=1).to(device)
        best_model_p3.load_state_dict(
            torch.load(os.path.join(result_dir, "phase_3", "model_cn_best"))
        )
        best_model_p3.eval()

        with torch.no_grad():
            # 使用最佳模型生成补全结果
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

            # 将输出合并到全局用于可视化
            completed_p1, _, _ = merge_local_to_global(
                output_p1, test_global_inputs, test_global_masks, metadata
            )

            completed_p3, _, _ = merge_local_to_global(
                output_p3, test_global_inputs, test_global_masks, metadata
            )

            # 创建比较网格
            comparison_grid = [
                test_global_inputs.cpu(),  # 输入
                test_global_targets.cpu(),  # 目标
                completed_p1.cpu(),        # 阶段1完成
                completed_p3.cpu(),        # 阶段3完成
            ]

            # 在Visdom中可视化比较
            # 使用自定义的高程图可视化
            def visualize_comparison(comparison_tensors, labels):
                """使用高程图着色方案可视化比较"""
                # 定义高程图着色函数
                def apply_colormap(tensor):
                    """将张量转换为带有高程图着色的PIL图像"""
                    # 确保输入是2D张量(单通道图像)或取第一个通道
                    if tensor.dim() > 2:
                        if tensor.size(0) == 1:
                            tensor = tensor.squeeze(0)  # 单通道图像
                        else:
                            tensor = tensor[0]  # 取第一个通道

                    # 转换为numpy数组并归一化到0-1范围
                    np_array = tensor.detach().cpu().numpy()
                    min_val = np_array.min()
                    max_val = np_array.max()
                    norm_array = (np_array - min_val) / (max_val - min_val + 1e-8)

                    # 应用matplotlib的jet色彩映射
                    colored_array = plt.cm.jet(norm_array)

                    # 转换为PIL图像 (colored_array是RGBA格式)
                    colored_array_rgb = (colored_array[:, :, :3] * 255).astype(np.uint8)
                    colored_image = Image.fromarray(colored_array_rgb)

                    return colored_image

                num_samples = min(2, len(comparison_tensors[0]))
                num_models = len(comparison_tensors)
                
                # 为每个模型创建彩色图像
                colored_images = []
                for model_idx in range(num_models):
                    for sample_idx in range(num_samples):
                        colored_img = apply_colormap(comparison_tensors[model_idx][sample_idx])
                        # 缩小图像以便显示
                        colored_img = colored_img.resize(
                            (colored_img.width // 2, colored_img.height // 2), 
                            Image.Resampling.LANCZOS
                        )
                        colored_images.append(colored_img)
                
                # 创建网格布局
                cols = num_samples
                rows = num_models
                grid_width = (colored_images[0].width + 10) * cols
                grid_height = (colored_images[0].height + 10) * rows + 30  # 额外空间用于标签
                grid_img = Image.new("RGB", (grid_width, grid_height), color=(255, 255, 255))
                
                # 添加标签
                draw = ImageDraw.Draw(grid_img)
                
                # 填充网格
                for i in range(rows):
                    # 添加行标签
                    draw.text((5, i * (colored_images[0].height + 10) + colored_images[0].height // 2), 
                              labels[i], fill=(0, 0, 0))
                    
                    for j in range(cols):
                        idx = i * cols + j
                        x = j * (colored_images[idx].width + 10)
                        y = i * (colored_images[idx].height + 10)
                        grid_img.paste(colored_images[idx], (x, y))
                
                return grid_img
            
            # 创建并显示比较图
            comparison_image = visualize_comparison(
                comparison_grid,
                ["输入", "目标", "阶段1完成", "阶段3完成"]
            )
            
            # 在Visdom中显示
            viz.image(
                np.array(comparison_image).transpose(2, 0, 1),
                opts=dict(
                    caption="模型比较",
                    title="最佳模型比较"
                ),
                win="best_models_comparison",
            )

        print("训练完成!")
    
    # 清理分布式环境
    cleanup()

def main():
    world_size = torch.cuda.device_count()
    print(f"使用 {world_size} 个GPU进行分布式训练")
    
    # 使用多进程启动
    mp.spawn(
        train,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()