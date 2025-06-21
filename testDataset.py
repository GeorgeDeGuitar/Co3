import torch
from DemDataset2 import DemDataset, fast_collate_fn
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import Subset, random_split
import gc
import matplotlib.pyplot as plt
from loss import smooth_mask_generate

json_dir = r"E:\KingCrimson Dataset\Simulate\data0\json"  # 局部json
array_dir = r"E:\KingCrimson Dataset\Simulate\data0\arraynmask\array"  # 全局array
mask_dir = r"E:\KingCrimson Dataset\Simulate\data0\arraynmask\mask"  # 全局mask
target_dir = (
    r"E:\KingCrimson Dataset\Simulate\data0\groundtruthstatus\statusarray" # 全局target
)
batch_size = 8  # 批次大小
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 内存管理工具函数
def cleanup_memory():
    """强制清理内存"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        

def prepare_batch_data(batch, device):
    """
    Prepare batch data by moving tensors to the specified device.
    优化：使用non_blocking转移和更好的内存管理
    """
    try:
        (
            local_inputs,
            local_masks,
            local_targets,
            global_inputs,
            global_masks,
            global_targets,
            metadata,
        ) = batch

        # 使用non_blocking转移提高效率
        local_inputs = local_inputs.to(device, dtype=torch.float32, non_blocking=True)
        local_masks = local_masks.to(device, dtype=torch.int, non_blocking=True)
        local_targets = local_targets.to(device, dtype=torch.float32, non_blocking=True)
        global_inputs = global_inputs.to(device, dtype=torch.float32, non_blocking=True)
        global_masks = global_masks.to(device, dtype=torch.int, non_blocking=True)
        global_targets = global_targets.to(device, dtype=torch.float32, non_blocking=True)

        return (
            local_inputs,
            local_masks,
            local_targets,
            global_inputs,
            global_masks,
            global_targets,
            metadata,
        )
    except Exception as e:
        print(f"准备批次数据时发生错误: {e}")
        cleanup_memory()
        raise
    
try:
    dataset = DemDataset(
        json_dir, array_dir, mask_dir, target_dir,
        min_valid_pixels=20, 
        max_valid_pixels=900,
        enable_synthetic_masks=True,  # 启用生成式mask
        synthetic_ratio=1.0,          # 每个原始数据生成1个合成数据
        analyze_data=False             # 分析数据质量
    )
    full_dataset = dataset
    print(f"数据集加载成功，总数据量: {len(full_dataset)}")
    print(f"  - 原始数据: {dataset.original_length}")
    if dataset.enable_synthetic_masks:
        print(f"  - 生成式数据: {dataset.synthetic_length}")
except Exception as e:
    print(f"加载数据集失败: {e}")
    raise

def simple_dataset_split(dataset, test_ratio=0.1, val_ratio=0.2, seed=42):
    """
    简洁的数据集分割方法
    
    Args:
        dataset: 完整数据集
        test_ratio: 测试集比例（从前面取）
        val_ratio: 验证集比例（从剩余数据中取）
        seed: 随机种子，确保可重现
        
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    total_size = len(dataset)
    test_size = int(total_size * test_ratio)
    
    # Step 1: 前test_ratio%作为测试集（固定不变）
    test_indices = list(range(test_size))
    remaining_indices = list(range(test_size, total_size))
    
    # Step 2: 剩余数据shuffle后分为训练集和验证集
    np.random.seed(seed)
    np.random.shuffle(remaining_indices)
    
    remaining_size = len(remaining_indices)
    val_size = int(remaining_size * val_ratio)
    
    val_indices = remaining_indices[:val_size]
    train_indices = remaining_indices[val_size:]
    
    # 创建子数据集
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    print(f"数据集分割完成:")
    print(f"  总数据: {total_size}")
    print(f"  测试集: {len(test_indices)} ({len(test_indices)/total_size:.1%}) - 索引 0 到 {test_size-1}")
    print(f"  验证集: {len(val_indices)} ({len(val_indices)/total_size:.1%})")
    print(f"  训练集: {len(train_indices)} ({len(train_indices)/total_size:.1%})")
    
    return train_dataset, val_dataset, test_dataset

if __name__ == '__main__':
    # ==================== 简洁的数据分割（仅需这几行）====================
    train_dataset, val_dataset, test_dataset = simple_dataset_split(
        full_dataset, 
        test_ratio=0.1,    # 10% 测试集
        val_ratio=0.1,    # 25% 验证集（从剩余90%中取）
        seed=42            # 固定种子确保可重现
    )

    # Create data loaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=fast_collate_fn,  # 使用优化的collate函数
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # 可视化 prepare_batch_data 提取的每个 batch 的 local/global 输入/目标/掩码六类数据（前6个样本）
    if True:
        for batch_idx, batch in enumerate(train_loader):
            try:
                (
                    local_inputs,
                    local_masks,
                    local_targets,
                    global_inputs,
                    global_masks,
                    global_targets,
                    metadata,
                ) = prepare_batch_data(batch, device)

                num_samples = min(6, local_inputs.shape[0])
                data_list = [
                    local_inputs[:num_samples],
                    local_masks[:num_samples],
                    local_targets[:num_samples],
                    global_inputs[:num_samples],
                    global_masks[:num_samples],
                    global_targets[:num_samples],
                ]
                titles = [
                    "local_inputs",
                    "local_masks",
                    "local_targets",
                    "global_inputs",
                    "global_masks",
                    "global_targets",
                ]

                fig, axes = plt.subplots(6, num_samples, figsize=(num_samples * 3, 18))
                for row in range(6):
                    for col in range(num_samples):
                        ax = axes[row, col]
                        img = data_list[row][col].detach().cpu().squeeze().numpy()
                        if "mask" in titles[row]:
                            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
                        else:
                            ax.imshow(img, cmap='jet')
                        if row == 0:
                            ax.set_title(f"Sample {col}")
                        if col == 0:
                            ax.set_ylabel(titles[row])
                        ax.axis('off')
                plt.tight_layout()
                plt.show()
                print(f"已显示第 {batch_idx+1} 个 batch，可视化 {num_samples} 个样本。")
                smooth_mask_generate(local_masks, sigma=0.5,visualize=True)
                input("按回车显示下一个 batch，Ctrl+C 可中断...")

            except Exception as e:
                print(f"批次数据准备或可视化失败: {e}")
                cleanup_memory()
                raise