import os
import numpy as np
import torch
from pathlib import Path
import pickle
from typing import Dict, List, Tuple
import time
from tqdm import tqdm
from data_management.SytheticDataset import MultiScaleDEMDataset


class PreGeneratedMaskDataset(MultiScaleDEMDataset):
    """使用预生成mask的优化数据集"""

    def __init__(
        self,
        reference_dir: str,
        mask_cache_dir: str,
        local_sizes: List[int] = [257],
        samples_per_file: int = 10,
        global_coverage_range: Tuple[float, float] = (0.3, 1.0),
        local_missing_range: Tuple[float, float] = (0.3, 0.7),
        num_global_masks: int = 300,
        num_local_masks: int = 300,
        regenerate_masks: bool = False,
        **kwargs,
    ):
        # 先初始化父类
        super().__init__(
            reference_dir=reference_dir,
            local_sizes=local_sizes,
            samples_per_file=samples_per_file,
            global_coverage_range=global_coverage_range,
            local_missing_range=local_missing_range,
            **kwargs,
        )

        self.mask_cache_dir = Path(mask_cache_dir)
        self.mask_cache_dir.mkdir(parents=True, exist_ok=True)

        self.num_global_masks = num_global_masks
        self.num_local_masks = num_local_masks

        # 加载或生成mask
        self._load_or_generate_masks(regenerate_masks)

    def _load_or_generate_masks(self, regenerate: bool):
        """加载或生成预缓存的mask"""
        global_mask_file = self.mask_cache_dir / "global_masks.pkl"
        local_mask_file = self.mask_cache_dir / "local_masks.pkl"

        if not regenerate and global_mask_file.exists() and local_mask_file.exists():
            print(" 加载预生成的mask...")
            with open(global_mask_file, "rb") as f:
                self.global_masks = pickle.load(f)
            with open(local_mask_file, "rb") as f:
                self.local_masks = pickle.load(f)
            print(
                f" 加载完成: {len(self.global_masks)} 个global masks, {len(self.local_masks)} 个local masks"
            )
        else:
            print(" 生成新的mask模板...")
            self._generate_mask_templates()
            # 保存
            with open(global_mask_file, "wb") as f:
                pickle.dump(self.global_masks, f)
            with open(local_mask_file, "wb") as f:
                pickle.dump(self.local_masks, f)
            print(" Mask模板生成并保存完成")

    def _generate_mask_templates(self):
        """生成mask模板"""
        # 生成global masks
        print(f"生成 {self.num_global_masks} 个global mask模板...")
        self.global_masks = []

        # 创建一个标准的initial mask (600x600)
        standard_shape = (600, 600)
        standard_initial_mask = np.ones(standard_shape, dtype=np.float32)

        for i in tqdm(range(self.num_global_masks), desc="Global masks"):
            # 随机选择覆盖率
            coverage_ratio = np.random.uniform(*self.global_coverage_range)
            # 生成global mask
            global_mask = self._generate_global_mask(
                standard_initial_mask, coverage_ratio
            )

            self.global_masks.append(
                {
                    "mask": global_mask,
                    "coverage_ratio": coverage_ratio,
                    "shape": standard_shape,
                }
            )

        # 生成local masks (每个尺寸都生成一些)
        print(f"生成 {self.num_local_masks} 个local mask模板...")
        self.local_masks = {size: [] for size in self.local_sizes}

        masks_per_size = self.num_local_masks // len(self.local_sizes)

        for size in self.local_sizes:
            for i in tqdm(range(masks_per_size), desc=f"Local masks {size}x{size}"):
                # 随机选择缺失率
                missing_ratio = np.random.uniform(*self.local_missing_range)
                # 生成local mask
                local_mask = self._generate_local_mask(
                    size, missing_ratio, self.enable_edge_connection
                )

                self.local_masks[size].append(
                    {"mask": local_mask, "missing_ratio": missing_ratio, "size": size}
                )

    def _apply_global_mask_to_initial(
        self, initial_mask: np.ndarray, global_mask_template: np.ndarray
    ) -> np.ndarray:
        """将预生成的global mask应用到initial mask上"""
        # 确保形状匹配
        if initial_mask.shape != global_mask_template.shape:
            raise ValueError(
                f"Shape mismatch: {initial_mask.shape} vs {global_mask_template.shape}"
            )

        # 取交集：只在initial_mask为1的地方应用global_mask
        return initial_mask * global_mask_template

    def _find_valid_local_region(
        self, global_mask: np.ndarray, local_size: int, min_valid_ratio: float = 0.95
    ):
        """快速找到有效的local区域"""
        h, w = global_mask.shape
        half_size = local_size // 2

        # 使用卷积快速计算每个位置的有效比例
        from scipy.ndimage import uniform_filter

        # 计算滑动窗口内的平均值（即有效比例）
        kernel_size = local_size
        valid_ratio_map = uniform_filter(
            global_mask.astype(np.float32), size=kernel_size, mode="constant"
        )

        # 找到满足条件的所有位置
        valid_y, valid_x = np.where(valid_ratio_map >= min_valid_ratio)

        # 过滤边界位置
        valid_positions = []
        for y, x in zip(valid_y, valid_x):
            if half_size <= y < h - half_size and half_size <= x < w - half_size:
                valid_positions.append((y, x))

        if not valid_positions and min_valid_ratio > 0.8:
            # 递归降低要求
            return self._find_valid_local_region(
                global_mask, local_size, min_valid_ratio - 0.1
            )

        return valid_positions

    def _generate_single_sample_optimized(
        self, reference_data: np.ndarray, sample_idx: int
    ) -> Dict:
        """优化的样本生成"""
        # 1. 生成initial mask
        initial_mask = self._generate_initial_mask(reference_data)

        # 2. 随机选择一个预生成的global mask
        global_mask_info = np.random.choice(self.global_masks)
        global_mask_template = global_mask_info["mask"]
        coverage_ratio = global_mask_info["coverage_ratio"]

        # 3. 应用global mask到initial mask
        global_mask = self._apply_global_mask_to_initial(
            initial_mask, global_mask_template
        )

        # 4. 随机选择local尺寸
        local_size = np.random.choice(self.local_sizes)

        # 5. 快速找到有效的local区域
        valid_positions = self._find_valid_local_region(global_mask, local_size)

        if not valid_positions:
            # 回退到中心位置
            center_y, center_x = (
                reference_data.shape[0] // 2,
                reference_data.shape[1] // 2,
            )
        else:
            center_y, center_x = valid_positions[
                np.random.randint(len(valid_positions))
            ]

        # 6. 提取local区域
        half_size = local_size // 2
        local_target = reference_data[
            center_y - half_size : center_y + half_size + 1,
            center_x - half_size : center_x + half_size + 1,
        ].copy()

        local_global_mask = global_mask[
            center_y - half_size : center_y + half_size + 1,
            center_x - half_size : center_x + half_size + 1,
        ].copy()

        # 7. 随机选择一个预生成的local mask
        local_mask_info = np.random.choice(self.local_masks[local_size])
        local_synthetic_mask = local_mask_info["mask"].copy()
        missing_ratio = local_mask_info["missing_ratio"]

        # 8. 合并masks
        local_mask = local_synthetic_mask * local_global_mask

        # 9. 生成inputs（与原方法相同）
        local_input = local_target.copy()
        valid_local_values = local_input[local_mask == 1]
        if len(valid_local_values) > 0:
            local_fill_value = np.mean(valid_local_values)
        else:
            local_fill_value = 0.0
        local_input[local_mask == 0] = local_fill_value

        # 10. 生成global input
        global_input = reference_data.copy()
        global_valid_values = global_input[global_mask == 1]
        if len(global_valid_values) > 0:
            global_fill_value = np.mean(global_valid_values)
            # print(f"global_fill_value: {global_fill_value}")
        else:
            global_fill_value = 0.0
        global_input[global_mask == 0] = global_fill_value

        # 11. 更新global mask和input
        local_start_y = center_y - half_size
        local_end_y = center_y + half_size + 1
        local_start_x = center_x - half_size
        local_end_x = center_x + half_size + 1

        global_mask_updated = global_mask.copy()
        local_missing_mask = local_mask == 0
        global_mask_updated[local_start_y:local_end_y, local_start_x:local_end_x][
            local_missing_mask
        ] = 0

        global_input[local_start_y:local_end_y, local_start_x:local_end_x][
            local_missing_mask
        ] = global_fill_value

        # 12. 生成metadata
        metadata = np.array(
            [
                center_y,
                center_x,
                local_size,
                sample_idx,
                coverage_ratio,
                missing_ratio,
                local_size,
            ],
            dtype=np.float32,
        )

        return {
            "local_input": local_input,
            "local_mask": local_mask,
            "local_target": local_target,
            "global_input": global_input,
            "global_mask": global_mask_updated,
            "global_target": reference_data.copy(),
            "metadata": metadata,
            "local_size": local_size,
            "initial_mask": initial_mask,
        }

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """重写getitem使用优化的生成方法"""
        file_idx = idx // self.samples_per_file
        sample_idx = idx % self.samples_per_file
        file_path = self.reference_files[file_idx]

        try:
            reference_data = self._load_reference_data(file_path)
            # 使用优化的生成方法
            sample = self._generate_single_sample_optimized(reference_data, sample_idx)

            return (
                torch.tensor(sample["local_input"], dtype=torch.float32).unsqueeze(0),
                torch.tensor(sample["local_mask"], dtype=torch.float32).unsqueeze(0),
                torch.tensor(sample["local_target"], dtype=torch.float32),
                torch.tensor(sample["global_input"], dtype=torch.float32).unsqueeze(0),
                torch.tensor(sample["global_mask"], dtype=torch.float32).unsqueeze(0),
                torch.tensor(sample["global_target"], dtype=torch.float32),
                sample["metadata"].astype(np.int32),
            )
        except Exception as e:
            print(f" 生成样本失败 (idx={idx}): {e}")
            import traceback

            traceback.print_exc()
            return self._get_fallback_sample()


def benchmark_generation():
    """基准测试对比生成速度"""
    config = {
        "reference_dir": r"F:\Dataset\Simulate\data1\ST2\statusarray",
        "local_sizes": [257],
        "samples_per_file": 10,
        "global_coverage_range": (0.5, 1.0),
        "local_missing_range": (0.3, 0.7),
        "seed": 42,
        "visualization_samples": 0,
    }

    # 测试原始版本
    print("\n 测试原始数据集生成速度...")
    original_dataset = MultiScaleDEMDataset(**config)

    start_time = time.time()
    for i in range(10):
        _ = original_dataset[i]
    original_time = time.time() - start_time
    print(f"原始版本: {original_time/10:.2f} 秒/样本")

    # 测试优化版本
    print("\n 测试优化数据集生成速度...")
    optimized_dataset = PreGeneratedMaskDataset(
        **config,
        mask_cache_dir="./mask_cache",
        num_global_masks=300,
        num_local_masks=300,
        regenerate_masks=False,  # 第二次运行时会加载缓存
    )

    start_time = time.time()
    for i in range(10):
        _ = optimized_dataset[i]
    optimized_time = time.time() - start_time
    print(f"优化版本: {optimized_time/10:.2f} 秒/样本")

    print(f"\n 速度提升: {original_time/optimized_time:.1f}x")


import matplotlib.pyplot as plt
import time
import numpy as np
from torch.utils.data import DataLoader


def visualize_and_benchmark_dataset(dataset, num_samples=5, display_time=1.0):
    """可视化数据集样本并测试加载速度"""

    print(f"\n 测试数据集加载性能...")
    print(f"   总样本数: {len(dataset)}")
    print(f"   将可视化 {num_samples} 个样本")

    # 性能统计
    load_times = []

    # 创建图形
    fig = plt.figure(figsize=(20, 4))

    for sample_idx in range(num_samples):
        # 记录加载时间
        start_time = time.time()

        # 加载数据
        data = dataset[sample_idx]

        load_time = time.time() - start_time
        load_times.append(load_time)

        # 解包数据
        (
            local_input,
            local_mask,
            local_target,
            global_input,
            global_mask,
            global_target,
            metadata,
        ) = data

        # 转换为numpy用于显示
        local_input_np = local_input.squeeze().numpy()
        local_mask_np = local_mask.squeeze().numpy()
        local_target_np = local_target.numpy()
        global_input_np = global_input.squeeze().numpy()
        global_mask_np = global_mask.squeeze().numpy()
        global_target_np = global_target.numpy()

        # 提取metadata信息
        center_y, center_x = int(metadata[0]), int(metadata[1])
        local_size = int(metadata[2])
        coverage_ratio = metadata[4]
        missing_ratio = metadata[5]

        # 清除之前的图
        plt.clf()

        # 创建子图
        axes = fig.subplots(1, 6)

        # 计算显示范围
        vmin = np.percentile(global_target_np[global_target_np != 0], 2)
        vmax = np.percentile(global_target_np[global_target_np != 0], 98)

        # 1. Global Target
        im1 = axes[0].imshow(global_target_np, cmap="terrain", vmin=vmin, vmax=vmax)
        axes[0].set_title(f"Global Target\n{global_target_np.shape}")
        # 标记local区域
        half_size = local_size // 2
        rect = plt.Rectangle(
            (center_x - half_size, center_y - half_size),
            local_size,
            local_size,
            fill=False,
            color="red",
            linewidth=2,
        )
        axes[0].add_patch(rect)
        axes[0].plot(center_x, center_y, "r+", markersize=10)

        # 2. Global Mask
        im2 = axes[1].imshow(global_mask_np, cmap="binary")
        axes[1].set_title(f"Global Mask\n覆盖率: {coverage_ratio:.2%}")

        # 3. Global Input (Masked)
        global_input_vis = global_input_np.copy()
        global_input_vis[global_mask_np == 0] = np.nan
        im3 = axes[2].imshow(global_input_vis, cmap="terrain", vmin=vmin, vmax=vmax)
        axes[2].set_title("Global Input")

        # 4. Local Target
        im4 = axes[3].imshow(local_target_np, cmap="terrain", vmin=vmin, vmax=vmax)
        axes[3].set_title(f"Local Target\n{local_size}×{local_size}")

        # 5. Local Mask
        im5 = axes[4].imshow(local_mask_np, cmap="binary")
        actual_missing = 1 - np.mean(local_mask_np)
        axes[4].set_title(f"Local Mask\n缺失率: {actual_missing:.2%}")

        # 6. Local Input (Masked)
        local_input_vis = local_input_np.copy()
        local_input_vis[local_mask_np == 0] = np.nan
        im6 = axes[5].imshow(local_input_vis, cmap="terrain", vmin=vmin, vmax=vmax)
        axes[5].set_title("Local Input")

        # 移除坐标轴
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        # 添加总标题
        fig.suptitle(
            f"样本 {sample_idx} - 加载时间: {load_time*1000:.1f}ms", fontsize=16
        )

        plt.tight_layout()
        plt.pause(display_time)  # 显示指定时间

        print(f"✓ 样本 {sample_idx}: {load_time*1000:.1f}ms 加载时间")

    plt.close()

    # 统计信息
    print(f"\n 加载性能统计:")
    print(f"   平均加载时间: {np.mean(load_times)*1000:.2f}ms")
    print(f"   最快加载时间: {np.min(load_times)*1000:.2f}ms")
    print(f"   最慢加载时间: {np.max(load_times)*1000:.2f}ms")
    print(f"   标准差: {np.std(load_times)*1000:.2f}ms")

    # 测试批量加载
    print(f"\n 测试DataLoader批量加载...")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

    batch_times = []
    for i, batch in enumerate(dataloader):
        if i >= 5:  # 只测试5个batch
            break
        start_time = time.time()
        # 解包数据
        (
            local_inputs,
            local_masks,
            local_targets,
            global_inputs,
            global_masks,
            global_targets,
            metadata,
        ) = batch
        batch_time = time.time() - start_time
        batch_times.append(batch_time)
        print(f"   Batch {i}: {batch_time*1000:.2f}ms, 形状: {local_inputs.shape}")

    print(f"\n   平均batch时间: {np.mean(batch_times)*1000:.2f}ms")

    return load_times


def test_mask_diversity(dataset, num_tests=100):
    """测试mask的多样性"""
    print(f"\n 测试mask多样性 (采样{num_tests}个)...")

    coverage_ratios = []
    missing_ratios = []

    for i in range(num_tests):
        data = dataset[i]
        metadata = data[6]
        coverage_ratios.append(metadata[4])
        missing_ratios.append(metadata[5])

    print(f"\n Global覆盖率分布:")
    print(f"   范围: [{np.min(coverage_ratios):.3f}, {np.max(coverage_ratios):.3f}]")
    print(f"   均值: {np.mean(coverage_ratios):.3f}")
    print(f"   标准差: {np.std(coverage_ratios):.3f}")

    print(f"\n Local缺失率分布:")
    print(f"   范围: [{np.min(missing_ratios):.3f}, {np.max(missing_ratios):.3f}]")
    print(f"   均值: {np.mean(missing_ratios):.3f}")
    print(f"   标准差: {np.std(missing_ratios):.3f}")

    # 绘制分布图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.hist(coverage_ratios, bins=20, alpha=0.7, color="blue", edgecolor="black")
    ax1.set_xlabel("Global Coverage Ratio")
    ax1.set_ylabel("Count")
    ax1.set_title("Global Coverage Distribution")
    ax1.grid(True, alpha=0.3)

    ax2.hist(missing_ratios, bins=20, alpha=0.7, color="red", edgecolor="black")
    ax2.set_xlabel("Local Missing Ratio")
    ax2.set_ylabel("Count")
    ax2.set_title("Local Missing Distribution")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 加载优化的数据集
    print("加载优化数据集...")

    optimized_dataset = PreGeneratedMaskDataset(
        reference_dir=r"e:\KingCrimson Dataset\Simulate\data0\ST2\statusarray",
        mask_cache_dir="./mask_cache_65",
        local_sizes=[65],
        samples_per_file=10,
        global_coverage_range=(0.5, 1.0),
        local_missing_range=(0.3, 0.7),
        num_global_masks=500,
        num_local_masks=500,
        regenerate_masks=True,  # 使用已有的mask
        seed=42,
        cache_reference_files=True,
        visualization_samples=0,
    )

    print(f"   数据集加载完成!")
    print(f"   总样本数: {len(optimized_dataset)}")
    print(f"   Global masks: {len(optimized_dataset.global_masks)}")
    print(f"   Local masks per size: {len(optimized_dataset.local_masks[257])}")

    # 可视化和性能测试
    load_times = visualize_and_benchmark_dataset(
        optimized_dataset,
        num_samples=100,
        display_time=10.0,  # 显示5个样本  # 每个显示1秒
    )

    # 测试mask多样性
    # test_mask_diversity(optimized_dataset, num_tests=100)

    # 对比原始版本（可选）
    print("\n" + "=" * 60)
    print("对比原始版本性能...")
