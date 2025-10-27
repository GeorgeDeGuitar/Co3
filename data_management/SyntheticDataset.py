import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
from typing import Tuple, List, Dict, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    binary_opening,
    binary_closing,
    label,
)

# 科学计算相关导入
try:
    from scipy.ndimage import binary_dilation, binary_erosion, label, binary_closing
    from skimage.morphology import disk
    from skimage.draw import ellipse

    SCIPY_AVAILABLE = True
except ImportError:
    print("警告: scipy/skimage未安装，将使用简化的mask生成方法")
    SCIPY_AVAILABLE = False


class MultiScaleDEMDataset(Dataset):
    """
    多尺寸DEM数据集
    支持从600x600参考文件生成不同尺寸的local数据用于训练
    """

    def __init__(
        self,
        reference_dir: str,
        local_sizes: List[int] = [64, 128, 256],  # 支持多种local尺寸
        samples_per_file: int = 10,  # 每个参考文件生成的样本数
        global_coverage_range: Tuple[float, float] = (
            0.3,
            1.0,
        ),  # global mask覆盖率范围
        local_missing_range: Tuple[float, float] = (0.3, 0.7),  # local mask缺失率范围
        max_edge_missing_ratio: float = 0.3,  # local中包含global边缘的最大比例
        enable_edge_connection: bool = True,  # 是否允许local mask与边缘相接
        edge_connection_prob: float = 0.02,  # 降低到2%的边缘相接概率
        seed: int = 42,
        cache_reference_files: bool = True,  # 是否缓存参考文件
        visualization_samples: int = 0,  # 可视化样本数量(0表示不可视化)
    ):
        """
        Args:
            reference_dir: 参考文件目录，包含600x600的.npy文件
            local_sizes: local数据的尺寸列表，必须为奇数
            samples_per_file: 每个参考文件生成的训练样本数
            global_coverage_range: global mask的覆盖率范围(30%-100%)
            local_missing_range: local mask的缺失率范围
            max_edge_missing_ratio: local中允许包含global边缘的最大比例
            enable_edge_connection: 是否允许local mask与边缘相接
            edge_connection_prob: 与边缘相接的概率(降低到2%)
            seed: 随机种子
            cache_reference_files: 是否预缓存参考文件到内存
            visualization_samples: 可视化的样本数量
        """

        # 参数验证
        for size in local_sizes:
            if size % 2 == 0:
                raise ValueError(f"local_size必须为奇数，当前值: {size}")
            if size >= 600:
                raise ValueError(f"local_size必须小于600，当前值: {size}")

        self.reference_dir = reference_dir
        self.local_sizes = sorted(local_sizes)  # 排序以便一致性
        self.samples_per_file = samples_per_file
        self.global_coverage_range = global_coverage_range
        self.local_missing_range = local_missing_range
        self.max_edge_missing_ratio = max_edge_missing_ratio
        self.enable_edge_connection = enable_edge_connection
        self.edge_connection_prob = edge_connection_prob
        self.cache_reference_files = cache_reference_files
        self.visualization_samples = visualization_samples

        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)

        # 扫描参考文件
        self.reference_files = self._scan_reference_files()

        # 计算总数据量
        self.total_samples = len(self.reference_files) * samples_per_file

        # 预缓存参考文件(可选)
        self.reference_cache = {}
        if cache_reference_files:
            self._cache_reference_files()

        print(f"🎯 多尺寸DEM数据集初始化完成:")
        print(f"   参考文件: {len(self.reference_files)} 个")
        print(f"   Local尺寸: {self.local_sizes}")
        print(f"   每文件样本数: {samples_per_file}")
        print(f"   总样本数: {self.total_samples}")
        print(
            f"   Global覆盖率: {global_coverage_range[0]:.1%} - {global_coverage_range[1]:.1%}"
        )
        print(
            f"   Local缺失率: {local_missing_range[0]:.1%} - {local_missing_range[1]:.1%}"
        )
        print(f"   边缘相接概率: {edge_connection_prob:.1%}")

        # 可视化部分样本
        """if visualization_samples > 0:
            self._visualize_samples(visualization_samples)"""

    def _scan_reference_files(self) -> List[str]:
        """扫描参考文件目录"""
        reference_files = []

        if not os.path.exists(self.reference_dir):
            raise ValueError(f"参考目录不存在: {self.reference_dir}")

        for file_path in Path(self.reference_dir).glob("*.npy"):
            try:
                # 快速验证文件格式
                data = np.load(file_path, mmap_mode="r")
                if data.shape == (600, 600):
                    reference_files.append(str(file_path))
                else:
                    print(f"⚠️ 跳过非600x600文件: {file_path}")
            except Exception as e:
                print(f"⚠️ 跳过无效文件 {file_path}: {e}")

        if len(reference_files) == 0:
            raise ValueError(
                f"在 {self.reference_dir} 中没有找到有效的600x600 .npy文件"
            )

        return sorted(reference_files)

    def _cache_reference_files(self):
        """预缓存参考文件到内存"""
        print("🔄 正在缓存参考文件...")

        for file_path in tqdm(self.reference_files, desc="缓存文件"):
            try:
                data = np.load(file_path).astype(np.float32)
                self.reference_cache[file_path] = data
            except Exception as e:
                print(f"⚠️ 缓存文件失败 {file_path}: {e}")

        print(f"✅ 已缓存 {len(self.reference_cache)} 个文件")

    def _load_reference_data(self, file_path: str) -> np.ndarray:
        """加载参考数据"""
        if self.cache_reference_files and file_path in self.reference_cache:
            return self.reference_cache[file_path].copy()
        else:
            return np.load(file_path).astype(np.float32)

    def _generate_initial_mask(self, data: np.ndarray) -> np.ndarray:
        """生成initial mask (值为0的位置mask=0)"""
        return (data != 0).astype(np.float32)

    def _generate_global_mask(
        self, initial_mask: np.ndarray, coverage_ratio: float
    ) -> np.ndarray:
        """
        改进版global mask生成：提高边缘多样性，确保更自然的边界
        """
        h, w = initial_mask.shape

        # 如果覆盖率接近1.0，直接返回initial_mask
        if coverage_ratio >= 0.95:
            return initial_mask.copy()

        # 计算目标像素数
        initial_valid_pixels = np.sum(initial_mask)
        target_pixels = int(initial_valid_pixels * coverage_ratio)

        if target_pixels <= 0:
            return np.zeros_like(initial_mask)

        # 选择生成策略，增加多样性
        strategy_choice = random.random()

        if strategy_choice < 0.3:  # 30% - 扩展式生成（从小核心扩展）
            current_mask = self._expansion_based_generation(initial_mask, target_pixels)
        elif strategy_choice < 0.6:  # 30% - 边界收缩式（从外向内）
            current_mask = self._boundary_contraction_generation(
                initial_mask, target_pixels
            )
        else:  # 40% - 混合式生成
            current_mask = self._mixed_generation(initial_mask, target_pixels)

        # 确保连通性和平滑性
        current_mask = self._ensure_connectivity_and_smoothness(
            current_mask, initial_mask
        )

        return current_mask

    def _expansion_based_generation(
        self, initial_mask: np.ndarray, target_pixels: int
    ) -> np.ndarray:
        """
        基于扩展的生成方法：从核心区域向外扩展
        """
        h, w = initial_mask.shape

        # 创建一个较小的核心区域作为起点
        core_ratio = random.uniform(0.1, 0.3)  # 核心区域占10%-30%
        core_pixels = max(int(target_pixels * core_ratio), 100)

        # 生成核心区域（在中心附近）
        center_margin = min(h // 4, w // 4)
        core_mask = self._generate_core_region(initial_mask, core_pixels, center_margin)

        # 从核心区域扩展到目标大小
        expansion_ratio = (target_pixels - np.sum(core_mask)) / (
            np.sum(initial_mask) - np.sum(core_mask)
        )
        expansion_ratio = max(0.0, min(1.0, expansion_ratio))

        final_mask = self._generate_partial_expansion(
            core_mask, initial_mask, initial_mask, expansion_ratio
        )

        return final_mask

    def _generate_core_region(
        self, initial_mask: np.ndarray, core_pixels: int, margin: int
    ) -> np.ndarray:
        """
        在中心区域生成核心mask
        """
        h, w = initial_mask.shape
        core_mask = np.zeros_like(initial_mask)

        # 选择核心中心点（避开边缘）
        center_y = random.randint(margin, h - margin - 1)
        center_x = random.randint(margin, w - margin - 1)

        # 确保中心点在initial_mask有效区域内
        valid_positions = np.where(initial_mask == 1)
        if len(valid_positions[0]) > 0:
            # 找到最接近目标中心的有效点
            distances = (valid_positions[0] - center_y) ** 2 + (
                valid_positions[1] - center_x
            ) ** 2
            closest_idx = np.argmin(distances)
            center_y = valid_positions[0][closest_idx]
            center_x = valid_positions[1][closest_idx]

        # 从中心点开始区域生长
        core_mask[center_y, center_x] = 1
        current_pixels = 1

        # 使用队列进行扩展
        expansion_queue = [(center_y, center_x)]
        visited = {(center_y, center_x)}

        directions = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]

        while current_pixels < core_pixels and expansion_queue:
            y, x = expansion_queue.pop(0)

            # 随机选择扩展方向
            random.shuffle(directions)

            for dy, dx in directions:
                if current_pixels >= core_pixels:
                    break

                # 随机扩展半径，增加不规则性
                radius = random.randint(1, 3)
                ny, nx = y + dy * radius, x + dx * radius

                if (
                    0 <= ny < h
                    and 0 <= nx < w
                    and (ny, nx) not in visited
                    and initial_mask[ny, nx] == 1
                ):

                    # 控制扩展概率，创造不规则边界
                    expand_prob = 0.7 - 0.3 * (
                        current_pixels / core_pixels
                    )  # 随着增长降低概率
                    if random.random() < expand_prob:
                        core_mask[ny, nx] = 1
                        current_pixels += 1
                        expansion_queue.append((ny, nx))
                        visited.add((ny, nx))

        return core_mask

    def _boundary_contraction_generation(
        self, initial_mask: np.ndarray, target_pixels: int
    ) -> np.ndarray:
        """
        边界收缩式生成：从initial_mask开始，随机收缩边界
        """
        h, w = initial_mask.shape
        current_mask = initial_mask.copy()
        current_pixels = np.sum(current_mask)

        to_remove = current_pixels - target_pixels
        if to_remove <= 0:
            return current_mask

        removed_pixels = 0
        max_iterations = 50
        iteration = 0

        while removed_pixels < to_remove and iteration < max_iterations:
            iteration += 1

            # 找到边界点
            if SCIPY_AVAILABLE:
                boundary = current_mask - binary_erosion(current_mask, disk(1))
            else:
                boundary = self._find_boundary_simple(current_mask)

            boundary_coords = np.where(boundary > 0)
            if len(boundary_coords[0]) == 0:
                break

            boundary_points = list(zip(boundary_coords[0], boundary_coords[1]))

            # 随机选择要移除的边界区域
            removal_strategy = random.choice(
                ["single_edge", "random_patches", "corner_bias"]
            )

            if removal_strategy == "single_edge":
                removed_pixels += self._remove_from_single_edge(
                    current_mask,
                    boundary_points,
                    min(to_remove - removed_pixels, len(boundary_points) // 3),
                )
            elif removal_strategy == "random_patches":
                removed_pixels += self._remove_random_patches(
                    current_mask,
                    boundary_points,
                    min(to_remove - removed_pixels, len(boundary_points) // 2),
                )
            else:  # corner_bias
                removed_pixels += self._remove_with_corner_bias(
                    current_mask,
                    boundary_points,
                    min(to_remove - removed_pixels, len(boundary_points) // 2),
                )

        return current_mask

    def _mixed_generation(
        self, initial_mask: np.ndarray, target_pixels: int
    ) -> np.ndarray:
        """
        混合式生成：结合多种策略
        """
        h, w = initial_mask.shape

        # 先用边界收缩生成一个中等大小的mask
        intermediate_ratio = random.uniform(0.6, 0.8)
        intermediate_pixels = int(np.sum(initial_mask) * intermediate_ratio)
        intermediate_mask = self._boundary_contraction_generation(
            initial_mask, intermediate_pixels
        )

        # 然后添加一些随机的"岛屿"区域
        remaining_pixels = target_pixels - np.sum(intermediate_mask)

        if remaining_pixels > 0:
            # 在intermediate_mask之外但initial_mask之内添加小区域
            available_region = initial_mask - intermediate_mask
            available_coords = np.where(available_region > 0)

            if len(available_coords[0]) > 0:
                # 添加几个小的连通区域
                num_islands = random.randint(1, min(5, remaining_pixels // 20))
                pixels_per_island = remaining_pixels // num_islands

                for _ in range(num_islands):
                    if remaining_pixels <= 0:
                        break

                    # 随机选择岛屿中心
                    idx = random.randint(0, len(available_coords[0]) - 1)
                    center_y, center_x = (
                        available_coords[0][idx],
                        available_coords[1][idx],
                    )

                    # 生成小岛屿
                    island_pixels = min(pixels_per_island, remaining_pixels)
                    island_mask = self._create_small_island(
                        initial_mask,
                        intermediate_mask,
                        center_y,
                        center_x,
                        island_pixels,
                    )

                    intermediate_mask = np.maximum(intermediate_mask, island_mask)
                    remaining_pixels = target_pixels - np.sum(intermediate_mask)

        return intermediate_mask

    def _remove_from_single_edge(
        self, mask: np.ndarray, boundary_points: list, max_remove: int
    ) -> int:
        """
        从单个边缘移除像素，创造不规则的凹陷
        """
        h, w = mask.shape

        # 按边缘分组
        edge_groups = {"top": [], "bottom": [], "left": [], "right": []}

        for y, x in boundary_points:
            if y < h // 4:
                edge_groups["top"].append((y, x))
            elif y > 3 * h // 4:
                edge_groups["bottom"].append((y, x))
            elif x < w // 4:
                edge_groups["left"].append((y, x))
            elif x > 3 * w // 4:
                edge_groups["right"].append((y, x))

        # 选择有点的边缘
        available_edges = [
            edge for edge, points in edge_groups.items() if len(points) > 0
        ]
        if not available_edges:
            return 0

        selected_edge = random.choice(available_edges)
        edge_points = edge_groups[selected_edge]

        # 从选定边缘创建不规则凹陷
        removed = 0
        start_point = random.choice(edge_points)

        # 创建不规则的移除区域
        to_remove = [start_point]
        processed = set()

        directions = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]

        while to_remove and removed < max_remove:
            y, x = to_remove.pop(0)

            if (y, x) in processed or mask[y, x] == 0:
                continue

            mask[y, x] = 0
            removed += 1
            processed.add((y, x))

            # 不规则扩展
            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                if (
                    0 <= ny < h
                    and 0 <= nx < w
                    and (ny, nx) not in processed
                    and mask[ny, nx] == 1
                    and random.random() < 0.4
                ):  # 低概率扩展，创造不规则性
                    to_remove.append((ny, nx))

        return removed

    def _remove_random_patches(
        self, mask: np.ndarray, boundary_points: list, max_remove: int
    ) -> int:
        """
        随机移除边界补丁
        """
        removed = 0
        num_patches = random.randint(3, min(8, len(boundary_points) // 10))

        for _ in range(num_patches):
            if removed >= max_remove or not boundary_points:
                break

            # 随机选择补丁中心
            patch_center = random.choice(boundary_points)
            patch_size = random.randint(3, min(15, max_remove - removed))

            # 移除补丁
            patch_removed = self._remove_patch_around_point(
                mask, patch_center, patch_size
            )
            removed += patch_removed

            # 更新边界点列表
            boundary_points = [(y, x) for y, x in boundary_points if mask[y, x] == 1]

        return removed

    def _remove_with_corner_bias(
        self, mask: np.ndarray, boundary_points: list, max_remove: int
    ) -> int:
        """
        偏向角落的移除策略
        """
        h, w = mask.shape
        removed = 0

        # 计算每个边界点到角落的距离
        corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]

        # 为边界点分配权重（距离角落越近权重越高）
        weighted_points = []
        for y, x in boundary_points:
            min_corner_dist = min(abs(y - cy) + abs(x - cx) for cy, cx in corners)
            weight = 1.0 / (1.0 + min_corner_dist * 0.1)  # 距离越近权重越高
            weighted_points.append(((y, x), weight))

        # 按权重排序
        weighted_points.sort(key=lambda item: item[1], reverse=True)

        # 优先移除权重高的点
        for (y, x), weight in weighted_points:
            if removed >= max_remove or mask[y, x] == 0:
                continue

            # 移除概率与权重相关
            if random.random() < weight:
                mask[y, x] = 0
                removed += 1

        return removed

    def _create_small_island(
        self,
        initial_mask: np.ndarray,
        current_mask: np.ndarray,
        center_y: int,
        center_x: int,
        target_pixels: int,
    ) -> np.ndarray:
        """
        创建小的岛屿区域
        """
        h, w = initial_mask.shape
        island_mask = np.zeros_like(initial_mask)

        if (
            initial_mask[center_y, center_x] == 0
            or current_mask[center_y, center_x] == 1
        ):
            return island_mask

        # 从中心点扩展
        island_mask[center_y, center_x] = 1
        current_pixels = 1

        expansion_queue = [(center_y, center_x)]
        visited = {(center_y, center_x)}

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while current_pixels < target_pixels and expansion_queue:
            y, x = expansion_queue.pop(0)

            for dy, dx in directions:
                if current_pixels >= target_pixels:
                    break

                ny, nx = y + dy, x + dx

                if (
                    0 <= ny < h
                    and 0 <= nx < w
                    and (ny, nx) not in visited
                    and initial_mask[ny, nx] == 1
                    and current_mask[ny, nx] == 0
                    and random.random() < 0.6
                ):

                    island_mask[ny, nx] = 1
                    current_pixels += 1
                    expansion_queue.append((ny, nx))
                    visited.add((ny, nx))

        return island_mask

    def _remove_patch_around_point(
        self, mask: np.ndarray, center: tuple, patch_size: int
    ) -> int:
        """
        在指定点周围移除一个补丁
        """
        h, w = mask.shape
        y, x = center
        removed = 0

        # 创建不规则的补丁形状
        max_radius = int(np.sqrt(patch_size / np.pi)) + 2

        for dy in range(-max_radius, max_radius + 1):
            for dx in range(-max_radius, max_radius + 1):
                ny, nx = y + dy, x + dx

                if (
                    0 <= ny < h
                    and 0 <= nx < w
                    and mask[ny, nx] == 1
                    and removed < patch_size
                ):

                    # 使用椭圆形状 + 随机性
                    dist = (dy * dy) / (max_radius * max_radius) + (dx * dx) / (
                        max_radius * max_radius
                    )
                    remove_prob = max(0, 1 - dist) * random.uniform(0.5, 1.0)

                    if random.random() < remove_prob:
                        mask[ny, nx] = 0
                        removed += 1

        return removed

    def _find_boundary_simple(self, mask: np.ndarray) -> np.ndarray:
        """
        简单的边界查找方法（当scipy不可用时）
        """
        h, w = mask.shape
        boundary = np.zeros_like(mask)

        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if mask[i, j] == 1:
                    # 检查8邻域
                    neighbors = mask[i - 1 : i + 2, j - 1 : j + 2]
                    if np.any(neighbors == 0):
                        boundary[i, j] = 1

        return boundary

    def _ensure_connectivity_and_smoothness(
        self, mask: np.ndarray, initial_mask: np.ndarray
    ) -> np.ndarray:
        """
        确保连通性和边缘平滑性
        """
        if not SCIPY_AVAILABLE:
            return mask

        # 确保不超出initial_mask的范围
        mask = np.minimum(mask, initial_mask)

        # 标记连通区域
        labeled_array, num_features = label(mask)

        if num_features > 1:
            # 保留最大的连通区域
            region_sizes = np.bincount(labeled_array.ravel())[1:]  # 忽略背景
            if len(region_sizes) > 0:
                max_label = np.argmax(region_sizes) + 1
                mask = (labeled_array == max_label).astype(np.float32)

        # 轻微的形态学平滑
        smoothing_iterations = random.randint(1, 3)
        for _ in range(smoothing_iterations):
            # 先闭合再开放，平滑边缘
            kernel_size = random.choice([3, 5])
            kernel = np.ones((kernel_size, kernel_size))

            mask = binary_closing(mask, kernel)
            mask = binary_opening(mask, kernel)

            # 确保不超出initial_mask
            mask = np.minimum(mask, initial_mask)

        return mask.astype(np.float32)

    def _generate_partial_expansion(
        self, original_mask, reference_mask, reference_data, expansion_ratio
    ):
        """
        改进的部分扩展方法，增加边缘多样性
        """
        h, w = original_mask.shape
        original_mask = (original_mask > 0.5).astype(np.float32)
        reference_mask = (reference_data != 0).astype(np.float32)

        # 计算可扩展区域
        expansion_candidates = reference_mask - original_mask
        expansion_candidates = np.maximum(expansion_candidates, 0)
        total_expansion_candidates = np.sum(expansion_candidates)

        if total_expansion_candidates == 0 or expansion_ratio <= 0.0:
            return original_mask.copy()

        target_expansion_pixels = int(total_expansion_candidates * expansion_ratio)
        if target_expansion_pixels >= total_expansion_candidates:
            return reference_mask.copy()

        # 初始化
        final_mask = original_mask.copy()
        expanded_pixels = 0

        # 使用边界点扩展，增加随机性和多样性
        if SCIPY_AVAILABLE:
            boundary = original_mask - binary_erosion(original_mask, disk(1))
        else:
            boundary = self._find_boundary_simple(original_mask)

        boundary_coords = np.where(boundary > 0)
        boundary_points = list(zip(boundary_coords[0], boundary_coords[1]))

        if len(boundary_points) == 0:
            return original_mask.copy()

        # 扩展队列 - 随机化起始点
        random.shuffle(boundary_points)
        expansion_queue = boundary_points.copy()
        visited = set(boundary_points)

        # 动态调整参数以增加多样性
        interruption_prob = random.uniform(0.02, 0.08)  # 随机中断概率
        max_radius = random.randint(2, 6)  # 随机最大半径
        smoothness_factor = random.uniform(0.7, 0.95)  # 随机平滑度

        # 随机选择扩展策略
        expansion_strategies = ["uniform", "directional", "clustered"]
        strategy = random.choice(expansion_strategies)

        while expanded_pixels < target_expansion_pixels and expansion_queue:
            y, x = expansion_queue.pop(0)

            if random.random() < interruption_prob:
                continue

            if strategy == "uniform":
                # 均匀扩展
                directions = [
                    (-1, 0),
                    (0, -1),
                    (0, 1),
                    (1, 0),
                    (-1, -1),
                    (-1, 1),
                    (1, -1),
                    (1, 1),
                ]
                random.shuffle(directions)
            elif strategy == "directional":
                # 方向性扩展 - 偏向某个方向
                primary_dirs = [(-1, 0), (0, -1), (0, 1), (1, 0)]
                secondary_dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                main_dir = random.choice(primary_dirs)
                directions = (
                    [main_dir] * 3 + primary_dirs + secondary_dirs
                )  # 偏向主方向
                random.shuffle(directions)
            else:  # clustered
                # 聚集性扩展 - 偏向已扩展区域附近
                directions = [(-1, 0), (0, -1), (0, 1), (1, 0)]
                random.shuffle(directions)

            radius = random.randint(1, max_radius)

            for dy, dx in directions:
                ny, nx = y + dy * radius, x + dx * radius

                if (
                    0 <= ny < h
                    and 0 <= nx < w
                    and (ny, nx) not in visited
                    and expansion_candidates[ny, nx] > 0
                ):

                    expand_prob = smoothness_factor

                    # 策略特定的概率调整
                    if strategy == "clustered":
                        # 检查周围已扩展区域的密度
                        local_density = 0
                        for check_dy in range(-2, 3):
                            for check_dx in range(-2, 3):
                                check_y, check_x = ny + check_dy, nx + check_dx
                                if (
                                    0 <= check_y < h
                                    and 0 <= check_x < w
                                    and final_mask[check_y, check_x] == 1
                                ):
                                    local_density += 1
                        expand_prob *= (
                            local_density / 25.0
                        ) + 0.3  # 基于局部密度调整概率

                    if random.random() < expand_prob:
                        final_mask[ny, nx] = 1
                        expanded_pixels += 1
                        expansion_queue.append((ny, nx))
                        visited.add((ny, nx))

                if expanded_pixels >= target_expansion_pixels:
                    break

            if expanded_pixels >= target_expansion_pixels:
                break

        # 最终平滑和连通性处理
        if SCIPY_AVAILABLE:
            # 多样化的平滑操作
            smoothing_type = random.choice(["closing", "opening", "both"])
            kernel_size = random.randint(3, 7)
            kernel = np.ones((kernel_size, kernel_size))

            if smoothing_type == "closing":
                final_mask = binary_closing(final_mask, kernel)
            elif smoothing_type == "opening":
                final_mask = binary_opening(final_mask, kernel)
            else:  # both
                final_mask = binary_closing(final_mask, kernel)
                final_mask = binary_opening(final_mask, disk(kernel_size - 2))

            # 确保最大连通区域
            labeled_array, num_features = label(final_mask)
            if num_features > 0:
                region_sizes = np.bincount(labeled_array.ravel())[1:]
                max_label = np.argmax(region_sizes) + 1
                final_mask = (labeled_array == max_label).astype(np.float32)

            # 确保不超出参考区域
            final_mask = np.minimum(final_mask, reference_mask)

        return final_mask

    def _edge_inward_erosion(self, mask: np.ndarray, target_pixels: int) -> np.ndarray:
        """从边缘向内腐蚀，确保连续性"""
        h, w = mask.shape
        current_mask = mask.copy()
        current_pixels = np.sum(current_mask)

        # 随机选择1-2个边缘
        edges = ["top", "bottom", "left", "right"]
        num_edges = random.choice([1, 2])
        selected_edges = random.sample(edges, num_edges)

        to_remove = current_pixels - target_pixels
        remove_per_edge = to_remove // num_edges

        for edge in selected_edges:
            current_mask = self._erode_from_edge_inward(
                current_mask, edge, remove_per_edge
            )
            if np.sum(current_mask) <= target_pixels:
                break

        return current_mask

    def _erode_from_edge_inward(
        self, mask: np.ndarray, edge: str, remove_pixels: int
    ) -> np.ndarray:
        """从指定边缘向内连续腐蚀"""
        h, w = mask.shape
        removed = 0

        # 限制腐蚀深度，确保从边缘连续向内
        max_depth = min(h // 3, w // 3, 50)

        if edge == "top":
            for depth in range(max_depth):
                if removed >= remove_pixels:
                    break
                row = depth
                if row >= h:
                    break
                # 在当前行找到所有有效像素
                valid_cols = [j for j in range(w) if mask[row, j] == 1]
                if not valid_cols:
                    continue
                # 移除这一行的像素
                for col in valid_cols:
                    if removed >= remove_pixels:
                        break
                    mask[row, col] = 0
                    removed += 1

        elif edge == "bottom":
            for depth in range(max_depth):
                if removed >= remove_pixels:
                    break
                row = h - 1 - depth
                if row < 0:
                    break
                valid_cols = [j for j in range(w) if mask[row, j] == 1]
                if not valid_cols:
                    continue
                for col in valid_cols:
                    if removed >= remove_pixels:
                        break
                    mask[row, col] = 0
                    removed += 1

        elif edge == "left":
            for depth in range(max_depth):
                if removed >= remove_pixels:
                    break
                col = depth
                if col >= w:
                    break
                valid_rows = [i for i in range(h) if mask[i, col] == 1]
                if not valid_rows:
                    continue
                for row in valid_rows:
                    if removed >= remove_pixels:
                        break
                    mask[row, col] = 0
                    removed += 1

        elif edge == "right":
            for depth in range(max_depth):
                if removed >= remove_pixels:
                    break
                col = w - 1 - depth
                if col < 0:
                    break
                valid_rows = [i for i in range(h) if mask[i, col] == 1]
                if not valid_rows:
                    continue
                for row in valid_rows:
                    if removed >= remove_pixels:
                        break
                    mask[row, col] = 0
                    removed += 1

        return mask

    def _gentle_uniform_erosion(
        self, mask: np.ndarray, target_pixels: int
    ) -> np.ndarray:
        """温和的均匀腐蚀，确保从边缘开始"""
        current_mask = mask.copy()
        current_pixels = np.sum(current_mask)

        # 使用小步长腐蚀
        max_iterations = 20
        iteration = 0

        while current_pixels > target_pixels and iteration < max_iterations:
            iteration += 1

            if SCIPY_AVAILABLE:
                # 使用小的腐蚀核
                erosion_size = 1
                kernel = disk(erosion_size)
                eroded_mask = binary_erosion(current_mask, kernel)

                # 确保腐蚀结果连通且从边缘开始
                eroded_mask = self._ensure_edge_connectivity(eroded_mask)
            else:
                eroded_mask = self._simple_edge_erosion(current_mask)

            new_pixels = np.sum(eroded_mask)
            if new_pixels > 0 and new_pixels < current_pixels:
                current_mask = eroded_mask
                current_pixels = new_pixels
            else:
                break

        return current_mask

    def _ensure_edge_connectivity(self, mask: np.ndarray) -> np.ndarray:
        """确保mask只有与边缘相连的区域"""
        if not SCIPY_AVAILABLE:
            return mask

        h, w = mask.shape

        # 找到所有连通区域
        labeled_mask, num_features = label(mask)

        if num_features <= 1:
            return mask

        # 找到与边缘相连的区域
        edge_connected_regions = set()

        # 检查四个边缘
        for i in range(h):
            if labeled_mask[i, 0] > 0:
                edge_connected_regions.add(labeled_mask[i, 0])
            if labeled_mask[i, w - 1] > 0:
                edge_connected_regions.add(labeled_mask[i, w - 1])
        for j in range(w):
            if labeled_mask[0, j] > 0:
                edge_connected_regions.add(labeled_mask[0, j])
            if labeled_mask[h - 1, j] > 0:
                edge_connected_regions.add(labeled_mask[h - 1, j])

        # 只保留与边缘相连的最大区域
        if edge_connected_regions:
            region_sizes = []
            for region_id in edge_connected_regions:
                size = np.sum(labeled_mask == region_id)
                region_sizes.append((size, region_id))

            # 选择最大的与边缘相连的区域
            region_sizes.sort(reverse=True)
            largest_region_id = region_sizes[0][1]

            # 创建新的mask，只保留最大的边缘连通区域
            new_mask = (labeled_mask == largest_region_id).astype(np.float32)
            return new_mask

        return mask

    def _simple_edge_erosion(self, mask: np.ndarray) -> np.ndarray:
        """简化版边缘腐蚀"""
        h, w = mask.shape
        eroded_mask = mask.copy()

        # 找到边界点并腐蚀
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if mask[i, j] == 1:
                    # 检查是否为边界点
                    neighbors = mask[i - 1 : i + 2, j - 1 : j + 2]
                    if np.any(neighbors == 0):  # 如果邻域中有0，说明是边界
                        if random.random() < 0.3:  # 30%概率腐蚀
                            eroded_mask[i, j] = 0

        return eroded_mask

    def _generate_concentrated_mask(
        self, mask: np.ndarray, target_pixels: int, local_size: int
    ) -> np.ndarray:
        """修正版集中分布mask生成：避免边缘出现缺失"""
        h, w = mask.shape

        # 确保簇中心远离边缘，增大边距
        margin = max(local_size // 6, 5)  # 至少5像素边距

        # 如果边距太大，调整
        if margin * 2 >= min(h, w):
            margin = max(2, min(h, w) // 8)

        # 确保有足够的内部空间
        inner_h = h - 2 * margin
        inner_w = w - 2 * margin

        if inner_h <= 0 or inner_w <= 0:
            # 如果没有足够内部空间，只在中心创建小的缺失
            center_y, center_x = h // 2, w // 2
            # 创建小的圆形缺失
            radius = min(3, np.sqrt(target_pixels / np.pi))
            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            small_mask = dist <= radius
            mask[small_mask] = 0
            return mask

        # 随机确定簇数量 (1-2个，减少复杂性)
        num_clusters = random.randint(1, 2)
        pixels_per_cluster = target_pixels // num_clusters

        for cluster_idx in range(num_clusters):
            if cluster_idx == num_clusters - 1:
                # 最后一个簇获得剩余像素
                cluster_pixels = target_pixels - np.sum(mask == 0)
            else:
                cluster_pixels = pixels_per_cluster

            if cluster_pixels <= 0:
                continue

            # 在内部区域选择簇中心
            center_y = random.randint(margin, h - margin - 1)
            center_x = random.randint(margin, w - margin - 1)

            # 生成紧凑的簇，限制最大半径
            max_radius = min(margin - 1, local_size // 8)  # 确保不会触及边缘
            cluster_mask = self._generate_compact_cluster(
                (h, w), center_y, center_x, cluster_pixels, max_radius
            )

            # 应用到主mask
            mask[cluster_mask] = 0

        return mask

    def _generate_compact_cluster(
        self,
        shape: Tuple[int, int],
        center_y: int,
        center_x: int,
        target_pixels: int,
        max_radius: int,
    ) -> np.ndarray:
        """生成紧凑的簇，确保不超出指定半径"""
        h, w = shape
        cluster_mask = np.zeros(shape, dtype=bool)

        # 使用区域生长，但严格限制范围
        if not (0 <= center_y < h and 0 <= center_x < w):
            return cluster_mask

        # 从中心开始
        cluster_mask[center_y, center_x] = True
        current_pixels = 1

        # 候选点队列
        candidates = [(center_y, center_x)]
        processed = {(center_y, center_x)}

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 只使用4连通

        while current_pixels < target_pixels and candidates:
            # 从候选点中选择距离中心最近的
            candidates.sort(
                key=lambda p: (p[0] - center_y) ** 2 + (p[1] - center_x) ** 2
            )

            new_candidates = []

            for y, x in candidates:
                if current_pixels >= target_pixels:
                    break

                # 添加邻居点
                for dy, dx in directions:
                    ny, nx = y + dy, x + dx

                    if (ny, nx) in processed:
                        continue

                    # 检查是否在范围内
                    if not (0 <= ny < h and 0 <= nx < w):
                        processed.add((ny, nx))
                        continue

                    # 检查距离限制
                    dist = np.sqrt((ny - center_y) ** 2 + (nx - center_x) ** 2)
                    if dist > max_radius:
                        processed.add((ny, nx))
                        continue

                    # 添加这个点
                    cluster_mask[ny, nx] = True
                    current_pixels += 1
                    processed.add((ny, nx))
                    new_candidates.append((ny, nx))

                    if current_pixels >= target_pixels:
                        break

            candidates = new_candidates

        return cluster_mask

    def _gentle_edge_erosion(self, mask: np.ndarray, target_pixels: int) -> np.ndarray:
        """温和的边缘腐蚀，参考原始方法但减少毛刺"""
        current_mask = mask.copy()
        current_pixels = np.sum(current_mask)

        # 使用多次小幅度腐蚀
        max_iterations = 15
        iteration = 0

        while current_pixels > target_pixels and iteration < max_iterations:
            iteration += 1

            if SCIPY_AVAILABLE:
                # 使用较小的腐蚀核，减少毛刺
                erosion_size = random.randint(1, 2)  # 减小腐蚀核尺寸
                kernel = disk(erosion_size)
                eroded_mask = binary_erosion(current_mask, kernel)

                # 应用平滑操作
                if random.random() < 0.5:
                    eroded_mask = binary_closing(eroded_mask, disk(1))
            else:
                eroded_mask = self._simple_gentle_erosion(current_mask)

            new_pixels = np.sum(eroded_mask)
            if new_pixels > 0 and new_pixels < current_pixels:
                current_mask = eroded_mask
                current_pixels = new_pixels
            else:
                break

        return current_mask

    def _selective_edge_erosion(
        self, mask: np.ndarray, target_pixels: int
    ) -> np.ndarray:
        """选择性边缘腐蚀"""
        h, w = mask.shape
        current_mask = mask.copy()
        current_pixels = np.sum(current_mask)
        to_remove = current_pixels - target_pixels

        if to_remove <= 0:
            return current_mask

        # 随机选择1-2个边缘
        edges = ["top", "bottom", "left", "right"]
        num_edges = random.choice([1, 2])
        selected_edges = random.sample(edges, num_edges)

        remove_per_edge = to_remove // num_edges

        for edge in selected_edges:
            current_mask = self._erode_edge_gradually(
                current_mask, edge, remove_per_edge
            )

        return current_mask

    def _erode_edge_gradually(
        self, mask: np.ndarray, edge: str, remove_pixels: int
    ) -> np.ndarray:
        """渐进式边缘腐蚀，避免剧烈毛刺"""
        h, w = mask.shape
        removed = 0

        # 创建温和的腐蚀模式
        max_depth = min(h // 4, w // 4, 20)  # 限制最大腐蚀深度

        for depth in range(max_depth):
            if removed >= remove_pixels:
                break

            # 在当前深度随机移除一些像素
            if edge == "top":
                row = depth
                if row < h:
                    valid_cols = [j for j in range(w) if mask[row, j] == 1]
                    remove_count = min(len(valid_cols), max(1, remove_pixels - removed))
                    if valid_cols:
                        selected_cols = random.sample(
                            valid_cols, min(remove_count, len(valid_cols))
                        )
                        for col in selected_cols:
                            mask[row, col] = 0
                            removed += 1
            elif edge == "bottom":
                row = h - 1 - depth
                if row >= 0:
                    valid_cols = [j for j in range(w) if mask[row, j] == 1]
                    remove_count = min(len(valid_cols), max(1, remove_pixels - removed))
                    if valid_cols:
                        selected_cols = random.sample(
                            valid_cols, min(remove_count, len(valid_cols))
                        )
                        for col in selected_cols:
                            mask[row, col] = 0
                            removed += 1
            elif edge == "left":
                col = depth
                if col < w:
                    valid_rows = [i for i in range(h) if mask[i, col] == 1]
                    remove_count = min(len(valid_rows), max(1, remove_pixels - removed))
                    if valid_rows:
                        selected_rows = random.sample(
                            valid_rows, min(remove_count, len(valid_rows))
                        )
                        for row in selected_rows:
                            mask[row, col] = 0
                            removed += 1
            elif edge == "right":
                col = w - 1 - depth
                if col >= 0:
                    valid_rows = [i for i in range(h) if mask[i, col] == 1]
                    remove_count = min(len(valid_rows), max(1, remove_pixels - removed))
                    if valid_rows:
                        selected_rows = random.sample(
                            valid_rows, min(remove_count, len(valid_rows))
                        )
                        for row in selected_rows:
                            mask[row, col] = 0
                            removed += 1

        return mask

    def _region_based_erosion(self, mask: np.ndarray, target_pixels: int) -> np.ndarray:
        """基于区域的遮挡方法，参考第二份代码的思路"""
        h, w = mask.shape
        current_mask = mask.copy()
        current_pixels = np.sum(current_mask)
        to_remove = current_pixels - target_pixels

        if to_remove <= 0:
            return current_mask

        # 选择几个起始点进行区域移除
        num_regions = random.randint(1, 3)
        remove_per_region = to_remove // num_regions

        for _ in range(num_regions):
            # 随机选择起始点
            valid_positions = np.where(current_mask == 1)
            if len(valid_positions[0]) == 0:
                break

            idx = random.randint(0, len(valid_positions[0]) - 1)
            start_y, start_x = valid_positions[0][idx], valid_positions[1][idx]

            # 从起始点进行区域移除
            current_mask = self._remove_region_from_point(
                current_mask, start_y, start_x, remove_per_region
            )

        return current_mask

    def _remove_region_from_point(
        self, mask: np.ndarray, start_y: int, start_x: int, target_remove: int
    ) -> np.ndarray:
        """从指定点开始移除区域"""
        h, w = mask.shape
        removed = 0
        to_process = [(start_y, start_x)]
        processed = set()

        directions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

        while removed < target_remove and to_process:
            y, x = to_process.pop(0)

            if (y, x) in processed or removed >= target_remove:
                continue

            if 0 <= y < h and 0 <= x < w and mask[y, x] == 1:
                mask[y, x] = 0
                removed += 1
                processed.add((y, x))

                # 添加邻居点，但控制扩展概率
                random.shuffle(directions)
                for dy, dx in directions:
                    ny, nx = y + dy, x + dx
                    if (
                        0 <= ny < h
                        and 0 <= nx < w
                        and (ny, nx) not in processed
                        and random.random() < 0.6
                    ):  # 控制扩展概率
                        to_process.append((ny, nx))

                # 随机中断，避免过于规则
                if random.random() < 0.15:
                    continue

        return mask

    def _simple_gentle_erosion(self, mask: np.ndarray) -> np.ndarray:
        """简化版温和腐蚀"""
        h, w = mask.shape
        eroded_mask = mask.copy()

        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if mask[i, j] == 1:
                    neighbors = mask[i - 1 : i + 2, j - 1 : j + 2]
                    neighbor_count = np.sum(neighbors)

                    # 更温和的腐蚀条件
                    if neighbor_count >= 7 and random.random() < 0.2:
                        eroded_mask[i, j] = 0

        return eroded_mask

    def _generate_local_mask(
        self,
        local_size: int,
        missing_ratio: float,
        enable_edge_connection: bool = False,
    ) -> np.ndarray:
        """
        生成多样化的local mask，支持多种形状和边界相接
        """
        shape = (local_size, local_size)
        missing_ratio_range = (missing_ratio * 0.8, missing_ratio * 1.2)  # 允许±20%变化
        num_clusters_range = (1, 4)
        cluster_size_range = (8, min(18, local_size // 3))

        # 10%概率与边界相接，90%概率内部生成
        edge_connection_prob = 0.1 if enable_edge_connection else 0.0
        scattered_probability = 0.2

        # 决定生成策略
        strategy_choice = np.random.random()

        if strategy_choice < edge_connection_prob:
            # 10% - 边界相接
            return self._generate_edge_connected_mask_local(shape, missing_ratio_range)
            """elif strategy_choice < edge_connection_prob + scattered_probability * (
                1 - edge_connection_prob
            ):
                # 18% - 分散分布（在剩余90%中的20%）
                return self._generate_scattered_mask_local(shape, missing_ratio_range)"""
        else:
            # 72% - 集中分布（在剩余90%中的80%）
            return self._generate_concentrated_mask_pattern_local(
                shape, missing_ratio_range, num_clusters_range, cluster_size_range
            )

    def _generate_edge_connected_mask_local(self, shape, missing_ratio_range):
        """
        生成与边界相接的mask（10%概率使用）
        """
        h, w = shape
        mask = np.ones(shape, dtype=np.float32)

        # 随机确定缺失比例
        target_missing_ratio = np.random.uniform(*missing_ratio_range)
        total_pixels = h * w
        target_missing_pixels = int(total_pixels * target_missing_ratio)

        # 选择边界相接的策略
        connection_strategy = np.random.choice(["single_edge", "corner", "multi_edge"])

        if connection_strategy == "single_edge":
            mask = self._create_single_edge_mask(mask, target_missing_pixels, h, w)
        elif connection_strategy == "corner":
            mask = self._create_corner_mask(mask, target_missing_pixels, h, w)
        else:  # multi_edge
            mask = self._create_multi_edge_mask(mask, target_missing_pixels, h, w)

        return mask

    def _create_single_edge_mask(self, mask, target_missing_pixels, h, w):
        """从单个边缘创建缺失区域"""
        # 选择一个边缘
        edge = np.random.choice(["top", "bottom", "left", "right"])

        removed = 0
        if edge == "top":
            # 从上边缘开始，创建不规则向下延伸
            start_x = np.random.randint(w // 4, 3 * w // 4)
            max_depth = min(h // 2, int(np.sqrt(target_missing_pixels * 2)))

            for depth in range(max_depth):
                if removed >= target_missing_pixels:
                    break
                y = depth
                # 创建变化的宽度
                width_at_depth = max(
                    1, int(w // 4 * (1 - depth / max_depth) + np.random.randint(-2, 3))
                )
                left_x = max(0, start_x - width_at_depth // 2)
                right_x = min(w, start_x + width_at_depth // 2)

                for x in range(left_x, right_x):
                    if removed < target_missing_pixels:
                        mask[y, x] = 0
                        removed += 1

        elif edge == "bottom":
            start_x = np.random.randint(w // 4, 3 * w // 4)
            max_depth = min(h // 2, int(np.sqrt(target_missing_pixels * 2)))

            for depth in range(max_depth):
                if removed >= target_missing_pixels:
                    break
                y = h - 1 - depth
                width_at_depth = max(
                    1, int(w // 4 * (1 - depth / max_depth) + np.random.randint(-2, 3))
                )
                left_x = max(0, start_x - width_at_depth // 2)
                right_x = min(w, start_x + width_at_depth // 2)

                for x in range(left_x, right_x):
                    if removed < target_missing_pixels:
                        mask[y, x] = 0
                        removed += 1

        elif edge == "left":
            start_y = np.random.randint(h // 4, 3 * h // 4)
            max_depth = min(w // 2, int(np.sqrt(target_missing_pixels * 2)))

            for depth in range(max_depth):
                if removed >= target_missing_pixels:
                    break
                x = depth
                height_at_depth = max(
                    1, int(h // 4 * (1 - depth / max_depth) + np.random.randint(-2, 3))
                )
                top_y = max(0, start_y - height_at_depth // 2)
                bottom_y = min(h, start_y + height_at_depth // 2)

                for y in range(top_y, bottom_y):
                    if removed < target_missing_pixels:
                        mask[y, x] = 0
                        removed += 1

        else:  # right
            start_y = np.random.randint(h // 4, 3 * h // 4)
            max_depth = min(w // 2, int(np.sqrt(target_missing_pixels * 2)))

            for depth in range(max_depth):
                if removed >= target_missing_pixels:
                    break
                x = w - 1 - depth
                height_at_depth = max(
                    1, int(h // 4 * (1 - depth / max_depth) + np.random.randint(-2, 3))
                )
                top_y = max(0, start_y - height_at_depth // 2)
                bottom_y = min(h, start_y + height_at_depth // 2)

                for y in range(top_y, bottom_y):
                    if removed < target_missing_pixels:
                        mask[y, x] = 0
                        removed += 1

        return mask

    def _create_corner_mask(self, mask, target_missing_pixels, h, w):
        """从角落创建L形缺失区域"""
        # 选择一个角落
        corner = np.random.choice(
            ["top_left", "top_right", "bottom_left", "bottom_right"]
        )

        # L形的两个臂长
        max_arm_length = min(h // 3, w // 3)
        arm1_length = np.random.randint(3, max_arm_length)
        arm2_length = np.random.randint(3, max_arm_length)
        arm_width = np.random.randint(2, min(5, max_arm_length // 2))

        removed = 0

        if corner == "top_left":
            # 水平臂
            for x in range(min(arm1_length, w)):
                for y in range(min(arm_width, h)):
                    if removed < target_missing_pixels:
                        mask[y, x] = 0
                        removed += 1
            # 垂直臂
            for y in range(min(arm2_length, h)):
                for x in range(min(arm_width, w)):
                    if (
                        removed < target_missing_pixels and mask[y, x] == 1
                    ):  # 避免重复设置
                        mask[y, x] = 0
                        removed += 1

        elif corner == "top_right":
            # 水平臂
            for x in range(max(0, w - arm1_length), w):
                for y in range(min(arm_width, h)):
                    if removed < target_missing_pixels:
                        mask[y, x] = 0
                        removed += 1
            # 垂直臂
            for y in range(min(arm2_length, h)):
                for x in range(max(0, w - arm_width), w):
                    if removed < target_missing_pixels and mask[y, x] == 1:
                        mask[y, x] = 0
                        removed += 1

        elif corner == "bottom_left":
            # 水平臂
            for x in range(min(arm1_length, w)):
                for y in range(max(0, h - arm_width), h):
                    if removed < target_missing_pixels:
                        mask[y, x] = 0
                        removed += 1
            # 垂直臂
            for y in range(max(0, h - arm2_length), h):
                for x in range(min(arm_width, w)):
                    if removed < target_missing_pixels and mask[y, x] == 1:
                        mask[y, x] = 0
                        removed += 1

        else:  # bottom_right
            # 水平臂
            for x in range(max(0, w - arm1_length), w):
                for y in range(max(0, h - arm_width), h):
                    if removed < target_missing_pixels:
                        mask[y, x] = 0
                        removed += 1
            # 垂直臂
            for y in range(max(0, h - arm2_length), h):
                for x in range(max(0, w - arm_width), w):
                    if removed < target_missing_pixels and mask[y, x] == 1:
                        mask[y, x] = 0
                        removed += 1

        return mask

    def _create_multi_edge_mask(self, mask, target_missing_pixels, h, w):
        """从多个边缘创建细长连接"""
        # 创建2-3个细长的边缘连接
        num_connections = np.random.randint(2, 4)
        pixels_per_connection = target_missing_pixels // num_connections

        removed = 0

        for conn_idx in range(num_connections):
            if removed >= target_missing_pixels:
                break

            # 随机选择起始边缘
            start_edge = np.random.choice(["top", "bottom", "left", "right"])

            # 连接长度和宽度
            connection_length = np.random.randint(3, min(h // 3, w // 3))
            connection_width = np.random.randint(1, 3)

            connection_removed = 0

            if start_edge == "top":
                start_x = np.random.randint(connection_width, w - connection_width)
                for depth in range(connection_length):
                    if (
                        removed >= target_missing_pixels
                        or connection_removed >= pixels_per_connection
                    ):
                        break
                    y = depth
                    if y >= h:
                        break
                    for dx in range(-connection_width, connection_width + 1):
                        x = start_x + dx
                        if 0 <= x < w and removed < target_missing_pixels:
                            mask[y, x] = 0
                            removed += 1
                            connection_removed += 1

            elif start_edge == "bottom":
                start_x = np.random.randint(connection_width, w - connection_width)
                for depth in range(connection_length):
                    if (
                        removed >= target_missing_pixels
                        or connection_removed >= pixels_per_connection
                    ):
                        break
                    y = h - 1 - depth
                    if y < 0:
                        break
                    for dx in range(-connection_width, connection_width + 1):
                        x = start_x + dx
                        if 0 <= x < w and removed < target_missing_pixels:
                            mask[y, x] = 0
                            removed += 1
                            connection_removed += 1

            elif start_edge == "left":
                start_y = np.random.randint(connection_width, h - connection_width)
                for depth in range(connection_length):
                    if (
                        removed >= target_missing_pixels
                        or connection_removed >= pixels_per_connection
                    ):
                        break
                    x = depth
                    if x >= w:
                        break
                    for dy in range(-connection_width, connection_width + 1):
                        y = start_y + dy
                        if 0 <= y < h and removed < target_missing_pixels:
                            mask[y, x] = 0
                            removed += 1
                            connection_removed += 1

            else:  # right
                start_y = np.random.randint(connection_width, h - connection_width)
                for depth in range(connection_length):
                    if (
                        removed >= target_missing_pixels
                        or connection_removed >= pixels_per_connection
                    ):
                        break
                    x = w - 1 - depth
                    if x < 0:
                        break
                    for dy in range(-connection_width, connection_width + 1):
                        y = start_y + dy
                        if 0 <= y < h and removed < target_missing_pixels:
                            mask[y, x] = 0
                            removed += 1
                            connection_removed += 1

        return mask

    def _generate_concentrated_mask_pattern_local(
        self, shape, missing_ratio_range, num_clusters_range, cluster_size_range
    ):
        """生成集中分布的mask模式 - 增强多样性"""
        h, w = shape
        mask = np.ones(shape, dtype=np.float32)

        # 随机确定缺失比例和簇数量
        target_missing_ratio = np.random.uniform(*missing_ratio_range)
        # print(f"target missing ratio: {target_missing_ratio}")
        num_clusters = 1  # np.random.randint(*num_clusters_range)
        # print(f"num clusters: {num_clusters}")

        total_pixels = h * w
        target_missing_pixels = int(total_pixels * target_missing_ratio)
        # print(f"target_missing_pixels: {target_missing_pixels}")

        # 追踪实际生成的缺失像素
        actual_missing_pixels = 0

        for cluster_idx in range(num_clusters):
            # 为每个簇分配像素数
            remaining_pixels = target_missing_pixels - actual_missing_pixels
            if cluster_idx == num_clusters - 1:
                # 最后一个簇获得剩余所有像素
                cluster_pixels = remaining_pixels
            else:
                # 前面的簇平均分配剩余像素
                cluster_pixels = remaining_pixels // (num_clusters - cluster_idx)

            if cluster_pixels <= 0:
                continue

            # print(f"Cluster {cluster_idx}: target pixels = {cluster_pixels}")

            # 选择簇中心，避免太靠近边缘
            margin = max(3, min(h, w) // 3)
            center_x = np.random.randint(margin, h - margin)
            center_y = np.random.randint(margin, w - margin)

            # 生成多样化的集中簇
            cluster_mask = self._generate_diverse_cluster_local(
                shape, center_x, center_y, cluster_pixels
            )

            # 打印生成的簇大小（形态学操作前）
            cluster_size_before = np.sum(cluster_mask)
            # print(f"  Cluster size before morphology: {cluster_size_before}")

            # 应用形态学操作确保集中性（可选）
            if True:  # 添加一个控制开关
                cluster_mask = self._apply_morphological_operations_local(cluster_mask)
                cluster_size_after = np.sum(cluster_mask)
                # print(f"  Cluster size after morphology: {cluster_size_after}")

            # 应用到主mask
            mask[cluster_mask] = 0
            actual_missing_pixels += np.sum(cluster_mask)

        # 打印最终统计
        final_missing_ratio = 1 - np.mean(mask)
        """print(
            f"Final missing ratio: {final_missing_ratio:.4f} (target was {target_missing_ratio:.4f})"
        )
        print(
            f"Final missing pixels: {int(total_pixels * final_missing_ratio)} (target was {target_missing_pixels})"
        )"""

        return mask

    def _apply_morphological_operations_local(self, mask):
        """应用形态学操作来进一步集中mask - 保守版本"""
        if not SCIPY_AVAILABLE:
            return mask

        # 使用更小的核来减少区域损失
        kernel_close = disk(1)  # 从 2 改为 1
        mask_closed = binary_dilation(mask, kernel_close)
        mask_closed = binary_erosion(mask_closed, kernel_close)

        # 可选：保留多个较大的连通区域而不是只保留最大的
        labeled_mask, num_labels = label(mask_closed)
        if num_labels > 1:
            # 计算所有区域的大小
            region_sizes = []
            for i in range(1, num_labels + 1):
                size = np.sum(labeled_mask == i)
                region_sizes.append((size, i))

            region_sizes.sort(reverse=True)

            # 保留占原始区域 80% 以上的所有大区域
            original_size = np.sum(mask)
            new_mask = np.zeros_like(mask_closed)
            accumulated_size = 0

            for size, region_id in region_sizes:
                new_mask[labeled_mask == region_id] = True
                accumulated_size += size
                # 如果已经恢复了80%的原始区域，停止
                if accumulated_size >= 0.8 * original_size:
                    break

            mask_closed = new_mask

        return mask_closed

    def _generate_diverse_cluster_local(self, shape, center_x, center_y, target_pixels):
        """生成多样化形状的集中簇 - 确保生成足够大的区域"""
        # 根据目标像素数估算需要的半径
        estimated_radius = int(np.sqrt(target_pixels / np.pi))

        # 确保半径不会太小
        actual_radius = max(estimated_radius, 10)

        print(
            f"  Generating cluster with estimated radius: {estimated_radius}, actual: {actual_radius}"
        )

        # 随机选择形状类型
        # 方法1：直接使用概率权重
        shape_types = ["growth", "ellipse", "multi_circle", "polygon"]
        shape_weights = [0.5, 0.2, 0.2, 0.1]  # 对应每个shape_type的概率

        shape_type = np.random.choice(shape_types, p=shape_weights)
        print(f"  Using shape type: {shape_type}")

        if shape_type == "ellipse":
            return self._generate_ellipse_cluster_local_v2(
                shape, center_x, center_y, actual_radius, target_pixels
            )
        elif shape_type == "polygon":
            return self._generate_polygon_cluster_local(
                shape, center_x, center_y, actual_radius, target_pixels
            )
        elif shape_type == "growth":
            return self._generate_growth_cluster_local_v2(
                shape, center_x, center_y, actual_radius, target_pixels
            )
        else:  # multi_circle
            return self._generate_multi_circle_cluster_local(
                shape, center_x, center_y, actual_radius, target_pixels
            )

    def _generate_ellipse_cluster_local_v2(
        self, shape, center_x, center_y, radius, target_pixels
    ):
        """生成椭圆形簇 - 改进版本"""
        h, w = shape
        mask = np.zeros(shape, dtype=bool)

        # 根据目标像素数调整半径
        # 椭圆面积 = π * a * b
        aspect_ratio = np.random.uniform(0.5, 2.0)
        a = radius * np.sqrt(aspect_ratio)
        b = radius / np.sqrt(aspect_ratio)

        # 迭代调整半径直到接近目标像素数
        scale = 1.0
        for _ in range(10):  # 最多迭代10次
            temp_mask = np.zeros(shape, dtype=bool)

            # 创建坐标网格
            y, x = np.ogrid[:h, :w]

            # 椭圆方程
            ellipse = ((x - center_y) / (b * scale)) ** 2 + (
                (y - center_x) / (a * scale)
            ) ** 2 <= 1
            temp_mask[ellipse] = True

            current_pixels = np.sum(temp_mask)
            if current_pixels >= target_pixels * 0.9:  # 允许10%的误差
                mask = temp_mask
                break

            # 调整缩放因子
            scale *= np.sqrt(target_pixels / current_pixels)

        return mask

    def _generate_growth_cluster_local_v2(
        self, shape, center_x, center_y, radius, target_pixels
    ):
        """生成生长型簇 - 使用区域生长算法"""
        h, w = shape
        mask = np.zeros(shape, dtype=bool)

        # 初始化种子点
        mask[center_x, center_y] = True

        # 待处理的边界点
        boundary = [(center_x, center_y)]
        directions = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]

        current_pixels = 1

        while boundary and current_pixels < target_pixels:
            # 随机选择一个边界点
            idx = np.random.randint(len(boundary))
            x, y = boundary.pop(idx)

            # 随机打乱方向
            np.random.shuffle(directions)

            for dx, dy in directions:
                nx, ny = x + dx, y + dy

                # 检查边界和是否已访问
                if 0 <= nx < h and 0 <= ny < w and not mask[nx, ny]:
                    # 根据距离中心的距离决定生长概率
                    dist = np.sqrt((nx - center_x) ** 2 + (ny - center_y) ** 2)

                    # 距离越远，生长概率越低
                    growth_prob = max(0.1, 1.0 - dist / (2 * radius))

                    if np.random.random() < growth_prob:
                        mask[nx, ny] = True
                        boundary.append((nx, ny))
                        current_pixels += 1

                        if current_pixels >= target_pixels:
                            break

        return mask

    def _generate_polygon_cluster_local(
        self, shape, center_x, center_y, max_radius, target_pixels
    ):
        """生成多边形缺失区域"""
        h, w = shape
        cluster_mask = np.zeros(shape, dtype=bool)

        # 生成不规则多边形的顶点
        num_vertices = np.random.randint(5, 9)  # 5-8个顶点
        angles = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)

        # 添加随机偏移创造不规则性
        angles += np.random.uniform(-0.4, 0.4, num_vertices)

        # 计算平均半径
        avg_radius = min(max_radius * 0.7, np.sqrt(target_pixels / np.pi))

        # 生成顶点
        vertices = []
        for angle in angles:
            # 随机化半径创造不规则形状
            radius = avg_radius * np.random.uniform(0.5, 1.5)
            radius = min(radius, max_radius)

            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)

            # 确保在范围内
            x = max(0, min(h - 1, x))
            y = max(0, min(w - 1, y))

            vertices.append((int(x), int(y)))

        # 填充多边形内部
        if len(vertices) >= 3:
            cluster_mask = self._fill_polygon_local(cluster_mask, vertices)

        return cluster_mask

    def _fill_polygon_local(self, mask, vertices):
        """简单的多边形填充算法"""
        h, w = mask.shape

        if len(vertices) < 3:
            return mask

        # 找到边界框
        min_x = max(0, min(v[0] for v in vertices))
        max_x = min(h - 1, max(v[0] for v in vertices))
        min_y = max(0, min(v[1] for v in vertices))
        max_y = min(w - 1, max(v[1] for v in vertices))

        # 使用点在多边形内测试
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                if self._point_in_polygon(x, y, vertices):
                    mask[x, y] = True

        return mask

    def _point_in_polygon(self, x, y, vertices):
        """点在多边形内测试（射线法）"""
        n = len(vertices)
        inside = False

        p1x, p1y = vertices[0]
        for i in range(1, n + 1):
            p2x, p2y = vertices[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def _generate_multi_circle_cluster_local(
        self, shape, center_x, center_y, max_radius, target_pixels
    ):
        """生成多个重叠圆形组成的簇"""
        h, w = shape
        cluster_mask = np.zeros(shape, dtype=bool)

        # 确定圆的数量
        num_circles = np.random.randint(2, 5)
        pixels_per_circle = target_pixels // num_circles

        for i in range(num_circles):
            # 在主中心附近选择子圆心
            offset_range = min(max_radius // 3, 6)
            sub_center_x = center_x + np.random.randint(-offset_range, offset_range + 1)
            sub_center_y = center_y + np.random.randint(-offset_range, offset_range + 1)

            # 确保在范围内
            sub_center_x = max(0, min(h - 1, sub_center_x))
            sub_center_y = max(0, min(w - 1, sub_center_y))

            # 计算子圆半径
            if i == num_circles - 1:
                # 最后一个圆调整以达到目标像素数
                remaining_pixels = target_pixels - np.sum(cluster_mask)
                radius = (
                    np.sqrt(remaining_pixels / np.pi) if remaining_pixels > 0 else 0
                )
            else:
                radius = np.sqrt(pixels_per_circle / np.pi)

            radius = min(radius, max_radius * 0.6)

            if radius > 0:
                # 生成子圆
                y, x = np.ogrid[:h, :w]
                dist = np.sqrt((x - sub_center_y) ** 2 + (y - sub_center_x) ** 2)
                sub_circle = dist <= radius

                cluster_mask = cluster_mask | sub_circle

        return cluster_mask

    def _generate_growth_cluster_local(
        self, shape, center_x, center_y, max_radius, target_pixels
    ):
        """使用区域生长方法生成集中的缺失区域 - 增加不规则性"""
        h, w = shape
        cluster_mask = np.zeros(shape, dtype=bool)

        # 从中心点开始
        if 0 <= center_x < h and 0 <= center_y < w:
            cluster_mask[center_x, center_y] = True
            current_pixels = 1
        else:
            return cluster_mask

        # 定义邻域(8连通)
        neighbors = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

        # 区域生长参数
        max_iterations = target_pixels * 2
        iteration = 0

        # 增加生长的随机性和方向偏好
        growth_randomness = np.random.uniform(0.3, 0.8)
        directional_bias = np.random.choice(
            [None, "horizontal", "vertical", "diagonal"]
        )

        while current_pixels < target_pixels and iteration < max_iterations:
            iteration += 1

            # 找到当前区域的边界点
            boundary_points = []

            for i in range(
                max(0, center_x - max_radius), min(h, center_x + max_radius + 1)
            ):
                for j in range(
                    max(0, center_y - max_radius), min(w, center_y + max_radius + 1)
                ):
                    if cluster_mask[i, j]:
                        # 检查8邻域
                        for di, dj in neighbors:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < h and 0 <= nj < w and not cluster_mask[ni, nj]:

                                # 计算距离中心的距离
                                dist = np.sqrt(
                                    (ni - center_x) ** 2 + (nj - center_y) ** 2
                                )
                                if dist <= max_radius:
                                    # 添加方向偏好权重
                                    bias_weight = 1.0
                                    if directional_bias == "horizontal":
                                        bias_weight = 1.0 + 0.5 * (
                                            1.0 - abs(di) / max(abs(di) + abs(dj), 1)
                                        )
                                    elif directional_bias == "vertical":
                                        bias_weight = 1.0 + 0.5 * (
                                            1.0 - abs(dj) / max(abs(di) + abs(dj), 1)
                                        )
                                    elif directional_bias == "diagonal":
                                        bias_weight = 1.0 + 0.3 * (abs(di) == abs(dj))

                                    boundary_points.append((ni, nj, dist, bias_weight))

            if not boundary_points:
                break

            # 根据距离和偏好权重排序
            boundary_points.sort(key=lambda x: x[2] / x[3])

            # 计算本次要添加的点数（增加随机性）
            remaining_pixels = target_pixels - current_pixels
            base_add = max(1, remaining_pixels // np.random.randint(3, 8))
            num_to_add = min(len(boundary_points), base_add, remaining_pixels)

            # 根据随机性决定选择策略
            if np.random.random() < growth_randomness:
                # 随机选择
                selected_indices = np.random.choice(
                    min(len(boundary_points), num_to_add * 2),
                    size=num_to_add,
                    replace=False,
                )
                selected_points = [boundary_points[i] for i in selected_indices]
            else:
                # 选择最优的点
                selected_points = boundary_points[:num_to_add]

            # 添加选中的点
            for ni, nj, _, _ in selected_points:
                if current_pixels >= target_pixels:
                    break
                cluster_mask[ni, nj] = True
                current_pixels += 1

        return cluster_mask

    def _generate_scattered_mask(
        self, mask: np.ndarray, target_pixels: int, local_size: int
    ) -> np.ndarray:
        """
        生成分散分布的mask，确保所有小簇都远离边缘
        """
        h, w = mask.shape

        # 多个小簇分散分布
        num_small_clusters = random.randint(3, min(8, max(3, local_size // 8)))
        pixels_per_cluster = max(1, target_pixels // num_small_clusters)

        # 增大边距，确保小簇不靠近边缘
        margin = max(3, local_size // 6)  # 更大的边距

        for cluster_idx in range(num_small_clusters):
            if np.sum(mask == 0) >= target_pixels:
                break

            cluster_pixels = min(pixels_per_cluster, target_pixels - np.sum(mask == 0))
            if cluster_pixels <= 0:
                continue

            # 随机选择小簇中心，确保远离边缘
            if margin * 2 >= min(h, w):
                center_y = h // 2
                center_x = w // 2
            else:
                center_y = random.randint(margin, h - margin - 1)
                center_x = random.randint(margin, w - margin - 1)

            # 生成小的圆形簇
            if cluster_pixels <= 4:
                # 很小的簇：直接设置几个点
                mask[center_y, center_x] = 0
                directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                for i, (dy, dx) in enumerate(directions):
                    if i >= cluster_pixels - 1:
                        break
                    ny, nx = center_y + dy, center_x + dx
                    if margin <= ny < h - margin and margin <= nx < w - margin:
                        mask[ny, nx] = 0
            else:
                # 较大簇：小圆形，但确保不超出边界
                radius = min(np.sqrt(cluster_pixels / np.pi), margin - 1)
                y, x = np.ogrid[:h, :w]
                dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                small_cluster = dist <= radius

                # 只在内部区域应用
                for i in range(h):
                    for j in range(w):
                        if (
                            small_cluster[i, j]
                            and margin <= i < h - margin
                            and margin <= j < w - margin
                        ):
                            mask[i, j] = 0

        return mask

    def _generate_compact_cluster(
        self,
        shape: Tuple[int, int],
        center_y: int,
        center_x: int,
        target_pixels: int,
        max_radius: int,
    ) -> np.ndarray:
        """
        生成紧凑的簇，严格限制在指定半径内，确保不触及边缘
        """
        h, w = shape
        cluster_mask = np.zeros(shape, dtype=bool)

        # 验证中心点是否有效
        if not (0 <= center_y < h and 0 <= center_x < w):
            return cluster_mask

        # 确保最大半径不会导致触及边缘
        margin = 1  # 至少3像素的边距
        max_allowed_radius = min(
            center_y - margin,
            center_x - margin,
            h - center_y - margin - 1,
            w - center_x - margin - 1,
            max_radius,
        )

        if max_allowed_radius <= 0:
            # 如果没有足够空间，只设置中心点
            cluster_mask[center_y, center_x] = True
            return cluster_mask

        # 从中心开始
        cluster_mask[center_y, center_x] = True
        current_pixels = 1

        # 候选点队列
        candidates = [(center_y, center_x)]
        processed = {(center_y, center_x)}

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 只使用4连通

        while current_pixels < target_pixels and candidates:
            # 从候选点中选择距离中心最近的
            candidates.sort(
                key=lambda p: (p[0] - center_y) ** 2 + (p[1] - center_x) ** 2
            )

            new_candidates = []

            for y, x in candidates:
                if current_pixels >= target_pixels:
                    break

                # 添加邻居点
                for dy, dx in directions:
                    ny, nx = y + dy, x + dx

                    if (ny, nx) in processed:
                        continue

                    # 检查是否在范围内
                    if not (0 <= ny < h and 0 <= nx < w):
                        processed.add((ny, nx))
                        continue

                    # 检查距离限制，确保不超出允许的半径
                    dist = np.sqrt((ny - center_y) ** 2 + (nx - center_x) ** 2)
                    if dist > max_allowed_radius:
                        processed.add((ny, nx))
                        continue

                    # 添加这个点
                    cluster_mask[ny, nx] = True
                    current_pixels += 1
                    processed.add((ny, nx))
                    new_candidates.append((ny, nx))

                    if current_pixels >= target_pixels:
                        break

            candidates = new_candidates

        return cluster_mask

    def _generate_cluster_shape(
        self, shape: Tuple[int, int], center_y: int, center_x: int, target_pixels: int
    ) -> np.ndarray:
        """生成多样化的簇形状"""
        h, w = shape
        cluster_mask = np.zeros(shape, dtype=bool)

        # 随机选择簇形状
        shape_type = random.choice(["ellipse", "growth", "irregular"])

        if shape_type == "ellipse":
            cluster_mask = self._generate_ellipse_cluster_new(
                shape, center_y, center_x, target_pixels
            )
        elif shape_type == "growth":
            cluster_mask = self._generate_growth_cluster_new(
                shape, center_y, center_x, target_pixels
            )
        else:  # irregular
            cluster_mask = self._generate_irregular_cluster(
                shape, center_y, center_x, target_pixels
            )

        return cluster_mask

    def _generate_ellipse_cluster_new(
        self, shape: Tuple[int, int], center_y: int, center_x: int, target_pixels: int
    ) -> np.ndarray:
        """生成椭圆形簇"""
        h, w = shape
        cluster_mask = np.zeros(shape, dtype=bool)

        if not SCIPY_AVAILABLE:
            # 简化版：圆形
            radius = np.sqrt(target_pixels / np.pi)
            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            cluster_mask = dist <= radius
            return cluster_mask

        # 估算椭圆参数
        aspect_ratio = random.uniform(0.5, 1.0)  # 椭圆长短轴比例

        # 计算半长轴和半短轴
        area = target_pixels
        semi_major = np.sqrt(area / (np.pi * aspect_ratio))
        semi_minor = semi_major * aspect_ratio

        # 限制在合理范围内
        max_radius = min(h // 3, w // 3)
        semi_major = min(semi_major, max_radius)
        semi_minor = min(semi_minor, max_radius)

        try:
            # 生成椭圆
            rr, cc = ellipse(center_y, center_x, semi_major, semi_minor, shape=shape)
            cluster_mask[rr, cc] = True
        except:
            # 回退到圆形
            radius = min(semi_major, semi_minor)
            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            cluster_mask = dist <= radius

        return cluster_mask

    def _generate_growth_cluster_new(
        self, shape: Tuple[int, int], center_y: int, center_x: int, target_pixels: int
    ) -> np.ndarray:
        """使用区域生长生成簇"""
        h, w = shape
        cluster_mask = np.zeros(shape, dtype=bool)

        if not (0 <= center_y < h and 0 <= center_x < w):
            return cluster_mask

        # 从中心点开始生长
        cluster_mask[center_y, center_x] = True
        current_pixels = 1

        # 8连通邻域
        directions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

        # 区域生长
        max_iterations = target_pixels * 3
        iteration = 0

        while current_pixels < target_pixels and iteration < max_iterations:
            iteration += 1

            # 找到边界点
            boundary_points = []
            for y in range(h):
                for x in range(w):
                    if cluster_mask[y, x]:
                        for dy, dx in directions:
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w and not cluster_mask[ny, nx]:
                                dist = np.sqrt(
                                    (ny - center_y) ** 2 + (nx - center_x) ** 2
                                )
                                boundary_points.append((ny, nx, dist))

            if not boundary_points:
                break

            # 去重并按距离排序
            boundary_points = list(set(boundary_points))
            boundary_points.sort(key=lambda x: x[2])

            # 选择要添加的点
            remaining = target_pixels - current_pixels
            num_to_add = min(len(boundary_points), max(1, remaining // 3), remaining)

            selected_points = boundary_points[:num_to_add]

            for ny, nx, _ in selected_points:
                if current_pixels >= target_pixels:
                    break
                cluster_mask[ny, nx] = True
                current_pixels += 1

        return cluster_mask

    def _generate_irregular_cluster(
        self, shape: Tuple[int, int], center_y: int, center_x: int, target_pixels: int
    ) -> np.ndarray:
        """生成不规则形状的簇"""
        h, w = shape
        cluster_mask = np.zeros(shape, dtype=bool)

        # 使用多个小圆的联合来创建不规则形状
        num_circles = random.randint(2, 4)
        pixels_per_circle = target_pixels // num_circles

        for i in range(num_circles):
            # 在中心点附近选择子圆心
            offset_range = min(h // 4, w // 4, 8)
            sub_center_y = center_y + random.randint(-offset_range, offset_range)
            sub_center_x = center_x + random.randint(-offset_range, offset_range)

            # 确保在范围内
            sub_center_y = max(0, min(h - 1, sub_center_y))
            sub_center_x = max(0, min(w - 1, sub_center_x))

            # 计算子圆半径
            radius = np.sqrt(pixels_per_circle / np.pi)

            # 生成子圆
            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - sub_center_x) ** 2 + (y - sub_center_y) ** 2)
            sub_circle = dist <= radius

            cluster_mask = cluster_mask | sub_circle

        return cluster_mask

    def _extract_local_region(
        self, global_data: np.ndarray, global_mask: np.ndarray, local_size: int
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """
        从global数据中提取local区域 - 修正版本
        确保提取的local区域中global_mask=1的部分占绝大多数
        """
        h, w = global_data.shape
        half_size = local_size // 2

        # 寻找有效的中心点位置，更严格的条件
        valid_centers = []
        for y in range(half_size, h - half_size):
            for x in range(half_size, w - half_size):
                # 提取候选区域
                local_region_mask = global_mask[
                    y - half_size : y + half_size + 1, x - half_size : x + half_size + 1
                ]

                # 检查global边缘区域的比例 - 更严格的条件
                valid_pixels = np.sum(local_region_mask == 1)
                valid_ratio = valid_pixels / (local_size * local_size)

                # 确保至少95%的区域是global_mask=1
                if valid_ratio >= 0.95:
                    valid_centers.append((y, x))

        if not valid_centers:
            # 如果没有95%的区域，降低到90%
            for y in range(half_size, h - half_size):
                for x in range(half_size, w - half_size):
                    local_region_mask = global_mask[
                        y - half_size : y + half_size + 1,
                        x - half_size : x + half_size + 1,
                    ]
                    valid_pixels = np.sum(local_region_mask == 1)
                    valid_ratio = valid_pixels / (local_size * local_size)

                    if valid_ratio >= 0.90:
                        valid_centers.append((y, x))

        if not valid_centers:
            # 如果还没有，降低到80%
            for y in range(half_size, h - half_size):
                for x in range(half_size, w - half_size):
                    local_region_mask = global_mask[
                        y - half_size : y + half_size + 1,
                        x - half_size : x + half_size + 1,
                    ]
                    valid_pixels = np.sum(local_region_mask == 1)
                    valid_ratio = valid_pixels / (local_size * local_size)

                    if valid_ratio >= 0.80:
                        valid_centers.append((y, x))

        if not valid_centers:
            # 最后回退：选择中心区域
            center_y = h // 2
            center_x = w // 2
        else:
            center_y, center_x = random.choice(valid_centers)

        # 提取local区域
        local_target = global_data[
            center_y - half_size : center_y + half_size + 1,
            center_x - half_size : center_x + half_size + 1,
        ].copy()
        local_global_mask = global_mask[
            center_y - half_size : center_y + half_size + 1,
            center_x - half_size : center_x + half_size + 1,
        ].copy()

        return local_target, local_global_mask, (center_y, center_x)

    def _generate_single_sample(
        self, reference_data: np.ndarray, sample_idx: int
    ) -> Dict:
        """生成单个训练样本 - 修正版本，确保local mask边缘为1"""

        # 1. 生成initial mask
        initial_mask = self._generate_initial_mask(reference_data)

        # 2. 生成global mask
        coverage_ratio = random.uniform(*self.global_coverage_range)
        global_mask = self._generate_global_mask(initial_mask, coverage_ratio)

        # 3. 随机选择local尺寸
        local_size = random.choice(self.local_sizes)

        # 4. 提取local区域 - 确保大部分区域都是global_mask=1
        local_target, local_global_mask, (center_y, center_x) = (
            self._extract_local_region(reference_data, global_mask, local_size)
        )

        # 5. 生成local synthetic mask - 在完全有效的区域内生成
        missing_ratio = random.uniform(*self.local_missing_range)

        # 检查local_global_mask的有效区域比例
        valid_ratio = np.mean(local_global_mask)

        if valid_ratio >= 0.95:
            # 如果95%以上区域有效，正常生成local mask

            local_synthetic_mask = self._generate_local_mask(
                local_size, missing_ratio, self.enable_edge_connection
            )
        else:
            # 如果有效区域不足95%，只在有效区域内生成缺失
            local_synthetic_mask = self._generate_local_mask_in_valid_region(
                local_global_mask, missing_ratio
            )

        # 6. 合并local mask (确保global无效区域也为0)
        local_mask = local_synthetic_mask * local_global_mask

        # 7. 生成local input
        local_input = local_target.copy()

        # 计算local_mask=1区域的均值用于填充local_mask=0的区域
        valid_local_values = local_input[local_mask == 1]
        if len(valid_local_values) > 0:
            local_fill_value = np.mean(valid_local_values)
        else:
            local_fill_value = 0.0

        # 将local_mask=0的区域设为local_mask=1区域的均值
        local_input[local_mask == 0] = local_fill_value

        # 8. 生成global input
        global_input = reference_data.copy()

        # 计算global_mask=1区域的均值作为全局填充值
        global_valid_values = global_input[global_mask == 1]
        if len(global_valid_values) > 0:
            global_fill_value = np.mean(global_valid_values)
        else:
            global_fill_value = 0.0

        # 先将global_mask=0的区域设为global均值
        global_input[global_mask == 0] = global_fill_value

        # 然后将local区域中local_mask=0的部分也设为global均值
        half_size = local_size // 2
        local_start_y = center_y - half_size
        local_end_y = center_y + half_size + 1
        local_start_x = center_x - half_size
        local_end_x = center_x + half_size + 1

        # 创建一个新的global_mask，将local缺失区域也标记为0
        global_mask_updated = global_mask.copy()
        local_missing_mask = local_mask == 0
        global_mask_updated[local_start_y:local_end_y, local_start_x:local_end_x][
            local_missing_mask
        ] = 0

        # 对应地，将global_input中的这些区域也设为global均值
        global_input[local_start_y:local_end_y, local_start_x:local_end_x][
            local_missing_mask
        ] = global_fill_value

        # 9. 生成metadata
        metadata = np.array(
            [
                center_y,
                center_x,  # local中心在global中的位置
                local_size,  # local尺寸
                sample_idx,  # 样本索引
                coverage_ratio,  # global覆盖率
                missing_ratio,  # local缺失率
                local_size,  # 重复local_size以保持兼容性
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

    def _generate_local_mask_in_valid_region(
        self, local_global_mask: np.ndarray, missing_ratio: float
    ) -> np.ndarray:
        """
        在有效区域内生成local mask，避免在global无效区域生成缺失
        """
        h, w = local_global_mask.shape
        local_mask = local_global_mask.copy()  # 从global mask开始

        # 找到所有有效位置
        valid_positions = np.where(local_global_mask == 1)
        total_valid_pixels = len(valid_positions[0])

        if total_valid_pixels == 0:
            return local_mask

        # 计算要移除的像素数（只在有效区域内）
        target_missing_pixels = int(total_valid_pixels * missing_ratio)
        if target_missing_pixels <= 0:
            return local_mask

        # 在有效区域内生成集中的缺失区域
        if target_missing_pixels < total_valid_pixels * 0.8:  # 如果缺失不超过80%
            # 选择一个或几个中心点进行集中缺失
            num_clusters = random.randint(
                1, min(3, max(1, target_missing_pixels // 20))
            )
            pixels_per_cluster = target_missing_pixels // num_clusters

            removed_pixels = 0

            for cluster_idx in range(num_clusters):
                if removed_pixels >= target_missing_pixels:
                    break

                cluster_pixels = min(
                    pixels_per_cluster, target_missing_pixels - removed_pixels
                )

                # 随机选择一个有效位置作为中心
                valid_indices = np.where(local_mask == 1)
                if len(valid_indices[0]) == 0:
                    break

                center_idx = random.randint(0, len(valid_indices[0]) - 1)
                center_y = valid_indices[0][center_idx]
                center_x = valid_indices[1][center_idx]

                # 从中心点开始区域生长式移除
                to_remove = [(center_y, center_x)]
                removed_in_cluster = 0
                processed = set()

                directions = [
                    (-1, -1),
                    (-1, 0),
                    (-1, 1),
                    (0, -1),
                    (0, 1),
                    (1, -1),
                    (1, 0),
                    (1, 1),
                ]

                while to_remove and removed_in_cluster < cluster_pixels:
                    y, x = to_remove.pop(0)

                    if (y, x) in processed:
                        continue

                    if 0 <= y < h and 0 <= x < w and local_mask[y, x] == 1:
                        local_mask[y, x] = 0
                        removed_pixels += 1
                        removed_in_cluster += 1
                        processed.add((y, x))

                        # 添加邻居
                        for dy, dx in directions:
                            ny, nx = y + dy, x + dx
                            if (
                                0 <= ny < h
                                and 0 <= nx < w
                                and (ny, nx) not in processed
                                and local_mask[ny, nx] == 1
                                and random.random() < 0.7
                            ):
                                to_remove.append((ny, nx))

                        if removed_pixels >= target_missing_pixels:
                            break
        else:
            # 如果缺失比例很高，随机移除
            valid_indices = list(zip(valid_positions[0], valid_positions[1]))
            random.shuffle(valid_indices)

            for i, (y, x) in enumerate(valid_indices):
                if i >= target_missing_pixels:
                    break
                local_mask[y, x] = 0

        return local_mask

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """获取训练样本"""

        # 计算文件索引和样本索引
        file_idx = idx // self.samples_per_file
        sample_idx = idx % self.samples_per_file

        # 获取参考文件
        file_path = self.reference_files[file_idx]

        try:
            # 加载参考数据
            reference_data = self._load_reference_data(file_path)

            # 生成样本
            sample = self._generate_single_sample(reference_data, sample_idx)

            # 转换为tensor
            return (
                torch.tensor(sample["local_input"], dtype=torch.float32).unsqueeze(
                    0
                ),  # (1, H, W)
                torch.tensor(sample["local_mask"], dtype=torch.float32).unsqueeze(
                    0
                ),  # (1, H, W)
                torch.tensor(sample["local_target"], dtype=torch.float32),  # (H, W)
                torch.tensor(sample["global_input"], dtype=torch.float32).unsqueeze(
                    0
                ),  # (1, H, W)
                torch.tensor(sample["global_mask"], dtype=torch.float32).unsqueeze(
                    0
                ),  # (1, H, W)
                torch.tensor(sample["global_target"], dtype=torch.float32),  # (H, W)
                sample["metadata"].astype(np.int32),  # metadata
            )

        except Exception as e:
            print(f" 生成样本失败 (idx={idx}): {e}")
            # 返回fallback数据
            return self._get_fallback_sample()

    def _get_fallback_sample(self) -> Tuple[torch.Tensor, ...]:
        """生成fallback样本"""
        local_size = self.local_sizes[0]  # 使用最小尺寸

        # 创建简单的测试数据
        local_input = torch.randn(1, local_size, local_size) * 0.1 + 0.5
        local_mask = torch.ones(1, local_size, local_size)
        local_target = torch.randn(local_size, local_size) * 0.1 + 0.5
        global_input = torch.randn(1, 600, 600) * 0.1 + 0.5
        global_mask = torch.ones(1, 600, 600)
        global_target = torch.randn(600, 600) * 0.1 + 0.5
        metadata = np.array(
            [300, 300, local_size, 0, 0.8, 0.5, local_size], dtype=np.int32
        )

        return (
            local_input,
            local_mask,
            local_target,
            global_input,
            global_mask,
            global_target,
            metadata,
        )

    def get_sample_info(self, idx: int) -> Dict:
        """获取样本信息(用于调试)"""
        file_idx = idx // self.samples_per_file
        sample_idx = idx % self.samples_per_file
        file_path = self.reference_files[file_idx]

        return {
            "file_idx": file_idx,
            "sample_idx": sample_idx,
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
        }

    def _visualize_samples(self, num_samples: int):
        """可视化生成的样本"""
        print(f" 正在可视化 {num_samples} 个样本...")

        fig, axes = plt.subplots(num_samples, 6, figsize=(18, 3 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_samples):
            try:
                # 获取样本
                sample_data = self[i * (self.total_samples // num_samples)]
                (
                    local_input,
                    local_mask,
                    local_target,
                    global_input,
                    global_mask,
                    global_target,
                    metadata,
                ) = sample_data

                # 转换为numpy用于显示
                local_input_np = local_input.squeeze().numpy()
                local_mask_np = local_mask.squeeze().numpy()
                local_target_np = local_target.numpy()
                global_input_np = global_input.squeeze().numpy()
                global_mask_np = global_mask.squeeze().numpy()
                global_target_np = global_target.numpy()

                # 获取尺寸信息
                local_size = metadata[2]
                center_y, center_x = metadata[0], metadata[1]

                # 计算颜色范围
                all_valid = np.concatenate(
                    [
                        local_target_np[local_target_np != 0],
                        global_target_np[global_target_np != 0],
                    ]
                )
                if len(all_valid) > 0:
                    vmin, vmax = np.percentile(all_valid, [2, 98])
                else:
                    vmin, vmax = 0, 1

                # 绘制图像
                axes[i, 0].imshow(
                    global_target_np, cmap="terrain", vmin=vmin, vmax=vmax
                )
                axes[i, 0].set_title(f"Global Target\n{global_target_np.shape}")
                axes[i, 0].plot(
                    center_x, center_y, "r+", markersize=10, markeredgewidth=2
                )

                axes[i, 1].imshow(global_mask_np, cmap="binary")
                axes[i, 1].set_title(
                    f"Global Mask\n覆盖率: {np.mean(global_mask_np):.1%}"
                )

                axes[i, 2].imshow(global_input_np, cmap="terrain", vmin=vmin, vmax=vmax)
                axes[i, 2].set_title("Global Input")

                axes[i, 3].imshow(local_target_np, cmap="terrain", vmin=vmin, vmax=vmax)
                axes[i, 3].set_title(f"Local Target\n{local_size}x{local_size}")

                axes[i, 4].imshow(local_mask_np, cmap="binary")
                missing_ratio = 1 - np.mean(local_mask_np)
                # 检查是否与边缘相接
                edge_connected = self._check_edge_connection(local_mask_np)
                edge_text = " (边缘)" if edge_connected else ""
                axes[i, 4].set_title(
                    f"Local Mask\n缺失: {missing_ratio:.1%}{edge_text}"
                )

                # 显示masked local input
                masked_local = local_input_np.copy()
                masked_local[local_mask_np == 0] = np.nan
                axes[i, 5].imshow(masked_local, cmap="terrain", vmin=vmin, vmax=vmax)
                axes[i, 5].set_title("Local Input (Masked)")

                # 移除坐标轴
                for ax in axes[i]:
                    ax.set_xticks([])
                    ax.set_yticks([])

            except Exception as e:
                print(f" 可视化样本 {i} 失败: {e}")
                for ax in axes[i]:
                    ax.text(
                        0.5,
                        0.5,
                        f"Error: {e}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )

        plt.tight_layout()
        plt.show()
        plt.pause(3.0)
        plt.close()

    def _check_edge_connection(self, mask: np.ndarray) -> bool:
        """检查mask是否与边缘相接"""
        h, w = mask.shape

        # 检查四个边缘是否有缺失值(0)
        edges = [
            mask[0, :],  # 上边缘
            mask[-1, :],  # 下边缘
            mask[:, 0],  # 左边缘
            mask[:, -1],  # 右边缘
        ]

        for edge in edges:
            if np.any(edge == 0):
                return True

        return False


def multiscale_collate_fn(batch):
    """
    多尺寸数据的collate函数
    处理不同尺寸的local数据
    """
    # 按local尺寸分组
    size_groups = {}

    for item in batch:
        if item is not None and len(item) == 7:
            local_size = item[6][2]  # metadata中的local_size
            if local_size not in size_groups:
                size_groups[local_size] = []
            size_groups[local_size].append(item)

    # 如果只有一种尺寸，直接处理
    if len(size_groups) == 1:
        size = list(size_groups.keys())[0]
        items = size_groups[size]

        local_inputs = torch.stack([item[0] for item in items])
        local_masks = torch.stack([item[1] for item in items])
        local_targets = torch.stack([item[2] for item in items])
        global_inputs = torch.stack([item[3] for item in items])
        global_masks = torch.stack([item[4] for item in items])
        global_targets = torch.stack([item[5] for item in items])
        metadata = [item[6] for item in items]

        return (
            local_inputs,
            local_masks,
            local_targets,
            global_inputs,
            global_masks,
            global_targets,
            metadata,
        )

    else:
        # 多种尺寸：选择最常见的尺寸
        size_counts = {size: len(items) for size, items in size_groups.items()}
        most_common_size = max(size_counts, key=size_counts.get)

        items = size_groups[most_common_size]

        # 如果数量不够一个batch，用其他尺寸补充
        target_batch_size = len(batch)
        while len(items) < target_batch_size:
            for size, group_items in size_groups.items():
                if size != most_common_size and group_items:
                    items.append(group_items[0])
                    if len(items) >= target_batch_size:
                        break
            break

        items = items[:target_batch_size]  # 截断到目标大小

        local_inputs = torch.stack([item[0] for item in items])
        local_masks = torch.stack([item[1] for item in items])
        local_targets = torch.stack([item[2] for item in items])
        global_inputs = torch.stack([item[3] for item in items])
        global_masks = torch.stack([item[4] for item in items])
        global_targets = torch.stack([item[5] for item in items])
        metadata = [item[6] for item in items]

        return (
            local_inputs,
            local_masks,
            local_targets,
            global_inputs,
            global_masks,
            global_targets,
            metadata,
        )


def create_multiscale_dataloader(
    reference_dir: str,
    local_sizes: List[int] = [64, 128, 256],
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    **dataset_kwargs,
) -> DataLoader:
    """
    创建多尺寸DEM数据加载器

    Args:
        reference_dir: 参考文件目录
        local_sizes: local数据尺寸列表
        batch_size: 批次大小
        shuffle: 是否随机打乱
        num_workers: 工作进程数
        **dataset_kwargs: 传递给数据集的其他参数

    Returns:
        DataLoader实例
    """

    dataset = MultiScaleDEMDataset(
        reference_dir=reference_dir, local_sizes=local_sizes, **dataset_kwargs
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=multiscale_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else 2,
    )

    return dataloader


def analyze_dataset_statistics(dataset: MultiScaleDEMDataset, num_samples: int = 100):
    """分析数据集统计信息"""
    print(f" 分析数据集统计信息 (采样 {num_samples} 个样本)...")

    local_size_counts = {size: 0 for size in dataset.local_sizes}
    global_coverage_stats = []
    local_missing_stats = []
    edge_connection_count = 0

    sample_indices = np.random.choice(
        len(dataset), min(num_samples, len(dataset)), replace=False
    )

    for idx in tqdm(sample_indices, desc="统计分析"):
        try:
            sample_data = dataset[idx]
            metadata = sample_data[6]

            local_size = metadata[2]
            local_size_counts[local_size] += 1

            global_coverage = metadata[4]
            local_missing = metadata[5]

            global_coverage_stats.append(global_coverage)
            local_missing_stats.append(local_missing)

            # 检查是否与边缘相接
            local_mask = sample_data[1].squeeze().numpy()
            if dataset._check_edge_connection(local_mask):
                edge_connection_count += 1

        except Exception as e:
            print(f" 统计样本 {idx} 失败: {e}")
            continue

    # 打印统计结果
    print(f"\n📈 数据集统计结果:")
    print(f"Local尺寸分布:")
    for size, count in local_size_counts.items():
        ratio = count / sum(local_size_counts.values()) * 100
        print(f"   {size}x{size}: {count} ({ratio:.1f}%)")

    if global_coverage_stats:
        print(f"\nGlobal覆盖率统计:")
        print(f"   均值: {np.mean(global_coverage_stats):.3f}")
        print(
            f"   范围: [{np.min(global_coverage_stats):.3f}, {np.max(global_coverage_stats):.3f}]"
        )

        print(f"\nLocal缺失率统计:")
        print(f"   均值: {np.mean(local_missing_stats):.3f}")
        print(
            f"   范围: [{np.min(local_missing_stats):.3f}, {np.max(local_missing_stats):.3f}]"
        )

        actual_edge_ratio = edge_connection_count / len(sample_indices) * 100
        print(
            f"\n边缘相接样本: {edge_connection_count}/{len(sample_indices)} ({actual_edge_ratio:.1f}%)"
        )
        print(f"   设置概率: {dataset.edge_connection_prob:.1%}")
        print(f"   实际比例: {actual_edge_ratio:.1f}%")


import os
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pickle
import h5py
from tqdm import tqdm
import json
from datetime import datetime


class PreGeneratedDEMDataset(Dataset):
    """预生成的DEM数据集，从文件加载"""

    def __init__(self, data_dir, file_format="npz"):
        """
        Args:
            data_dir: 预生成数据的目录
            file_format: 数据格式 ('npz', 'h5', 'pt')
        """
        self.data_dir = Path(data_dir)
        self.file_format = file_format

        # 读取元数据
        with open(self.data_dir / "metadata.json", "r") as f:
            self.metadata = json.load(f)

        self.total_samples = self.metadata["total_samples"]
        self.local_sizes = self.metadata["local_sizes"]

        # 根据格式设置文件列表
        if file_format == "h5":
            self.data_file = self.data_dir / "data.h5"
            self._open_h5_file()
        else:
            self.sample_files = sorted(self.data_dir.glob(f"sample_*.{file_format}"))

    def _open_h5_file(self):
        """打开HDF5文件"""
        self.h5_file = h5py.File(self.data_file, "r")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        if self.file_format == "npz":
            # NPZ格式
            file_path = self.data_dir / f"sample_{idx:06d}.npz"
            data = np.load(file_path)

            return (
                torch.from_numpy(data["local_input"])
                .float()
                .unsqueeze(0),  # 添加channel维度
                torch.from_numpy(data["local_mask"]).float().unsqueeze(0),
                torch.from_numpy(data["local_target"]).float(),  # 没有channel维度
                torch.from_numpy(data["global_input"]).float().unsqueeze(0),
                torch.from_numpy(data["global_mask"]).float().unsqueeze(0),
                torch.from_numpy(data["global_target"]).float(),
                data["metadata"].astype(np.int32),  # 保持与原始数据集一致
            )

        elif self.file_format == "h5":
            # HDF5格式
            group = self.h5_file[f"sample_{idx:06d}"]

            # 重建metadata数组
            metadata = np.array(
                [group.attrs.get(f"metadata_{i}", 0) for i in range(7)], dtype=np.int32
            )

            return (
                torch.from_numpy(group["local_input"][:]).float().unsqueeze(0),
                torch.from_numpy(group["local_mask"][:]).float().unsqueeze(0),
                torch.from_numpy(group["local_target"][:]).float(),
                torch.from_numpy(group["global_input"][:]).float().unsqueeze(0),
                torch.from_numpy(group["global_mask"][:]).float().unsqueeze(0),
                torch.from_numpy(group["global_target"][:]).float(),
                metadata,
            )

        elif self.file_format == "pt":
            # PyTorch格式
            file_path = self.data_dir / f"sample_{idx:06d}.pt"
            data = torch.load(file_path)

            # 如果保存的是字典格式
            if isinstance(data, dict):
                return (
                    data["local_input"],
                    data["local_mask"],
                    data["local_target"],
                    data["global_input"],
                    data["global_mask"],
                    data["global_target"],
                    data["metadata"],
                )
            else:
                # 如果保存的是tuple格式
                return data

    def __del__(self):
        if hasattr(self, "h5_file"):
            self.h5_file.close()


def pregenerate_and_save_dataset(
    dataset, save_dir, file_format="npz", num_samples=None
):
    """
    预生成数据集并保存到文件

    Args:
        dataset: MultiScaleDEMDataset实例
        save_dir: 保存目录
        file_format: 保存格式 ('npz', 'h5', 'pt')
        num_samples: 要生成的样本数，None表示全部

    Returns:
        生成时间统计
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if num_samples is None:
        num_samples = len(dataset)
    else:
        num_samples = min(num_samples, len(dataset))

    print(f"\n 开始预生成数据集...")
    print(f"   保存目录: {save_dir}")
    print(f"   文件格式: {file_format}")
    print(f"   样本数量: {num_samples}")

    start_time = time.time()
    generation_times = []
    save_times = []

    # 保存元数据
    # 修改 pregenerate_and_save_dataset 函数中的元数据部分
    ref_data = dataset._load_reference_data(dataset.reference_files[0])
    metadata_dict = {
        "total_samples": num_samples,
        "local_sizes": dataset.local_sizes,
        "global_size": ref_data.shape,  # 使用实际数据的形状
        "creation_time": datetime.now().isoformat(),
        "global_coverage_range": dataset.global_coverage_range,
        "local_missing_range": dataset.local_missing_range,
        "samples_per_file": dataset.samples_per_file,
    }

    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata_dict, f, indent=2)

    if file_format == "h5":
        # HDF5格式 - 所有数据保存在一个文件中
        h5_file = h5py.File(save_dir / "data.h5", "w")

    # 使用进度条
    with tqdm(total=num_samples, desc="生成数据") as pbar:
        for idx in range(num_samples):
            # 生成数据
            gen_start = time.time()
            data = dataset[idx]
            gen_time = time.time() - gen_start
            generation_times.append(gen_time)

            # 解包数据 - 按照您的__getitem__返回顺序
            (
                local_input,
                local_mask,
                local_target,
                global_input,
                global_mask,
                global_target,
                metadata,
            ) = data

            # 保存数据
            save_start = time.time()

            if file_format == "npz":
                # NPZ格式 - 移除squeeze操作，因为数据已经有channel维度
                np.savez_compressed(
                    save_dir / f"sample_{idx:06d}.npz",
                    local_input=local_input.squeeze(0).numpy(),  # 移除channel维度保存
                    local_mask=local_mask.squeeze(0).numpy(),
                    local_target=local_target.numpy(),  # 这个没有channel维度
                    global_input=global_input.squeeze(0).numpy(),
                    global_mask=global_mask.squeeze(0).numpy(),
                    global_target=global_target.numpy(),
                    metadata=metadata,  # 已经是numpy array
                )

            elif file_format == "h5":
                # HDF5格式
                group = h5_file.create_group(f"sample_{idx:06d}")
                group.create_dataset(
                    "local_input",
                    data=local_input.squeeze(0).numpy(),
                    compression="gzip",
                )
                group.create_dataset(
                    "local_mask", data=local_mask.squeeze(0).numpy(), compression="gzip"
                )
                group.create_dataset(
                    "local_target", data=local_target.numpy(), compression="gzip"
                )
                group.create_dataset(
                    "global_input",
                    data=global_input.squeeze(0).numpy(),
                    compression="gzip",
                )
                group.create_dataset(
                    "global_mask",
                    data=global_mask.squeeze(0).numpy(),
                    compression="gzip",
                )
                group.create_dataset(
                    "global_target", data=global_target.numpy(), compression="gzip"
                )
                # HDF5不支持直接存储numpy object数组，转换为属性
                for i, val in enumerate(metadata):
                    group.attrs[f"metadata_{i}"] = float(val)

            elif file_format == "pt":
                # PyTorch格式 - 直接保存所有tensor
                torch.save(
                    {
                        "local_input": local_input,
                        "local_mask": local_mask,
                        "local_target": local_target,
                        "global_input": global_input,
                        "global_mask": global_mask,
                        "global_target": global_target,
                        "metadata": metadata,
                    },
                    save_dir / f"sample_{idx:06d}.pt",
                )

            save_time = time.time() - save_start
            save_times.append(save_time)

            # 更新进度条
            pbar.update(1)
            pbar.set_postfix(
                {"gen_time": f"{gen_time:.3f}s", "save_time": f"{save_time:.3f}s"}
            )

            # 每100个样本输出一次统计
            if (idx + 1) % 100 == 0:
                print(f"\n  已生成 {idx + 1} 个样本")
                print(f"  当前平均生成时间: {np.mean(generation_times):.3f}秒/样本")
                print(f"  当前平均保存时间: {np.mean(save_times):.3f}秒/样本")

    if file_format == "h5":
        h5_file.close()

    total_time = time.time() - start_time

    # 计算统计信息
    stats = {
        "total_time": total_time,
        "avg_generation_time": np.mean(generation_times),
        "avg_save_time": np.mean(save_times),
        "total_generation_time": np.sum(generation_times),
        "total_save_time": np.sum(save_times),
        "samples_per_second": num_samples / total_time,
    }

    # 保存统计信息
    with open(save_dir / "generation_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n 数据生成完成!")
    print(f"   总时间: {total_time:.2f}秒")
    print(f"   平均生成时间: {stats['avg_generation_time']:.3f}秒/样本")
    print(f"   平均保存时间: {stats['avg_save_time']:.3f}秒/样本")
    print(f"   生成速度: {stats['samples_per_second']:.2f}样本/秒")

    # 计算存储空间
    total_size = sum(f.stat().st_size for f in save_dir.rglob("*")) / (1024**2)
    print(f"   总存储空间: {total_size:.2f} MB")
    print(f"   平均每样本: {total_size/num_samples:.2f} MB")

    return stats


def test_data_loading(data_dir, file_format="npz", num_tests=10):
    """测试数据加载速度"""
    print(f"\n 测试数据加载...")
    print(f"   数据目录: {data_dir}")
    print(f"   文件格式: {file_format}")

    # 创建数据集
    dataset = PreGeneratedDEMDataset(data_dir, file_format)

    # 随机测试
    load_times = []
    indices = np.random.randint(0, len(dataset), num_tests)

    print(f"\n 随机加载测试 ({num_tests}个样本):")
    for i, idx in enumerate(indices):
        start_time = time.time()
        data = dataset[idx]
        load_time = time.time() - start_time
        load_times.append(load_time)
        print(f"   样本 {idx}: {load_time*1000:.2f} ms")

    print(f"\n   平均加载时间: {np.mean(load_times)*1000:.2f} ms")
    print(f"   最大加载时间: {np.max(load_times)*1000:.2f} ms")
    print(f"   最小加载时间: {np.min(load_times)*1000:.2f} ms")

    # 测试DataLoader
    print(f"\n DataLoader测试:")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    batch_times = []
    for i, batch in enumerate(dataloader):
        if i >= 5:  # 只测试5个batch
            break
        start_time = time.time()
        # 模拟数据处理
        _ = batch
        batch_time = time.time() - start_time
        batch_times.append(batch_time)
        print(f"   Batch {i+1}: {batch_time*1000:.2f} ms")

    print(f"\n   平均batch时间: {np.mean(batch_times)*1000:.2f} ms")

    return load_times, batch_times


def compare_formats(dataset, save_base_dir, num_samples=100):
    """比较不同保存格式的性能"""
    formats = ["npz", "h5", "pt"]
    results = {}

    print(f"\n 比较不同保存格式 (各{num_samples}个样本)")
    print("=" * 60)

    for fmt in formats:
        print(f"\n🔸 测试格式: {fmt}")
        save_dir = Path(save_base_dir) / f"format_{fmt}"

        # 生成并保存
        gen_stats = pregenerate_and_save_dataset(
            dataset, save_dir, file_format=fmt, num_samples=num_samples
        )

        # 测试加载
        load_times, batch_times = test_data_loading(save_dir, file_format=fmt)

        results[fmt] = {
            "generation": gen_stats,
            "load_times": load_times,
            "batch_times": batch_times,
        }

    # 打印比较结果
    print("\n 格式比较总结:")
    print("=" * 60)
    print(
        f"{'格式':<10} {'生成时间':<15} {'保存时间':<15} {'加载时间':<15} {'存储空间':<15}"
    )
    print("-" * 60)

    for fmt in formats:
        gen = results[fmt]["generation"]
        load = results[fmt]["load_times"]

        save_dir = Path(save_base_dir) / f"format_{fmt}"
        total_size = sum(f.stat().st_size for f in save_dir.rglob("*")) / (1024**2)

        print(
            f"{fmt:<10} "
            f"{gen['avg_generation_time']*1000:<15.2f} "
            f"{gen['avg_save_time']*1000:<15.2f} "
            f"{np.mean(load)*1000:<15.2f} "
            f"{total_size:<15.2f}"
        )

    return results


# 便捷的批量生成脚本
def batch_pregenerate(
    reference_dir,
    save_dir,
    total_samples=10000,
    samples_per_batch=1000,
    local_sizes=[257],
    file_format="npz",
):
    """
    批量预生成数据集，适合生成大量数据

    Args:
        reference_dir: 参考数据目录
        save_dir: 保存目录
        total_samples: 总样本数
        samples_per_batch: 每批生成的样本数
        local_sizes: local尺寸列表
        file_format: 保存格式
    """
    from datetime import datetime

    # 配置
    config = {
        "reference_dir": reference_dir,
        "local_sizes": local_sizes,
        "samples_per_file": 10,
        "global_coverage_range": (0.5, 1.0),
        "local_missing_range": (0.3, 0.7),
        "max_edge_missing_ratio": 0.3,
        "enable_edge_connection": True,
        "edge_connection_prob": 0.02,
        "seed": 41,
        "cache_reference_files": True,
        "visualization_samples": 0,
    }

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # 记录开始时间
    start_time = datetime.now()
    print(f"\n 开始批量生成数据集")
    print(f"   开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   总样本数: {total_samples}")
    print(f"   每批样本数: {samples_per_batch}")
    print(f"   保存格式: {file_format}")
    print(f"   保存目录: {save_path}")

    # 分批生成
    num_batches = (total_samples + samples_per_batch - 1) // samples_per_batch

    all_stats = []

    for batch_idx in range(num_batches):
        batch_start = batch_idx * samples_per_batch
        batch_end = min((batch_idx + 1) * samples_per_batch, total_samples)
        batch_size = batch_end - batch_start

        print(f"\n 处理批次 {batch_idx + 1}/{num_batches}")
        print(f"   样本范围: {batch_start} - {batch_end-1}")

        # 创建批次目录
        batch_dir = save_path / f"batch_{batch_idx:04d}"

        # 创建数据集（每批都重新创建以避免内存问题）
        dataset = MultiScaleDEMDataset(**config)

        # 生成并保存
        stats = pregenerate_and_save_dataset(
            dataset, batch_dir, file_format=file_format, num_samples=batch_size
        )

        all_stats.append(stats)

        # 清理内存
        del dataset

        # 估计剩余时间
        if batch_idx > 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            avg_time_per_batch = elapsed / (batch_idx + 1)
            remaining_batches = num_batches - batch_idx - 1
            eta = remaining_batches * avg_time_per_batch
            print(f"   预计剩余时间: {eta/60:.1f} 分钟")

    # 创建主元数据文件
    main_metadata = {
        "total_samples": total_samples,
        "num_batches": num_batches,
        "samples_per_batch": samples_per_batch,
        "local_sizes": local_sizes,
        "file_format": file_format,
        "creation_time": start_time.isoformat(),
        "completion_time": datetime.now().isoformat(),
        "batch_dirs": [f"batch_{i:04d}" for i in range(num_batches)],
        "config": config,
    }

    with open(save_path / "main_metadata.json", "w") as f:
        json.dump(main_metadata, f, indent=2)

    # 汇总统计
    total_time = (datetime.now() - start_time).total_seconds()
    avg_gen_time = np.mean([s["avg_generation_time"] for s in all_stats])
    avg_save_time = np.mean([s["avg_save_time"] for s in all_stats])

    print(f"\n 批量生成完成!")
    print(f"   完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   总耗时: {total_time/60:.1f} 分钟")
    print(f"   平均生成速度: {total_samples/total_time:.2f} 样本/秒")
    print(f"   平均生成时间: {avg_gen_time:.3f} 秒/样本")
    print(f"   平均保存时间: {avg_save_time:.3f} 秒/样本")

    # 计算总存储空间
    total_size = sum(f.stat().st_size for f in save_path.rglob("*")) / (1024**3)
    print(f"   总存储空间: {total_size:.2f} GB")
    print(f"   平均每样本: {total_size*1024/total_samples:.2f} MB")


class BatchPreGeneratedDataset(Dataset):
    """从批量生成的目录加载数据集"""

    def __init__(self, data_dir, file_format="npz"):
        self.data_dir = Path(data_dir)
        self.file_format = file_format

        # 读取主元数据
        with open(self.data_dir / "main_metadata.json", "r") as f:
            self.main_metadata = json.load(f)

        self.total_samples = self.main_metadata["total_samples"]
        self.samples_per_batch = self.main_metadata["samples_per_batch"]
        self.batch_dirs = self.main_metadata["batch_dirs"]

        # 预加载每个批次的数据集（延迟加载）
        self.batch_datasets = {}

    def _get_batch_dataset(self, batch_idx):
        """延迟加载批次数据集"""
        if batch_idx not in self.batch_datasets:
            batch_dir = self.data_dir / self.batch_dirs[batch_idx]
            self.batch_datasets[batch_idx] = PreGeneratedDEMDataset(
                batch_dir, self.file_format
            )
        return self.batch_datasets[batch_idx]

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # 计算批次索引和批次内索引
        batch_idx = idx // self.samples_per_batch
        within_batch_idx = idx % self.samples_per_batch

        # 获取对应批次的数据集
        batch_dataset = self._get_batch_dataset(batch_idx)

        # 返回数据
        return batch_dataset[within_batch_idx]

    def __del__(self):
        # 清理所有打开的数据集
        for dataset in self.batch_datasets.values():
            if hasattr(dataset, "__del__"):
                dataset.__del__()


# 使用示例
if __name__ == "__main__":  # 设为True运行批量生成示例
    try:
        # 批量生成大数据集
        batch_pregenerate(
            reference_dir=r"F:\Dataset\Simulate\data0\ST2\statusarray",
            save_dir=r"F:\Dataset\Simulate\data0\ST2\pregenerated_large",
            total_samples=10000,  # 生成5万个样本
            samples_per_batch=1000,  # 每批5000个
            local_sizes=[257],
            file_format="h5",  # 使用HDF5格式节省空间
        )  #  数据集创建成功! 总样本数: {len(dataset)}")

        # 选择操作
        print("\n请选择操作:")
        print("1. 预生成并保存数据集")
        print("2. 测试加载已保存的数据集")
        print("3. 比较不同保存格式")

        choice = input("\n请输入选择 (1/2/3): ").strip()

        if choice == "1":
            # 预生成并保存
            save_dir = (
                Path(PREGENERATE_CONFIG["save_base_dir"])
                / PREGENERATE_CONFIG["file_format"]
            )
            stats = pregenerate_and_save_dataset(
                dataset,
                save_dir,
                file_format=PREGENERATE_CONFIG["file_format"],
                num_samples=PREGENERATE_CONFIG["num_samples"],
            )

            # 测试加载
            print("\n" + "=" * 60)
            test_data_loading(
                save_dir,
                PREGENERATE_CONFIG["file_format"],
                PREGENERATE_CONFIG["test_samples"],
            )

        elif choice == "2":
            # 只测试加载
            save_dir = (
                Path(PREGENERATE_CONFIG["save_base_dir"])
                / PREGENERATE_CONFIG["file_format"]
            )
            if not save_dir.exists():
                print(f" 错误: 目录 {save_dir} 不存在!")
                print("   请先运行选项1生成数据集")
            else:
                test_data_loading(
                    save_dir,
                    PREGENERATE_CONFIG["file_format"],
                    PREGENERATE_CONFIG["test_samples"],
                )

                # 测试实际使用
                print("\n 测试DataLoader实际使用:")
                pre_dataset = PreGeneratedDEMDataset(
                    save_dir, PREGENERATE_CONFIG["file_format"]
                )
                dataloader = DataLoader(
                    pre_dataset, batch_size=8, shuffle=True, num_workers=2
                )

                start_time = time.time()
                for i, batch in enumerate(dataloader):
                    if i >= 10:
                        break
                    # 模拟训练
                    pass

                elapsed = time.time() - start_time
                print(f"   处理10个batch用时: {elapsed:.2f}秒")

        elif choice == "3":
            # 比较格式
            compare_results = compare_formats(
                dataset,
                PREGENERATE_CONFIG["save_base_dir"],
                num_samples=100,  # 每种格式测试100个样本
            )

        else:
            print(" 无效选择!")

    except Exception as e:
        print(f" 错误: {e}")
        import traceback

        traceback.print_exc()

    print("\n 完成!")
"""    # 修改后的main函数
if __name__ == "__main__":
    # 配置参数
    CONFIG = {
        "reference_dir": r"F:\Dataset\Simulate\data1\ST2\statusarray",
        "local_sizes": [257],
        "samples_per_file": 10,
        "global_coverage_range": (0.5, 1.0),
        "local_missing_range": (0.3, 0.7),
        "max_edge_missing_ratio": 0.3,
        "enable_edge_connection": True,
        "edge_connection_prob": 0.02,
        "seed": 41,
        "cache_reference_files": True,
        "visualization_samples": 0,  # 预生成时关闭可视化
    }

    # 预生成配置
    PREGENERATE_CONFIG = {
        "save_base_dir": r"F:\Dataset\Simulate\data1\ST2\pregenerated",
        "num_samples": 1000,  # 预生成1000个样本
        "file_format": "npz",  # 可选: 'npz', 'h5', 'pt'
        "test_samples": 20,  # 测试加载的样本数
    }

    print(" 预生成DEM数据集")
    print("=" * 60)

    try:
        # 创建原始数据集
        print("\n 创建原始数据集...")
        dataset = MultiScaleDEMDataset(**CONFIG)
        print(f" 数据集创建成功! 总样本数: {len(dataset)}")

        # 选择操作
        print("\n请选择操作:")
        print("1. 预生成并保存数据集")
        print("2. 测试加载已保存的数据集")
        print("3. 比较不同保存格式")
        print("4. 验证数据一致性")

        choice = input("\n请输入选择 (1/2/3/4): ").strip()

        if choice == "1":
            # 预生成并保存
            save_dir = (
                Path(PREGENERATE_CONFIG["save_base_dir"])
                / PREGENERATE_CONFIG["file_format"]
            )
            stats = pregenerate_and_save_dataset(
                dataset,
                save_dir,
                file_format=PREGENERATE_CONFIG["file_format"],
                num_samples=PREGENERATE_CONFIG["num_samples"],
            )

            # 测试加载
            print("\n" + "=" * 60)
            test_data_loading(
                save_dir,
                PREGENERATE_CONFIG["file_format"],
                PREGENERATE_CONFIG["test_samples"],
            )

        elif choice == "2":
            # 只测试加载
            save_dir = (
                Path(PREGENERATE_CONFIG["save_base_dir"])
                / PREGENERATE_CONFIG["file_format"]
            )
            if not save_dir.exists():
                print(f" 错误: 目录 {save_dir} 不存在!")
                print("   请先运行选项1生成数据集")
            else:
                test_data_loading(
                    save_dir,
                    PREGENERATE_CONFIG["file_format"],
                    PREGENERATE_CONFIG["test_samples"],
                )

                # 测试实际使用
                print("\n 测试DataLoader实际使用:")
                pre_dataset = PreGeneratedDEMDataset(
                    save_dir, PREGENERATE_CONFIG["file_format"]
                )
                dataloader = DataLoader(
                    pre_dataset, batch_size=8, shuffle=True, num_workers=2
                )

                start_time = time.time()
                for i, batch in enumerate(dataloader):
                    if i >= 10:
                        break
                    # 验证数据形状
                    (
                        local_inputs,
                        local_masks,
                        local_targets,
                        global_inputs,
                        global_masks,
                        global_targets,
                        metadata,
                    ) = batch
                    if i == 0:
                        print(f"   Batch shapes:")
                        print(f"     local_inputs: {local_inputs.shape}")
                        print(f"     local_masks: {local_masks.shape}")
                        print(f"     local_targets: {local_targets.shape}")
                        print(f"     global_inputs: {global_inputs.shape}")
                        print(f"     global_masks: {global_masks.shape}")
                        print(f"     global_targets: {global_targets.shape}")
                        print(f"     metadata: {metadata.shape}")

                elapsed = time.time() - start_time
                print(f"\n   处理10个batch用时: {elapsed:.2f}秒")

        elif choice == "3":
            # 比较格式
            compare_results = compare_formats(
                dataset,
                PREGENERATE_CONFIG["save_base_dir"],
                num_samples=100,  # 每种格式测试100个样本
            )

        elif choice == "4":
            # 验证数据一致性
            print("\n🔍 验证数据一致性...")
            save_dir = (
                Path(PREGENERATE_CONFIG["save_base_dir"])
                / PREGENERATE_CONFIG["file_format"]
            )

            if not save_dir.exists():
                print(f" 错误: 目录 {save_dir} 不存在!")
            else:
                # 创建预生成数据集
                pre_dataset = PreGeneratedDEMDataset(
                    save_dir, PREGENERATE_CONFIG["file_format"]
                )

                # 随机选择几个样本进行比较
                test_indices = np.random.randint(
                    0, min(len(dataset), len(pre_dataset)), 5
                )

                print(f"\n比较 {len(test_indices)} 个样本...")
                all_match = True

                for idx in test_indices:
                    # 从原始数据集获取
                    orig_data = dataset[idx]
                    # 从预生成数据集获取
                    pre_data = pre_dataset[idx]

                    # 比较每个组件
                    match = True
                    for i, (orig, pre) in enumerate(zip(orig_data, pre_data)):
                        if isinstance(orig, torch.Tensor) and isinstance(
                            pre, torch.Tensor
                        ):
                            if not torch.allclose(orig, pre, rtol=1e-5):
                                print(f"   样本 {idx} 的第 {i} 个组件不匹配!")
                                match = False
                                all_match = False
                        elif isinstance(orig, np.ndarray) and isinstance(
                            pre, np.ndarray
                        ):
                            if not np.allclose(orig, pre, rtol=1e-5):
                                print(f"   样本 {idx} 的metadata不匹配!")
                                match = False
                                all_match = False

                    if match:
                        print(f"    样本 {idx} 完全匹配")

                if all_match:
                    print("\n 所有测试样本完全匹配！数据一致性验证通过。")
                else:
                    print("\n 发现数据不一致！")

        else:
            print(" 无效选择!")

    except Exception as e:
        print(f" 错误: {e}")
        import traceback

        traceback.print_exc()

    print("\n 完成!")"""
