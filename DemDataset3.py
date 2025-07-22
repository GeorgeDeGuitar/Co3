import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
import time
import itertools
import random
import os

# 科学计算相关导入
try:
    from scipy.ndimage import binary_dilation, binary_erosion, label
    from skimage.morphology import disk
    from skimage.draw import ellipse, polygon

    SCIPY_AVAILABLE = True
except ImportError:
    print("警告: scipy或skimage未安装，将使用简化的mask生成方法")
    SCIPY_AVAILABLE = False


class DemDataset(Dataset):
    def __init__(
        self,
        jsonDir,
        arrayDir,
        maskDir,
        targetDir,
        transform=None,
        min_valid_pixels=20,
        max_valid_pixels=800,
        analyze_data=False,
        enable_synthetic_masks=True,
        synthetic_ratio=1.0,
    ):
        """
        jsonDir: 存放 json 子文件夹的根目录
        arrayDir: 存放 array (.npy) 文件的目录
        maskDir: 存放 mask (.npy) 文件的目录
        targetDir: 存放 target (rs{id}.xyz) 文件的目录
        transform: 数据增强或归一化（可选）
        min_valid_pixels: local mask中最小有效像素数量
        max_valid_pixels: local mask中最大有效像素数量
        enable_synthetic_masks: 是否启用生成式mask增强
        synthetic_ratio: 生成式数据占原始数据的比例（1.0表示每个原始数据生成1个合成数据）
        """
        self.jsonDir = jsonDir
        self.arrayDir = arrayDir
        self.maskDir = maskDir
        self.targetDir = targetDir
        self.transform = transform
        self.min_valid_pixels = min_valid_pixels
        self.max_valid_pixels = max_valid_pixels
        self.enable_synthetic_masks = enable_synthetic_masks
        self.synthetic_ratio = synthetic_ratio
        self.kernel_size = 33  # 网格大小
        self.target_size = 600  # 目标大小
        self.cachedir = r"E:\KingCrimson Dataset\Simulate\data0\traindata"

        # 先建立global文件的索引
        self.global_file_index = self._build_global_file_index()
        self.original_data_groups = self._groupJsonFiles()
        self.original_length = len(self.original_data_groups)

        # 如果启用生成式mask，计算总长度
        if self.enable_synthetic_masks:
            self.synthetic_length = int(self.original_length * self.synthetic_ratio)
            self.total_length = self.original_length + self.synthetic_length
        else:
            self.total_length = self.original_length

        # 预加载一个有效数据作为fallback
        self.fallback_data = self._load_fallback_data()

        # 数据质量分析
        if analyze_data:
            self.data_quality_stats = self._analyze_data_quality()
            self._adjust_pixel_thresholds()

        # 预加载多个有效数据作为fallback池
        self.fallback_pool = self._load_fallback_pool(pool_size=30)
        self.fallback_index = 0
        self.total_requests = 0  # 添加请求计数器

        print(f"数据集初始化完成:")
        print(f"  原始数据: {self.original_length}")
        if self.enable_synthetic_masks:
            print(f"  生成式数据: {self.synthetic_length}")
            print(f"  总数据量: {self.total_length}")

    def generate_clustered_mask(
        self,
        shape=(33, 33),
        missing_ratio_range=(0.3, 0.7),
        num_clusters_range=(1, 4),
        cluster_size_range=(8, 18),
        scattered_probability=0.2,
    ):  # 非常低的分散概率
        """
        生成缺失mask，主要是集中分布，极少分散分布

        参数:
        - shape: mask形状
        - missing_ratio_range: 缺失像素比例范围
        - num_clusters_range: 缺失区域簇数量范围（1-2个簇）
        - cluster_size_range: 每个簇的大小范围
        - scattered_probability: 生成分散分布mask的概率（0.05表示5%概率）

        返回:
        - mask: 0表示缺失，1表示有效
        """
        # 决定生成集中还是分散的mask（主要是集中）
        if np.random.random() < scattered_probability:
            # 生成分散分布的mask（极少情况）
            return self._generate_scattered_mask(shape, missing_ratio_range)
        else:
            # 生成集中分布的mask（主要情况）
            return self._generate_concentrated_mask_pattern(
                shape, missing_ratio_range, num_clusters_range, cluster_size_range
            )

    def _generate_concentrated_mask_pattern(
        self, shape, missing_ratio_range, num_clusters_range, cluster_size_range
    ):
        """生成集中分布的mask模式"""
        h, w = shape
        mask = np.ones(shape, dtype=np.float32)

        # 随机确定缺失比例和簇数量
        target_missing_ratio = np.random.uniform(*missing_ratio_range)
        num_clusters = np.random.randint(*num_clusters_range)

        total_pixels = h * w
        target_missing_pixels = int(total_pixels * target_missing_ratio)

        for cluster_idx in range(num_clusters):
            # 为每个簇分配像素数
            if cluster_idx == num_clusters - 1:
                # 最后一个簇获得剩余所有像素
                cluster_pixels = target_missing_pixels - np.sum(mask == 0)
            else:
                # 前面的簇平均分配
                cluster_pixels = target_missing_pixels // num_clusters

            if cluster_pixels <= 0:
                continue

            # 选择簇中心，避免太靠近边缘
            margin = 5
            center_x = np.random.randint(margin, h - margin)
            center_y = np.random.randint(margin, w - margin)

            # 生成集中的簇
            cluster_mask = self._generate_concentrated_cluster(
                shape, center_x, center_y, cluster_size_range[1], cluster_pixels
            )

            # 应用形态学操作确保集中性
            cluster_mask = self._apply_morphological_operations(cluster_mask)

            # 应用到主mask
            mask[cluster_mask] = 0

        return mask

    def _generate_scattered_mask(self, shape, missing_ratio_range):
        """
        生成分散分布的mask（极少使用）
        """
        h, w = shape
        mask = np.ones(shape, dtype=np.float32)

        # 随机确定缺失比例
        target_missing_ratio = np.random.uniform(*missing_ratio_range)
        total_pixels = h * w
        target_missing_pixels = int(total_pixels * target_missing_ratio)

        # 多个小簇分散分布
        num_small_clusters = np.random.randint(5, 10)  # 5-9个小簇
        pixels_per_cluster = target_missing_pixels // num_small_clusters
        remaining_pixels = target_missing_pixels % num_small_clusters

        for cluster_idx in range(num_small_clusters):
            cluster_pixels = pixels_per_cluster
            if cluster_idx < remaining_pixels:
                cluster_pixels += 1

            if cluster_pixels <= 0:
                continue

            # 随机选择小簇中心
            center_x = np.random.randint(2, h - 2)
            center_y = np.random.randint(2, w - 2)

            # 生成小的圆形簇
            radius = min(2, np.sqrt(cluster_pixels / np.pi))
            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - center_y) ** 2 + (y - center_x) ** 2)
            small_cluster = dist <= radius

            mask[small_cluster] = 0

        return mask

    def _generate_concentrated_cluster(
        self, shape, center_x, center_y, max_radius, target_pixels
    ):
        """生成集中的不规则形状缺失簇"""
        # 方法1: 椭圆形缺失区域 (50%概率)
        if np.random.random() < 0.3:
            cluster_mask = self._generate_ellipse_cluster(
                shape, center_x, center_y, max_radius, target_pixels
            )
        # 方法2: 区域生长 (50%概率)
        else:
            cluster_mask = self._generate_growth_cluster(
                shape, center_x, center_y, max_radius, target_pixels
            )

        return cluster_mask

    def _generate_ellipse_cluster(
        self, shape, center_x, center_y, max_radius, target_pixels
    ):
        """生成椭圆形缺失区域"""
        h, w = shape
        cluster_mask = np.zeros(shape, dtype=bool)

        if not SCIPY_AVAILABLE:
            # 如果skimage不可用，使用简单的圆形
            y, x = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((x - center_y) ** 2 + (y - center_x) ** 2)
            radius = min(max_radius * 0.6, np.sqrt(target_pixels / np.pi))
            cluster_mask = dist_from_center <= radius
            return cluster_mask

        # 估算椭圆参数
        estimated_area = target_pixels

        # 随机椭圆长短轴比例
        aspect_ratio = np.random.uniform(0.4, 1.0)

        # 计算半长轴和半短轴
        semi_major = min(
            max_radius * 0.8, np.sqrt(estimated_area / (np.pi * aspect_ratio))
        )
        semi_minor = semi_major * aspect_ratio

        # 限制在图像范围内
        semi_major = min(semi_major, min(h // 3, w // 3))
        semi_minor = min(semi_minor, min(h // 3, w // 3))

        try:
            # 生成椭圆
            rr, cc = ellipse(center_x, center_y, semi_major, semi_minor, shape=shape)
            cluster_mask[rr, cc] = True
        except:
            # 如果椭圆生成失败，回退到圆形
            y, x = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((x - center_y) ** 2 + (y - center_x) ** 2)
            radius = min(max_radius * 0.6, np.sqrt(target_pixels / np.pi))
            cluster_mask = dist_from_center <= radius

        return cluster_mask

    def _generate_growth_cluster(
        self, shape, center_x, center_y, max_radius, target_pixels
    ):
        """使用区域生长方法生成集中的缺失区域"""
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

        # 区域生长
        max_iterations = target_pixels * 2
        iteration = 0

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
                                    boundary_points.append((ni, nj, dist))

            if not boundary_points:
                break

            # 去重并排序
            boundary_points = list(set(boundary_points))
            boundary_points.sort(key=lambda x: x[2])

            # 计算本次要添加的点数
            remaining_pixels = target_pixels - current_pixels
            num_to_add = min(
                len(boundary_points), max(1, remaining_pixels // 5), remaining_pixels
            )

            # 选择距离中心最近的点进行扩展
            selected_points = boundary_points[:num_to_add]

            # 添加选中的点
            for ni, nj, _ in selected_points:
                if current_pixels >= target_pixels:
                    break
                cluster_mask[ni, nj] = True
                current_pixels += 1

        return cluster_mask

    def _check_target_quality(self, target_data):
        """
        检查目标数据的质量，过滤掉数据基本在0和1附近分布的数据

        参数:
        - target_data: numpy数组，形状为(33, 33)

        返回:
        - bool: True表示数据质量好，False表示需要过滤
        """
        try:
            # 获取有效数据（排除空值）
            valid_data = target_data[target_data != 100.0]

            if len(valid_data) == 0:
                return False

            # 定义0和1附近的范围
            zero_range = 0.2  # 0附近的范围：[-0.2, 0.2]
            one_range = 0.2  # 1附近的范围：[0.8, 1.2]

            # 检查数据是否主要分布在0附近
            near_zero = np.abs(valid_data) <= zero_range
            zero_ratio = np.sum(near_zero) / len(valid_data)

            # 检查数据是否主要分布在1附近
            near_one = np.abs(valid_data - 1.0) <= one_range
            one_ratio = np.sum(near_one) / len(valid_data)

            # 检查数据是否主要分布在0或1附近
            near_zero_or_one = near_zero | near_one
            zero_one_ratio = np.sum(near_zero_or_one) / len(valid_data)

            # 过滤条件：如果超过80%的数据都在0或1附近，则过滤
            filter_threshold = 0.8
            should_filter = zero_one_ratio > filter_threshold

            # 可选：打印调试信息
            if should_filter:
                print(
                    f"过滤数据: {zero_one_ratio:.2%}的像素在0/1附近 "
                    f"(0附近: {zero_ratio:.2%}, 1附近: {one_ratio:.2%})"
                )

            return not should_filter

        except Exception as e:
            print(f"目标数据质量检查失败: {e}")
            return False

    def _apply_morphological_operations(self, mask):
        """应用形态学操作来进一步集中mask"""
        if not SCIPY_AVAILABLE:
            # 如果scipy不可用，返回原始mask
            return mask

        # 闭运算：填充小洞
        kernel_close = disk(2)
        mask_closed = binary_dilation(mask, kernel_close)
        mask_closed = binary_erosion(mask_closed, kernel_close)

        # 去除小的分离区域，只保留最大的连通区域
        labeled_mask, num_labels = label(mask_closed)
        if num_labels > 1:
            # 找到最大的连通区域
            region_sizes = []
            for i in range(1, num_labels + 1):
                size = np.sum(labeled_mask == i)
                region_sizes.append((size, i))

            # 保留最大的1个区域
            region_sizes.sort(reverse=True)

            new_mask = np.zeros_like(mask_closed)
            if region_sizes:
                _, region_id = region_sizes[0]
                new_mask[labeled_mask == region_id] = True

            mask_closed = new_mask

        return mask_closed

    def apply_mask_to_data(self, target_data, mask):
        """
        将mask应用到目标数据上生成新的输入数据

        参数:
        - target_data: 原始目标数据 (33, 33)
        - mask: 生成的mask (33, 33), 0表示缺失

        返回:
        - masked_data: 应用mask后的数据
        - applied_mask: 实际应用的mask
        """
        masked_data = target_data.copy()

        # 将mask为0的位置设为空值(100.0)
        empty_value = 100.0
        masked_data[mask == 0] = empty_value

        # 生成对应的输入mask (1表示有效，0表示无效)
        applied_mask = mask.copy()

        return masked_data, applied_mask

    def _build_global_file_index(self):
        """建立global文件的索引，避免重复搜索"""
        global_index = {}

        # 索引array和mask文件
        for filename in os.listdir(self.arrayDir):
            if filename.endswith("_a.npy"):
                parts = filename.split("_")
                if len(parts) >= 6:
                    id_val = parts[0]
                    xbegain = parts[1]
                    ybegain = parts[2]
                    xnum = parts[3]
                    ynum = parts[4]

                    if id_val not in global_index:
                        global_index[id_val] = {}

                    global_index[id_val]["array_file"] = os.path.join(
                        self.arrayDir, filename
                    )
                    global_index[id_val]["meta"] = {
                        "xbegain": int(xbegain),
                        "ybegain": int(ybegain),
                        "xnum": int(xnum),
                        "ynum": int(ynum),
                    }

        # 索引mask文件
        for filename in os.listdir(self.maskDir):
            if filename.endswith("_m.npy"):
                parts = filename.split("_")
                if len(parts) >= 6:
                    id_val = parts[0]
                    if id_val in global_index:
                        global_index[id_val]["mask_file"] = os.path.join(
                            self.maskDir, filename
                        )

        # 索引target文件
        for filename in os.listdir(self.targetDir):
            if filename.startswith("rs_") and filename.endswith(".npy"):
                parts = filename.split("_")
                if len(parts) >= 2:
                    id_val = parts[1]
                    if id_val in global_index:
                        global_index[id_val]["target_file"] = os.path.join(
                            self.targetDir, filename
                        )

        print(f"建立global文件索引完成，共找到 {len(global_index)} 个ID的文件")
        return global_index

    def _analyze_data_quality(self):
        """分析数据质量，统计有效像素分布"""
        print("正在分析数据质量...")
        pixel_counts = []
        valid_samples = 0
        total_analyzed = 0

        # 随机采样分析
        sample_size = min(1000, len(self.original_data_groups))
        sample_indices = np.random.choice(
            len(self.original_data_groups), sample_size, replace=False
        )

        for idx in sample_indices:
            total_analyzed += 1
            try:
                key = next(
                    itertools.islice(self.original_data_groups.keys(), idx, idx + 1)
                )
                group_data = self.original_data_groups[key]
                json_files = group_data["json_files"]
                meta = group_data["meta"]

                id_val = str(meta["id"])
                if id_val not in self.global_file_index:
                    continue

                # 只读取第一个json文件进行快速检查
                if json_files:
                    with open(json_files[0], "r", encoding="utf-8") as f:
                        json_content = json.load(f)
                        sample_data = np.array(json_content["data"])

                    # 快速计算有效像素
                    mask = (sample_data != 100).astype(np.float32)
                    valid_pixel_count = np.sum(mask == 1)
                    pixel_counts.append(valid_pixel_count)

                    if (
                        self.min_valid_pixels
                        <= valid_pixel_count
                        <= self.max_valid_pixels
                    ):
                        valid_samples += 1

            except Exception as e:
                continue

            if total_analyzed % 100 == 0:
                print(f"已分析 {total_analyzed}/{sample_size} 样本")

        pixel_counts = np.array(pixel_counts)
        stats = {
            "total_analyzed": total_analyzed,
            "valid_samples": valid_samples,
            "valid_ratio": valid_samples / max(total_analyzed, 1),
            "pixel_mean": np.mean(pixel_counts) if len(pixel_counts) > 0 else 0,
            "pixel_std": np.std(pixel_counts) if len(pixel_counts) > 0 else 0,
            "pixel_min": np.min(pixel_counts) if len(pixel_counts) > 0 else 0,
            "pixel_max": np.max(pixel_counts) if len(pixel_counts) > 0 else 0,
            "pixel_median": np.median(pixel_counts) if len(pixel_counts) > 0 else 0,
        }

        print(f"\n数据质量分析结果:")
        print(f"  分析样本数: {stats['total_analyzed']}")
        print(f"  有效样本数: {stats['valid_samples']}")
        print(f"  有效样本比例: {stats['valid_ratio']:.2%}")
        print(
            f"  有效像素统计: 均值={stats['pixel_mean']:.0f}, 中位数={stats['pixel_median']:.0f}"
        )
        print(f"  有效像素范围: [{stats['pixel_min']:.0f}, {stats['pixel_max']:.0f}]")
        print(f"  当前过滤范围: [{self.min_valid_pixels}, {self.max_valid_pixels}]")

        return stats

    def _adjust_pixel_thresholds(self):
        """根据数据质量分析结果调整像素阈值"""
        if not hasattr(self, "data_quality_stats"):
            return

        stats = self.data_quality_stats

        # 如果有效样本比例太低（<30%），自动调整阈值
        if stats["valid_ratio"] < 0.3:
            print(
                f"\n检测到有效样本比例过低({stats['valid_ratio']:.2%})，建议调整阈值:"
            )

            # 建议使用更宽松的范围
            suggested_min = max(10, int(stats["pixel_median"] * 0.5))
            suggested_max = min(1089, int(stats["pixel_median"] * 1.5))

            print(f"  建议最小值: {suggested_min} (当前: {self.min_valid_pixels})")
            print(f"  建议最大值: {suggested_max} (当前: {self.max_valid_pixels})")
            print("  可以在创建数据集时手动调整这些值以提高数据利用率")

    def _groupJsonFiles(self):
        """重新设计的分组逻辑，确保ID匹配正确"""
        groups = {}

        for subfolder in os.listdir(self.jsonDir):
            subfolder_path = os.path.join(self.jsonDir, subfolder)
            if not os.path.isdir(subfolder_path):
                continue

            for filename in os.listdir(subfolder_path):
                if filename.endswith(".json"):
                    parts = filename.split("_")
                    if len(parts) < 4:
                        continue

                    centerx = parts[0]
                    centery = parts[1]
                    id_val = parts[2]
                    id_adj = parts[3].split(".")[0]

                    # 检查这个id_val是否在global文件索引中存在
                    if id_val not in self.global_file_index:
                        continue

                    # 使用centerx, centery, id_val作为主键
                    key = (centerx, centery, id_val)

                    if key not in groups:
                        groups[key] = {
                            "json_files": [],
                            "meta": {
                                "centerx": int(centerx),
                                "centery": int(centery),
                                "id": int(id_val),
                                **self.global_file_index[id_val]["meta"],
                            },
                        }

                    json_path = os.path.join(subfolder_path, filename)
                    groups[key]["json_files"].append(json_path)

        print(f"分组完成，共有 {len(groups)} 个数据组")
        return groups

    def _load_fallback_pool(self, pool_size=30):
        """预加载多个有效的真实数据作为fallback池"""
        print(f"正在加载fallback数据池 (目标数量: {pool_size})...")
        fallback_pool = []

        max_attempts = min(500, len(self.original_data_groups))
        attempts = 0

        for idx in range(max_attempts):
            if len(fallback_pool) >= pool_size:
                break

            attempts += 1
            try:
                key = next(
                    itertools.islice(self.original_data_groups.keys(), idx, idx + 1)
                )
                group_data = self.original_data_groups[key]
                json_files = group_data["json_files"]
                meta = group_data["meta"]

                centerx = meta["centerx"]
                centery = meta["centery"]
                id_val = meta["id"]
                xbegain = meta["xbegain"]
                ybegain = meta["ybegain"]
                xnum = meta["xnum"]
                ynum = meta["ynum"]

                global_info = self.global_file_index.get(str(id_val))
                if not global_info:
                    continue

                # 读取数据
                input_data = []
                for json_path in json_files:
                    with open(json_path, "r", encoding="utf-8") as f:
                        json_content = json.load(f)
                        input_data.append(np.array(json_content["data"]))

                input_data_stack = np.stack(input_data, axis=0)
                merged_input_data, skip = merge_input_channels(
                    input_data_stack, empty_value=100
                )

                if skip:
                    continue

                input_processed, input_masks = self.preprocess_data(merged_input_data)

                # 检查有效性
                valid_pixel_count = np.sum(input_masks == 1)
                if not (
                    self.min_valid_pixels <= valid_pixel_count <= self.max_valid_pixels
                ):
                    continue

                # 加载global文件
                id_array = np.load(global_info["array_file"], mmap_mode="r").astype(
                    np.float32
                )
                id_mask = np.load(global_info["mask_file"]).astype(np.float32)
                id_target = np.load(global_info["target_file"], mmap_mode="r").astype(
                    np.float32
                )

                # 边界检查
                minx = int(centerx - xbegain) - self.kernel_size // 2
                maxx = int(centerx - xbegain) + self.kernel_size // 2 + 1
                miny = int(centery - ybegain) - self.kernel_size // 2
                maxy = int(centery - ybegain) + self.kernel_size // 2 + 1

                if (
                    minx < 0
                    or maxx > id_target.shape[0]
                    or miny < 0
                    or maxy > id_target.shape[1]
                    or maxx - minx != self.kernel_size
                    or maxy - miny != self.kernel_size
                ):
                    continue

                input_target = id_target[minx:maxx, miny:maxy].copy()

                # 创建副本
                id_array = id_array.copy()
                id_mask = id_mask.copy()

                mask_zero_values = id_array[id_mask == 0]
                if len(mask_zero_values) > 0:
                    mask_zero_value = mask_zero_values[0]
                else:
                    mask_zero_value = 100

                id_array[minx:maxx, miny:maxy] = mask_zero_value
                id_mask[minx:maxx, miny:maxy] = 0

                metadata = np.array(
                    [
                        float(centerx),
                        float(centery),
                        float(xbegain),
                        float(ybegain),
                        float(xnum),
                        float(ynum),
                        float(id_val),
                    ]
                )
                '''## 归一化
                # 基于global_input的数据范围进行归一化（已经用均值填充，无需考虑mask）
                global_min = id_array.min()
                global_max = id_array.max()
                
                # 避免除零错误
                if global_max - global_min > 1e-8:
                    # 归一化到[0, 1]
                    input_processed = (input_processed - global_min) / (global_max - global_min)
                    input_target = (input_target - global_min) / (global_max - global_min)
                    id_array = (id_array - global_min) / (global_max - global_min)
                    id_target = (id_target - global_min) / (global_max - global_min)
                else:
                    # 如果范围太小，直接设为0.5
                    global_input = torch.full_like(global_input, 0.5)
                    local_input = torch.full_like(local_input, 0.5)
                    id_array = torch.full_like(id_array, 0.5)
                    id_target = torch.full_like(id_target, 0.5)'''

                fallback_data = (
                    torch.tensor(input_processed, dtype=torch.float32),
                    torch.tensor(input_masks, dtype=torch.int),
                    torch.tensor(input_target, dtype=torch.float32),
                    torch.tensor(id_array, dtype=torch.float32),
                    torch.tensor(id_mask, dtype=torch.int),
                    torch.tensor(id_target, dtype=torch.float32),
                    metadata.astype(int),
                )

                fallback_pool.append(fallback_data)
                if len(fallback_pool) % 10 == 0:
                    print(f"已加载 {len(fallback_pool)}/{pool_size} 个fallback数据")

            except Exception as e:
                continue

        if len(fallback_pool) == 0:
            print("警告: 无法找到任何有效的fallback数据")
            return None
        else:
            print(f"成功创建fallback数据池，包含 {len(fallback_pool)} 个不同的数据样本")
            return fallback_pool

    def _get_fallback_data(self):
        """从fallback池中获取数据"""
        if self.fallback_pool is None or len(self.fallback_pool) == 0:
            return self._get_placeholder_data()

        # 基于当前请求总数的确定性选择
        selection_index = self.total_requests % len(self.fallback_pool)
        return self.fallback_pool[selection_index]

    def _load_fallback_data(self):
        """预加载一个有效的真实数据作为fallback"""
        print("正在加载单个fallback数据...")

        for idx in range(min(100, len(self.original_data_groups))):
            try:
                key = next(
                    itertools.islice(self.original_data_groups.keys(), idx, idx + 1)
                )
                group_data = self.original_data_groups[key]
                json_files = group_data["json_files"]
                meta = group_data["meta"]

                centerx = meta["centerx"]
                centery = meta["centery"]
                id_val = meta["id"]
                xbegain = meta["xbegain"]
                ybegain = meta["ybegain"]
                xnum = meta["xnum"]
                ynum = meta["ynum"]

                global_info = self.global_file_index.get(str(id_val))
                if not global_info:
                    continue

                # 读取数据
                input_data = []
                for json_path in json_files:
                    with open(json_path, "r", encoding="utf-8") as f:
                        json_content = json.load(f)
                        input_data.append(np.array(json_content["data"]))

                input_data_stack = np.stack(input_data, axis=0)
                merged_input_data, skip = merge_input_channels(
                    input_data_stack, empty_value=100
                )

                if skip:
                    continue

                input_processed, input_masks = self.preprocess_data(merged_input_data)

                # 检查有效性
                valid_pixel_count = np.sum(input_masks == 1)
                if not (
                    self.min_valid_pixels <= valid_pixel_count <= self.max_valid_pixels
                ):
                    continue

                # 加载global文件
                id_array = np.load(global_info["array_file"], mmap_mode="r").astype(
                    np.float32
                )
                id_mask = np.load(global_info["mask_file"]).astype(np.float32)
                id_target = np.load(global_info["target_file"], mmap_mode="r").astype(
                    np.float32
                )

                # 边界检查
                minx = int(centerx - xbegain) - self.kernel_size // 2
                maxx = int(centerx - xbegain) + self.kernel_size // 2 + 1
                miny = int(centery - ybegain) - self.kernel_size // 2
                maxy = int(centery - ybegain) + self.kernel_size // 2 + 1

                if (
                    minx < 0
                    or maxx > id_target.shape[0]
                    or miny < 0
                    or maxy > id_target.shape[1]
                    or maxx - minx != self.kernel_size
                    or maxy - miny != self.kernel_size
                ):
                    continue

                input_target = id_target[minx:maxx, miny:maxy].copy()

                # 创建副本
                id_array = id_array.copy()
                id_mask = id_mask.copy()

                mask_zero_values = id_array[id_mask == 0]
                if len(mask_zero_values) > 0:
                    mask_zero_value = mask_zero_values[0]
                else:
                    mask_zero_value = 100

                id_array[minx:maxx, miny:maxy] = mask_zero_value
                id_mask[minx:maxx, miny:maxy] = 0

                metadata = np.array(
                    [
                        float(centerx),
                        float(centery),
                        float(xbegain),
                        float(ybegain),
                        float(xnum),
                        float(ynum),
                        float(id_val),
                    ]
                )

                '''## 归一化
                # 基于global_input的数据范围进行归一化（已经用均值填充，无需考虑mask）
                global_min = id_array.min()
                global_max = id_array.max()
                
                # 避免除零错误
                if global_max - global_min > 1e-8:
                    # 归一化到[0, 1]
                    input_processed = (input_processed - global_min) / (global_max - global_min)
                    input_target = (input_target - global_min) / (global_max - global_min)
                    id_array = (id_array - global_min) / (global_max - global_min)
                    id_target = (id_target - global_min) / (global_max - global_min)
                else:
                    # 如果范围太小，直接设为0.5
                    global_input = torch.full_like(global_input, 0.5)
                    local_input = torch.full_like(local_input, 0.5)
                    id_array = torch.full_like(id_array, 0.5)
                    id_target = torch.full_like(id_target, 0.5)'''

                fallback_data = (
                    torch.tensor(input_processed, dtype=torch.float32),
                    torch.tensor(input_masks, dtype=torch.int),
                    torch.tensor(input_target, dtype=torch.float32),
                    torch.tensor(id_array, dtype=torch.float32),
                    torch.tensor(id_mask, dtype=torch.int),
                    torch.tensor(id_target, dtype=torch.float32),
                    metadata.astype(int),
                )

                print(f"成功加载fallback数据: ID={id_val}")
                return fallback_data

            except Exception as e:
                continue

        print("警告: 无法找到有效的fallback数据")
        return None

    def __len__(self):
        return self.total_length

    def preprocess_data(self, data):
        mask = data != 100
        valid_data = data[mask]
        if valid_data.size > 0:
            mean = valid_data.mean()
            data[~mask] = mean
        return data, mask.astype(np.float32)

    def _get_original_item(self, idx):
        """获取原始数据项"""
        max_retries = 5

        for retry in range(max_retries):
            try:
                # 计算实际使用的索引
                actual_idx = (idx + retry) % len(self.original_data_groups)

                # 获取数据组键
                key = next(
                    itertools.islice(
                        self.original_data_groups.keys(), actual_idx, actual_idx + 1
                    )
                )
                group_data = self.original_data_groups[key]
                json_files = group_data["json_files"]
                meta = group_data["meta"]

                # 提取元数据
                centerx = meta["centerx"]
                centery = meta["centery"]
                id_val = meta["id"]
                xbegain = meta["xbegain"]
                ybegain = meta["ybegain"]
                xnum = meta["xnum"]
                ynum = meta["ynum"]

                # 快速检查：验证global文件存在
                global_info = self.global_file_index.get(str(id_val))
                if not global_info:
                    continue

                # 1. 读取输入数据
                input_data = []
                for json_path in json_files:
                    with open(json_path, "r", encoding="utf-8") as f:
                        json_content = json.load(f)
                        input_data.append(np.array(json_content["data"]))

                # 2. 快速检查local mask有效性
                input_data_stack = np.stack(input_data, axis=0)
                merged_input_data, skip = merge_input_channels(
                    input_data_stack, empty_value=100
                )

                if skip:
                    continue

                input_processed, input_masks = self.preprocess_data(merged_input_data)

                # 检查local mask的有效性
                valid_pixel_count = np.sum(input_masks == 1)
                if not (
                    self.min_valid_pixels <= valid_pixel_count <= self.max_valid_pixels
                ):
                    continue

                # 3. 加载global文件
                id_array = np.load(global_info["array_file"], mmap_mode="r").astype(
                    np.float32
                )
                id_mask = np.load(global_info["mask_file"]).astype(np.float32)
                id_target = np.load(global_info["target_file"], mmap_mode="r").astype(
                    np.float32
                )

                # 4. 提取local的target并进行边界检查
                minx = int(centerx - xbegain) - self.kernel_size // 2
                maxx = int(centerx - xbegain) + self.kernel_size // 2 + 1
                miny = int(centery - ybegain) - self.kernel_size // 2
                maxy = int(centery - ybegain) + self.kernel_size // 2 + 1

                # 严格的边界检查
                if (
                    minx < 0
                    or maxx > id_target.shape[0]
                    or miny < 0
                    or maxy > id_target.shape[1]
                    or maxx - minx != self.kernel_size
                    or maxy - miny != self.kernel_size
                ):
                    continue

                input_target = id_target[minx:maxx, miny:maxy].copy()

                # 验证提取的target尺寸
                if input_target.shape != (self.kernel_size, self.kernel_size):
                    continue

                # **新增：目标数据质量检查**
                if not self._check_target_quality(input_target):
                    continue

                # 5. 剔除掉local在global中的部分
                mask_zero_values = id_array[id_mask == 0]
                if len(mask_zero_values) > 0:
                    mask_zero_value = mask_zero_values[0]
                else:
                    mask_zero_value = 100

                # 创建副本以避免修改原始数据
                id_array = id_array.copy()
                id_mask = id_mask.copy()
                id_array[minx:maxx, miny:maxy] = mask_zero_value
                id_mask[minx:maxx, miny:maxy] = 0

                # 构建元数据数组
                metadata = np.array(
                    [
                        float(centerx),
                        float(centery),
                        float(xbegain),
                        float(ybegain),
                        float(xnum),
                        float(ynum),
                        float(id_val),
                    ]
                )

                ## 归一化
                # 基于global_input的数据范围进行归一化（已经用均值填充，无需考虑mask）
                global_min = id_array.min()
                global_max = id_array.max()
                
                '''# 避免除零错误
                if global_max - global_min > 1e-8:
                    # 归一化到[0, 1]
                    input_processed = (input_processed - global_min) / (global_max - global_min)
                    input_target = (input_target - global_min) / (global_max - global_min)
                    id_array = (id_array - global_min) / (global_max - global_min)
                    id_target = (id_target - global_min) / (global_max - global_min)
                else:
                    # 如果范围太小，直接设为0.5
                    global_input = torch.full_like(global_input, 0.5)
                    local_input = torch.full_like(local_input, 0.5)
                    id_array = torch.full_like(id_array, 0.5)
                    id_target = torch.full_like(id_target, 0.5)

                # 转换为张量并返回
                return (
                    torch.tensor(input_target*input_masks+input_processed*(1-input_masks), dtype=torch.float32),
                    torch.tensor(input_masks, dtype=torch.int),
                    torch.tensor(input_target, dtype=torch.float32),
                    torch.tensor(id_array, dtype=torch.float32),
                    torch.tensor(id_mask, dtype=torch.int),
                    torch.tensor(id_target, dtype=torch.float32),
                    metadata.astype(int),
                )'''

            except Exception as e:
                if retry == max_retries - 1:
                    print(
                        f"Failed to load original data after {max_retries} retries for index {idx}: {e}"
                    )
                continue

        # 如果所有重试都失败，返回fallback数据
        return self._get_fallback_data()

    def _get_synthetic_item(self, idx):
        """生成基于生成式mask的数据项"""
        try:
            # 从原始数据中选择一个作为基础
            original_idx = (idx - self.original_length) % self.original_length

            # 获取原始数据的所有组件
            key = next(
                itertools.islice(
                    self.original_data_groups.keys(), original_idx, original_idx + 1
                )
            )
            group_data = self.original_data_groups[key]
            json_files = group_data["json_files"]
            meta = group_data["meta"]

            centerx = meta["centerx"]
            centery = meta["centery"]
            id_val = meta["id"]
            xbegain = meta["xbegain"]
            ybegain = meta["ybegain"]
            xnum = meta["xnum"]
            ynum = meta["ynum"]

            global_info = self.global_file_index.get(str(id_val))
            if not global_info:
                return self._get_fallback_data()

            # 加载global文件
            id_array = np.load(global_info["array_file"], mmap_mode="r").astype(
                np.float32
            )
            id_mask = np.load(global_info["mask_file"]).astype(np.float32)
            id_target = np.load(global_info["target_file"], mmap_mode="r").astype(
                np.float32
            )

            # 提取local target
            minx = int(centerx - xbegain) - self.kernel_size // 2
            maxx = int(centerx - xbegain) + self.kernel_size // 2 + 1
            miny = int(centery - ybegain) - self.kernel_size // 2
            maxy = int(centery - ybegain) + self.kernel_size // 2 + 1

            # 边界检查
            if (
                minx < 0
                or maxx > id_target.shape[0]
                or miny < 0
                or maxy > id_target.shape[1]
                or maxx - minx != self.kernel_size
                or maxy - miny != self.kernel_size
            ):
                return self._get_fallback_data()

            input_target = id_target[minx:maxx, miny:maxy].copy()

            # **新增：目标数据质量检查**
            if not self._check_target_quality(input_target):
                return self._get_fallback_data()

            # 生成主要集中分布的mask（分散概率极低：5%）
            synthetic_mask = self.generate_clustered_mask(
                shape=(self.kernel_size, self.kernel_size),
                missing_ratio_range=(0.25, 0.75),  # 20%-80%的缺失比例
                num_clusters_range=(1, 4),  # 1-2个缺失簇
                cluster_size_range=(6, 18),  # 簇大小范围
                scattered_probability=0.2,  # 仅5%概率生成分散mask
            )

            # 将mask应用到目标数据上生成新的输入
            synthetic_input_data, synthetic_input_mask = self.apply_mask_to_data(
                input_target, synthetic_mask
            )

            # 预处理生成的输入数据
            synthetic_input_data = synthetic_input_data[np.newaxis, :]  # 添加通道维度
            processed_input, processed_mask = self.preprocess_data(synthetic_input_data)

            # 修改全局数据以反映新的局部输入
            modified_id_array = id_array.copy()
            modified_id_mask = id_mask.copy()

            # 在全局数组中嵌入生成的局部数据
            modified_id_array[minx:maxx, miny:maxy] = synthetic_input_data.squeeze()
            modified_id_mask[minx:maxx, miny:maxy] = synthetic_input_mask

            # 构建元数据
            metadata = np.array(
                [
                    float(centerx),
                    float(centery),
                    float(xbegain),
                    float(ybegain),
                    float(xnum),
                    float(ynum),
                    float(id_val),
                ]
            )

            '''# 归一化
            # 基于global_input的数据范围进行归一化（已经用均值填充，无需考虑mask）
            global_min = id_array.min()
            global_max = id_array.max()
            
            # 避免除零错误
            if global_max - global_min > 1e-8:
                # 归一化到[0, 1]
                processed_input = (processed_input - global_min) / (global_max - global_min)
                input_target = (input_target - global_min) / (global_max - global_min)
                modified_id_array = (modified_id_array - global_min) / (global_max - global_min)
                id_target = (id_target - global_min) / (global_max - global_min)
            else:
                # 如果范围太小，直接设为0.5
                global_input = torch.full_like(global_input, 0.5)
                local_input = torch.full_like(local_input, 0.5)
                modified_id_array = torch.full_like(modified_id_array, 0.5)
                id_target = torch.full_like(id_target, 0.5)'''

            return (
                torch.tensor(input_target*processed_mask+processed_input*(1-processed_mask), dtype=torch.float32),
                torch.tensor(processed_mask, dtype=torch.int),
                torch.tensor(input_target, dtype=torch.float32),  # 目标保持不变
                torch.tensor(modified_id_array, dtype=torch.float32),
                torch.tensor(modified_id_mask, dtype=torch.int),
                torch.tensor(id_target, dtype=torch.float32),
                metadata.astype(int),
            )

        except Exception as e:
            print(f"生成合成数据失败 (idx={idx}): {e}")
            return self._get_fallback_data()

    def __getitem__(self, idx):
        """
        快速的__getitem__实现，支持原始数据和生成式mask数据
        """
        self.total_requests += 1

        if idx < self.original_length:
            # 返回原始数据
            return self._get_original_item(idx)
        else:
            # 返回生成式mask数据
            return self._get_synthetic_item(idx)

    def _get_placeholder_data(self):
        """返回占位数据，避免训练中断 - 仅作为最后手段"""
        print("警告: 正在创建占位数据，这可能影响训练质量")

        # 创建一个最小的有效数据作为占位符
        input_data = np.random.randn(1, 33, 33).astype(np.float32) * 0.1 - 30
        input_mask = np.ones((1, 33, 33), dtype=np.int32)
        input_target = np.random.randn(33, 33).astype(np.float32) * 0.1 - 30
        id_array = np.random.randn(600, 600).astype(np.float32) * 0.1 - 30
        id_mask = np.ones((600, 600), dtype=np.int32)
        id_target = np.random.randn(600, 600).astype(np.float32) * 0.1 - 30

        centerx = 300
        centery = 300
        xbegain = 283
        ybegain = 283
        xnum = 600
        ynum = 600
        id_val = -1

        metadata = np.array(
            [centerx, centery, xbegain, ybegain, xnum, ynum, id_val], dtype=int
        )

        return (
            torch.tensor(input_data, dtype=torch.float32),
            torch.tensor(input_mask, dtype=torch.int),
            torch.tensor(input_target, dtype=torch.float32),
            torch.tensor(id_array, dtype=torch.float32),
            torch.tensor(id_mask, dtype=torch.int),
            torch.tensor(id_target, dtype=torch.float32),
            metadata,
        )

    def visualize_synthetic_mask(self, num_samples=4, save_dir=None):
        """
        可视化生成的mask效果
        """
        if not self.enable_synthetic_masks:
            print("未启用生成式mask，无法可视化")
            return

        if self.total_length <= self.original_length:
            print("没有生成式数据可供可视化")
            return

        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_samples):
            # 获取合成数据索引
            synthetic_idx = self.original_length + i

            # 获取合成数据
            (
                input_data,
                input_mask,
                target_data,
                global_array,
                global_mask,
                global_target,
                metadata,
            ) = self[synthetic_idx]

            # 获取对应的原始数据进行比较
            original_idx = i % self.original_length
            (
                orig_input_data,
                orig_input_mask,
                orig_target_data,
                orig_global_array,
                orig_global_mask,
                orig_global_target,
                orig_metadata,
            ) = self[original_idx]

            # 转换为numpy用于可视化
            input_data_np = input_data.squeeze().numpy()
            input_mask_np = input_mask.squeeze().numpy()
            target_data_np = target_data.squeeze().numpy()
            orig_input_data_np = orig_input_data.squeeze().numpy()

            # 计算统一的颜色范围
            all_data = [input_data_np, target_data_np, orig_input_data_np]
            valid_data = []
            for d in all_data:
                valid_values = d[d != 100]
                if len(valid_values) > 0:
                    valid_data.extend(valid_values)

            if valid_data:
                vmin = np.min(valid_data)
                vmax = np.max(valid_data)
            else:
                vmin, vmax = 0, 1

            # 1. 原始输入
            im1 = axes[i, 0].imshow(
                orig_input_data_np, cmap="terrain", vmin=vmin, vmax=vmax
            )
            axes[i, 0].set_title(f"原始输入 {i+1}")

            # 2. 生成的输入（带mask）
            masked_display = input_data_np.copy()
            masked_display[input_mask_np == 0] = np.nan
            im2 = axes[i, 1].imshow(
                masked_display, cmap="terrain", vmin=vmin, vmax=vmax
            )
            axes[i, 1].set_title(f"生成输入 {i+1}")

            # 3. 生成的mask
            im3 = axes[i, 2].imshow(input_mask_np, cmap="binary", vmin=0, vmax=1)
            missing_ratio = 1 - input_mask_np.mean()
            if SCIPY_AVAILABLE:
                labeled_mask, num_regions = label(input_mask_np == 0)
                mask_type = "集中" if num_regions <= 2 else "分散"
                axes[i, 2].set_title(
                    f"生成mask {i+1}\n缺失: {missing_ratio:.1%} ({mask_type})"
                )
            else:
                axes[i, 2].set_title(f"生成mask {i+1}\n缺失: {missing_ratio:.1%}")

            # 4. 目标数据
            im4 = axes[i, 3].imshow(
                target_data_np, cmap="terrain", vmin=vmin, vmax=vmax
            )
            axes[i, 3].set_title(f"目标数据 {i+1}")

            # 添加colorbar（只在第一行）
            if i == 0:
                plt.colorbar(im1, ax=axes[i, 0])
                plt.colorbar(im2, ax=axes[i, 1])
                plt.colorbar(im3, ax=axes[i, 2])
                plt.colorbar(im4, ax=axes[i, 3])

        plt.tight_layout()

        if save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, "synthetic_mask_visualization.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"可视化结果已保存到: {save_path}")
        else:
            plt.show()

        plt.close()


def fast_collate_fn(batch):
    """快速的collate函数，最小化处理开销"""
    inputs = []
    input_masks = []
    input_targets = []
    id_arrays = []
    id_masks = []
    id_targets = []
    metadata = []

    for item in batch:
        if item is not None and len(item) == 7:
            inputs.append(item[0])
            input_masks.append(item[1])
            input_targets.append(item[2])
            id_arrays.append(item[3])
            id_masks.append(item[4])
            id_targets.append(item[5])
            metadata.append(item[6])

    # 如果没有有效数据，使用占位数据
    if len(inputs) == 0:
        print("警告: 批次中没有有效数据，使用占位数据")
        temp_dataset = DemDataset.__new__(DemDataset)
        placeholder = temp_dataset._get_placeholder_data()
        inputs = [placeholder[0]]
        input_masks = [placeholder[1]]
        input_targets = [placeholder[2]]
        id_arrays = [placeholder[3]]
        id_masks = [placeholder[4]]
        id_targets = [placeholder[5]]
        metadata = [placeholder[6]]

    # 验证数据尺寸
    for i, (inp, mask, target, id_arr, id_m, id_t, meta) in enumerate(
        zip(
            inputs,
            input_masks,
            input_targets,
            id_arrays,
            id_masks,
            id_targets,
            metadata,
        )
    ):
        if inp.shape[0] == 0 or inp.shape[1] == 0 or inp.shape[2] == 0:
            print(f"警告: 检测到无效尺寸，使用占位数据")
            temp_dataset = DemDataset.__new__(DemDataset)
            placeholder = temp_dataset._get_placeholder_data()
            inputs[i] = placeholder[0]
            input_masks[i] = placeholder[1]
            input_targets[i] = placeholder[2]
            id_arrays[i] = placeholder[3]
            id_masks[i] = placeholder[4]
            id_targets[i] = placeholder[5]
            metadata[i] = placeholder[6]

    # 堆叠张量
    try:
        inputs = torch.stack(inputs)
        input_masks = torch.stack(input_masks)
        input_targets = torch.stack(input_targets).unsqueeze(1)
        id_arrays = torch.stack(id_arrays).unsqueeze(1)
        id_masks = torch.stack(id_masks).unsqueeze(1)
        id_targets = torch.stack(id_targets).unsqueeze(1)

        return (
            inputs,
            input_masks,
            input_targets,
            id_arrays,
            id_masks,
            id_targets,
            metadata,
        )

    except Exception as e:
        print(f"堆叠张量时发生错误: {e}")
        # 创建占位批次
        temp_dataset = DemDataset.__new__(DemDataset)
        placeholder = temp_dataset._get_placeholder_data()

        batch_size = len(inputs) if inputs else 1
        return (
            placeholder[0].unsqueeze(0).repeat(batch_size, 1, 1, 1),
            placeholder[1].unsqueeze(0).repeat(batch_size, 1, 1, 1),
            placeholder[2].unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1, 1),
            placeholder[3].unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1, 1),
            placeholder[4].unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1, 1),
            placeholder[5].unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1, 1),
            [placeholder[6]] * batch_size,
        )


def merge_input_channels(input_data, empty_value=100.0, visualize=False):
    """合并多个输入通道，采用非空优先策略 - 优化版本"""
    # 创建空值掩码 (True表示有效值，False表示空值)
    valid_mask = input_data != empty_value

    # 统计每个位置的有效值数量
    valid_count = np.sum(valid_mask, axis=0, keepdims=True)

    # 快速检查是否需要跳过此批次
    total_valid = np.sum(valid_count > 0)
    skip_batch = total_valid < 10

    if skip_batch:
        return np.full((1, input_data.shape[1], input_data.shape[2]), empty_value), True

    # 将空值替换为0，以便于计算总和
    temp_data = np.where(valid_mask, input_data, 0)

    # 计算有效值的总和和均值
    valid_sum = np.sum(temp_data, axis=0, keepdims=True)
    merged_data = np.divide(
        valid_sum, valid_count, out=np.zeros_like(valid_sum), where=valid_count != 0
    )

    # 将没有有效值的位置设为空值
    merged_data[valid_count == 0] = empty_value

    return merged_data, False


# 在DemDataset的main函数中添加数据分布统计代码

if __name__ == "__main__":
   jsonDir = r"E:\KingCrimson Dataset\Simulate\data0\json"
   arrayDir = r"E:\KingCrimson Dataset\Simulate\data0\arraynmask_a\array"
   maskDir = r"E:\KingCrimson Dataset\Simulate\data0\arraynmask_a\mask"
   targetDir = r"E:\KingCrimson Dataset\Simulate\data0\groundtruthstatus\statusarray"

   # 创建数据集
   dataset = DemDataset(
       jsonDir,
       arrayDir,
       maskDir,
       targetDir,
       min_valid_pixels=20,
       max_valid_pixels=800,
       enable_synthetic_masks=True,
       synthetic_ratio=1.0,
   )

   print(f"数据集大小: {len(dataset)}")

   # 创建DataLoader
   dataloader = DataLoader(
       dataset,
       batch_size=8,
       shuffle=True,
       collate_fn=fast_collate_fn,
       num_workers=2,
       pin_memory=True,
       prefetch_factor=2,
       persistent_workers=True,
   )

   # ===================== 数据分布统计 =====================
   print("\n" + "="*60)
   print("📊 开始统计数据分布信息...")
   print("="*60)
   
   # 统计变量
   all_input_processed = []
   all_input_targets = []
   all_id_arrays = []
   all_id_targets = []
   
   # 分布统计
   input_stats = {'min': [], 'max': [], 'mean': [], 'std': []}
   target_stats = {'min': [], 'max': [], 'mean': [], 'std': []}
   global_stats = {'min': [], 'max': [], 'mean': [], 'std': []}
   
   # 归一化前的原始数据统计
   raw_input_stats = {'min': [], 'max': [], 'mean': [], 'std': []}
   raw_target_stats = {'min': [], 'max': [], 'mean': [], 'std': []}
   raw_global_stats = {'min': [], 'max': [], 'mean': [], 'std': []}
   
   # 采样统计（避免内存溢出）
   sample_count = 0
   max_samples = 100  # 统计前100个batch
   
   print(f"正在采样前 {max_samples} 个batch进行统计...")
   
   for batch_idx, batch_data in enumerate(dataloader):
       if batch_idx >= max_samples:
           break
           
       (inputs, input_masks, input_targets, 
        id_arrays, id_masks, id_targets, metadata) = batch_data
       
       batch_size = inputs.shape[0]
       sample_count += batch_size
       
       # 打印前几个样本的详细信息
       if batch_idx < 3:
           print(f"\n--- Batch {batch_idx + 1} 详细信息 ---")
           for i in range(min(3, batch_size)):
               inp = inputs[i].squeeze()
               inp_tgt = input_targets[i].squeeze()
               id_arr = id_arrays[i].squeeze()
               id_tgt = id_targets[i].squeeze()
               
               print(f"样本 {i+1}:")
               print(f"  local_input   - min: {inp.min():.4f}, max: {inp.max():.4f}, mean: {inp.mean():.4f}")
               print(f"  local_target  - min: {inp_tgt.min():.4f}, max: {inp_tgt.max():.4f}, mean: {inp_tgt.mean():.4f}")
               print(f"  global_array  - min: {id_arr.min():.4f}, max: {id_arr.max():.4f}, mean: {id_arr.mean():.4f}")
               print(f"  global_target - min: {id_tgt.min():.4f}, max: {id_tgt.max():.4f}, mean: {id_tgt.mean():.4f}")
               
               # 检查是否有异常值
               abnormal = False
               if inp.min() < -0.1 or inp.max() > 1.1:
                   print(f"  ⚠️  local_input 超出[0,1]范围！")
                   abnormal = True
               if inp_tgt.min() < -0.1 or inp_tgt.max() > 1.1:
                   print(f"  ⚠️  local_target 超出[0,1]范围！")
                   abnormal = True
               if id_arr.min() < -0.1 or id_arr.max() > 1.1:
                   print(f"  ⚠️  global_array 超出[0,1]范围！")
                   abnormal = True
               if id_tgt.min() < -0.1 or id_tgt.max() > 1.1:
                   print(f"  ⚠️  global_target 超出[0,1]范围！")
                   abnormal = True

               # 如果有异常值则进行对比可视化
               if abnormal:
                   import matplotlib.pyplot as plt
                   fig, axs = plt.subplots(1, 2, figsize=(10, 4))
                   im0 = axs[0].imshow(id_arr.numpy(), cmap='terrain')
                   axs[0].set_title('global_array')
                   plt.colorbar(im0, ax=axs[0])
                   im1 = axs[1].imshow(id_tgt.numpy(), cmap='terrain')
                   axs[1].set_title('global_target')
                   plt.colorbar(im1, ax=axs[1])
                   plt.suptitle(f"异常样本 {i+1} global_array vs global_target")
                   plt.tight_layout()
                   plt.show()
       
       # 统计每个batch的分布
       for i in range(batch_size):
           inp = inputs[i].squeeze()
           inp_tgt = input_targets[i].squeeze()
           id_arr = id_arrays[i].squeeze()
           id_tgt = id_targets[i].squeeze()
           
           # 只统计有效值（排除填充值）
           inp_valid = inp[inp != 0.5]  # 假设0.5是填充值
           inp_tgt_valid = inp_tgt[inp_tgt != 0.5]
           id_arr_valid = id_arr[id_arr != 0.5]
           id_tgt_valid = id_tgt[id_tgt != 0.5]
           
           # Local input统计
           if len(inp_valid) > 0:
               input_stats['min'].append(inp_valid.min().item())
               input_stats['max'].append(inp_valid.max().item())
               input_stats['mean'].append(inp_valid.mean().item())
               input_stats['std'].append(inp_valid.std().item())
           
           # Local target统计
           if len(inp_tgt_valid) > 0:
               target_stats['min'].append(inp_tgt_valid.min().item())
               target_stats['max'].append(inp_tgt_valid.max().item())
               target_stats['mean'].append(inp_tgt_valid.mean().item())
               target_stats['std'].append(inp_tgt_valid.std().item())
           
           # Global统计
           if len(id_tgt_valid) > 0:
               global_stats['min'].append(id_tgt_valid.min().item())
               global_stats['max'].append(id_tgt_valid.max().item())
               global_stats['mean'].append(id_tgt_valid.mean().item())
               global_stats['std'].append(id_tgt_valid.std().item())
       
       # 进度显示
       if (batch_idx + 1) % 20 == 0:
           print(f"已处理 {batch_idx + 1}/{max_samples} 个batch...")
   
   # ===================== 统计结果分析 =====================
   print(f"\n" + "="*60)
   print(f"📈 数据分布统计结果 (基于 {sample_count} 个样本)")
   print("="*60)
   
   def print_stats(stats_dict, name):
       if len(stats_dict['min']) > 0:
           print(f"\n{name} 统计:")
           print(f"  最小值: {np.min(stats_dict['min']):.6f} ~ {np.max(stats_dict['min']):.6f}")
           print(f"  最大值: {np.min(stats_dict['max']):.6f} ~ {np.max(stats_dict['max']):.6f}")
           print(f"  均值:   {np.min(stats_dict['mean']):.6f} ~ {np.max(stats_dict['mean']):.6f}")
           print(f"  标准差: {np.min(stats_dict['std']):.6f} ~ {np.max(stats_dict['std']):.6f}")
           
           # 检查归一化情况
           all_mins = np.array(stats_dict['min'])
           all_maxs = np.array(stats_dict['max'])
           
           out_of_range_min = np.sum(all_mins < -0.01)
           out_of_range_max = np.sum(all_maxs > 1.01)
           
           print(f"  超出[0,1]范围的样本:")
           print(f"    最小值<0: {out_of_range_min}/{len(all_mins)} ({out_of_range_min/len(all_mins)*100:.1f}%)")
           print(f"    最大值>1: {out_of_range_max}/{len(all_maxs)} ({out_of_range_max/len(all_maxs)*100:.1f}%)")
           
           if out_of_range_min > 0 or out_of_range_max > 0:
               print(f"  ⚠️  {name} 数据未正确归一化到[0,1]!")
               print(f"      极值范围: [{np.min(all_mins):.4f}, {np.max(all_maxs):.4f}]")
           else:
               print(f"  ✅ {name} 数据正确归一化到[0,1]")
       else:
           print(f"\n{name}: 无有效数据")
   
   # 打印各类数据的统计
   print_stats(input_stats, "Local Input")
   print_stats(target_stats, "Local Target") 
   print_stats(global_stats, "Global Data")
   
   # ===================== 特殊值分析 =====================
   print(f"\n" + "="*60)
   print("🔍 特殊值分析")
   print("="*60)
   
   # 重新采样检查特殊值
   special_value_count = {'0.5': 0, 'negative': 0, 'greater_than_1': 0, 'total_pixels': 0}
   
   print("检查特殊值分布...")
   for batch_idx, batch_data in enumerate(dataloader):
       if batch_idx >= 20:  # 检查前20个batch
           break
           
       (inputs, input_masks, input_targets, 
        id_arrays, id_masks, id_targets, metadata) = batch_data
       
       for i in range(inputs.shape[0]):
           inp = inputs[i].squeeze()
           
           total_pixels = inp.numel()
           special_value_count['total_pixels'] += total_pixels
           
           # 统计特殊值
           count_05 = torch.sum(torch.abs(inp - 0.5) < 1e-6).item()
           count_neg = torch.sum(inp < 0).item()
           count_gt1 = torch.sum(inp > 1).item()
           
           special_value_count['0.5'] += count_05
           special_value_count['negative'] += count_neg
           special_value_count['greater_than_1'] += count_gt1
   
   print(f"\n特殊值统计 (基于 {special_value_count['total_pixels']} 个像素):")
   print(f"  值=0.5的像素: {special_value_count['0.5']} ({special_value_count['0.5']/special_value_count['total_pixels']*100:.2f}%)")
   print(f"  负值像素: {special_value_count['negative']} ({special_value_count['negative']/special_value_count['total_pixels']*100:.2f}%)")
   print(f"  >1的像素: {special_value_count['greater_than_1']} ({special_value_count['greater_than_1']/special_value_count['total_pixels']*100:.2f}%)")
   
   # ===================== 结论和建议 =====================
   print(f"\n" + "="*60)
   print("💡 结论和建议")
   print("="*60)
   
   total_problematic = special_value_count['negative'] + special_value_count['greater_than_1']
   if total_problematic > 0:
       print(f"❌ 发现 {total_problematic} 个像素超出[0,1]范围")
       print(f"   这表明数据归一化存在问题")
       print(f"   建议检查归一化逻辑，特别是:")
       print(f"   1. 归一化基准的选择（global_min, global_max的计算）")
       print(f"   2. 是否有数据在归一化前/后被错误处理")
       print(f"   3. 检查else分支是否被正确执行")
   else:
       print(f"✅ 所有数据都在[0,1]范围内，归一化正确")
   
   if special_value_count['0.5'] > special_value_count['total_pixels'] * 0.1:
       print(f"⚠️  发现大量0.5值 ({special_value_count['0.5']/special_value_count['total_pixels']*100:.1f}%)")
       print(f"   这可能表明:")
       print(f"   1. 大量数据触发了else分支（范围过小）")
       print(f"   2. 大量无效数据被填充为0.5")
       print(f"   建议检查数据质量和填充策略")
   
   print(f"\n" + "="*60)
   print("📊 统计完成")
   print("="*60)
