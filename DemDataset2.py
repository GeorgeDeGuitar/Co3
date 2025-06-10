import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
import time
import itertools
from scipy.ndimage import binary_dilation, binary_erosion, label
from skimage.morphology import disk, square, diamond
from skimage.draw import ellipse, polygon
import random
from concurrent.futures import ThreadPoolExecutor, as_completed


class EnhancedDemDataset(Dataset):
    def __init__(self, jsonDir, arrayDir, maskDir, targetDir, transform=None, 
                 enable_synthetic_masks=True, synthetic_ratio=1.0,
                 use_cache=True, cache_dir=None, num_workers_cache=4):
        """
        增强的DEM数据集，支持生成式mask数据增强
        
        参数:
        - enable_synthetic_masks: 是否启用生成式mask增强
        - synthetic_ratio: 生成式数据占原始数据的比例（1.0表示每个原始数据生成1个合成数据）
        - use_cache: 是否使用缓存加速数据加载
        - cache_dir: 缓存目录，如果为None则使用默认目录
        - num_workers_cache: 缓存生成时的并行worker数量
        """
        self.jsonDir = jsonDir
        self.arrayDir = arrayDir
        self.maskDir = maskDir
        self.targetDir = targetDir
        self.transform = transform
        self.enable_synthetic_masks = enable_synthetic_masks
        self.synthetic_ratio = synthetic_ratio
        self.use_cache = use_cache
        self.num_workers_cache = num_workers_cache
        
        # 缓存设置
        if cache_dir is None:
            self.cache_dir = os.path.join(os.path.dirname(jsonDir), "dataset_cache")
        else:
            self.cache_dir = cache_dir
        
        # 原始数据组
        print("🔍 扫描数据文件...")
        self.original_data_groups = self._groupJsonFiles()
        self.original_length = len(self.original_data_groups)
        
        # 如果启用生成式mask，计算总长度
        if self.enable_synthetic_masks:
            self.synthetic_length = int(self.original_length * self.synthetic_ratio)
            self.total_length = self.original_length + self.synthetic_length
        else:
            self.total_length = self.original_length
            
        self.kernel_size = 33
        self.target_size = 600
        
        # 初始化缓存
        if self.use_cache:
            self._init_cache()
        
        print(f"📊 数据集统计:")
        print(f"  原始数据: {self.original_length}")
        if self.enable_synthetic_masks:
            print(f"  生成式数据: {self.synthetic_length}")
            print(f"  总数据量: {self.total_length}")
        if self.use_cache:
            print(f"  缓存目录: {self.cache_dir}")

    def _init_cache(self):
        """初始化缓存系统"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            print(f"📁 创建缓存目录: {self.cache_dir}")
        
        # 缓存版本控制
        self.cache_version_file = os.path.join(self.cache_dir, "cache_version.txt")
        self.current_version = "v1.2"  # 增加版本号以支持新的mask生成
        
        # 检查缓存版本
        cache_valid = False
        if os.path.exists(self.cache_version_file):
            try:
                with open(self.cache_version_file, 'r') as f:
                    cached_version = f.read().strip()
                if cached_version == self.current_version:
                    cache_valid = True
                else:
                    print(f"🔄 缓存版本更新 ({cached_version} -> {self.current_version})")
            except:
                pass
        
        if not cache_valid:
            print("🚀 生成数据缓存以加速后续加载...")
            self._build_cache()
            with open(self.cache_version_file, 'w') as f:
                f.write(self.current_version)
    
    def _build_cache(self):
        """构建数据缓存"""
        start_time = time.time()
        cache_metadata_file = os.path.join(self.cache_dir, "metadata.npy")
        
        print(f"📦 开始构建缓存，预计需要处理 {self.original_length} 个原始数据...")
        
        # 创建子目录
        subdirs = ['arrays', 'masks', 'targets', 'metadata']
        for subdir in subdirs:
            subdir_path = os.path.join(self.cache_dir, subdir)
            if not os.path.exists(subdir_path):
                os.makedirs(subdir_path)
        
        # 并行处理数据
        def cache_single_item(idx):
            try:
                data_dict = self._get_original_data_no_cache(idx)
                
                # 保存到缓存
                cache_prefix = os.path.join(self.cache_dir, f"item_{idx:06d}")
                
                np.save(f"{cache_prefix}_input_data.npy", 
                       np.stack(data_dict['input_data'], axis=0).astype(np.float32))
                np.save(f"{cache_prefix}_input_target.npy", 
                       data_dict['input_target'].astype(np.float32))
                np.save(f"{cache_prefix}_id_array.npy", 
                       data_dict['id_array'].astype(np.float32))
                np.save(f"{cache_prefix}_id_mask.npy", 
                       data_dict['id_mask'].astype(np.float32))
                np.save(f"{cache_prefix}_id_target.npy", 
                       data_dict['id_target'].astype(np.float32))
                np.save(f"{cache_prefix}_metadata.npy", 
                       data_dict['metadata'].astype(np.float32))
                
                return idx, True
            except Exception as e:
                print(f"❌ 缓存第 {idx} 个数据失败: {e}")
                return idx, False
        
        # 使用线程池并行处理
        successful_items = []
        with ThreadPoolExecutor(max_workers=self.num_workers_cache) as executor:
            futures = {executor.submit(cache_single_item, idx): idx 
                      for idx in range(self.original_length)}
            
            for future in as_completed(futures):
                idx, success = future.result()
                if success:
                    successful_items.append(idx)
                
                # 显示进度
                if len(successful_items) % 100 == 0:
                    elapsed = time.time() - start_time
                    progress = len(successful_items) / self.original_length
                    eta = elapsed / progress - elapsed if progress > 0 else 0
                    print(f"⏳ 缓存进度: {len(successful_items)}/{self.original_length} "
                          f"({progress*100:.1f}%) ETA: {eta:.1f}s")
        
        # 保存成功的索引列表
        np.save(cache_metadata_file, np.array(successful_items))
        
        elapsed = time.time() - start_time
        print(f"✅ 缓存构建完成! 耗时: {elapsed:.1f}s, 成功: {len(successful_items)}/{self.original_length}")
    
    def _get_original_data_no_cache(self, idx):
        """获取原始数据（不使用缓存）- 用于构建缓存"""
        return self._get_original_data_internal(idx)

    def _groupJsonFiles(self):
        """原始的数据分组逻辑，保持不变"""
        groups = {}
        for subfolder in os.listdir(self.jsonDir):
            subfolder_path = os.path.join(self.jsonDir, subfolder)
            print(f"processing folder: {subfolder}")
            if not os.path.isdir(subfolder_path):
                continue

            for filename in os.listdir(subfolder_path):
                if filename.endswith(".json"):
                    parts = filename.split("_")
                    if len(parts) < 4:
                        continue

                    centerx, centery, id_val, id_adj = (
                        parts[0], parts[1], parts[2], parts[3].split(".")[0]
                    )
                    key = (centerx, centery, id_val, id_adj)

                    if key not in groups:
                        groups[key] = {
                            "json_files": [],
                            "meta": {
                                "centerx": centerx,
                                "centery": centery,
                                "id": id_val,
                            },
                        }

                    json_path = os.path.join(subfolder_path, filename)
                    groups[key]["json_files"].append(json_path)

                    # 获取全局数据信息
                    for filename in os.listdir(self.arrayDir):
                        if filename.endswith("_a.npy") and filename.startswith(f"{id_val}_"):
                            parts = filename.split("_")
                            if len(parts) >= 6:
                                id_val = parts[0]
                                xbegain = parts[1]
                                ybegain = parts[2]
                                xnum = parts[3]
                                ynum = parts[4]
                                if "xbegain" not in groups[key]["meta"]:
                                    groups[key]["meta"]["xbegain"] = xbegain
                                    groups[key]["meta"]["ybegain"] = ybegain
                                    groups[key]["meta"]["xnum"] = xnum
                                    groups[key]["meta"]["ynum"] = ynum
                                    groups[key]["meta"]["id"] = id_val
        return groups

    def generate_clustered_mask(self, shape=(33, 33), 
                              missing_ratio_range=(0.0, 0.2),
                              num_clusters_range=(1, 2),
                              cluster_size_range=(8, 18),
                              scattered_probability=0.2):
        """
        生成缺失mask，支持集中和分散分布
        
        参数:
        - shape: mask形状
        - missing_ratio_range: 缺失像素比例范围
        - num_clusters_range: 缺失区域簇数量范围（1-2个簇）
        - cluster_size_range: 每个簇的大小范围
        - scattered_probability: 生成分散分布mask的概率（0.2表示20%概率）
        
        返回:
        - mask: 0表示缺失，1表示有效
        """
        # 决定生成集中还是分散的mask
        if np.random.random() < scattered_probability:
            # 生成分散分布的mask
            return self._generate_scattered_mask(shape, missing_ratio_range)
        else:
            # 生成集中分布的mask（原有逻辑）
            return self._generate_concentrated_mask_pattern(
                shape, missing_ratio_range, num_clusters_range, cluster_size_range
            )
    
    def _generate_concentrated_mask_pattern(self, shape, missing_ratio_range, 
                                          num_clusters_range, cluster_size_range):
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
                shape, center_x, center_y, 
                cluster_size_range[1], cluster_pixels
            )
            
            # 应用形态学操作确保集中性
            cluster_mask = self._apply_morphological_operations(cluster_mask)
            
            # 应用到主mask
            mask[cluster_mask] = 0
        
        return mask
    
    def _generate_scattered_mask(self, shape, missing_ratio_range):
        """
        生成分散分布的mask
        
        参数:
        - shape: mask形状
        - missing_ratio_range: 缺失像素比例范围
        
        返回:
        - mask: 0表示缺失，1表示有效
        """
        h, w = shape
        mask = np.ones(shape, dtype=np.float32)
        
        # 随机确定缺失比例
        target_missing_ratio = np.random.uniform(*missing_ratio_range)
        total_pixels = h * w
        target_missing_pixels = int(total_pixels * target_missing_ratio)
        
        # 方法1: 随机像素分散缺失 (30%概率)
        if np.random.random() < 0.3:
            # 完全随机分散
            all_positions = [(i, j) for i in range(h) for j in range(w)]
            missing_positions = np.random.choice(
                len(all_positions), target_missing_pixels, replace=False
            )
            for idx in missing_positions:
                i, j = all_positions[idx]
                mask[i, j] = 0
                
        # 方法2: 多个小簇分散分布 (40%概率)
        elif np.random.random() < 0.7:
            num_small_clusters = np.random.randint(5, 12)  # 5-11个小簇
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
                
                # 生成小的圆形或方形簇
                if np.random.random() < 0.5:
                    # 小圆形
                    radius = min(3, np.sqrt(cluster_pixels / np.pi))
                    y, x = np.ogrid[:h, :w]
                    dist = np.sqrt((x - center_y)**2 + (y - center_x)**2)
                    small_cluster = dist <= radius
                else:
                    # 小方形
                    size = min(3, int(np.sqrt(cluster_pixels)))
                    x1 = max(0, center_x - size//2)
                    x2 = min(h, center_x + size//2 + 1)
                    y1 = max(0, center_y - size//2)
                    y2 = min(w, center_y + size//2 + 1)
                    small_cluster = np.zeros(shape, dtype=bool)
                    small_cluster[x1:x2, y1:y2] = True
                
                mask[small_cluster] = 0
                
        # 方法3: 条带状分散缺失 (30%概率)
        else:
            # 生成随机条带
            num_stripes = np.random.randint(3, 8)  # 3-7条
            pixels_per_stripe = target_missing_pixels // num_stripes
            
            for stripe_idx in range(num_stripes):
                if np.random.random() < 0.5:
                    # 水平条带
                    stripe_height = np.random.randint(1, 4)
                    stripe_y = np.random.randint(0, h - stripe_height)
                    stripe_x_start = np.random.randint(0, w // 3)
                    stripe_x_end = np.random.randint(w // 3 * 2, w)
                    mask[stripe_y:stripe_y+stripe_height, stripe_x_start:stripe_x_end] = 0
                else:
                    # 垂直条带
                    stripe_width = np.random.randint(1, 4)
                    stripe_x = np.random.randint(0, w - stripe_width)
                    stripe_y_start = np.random.randint(0, h // 3)
                    stripe_y_end = np.random.randint(h // 3 * 2, h)
                    mask[stripe_y_start:stripe_y_end, stripe_x:stripe_x+stripe_width] = 0
        
        # 确保不超过目标缺失像素数
        current_missing = np.sum(mask == 0)
        if current_missing > target_missing_pixels:
            # 随机恢复一些像素
            missing_positions = np.where(mask == 0)
            excess = current_missing - target_missing_pixels
            restore_indices = np.random.choice(len(missing_positions[0]), excess, replace=False)
            mask[missing_positions[0][restore_indices], missing_positions[1][restore_indices]] = 1
        elif current_missing < target_missing_pixels:
            # 随机添加一些缺失像素
            valid_positions = np.where(mask == 1)
            deficit = target_missing_pixels - current_missing
            if len(valid_positions[0]) > 0:
                add_indices = np.random.choice(
                    len(valid_positions[0]), 
                    min(deficit, len(valid_positions[0])), 
                    replace=False
                )
                mask[valid_positions[0][add_indices], valid_positions[1][add_indices]] = 0
        
        return mask

    def _generate_concentrated_cluster(self, shape, center_x, center_y, max_radius, target_pixels):
        """
        生成集中的不规则形状缺失簇
        
        参数:
        - shape: 总形状
        - center_x, center_y: 簇中心坐标
        - max_radius: 最大半径
        - target_pixels: 目标像素数量
        """
        # 方法1: 椭圆形缺失区域 (40%概率)
        if np.random.random() < 0.2:
            cluster_mask = self._generate_ellipse_cluster(
                shape, center_x, center_y, max_radius, target_pixels
            )
        
        # 方法2: 多边形缺失区域 (30%概率)
        elif np.random.random() < 0.2:
            cluster_mask = self._generate_polygon_cluster(
                shape, center_x, center_y, max_radius, target_pixels
            )
        
        # 方法3: 连通区域生长 (30%概率)
        else:
            cluster_mask = self._generate_growth_cluster(
                shape, center_x, center_y, max_radius, target_pixels
            )
        
        return cluster_mask

    def _generate_ellipse_cluster(self, shape, center_x, center_y, max_radius, target_pixels):
        """生成椭圆形缺失区域"""
        h, w = shape
        cluster_mask = np.zeros(shape, dtype=bool)
        
        # 估算椭圆参数
        estimated_area = target_pixels
        
        # 随机椭圆长短轴比例
        aspect_ratio = np.random.uniform(0.4, 1.0)  # 长短轴比例
        
        # 计算半长轴和半短轴
        semi_major = min(max_radius * 0.8, np.sqrt(estimated_area / (np.pi * aspect_ratio)))
        semi_minor = semi_major * aspect_ratio
        
        # 限制在图像范围内
        semi_major = min(semi_major, min(h//3, w//3))
        semi_minor = min(semi_minor, min(h//3, w//3))
        
        try:
            # 生成椭圆
            rr, cc = ellipse(center_x, center_y, semi_major, semi_minor, shape=shape)
            cluster_mask[rr, cc] = True
        except:
            # 如果椭圆生成失败，回退到圆形
            y, x = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((x - center_y)**2 + (y - center_x)**2)
            radius = min(max_radius * 0.6, np.sqrt(target_pixels / np.pi))
            cluster_mask = dist_from_center <= radius
        
        return cluster_mask

    def _generate_polygon_cluster(self, shape, center_x, center_y, max_radius, target_pixels):
        """生成多边形缺失区域"""
        h, w = shape
        cluster_mask = np.zeros(shape, dtype=bool)
        
        # 生成不规则多边形顶点
        num_vertices = np.random.randint(5, 8)  # 5-7个顶点
        
        # 估算合适的半径
        estimated_radius = min(max_radius * 0.7, np.sqrt(target_pixels / np.pi) * 1.1)
        
        # 生成顶点
        angles = np.linspace(0, 2*np.pi, num_vertices, endpoint=False)
        vertices_x = []
        vertices_y = []
        
        for angle in angles:
            # 随机化半径，创造不规则形状
            radius_variation = np.random.uniform(0.7, 1.3)
            current_radius = estimated_radius * radius_variation
            
            x = center_x + current_radius * np.cos(angle)
            y = center_y + current_radius * np.sin(angle)
            
            # 确保顶点在图像范围内
            x = np.clip(x, 1, h-2)
            y = np.clip(y, 1, w-2)
            
            vertices_x.append(x)
            vertices_y.append(y)
        
        try:
            # 生成多边形
            rr, cc = polygon(vertices_x, vertices_y, shape=shape)
            cluster_mask[rr, cc] = True
        except:
            # 如果多边形生成失败，回退到圆形
            y, x = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((x - center_y)**2 + (y - center_x)**2)
            radius = min(max_radius * 0.6, np.sqrt(target_pixels / np.pi))
            cluster_mask = dist_from_center <= radius
        
        return cluster_mask

    def _generate_growth_cluster(self, shape, center_x, center_y, max_radius, target_pixels):
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
        neighbors = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        # 区域生长
        max_iterations = target_pixels * 2  # 防止无限循环
        iteration = 0
        
        while current_pixels < target_pixels and iteration < max_iterations:
            iteration += 1
            
            # 找到当前区域的边界点
            boundary_points = []
            
            for i in range(max(0, center_x - max_radius), min(h, center_x + max_radius + 1)):
                for j in range(max(0, center_y - max_radius), min(w, center_y + max_radius + 1)):
                    if cluster_mask[i, j]:
                        # 检查8邻域
                        for di, dj in neighbors:
                            ni, nj = i + di, j + dj
                            if (0 <= ni < h and 0 <= nj < w and 
                                not cluster_mask[ni, nj]):
                                
                                # 计算距离中心的距离
                                dist = np.sqrt((ni - center_x)**2 + (nj - center_y)**2)
                                if dist <= max_radius:
                                    boundary_points.append((ni, nj, dist))
            
            if not boundary_points:
                break
            
            # 去重
            boundary_points = list(set(boundary_points))
            
            # 按距离中心的距离排序，优先选择距离中心近的点
            boundary_points.sort(key=lambda x: x[2])
            
            # 计算本次要添加的点数
            remaining_pixels = target_pixels - current_pixels
            num_to_add = min(len(boundary_points), 
                           max(1, remaining_pixels // 5),
                           remaining_pixels)
            
            # 选择距离中心最近的点进行扩展
            selected_points = boundary_points[:num_to_add]
            
            # 添加选中的点
            for ni, nj, _ in selected_points:
                if current_pixels >= target_pixels:
                    break
                cluster_mask[ni, nj] = True
                current_pixels += 1
        
        return cluster_mask

    def _apply_morphological_operations(self, mask):
        """
        应用形态学操作来进一步集中mask
        """
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
            keep_regions = 1
            
            new_mask = np.zeros_like(mask_closed)
            for i in range(min(keep_regions, len(region_sizes))):
                _, region_id = region_sizes[i]
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

    def __len__(self):
        return self.total_length

    def preprocess_data(self, data):
        """原始的数据预处理逻辑"""
        mask = data != 100
        valid_data = data[mask]
        if valid_data.size > 0:
            mean = valid_data.mean()
            data[~mask] = mean
        return data, mask.astype(np.float32)

    def _get_original_data(self, idx):
        """获取原始数据的逻辑，支持缓存加速"""
        if self.use_cache:
            return self._get_original_data_cached(idx)
        else:
            return self._get_original_data_internal(idx)
    
    def _get_original_data_cached(self, idx):
        """从缓存加载原始数据"""
        try:
            cache_prefix = os.path.join(self.cache_dir, f"item_{idx:06d}")
            
            # 检查缓存文件是否存在
            required_files = [
                f"{cache_prefix}_input_data.npy",
                f"{cache_prefix}_input_target.npy", 
                f"{cache_prefix}_id_array.npy",
                f"{cache_prefix}_id_mask.npy",
                f"{cache_prefix}_id_target.npy",
                f"{cache_prefix}_metadata.npy"
            ]
            
            if not all(os.path.exists(f) for f in required_files):
                # 缓存文件不完整，回退到直接加载
                return self._get_original_data_internal(idx)
            
            # 从缓存加载
            input_data_stack = np.load(f"{cache_prefix}_input_data.npy", mmap_mode='r')
            input_data = [input_data_stack[i] for i in range(input_data_stack.shape[0])]
            
            data_dict = {
                'input_data': input_data,
                'input_target': np.load(f"{cache_prefix}_input_target.npy", mmap_mode='r'),
                'id_array': np.load(f"{cache_prefix}_id_array.npy", mmap_mode='r'),
                'id_mask': np.load(f"{cache_prefix}_id_mask.npy", mmap_mode='r'),
                'id_target': np.load(f"{cache_prefix}_id_target.npy", mmap_mode='r'),
                'metadata': np.load(f"{cache_prefix}_metadata.npy", mmap_mode='r'),
            }
            
            # 计算局部区域坐标
            metadata = data_dict['metadata']
            centerx, centery, xbegain, ybegain = metadata[:4]
            minx = int(centerx - xbegain) - self.kernel_size // 2
            maxx = int(centerx - xbegain) + self.kernel_size // 2 + 1
            miny = int(centery - ybegain) - self.kernel_size // 2
            maxy = int(centery - ybegain) + self.kernel_size // 2 + 1
            
            # 计算mask_zero_values
            mask_zero_values = data_dict['id_array'][data_dict['id_mask'] == 0]
            mask_zero_values = mask_zero_values[0] if len(mask_zero_values) > 0 else 0
            
            data_dict.update({
                'minx': minx, 'maxx': maxx, 'miny': miny, 'maxy': maxy,
                'mask_zero_values': mask_zero_values
            })
            
            return data_dict
            
        except Exception as e:
            print(f"⚠️  缓存加载失败 (idx={idx}): {e}, 回退到直接加载")
            return self._get_original_data_internal(idx)
    
    def _get_original_data_internal(self, idx):
        """获取原始数据的内部实现（原有逻辑）"""
        key = next(itertools.islice(self.original_data_groups.keys(), idx, idx + 1))
        group_data = self.original_data_groups[key]
        json_files = group_data["json_files"]
        meta = group_data["meta"]

        # 提取元数据
        centerx = int(meta["centerx"])
        centery = int(meta["centery"])
        id_val = int(meta["id"])
        xbegain = int(meta.get("xbegain", 0))
        ybegain = int(meta.get("ybegain", 0))
        xnum = int(meta.get("xnum", 0))
        ynum = int(meta.get("ynum", 0))

        metadata = np.array([
            float(centerx), float(centery), float(xbegain), 
            float(ybegain), float(xnum), float(ynum), float(id_val)
        ])

        # 读取输入数据
        input_data = []
        for json_path in json_files:
            with open(json_path, "r", encoding="utf-8") as f:
                json_content = json.load(f)
                input_data.append(np.array(json_content["data"]))

        # 读取全局数据
        array_filename = f"{id_val}_{xbegain}_{ybegain}_{xnum}_{ynum}_a.npy"
        array_path = os.path.join(self.arrayDir, array_filename)
        if os.path.exists(array_path):
            id_array = np.load(array_path, mmap_mode='r').astype(np.float32)
        else:
            print(f"Warning: Array file {array_path} not found.")
            id_array = np.zeros((xnum, ynum))

        # 读取全局mask
        mask_filename = f"{id_val}_{xbegain}_{ybegain}_{xnum}_{ynum}_m.npy"
        mask_path = os.path.join(self.maskDir, mask_filename)
        if os.path.exists(mask_path):
            id_mask = np.load(mask_path).astype(np.float32)
        else:
            print(f"Warning: Mask file {mask_path} not found.")
            id_mask = np.zeros((xnum, ynum))

        # 读取目标数据
        target_filename = None
        for filename in os.listdir(self.targetDir):
            if filename.startswith(f"rs_{id_val}_") and filename.endswith(".npy"):
                target_filename = filename
                break
        
        if target_filename:
            target_path = os.path.join(self.targetDir, target_filename)
            id_target = np.load(target_path, mmap_mode='r').astype(np.float32)
        else:
            print(f"Warning: Target file not found for id {id_val}")
            id_target = np.zeros((int(xnum), int(ynum)))

        # 提取局部target
        minx = int(centerx - xbegain) - self.kernel_size // 2
        maxx = int(centerx - xbegain) + self.kernel_size // 2 + 1
        miny = int(centery - ybegain) - self.kernel_size // 2
        maxy = int(centery - ybegain) + self.kernel_size // 2 + 1
        
        input_target = id_target[minx:maxx, miny:maxy].copy()

        # 剔除局部区域从全局数据
        mask_zero_values = id_array[id_mask == 0][0] if np.any(id_mask == 0) else 0
        id_array[minx:maxx, miny:maxy] = mask_zero_values
        id_mask[minx:maxx, miny:maxy] = 0

        return {
            'input_data': input_data,
            'input_target': input_target,
            'id_array': id_array,
            'id_mask': id_mask,
            'id_target': id_target,
            'metadata': metadata,
            'minx': minx, 'maxx': maxx, 'miny': miny, 'maxy': maxy,
            'mask_zero_values': mask_zero_values
        }

    def __getitem__(self, idx):
        if idx < self.original_length:
            # 返回原始数据
            return self._get_original_item(idx)
        else:
            # 返回生成式mask数据
            return self._get_synthetic_item(idx)

    def _get_original_item(self, idx):
        """获取原始数据项"""
        data_dict = self._get_original_data(idx)
        
        # 处理输入数据
        input_data = np.stack(data_dict['input_data'], axis=0)
        merged_input_data, _ = merge_input_channels(input_data, empty_value=100)
        input_data, input_masks = self.preprocess_data(merged_input_data)

        return (
            torch.tensor(input_data, dtype=torch.float32),
            torch.tensor(input_masks, dtype=torch.int),
            torch.tensor(data_dict['input_target'], dtype=torch.float32),
            torch.tensor(data_dict['id_array'], dtype=torch.float32),
            torch.tensor(data_dict['id_mask'], dtype=torch.int),
            torch.tensor(data_dict['id_target'], dtype=torch.float32),
            data_dict['metadata'].astype(int),
        )

    def _get_synthetic_item(self, idx):
        """生成基于生成式mask的数据项"""
        # 从原始数据中随机选择一个作为基础
        original_idx = (idx - self.original_length) % self.original_length
        data_dict = self._get_original_data(original_idx)
        
        # 生成混合分布的mask
        synthetic_mask = self.generate_clustered_mask(
            shape=(self.kernel_size, self.kernel_size),
            missing_ratio_range=(0.25, 0.75),  # 25%-75%的缺失比例
            num_clusters_range=(1, 2),         # 1-2个缺失簇
            cluster_size_range=(6, 15),        # 簇大小范围
            scattered_probability=0.2          # 20%概率生成分散mask
        )
        
        # 将mask应用到目标数据上生成新的输入
        synthetic_input_data, synthetic_input_mask = self.apply_mask_to_data(
            data_dict['input_target'], synthetic_mask
        )
        
        # 预处理生成的输入数据
        synthetic_input_data = synthetic_input_data[np.newaxis, :]  # 添加通道维度
        processed_input, processed_mask = self.preprocess_data(synthetic_input_data)
        
        # 修改全局数据以反映新的局部输入
        modified_id_array = data_dict['id_array'].copy()
        modified_id_mask = data_dict['id_mask'].copy()
        
        # 在全局数组中嵌入生成的局部数据
        minx, maxx = data_dict['minx'], data_dict['maxx']
        miny, maxy = data_dict['miny'], data_dict['maxy']
        
        # 将生成的输入数据嵌入到全局数组中
        modified_id_array[minx:maxx, miny:maxy] = synthetic_input_data.squeeze()
        modified_id_mask[minx:maxx, miny:maxy] = synthetic_input_mask

        return (
            torch.tensor(processed_input, dtype=torch.float32),
            torch.tensor(processed_mask, dtype=torch.int),
            torch.tensor(data_dict['input_target'], dtype=torch.float32),  # 目标保持不变
            torch.tensor(modified_id_array, dtype=torch.float32),
            torch.tensor(modified_id_mask, dtype=torch.int),
            torch.tensor(data_dict['id_target'], dtype=torch.float32),
            data_dict['metadata'].astype(int),
        )

    def visualize_synthetic_mask(self, idx_list=None, save_dir=None):
        """
        可视化生成的mask效果
        
        参数:
        - idx_list: 要可视化的索引列表，如果为None则随机选择
        - save_dir: 保存图像的目录
        """
        if idx_list is None:
            # 随机选择一些合成数据进行可视化
            synthetic_indices = list(range(self.original_length, min(self.original_length + 8, self.total_length)))
        else:
            synthetic_indices = [idx for idx in idx_list if idx >= self.original_length]
        
        if not synthetic_indices:
            print("没有合成数据可供可视化")
            return
            
        fig, axes = plt.subplots(len(synthetic_indices), 4, figsize=(16, 4 * len(synthetic_indices)))
        if len(synthetic_indices) == 1:
            axes = axes.reshape(1, -1)
        
        for i, idx in enumerate(synthetic_indices):
            # 获取合成数据
            (input_data, input_mask, target_data, 
             global_array, global_mask, global_target, metadata) = self[idx]
            
            # 获取对应的原始数据进行比较
            original_idx = (idx - self.original_length) % self.original_length
            (orig_input_data, orig_input_mask, orig_target_data, 
             orig_global_array, orig_global_mask, orig_global_target, orig_metadata) = self[original_idx]
            
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
            im1 = axes[i, 0].imshow(orig_input_data_np, cmap='terrain', vmin=vmin, vmax=vmax)
            axes[i, 0].set_title(f'原始输入 (idx {original_idx})')
            
            # 2. 生成的输入（带mask）
            masked_display = input_data_np.copy()
            masked_display[input_mask_np == 0] = np.nan  # 将缺失区域设为NaN以在可视化中显示
            im2 = axes[i, 1].imshow(masked_display, cmap='terrain', vmin=vmin, vmax=vmax)
            axes[i, 1].set_title(f'生成输入 (idx {idx})')
            
            # 3. 生成的mask
            im3 = axes[i, 2].imshow(input_mask_np, cmap='binary', vmin=0, vmax=1)
            axes[i, 2].set_title(f'生成mask (缺失: {1-input_mask_np.mean():.1%})')
            
            # 4. 目标数据（保持不变）
            im4 = axes[i, 3].imshow(target_data_np, cmap='terrain', vmin=vmin, vmax=vmax)
            axes[i, 3].set_title('目标数据')
            
            # 添加colorbar
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
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"可视化结果已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()


def merge_input_channels(input_data, empty_value=100.0, visualize=False):
    """
    合并多个输入通道，采用非空优先策略

    参数:
    input_data: 形状为[N, H, W]的数组，其中N是通道数
    empty_value: 表示空值的数值

    返回:
    merged_data: 形状为[1, H, W]的数组
    """
    # 创建空值掩码 (True表示有效值，False表示空值)
    valid_mask = (input_data != empty_value).astype(np.uint8)

    # 统计每个位置的有效值数量
    valid_count = np.sum(valid_mask, axis=0, keepdims=True)

    # 将空值替换为0，以便于计算总和
    temp_data = np.where(valid_mask, input_data, 0)

    # 计算有效值的总和
    valid_sum = np.sum(temp_data, axis=0, keepdims=True)

    # 计算有效值的均值 (避免除以零)
    merged_data = np.zeros_like(valid_sum)
    non_empty_positions = valid_count > 0
    merged_data[non_empty_positions] = (
        valid_sum[non_empty_positions] / valid_count[non_empty_positions]
    )

    # 将没有有效值的位置设为空值
    merged_data[valid_count == 0] = empty_value

    return merged_data, False


# 自定义 collate_fn，处理变长张量
def custom_collate_fn(batch):
    # batch 是一个列表，每个元素是一个7元素的元组
    # 将每个元组拆分成7个列表
    inputs = []
    input_masks = []
    input_targets = []
    id_arrays = []
    id_masks = []
    id_targets = []
    metadata = []

    for item in batch:
        inputs.append(item[0])
        input_masks.append(item[1])
        input_targets.append(item[2])
        id_arrays.append(item[3])
        id_masks.append(item[4])
        id_targets.append(item[5])
        metadata.append(item[6])

    # 现在将列表转换为批次张量
    inputs = torch.stack(inputs)
    input_masks = torch.stack(input_masks)
    input_targets = torch.stack(input_targets).unsqueeze(1)
    id_arrays = torch.stack(id_arrays).unsqueeze(1)
    id_masks = torch.stack(id_masks).unsqueeze(1)
    id_targets = torch.stack(id_targets).unsqueeze(1)

    return inputs, input_masks, input_targets, id_arrays, id_masks, id_targets, metadata


def test_mask_generation():
    """测试混合式mask生成效果"""
    print("🧪 测试混合式mask生成算法...")
    
    # 创建临时数据集实例进行测试
    class TestDataset:
        def __init__(self):
            self.kernel_size = 33
            
        def generate_clustered_mask(self, *args, **kwargs):
            # 创建一个EnhancedDemDataset的临时实例
            temp_dataset = type('temp', (), {})()
            temp_dataset.kernel_size = 33
            
            # 绑定所有方法
            temp_dataset.generate_clustered_mask = EnhancedDemDataset.generate_clustered_mask.__get__(temp_dataset)
            temp_dataset._generate_concentrated_mask_pattern = EnhancedDemDataset._generate_concentrated_mask_pattern.__get__(temp_dataset)
            temp_dataset._generate_scattered_mask = EnhancedDemDataset._generate_scattered_mask.__get__(temp_dataset)
            temp_dataset._generate_concentrated_cluster = EnhancedDemDataset._generate_concentrated_cluster.__get__(temp_dataset)
            temp_dataset._generate_ellipse_cluster = EnhancedDemDataset._generate_ellipse_cluster.__get__(temp_dataset)
            temp_dataset._generate_polygon_cluster = EnhancedDemDataset._generate_polygon_cluster.__get__(temp_dataset)
            temp_dataset._generate_growth_cluster = EnhancedDemDataset._generate_growth_cluster.__get__(temp_dataset)
            temp_dataset._apply_morphological_operations = EnhancedDemDataset._apply_morphological_operations.__get__(temp_dataset)
            
            return temp_dataset.generate_clustered_mask(*args, **kwargs)
    
    test_dataset = TestDataset()
    
    # 生成测试mask - 分别测试集中和分散模式
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    for i in range(12):
        row = i // 4
        col = i % 4
        
        # 生成mask - 设置不同的分散概率来演示两种模式
        if i < 6:
            # 前6个：主要集中模式（10%分散概率）
            mask = test_dataset.generate_clustered_mask(
                shape=(33, 33),
                missing_ratio_range=(0.25, 0.75),
                num_clusters_range=(1, 2),
                cluster_size_range=(6, 15),
                scattered_probability=0.1
            )
            mask_type = "集中模式"
        else:
            # 后6个：主要分散模式（90%分散概率）
            mask = test_dataset.generate_clustered_mask(
                shape=(33, 33),
                missing_ratio_range=(0.25, 0.75),
                num_clusters_range=(1, 2),
                cluster_size_range=(6, 15),
                scattered_probability=0.9
            )
            mask_type = "分散模式"
        
        # 计算缺失比例
        missing_ratio = 1 - (mask.sum() / mask.size)
        
        # 计算连通区域数
        labeled_mask, num_regions = label(mask == 0)
        
        # 可视化
        axes[row, col].imshow(mask, cmap='binary', vmin=0, vmax=1)
        axes[row, col].set_title(f'{mask_type} {i%6+1}\n缺失: {missing_ratio:.1%}, 区域: {num_regions}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.suptitle('混合式Mask生成测试结果 (集中 vs 分散)', fontsize=16, y=0.98)
    
    # 保存测试结果
    plt.savefig('mixed_masks_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ 混合式mask生成测试完成！")
    print("📊 测试结果已保存为 mixed_masks_test.png")
    print("🎯 新算法特点:")
    print("   - 支持集中和分散两种分布模式")
    print("   - 可控的分散概率（scattered_probability参数）")
    print("   - 集中模式: 1-2个大型连续区域")
    print("   - 分散模式: 多种分散策略（随机、小簇、条带）")
    print("   - 智能缓存系统加速数据加载")


def benchmark_dataset_loading(dataset, num_samples=50):
    """性能基准测试"""
    print(f"📊 开始性能基准测试 (样本数: {num_samples})")
    
    # 测试原始数据加载
    start_time = time.time()
    for i in range(min(num_samples, dataset.original_length)):
        _ = dataset._get_original_item(i)
    original_time = time.time() - start_time
    
    # 测试合成数据生成
    if dataset.enable_synthetic_masks and dataset.total_length > dataset.original_length:
        start_time = time.time()
        for i in range(dataset.original_length, min(dataset.original_length + num_samples, dataset.total_length)):
            _ = dataset._get_synthetic_item(i)
        synthetic_time = time.time() - start_time
    else:
        synthetic_time = 0
    
    print(f"⏱️  性能结果:")
    print(f"   原始数据加载: {original_time:.2f}s ({original_time/num_samples*1000:.1f}ms/样本)")
    if synthetic_time > 0:
        print(f"   合成数据生成: {synthetic_time:.2f}s ({synthetic_time/num_samples*1000:.1f}ms/样本)")
    
    return original_time, synthetic_time


def create_fast_dataset(jsonDir, arrayDir, maskDir, targetDir, **kwargs):
    """创建优化的快速数据集"""
    # 默认的快速配置
    fast_config = {
        'enable_synthetic_masks': True,
        'synthetic_ratio': 1.0,
        'use_cache': True,
        'num_workers_cache': 6,  # 增加缓存生成的并行度
    }
    fast_config.update(kwargs)
    
    print("🚀 创建高性能数据集配置:")
    for key, value in fast_config.items():
        print(f"   {key}: {value}")
    
    return EnhancedDemDataset(jsonDir, arrayDir, maskDir, targetDir, **fast_config)


# 示例使用
if __name__ == "__main__":
    # 测试mask生成
    print("🧪 测试混合式mask生成算法...")
    test_mask_generation()
    
    # 性能测试
    print("\n" + "="*50)
    print("⚡ 性能测试")
    print("="*50)
    
    # 测试缓存性能
    jsonDir = r"E:\KingCrimson Dataset\Simulate\data0\json"
    arrayDir = r"E:\KingCrimson Dataset\Simulate\data0\arraynmask\array"
    maskDir = r"E:\KingCrimson Dataset\Simulate\data0\arraynmask\mask"
    targetDir = r"E:\KingCrimson Dataset\Simulate\data0\groundtruthstatus\statusarray"
    
    print("测试数据加载性能...")
    
    # 创建小规模测试数据集
    if os.path.exists(jsonDir):
        print("📊 无缓存加载测试...")
        start_time = time.time()
        dataset_no_cache = EnhancedDemDataset(
            jsonDir, arrayDir, maskDir, targetDir,
            enable_synthetic_masks=False,
            use_cache=False
        )
        # 测试加载前10个数据
        for i in range(10):
            _ = dataset_no_cache[i]
        no_cache_time = time.time() - start_time
        print(f"无缓存加载10个数据耗时: {no_cache_time:.2f}s")
        
        print("📊 有缓存加载测试...")
        start_time = time.time()
        dataset_with_cache = EnhancedDemDataset(
            jsonDir, arrayDir, maskDir, targetDir,
            enable_synthetic_masks=False,
            use_cache=True
        )
        # 测试加载前10个数据
        for i in range(10):
            _ = dataset_with_cache[i]
        cache_time = time.time() - start_time
        print(f"有缓存加载10个数据耗时: {cache_time:.2f}s")
        
        if no_cache_time > 0:
            speedup = no_cache_time / cache_time
            print(f"🚀 缓存加速比: {speedup:.1f}x")
    
    print("\n" + "="*50)
    print("🚀 数据集使用示例")
    print("="*50)

    # 创建增强数据集 - 使用快速配置
    dataset = create_fast_dataset(
        jsonDir, arrayDir, maskDir, targetDir,
        synthetic_ratio=1.0,           # 每个原始数据生成1个合成数据
        use_cache=True,                # 启用缓存加速
        num_workers_cache=6            # 使用6个线程并行构建缓存
    )
    
    print(f"📈 增强后数据集大小: {len(dataset)}")
    
    # 性能基准测试
    if len(dataset) > 0:
        benchmark_dataset_loading(dataset, num_samples=20)
    
    # 测试数据加载
    dataloader = DataLoader(
        dataset, batch_size=4, shuffle=True, 
        collate_fn=custom_collate_fn, num_workers=2
    )
    
    # 可视化生成式mask效果
    if len(dataset) > dataset.original_length:
        print("🎨 生成数据增强可视化...")
        dataset.visualize_synthetic_mask(save_dir="./mask_visualization_results")
    
    # 测试训练循环
    print("🔄 测试数据加载器...")
    for i, batch in enumerate(dataloader):
        inputs, input_masks, input_targets, id_arrays, id_masks, id_targets, metadata = batch
        print(f"批次 {i+1}:")
        print(f"  输入形状: {inputs.shape}")
        print(f"  掩码形状: {input_masks.shape}")
        print(f"  目标形状: {input_targets.shape}")
        
        # 分析mask特征
        concentrated_count = 0
        scattered_count = 0
        
        for j in range(inputs.shape[0]):
            mask = input_masks[j].cpu().numpy().squeeze()
            missing_ratio = 1.0 - (mask.sum() / mask.size)
            labeled_mask, num_regions = label(mask == 0)
            
            # 简单判断mask类型（基于连通区域数量）
            mask_type = "集中" if num_regions <= 2 else "分散"
            if mask_type == "集中":
                concentrated_count += 1
            else:
                scattered_count += 1
                
            print(f"    样本{j+1}: 缺失{missing_ratio:.1%}, {num_regions}个区域 ({mask_type})")
        
        print(f"  批次统计: {concentrated_count}个集中, {scattered_count}个分散")
        
        if i >= 1:  # 只测试2个批次
            break
    
    print("✅ 测试完成！数据集可以正常使用。")
    print("🎯 优化总结:")
    print("   - 支持集中(80%)和分散(20%)混合mask模式")
    print("   - 智能缓存系统显著加速数据加载")
    print("   - 并行缓存构建提高初始化速度") 
    print("   - 内存映射减少RAM占用")
    print("   - 线程池加速I/O密集型操作")