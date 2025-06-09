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

class DemDataset(Dataset):
    def __init__(self, jsonDir, arrayDir, maskDir, targetDir, transform=None, 
                 min_valid_pixels=20, max_valid_pixels=800, analyze_data=False):
        """
        jsonDir: 存放 json 子文件夹的根目录
        arrayDir: 存放 array (.npy) 文件的目录
        maskDir: 存放 mask (.npy) 文件的目录
        targetDir: 存放 target (rs{id}.xyz) 文件的目录
        transform: 数据增强或归一化（可选）
        min_valid_pixels: local mask中最小有效像素数量
        max_valid_pixels: local mask中最大有效像素数量
        """
        self.jsonDir = jsonDir
        self.arrayDir = arrayDir
        self.maskDir = maskDir
        self.targetDir = targetDir
        self.transform = transform
        self.min_valid_pixels = min_valid_pixels
        self.max_valid_pixels = max_valid_pixels
        self.kernel_size = 33  # 网格大小
        self.target_size = 600  # 目标大小
        self.cachedir = r"E:\KingCrimson Dataset\Simulate\data0\traindata"
        
        # 先建立global文件的索引
        self.global_file_index = self._build_global_file_index()
        self.dataGroups = self._groupJsonFiles()
        
        # 预加载一个有效数据作为fallback
        self.fallback_data = self._load_fallback_data()
        
        # 数据质量分析
        if analyze_data:
            self.data_quality_stats = self._analyze_data_quality()
            self._adjust_pixel_thresholds()
        
        # 预加载多个有效数据作为fallback池
        self.fallback_pool = self._load_fallback_pool(pool_size=30)  # 加载10个fallback数据
        self.fallback_index = 0  # 用于轮换fallback数据
        
        print(f"数据集初始化完成: 总数据{len(self.dataGroups)}")

    def _load_fallback_pool(self, pool_size=10):
        """预加载多个有效的真实数据作为fallback池"""
        print(f"正在加载fallback数据池 (目标数量: {pool_size})...")
        fallback_pool = []
        
        max_attempts = min(500, len(self.dataGroups))  # 最多尝试500个样本
        attempts = 0
        
        for idx in range(max_attempts):
            if len(fallback_pool) >= pool_size:
                break
                
            attempts += 1
            try:
                key = next(itertools.islice(self.dataGroups.keys(), idx, idx + 1))
                group_data = self.dataGroups[key]
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
                merged_input_data, skip = merge_input_channels(input_data_stack, empty_value=100)
                
                if skip:
                    continue
                    
                input_processed, input_masks = self.preprocess_data(merged_input_data)
                
                # 检查有效性
                valid_pixel_count = np.sum(input_masks == 1)
                if not (self.min_valid_pixels <= valid_pixel_count <= self.max_valid_pixels):
                    continue

                # 加载global文件
                id_array = np.load(global_info['array_file'], mmap_mode='r').astype(np.float32)
                id_mask = np.load(global_info['mask_file']).astype(np.float32)
                id_target = np.load(global_info['target_file'], mmap_mode='r').astype(np.float32)

                # 边界检查
                minx = int(centerx - xbegain) - self.kernel_size // 2
                maxx = int(centerx - xbegain) + self.kernel_size // 2 + 1
                miny = int(centery - ybegain) - self.kernel_size // 2
                maxy = int(centery - ybegain) + self.kernel_size // 2 + 1
                
                if (minx < 0 or maxx > id_target.shape[0] or 
                    miny < 0 or maxy > id_target.shape[1] or
                    maxx - minx != self.kernel_size or 
                    maxy - miny != self.kernel_size):
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

                metadata = np.array([
                    float(centerx), float(centery), float(xbegain), 
                    float(ybegain), float(xnum), float(ynum), float(id_val)
                ])

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
                print(f"成功加载fallback数据 {len(fallback_pool)}/{pool_size}: ID={id_val}, valid_pixels={valid_pixel_count}")
                
            except Exception as e:
                continue
        
        if len(fallback_pool) == 0:
            print("警告: 无法找到任何有效的fallback数据")
            return None
        else:
            print(f"成功创建fallback数据池，包含 {len(fallback_pool)} 个不同的数据样本")
            
            # 显示fallback池的统计信息
            ids = [data[6][6] for data in fallback_pool]  # 提取ID
            valid_pixels = [torch.sum(data[1] == 1).item() for data in fallback_pool]  # 提取有效像素数
            print(f"Fallback池统计:")
            print(f"  ID范围: {min(ids)} - {max(ids)}")
            print(f"  有效像素范围: {min(valid_pixels)} - {max(valid_pixels)}")
            print(f"  平均有效像素: {np.mean(valid_pixels):.0f}")
            
            return fallback_pool

    def _get_fallback_data(self):
        """从fallback池中获取数据，支持多种选择策略"""
        if self.fallback_pool is None or len(self.fallback_pool) == 0:
            return self._get_placeholder_data()
        
        # 策略1: 轮换使用（默认）
        if hasattr(self, 'fallback_strategy') and self.fallback_strategy == 'round_robin':
            selected_data = self.fallback_pool[self.fallback_index % len(self.fallback_pool)]
            self.fallback_index += 1
            return selected_data
        
        # 策略2: 随机选择
        elif hasattr(self, 'fallback_strategy') and self.fallback_strategy == 'random':
            return np.random.choice(self.fallback_pool)
        
        # 策略3: 基于当前索引的确定性选择（避免随机性影响reproducibility）
        else:
            # 使用当前请求总数来确定选择哪个fallback数据
            selection_index = self.total_requests % len(self.fallback_pool)
            return self.fallback_pool[selection_index]

    def set_fallback_strategy(self, strategy='deterministic'):
        """
        设置fallback数据选择策略
        
        Args:
            strategy (str): 选择策略
                - 'deterministic': 基于请求数确定性选择（默认，保证reproducibility）
                - 'round_robin': 轮换使用
                - 'random': 随机选择
        """
        valid_strategies = ['deterministic', 'round_robin', 'random']
        if strategy not in valid_strategies:
            raise ValueError(f"策略必须是以下之一: {valid_strategies}")
        
        self.fallback_strategy = strategy
        print(f"Fallback选择策略已设置为: {strategy}")

    def _load_fallback_data(self):
        """预加载一个有效的真实数据作为fallback"""
        print("正在加载fallback数据...")
        
        for idx in range(min(100, len(self.dataGroups))):  # 最多尝试100个样本
            try:
                key = next(itertools.islice(self.dataGroups.keys(), idx, idx + 1))
                group_data = self.dataGroups[key]
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
                merged_input_data, skip = merge_input_channels(input_data_stack, empty_value=100)
                
                if skip:
                    continue
                    
                input_processed, input_masks = self.preprocess_data(merged_input_data)
                
                # 检查有效性
                valid_pixel_count = np.sum(input_masks == 1)
                if not (self.min_valid_pixels <= valid_pixel_count <= self.max_valid_pixels):
                    continue

                # 加载global文件
                id_array = np.load(global_info['array_file'], mmap_mode='r').astype(np.float32)
                id_mask = np.load(global_info['mask_file']).astype(np.float32)
                id_target = np.load(global_info['target_file'], mmap_mode='r').astype(np.float32)

                # 边界检查
                minx = int(centerx - xbegain) - self.kernel_size // 2
                maxx = int(centerx - xbegain) + self.kernel_size // 2 + 1
                miny = int(centery - ybegain) - self.kernel_size // 2
                maxy = int(centery - ybegain) + self.kernel_size // 2 + 1
                
                if (minx < 0 or maxx > id_target.shape[0] or 
                    miny < 0 or maxy > id_target.shape[1] or
                    maxx - minx != self.kernel_size or 
                    maxy - miny != self.kernel_size):
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

                metadata = np.array([
                    float(centerx), float(centery), float(xbegain), 
                    float(ybegain), float(xnum), float(ynum), float(id_val)
                ])

                fallback_data = (
                    torch.tensor(input_processed, dtype=torch.float32),
                    torch.tensor(input_masks, dtype=torch.int),
                    torch.tensor(input_target, dtype=torch.float32),
                    torch.tensor(id_array, dtype=torch.float32),
                    torch.tensor(id_mask, dtype=torch.int),
                    torch.tensor(id_target, dtype=torch.float32),
                    metadata.astype(int),
                )
                
                print(f"成功加载fallback数据: ID={id_val}, valid_pixels={valid_pixel_count}")
                return fallback_data
                
            except Exception as e:
                continue
        
        # 如果无法找到有效数据，返回None
        print("警告: 无法找到有效的fallback数据")
        return None

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
                    
                    global_index[id_val]['array_file'] = os.path.join(self.arrayDir, filename)
                    global_index[id_val]['meta'] = {
                        'xbegain': int(xbegain),
                        'ybegain': int(ybegain),
                        'xnum': int(xnum),
                        'ynum': int(ynum)
                    }
        
        # 索引mask文件
        for filename in os.listdir(self.maskDir):
            if filename.endswith("_m.npy"):
                parts = filename.split("_")
                if len(parts) >= 6:
                    id_val = parts[0]
                    if id_val in global_index:
                        global_index[id_val]['mask_file'] = os.path.join(self.maskDir, filename)
        
        # 索引target文件
        for filename in os.listdir(self.targetDir):
            if filename.startswith("rs_") and filename.endswith(".npy"):
                parts = filename.split("_")
                if len(parts) >= 2:
                    id_val = parts[1]
                    if id_val in global_index:
                        global_index[id_val]['target_file'] = os.path.join(self.targetDir, filename)
        
        print(f"建立global文件索引完成，共找到 {len(global_index)} 个ID的文件")
        return global_index

    def _analyze_data_quality(self):
        """分析数据质量，统计有效像素分布"""
        print("正在分析数据质量...")
        pixel_counts = []
        valid_samples = 0
        total_analyzed = 0
        
        # 随机采样分析，不需要分析所有数据
        sample_size = min(1000, len(self.dataGroups))
        sample_indices = np.random.choice(len(self.dataGroups), sample_size, replace=False)
        
        for idx in sample_indices:
            total_analyzed += 1
            try:
                key = next(itertools.islice(self.dataGroups.keys(), idx, idx + 1))
                group_data = self.dataGroups[key]
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
                    
                    if self.min_valid_pixels <= valid_pixel_count <= self.max_valid_pixels:
                        valid_samples += 1
                        
            except Exception as e:
                continue
            
            if total_analyzed % 100 == 0:
                print(f"已分析 {total_analyzed}/{sample_size} 样本")
        
        pixel_counts = np.array(pixel_counts)
        stats = {
            'total_analyzed': total_analyzed,
            'valid_samples': valid_samples,
            'valid_ratio': valid_samples / max(total_analyzed, 1),
            'pixel_mean': np.mean(pixel_counts) if len(pixel_counts) > 0 else 0,
            'pixel_std': np.std(pixel_counts) if len(pixel_counts) > 0 else 0,
            'pixel_min': np.min(pixel_counts) if len(pixel_counts) > 0 else 0,
            'pixel_max': np.max(pixel_counts) if len(pixel_counts) > 0 else 0,
            'pixel_median': np.median(pixel_counts) if len(pixel_counts) > 0 else 0,
        }
        
        print(f"\n数据质量分析结果:")
        print(f"  分析样本数: {stats['total_analyzed']}")
        print(f"  有效样本数: {stats['valid_samples']}")
        print(f"  有效样本比例: {stats['valid_ratio']:.2%}")
        print(f"  有效像素统计: 均值={stats['pixel_mean']:.0f}, 中位数={stats['pixel_median']:.0f}")
        print(f"  有效像素范围: [{stats['pixel_min']:.0f}, {stats['pixel_max']:.0f}]")
        print(f"  当前过滤范围: [{self.min_valid_pixels}, {self.max_valid_pixels}]")
        
        return stats
    
    def _adjust_pixel_thresholds(self):
        """根据数据质量分析结果调整像素阈值"""
        if not hasattr(self, 'data_quality_stats'):
            return
            
        stats = self.data_quality_stats
        
        # 如果有效样本比例太低（<30%），自动调整阈值
        if stats['valid_ratio'] < 0.3:
            print(f"\n检测到有效样本比例过低({stats['valid_ratio']:.2%})，建议调整阈值:")
            
            # 建议使用更宽松的范围
            suggested_min = max(10, int(stats['pixel_median'] * 0.5))
            suggested_max = min(1089, int(stats['pixel_median'] * 1.5))
            
            print(f"  建议最小值: {suggested_min} (当前: {self.min_valid_pixels})")
            print(f"  建议最大值: {suggested_max} (当前: {self.max_valid_pixels})")
            
            # 可以选择自动调整或手动调整
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
                                **self.global_file_index[id_val]['meta']
                            },
                        }

                    json_path = os.path.join(subfolder_path, filename)
                    groups[key]["json_files"].append(json_path)

        print(f"分组完成，共有 {len(groups)} 个数据组")
        return groups

    def __len__(self):
        return len(self.dataGroups)

    def preprocess_data(self, data):
        mask = data != 100
        valid_data = data[mask]
        if valid_data.size > 0:
            mean = valid_data.mean()
            data[~mask] = mean
        return data, mask.astype(np.float32)

    def __getitem__(self, idx):
        """
        快速的__getitem__实现，在运行时进行轻量级检查
        """
        max_retries = 5  # 最大重试次数
        
        for retry in range(max_retries):
            try:
                # 计算实际使用的索引
                actual_idx = (idx + retry) % len(self.dataGroups)
                
                # 获取数据组键
                key = next(itertools.islice(self.dataGroups.keys(), actual_idx, actual_idx + 1))
                group_data = self.dataGroups[key]
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

                # 1. 读取输入数据（多个json文件）
                input_data = []
                for json_path in json_files:
                    with open(json_path, "r", encoding="utf-8") as f:
                        json_content = json.load(f)
                        input_data.append(np.array(json_content["data"]))

                # 2. 快速检查local mask有效性
                input_data_stack = np.stack(input_data, axis=0)
                merged_input_data, skip = merge_input_channels(input_data_stack, empty_value=100)
                
                if skip:
                    continue
                    
                input_processed, input_masks = self.preprocess_data(merged_input_data)
                
                # 检查local mask的有效性
                valid_pixel_count = np.sum(input_masks == 1)
                if not (self.min_valid_pixels <= valid_pixel_count <= self.max_valid_pixels):
                    continue

                # 3. 加载global文件（只有通过检查才加载）
                id_array = np.load(global_info['array_file'], mmap_mode='r').astype(np.float32)
                id_mask = np.load(global_info['mask_file']).astype(np.float32)
                id_target = np.load(global_info['target_file'], mmap_mode='r').astype(np.float32)

                # 4. 提取local的target并进行边界检查
                minx = int(centerx - xbegain) - self.kernel_size // 2
                maxx = int(centerx - xbegain) + self.kernel_size // 2 + 1
                miny = int(centery - ybegain) - self.kernel_size // 2
                maxy = int(centery - ybegain) + self.kernel_size // 2 + 1
                
                # 严格的边界检查
                if (minx < 0 or maxx > id_target.shape[0] or 
                    miny < 0 or maxy > id_target.shape[1] or
                    maxx - minx != self.kernel_size or 
                    maxy - miny != self.kernel_size):
                    print(f"边界检查失败: ID={id_val}, minx={minx}, maxx={maxx}, miny={miny}, maxy={maxy}")
                    print(f"目标形状: {id_target.shape}, 期望大小: {self.kernel_size}")
                    continue
                    
                input_target = id_target[minx:maxx, miny:maxy].copy()

                # 验证提取的target尺寸
                if input_target.shape != (self.kernel_size, self.kernel_size):
                    print(f"提取的target尺寸错误: {input_target.shape}, 期望: ({self.kernel_size}, {self.kernel_size})")
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
                metadata = np.array([
                    float(centerx), float(centery), float(xbegain), 
                    float(ybegain), float(xnum), float(ynum), float(id_val)
                ])

                # 转换为张量并返回
                return (
                    torch.tensor(input_processed, dtype=torch.float32),
                    torch.tensor(input_masks, dtype=torch.int),
                    torch.tensor(input_target, dtype=torch.float32),
                    torch.tensor(id_array, dtype=torch.float32),
                    torch.tensor(id_mask, dtype=torch.int),
                    torch.tensor(id_target, dtype=torch.float32),
                    metadata.astype(int),
                )
                
            except Exception as e:
                # 如果出现异常，尝试下一个索引
                if retry == max_retries - 1:
                    # 最后一次重试失败，返回一个默认数据或抛出异常
                    print(f"Failed to load data after {max_retries} retries for index {idx}: {e}")
                continue
        
        # 如果所有重试都失败，返回真实的fallback数据
        if self.fallback_data is not None:
            # print(f"使用fallback数据替代索引 {idx}")
            return self.fallback_data
        else:
            # 最后的手段：返回占位数据
            print(f"警告: 使用占位数据替代索引 {idx}")
            return self._get_placeholder_data()

    def _get_placeholder_data(self):
        """返回占位数据，避免训练中断 - 仅作为最后手段"""
        print("警告: 正在创建占位数据，这可能影响训练质量")
        
        # 创建一个最小的有效数据作为占位符，确保尺寸正确
        input_data = np.random.randn(1, 33, 33).astype(np.float32) * 0.1 - 30  # 接近真实DEM数据范围
        input_mask = np.ones((1, 33, 33), dtype=np.int32)
        input_target = np.random.randn(33, 33).astype(np.float32) * 0.1 - 30
        id_array = np.random.randn(600, 600).astype(np.float32) * 0.1 - 30
        id_mask = np.ones((600, 600), dtype=np.int32)
        id_target = np.random.randn(600, 600).astype(np.float32) * 0.1 - 30
        
        # 确保metadata的centerx, centery在合理范围内，避免索引越界
        centerx = 300  # 600的中心位置
        centery = 300
        xbegain = 283  # 确保33x33的patch在600x600范围内
        ybegain = 283
        xnum = 600
        ynum = 600
        id_val = -1  # 特殊ID标识占位数据，使用负数避免与真实ID冲突
        
        metadata = np.array([centerx, centery, xbegain, ybegain, xnum, ynum, id_val], dtype=int)
        
        return (
            torch.tensor(input_data, dtype=torch.float32),
            torch.tensor(input_mask, dtype=torch.int),
            torch.tensor(input_target, dtype=torch.float32),
            torch.tensor(id_array, dtype=torch.float32),
            torch.tensor(id_mask, dtype=torch.int),
            torch.tensor(id_target, dtype=torch.float32),
            metadata,
        )


def fast_collate_fn(batch):
    """
    快速的collate函数，最小化处理开销
    """
    # 直接处理，不进行复杂的过滤
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

    # 如果没有有效数据，尝试使用fallback数据
    if len(inputs) == 0:
        print("警告: 批次中没有有效数据，尝试创建替代批次")
        # 这种情况应该很少见，因为__getitem__已经有了fallback机制
        # 创建一个临时dataset实例来获取fallback数据
        temp_dataset = DemDataset.__new__(DemDataset)
        temp_dataset.fallback_pool = None  # 避免递归
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
        zip(inputs, input_masks, input_targets, id_arrays, id_masks, id_targets, metadata)
    ):
        # 检查并修复可能的尺寸问题
        if inp.shape[0] == 0 or inp.shape[1] == 0 or inp.shape[2] == 0:
            print(f"警告: 检测到无效尺寸 input shape: {inp.shape}，使用占位数据")
            temp_dataset = DemDataset.__new__(DemDataset)
            temp_dataset.fallback_pool = None
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

        return inputs, input_masks, input_targets, id_arrays, id_masks, id_targets, metadata
        
    except Exception as e:
        print(f"堆叠张量时发生错误: {e}")
        # 如果堆叠失败，创建完全的占位批次
        temp_dataset = DemDataset.__new__(DemDataset)
        temp_dataset.fallback_pool = None
        placeholder = temp_dataset._get_placeholder_data()
        
        batch_size = len(inputs) if inputs else 1
        return (
            placeholder[0].unsqueeze(0).repeat(batch_size, 1, 1, 1),
            placeholder[1].unsqueeze(0).repeat(batch_size, 1, 1, 1),
            placeholder[2].unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1, 1),
            placeholder[3].unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1, 1),
            placeholder[4].unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1, 1),
            placeholder[5].unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1, 1),
            [placeholder[6]] * batch_size
        )


def merge_input_channels(input_data, empty_value=100.0, visualize=False):
    """合并多个输入通道，采用非空优先策略 - 优化版本"""
    # 创建空值掩码 (True表示有效值，False表示空值)
    valid_mask = (input_data != empty_value)

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
    merged_data = np.divide(valid_sum, valid_count, out=np.zeros_like(valid_sum), where=valid_count!=0)

    # 将没有有效值的位置设为空值
    merged_data[valid_count == 0] = empty_value

    return merged_data, False


# 示例使用
if __name__ == "__main__":
    jsonDir = r"E:\KingCrimson Dataset\Simulate\data0\json"
    arrayDir = r"E:\KingCrimson Dataset\Simulate\data0\arraynmask\array"
    maskDir = r"E:\KingCrimson Dataset\Simulate\data0\arraynmask\mask"
    targetDir = r"E:\KingCrimson Dataset\Simulate\data0\groundtruthstatus\statusarray"

    # 创建数据集
    dataset = DemDataset(
        jsonDir, arrayDir, maskDir, targetDir,
        min_valid_pixels=20,
        max_valid_pixels=800
    )
    
    # 使用标准DataLoader，但优化设置
    dataloader = DataLoader(
        dataset, 
        batch_size=8, 
        shuffle=True, 
        collate_fn=fast_collate_fn,
        num_workers=2,  # 可以使用多进程
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True  # 保持worker进程
    )
    
    print(f"数据集大小: {len(dataset)}")

    # 测试数据加载速度
    start_time = time.time()
    valid_batches = 0
    
    for i, batch_data in enumerate(dataloader):
        inputs, input_masks, input_targets, id_arrays, id_masks, id_targets, metadata = batch_data
        valid_batches += 1
        
        if i == 0:  # 只打印第一个批次的详细信息
            print(f"批次 {i+1}, 样本数: {len(inputs)}")
            for j in range(min(3, len(metadata))):  # 只打印前3个样本
                id_val = metadata[j][6].item()
                valid_pixels = torch.sum(input_masks[j] == 1).item()
                print(f"  样本 {j+1}: ID={id_val}, valid_pixels={valid_pixels}")
        
        if i >= 10:  # 测试10个批次来评估速度
            break
    
    end_time = time.time()
    print(f"测试完成，成功加载 {valid_batches} 个有效批次")
    print(f"处理时间: {end_time - start_time:.2f}秒")
    print(f"平均每批次时间: {(end_time - start_time) / valid_batches:.2f}秒")