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
    def __init__(self, jsonDir, arrayDir, maskDir, targetDir, transform=None):
        """
        jsonDir: 存放 json 子文件夹的根目录
        arrayDir: 存放 array (.npy) 文件的目录
        maskDir: 存放 mask (.npy) 文件的目录
        targetDir: 存放 target (rs{id}.xyz) 文件的目录
        transform: 数据增强或归一化（可选）
        """
        self.jsonDir = jsonDir
        self.arrayDir = arrayDir
        self.maskDir = maskDir
        self.targetDir = targetDir
        self.transform = transform
        self.dataGroups = self._groupJsonFiles()
        self.kernel_size = 33  # 网格大小
        self.target_size = 600  # 目标大小
        self.cachedir = r"E:\KingCrimson Dataset\Simulate\data0\traindata"
        # self.meanz = -29.68914775044842

    def _groupJsonFiles(self):
        groups = {}
        for subfolder in os.listdir(self.jsonDir):
            subfolder_path = os.path.join(self.jsonDir, subfolder)
            print(f"processing folder: {subfolder}")
            if not os.path.isdir(subfolder_path):
                continue

            for filename in os.listdir(subfolder_path):
                # print(f"Processing file: {filename}")
                if filename.endswith(".json"):
                    parts = filename.split("_")
                    if len(parts) < 4:  # 确保文件名格式正确
                        continue

                    centerx, centery, id_val, id_adj = (
                        parts[0],
                        parts[1],
                        parts[2],
                        parts[3].split(".")[0],
                    )
                    key = (centerx, centery, id_val, id_adj)  # 三级分组: centerx, centery, id

                    if key not in groups:
                        groups[key] = {
                            "json_files": [],
                            "meta": {
                                "centerx": centerx,  # 核x绝对坐标
                                "centery": centery,  # 核y绝对坐标
                                "id": id_val,  # 自身栅格点云序列
                            },
                        }

                    json_path = os.path.join(subfolder_path, filename)
                    groups[key]["json_files"].append(json_path)

                    # 获取 xbegain, ybegain, xnum, ynum 信息
                    # 这些信息可以从json文件中读取，或者从文件名中提取
                    for filename in os.listdir(self.arrayDir):
                        if filename.endswith("_a.npy") and filename.startswith(
                            f"{id_val}_"
                        ):
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

                    """with open(json_path, "r", encoding="utf-8") as f:
                        json_content = json.load(f)
                        if (
                            "xbegain" in json_content
                            and "xbegain" not in groups[key]["meta"]
                        ):
                            groups[key]["meta"]["xbegain"] = json_content["xbegain"]
                            groups[key]["meta"]["ybegain"] = json_content["ybegain"]
                            groups[key]["meta"]["xnum"] = json_content["xnum"]
                            groups[key]["meta"]["ynum"] = json_content["ynum"]"""
        # 删除 json_files 数量为 1 的键
        """keys_to_remove = [
            key for key, group in groups.items() if len(group["json_files"]) == 1
        ]
        for key in keys_to_remove:
            del groups[key]"""
        '''new_groups = {}
        for key, group in groups.items():
            # print(group)
            if len(group["json_files"]) <= 1:
                new_groups[key] = group
                continue
            sorted_files = sorted(
                group["json_files"],
                key=lambda x: int(x.split("_")[-1].split(".")[0]),
            )
            id_adj_groups = {}
            current_paths = []
            for file in sorted_files:
                try:
                    id_adj = int(file.split("_")[-1].split(".")[0])
                except ValueError:
                    print(f"Error parsing id_adj from filename: {file}")
                    continue
                current_paths.append(file)
                id_adj_groups[id_adj] = current_paths.copy()
            for id_adj, paths in id_adj_groups.items():
                new_key = f"{key}{id_adj}"
                new_groups[new_key] = {
                    "json_files": paths,
                    "meta": group["meta"].copy(),
                }'''
                # print(new_groups[new_key])

        return groups

    def __len__(self):
        return len(self.dataGroups)
    
    def writecache(self, id, indata, inmask, intarget, iddata, idmask, idtarget, metadata):
        """
        将相关数据写入指定文件夹，每种输入类型使用单独的子文件夹，数据量过大时自动新建子文件夹。
        参数:
        - indata: 输入数据
        - inmask: 输入掩码
        - iddata: 全局数据
        - idmask: 全局掩码
        - idtarget: 全局目标
        - metadata: 元数据
        """
        output_dir = self.cachedir  # 输出文件夹路径
        indata_dir = os.path.join(output_dir, "input_data")
        inmask_dir = os.path.join(output_dir, "input_masks")
        itarget_dir = os.path.join(output_dir, "input_target")
        iddata_dir = os.path.join(output_dir, "id_array")
        idmask_dir = os.path.join(output_dir, "id_mask")
        idtarget_dir = os.path.join(output_dir, "id_target")
        metadata_dir = os.path.join(output_dir, "metadata")
        for idx, item in enumerate([indata_dir, inmask_dir, itarget_dir,iddata_dir, idmask_dir, idtarget_dir, metadata_dir]):
            if not os.path.exists(item):
                os.makedirs(item, exist_ok=True)
                print(f"create folder: {item}")
            subitem = os.path.join(item, str(id//5000))
            if id//5000 == 1000:
                break
            if not os.path.exists(subitem):
                os.makedirs(subitem, exist_ok=True)
                # print(f"create subfolder: {subitem}")
            idx=2
            match idx:
                case 0:
                    np.save(os.path.join(subitem, f"{id}_input_data.npy"), indata.astype(np.float32))
                case 1:
                    np.save(os.path.join(subitem, f"{id}_input_masks.npy"), inmask.astype(int))
                case 2:
                    np.save(os.path.join(subitem, f"{id}_input_target.npy"), intarget.astype(np.float32))
                case 3:
                    np.save(os.path.join(subitem, f"{id}_id_array.npy"), iddata.astype(np.float32))
                case 4:
                    np.save(os.path.join(subitem, f"{id}_id_mask.npy"), idmask.astype(int))
                case 5:
                    np.save(os.path.join(subitem, f"{id}_id_target.npy"), idtarget.astype(np.float32))
                case 6:
                    np.save(os.path.join(subitem, f"{id}_metadata.npy"), metadata.astype(int))        
            print(f"wrote {id} data")
            
            

    def preprocess_data(self, data):
        mask = data != 100  # 1 表示有效值，0 表示空值
        valid_data = data[mask]
        if valid_data.size > 0:
            mean = valid_data.mean()
            """std = valid_data.std()
            if std > 0:
                data[mask] = (data[mask] - mean) / std"""
            data[~mask] = mean  # 空值设为0，模型将学习补全
        return data, mask.astype(np.float32)

    def __getitem__(self, idx):
        # key = list(self.dataGroups.keys())[idx]  # 获取 (centerx, centery, id) 键
        key = next(itertools.islice(self.dataGroups.keys(), idx, idx + 1))
        group_data = self.dataGroups[key]
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
        id = int(meta.get("id", 0))

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
        # print(f"id:{id}, id_val:{id_val}")

        # 1. 读取输入数据（多个json文件） local input
        input_data = []
        for json_path in json_files:
            with open(json_path, "r", encoding="utf-8") as f:
                json_content = json.load(f)
                input_data.append(np.array(json_content["data"]))
        # 2. 读取id对应的array文件 global input
        array_filename = f"{id_val}_{xbegain}_{ybegain}_{xnum}_{ynum}_a.npy"
        array_path = os.path.join(self.arrayDir, array_filename)
        id_array = None
        if os.path.exists(array_path):
            id_array = np.load(array_path,mmap_mode='r').astype(np.float32)
        else:
            print(f"Warning: Array file {array_path} not found.")
            id_array = np.zeros((xnum, ynum))  # 创建默认数组

        # 3. 读取id对应的mask文件 global mask
        mask_filename = f"{id_val}_{xbegain}_{ybegain}_{xnum}_{ynum}_m.npy"
        mask_path = os.path.join(self.maskDir, mask_filename)
        id_mask = None
        if os.path.exists(mask_path):
            id_mask = np.load(mask_path).astype(np.float32)

        else:
            print(f"Warning: Mask file {mask_path} not found.")
            id_mask = np.zeros((xnum, ynum))  # 创建默认掩码

        # 4. 读取id对应的target文件 global groundtruth
        for filename in os.listdir(self.targetDir):
            if filename.startswith(f"rs_{id_val}_") and filename.endswith(".npy"):
                target_filename = filename
                break
        target_path = os.path.join(self.targetDir, target_filename)
        if os.path.exists(target_path):
            id_target = np.load(target_path,mmap_mode='r').astype(np.float32)  # [:, ::-1].copy().T

            # id_target -= self.meanz  # 减去均值

            # ##增加target扩张为600的代码
        else:
            print(f"Warning: Target file {target_path} not found.")
            # id_target = np.zeros((int(xnum), int(ynum)))  # 创建默认target

        # 5. 提取local的target
        [minx, maxx] = [
            int(centerx - xbegain) - self.kernel_size // 2,
            int(centerx - xbegain) + self.kernel_size // 2 + 1,
        ]
        [miny, maxy] = [
            int(centery - ybegain) - self.kernel_size // 2,
            int(centery - ybegain) + self.kernel_size // 2 + 1,
        ]
        input_target = id_target[minx:maxx, miny:maxy].view()  # 提取局部target

        # 剔除掉local在global中的部分
        # 查询 id_array 在 id_mask=0 的位置的值
        mask_zero_values = id_array[id_mask == 0][0]

        # 将这些值赋给 [minx:maxx, miny:maxy] 范围内的 id_array
        id_array[minx:maxx, miny:maxy] = mask_zero_values
        id_mask[minx:maxx, miny:maxy] = 0

        # 预处理输入数据
        input_data = np.stack(input_data, axis=0)  # (N, H, W)
        # print(f"input_data shape: {input_data.shape}")

        # 应用合并通道的处理
        merged_input_data, skip = merge_input_channels(
            input_data, empty_value=100
        )  # 得到 (1, H, W)
        # 如果需要跳过此批次，返回一个标记值或重新调用另一个索引

        '''if skip:
            if idx + 1 < len(self.dataGroups):
                return self.__getitem__(idx + 1)  # 跳过此批次，递归调用下一个索引'''
        
         # 继续使用合并后的数据进行处理
        input_data, input_masks = self.preprocess_data(merged_input_data)
        
        # self.writecache(idx, input_data, input_masks,input_target, id_array, id_mask, id_target, metadata)

        # 转换为张量
        return (
            torch.tensor(input_data, dtype=torch.float32),  # 输入数据组
            torch.tensor(input_masks, dtype=torch.int),  # 输入掩码组
            torch.tensor(input_target, dtype=torch.float32),  # 目标网格
            torch.tensor(id_array, dtype=torch.float32),  # ID对应的数组
            torch.tensor(id_mask, dtype=torch.int),
            torch.tensor(id_target, dtype=torch.float32),
            # ID对应的掩码
            metadata.astype(int),  # 元数据
        )


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

    ''# 统计每个位置的有效值数量
    valid_count = np.sum(valid_mask, axis=0, keepdims=True)
    '''# 检查是否需要跳过此批次
    merged_mask = (valid_count.squeeze() > 0).astype(np.uint8)
    true_count = np.sum(merged_mask)
    skip_batch = true_count == 1089'''


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

    # 可视化每个通道的掩码
    if visualize:
        vmin = min(np.nanmin(t) for t in input_data)
        vmax = max(np.nanmax(t) for t in input_data)
        num_channels = input_data.shape[0]
        print(f"通道数: {num_channels}")
        num_cols = min(4, num_channels + 1)
        num_rows = (num_channels + num_cols - 1) // num_cols

        plt.figure(figsize=(num_cols * 3, num_rows * 3))
        for i in range(num_channels):
            channel_mask = valid_mask[i]

            # 打印每个通道掩码的统计信息，帮助调试
            print(
                f"通道 {i} 掩码统计: True值数量={np.sum(channel_mask)}, 总像素数={channel_mask.size}"
            )

            plt.subplot(num_rows, num_cols, i + 1)
            # 确保使用严格的布尔值显示
            # plt.imshow(channel_mask, cmap="terrain", vmin=vmin, vmax=vmax)
            plt.imshow(channel_mask, cmap="binary", vmin=0, vmax=1)
            plt.title(f"通道 {i+1} 掩码")
            plt.colorbar(label="有效(1)/无效(0)")

        # 显示合并后的掩码
        merged_mask = valid_count.squeeze() > 0
        print(
            f"合并掩码统计: True值数量={np.sum(merged_mask)}, 总像素数={merged_mask.size}"
        )

        plt.subplot(
            num_rows,
            num_cols,
            num_channels + 1,  # if num_channels < num_rows * num_cols else 1,
        )
        plt.imshow(merged_mask, cmap="binary", vmin=0, vmax=1)
        plt.title("合并后掩码")
        plt.colorbar(label="有效(1)/无效(0)")

        plt.tight_layout()
        plt.suptitle("输入通道掩码可视化")
        plt.subplots_adjust(top=0.9)
        plt.show()

    return merged_data, False # skip_batch


def xyz_to_grid(xyz_data, grid_size=33):
    """将xyz点转换为网格数据"""
    if not xyz_data:
        return np.full((grid_size, grid_size), 0.0)

    x_coords = [point[0] for point in xyz_data]
    y_coords = [point[1] for point in xyz_data]
    z_coords = [point[2] for point in xyz_data]

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # 防止除以零
    x_range = x_max - x_min
    y_range = y_max - y_min

    if x_range == 0:
        x_range = 1
    if y_range == 0:
        y_range = 1

    z_array = np.full((grid_size, grid_size), 0.0)

    for x, y, z in xyz_data:
        row = int(((x - x_min) / x_range) * (grid_size - 1))
        col = int(((y - y_min) / y_range) * (grid_size - 1))
        row = min(max(row, 0), grid_size - 1)
        col = min(max(col, 0), grid_size - 1)
        z_array[row, col] = z

    return z_array


def visualize_dem_data(
    inputs,
    input_masks,
    input_targets,
    id_arrays,
    id_masks,
    id_targets,
    metadata,
    batch_idx,
    save_dir=None,
):
    """
    可视化每个批次中的DEM数据和掩码

    参数:
    - inputs: 输入数据列表
    - input_masks: 输入掩码列表
    - input_targets: 输入目标列表
    - id_arrays: ID数组列表
    - id_masks: ID掩码列表
    - id_targets: ID目标列表
    - metadata: 元数据张量
    - batch_idx: 批次索引
    - save_dir: 保存图像的目录，如果为None则不保存
    """
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 为每个样本创建可视化
    for i in range(len(inputs)):
        input_data = inputs[i].numpy()  # 转换为NumPy数组
        input_mask = input_masks[i].numpy()  # 转换掩码为NumPy数组
        input_target = input_targets[i].numpy()
        id_array = id_arrays[i].numpy()
        id_mask = id_masks[i].numpy()  # 全局掩码
        id_target = id_targets[i].numpy()

        # 提取元数据
        centerx, centery, xbegain, ybegain, xnum, ynum, id_val = metadata[i].tolist()
        """print(
            f"centerx: {centerx}, centery: {centery}, xbegain:{xbegain}, ybegain:{ybegain}, xnum:{xnum}, ynum:{ynum}, id_val: {id_val}"
        )"""

        # 创建3x2的子图布局
        fig, axs = plt.subplots(2, 3, figsize=(14, 16))
        """plt.subplots_adjust(
            left=0.3, bottom=0.1, hspace=0.4, wspace=0.4
        )  # 调整子图之间的间距"""
        fig.suptitle(
            f"batch {batch_idx+1} id {i+1}\nID: {id_val}, center: ({centerx}, {centery})",
            fontsize=16,
        )

        # 计算所有高程数据的统一颜色范围（以目标数据为准）
        all_targets = [input_target, id_target]
        valid_targets = [t for t in all_targets if np.any(np.isfinite(t))]

        if valid_targets:
            vmin = min(np.nanmin(t) for t in valid_targets)
            vmax = max(np.nanmax(t) for t in valid_targets)
        else:
            vmin, vmax = 0, 1

        # 1. 可视化局部输入数据
        # 如果输入数据有多个通道，只取第一个通道进行可视化
        if len(input_data.shape) > 2:
            input_data_viz = input_data[0]
        else:
            input_data_viz = input_data

        im1 = axs[0, 0].imshow(
            input_data_viz.squeeze(), cmap="terrain", vmin=vmin, vmax=vmax
        )
        axs[0, 0].set_title("local input(33x33)", fontsize=10)
        axs[0, 0].set_xlabel("X", fontsize=10)
        axs[0, 0].set_ylabel("Y", fontsize=10)
        # 共用colorbar，不在这里添加

        # 2. 可视化局部目标数据
        im2 = axs[0, 1].imshow(
            input_target.squeeze(), cmap="terrain", vmin=vmin, vmax=vmax
        )
        axs[0, 1].set_title("local target(33x33)", fontsize=10)
        axs[0, 1].set_xlabel("X", fontsize=10)
        axs[0, 1].set_ylabel("Y", fontsize=10)
        # 添加共用的colorbar
        cbar_ax1 = fig.add_axes([0.02, 0.4, 0.02, 0.2])  # 调整位置和大小
        plt.colorbar(im2, cax=cbar_ax1, label="elevation")

        # 3. 可视化全局输入数据
        im3 = axs[1, 0].imshow(id_array.squeeze(), cmap="terrain", vmin=vmin, vmax=vmax)
        axs[1, 0].set_title(f"global input({xnum}x{ynum})")
        axs[1, 0].set_xlabel("X", fontsize=10)
        axs[1, 0].set_ylabel("Y", fontsize=10)

        # 在全局图上标记局部区域
        minx = int(centerx - xbegain) - 33 // 2 - 1
        maxx = int(centerx - xbegain) + 33 // 2
        miny = int(centery - ybegain) - 33 // 2 - 1
        maxy = int(centery - ybegain) + 33 // 2
        # 确保边界在有效范围内
        """minx = max(0, min(minx, id_array.shape[0] - 1))
        maxx = max(0, min(maxx, id_array.shape[0]))
        miny = max(0, min(miny, id_array.shape[1] - 1))
        maxy = max(0, min(maxy, id_array.shape[1]))"""

        rect = plt.Rectangle(
            (miny, minx),
            maxy - miny,
            maxx - minx,
            edgecolor="red",
            facecolor="none",
            linewidth=2,
        )
        axs[1, 0].add_patch(rect)
        # 共用colorbar，不在这里添加

        # 4. 可视化全局目标数据
        im4 = axs[1, 1].imshow(
            id_target.squeeze(), cmap="terrain", vmin=vmin, vmax=vmax
        )
        axs[1, 1].set_title(f"global target({id_target.shape[0]}x{id_target.shape[1]})")
        axs[1, 1].set_xlabel("X", fontsize=10)
        axs[1, 1].set_ylabel("Y", fontsize=10)

        # 在全局目标图上也标记局部区域
        rect2 = plt.Rectangle(
            (miny, minx),
            maxy - miny,
            maxx - minx,
            edgecolor="red",
            facecolor="none",
            linewidth=2,
        )
        axs[1, 1].add_patch(rect2)
        # 为第二行添加共用的colorbar
        # cbar_ax2 = fig.add_axes([0.92, 0.4, 0.02, 0.2])  # 调整位置和大小
        # plt.colorbar(im4, cax=cbar_ax2, label="elevation")

        # 5. 可视化局部掩码
        if len(input_mask.shape) > 2:
            input_mask_viz = input_mask[0]
        else:
            input_mask_viz = input_mask

        im5 = axs[0, 2].imshow(input_mask_viz.squeeze(), cmap="binary", vmin=0, vmax=1)
        axs[0, 2].set_title("local mask(33x33)", fontsize=10)
        axs[0, 2].set_xlabel("X", fontsize=10)
        axs[0, 2].set_ylabel("Y", fontsize=10)
        # 掩码不使用colorbar

        # 6. 可视化全局掩码
        im6 = axs[1, 2].imshow(id_mask.squeeze(), cmap="binary", vmin=0, vmax=1)
        axs[1, 2].set_title(f"global mask({id_mask.shape[0]}x{id_mask.shape[1]})")
        axs[1, 2].set_xlabel("X", fontsize=10)
        axs[1, 2].set_ylabel("Y", fontsize=10)

        # 在全局掩码上标记局部区域
        rect_mask = plt.Rectangle(
            (miny, minx),
            maxy - miny,
            maxx - minx,
            edgecolor="red",
            facecolor="none",
            linewidth=2,
        )
        axs[1, 2].add_patch(rect_mask)
        # 掩码不使用colorbar

        plt.tight_layout(
            w_pad=1.0, rect=[0, 0, 0.9, 0.96]
        )  # 调整布局，给标题和colorbar留出空间
        plt.subplots_adjust(
            left=0.12, right=0.94, bottom=0.05, hspace=0.4, wspace=0.4
        )  # 调整子图之间的间距

        # 保存图像或显示
        if save_dir:
            save_path = os.path.join(
                save_dir, f"batch_{batch_idx+1}_sample_{i+1}_id_{id_val}.png"
            )
            plt.savefig(save_path, dpi=150)
            print(f"已保存图像: {save_path}")
        else:
            plt.show()

        plt.close(fig)


# 示例使用
if __name__ == "__main__":
    jsonDir = r"E:\KingCrimson Dataset\Simulate\data0\json"  # 局部json
    arrayDir = r"E:\KingCrimson Dataset\Simulate\data0\arraynmask\array"  # 全局array
    maskDir = r"E:\KingCrimson Dataset\Simulate\data0\arraynmask\mask"  # 全局mask
    targetDir = (
        r"E:\KingCrimson Dataset\Simulate\data0\groundtruthstatus\statusarray" # 全局target
    )

    dataset = DemDataset(jsonDir, arrayDir, maskDir, targetDir)
    dataloader = DataLoader(
        dataset, batch_size=4, shuffle=False, collate_fn=custom_collate_fn,num_workers=8,prefetch_factor=2  # 预取数据
    )
    print(f"数据集大小: {len(dataset)}")

    # 调用 __getitem__ 方法
    for i, item in enumerate(dataloader):
        inputs, input_masks, input_targets, id_arrays, id_masks, id_targets, metadata = item
    
    print("over")
    exit()
    

    # 过滤掉 __getitem__ 返回 None 的数据
    valid_data = [data for data in dataset if data is not None]
    dataloader = DataLoader(
        valid_data, batch_size=4, shuffle=True, collate_fn=custom_collate_fn
    )

    # 测试数据加载并为每个批次生成统一的可视化
    for i, (
        inputs,
        input_masks,
        input_targets,
        id_arrays,
        id_masks,
        id_targets,
        metadata,
    ) in enumerate(dataloader):
        print(f"\n批次 {i+1}:")

        # 打印数据形状信息
        print(f"输入数据: {[input.shape for input in inputs]}")  # 33*33
        print(f"输入掩码: {[mask.shape for mask in input_masks]}")  # 33*33
        print(f"目标数据: {[target.shape for target in input_targets]}")  # 33*33
        print(f"ID数组: {[arr.shape for arr in id_arrays]}")  # 600*600 global
        print(f"ID掩码: {[mask.shape for mask in id_masks]}")  # 600*600 globalmask
        print(
            f"目标数据: {[target.shape for target in id_targets]}"
        )  # 600*600 groundtruth

        # 可视化当前批次的所有样本
        # visualize_dem_data(inputs, input_targets, id_arrays, id_targets, metadata, i)
        # 在main函数的循环中修改调用方式
        visualize_dem_data(
            inputs,
            input_masks,  # 添加了局部掩码
            input_targets,
            id_arrays,
            id_masks,  # 添加了全局掩码
            id_targets,
            metadata,
            i,
        )
        # 只处理第一个批次，然后退出循环
        # break
