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

class DEMData(Dataset):
    def __init__(self,cachedir,transform=None):
        """
        jsonDir: 存放 json 子文件夹的根目录
        arrayDir: 存放 array (.npy) 文件的目录
        maskDir: 存放 mask (.npy) 文件的目录
        targetDir: 存放 target (rs{id}.xyz) 文件的目录
        transform: 数据增强或归一化（可选）
        """
        
        self.kernel_size = 33  # 网格大小
        self.target_size = 600  # 目标大小
        self.cachedir = r"E:\KingCrimson Dataset\Simulate\data0\traindata"
        self.linputDir = os.path.join(self.cachedir, "input_data")
        self.lmaskDir = os.path.join(self.cachedir, "input_masks")
        self.ltargetDir = os.path.join(self.cachedir, "input_target")
        self.ginputDir = os.path.join(self.cachedir, "id_array")
        self.gmaskDir = os.path.join(self.cachedir, "id_mask")
        self.gtargetDir = os.path.join(self.cachedir, "id_target")
        self.metadataDir = os.path.join(self.cachedir, "metadata")
        self.transform = transform
        self.len=0
        self.dataGroup={}
        self.__readitem__()
    
    def __readitem__(self):
        total_files_num = 0
        for root, dirs, files in os.walk(self.linputDir):
            total_files_num += len(files)
            self.len=total_files_num
        print(f"Total files: {total_files_num}")
        '''from tqdm import tqdm  # 导入 tqdm 库用于显示进度条

        for i in tqdm(range(total_files_num), desc="Loading data"):
            subset = str(i // 5000)
            linput_path = os.path.join(self.linputDir, subset, f"{i}_input_data.npy")
            lmask_path = os.path.join(self.lmaskDir, subset, f"{i}_input_masks.npy")
            ltarget_path = os.path.join(self.ltargetDir, subset, f"{i}_input_target.npy")
            ginput_path = os.path.join(self.ginputDir, subset, f"{i}_id_array.npy")
            gmask_path = os.path.join(self.gmaskDir, subset, f"{i}_id_mask.npy")
            gtarget_path = os.path.join(self.gtargetDir, subset, f"{i}_id_target.npy")
            metadata_path = os.path.join(self.metadataDir, subset, f"{i}_metadata.npy")
            # 读取数据
            self.dataGroup[i] = {}
            self.dataGroup[i]["linput"] = np.load(linput_path,mmap_mode='r').astype(np.float32)
            self.dataGroup[i]["lmask"] = np.load(lmask_path,mmap_mode='r').astype(int)
            self.dataGroup[i]["ltarget"] = np.load(ltarget_path,mmap_mode='r').astype(np.float32)
            self.dataGroup[i]["ginput"] = np.load(ginput_path,mmap_mode='r').astype(np.float32)
            self.dataGroup[i]["gmask"] = np.load(gmask_path,mmap_mode='r').astype(int)
            self.dataGroup[i]["gtarget"] = np.load(gtarget_path,mmap_mode='r').astype(np.float32)
            self.dataGroup[i]["metadata"] = np.load(metadata_path,mmap_mode='r').astype(int)'''
        
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        subset = str(idx//5000)
        linput_path = os.path.join(self.linputDir, subset, f"{idx}_input_data.npy")
        lmask_path = os.path.join(self.lmaskDir, subset, f"{idx}_input_masks.npy")
        ltarget_path = os.path.join(self.ltargetDir, subset, f"{idx}_input_target.npy")
        ginput_path = os.path.join(self.ginputDir, subset, f"{idx}_id_array.npy")
        gmask_path = os.path.join(self.gmaskDir, subset, f"{idx}_id_mask.npy")
        gtarget_path = os.path.join(self.gtargetDir, subset, f"{idx}_id_target.npy")
        metadata_path = os.path.join(self.metadataDir, subset, f"{idx}_metadata.npy")
        # 读取数据
        linput = np.load(linput_path,mmap_mode='r').astype(np.float32)
        lmask = np.load(lmask_path,mmap_mode='r').astype(int)
        ltarget = np.load(ltarget_path,mmap_mode='r').astype(np.float32)
        ginput = np.load(ginput_path,mmap_mode='r').astype(np.float32)
        gmask = np.load(gmask_path,mmap_mode='r').astype(int)
        gtarget = np.load(gtarget_path,mmap_mode='r').astype(np.float32)
        metadata = np.load(metadata_path,mmap_mode='r').astype(int)
        # 转换为张量
        return (
            torch.tensor(linput, dtype=torch.float32),  # 输入数据组
            torch.tensor(lmask, dtype=torch.int),  # 输入掩码组
            torch.tensor(ltarget, dtype=torch.float32),  # 目标网格
            torch.tensor(ginput, dtype=torch.float32),  # ID对应的数组
            torch.tensor(gmask, dtype=torch.int),
            torch.tensor(gtarget, dtype=torch.float32),
            # ID对应的掩码
            metadata.astype(int),  # 元数据
        )
        '''return (
            torch.tensor(self.dataGroup[i]["linput"], dtype=torch.float32),  # 输入数据组
            torch.tensor(self.dataGroup[i]["lmask"], dtype=torch.int),  # 输入掩码组
            torch.tensor(self.dataGroup[i]["ltarget"], dtype=torch.float32),  # 目标网格
            torch.tensor(self.dataGroup[i]["ginput"], dtype=torch.float32),  # ID对应的数组
            torch.tensor(self.dataGroup[i]["gmask"], dtype=torch.int),
            torch.tensor(self.dataGroup[i]["gtarget"], dtype=torch.float32),
            # ID对应的掩码
            self.dataGroup[i]["metadata"].astype(int),  # 元数据
        )'''
    
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

# 示例使用
if __name__ == "__main__":
    jsonDir = r"E:\KingCrimson Dataset\Simulate\data0\json"  # 局部json
    arrayDir = r"E:\KingCrimson Dataset\Simulate\data0\arraynmask\array"  # 全局array
    maskDir = r"E:\KingCrimson Dataset\Simulate\data0\arraynmask\mask"  # 全局mask
    targetDir = (
        r"E:\KingCrimson Dataset\Simulate\data0\groundtruthstatus\statusarray" # 全局target
    )
    cacheDir = r"E:\KingCrimson Dataset\Simulate\data0\traindata"  # 缓存目录

    dataset = DEMData(cacheDir)
    dataloader = DataLoader(
        dataset, batch_size=4, shuffle=False, collate_fn=custom_collate_fn,num_workers=8,prefetch_factor=2  # 预取数据
    )