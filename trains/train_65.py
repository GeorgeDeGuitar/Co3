import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from torch.nn.utils import spectral_norm
from torch.optim import Adadelta, Adam
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw
import visdom
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from losses.loss import completion_network_loss
import torch.nn.functional as F

# Import your models
from models.CompletionNet import CompletionNetwork
from models.ContextDis import ContextDiscriminator

# Import your custom dataset
## from DemDataset1 import DemDataset
from data_manage.fastdatageneration import PreGeneratedMaskDataset

# from DEMData import DEMData

# Import the custom_collate_fn function from your dataset module
## from DemDataset import custom_collate_fn
# from DEMData import custom_collate_fn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.SurfaceDis import BoundaryQualityDiscriminator

import random
import gc
import copy
import glob
import psutil
import traceback
import tempfile
import shutil
import os
import re
import subprocess
import time
from contextlib import contextmanager
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


@contextmanager
def visdom_server():
    """上下文管理器，自动管理Visdom服务器的启动和关闭"""
    process = None
    # try:
    # 启动服务器
    process = subprocess.Popen(
        ["python", "-m", "visdom.server"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    print("正在启动Visdom服务器...")
    time.sleep(3)  # 等待服务器启动
    yield process
    """finally:
        # 确保服务器被关闭
        if process:
            process.terminate()
            process.wait()
            print("Visdom服务器已关闭")"""


# 稳定性追踪器
class StabilityTracker:
    def __init__(self):
        self.real_acc_ema = 0.5
        self.fake_acc_ema = 0.5
        self.recon_loss_ema = float("inf")
        self.alpha = 0.95
        self.collapse_threshold = 0.15
        self.recovery_steps = 0

    def update(self, real_acc, fake_acc, recon_loss):
        self.real_acc_ema = self.alpha * self.real_acc_ema + (1 - self.alpha) * real_acc
        self.fake_acc_ema = self.alpha * self.fake_acc_ema + (1 - self.alpha) * fake_acc
        self.recon_loss_ema = (
            self.alpha * self.recon_loss_ema + (1 - self.alpha) * recon_loss
        )

    def is_collapsed(self):
        return (
            self.real_acc_ema < self.collapse_threshold
            or self.fake_acc_ema < self.collapse_threshold
        )

    def get_loss_weights(self):
        if (
            self.real_acc_ema < 0.25
            and self.fake_acc_ema > 0.75
            or self.fake_acc_ema - self.real_acc_ema >= 0.4
        ):
            print("strenthening real sample weights")
            return 3.0, 0.3  # 加强真实样本权重
        elif (
            self.fake_acc_ema < 0.25
            and self.real_acc_ema > 0.75
            or self.real_acc_ema - self.fake_acc_ema >= 0.4
        ):
            print("strenthening fake sample weights")
            return 0.3, 3.0  # 加强假样本权重
        else:
            return 1.0, 1.0


# 内存管理工具函数
def cleanup_memory():
    """强制清理内存"""
    gc.collect()
    if torch.cuda.device_count() > 1:
        for i in range(torch.cuda.device_count()):
            torch.cuda.empty_cache()
            torch.cuda.synchronize(i)
    else:
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()


def get_gpu_memory_usage():
    """获取GPU内存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        return allocated, reserved
    return 0, 0


def check_memory_and_cleanup(threshold_gb=10.0):
    """检查内存使用并在超过阈值时清理"""
    if torch.cuda.device_count() > 1:
        total_allocated = 0
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            total_allocated += allocated
            if allocated > threshold_gb:
                print(f"GPU {i} 内存使用过高: {allocated:.2f}GB")

        if total_allocated > threshold_gb * torch.cuda.device_count():
            print(f"总GPU内存使用过高: {total_allocated:.2f}GB, 进行清理...")
            cleanup_memory()
    else:
        # 原有的单GPU逻辑
        allocated, reserved = get_gpu_memory_usage()
        if allocated > threshold_gb:
            print(f"GPU内存使用过高: {allocated:.2f}GB, 进行清理...")
            cleanup_memory()


def safe_tensor_operation(func, *args, **kwargs):
    """安全的tensor操作，自动处理内存不足"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"内存不足，第{attempt+1}次重试...")
                cleanup_memory()
                if attempt == max_retries - 1:
                    raise
            else:
                raise


# setx PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:98


def save_checkpoint(
    model_cn,
    model_cd,
    model_bqd,
    opt_cn,
    opt_cd,
    opt_bqd,
    step,
    phase,
    result_dir,
    best_val_loss=None,
    best_acc=None,
    cn_lr=None,
    cd_lr=None,
    bqd_lr=None,
    schedulers=None,  # 新增
    scalers=None,  # 新增
    stability_tracker=None,  # 新增
):
    """完整的检查点保存"""
    try:
        # 处理多GPU模型的state_dict
        def get_model_state_dict(model):
            if model is None:
                return None
            # 如果是DataParallel包装的模型，使用.module获取原始模型
            if hasattr(model, "module"):
                return model.module.state_dict()
            else:
                return model.state_dict()

        checkpoint = {
            # 模型状态
            "model_cn_state_dict": get_model_state_dict(model_cn),
            "model_cd_state_dict": get_model_state_dict(model_cd),
            "model_bqd_state_dict": get_model_state_dict(model_bqd),
            # 优化器状态
            "opt_cn_state_dict": opt_cn.state_dict() if opt_cn is not None else None,
            "opt_cd_state_dict": opt_cd.state_dict() if opt_cd is not None else None,
            "opt_bqd_state_dict": opt_bqd.state_dict() if opt_bqd is not None else None,
            # 训练状态
            "step": step,
            "phase": phase,
            "best_val_loss": best_val_loss,
            "best_acc": best_acc,
            "cn_lr": cn_lr,
            "cd_lr": cd_lr,
            "bqd_lr": bqd_lr,
            # 随机数状态
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),  # 添加Python random状态
            # CUDA随机数状态
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            # 学习率调度器状态
            "schedulers": {},
            "scalers": {},
            "stability_tracker": None,
        }

        # 保存调度器状态
        if schedulers:
            for name, scheduler in schedulers.items():
                if scheduler is not None:
                    checkpoint["schedulers"][name] = scheduler.state_dict()

        # 保存缩放器状态
        if scalers:
            for name, scaler in scalers.items():
                if scaler is not None:
                    checkpoint["scalers"][name] = scaler.state_dict()

        # 保存稳定性追踪器
        if stability_tracker is not None:
            checkpoint["stability_tracker"] = {
                "real_acc_ema": stability_tracker.real_acc_ema,
                "fake_acc_ema": stability_tracker.fake_acc_ema,
                "recon_loss_ema": stability_tracker.recon_loss_ema,
                "alpha": stability_tracker.alpha,
                "recovery_steps": stability_tracker.recovery_steps,
            }

        # 确保结果目录存在
        os.makedirs(result_dir, exist_ok=True)

        checkpoint_filename = f"checkpoint_phase{phase}_step{step}.pth"
        checkpoint_path = os.path.join(result_dir, checkpoint_filename)

        # 使用临时文件安全保存
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".pth.tmp", dir=result_dir
        ) as tmp_file:
            temp_path = tmp_file.name

        try:
            # 保存到临时文件
            print(f"正在保存检查点到临时文件: {temp_path}")
            torch.save(checkpoint, temp_path)

            # 验证临时文件完整性
            print("验证检查点文件完整性...")
            test_checkpoint = torch.load(
                temp_path, map_location="cpu", weights_only=False
            )
            required_keys = ["model_cn_state_dict", "step", "phase"]
            for key in required_keys:
                if key not in test_checkpoint:
                    raise ValueError(f"检查点验证失败：缺少键 {key}")
            del test_checkpoint  # 释放内存

            # 如果验证通过，移动临时文件到最终位置
            if os.path.exists(checkpoint_path):
                # 备份现有文件
                backup_path = checkpoint_path + ".backup"
                shutil.move(checkpoint_path, backup_path)
                print(f"备份现有检查点到: {backup_path}")

            shutil.move(temp_path, checkpoint_path)
            print(f"成功保存检查点到: {checkpoint_path}")

            # 保存最新检查点的副本
            latest_checkpoint_path = os.path.join(result_dir, "latest_checkpoint.pth")
            shutil.copy2(checkpoint_path, latest_checkpoint_path)
            print(f"保存最新检查点到: {latest_checkpoint_path}")

            # 现在安全地删除旧文件（保留最近的3个）
            cleanup_old_checkpoints(result_dir, phase, step, keep_count=3)

        except Exception as e:
            # 如果保存失败，清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e

        # 清理checkpoint变量
        del checkpoint
        cleanup_memory()

        return checkpoint_path

    except Exception as e:
        print(f"保存检查点时发生错误: {e}")
        cleanup_memory()
        raise


def cleanup_old_checkpoints(result_dir, current_phase, current_step, keep_count=3):
    """
    清理旧的检查点文件，保留最近的几个
    """
    try:
        # 找到同一阶段的所有检查点文件
        pattern = os.path.join(result_dir, f"checkpoint_phase{current_phase}_step*.pth")
        checkpoint_files = glob.glob(pattern)

        # 排除当前步数的文件
        current_filename = f"checkpoint_phase{current_phase}_step{current_step}.pth"
        checkpoint_files = [
            f for f in checkpoint_files if not f.endswith(current_filename)
        ]

        # 按步数排序（从文件名提取步数）
        def extract_step(filepath):
            try:
                filename = os.path.basename(filepath)
                # 提取step数字
                step_part = filename.split("_step")[1].split(".pth")[0]
                return int(step_part)
            except:
                return 0

        checkpoint_files.sort(key=extract_step, reverse=True)

        # 删除多余的文件
        files_to_delete = checkpoint_files[keep_count:]
        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                print(f"删除旧检查点: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"删除文件失败 {file_path}: {e}")

        # 同时清理备份文件
        backup_files = glob.glob(os.path.join(result_dir, "*.backup"))
        backup_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        for backup_file in backup_files[keep_count:]:
            try:
                os.remove(backup_file)
                print(f"删除旧备份: {os.path.basename(backup_file)}")
            except Exception as e:
                print(f"删除备份文件失败 {backup_file}: {e}")

    except Exception as e:
        print(f"清理旧检查点时出错: {e}")


def load_checkpoint(
    checkpoint_path,
    model_cn,
    model_cd,
    model_bqd,
    opt_cn,
    opt_cd,
    opt_bqd,
    device,
    schedulers=None,
    scalers=None,
):
    """完整的检查点加载"""
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )

        # 处理多GPU模型的state_dict加载
        def load_model_state_dict(model, state_dict_key):
            if model is None or checkpoint.get(state_dict_key) is None:
                return

            state_dict = checkpoint[state_dict_key]

            # 处理DataParallel模型
            if hasattr(model, "module"):
                # 当前是DataParallel，检查state_dict是否有'module.'前缀
                sample_key = list(state_dict.keys())[0]
                if not sample_key.startswith("module."):
                    # state_dict没有'module.'前缀，需要添加
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        new_state_dict["module." + k] = v
                    model.load_state_dict(new_state_dict)
                else:
                    model.load_state_dict(state_dict)
            else:
                # 当前不是DataParallel，检查state_dict是否有'module.'前缀
                sample_key = list(state_dict.keys())[0]
                if sample_key.startswith("module."):
                    # state_dict有'module.'前缀，需要移除
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        new_state_dict[k[7:]] = v  # 移除'module.'前缀
                    model.load_state_dict(new_state_dict)
                else:
                    model.load_state_dict(state_dict)

        # 加载模型
        load_model_state_dict(model_cn, "model_cn_state_dict")
        # load_model_state_dict(model_cd, "model_cd_state_dict")
        load_model_state_dict(model_bqd, "model_bqd_state_dict")

        # 加载优化器
        if opt_cn and checkpoint.get("opt_cn_state_dict"):
            opt_cn.load_state_dict(checkpoint["opt_cn_state_dict"])
        if opt_cd and checkpoint.get("opt_cd_state_dict") and False:
            opt_cd.load_state_dict(checkpoint["opt_cd_state_dict"])
        if opt_bqd and checkpoint.get("opt_bqd_state_dict"):
            opt_bqd.load_state_dict(checkpoint["opt_bqd_state_dict"])

        # 恢复随机数状态
        """if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        if "numpy_rng_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_rng_state"])
        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])
        if "cuda_rng_state" in checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])"""

        # 恢复调度器状态
        if schedulers and "schedulers" in checkpoint:
            for name, scheduler in schedulers.items():
                if scheduler is not None and name in checkpoint["schedulers"]:
                    scheduler.load_state_dict(checkpoint["schedulers"][name])

        # 恢复缩放器状态
        if scalers and "scalers" in checkpoint:
            for name, scaler in scalers.items():
                if scaler is not None and name in checkpoint["scalers"]:
                    scaler.load_state_dict(checkpoint["scalers"][name])

        # 恢复稳定性追踪器
        stability_tracker = None
        if "stability_tracker" in checkpoint and checkpoint["stability_tracker"]:
            stability_tracker = StabilityTracker()
            st_data = checkpoint["stability_tracker"]
            stability_tracker.real_acc_ema = st_data["real_acc_ema"]
            stability_tracker.fake_acc_ema = st_data["fake_acc_ema"]
            stability_tracker.recon_loss_ema = st_data["recon_loss_ema"]
            stability_tracker.alpha = st_data.get("alpha", 0.95)
            stability_tracker.recovery_steps = st_data.get("recovery_steps", 0)

        # 提取其他信息
        step = checkpoint["step"]
        phase = checkpoint["phase"]
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        best_acc = checkpoint.get("best_acc", float("-inf"))
        cn_lr = checkpoint.get("cn_lr", None)
        cd_lr = checkpoint.get("cd_lr", None)
        bqd_lr = checkpoint.get("bqd_lr", None)

        print(f"成功恢复检查点：阶段 {phase}，步骤 {step}")
        print(f"随机数状态已恢复")

        return (
            step,
            phase,
            best_val_loss,
            best_acc,
            cn_lr,
            cd_lr,
            bqd_lr,
            stability_tracker,
        )

    except Exception as e:
        print(f"加载检查点失败: {e}")
        raise


def ensure_scalar_loss(loss):
    """
    确保loss是标量，处理多GPU情况
    """
    if isinstance(loss, torch.Tensor):
        if loss.dim() > 0:
            return loss.mean()
        else:
            return loss
    else:
        return loss


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
        global_targets = global_targets.to(
            device, dtype=torch.float32, non_blocking=True
        )

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
    print(
        f"  测试集: {len(test_indices)} ({len(test_indices)/total_size:.1%}) - 索引 0 到 {test_size-1}"
    )
    print(f"  验证集: {len(val_indices)} ({len(val_indices)/total_size:.1%})")
    print(f"  训练集: {len(train_indices)} ({len(train_indices)/total_size:.1%})")

    return train_dataset, val_dataset, test_dataset


def generate_target_mask(batch_global_masks, metadata):
    """
    生成target mask，并验证是否包含global_masks的所有1值

    Args:
        batch_global_masks: 全局掩码张量 [batch_size, channels, height, width], dtype=torch.int
        metadata: numpy数组或tuple列表，每个元素格式为 [centerx, centery, xbegain, ybegain, xnum, ynum, id_val]

    Returns:
        target_masks: 生成的target mask张量 [batch_size, channels, height, width], dtype与global_masks相同
        success_flags: 每个样本是否成功生成的标志列表
    """
    batch_size = batch_global_masks.shape[0]
    device = batch_global_masks.device
    dtype = batch_global_masks.dtype  # 保持与global_masks相同的数据类型

    # 创建与batch_global_masks相同shape和dtype的target_masks
    target_masks = torch.zeros_like(batch_global_masks, dtype=dtype, device=device)
    success_flags = []

    for i in range(batch_size):
        try:
            meta = metadata[i]

            # 处理不同类型的metadata (numpy数组、tuple、list等)
            if hasattr(meta, "__getitem__"):  # 可索引对象
                # metadata数组索引: [centerx, centery, xbegain, ybegain, xnum, ynum, id_val]
                #                    [0,       1,       2,       3,       4,     5,     6]

                # 安全地提取xnum和ynum，并转换为整数
                try:
                    xnum_raw = meta[4]
                    ynum_raw = meta[5]

                    # 处理可能的tensor、numpy或其他数值类型
                    if hasattr(xnum_raw, "item"):  # tensor或numpy标量
                        xnum = int(xnum_raw.item())
                    else:
                        xnum = int(float(xnum_raw))  # 先转float再转int，处理字符串情况

                    if hasattr(ynum_raw, "item"):
                        ynum = int(ynum_raw.item())
                    else:
                        ynum = int(float(ynum_raw))

                except (ValueError, TypeError, IndexError) as e:
                    print(f"ERROR: Batch {i}: 无法解析metadata中的xnum/ynum: {e}")
                    print(f"  Metadata: {meta}")
                    print(f"  Metadata type: {type(meta)}")
                    success_flags.append(False)
                    continue
            else:
                print(f"ERROR: Batch {i}: metadata格式不支持: {type(meta)}")
                success_flags.append(False)
                continue

            # 获取图像尺寸
            batch_size_dim, channels, height, width = batch_global_masks.shape

            # 第一次尝试：使用原始xnum和ynum
            target_mask = create_mask(
                height, width, ynum, xnum, device, dtype, channels
            )

            # 确保target_mask是tensor而不是tuple - 关键修复点1
            if isinstance(target_mask, tuple):
                target_mask = target_mask[0]  # 如果是tuple，取第一个元素

            # 验证是否包含global_masks的所有1值
            global_mask = batch_global_masks[i]
            if validate_mask_inclusion(global_mask, target_mask):
                target_masks[i] = target_mask
                success_flags.append(True)
                # print(f"Batch {i}: 成功生成target mask (xnum={xnum}, ynum={ynum})")
            else:
                print(f"Batch {i}: 第一次验证失败，交换xnum和ynum重试...")

                # 第二次尝试：交换xnum和ynum
                target_mask = create_mask(
                    height, width, xnum, ynum, device, dtype, channels
                )

                # 确保target_mask是tensor而不是tuple - 关键修复点2
                if isinstance(target_mask, tuple):
                    target_mask = target_mask[0]

                if validate_mask_inclusion(global_mask, target_mask):
                    target_masks[i] = target_mask
                    success_flags.append(True)
                    print(
                        f"Batch {i}: 交换后成功生成target mask (xnum={ynum}, ynum={xnum})"
                    )
                else:
                    print(f"ERROR: Batch {i}: 交换xnum和ynum后仍然验证失败!")
                    print(f"  原始参数: xnum={xnum}, ynum={ynum}")
                    print(f"  图像尺寸: {height}x{width}")
                    print(f"  Global mask中1的数量: {global_mask.sum().item()}")
                    print(f"  Target mask尺寸: {target_mask.shape}")
                    success_flags.append(False)

        except Exception as e:
            print(f"ERROR: Batch {i}: 处理metadata时发生异常: {e}")
            print(f"  Metadata: {metadata[i] if i < len(metadata) else 'N/A'}")
            print(
                f"  Metadata type: {type(metadata[i]) if i < len(metadata) else 'N/A'}"
            )
            success_flags.append(False)

    # 确保返回的是纯tensor，不是tuple - 关键修复点3
    if isinstance(target_masks, tuple):
        target_masks = target_masks[0]

    assert isinstance(
        target_masks, torch.Tensor
    ), f"target_masks应该是tensor，但得到了{type(target_masks)}"
    assert (
        target_masks.shape == batch_global_masks.shape
    ), f"shape不匹配: {target_masks.shape} vs {batch_global_masks.shape}"

    return target_masks, success_flags


def create_mask(height, width, x_size, y_size, device, dtype, channels):
    """
    在左上角创建指定尺寸的mask

    Args:
        height, width: 图像尺寸
        x_size, y_size: mask的尺寸
        device: 设备类型
        dtype: 数据类型 (torch.int)
        channels: 通道数

    Returns:
        mask: 生成的mask张量 [channels, height, width] - 纯tensor，不是tuple
    """
    # 确保所有参数都是整数
    height = int(height)
    width = int(width)
    x_size = int(x_size)
    y_size = int(y_size)
    channels = int(channels)

    # 确保mask尺寸不超过图像尺寸
    x_size = min(x_size, width)
    y_size = min(y_size, height)

    # 创建与global_masks相同格式的mask
    mask = torch.zeros(channels, height, width, device=device, dtype=dtype)
    mask[:, :y_size, :x_size] = 1

    # 确保返回的是tensor，不是tuple - 额外保护
    if isinstance(mask, tuple):
        mask = mask[0]

    assert isinstance(mask, torch.Tensor), f"mask应该是tensor，但得到了{type(mask)}"

    return mask


def validate_mask_inclusion(global_mask, target_mask):
    """
    验证target_mask是否包含global_mask的所有1值

    Args:
        global_mask: 全局掩码
        target_mask: 目标掩码

    Returns:
        bool: 是否包含所有1值
    """
    # 确保输入都是tensor
    if isinstance(global_mask, tuple):
        global_mask = global_mask[0]
    if isinstance(target_mask, tuple):
        target_mask = target_mask[0]

    # 检查是否存在global_mask为1但target_mask为0的位置
    invalid_positions = (global_mask == 1) & (target_mask == 0)
    return not invalid_positions.any().item()


def safe_extract_metadata(meta, index, default=0):
    """
    安全地从metadata中提取数值

    Args:
        meta: metadata对象 (可能是numpy数组、tuple、list等)
        index: 索引位置
        default: 默认值

    Returns:
        提取并转换后的数值
    """
    try:
        if hasattr(meta, "__getitem__") and len(meta) > index:
            value = meta[index]

            # 处理不同的数值类型
            if hasattr(value, "item"):  # tensor或numpy标量
                return value.item()
            elif isinstance(value, (int, float)):
                return float(value)
            elif isinstance(value, str):
                return float(value)
            else:
                return float(value)
        else:
            return default
    except Exception as e:
        print(f"Warning: 提取metadata[{index}]时出错: {e}, 使用默认值 {default}")
        return default


# 安全的类型转换函数
def safe_tensor_float(tensor_or_tuple):
    """
    安全地将tensor或tuple转换为float tensor

    Args:
        tensor_or_tuple: 可能是tensor或包含tensor的tuple

    Returns:
        float类型的tensor
    """
    if isinstance(tensor_or_tuple, tuple):
        # 如果是tuple，假设第一个元素是tensor
        tensor = tensor_or_tuple[0]
    else:
        tensor = tensor_or_tuple

    if isinstance(tensor, torch.Tensor):
        return tensor.float()
    else:
        raise TypeError(
            f"Expected tensor or tuple containing tensor, got {type(tensor)}"
        )


def nomarlize(local_inputs, local_targets, global_inputs, global_targets):
    ## 归一化
    # 基于global_input的数据范围进行归一化（已经用均值填充，无需考虑mask）
    global_min = global_inputs.min()
    global_max = global_inputs.max()

    # 避免除零错误
    if global_max - global_min > 1e-8:
        # 归一化到[0, 1]
        local_inputs = (local_inputs - global_min) / (global_max - global_min)
        local_targets = (local_targets - global_min) / (global_max - global_min)
        global_inputs = (global_inputs - global_min) / (global_max - global_min)
        global_targets = (global_targets - global_min) / (global_max - global_min)
        return local_inputs, local_targets, global_inputs, global_targets
    else:
        print("Warning: global_inputs的最大值和最小值相等，无法进行归一化。")
        return local_inputs, local_targets, global_inputs, global_targets


def train(dir, envi, cuda, batch, test=False, resume_from=None, Phase=1):
    try:
        # =================================================
        # Configuration settings (replace with your settings)
        # =================================================
        if test:
            json_dir = r"E:\KingCrimson Dataset\Simulate\data0\testjson"
        else:
            json_dir = r"E:\KingCrimson Dataset\Simulate\data0\json"  # 局部json
        array_dir = (
            r"E:\KingCrimson Dataset\Simulate\data0\arraynmask_a\array"  # 全局array
        )
        mask_dir = (
            r"E:\KingCrimson Dataset\Simulate\data0\arraynmask_a\mask"  # 全局mask
        )
        target_dir = r"E:\KingCrimson Dataset\Simulate\data0\groundtruthstatus\statusarray"  # 全局target
        result_dir = rf"E:\KingCrimson Dataset\Simulate\data0\{dir}"  # 结果保存路径

        # Training parameters
        steps_1 = 50000  # 160000  # Phase 1 training steps
        steps_2 = 50000  # Phase 2 training steps
        steps_3 = 200000  # Phase 3 training steps

        snaperiod_1 = 500  # How often to save snapshots in phase 1
        snaperiod_2 = 500  # How often to save snapshots in phase 2
        snaperiod_3 = 500  # How often to save snapshots in phase 3

        batch_size = batch  # Batch size
        alpha = 4e-1  # Alpha parameter for loss weighting
        validation_split = 0.1  # Percentage of data to use for validation

        # 初始内存清理
        cleanup_memory()

        # Hardware setup
        if not torch.cuda.is_available():
            raise Exception("At least one GPU must be available.")
        # 自动检测和设置GPU
        if torch.cuda.device_count() > 1 and cuda == "multi":
            print(f"检测到 {torch.cuda.device_count()} 个GPU")
            use_multi_gpu = True
            # 自动设置主设备为cuda:0
            device = torch.device("cuda:0")
            print(f"使用多GPU训练，主设备: {device}")
            print(f"可用GPU: {[f'cuda:{i}' for i in range(torch.cuda.device_count())]}")
        else:
            use_multi_gpu = False
            # 使用传入的cuda参数
            device = torch.device(cuda)
            print(f"使用单GPU训练: {device}")

        torch.cuda.set_device(device)

        # Initialize Visdom with error handling
        try:
            viz = visdom.Visdom(env=envi)
            viz.close(env=envi)
        except Exception as e:
            print(f"Visdom初始化失败: {e}")
            viz = None

        # =================================================
        # Preparation
        # =================================================
        # Create result directories
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        for phase in ["phase_1", "phase_2", "phase_3", "validation", "bqd"]:
            if not os.path.exists(os.path.join(result_dir, phase)):
                os.makedirs(os.path.join(result_dir, phase))

        cleanup_memory()

        size = 65

        # 数据集参数配置
        reference_dir = (
            r"e:\KingCrimson Dataset\Simulate\data0\ST2\statusarray"  # 参考数据目录
        )
        mask_cache_dir = f"./mask_cache_{size}"  # mask缓存目录
        local_sizes = [size]  # 支持多种局部尺寸
        samples_per_file = 20  # 每个文件生成的样本数
        global_coverage_range = (0.5, 1.0)  # 全局覆盖率范围
        local_missing_range = (0.3, 0.7)  # 局部缺失率范围
        num_global_masks = 500  # 全局mask模板数量
        num_local_masks = 500  # 局部mask模板数量
        batch_size = 8  # 批次大小
        validation_split = 0.1  # 验证集比例

        # 创建数据集
        print("🚀 加载预生成mask数据集...")
        try:
            dataset = PreGeneratedMaskDataset(
                reference_dir=reference_dir,
                mask_cache_dir=mask_cache_dir,
                local_sizes=local_sizes,
                samples_per_file=samples_per_file,
                global_coverage_range=global_coverage_range,
                local_missing_range=local_missing_range,
                num_global_masks=num_global_masks,
                num_local_masks=num_local_masks,
                regenerate_masks=False,  # 使用已有的预生成mask
                seed=42,
                cache_reference_files=True,
                visualization_samples=0,
                enable_edge_connection=True,  # 启用边缘连接
            )
            full_dataset = dataset
            print(f"✅ 数据集加载成功!")
            print(f"   总样本数: {len(full_dataset)}")
            print(f"   Global masks: {len(dataset.global_masks)}")
            print(f"   Local masks数量:")
            for size in local_sizes:
                if size in dataset.local_masks:
                    print(f"     {size}x{size}: {len(dataset.local_masks[size])}")
        except Exception as e:
            print(f"❌ 加载数据集失败: {e}")
            raise

        # 分割数据集
        dataset_size = len(full_dataset)
        val_size = int(validation_split * dataset_size)
        train_size = dataset_size - val_size

        print(f"\n📊 数据集分割:")
        print(f"   总数据量: {dataset_size}")
        print(f"   训练集: {train_size} ({100*(1-validation_split):.0f}%)")
        print(f"   验证集: {val_size} ({100*validation_split:.0f}%)")

        # 使用固定种子确保可重现性
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size], generator=generator
        )

        # 创建训练数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,  # 启用shuffle
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            drop_last=True,  # 丢弃最后不完整的批次
        )

        # 创建验证数据加载器
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=True,  # 启用shuffle
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            drop_last=False,
        )

        # 创建验证子集（用于快速验证）
        val_subset_size = min(100, len(val_dataset))
        val_indices = np.random.choice(
            len(val_dataset), size=val_subset_size, replace=False
        )
        val_subset = torch.utils.data.Subset(val_dataset, val_indices)

        val_subset_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=True,  # 启用shuffle
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            drop_last=False,
        )

        # =================================================
        # Setup Models
        # =================================================
        # Initialize models with error handling
        try:
            model_cn = CompletionNetwork(input_channels=1, local_size=size).to(device)
            model_cd = ContextDiscriminator(
                local_input_channels=1,
                local_input_size=size,
                global_input_channels=1,
                global_input_size=600,
            ).to(device)
            model_bqd = BoundaryQualityDiscriminator(
                input_channels=1, input_size=size
            ).to(device)
            model_cn = model_cn.float()
            model_cd = model_cd.float()
            model_bqd = model_bqd.float()

            # 包装为多GPU模型
            if use_multi_gpu:
                model_cn = nn.DataParallel(model_cn)
                model_cd = nn.DataParallel(model_cd)
                model_bqd = nn.DataParallel(model_bqd)
                print("模型已包装为DataParallel")
        except Exception as e:
            print(f"模型初始化失败: {e}")
            cleanup_memory()
            raise

        # Setup optimizers
        opt_cn_p1 = Adadelta(model_cn.parameters())
        opt_bqd = torch.optim.Adam(
            model_bqd.parameters(), lr=5e-4, betas=(0.5, 0.999), eps=1e-8
        )
        # opt_cd_p1 = Adadelta(model_cd.parameters())
        opt_cd_p2 = torch.optim.Adam(
            model_cd.parameters(), lr=1e-5, betas=(0.5, 0.999), eps=1e-8
        )
        opt_cn_p3 = torch.optim.Adam(
            model_cn.parameters(), lr=1e-4, betas=(0.5, 0.999), eps=1e-8
        )
        opt_cd_p3 = torch.optim.Adam(
            model_cd.parameters(), lr=1e-5, betas=(0.5, 0.999), eps=1e-8
        )

        # 初始化训练状态变量
        phase = Phase  # 从第二阶段开始
        step_phase1 = 0
        step_phase2 = 0
        step_phase3 = 0
        best_val_loss = float("inf")
        best_bqd_loss = float("inf")
        best_acc = float("-inf")
        best_val_loss_joint = float("inf")
        cn_lr = None
        cd_lr = None
        bqd_lr = None

        # 获取当前本地时间
        current_time = time.localtime()

        # 获取年月日
        year = current_time.tm_year
        month = current_time.tm_mon
        day = current_time.tm_mday
        hour = current_time.tm_hour  # 24小时制 (0-23)
        minute = current_time.tm_min  # 分钟 (0-59)
        # 如果指定了检查点文件，则加载它恢复训练
        interrupt = f"_{year}_{month}_{day}_{hour}_{minute}"
        if resume_from:
            try:
                # 根据不同阶段准备相应的优化器和调度器
                schedulers_dict = None
                scalers_dict = None
                loaded_stability_tracker = None

                # 首次加载获取基本信息
                (
                    step,
                    phase,
                    best_val_loss,
                    best_acc,
                    cn_lr,
                    cd_lr,
                    bqd_lr,
                    loaded_stability_tracker,
                ) = load_checkpoint(
                    resume_from,
                    model_cn,
                    model_cd,
                    model_bqd,
                    None,  # 先不加载优化器，根据phase决定
                    None,
                    None,
                    device,
                )

                print(f"检查点信息: 阶段={phase}, 步骤={step}")

                # 根据加载的阶段设置相应的训练状态
                if phase == 1:
                    print("恢复Phase 1训练状态...")
                    # 重新加载，使用正确的优化器
                    (
                        step,
                        phase,
                        best_val_loss,
                        best_acc,
                        cn_lr,
                        cd_lr,
                        bqd_lr,
                        loaded_stability_tracker,
                    ) = load_checkpoint(
                        resume_from,
                        model_cn,
                        model_cd,
                        model_bqd,
                        opt_cn_p1,  # Phase 1使用的优化器
                        None,  # Phase 1不使用CD优化器
                        opt_bqd,
                        device,
                        schedulers=None,  # Phase 1通常不使用调度器
                        scalers=None,
                    )

                    step_phase1 = step
                    step_phase2 = 0
                    step_phase3 = 0
                    # best_val_loss保持加载的值

                elif phase == 2:
                    print("恢复Phase 2训练状态...")
                    # Phase 2完成后切换到Phase 3的优化器设置
                    (
                        step,
                        phase,
                        best_val_loss,
                        best_acc,
                        cn_lr,
                        cd_lr,
                        bqd_lr,
                        loaded_stability_tracker,
                    ) = load_checkpoint(
                        resume_from,
                        model_cn,
                        model_cd,
                        model_bqd,
                        opt_cn_p1,  # 保持Phase 1的CN优化器
                        opt_cd_p2,  # Phase 2的CD优化器
                        opt_bqd,
                        device,
                        schedulers=None,  # Phase 2的调度器
                        scalers=None,
                    )

                    step_phase1 = steps_1
                    step_phase2 = step
                    step_phase3 = 0
                    # best_acc保持加载的值

                elif phase == 3:
                    print("恢复Phase 3训练状态...")

                    # Phase 3需要初始化调度器和缩放器
                    # 注意：这些变量在Phase 3代码中才会创建，这里需要先创建
                    scheduler_cn = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        opt_cn_p3, mode="min", factor=0.7, patience=8, min_lr=1e-6
                    )
                    scheduler_cd = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        opt_cd_p3, mode="max", factor=0.7, patience=5, min_lr=1e-7
                    )

                    scaler_cn = GradScaler(
                        enabled=True, init_scale=2**10, growth_interval=2000
                    )
                    scaler_cd = GradScaler(
                        enabled=True, init_scale=2**10, growth_interval=2000
                    )
                    scaler_bqd_p3 = GradScaler()

                    # 准备调度器和缩放器字典
                    schedulers_dict = {"cn": scheduler_cn, "cd": scheduler_cd}
                    scalers_dict = {
                        "cn": scaler_cn,
                        "cd": scaler_cd,
                        "bqd": scaler_bqd_p3,
                    }

                    # 重新加载，包含完整的状态
                    (
                        step,
                        phase,
                        best_val_loss,
                        best_acc,
                        cn_lr,
                        cd_lr,
                        bqd_lr,
                        loaded_stability_tracker,
                    ) = load_checkpoint(
                        resume_from,
                        model_cn,
                        model_cd,
                        model_bqd,
                        opt_cn_p3,  # Phase 3的CN优化器
                        opt_cd_p3,  # Phase 3的CD优化器
                        opt_bqd,
                        device,
                        schedulers=schedulers_dict,
                        scalers=scalers_dict,
                    )

                    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    pretrained_weights_path_cd = r"e:\KingCrimson Dataset\Simulate\data0\results37\phase_2\model_cd_step63000"
                    if os.path.exists(pretrained_weights_path_cd):
                        try:
                            model_cd.load_state_dict(
                                torch.load(
                                    pretrained_weights_path_cd,
                                    map_location=device,
                                    weights_only=True,
                                )
                            )
                            print(
                                f"Phase3: Loaded pre-trained weights from {pretrained_weights_path_cd}"
                            )
                        except Exception as e:
                            print(f"加载CD预训练权重失败: {e}")

                    step_phase1 = steps_1
                    step_phase2 = steps_2
                    step_phase3 = step
                    best_val_loss_joint = best_val_loss

                    # 恢复稳定性追踪器
                    if loaded_stability_tracker:
                        stability_tracker = loaded_stability_tracker
                        print(
                            f"恢复稳定性追踪器: 真实准确率EMA={stability_tracker.real_acc_ema:.3f}, "
                            f"假样本准确率EMA={stability_tracker.fake_acc_ema:.3f}"
                        )
                    else:
                        # 如果没有保存的追踪器，创建新的
                        stability_tracker = StabilityTracker()
                        print("创建新的稳定性追踪器")

                else:
                    raise ValueError(f"未知的训练阶段: {phase}")

                # 应用学习率（如果检查点中保存了的话）
                if cn_lr is not None and phase in [1, 3]:
                    if phase == 1:
                        opt_cn_p1.param_groups[0]["lr"] = cn_lr
                        print(f"恢复Phase 1 CN学习率: {cn_lr}")
                    elif phase == 3:
                        opt_cn_p3.param_groups[0]["lr"] = cn_lr
                        print(f"恢复Phase 3 CN学习率: {cn_lr}")

                if cd_lr is not None and phase in [2, 3]:
                    if phase == 2:
                        opt_cd_p2.param_groups[0]["lr"] = cd_lr
                        print(f"恢复Phase 2 CD学习率: {cd_lr}")
                    elif phase == 3:
                        opt_cd_p3.param_groups[0]["lr"] = cd_lr
                        print(f"恢复Phase 3 CD学习率: {cd_lr}")

                if bqd_lr is not None:
                    opt_bqd.param_groups[0]["lr"] = bqd_lr
                    print(f"恢复BQD学习率: {bqd_lr}")

                print(f"✅ 成功恢复训练状态:")
                print(f"   当前阶段: {phase}")
                print(f"   Phase 1 步骤: {step_phase1}")
                print(f"   Phase 2 步骤: {step_phase2}")
                print(f"   Phase 3 步骤: {step_phase3}")
                # 安全地显示最佳值
                if best_val_loss is not None and best_val_loss != float("inf"):
                    print(f"   最佳验证损失: {best_val_loss:.4f}")
                else:
                    print(f"   最佳验证损失: 未设置")

                if best_acc is not None and best_acc != float("-inf"):
                    print(f"   最佳准确率: {best_acc:.4f}")
                else:
                    print(f"   最佳准确率: 未设置")

                # 根据阶段显示特定信息
                if phase == 1:
                    print("   -> 将继续Phase 1训练（生成器+BQD预训练）")
                elif phase == 2:
                    print("   -> 将继续Phase 2训练（判别器训练）")
                elif phase == 3:
                    print("   -> 将继续Phase 3训练（联合对抗训练）")
                    if loaded_stability_tracker:
                        print(f"   -> 稳定性追踪器已恢复")

            except Exception as e:
                print(f"❌ 加载检查点失败: {e}")
                print("将从头开始训练...")
                # 重置所有训练状态
                phase = Phase  # 使用命令行参数指定的阶段
                step_phase1 = 0
                step_phase2 = 0
                step_phase3 = 0
                best_val_loss = float("inf")
                best_bqd_loss = float("inf")
                best_acc = float("-inf")
                best_val_loss_joint = float("inf")
                cn_lr = None
                cd_lr = None
                bqd_lr = None
                cleanup_memory()
                raise
        else:
            step = 0
            if phase == 1:
                """pretrained_weights_path = r"E:\KingCrimson Dataset\Simulate\data0\results18\phase_1\model_cn_step26500"
                if os.path.exists(pretrained_weights_path):
                    try:
                        model_cn.load_state_dict(torch.load(pretrained_weights_path, map_location=device, weights_only=True))
                        print(f"Phase1: Loaded pre-trained weights from {pretrained_weights_path}")
                    except Exception as e:
                        print(f"加载预训练权重失败: {e}")"""

            if phase == 2:
                # Load pre-trained weights if available
                pretrained_weights_path = r"e:\KingCrimson Dataset\Simulate\data0\results46\phase_1\model_cn_step50000"
                if os.path.exists(pretrained_weights_path):
                    try:
                        model_cn.load_state_dict(
                            torch.load(
                                pretrained_weights_path,
                                map_location=device,
                                weights_only=True,
                            )
                        )
                        print(
                            f"Phase2: Loaded pre-trained weights from {pretrained_weights_path}"
                        )
                    except Exception as e:
                        print(f"加载预训练权重失败: {e}")
                pretrained_weights_path_bqd = r"e:\KingCrimson Dataset\Simulate\data0\results46\bqd\model_bqd_best"
                if os.path.exists(pretrained_weights_path_bqd):
                    try:
                        model_bqd.load_state_dict(
                            torch.load(
                                pretrained_weights_path_bqd,
                                map_location=device,
                                weights_only=True,
                            )
                        )
                        print(
                            f"Phase3: Loaded pre-trained weights from {pretrained_weights_path_bqd}"
                        )
                    except Exception as e:
                        print(f"加载CD预训练权重失败: {e}")
            elif phase == 3:
                # Load pre-trained weights if available
                pretrained_weights_path_cn = r"e:\KingCrimson Dataset\Simulate\data0\results46\phase_1\model_cn_step50000"
                if os.path.exists(pretrained_weights_path_cn):
                    try:
                        model_cn.load_state_dict(
                            torch.load(
                                pretrained_weights_path_cn,
                                map_location=device,
                                weights_only=True,
                            )
                        )
                        print(
                            f"Phase3: Loaded pre-trained weights from {pretrained_weights_path_cn}"
                        )
                    except Exception as e:
                        print(f"加载CN预训练权重失败: {e}")

                pretrained_weights_path_cd = r"e:\KingCrimson Dataset\Simulate\data0\results46\phase_2\model_cd_step50000"
                if os.path.exists(pretrained_weights_path_cd):
                    try:
                        model_cd.load_state_dict(
                            torch.load(
                                pretrained_weights_path_cd,
                                map_location=device,
                                weights_only=True,
                            )
                        )
                        print(
                            f"Phase3: Loaded pre-trained weights from {pretrained_weights_path_cd}"
                        )
                    except Exception as e:
                        print(f"加载CD预训练权重失败: {e}")

                pretrained_weights_path_bqd = r"e:\KingCrimson Dataset\Simulate\data0\results46\bqd\model_bqd_best"
                if os.path.exists(pretrained_weights_path_bqd):
                    try:
                        model_bqd.load_state_dict(
                            torch.load(
                                pretrained_weights_path_bqd,
                                map_location=device,
                                weights_only=True,
                            )
                        )
                        print(
                            f"Phase3: Loaded pre-trained weights from {pretrained_weights_path_bqd}"
                        )
                    except Exception as e:
                        print(f"加载CD预训练权重失败: {e}")

        # Create windows for the different phases and metrics
        loss_windows = {}
        if viz is not None:
            try:
                loss_windows = {
                    "phase1_train": viz.line(
                        Y=torch.zeros((1, 3)),
                        X=torch.tensor([step]),
                        opts=dict(
                            title="Phase 1 Training Loss",
                            legend=["Generator Loss", "Din Loss", "Dout Loss"],
                            linecolor=np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]]),
                        ),
                    ),
                    "phase1_val": viz.line(
                        Y=torch.zeros((1, 2)),
                        X=torch.tensor([step]),
                        opts=dict(
                            title="Phase 1 Validation Loss",
                            legend=["G loss", "D Loss"],
                            linecolor=np.array([[255, 0, 0], [0, 255, 0]]),
                        ),
                    ),
                    "phase2_disc": viz.line(
                        Y=torch.zeros(1),
                        X=torch.tensor([step]),
                        opts=dict(title="Phase 2 Discriminator Loss"),
                    ),
                    "phase2_acc": viz.line(
                        Y=torch.zeros((1, 3)),
                        X=torch.tensor([step]),
                        opts=dict(
                            title="Phase 2 Discriminator Accuracy",
                            legend=["avg acc", "fake acc", "real acc"],
                            linecolor=np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]]),
                        ),
                    ),
                    "phase3_gen": viz.line(
                        Y=torch.zeros((1, 3)),
                        X=torch.tensor([step]),
                        opts=dict(
                            title="Phase 3 Generator Loss",
                            legend=[
                                "gen loss",
                                "bqd loss",
                                "boundary loss",
                            ],  # 移到opts内部
                            linecolor=np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]]),
                        ),
                    ),
                    "phase3_disc": viz.line(
                        Y=torch.zeros(1),
                        X=torch.tensor([step]),
                        opts=dict(title="Phase 3 Discriminator Loss"),
                    ),
                    "phase3_val": viz.line(
                        Y=torch.zeros((1, 6)),
                        X=torch.tensor([step]),
                        opts=dict(
                            title="Phase 3 Validation Loss",
                            legend=[
                                "Recon Loss",
                                "Adv Loss",
                                "Real Acc",
                                "Fake Acc",
                                "Disc Acc",
                                "BQD Loss",
                            ],
                            linecolor=np.array(
                                [
                                    [255, 0, 0],
                                    [0, 255, 0],
                                    [0, 0, 255],
                                    [255, 255, 0],
                                    [0, 255, 255],
                                    [177, 177, 0],
                                ]
                            ),
                        ),
                    ),
                    "phase3_fake_acc": viz.line(
                        Y=torch.zeros((1, 2)),
                        X=torch.tensor([step]),
                        opts=dict(
                            title="Phase 3 Discriminator Accuracy",
                            legend=["Fake Acc", "Real Acc"],
                            linecolor=np.array([[255, 0, 0], [0, 0, 255]]),
                        ),
                    ),
                }
            except Exception as e:
                print(f"创建Visdom窗口失败: {e}")
                traceback.print_exc()  # 打印详细的错误堆栈信息
                loss_windows = {}

        # 在批次大小配置部分
        if use_multi_gpu:
            # 每个GPU的有效批次大小
            per_gpu_batch_size = batch_size
            # 总的有效批次大小
            effective_batch_size = batch_size * torch.cuda.device_count()
            print(
                f"多GPU训练: 每GPU批次={per_gpu_batch_size}, 总批次={effective_batch_size}"
            )
        else:
            effective_batch_size = batch_size
        target_batch_size = 32  # 目标有效batch size
        actual_batch_size = batch_size  # 当前的batch_size (8)
        if use_multi_gpu:
            # 多GPU时，实际的有效batch size是 actual_batch_size * num_gpus
            effective_batch_size = actual_batch_size * torch.cuda.device_count()
        else:
            effective_batch_size = actual_batch_size

        accumulation_steps = max(1, target_batch_size // effective_batch_size)
        print(
            f"配置梯度累积: {accumulation_steps} steps, 有效batch size: {target_batch_size}"
        )

        def visualize_results(
            local_targets,
            local_inputs,
            outputs,
            edge_inputs=None,
            edge_outputs=None,
            edge_targets=None,
            completed=None,
            step=0,
            phase="phase_1",
            num_test_completions=8,
        ):
            """
            在Visdom中可视化输入、目标、输出和完成的图像，使用高程图着色方案。
            增加edge_inputs和edge_outputs的对比可视化。
            """
            if viz is None:
                return

            try:

                def apply_colormap(tensor, grayscale=False):
                    """将张量转换为带有高程图着色的PIL图像"""
                    # 确保输入是2D张量(单通道图像)或取第一个通道
                    if tensor.dim() > 2:
                        if tensor.size(0) == 1:
                            tensor = tensor.squeeze(0)
                        else:
                            tensor = tensor[0]

                    # 转换为numpy数组并归一化到0-1范围
                    np_array = tensor.detach().cpu().numpy()
                    min_val = np_array.min()
                    max_val = np_array.max()
                    norm_array = (np_array - min_val) / (max_val - min_val + 1e-8)

                    if grayscale:
                        # 直接使用归一化的数组作为灰度图
                        gray_array = (norm_array * 255).astype(np.uint8)
                        colored_image = Image.fromarray(gray_array, mode="L")
                    else:
                        # 应用matplotlib的jet色彩映射
                        colored_array = plt.cm.jet(norm_array)
                        # 转换为PIL图像 (colored_array是RGBA格式)
                        colored_array_rgb = (colored_array[:, :, :3] * 255).astype(
                            np.uint8
                        )
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
                        colored_img = apply_colormap(tensor_batch[j])
                        colored_patches.append(colored_img)

                # 创建网格布局
                cols = num_samples
                rows = 3
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
                        caption=f"{phase} local patch step {step}",
                        title=f"{phase} local patch",
                    ),
                    win=f"{phase}_patches",
                )

                # 处理completed图像
                if completed is not None:
                    completed_samples = min(num_samples, len(completed))
                    colored_completed = []

                    for i in range(completed_samples):
                        colored_img = apply_colormap(completed[i])
                        colored_completed.append(colored_img)

                    # 创建completed图像网格
                    cols_c = min(2, completed_samples)
                    rows_c = (completed_samples + cols_c - 1) // cols_c
                    completed_grid_width = (colored_completed[0].width + 5) * cols_c
                    completed_grid_height = (colored_completed[0].height + 5) * rows_c
                    completed_grid_img = Image.new(
                        "RGB", (completed_grid_width, completed_grid_height)
                    )

                    # 填充网格
                    for i in range(completed_samples):
                        x = (i % cols_c) * (colored_completed[i].width + 5)
                        y = (i // cols_c) * (colored_completed[i].height + 5)
                        completed_grid_img.paste(colored_completed[i], (x, y))

                    # 缩小图像为原来的一半
                    completed_grid_img = completed_grid_img.resize(
                        (completed_grid_img.width // 2, completed_grid_img.height // 2),
                        Image.Resampling.LANCZOS,
                    )

                    viz.image(
                        np.array(completed_grid_img).transpose(2, 0, 1),
                        opts=dict(
                            caption=f"{phase} global patch step {step}",
                            title=f"{phase} global patch",
                        ),
                        win=f"{phase}_completed",
                    )

                # ========== 增加edge_inputs、edge_outputs和edge_targets三行对比可视化（灰度图） ==========
                if edge_inputs is not None and edge_outputs is not None:
                    num_edge = min(num_samples, len(edge_inputs), len(edge_outputs))
                    # 检查是否有edge_targets
                    has_edge_targets = (
                        edge_targets is not None and len(edge_targets) >= num_edge
                    )

                    colored_edge_in = []
                    colored_edge_out = []
                    colored_edge_target = []

                    # 生成灰度图像
                    for i in range(num_edge):
                        colored_edge_in.append(
                            apply_colormap(edge_inputs[i], grayscale=True)
                        )
                        colored_edge_out.append(
                            apply_colormap(edge_outputs[i], grayscale=True)
                        )
                        if has_edge_targets:
                            colored_edge_target.append(
                                apply_colormap(edge_targets[i], grayscale=True)
                            )

                    # 创建3行num_edge列的网格，第一行为in，第二行为out，第三行为target
                    edge_width = colored_edge_in[0].width
                    edge_height = colored_edge_in[0].height
                    gap_size = 5
                    gray_value = 128  # 中等灰度值 (0-255范围)

                    edge_grid_width = (edge_width + gap_size) * num_edge
                    edge_grid_height = (edge_height + gap_size) * (
                        3 if has_edge_targets else 2
                    )

                    # 创建灰度图网格，使用灰色背景
                    edge_grid_img = Image.new(
                        "L", (edge_grid_width, edge_grid_height), color=gray_value
                    )

                    # 第一行: edge_inputs
                    for i in range(num_edge):
                        x = i * (edge_width + gap_size)
                        y = 0
                        edge_grid_img.paste(colored_edge_target[i], (x, y))

                    # 第二行: edge_outputs
                    for i in range(num_edge):
                        x = i * (edge_width + gap_size)
                        y = edge_height + gap_size
                        edge_grid_img.paste(colored_edge_in[i], (x, y))

                    # 第三行: edge_targets (如果存在)
                    if has_edge_targets:
                        for i in range(num_edge):
                            x = i * (edge_width + gap_size)
                            y = (edge_height + gap_size) * 2
                            edge_grid_img.paste(colored_edge_out[i], (x, y))

                    # 放大和local_inputs一样
                    edge_grid_img = edge_grid_img.resize(
                        (edge_grid_img.width * 3, edge_grid_img.height * 3),
                        Image.NEAREST,
                    )

                    # 转换为numpy数组时需要注意灰度图的处理
                    edge_array = np.array(edge_grid_img)
                    # 灰度图需要添加通道维度给Visdom
                    if edge_array.ndim == 2:
                        edge_array = edge_array[np.newaxis, :, :]  # 添加通道维度

                    # 构建caption，根据是否有targets调整
                    caption_text = f"{phase} edge extraction step {step}"
                    if has_edge_targets:
                        caption_text += " (Input/Output/Target)"
                    else:
                        caption_text += " (Input/Output)"

                    viz.image(
                        edge_array,
                        opts=dict(
                            caption=caption_text,
                            title=f"{phase} edge extraction",
                        ),
                        win=f"{phase}_edge_compare",
                    )

                    # 清理edge相关变量
                    del colored_edge_in, colored_edge_out
                    if has_edge_targets:
                        del colored_edge_target
                    del edge_grid_img

                # 清理变量
                del colored_patches, grid_img
                if "colored_completed" in locals():
                    del colored_completed, completed_grid_img
                cleanup_memory()

            except Exception as e:
                print(f"可视化过程中发生错误: {e}")
                traceback.print_exc()  # 打印详细的错误堆栈信息
                cleanup_memory()

        # Helper function to evaluate model on validation set
        def validate(model_cn, model_bqd, val_subset_loader, device):
            """优化的验证函数，增加错误处理和内存管理"""
            try:
                model_cn.eval()
                val_loss = 0.0
                val_bqd_loss = 0.0
                num_batches = 0

                with torch.no_grad():
                    for batch in val_subset_loader:
                        try:
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

                            # Forward pass with memory management
                            outputs = safe_tensor_operation(
                                model_cn,
                                batch_local_inputs,
                                batch_local_masks,
                                batch_global_inputs,
                                batch_global_masks,
                            )

                            completed, completed_mask, pos = merge_local_to_global(
                                outputs,
                                batch_global_inputs,
                                batch_global_masks,
                                metadata,
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

                            nld_li = normalization(
                                batch_local_inputs, batch_global_inputs
                            )
                            nld_lo = normalization(outputs, batch_global_inputs)
                            nld_lt = normalization(
                                batch_local_targets, batch_global_inputs
                            )

                            output1, loss1, _ = model_bqd(nld_li, batch_local_masks)
                            output2, loss2, loss2_mask = model_bqd(
                                nld_lo, batch_local_masks
                            )
                            nld_lt = nld_lt.unsqueeze(1)
                            # print(f"local size: {nld_li.size()}, mask size: {torch.zeros_like(outputs).size()}")
                            # print(f"nld_lt size: {nld_lt.size()}, nld_lo size: {nld_lo.size()}")
                            output3, _, _ = model_bqd(nld_lt, torch.zeros_like(nld_lt))
                            loss_bqd = (loss1 + loss2) / 2
                            val_bqd_loss += loss_bqd.item()
                            val_loss += loss2_mask.item()

                            num_batches += 1

                            # 清理临时变量
                            del (
                                # outputs,
                                # completed,
                                completed_mask,
                                pos,
                                loss,
                                # output1,
                                # output2,
                                loss1,
                                loss2,
                                loss_bqd,
                            )
                            del batch_data

                        except Exception as e:
                            print(f"验证批次处理失败: {e}")
                            traceback.print_exc()  # 打印详细的错误堆栈信息
                            cleanup_memory()
                            continue

                # Calculate average validation loss
                avg_val_loss = val_loss / max(num_batches, 1)
                avg_bqd_val_loss = val_bqd_loss / max(num_batches, 1)
                model_cn.train()
                model_bqd.train()

                return (
                    avg_val_loss,
                    avg_bqd_val_loss,
                    batch_local_targets,
                    batch_local_inputs,
                    outputs,
                    output1,
                    output2,
                    output3,
                    completed,
                )

            except Exception as e:
                print(f"验证过程中发生错误: {e}")
                cleanup_memory()
                model_cn.train()
                return float("inf")

        def validateAll(model_cn, model_cd, model_bqd, val_subset_loader, device):
            """全面验证生成器和判别器性能，优化内存管理"""
            try:
                model_cn.eval()
                model_cd.eval()
                model_bqd.eval()

                # 初始化各种损失和指标
                recon_loss_sum = 0.0
                adv_loss_sum = 0.0
                bqd_loss_sum = 0.0
                real_acc_sum = 0.0
                fake_acc_sum = 0.0
                total_batches = 0

                with torch.no_grad():
                    for batch in val_subset_loader:
                        try:
                            total_batches += 1

                            # 准备批次数据
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

                            # 1. 评估生成器重建性能
                            outputs = safe_tensor_operation(
                                model_cn,
                                batch_local_inputs,
                                batch_local_masks,
                                batch_global_inputs,
                                batch_global_masks,
                            )

                            # 计算全局完成结果
                            completed, completed_mask, pos = merge_local_to_global(
                                outputs,
                                batch_global_inputs,
                                batch_global_masks,
                                metadata,
                            )

                            # 计算重建损失
                            recon_loss = completion_network_loss(
                                outputs,
                                batch_local_targets,
                                batch_local_masks,
                                batch_global_targets,
                                completed,
                                completed_mask,
                                pos,
                            )
                            recon_loss_sum += recon_loss.item()

                            # 2. 评估对抗性性能
                            batch_size = batch_local_targets.size(0)
                            real_labels = torch.ones(batch_size, 1, device=device)
                            fake_labels = torch.zeros(batch_size, 1, device=device)

                            nld_o = normalization(outputs, batch_global_inputs)
                            _, _, mask_loss = model_bqd(nld_o, batch_local_masks)
                            mask_loss = ensure_scalar_loss(mask_loss)
                            bqd_loss_sum += mask_loss.item()
                            # recon_loss_sum += mask_loss.item()

                            # 将生成器输出传递给判别器
                            fake_global_embedded, fake_global_mask_embedded, _ = (
                                merge_local_to_global(
                                    outputs,
                                    batch_global_inputs,
                                    batch_global_masks,
                                    metadata,
                                )
                            )

                            fake_predictions, _, _ = safe_tensor_operation(
                                model_cd,
                                outputs,
                                batch_local_masks,
                                fake_global_embedded,
                                fake_global_mask_embedded,
                            )

                            # 将真实样本传递给判别器
                            real_global_embedded, real_global_mask_embedded, _ = (
                                merge_local_to_global(
                                    batch_local_targets,
                                    batch_global_inputs,
                                    batch_global_masks,
                                    metadata,
                                )
                            )

                            real_predictions, _, _ = safe_tensor_operation(
                                model_cd,
                                batch_local_targets,
                                batch_local_masks,
                                real_global_embedded,
                                real_global_mask_embedded,
                            )

                            # 计算对抗性损失 - 生成器视角（希望判别器将生成样本识别为真）
                            adv_loss = F.binary_cross_entropy_with_logits(
                                fake_predictions, real_labels
                            )
                            adv_loss_sum += adv_loss.item()

                            # 计算判别器准确率
                            real_acc = (
                                (torch.sigmoid(real_predictions) >= 0.5)
                                .float()
                                .mean()
                                .item()
                            )
                            fake_acc = (
                                (torch.sigmoid(fake_predictions) < 0.5)
                                .float()
                                .mean()
                                .item()
                            )

                            real_acc_sum += real_acc
                            fake_acc_sum += fake_acc

                            # 清理临时变量
                            del outputs, completed, completed_mask, pos, recon_loss
                            del fake_global_embedded, fake_global_mask_embedded
                            del real_global_embedded, real_global_mask_embedded
                            del fake_predictions, real_predictions, adv_loss
                            del batch_data

                        except Exception as e:
                            print(f"验证批次处理失败: {e}")
                            cleanup_memory()
                            continue

                # 计算平均值
                avg_recon_loss = (
                    recon_loss_sum + 0.1 * adv_loss_sum + bqd_loss_sum * 0.1
                ) / max(total_batches, 1)
                avg_adv_loss = adv_loss_sum / max(total_batches, 1)
                avg_real_acc = real_acc_sum / max(total_batches, 1)
                avg_fake_acc = fake_acc_sum / max(total_batches, 1)
                avg_disc_acc = (avg_real_acc + avg_fake_acc) / 2
                bqd_loss = bqd_loss_sum / max(total_batches, 1)

                # 将模型设回训练模式
                model_cn.train()
                model_cd.train()
                model_bqd.train()

                return {
                    "recon_loss": avg_recon_loss,
                    "adv_loss": avg_adv_loss,
                    "real_acc": avg_real_acc,
                    "fake_acc": avg_fake_acc,
                    "disc_acc": avg_disc_acc,
                    "bqd_loss": bqd_loss,
                }

            except Exception as e:
                print(f"全面验证过程中发生错误: {e}")
                cleanup_memory()
                model_cn.train()
                model_cd.train()
                model_bqd.train()
                return {
                    "recon_loss": float("inf"),
                    "adv_loss": float("inf"),
                    "real_acc": 0.0,
                    "fake_acc": 0.0,
                    "disc_acc": 0.0,
                    "bqd_loss": 0.0,
                }

        def merge_local_to_global(local_data, global_data, global_mask, metadata):
            """
            Merge local patch data into global data based on metadata.

            metadata格式: [center_y, center_x, local_size, sample_idx, coverage_ratio, missing_ratio, local_size]
            其中center_y, center_x是local中心在global中的位置
            """
            try:
                local_data = local_data.unsqueeze(1)  # 确保local_data是4D张量
                merged_data = global_data.clone()
                merged_mask = global_mask.clone()
                pos = []

                for i in range(len(metadata)):
                    # 提取metadata信息
                    center_y = int(metadata[i][0])  # local中心在global中的y坐标
                    center_x = int(metadata[i][1])  # local中心在global中的x坐标
                    local_size = int(metadata[i][2])  # local尺寸

                    # 计算local区域在global中的边界
                    half_size = local_size // 2
                    min_y = center_y - half_size
                    max_y = center_y + half_size + 1  # +1因为Python切片是左闭右开
                    min_x = center_x - half_size
                    max_x = center_x + half_size + 1

                    # 边界检查
                    if (
                        min_y < 0
                        or min_x < 0
                        or max_y > global_data.size(2)
                        or max_x > global_data.size(3)
                    ):
                        print(
                            f"警告: 样本{i}位置越界 - center:({center_y},{center_x}), size:{local_size}"
                        )
                        print(f"       范围: y[{min_y},{max_y}), x[{min_x},{max_x})")
                        print(f"       global size: {global_data.shape}")
                        continue

                    # 嵌入local数据到global中
                    merged_data[i, :, min_y:max_y, min_x:max_x] = local_data[i, :, :, :]
                    merged_mask[i, :, min_y:max_y, min_x:max_x] = 1

                    pos.append([min_y, max_y, min_x, max_x])

                return merged_data, merged_mask, pos

            except Exception as e:
                print(f"合并错误: {e}")
                print(f"local_data shape: {local_data.shape}")
                print(f"global_data shape: {global_data.shape}")
                for i, meta in enumerate(metadata):
                    print(
                        f"metadata[{i}]: center=({meta[0]},{meta[1]}), size={meta[2]}"
                    )
                raise

        def normalization(inputs, global_inputs):
            # print(f"normalization inputs shape: {inputs.shape}, global_inputs shape: {global_inputs.shape}")
            gmin = global_inputs.min()
            gmax = global_inputs.max()

            # 避免除零错误
            if gmax - gmin > 1e-8:
                # 归一化到[0, 1]
                inputs = (inputs - gmin) / (gmax - gmin)
            return inputs

        # =================================================
        # Training Phase 1: Train Completion Network only
        # =================================================

        if step_phase1 < steps_1 and phase == 1:
            print("开始/继续第1阶段训练...")
            try:
                model_cn.train()
                model_bqd.train()
                pbar = tqdm(total=steps_1, initial=step_phase1)
                step = step_phase1
                scaler = GradScaler()
                scaler_bqd = GradScaler()
                opt_cn_p1 = Adadelta(model_cn.parameters())

                if bqd_lr is not None:
                    opt_bqd.param_groups[0]["lr"] = bqd_lr

                # 梯度累积计数器
                accumulated_step = 0

                while step < steps_1:
                    for batch in train_loader:
                        try:
                            # 定期检查内存
                            if step % 100 == 0:
                                check_memory_and_cleanup(threshold_gb=6.0)

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

                            # ==================== 修复：正确实现梯度累积 ====================

                            #### 第1次backward - BQD网络学习识别生成输出边缘 ####
                            for param in model_cn.parameters():
                                param.requires_grad = False
                            for param in model_bqd.parameters():
                                param.requires_grad = True

                            with torch.no_grad():
                                outputs = safe_tensor_operation(
                                    model_cn,
                                    batch_local_inputs,
                                    batch_local_masks,
                                    batch_global_inputs,
                                    batch_global_masks,
                                )
                            nld_o = normalization(outputs, batch_global_inputs)
                            bqd_outputs_out, bqd_loss_out, _ = model_bqd(
                                nld_o.detach(),
                                batch_local_masks,
                            )
                            bqd_loss_out = ensure_scalar_loss(bqd_loss_out)
                            bqd_loss_out_scaled = bqd_loss_out / accumulation_steps
                            scaler_bqd.scale(bqd_loss_out_scaled).backward(
                                retain_graph=True
                            )
                            # bqd_loss_out_scaled.backward(retain_graph=True)

                            #### 第2次backward - BQD网络学习识别原始输入边缘 ####\
                            nld_li = normalization(
                                batch_local_inputs, batch_global_inputs
                            )
                            bqd_outputs_in, bqd_loss_in, _ = model_bqd(
                                nld_li, batch_local_masks
                            )
                            bqd_loss_in = ensure_scalar_loss(bqd_loss_in)
                            bqd_loss_in_scaled = bqd_loss_in / accumulation_steps
                            scaler_bqd.scale(bqd_loss_in_scaled).backward(
                                retain_graph=True
                            )
                            # bqd_loss_in_scaled.backward(retain_graph=True)

                            #### 第3次backward - 补全网络的重建+对抗损失 ####
                            for param in model_bqd.parameters():
                                param.requires_grad = False
                            for param in model_cn.parameters():
                                param.requires_grad = True

                            # Forward pass for completion network
                            outputs = safe_tensor_operation(
                                model_cn,
                                batch_local_inputs,
                                batch_local_masks,
                                batch_global_inputs,
                                batch_global_masks,
                            )

                            completed, completed_mask, pos = merge_local_to_global(
                                outputs,
                                batch_global_inputs,
                                batch_global_masks,
                                metadata,
                            )

                            # 重建损失
                            recon_loss = completion_network_loss(
                                outputs,
                                batch_local_targets,
                                batch_local_masks,
                                batch_global_targets,
                                completed,
                                completed_mask,
                                pos,
                            )

                            # BQD对抗损失（让生成结果欺骗BQD）
                            nld_lo = normalization(outputs, batch_global_inputs)
                            _, _, bqd_loss_mask_adversarial = model_bqd(
                                nld_lo, batch_local_masks
                            )
                            bqd_loss_mask_adversarial = ensure_scalar_loss(
                                bqd_loss_mask_adversarial
                            )

                            # 组合损失并应用梯度累积缩放
                            total_cn_loss = (
                                recon_loss + bqd_loss_mask_adversarial
                            ) / accumulation_steps
                            # scaler.scale(total_cn_loss).backward()
                            total_cn_loss.backward()

                            accumulated_step += 1

                            # ==================== 修复：正确的梯度累积更新 ====================
                            if accumulated_step % accumulation_steps == 0:
                                # 更新补全网络
                                # scaler.unscale_(opt_cn_p1)
                                torch.nn.utils.clip_grad_norm_(
                                    model_cn.parameters(), max_norm=1.0
                                )
                                # scaler.step(opt_cn_p1)
                                opt_cn_p1.step()
                                # scaler.update()
                                opt_cn_p1.zero_grad(set_to_none=True)
                            if accumulated_step % (accumulation_steps * 2) == 0:
                                # 更新BQD网络
                                scaler_bqd.unscale_(opt_bqd)
                                torch.nn.utils.clip_grad_norm_(
                                    model_bqd.parameters(), max_norm=1.0
                                )
                                scaler_bqd.step(opt_bqd)
                                # opt_bqd.step()
                                scaler_bqd.update()
                                opt_bqd.zero_grad(set_to_none=True)

                            # Update Visdom plot for training loss
                            if (
                                viz is not None
                                and "phase1_train" in loss_windows
                                and step % 100 == 0
                            ):
                                try:
                                    y0 = (
                                        recon_loss.item()
                                        if recon_loss.item() < 1000
                                        else -1
                                    )
                                    y = torch.tensor(
                                        [y0, bqd_loss_in.item(), bqd_loss_out.item()]
                                    ).float()
                                    viz.line(
                                        Y=y.unsqueeze(0),  # shape (1, 3)
                                        X=torch.tensor([step]).float(),
                                        win=loss_windows["phase1_train"],
                                        update="append",
                                    )
                                except Exception as e:
                                    print(f"Visdom phase1_train plot error: {e}")
                                    pass

                            # 在每个snaperiod_1步保存检查点
                            if step % snaperiod_1 == 0:
                                try:
                                    # 准备Phase 1的状态信息
                                    phase1_schedulers = None  # Phase 1通常不使用调度器
                                    phase1_scalers = {
                                        "cn": scaler,  # Phase 1的CN缩放器
                                        "bqd": scaler_bqd,  # Phase 1的BQD缩放器
                                    }

                                    """save_checkpoint(
                                        model_cn,
                                        None,  # Phase 1不保存CD模型
                                        model_bqd,
                                        opt_cn_p1,  # Phase 1的CN优化器
                                        None,  # Phase 1不使用CD优化器
                                        opt_bqd,
                                        step,
                                        1,  # phase = 1
                                        result_dir,
                                        best_val_loss=best_val_loss,
                                        best_acc=None,  # Phase 1不追踪准确率
                                        cn_lr=opt_cn_p1.param_groups[0]["lr"],
                                        cd_lr=None,  # Phase 1不使用CD
                                        bqd_lr=opt_bqd.param_groups[0]["lr"],
                                        schedulers=phase1_schedulers,
                                        scalers=phase1_scalers,
                                        stability_tracker=None,  # Phase 1不使用稳定性追踪器
                                    )"""

                                    print(f"✅ Phase 1 检查点已保存 (step {step})")

                                except Exception as e:
                                    print(f"❌ Phase 1 保存检查点失败: {e}")

                            step += 1
                            pbar.set_description(
                                f"Phase 1 | train loss: {recon_loss.item():.5f}, adv loss: {bqd_loss_mask_adversarial.item():.5f}"
                            )
                            pbar.update(1)

                            # 清理临时变量
                            del (
                                outputs,
                                completed,
                                completed_mask,
                                pos,
                                recon_loss,
                                bqd_outputs_in,
                                bqd_loss_in,
                                bqd_outputs_out,
                                bqd_loss_out,
                            )
                            del batch_data

                            # Testing and saving snapshots
                            if step % snaperiod_1 == 0:
                                try:

                                    (
                                        val_loss,
                                        val_bqd_loss,
                                        val_local_targets,
                                        val_local_inputs,
                                        test_output,
                                        val_in,
                                        val_out,
                                        val_target,
                                        completed,
                                    ) = validate(
                                        model_cn, model_bqd, val_subset_loader, device
                                    )
                                    print(
                                        f"Step {step}, Validation Loss: {val_loss:.5f}"
                                    )

                                    # Update Visdom plot for validation loss
                                    if viz is not None and "phase1_val" in loss_windows:
                                        try:
                                            viz.line(
                                                Y=torch.tensor(
                                                    [[val_loss, val_bqd_loss]]
                                                ),
                                                X=torch.tensor([step]),
                                                win=loss_windows["phase1_val"],
                                                update="append",
                                            )
                                        except:
                                            pass

                                    # Save if best model
                                    if val_loss < best_val_loss:
                                        best_val_loss = val_loss
                                        best_model_path = os.path.join(
                                            result_dir, "phase_1", "model_cn_best"
                                        )
                                        torch.save(
                                            model_cn.state_dict(), best_model_path
                                        )

                                    if val_bqd_loss < best_bqd_loss:
                                        best_bqd_loss = val_bqd_loss
                                        best_bqd_model_path = os.path.join(
                                            result_dir, "bqd", "model_bqd_best"
                                        )
                                        torch.save(
                                            model_bqd.state_dict(), best_bqd_model_path
                                        )

                                    """# Generate some test completions
                                    model_cn.eval()
                                    model_bqd.eval()
                                    with torch.no_grad():
                                        try:
                                            # Get a batch from validation set
                                            val_batch = next(iter(val_subset_loader))

                                            # Prepare validation batch
                                            val_data = prepare_batch_data(
                                                val_batch, device
                                            )
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
                                            test_output = safe_tensor_operation(
                                                model_cn,
                                                val_local_inputs,
                                                val_local_masks,
                                                val_global_inputs,
                                                val_global_masks,
                                            )

                                            # Merge local output to global for visualization
                                            completed, completed_mask, _ = (
                                                merge_local_to_global(
                                                    test_output,
                                                    val_global_inputs,
                                                    val_global_masks,
                                                    metadata,
                                                )
                                            )

                                            val_in, loss1, _ = model_bqd(
                                                val_local_inputs, val_local_masks
                                            )
                                            val_out, loss2, loss2_mask = model_bqd(
                                                test_output, val_local_masks
                                            )"""
                                    with torch.no_grad():
                                        try:
                                            # Visualize results in Visdom
                                            visualize_results(
                                                val_local_targets,
                                                val_local_inputs,
                                                test_output,
                                                val_in,
                                                val_out,
                                                val_target,
                                                completed,
                                                step,
                                                "phase_1",
                                            )

                                            # 保存模型
                                            folder = os.path.join(result_dir, "phase_1")

                                            # 找到所有匹配的检查点文件（不要求.pth扩展名）
                                            pattern = os.path.join(
                                                folder, "model_cn_step*"
                                            )
                                            old_files = glob.glob(pattern)

                                            # 删除step小于当前step-1000的文件
                                            for file in old_files:
                                                try:
                                                    # 从文件名中提取step数字
                                                    basename = os.path.basename(file)
                                                    # 修改正则表达式，使.pth扩展名可选
                                                    match = re.search(
                                                        r"model_cn_step(\d+)(?:\.pth)?$",
                                                        basename,
                                                    )
                                                    if match:
                                                        file_step = int(match.group(1))
                                                        # 只删除step小于当前step-1000的文件
                                                        if file_step < step - 1000:
                                                            os.remove(file)
                                                            print(
                                                                f"删除旧文件: {basename} (step={file_step})"
                                                            )
                                                    else:
                                                        print(
                                                            f"无法解析文件名: {basename}"
                                                        )
                                                except Exception as e:
                                                    print(f"删除文件失败: {e}")

                                            bqd_folder = os.path.join(result_dir, "bqd")
                                            bqd_pattern = os.path.join(
                                                bqd_folder, "model_bqd_step*"
                                            )
                                            old_bqd_files = glob.glob(bqd_pattern)
                                            # 删除step小于当前step-1000的文件
                                            for file in old_bqd_files:
                                                try:
                                                    # 从文件名中提取step数字
                                                    basename = os.path.basename(file)
                                                    # 修改正则表达式，使.pth扩展名可选
                                                    match = re.search(
                                                        r"model_bqd_step(\d+)(?:\.pth)?$",
                                                        basename,
                                                    )
                                                    if match:
                                                        file_step = int(match.group(1))
                                                        # 只删除step小于当前step-1000的文件
                                                        if file_step < step - 1000:
                                                            os.remove(file)
                                                            print(
                                                                f"删除旧文件: {basename} (step={file_step})"
                                                            )
                                                    else:
                                                        print(
                                                            f"无法解析文件名: {basename}"
                                                        )
                                                except Exception as e:
                                                    print(f"删除文件失败: {e}")

                                            # 保存当前模型
                                            model_path = os.path.join(
                                                result_dir,
                                                "phase_1",
                                                f"model_cn_step{step}",
                                            )
                                            torch.save(
                                                model_cn.state_dict(), model_path
                                            )
                                            print(f"保存模型: model_cn_step{step}")

                                            # 保存当前BQD模型
                                            bqd_model_path = os.path.join(
                                                result_dir,
                                                "bqd",
                                                f"model_bqd_step{step}",
                                            )
                                            torch.save(
                                                model_bqd.state_dict(), bqd_model_path
                                            )
                                            print(f"保存模型: model_bqd_step{step}")

                                            # 清理变量
                                            del test_output, completed

                                            del (
                                                # val_batch,
                                                val_local_inputs,
                                                # val_local_masks,
                                                val_local_targets,
                                            )
                                            """del (
                                                val_global_inputs,
                                                val_global_masks,
                                                val_global_targets,
                                            )
"""
                                        except Exception as e:
                                            print(f"生成测试完成图像时失败: {e}")
                                            cleanup_memory()

                                    model_cn.train()
                                    model_bqd.train()

                                except Exception as e:
                                    print(f"验证步骤失败: {e}")
                                    cleanup_memory()

                            if step >= steps_1:
                                break

                        except Exception as e:
                            print(f"训练批次处理失败: {e}")
                            traceback.print_exc()  # 打印详细的错误堆栈信息
                            cleanup_memory()
                            continue

                pbar.close()
                phase = 2  # 切换到第二阶段

            except Exception as e:
                print(f"Phase 1训练过程中发生严重错误: {e}")
                cleanup_memory()
                raise

        # =================================================
        # Training Phase 2: Train Context Discriminator only
        # =================================================

        def balanced_discriminator_loss(
            real_preds, fake_preds, real_labels, fake_labels
        ):
            """
            平衡的判别器损失，特别处理不平衡情况
            """
            # 基础MSE损失
            real_loss = F.mse_loss(real_preds, real_labels)
            fake_loss = F.mse_loss(fake_preds, fake_labels)

            # 计算真假分类情况
            real_accuracy = (real_preds > 0.5).float().mean()
            fake_accuracy = (fake_preds < 0.5).float().mean()

            # 如果分类严重不平衡
            if fake_accuracy > 0.8 and real_accuracy < 0.2:
                # 大幅增加真实样本权重，减少假样本权重
                return real_loss * 5.0 + fake_loss * 0.1
            elif real_accuracy > 0.8 and fake_accuracy < 0.2:
                # 反之亦然
                return real_loss * 0.1 + fake_loss * 5.0
            else:
                # 相对平衡情况
                return (real_loss + fake_loss) / 2.0

        def weighted_bce_loss(predictions, targets, pos_weight=2.0, neg_weight=2.0):
            """改进的加权二元交叉熵损失，防止数值问题"""
            # 使用更安全的epsilon值
            eps = 1e-6

            # 使用更严格的clamp确保预测值在有效范围内
            predictions = torch.clamp(predictions, eps, 1.0 - eps)

            # 使用更稳定的形式计算损失
            pos_part = -targets * torch.log(predictions) * pos_weight
            neg_part = -(1.0 - targets) * torch.log(1.0 - predictions) * neg_weight

            # 检查无效值并替换
            pos_part = torch.where(
                torch.isnan(pos_part), torch.zeros_like(pos_part), pos_part
            )
            neg_part = torch.where(
                torch.isnan(neg_part), torch.zeros_like(neg_part), neg_part
            )

            # 计算均值前检查总损失
            loss = pos_part + neg_part
            loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

            return loss.mean()

        def feature_contrastive_loss(
            real_local_feat,
            real_global_feat,
            fake_local_feat,
            fake_global_feat,
        ):
            """特征对比损失 - 鼓励真假样本特征分离"""

            # 计算真假特征均值
            real_local_mean = real_local_feat.mean(0)
            fake_local_mean = fake_local_feat.mean(0)
            real_global_mean = real_global_feat.mean(0)
            fake_global_mean = fake_global_feat.mean(0)

            # 计算欧氏距离
            local_distance = torch.sqrt(
                torch.sum((real_local_mean - fake_local_mean) ** 2) + 1e-6
            )
            global_distance = torch.sqrt(
                torch.sum((real_global_mean - fake_global_mean) ** 2) + 1e-6
            )
            # boundary_distance = torch.sqrt(torch.sum((real_boudnary_mean - fake_boundary_mean) ** 2) + 1e-6)

            # 对比损失 - 我们希望最大化距离（所以使用负号）
            contrastive_loss = (
                1.0 / local_distance + 1.0 / global_distance
            )  #  + 1.0 / boundary_distance

            return contrastive_loss

        def enhanced_feature_contrastive_loss(
            real_local_feat,
            real_global_feat,
            fake_local_feat,
            fake_global_feat,
            alpha=1.0,
            beta=0.6,
            gamma=0.6,
        ):
            """
            增强版特征对比损失 - 基于你的原始设计，加入余弦分离和中心分离

            Args:
                real_local_feat: [B, D] 真实样本局部特征
                real_global_feat: [B, D] 真实样本全局特征
                fake_local_feat: [B, D] 生成样本局部特征
                fake_global_feat: [B, D] 生成样本全局特征
                alpha: 原始欧几里得距离权重
                beta: 余弦分离权重
                gamma: 中心分离权重

            Returns:
                total_loss: 组合损失
                loss_info: 各项损失的详细信息
            """

            # ========== 1. 原始的欧几里得距离分离 (你的原始设计) ==========
            # 计算真假特征均值
            real_local_mean = real_local_feat.mean(0)
            fake_local_mean = fake_local_feat.mean(0)
            real_global_mean = real_global_feat.mean(0)
            fake_global_mean = fake_global_feat.mean(0)

            # 计算欧氏距离
            local_distance = torch.sqrt(
                torch.sum((real_local_mean - fake_local_mean) ** 2) + 1e-6
            )
            global_distance = torch.sqrt(
                torch.sum((real_global_mean - fake_global_mean) ** 2) + 1e-6
            )

            # 原始对比损失 - 我们希望最大化距离（所以使用负号）
            euclidean_loss = 1.0 / (local_distance + 1e-6) + 1.0 / (
                global_distance + 1e-6
            )

            # ========== 2. 余弦分离损失 ==========
            # 归一化特征到单位球面
            real_local_norm = F.normalize(real_local_feat, p=2, dim=1)
            real_global_norm = F.normalize(real_global_feat, p=2, dim=1)
            fake_local_norm = F.normalize(fake_local_feat, p=2, dim=1)
            fake_global_norm = F.normalize(fake_global_feat, p=2, dim=1)

            # 计算真假样本特征的余弦相似度
            local_cosine_sim = F.cosine_similarity(
                real_local_norm, fake_local_norm, dim=1
            )
            global_cosine_sim = F.cosine_similarity(
                real_global_norm, fake_global_norm, dim=1
            )

            # 余弦分离损失 - 希望相似度接近0（正交）
            cosine_separation_loss = (
                local_cosine_sim.abs().mean() + global_cosine_sim.abs().mean()
            ) / 2.0

            # ========== 3. 中心分离损失 ==========
            # 计算归一化后的中心
            real_local_center = real_local_norm.mean(0, keepdim=True)  # [1, D]
            real_global_center = real_global_norm.mean(0, keepdim=True)  # [1, D]
            fake_local_center = fake_local_norm.mean(0, keepdim=True)  # [1, D]
            fake_global_center = fake_global_norm.mean(0, keepdim=True)  # [1, D]

            # 归一化中心向量
            real_local_center_norm = F.normalize(real_local_center, p=2, dim=1)
            real_global_center_norm = F.normalize(real_global_center, p=2, dim=1)
            fake_local_center_norm = F.normalize(fake_local_center, p=2, dim=1)
            fake_global_center_norm = F.normalize(fake_global_center, p=2, dim=1)

            # 计算中心间的余弦相似度
            local_center_sim = F.cosine_similarity(
                real_local_center_norm, fake_local_center_norm, dim=1
            )
            global_center_sim = F.cosine_similarity(
                real_global_center_norm, fake_global_center_norm, dim=1
            )

            # 中心分离损失 - 希望中心相似度尽可能小
            center_separation_loss = (
                local_center_sim.abs() + global_center_sim.abs()
            ) / 2.0

            # ========== 4. 组合损失 ==========
            total_loss = (
                alpha * euclidean_loss  # 原始欧几里得距离
                + beta * cosine_separation_loss  # 余弦分离
                + gamma * center_separation_loss  # 中心分离
            )
            # print(f"euclidean_loss: {euclidean_loss.item()}, cosine_separation: {cosine_separation_loss.item()}, center_separation: {center_separation_loss.item()}")

            # 返回详细信息用于调试
            loss_info = {
                "total_loss": total_loss.item(),
                "euclidean_loss": euclidean_loss.item(),
                "cosine_separation": cosine_separation_loss.item(),
                "center_separation": center_separation_loss.item(),
                "local_euclidean_distance": local_distance.item(),
                "global_euclidean_distance": global_distance.item(),
                "local_cosine_similarity": local_cosine_sim.mean().item(),
                "global_cosine_similarity": global_cosine_sim.mean().item(),
                "local_center_similarity": local_center_sim.item(),
                "global_center_similarity": global_center_sim.item(),
            }

            return total_loss  # , loss_info

        def gradient_penalty(
            discriminator,
            real_local_data,
            real_local_mask,
            real_global_data,
            real_global_mask,
            fake_local_data,
            fake_local_mask,
            fake_global_data,
            fake_global_mask,
        ):
            """内存效率更高的梯度惩罚计算"""
            try:
                batch_size = real_local_data.size(0)

                # 只使用较小的批量计算梯度惩罚
                max_samples = min(batch_size, 4)  # 一次最多处理4个样本

                # 随机选择样本
                indices = torch.randperm(batch_size)[:max_samples]

                real_local_sample = real_local_data[indices]
                real_local_mask_sample = real_local_mask[indices]
                real_global_sample = real_global_data[indices]
                real_global_mask_sample = real_global_mask[indices]

                fake_local_sample = fake_local_data[indices]
                fake_local_mask_sample = fake_local_mask[indices]
                fake_global_sample = fake_global_data[indices]
                fake_global_mask_sample = fake_global_mask[indices]

                # 以下计算与原始函数相同，但使用较小的样本
                alpha = torch.rand(max_samples, 1, 1, 1, device=real_local_data.device)

                interpolates_local = (
                    alpha * real_local_sample + (1 - alpha) * fake_local_sample
                )
                interpolates_global = (
                    alpha * real_global_sample + (1 - alpha) * fake_global_sample
                )

                interpolates_local_mask = (
                    alpha * real_local_mask_sample
                    + (1 - alpha) * fake_local_mask_sample
                ).round()
                interpolates_global_mask = (
                    alpha * real_global_mask_sample
                    + (1 - alpha) * fake_global_mask_sample
                ).round()

                interpolates_local.requires_grad_(True)
                interpolates_global.requires_grad_(True)

                d_interpolates, _, _ = discriminator(
                    interpolates_local,
                    interpolates_local_mask,
                    interpolates_global,
                    interpolates_global_mask,
                )

                fake = torch.ones(d_interpolates.size(), device=real_local_data.device)

                gradients = torch.autograd.grad(
                    outputs=d_interpolates,
                    inputs=[interpolates_local, interpolates_global],
                    grad_outputs=fake,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                )

                gradients_local = gradients[0].view(max_samples, -1)
                gradients_global = gradients[1].view(max_samples, -1)

                gradients_all = torch.cat([gradients_local, gradients_global], dim=1)
                gradient_norm = gradients_all.norm(2, dim=1)

                gradient_penalty = ((gradient_norm - 1) ** 2).mean()

                # 清理临时变量
                del interpolates_local, interpolates_global, d_interpolates
                del gradients, gradients_local, gradients_global, gradients_all

                return gradient_penalty

            except Exception as e:
                print(f"计算梯度惩罚时发生错误: {e}")
                return torch.tensor(0.0, device=real_local_data.device)

        def log_scale_balance_loss(loss_real, loss_fake, lambda_balance=1.0):
            """
            对数比例平衡损失，专门处理数量级差异
            """
            # 添加小常数防止log(0)
            epsilon = 1e-8

            # 计算对数比例
            log_ratio = torch.log(loss_real + epsilon) - torch.log(loss_fake + epsilon)

            # 平衡损失是对数比例的平方，当比例为1(对数为0)时最小
            balance_loss = lambda_balance * log_ratio**2

            return balance_loss

        def r1_regularization(
            discriminator,
            real_data,
            real_mask,
            real_global,
            real_global_mask,
            weight=1e3,
        ):
            """R1正则化 - 在真实数据上的梯度惩罚"""
            try:
                # 需要梯度
                real_data.requires_grad_(True)

                # 前向传播
                real_pred, _, _ = discriminator(
                    real_data, real_mask, real_global, real_global_mask
                )

                # 对所有样本的预测求和
                pred_sum = real_pred.sum()

                # 计算梯度
                gradients = torch.autograd.grad(
                    outputs=pred_sum,
                    inputs=real_data,
                    create_graph=True,
                    retain_graph=True,
                )[0]

                # 梯度平方范数
                grad_norm_squared = (
                    gradients.pow(2).reshape(gradients.shape[0], -1).sum(1)
                )

                # R1正则化
                r1_penalty = weight * grad_norm_squared.mean() / 2

                # 清理梯度变量
                del gradients, grad_norm_squared

                return r1_penalty

            except Exception as e:
                print(f"计算R1正则化时发生错误: {e}")
                return torch.tensor(0.0, device=real_data.device)

        best_acc = float("-inf")

        if step_phase2 < steps_2 and phase == 2:
            print("开始/继续第2阶段训练...")
            try:
                model_cd.load_state_dict(
                    torch.load(
                        r"e:\KingCrimson Dataset\Simulate\data0\results46\phase_2\model_cd_step42500",
                        map_location=device,
                        weights_only=True,
                    )
                )
                step_phase2 = 42500
                # 使用Adam优化器
                opt_cd = torch.optim.Adam(
                    model_cd.parameters(), lr=1e-5, betas=(0.5, 0.999), eps=1e-8
                )

                # 创建梯度缩放器，用于自动混合精度训练
                scaler2 = GradScaler(enabled=True)

                # 创建学习率调度器 - 只使用一种调度器
                scheduler = CosineAnnealingLR(opt_cd_p2, T_max=steps_2, eta_min=1e-6)

                # 实例噪声初始值和最小值
                start_noise = 0.1
                min_noise = 0.001

                # 跟踪性能指标
                running_loss = 0.0
                running_real_acc = 0.0
                running_fake_acc = 0.0
                patience_counter = 0
                max_patience = 10  # 增加耐心值，避免过早降低学习率

                pbar = tqdm(total=steps_2, initial=step_phase2)
                step = step_phase2
                while step < steps_2:
                    for batch in train_loader:
                        try:
                            # 定期检查内存
                            if step % 50 == 0:
                                check_memory_and_cleanup(threshold_gb=8.0)

                            # 准备批次数据
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

                            # 使用自动混合精度
                            with autocast(
                                device_type=device.type,
                                dtype=torch.float32,
                                enabled=True,
                            ):
                                # nld_li, nldlt, nld_gi, nld_gt = nomarlize(batch_local_inputs, batch_local_targets, batch_global_inputs, batch_global_targets)
                                # 生成假样本
                                with torch.no_grad():
                                    fake_outputs = safe_tensor_operation(
                                        model_cn,
                                        batch_local_inputs,
                                        batch_local_masks,
                                        batch_global_inputs,
                                        batch_global_masks,
                                    )

                                # 计算当前噪声水平 - 随着训练进展减少
                                noise_level = max(
                                    min_noise, start_noise * (1.0 - step / steps_2)
                                )

                                # 应用实例噪声
                                # 限制噪声水平在0-1之间
                                noise_level = max(0.0, min(noise_level, 1.0))
                                batch_local_targets_noisy = (
                                    batch_local_targets
                                    + torch.randn_like(batch_local_targets)
                                    * noise_level
                                )
                                fake_outputs_noisy = (
                                    fake_outputs
                                    + torch.randn_like(fake_outputs) * noise_level
                                )

                                # 嵌入假输出到全局
                                fake_global_embedded, fake_global_mask_embedded, _ = (
                                    merge_local_to_global(
                                        fake_outputs_noisy,
                                        batch_global_inputs,
                                        batch_global_masks,
                                        metadata,
                                    )
                                )

                                # 嵌入真实输入到全局
                                real_global_embedded, real_global_mask_embedded, _ = (
                                    merge_local_to_global(
                                        batch_local_targets_noisy,
                                        batch_global_inputs,
                                        batch_global_masks,
                                        metadata,
                                    )
                                )

                                # 创建平滑标签
                                batch_size = batch_local_targets.size(0)
                                real_labels = (
                                    torch.ones(batch_size, 1).to(device) * 0.95
                                )  # 标签平滑
                                fake_labels = (
                                    torch.zeros(batch_size, 1).to(device) + 0.05
                                )  # 标签平滑

                                # 训练判别器处理假样本
                                fake_predictions, fake_lf, fake_gf = (
                                    safe_tensor_operation(
                                        model_cd,
                                        fake_outputs_noisy,
                                        batch_local_masks,
                                        fake_global_embedded,
                                        fake_global_mask_embedded,
                                    )
                                )

                                # 训练判别器处理真实样本
                                real_predictions, real_lf, real_gf = (
                                    safe_tensor_operation(
                                        model_cd,
                                        batch_local_targets_noisy,
                                        batch_local_masks,
                                        real_global_embedded,
                                        real_global_mask_embedded,
                                    )
                                )

                                # 使用BCEWithLogitsLoss代替BCE或自定义损失
                                loss_real = F.binary_cross_entropy_with_logits(
                                    real_predictions, real_labels
                                )
                                loss_fake = F.binary_cross_entropy_with_logits(
                                    fake_predictions, fake_labels
                                )

                                # 计算特征匹配损失
                                fm_weight = 4
                                fm_loss = (
                                    feature_contrastive_loss(
                                        real_lf,
                                        real_gf,
                                        fake_lf,
                                        fake_gf,
                                    )
                                    * fm_weight
                                )
                                if (
                                    fm_loss.mean() > loss_real.mean()
                                    and fm_loss.mean() > loss_fake.mean()
                                ):
                                    fm_loss *= 1
                                    # print("fm_loss weight reduced")
                                lsb_loss = log_scale_balance_loss(loss_real, loss_fake)
                                r1_loss = r1_regularization(
                                    model_cd,
                                    batch_local_targets_noisy,
                                    batch_local_masks,
                                    real_global_embedded,
                                    real_global_mask_embedded,
                                )

                                # 使用sigmoid计算预测概率，用于准确率计算
                                with torch.no_grad():
                                    real_probs = torch.sigmoid(real_predictions)
                                    fake_probs = torch.sigmoid(fake_predictions)

                                    real_acc = (real_probs >= 0.5).float().mean().item()
                                    fake_acc = (fake_probs < 0.5).float().mean().item()
                                    avg_acc = (real_acc + fake_acc) / 2

                                # 动态调整损失权重，平衡训练
                                if real_acc < 0.1 and fake_acc > 0.9:
                                    loss_real *= 5.0
                                    loss_fake *= 0.5
                                    print("loss real weight strengthened")
                                elif fake_acc < 0.1 and real_acc > 0.9:
                                    loss_real *= 0.5
                                    loss_fake *= 5.0
                                    print("loss fake weight strengthened")
                                loss = (
                                    loss_real + loss_fake + fm_loss + lsb_loss + r1_loss
                                )

                            # 使用梯度缩放器进行反向传播
                            scaler2.scale(loss).backward()
                            # loss.backward()

                            # 梯度裁剪
                            scaler2.unscale_(opt_cd_p2)
                            torch.nn.utils.clip_grad_norm_(
                                model_cd.parameters(), max_norm=1.0
                            )

                            # 更新参数
                            scaler2.step(opt_cd_p2)
                            # opt_cd_p2.step()
                            scaler2.update()
                            opt_cd_p2.zero_grad(set_to_none=True)

                            # 更新学习率 - 只使用调度器
                            scheduler.step()

                            # 更新运行指标
                            running_loss += loss.item()
                            running_real_acc += real_acc
                            running_fake_acc += fake_acc

                            # 定期打印详细统计信息
                            if step % 100 == 0:
                                avg_loss = running_loss / min(
                                    100, step - step_phase2 + 1
                                )
                                avg_real_acc = running_real_acc / min(
                                    100, step - step_phase2 + 1
                                )
                                avg_fake_acc = running_fake_acc / min(
                                    100, step - step_phase2 + 1
                                )

                                print(f"\n统计信息(最近100步):")
                                print(f"平均损失: {avg_loss:.4f}")
                                print(f"真实样本平均准确率: {avg_real_acc:.4f}")
                                print(f"虚假样本平均准确率: {avg_fake_acc:.4f}")
                                print(f"当前学习率: {scheduler.get_last_lr()[0]:.6f}")

                                # 重置运行指标
                                running_loss = 0.0
                                running_real_acc = 0.0
                                running_fake_acc = 0.0

                            # 更新进度条
                            current_lr = scheduler.get_last_lr()[0]
                            pbar.set_description(
                                f"Phase 2 | loss: {loss.item():.4f}, acc: {(real_acc+fake_acc)/2:.3f}, loss_rf:{(loss_real+loss_fake)/2:.3f}, lr: {current_lr:.1e}"
                            )
                            pbar.update(1)

                            # Visdom可视化
                            if viz is not None and step % 100 == 0:
                                try:
                                    if "phase2_disc" in loss_windows:
                                        viz.line(
                                            Y=torch.tensor([loss.item()]),
                                            X=torch.tensor([step]),
                                            win=loss_windows["phase2_disc"],
                                            update="append",
                                        )

                                    if "phase2_acc" in loss_windows:
                                        viz.line(
                                            Y=torch.tensor(
                                                [[avg_acc, real_acc, fake_acc]]
                                            ),
                                            X=torch.tensor([step]),
                                            win=loss_windows["phase2_acc"],
                                            update="append",
                                        )
                                except:
                                    pass

                            step += 1

                            # 清理临时变量
                            del (
                                fake_outputs,
                                fake_outputs_noisy,
                                batch_local_targets_noisy,
                            )
                            del fake_global_embedded, fake_global_mask_embedded
                            del real_global_embedded, real_global_mask_embedded
                            del (
                                fake_predictions,
                                real_predictions,
                                fake_lf,
                                fake_gf,
                                real_lf,
                                real_gf,
                            )
                            del loss_real, loss_fake, fm_loss, lsb_loss, r1_loss, loss
                            del batch_data

                            # 在每个snaperiod_2步保存检查点和进行验证
                            if step % snaperiod_2 == 0:
                                try:
                                    # 评估判别器
                                    model_cd.eval()
                                    val_correct = 0
                                    val_total = 0
                                    with torch.no_grad():
                                        for val_batch in val_subset_loader:
                                            try:
                                                # 准备验证批次
                                                val_data = prepare_batch_data(
                                                    val_batch, device
                                                )
                                                (
                                                    val_local_inputs,
                                                    val_local_masks,
                                                    val_local_targets,
                                                    val_global_inputs,
                                                    val_global_masks,
                                                    val_global_targets,
                                                    val_metadata,
                                                ) = val_data
                                                # 生成假样本
                                                val_fake_outputs = (
                                                    safe_tensor_operation(
                                                        model_cn,
                                                        val_local_inputs,
                                                        val_local_masks,
                                                        val_global_inputs,
                                                        val_global_masks,
                                                    )
                                                )

                                                # 嵌入
                                                (
                                                    val_fake_global_embedded,
                                                    val_fake_global_mask_embedded,
                                                    _,
                                                ) = merge_local_to_global(
                                                    val_fake_outputs,
                                                    val_global_inputs,
                                                    val_global_masks,
                                                    val_metadata,
                                                )

                                                (
                                                    val_real_global_embedded,
                                                    val_real_global_mask_embedded,
                                                    _,
                                                ) = merge_local_to_global(
                                                    val_local_targets,
                                                    val_global_inputs,
                                                    val_global_masks,
                                                    val_metadata,
                                                )
                                                # 获取判别器预测
                                                val_fake_preds, _, _ = (
                                                    safe_tensor_operation(
                                                        model_cd,
                                                        val_fake_outputs,
                                                        val_local_masks,
                                                        val_fake_global_embedded,
                                                        val_fake_global_mask_embedded,
                                                    )
                                                )

                                                val_real_preds, _, _ = (
                                                    safe_tensor_operation(
                                                        model_cd,
                                                        val_local_targets,
                                                        val_local_masks,
                                                        val_real_global_embedded,
                                                        val_real_global_mask_embedded,
                                                    )
                                                )
                                                val_fake_preds = torch.sigmoid(
                                                    val_fake_preds
                                                )
                                                val_real_preds = torch.sigmoid(
                                                    val_real_preds
                                                )
                                                # 计算准确率
                                                val_total += val_fake_preds.size(0) * 2
                                                val_correct += (
                                                    (val_fake_preds < 0.5).sum()
                                                    + (val_real_preds >= 0.5).sum()
                                                ).item()
                                                print(
                                                    f"validation: fake_preds: {(val_fake_preds<0.5).sum()}/{val_fake_preds.size(0)}, real_preds: {(val_real_preds>0.5).sum()}/{val_real_preds.size(0)}"
                                                )

                                                # 清理验证变量
                                                del (
                                                    val_fake_outputs,
                                                    val_fake_global_embedded,
                                                    val_fake_global_mask_embedded,
                                                )
                                                del (
                                                    val_real_global_embedded,
                                                    val_real_global_mask_embedded,
                                                )
                                                del val_fake_preds, val_real_preds
                                                del val_data

                                            except Exception as e:
                                                print(f"验证批次处理失败: {e}")
                                                cleanup_memory()
                                                continue

                                    val_accuracy = val_correct / max(val_total, 1)
                                    print(f"\n验证准确率: {val_accuracy:.4f}")

                                    # 检查是否有改进 - 移除手动学习率调整
                                    if val_accuracy > best_acc:
                                        best_acc = val_accuracy
                                        best_model_path = os.path.join(
                                            result_dir, "phase_2", "model_cd_best"
                                        )
                                        torch.save(
                                            model_cd.state_dict(), best_model_path
                                        )
                                        print(
                                            f"发现新的最佳模型，准确率: {val_accuracy:.4f}"
                                        )
                                        patience_counter = 0
                                    else:
                                        patience_counter += 1
                                        print(
                                            f"未见改善。耐心计数: {patience_counter}/{max_patience}"
                                        )
                                        """# 如果长时间没有改善，降低学习率
                                        if patience_counter >= max_patience:
                                            for param_group in opt_cd.param_groups:
                                                param_group['lr'] *= 0.5
                                            print(f"性能停滞，将学习率降低到: {opt_cd.param_groups[0]['lr']:.6f}")
                                            patience_counter = 0"""

                                    # 保存模型

                                    # 保存模型

                                    # 找到所有匹配的检查点文件

                                    # 删除step小于当前step-1000的文件

                                    folder = os.path.join(result_dir, "phase_2")
                                    pattern = os.path.join(folder, "model_cd_step*")
                                    old_files = glob.glob(pattern)
                                    for file in old_files:
                                        try:
                                            # 从文件名中提取step数字
                                            basename = os.path.basename(file)
                                            match = re.search(
                                                r"model_cd_step(\d+)(?:\.pth)?$",
                                                basename,
                                            )
                                            if match:
                                                file_step = int(match.group(1))
                                                # 只删除step小于当前step-1000的文件
                                                if file_step < step - 1000:
                                                    os.remove(file)
                                                    print(
                                                        f"删除旧文件: {basename} (step={file_step})"
                                                    )
                                            else:
                                                print(f"无法解析文件名: {basename}")
                                        except Exception as e:
                                            print(f"删除文件失败: {e}")
                                    model_path = os.path.join(
                                        result_dir, "phase_2", f"model_cd_step{step}"
                                    )
                                    torch.save(model_cd.state_dict(), model_path)

                                    # 回到训练模式
                                    model_cd.train()

                                    # 准备Phase 2专用的状态
                                    phase1_scalers = {}
                                    if "scaler" in locals() and scaler is not None:
                                        phase1_scalers["cn"] = scaler
                                    if (
                                        "scaler_bqd" in locals()
                                        and scaler_bqd is not None
                                    ):
                                        phase1_scalers["bqd"] = scaler_bqd

                                    """save_checkpoint(
                                        model_cn,
                                        None,              # Phase 1不需要CD
                                        model_bqd,
                                        opt_cn_p1,
                                        None,              # Phase 1不需要CD优化器  
                                        opt_bqd,
                                        step,
                                        1,
                                        result_dir,
                                        best_val_loss=best_val_loss,
                                        best_acc=None,
                                        cn_lr=opt_cn_p1.param_groups[0]["lr"],
                                        cd_lr=None,
                                        bqd_lr=opt_bqd.param_groups[0]["lr"],
                                        schedulers=None,   # Phase 1不使用调度器
                                        scalers=phase1_scalers if phase1_scalers else None,
                                        stability_tracker=None,
                                    )"""
                                    """save_checkpoint(
                                        model_cn,
                                        model_cd,
                                        model_bqd,
                                        opt_cn_p1,
                                        opt_cd_p2,
                                        opt_bqd,
                                        step,
                                        2,
                                        result_dir,
                                        best_acc=best_acc,
                                        cd_lr=opt_cd_p2.param_groups[0]["lr"],
                                    )"""

                                except Exception as e:
                                    print(f"验证步骤失败: {e}")
                                    cleanup_memory()

                            # 判断是否完成训练
                            if step >= steps_2:
                                break

                        except Exception as e:
                            print(f"训练批次处理失败: {e}")
                            traceback.print_exc()  # 打印详细的错误堆栈信息
                            cleanup_memory()
                            continue

                pbar.close()
                print(f"Phase 2 训练完成! 最佳判别器准确率: {best_acc:.4f}")
                phase = 3

            except Exception as e:
                print(f"Phase 2训练过程中发生严重错误: {e}")
                cleanup_memory()
                raise

        # =================================================
        # Training Phase 3: 优化后的联合训练
        # =================================================

        if step_phase3 < steps_3 and phase == 3:
            print("开始/继续第3阶段训练...")
            try:
                cleanup_memory()
                pbar = tqdm(total=steps_3, initial=step_phase3)
                step = step_phase3

                # ==================== 优化配置 ====================
                # 梯度累积设置
                target_batch_size = 32
                actual_batch_size = batch_size  # 当前的batch_size (8)
                accumulation_steps = target_batch_size // actual_batch_size  # 2
                print(
                    f"使用梯度累积: {accumulation_steps} steps, 有效batch size: {target_batch_size}"
                )

                # 重新初始化优化器 - 使用更保守的学习率

                if cn_lr is not None:
                    opt_cn_p3.param_groups[0]["lr"] = cn_lr  # 设置初始学习率
                if cd_lr is not None:
                    opt_cd_p3.param_groups[0]["lr"] = cd_lr  # 设置初始学习率
                if bqd_lr is not None:
                    opt_bqd.param_groups[0]["lr"] = bqd_lr

                # 创建梯度缩放器
                scaler_cn = GradScaler(
                    enabled=True, init_scale=2**10, growth_interval=2000
                )
                scaler_cd = GradScaler(
                    enabled=True, init_scale=2**10, growth_interval=2000
                )
                scaler_bqd_p3 = GradScaler()

                # 学习率调度器
                scheduler_cn = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt_cn_p3, mode="min", factor=0.7, patience=8, min_lr=1e-6
                )
                scheduler_cd = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt_cd_p3, mode="max", factor=0.7, patience=5, min_lr=1e-7
                )
                stability_tracker = StabilityTracker()

                # 损失权重
                alpha_d = 1.0
                alpha_g = 0.1  # 降低对抗损失权重
                fm_weight = 2.0  # 降低特征匹配权重

                # 噪声设置
                start_noise = 0.005  # 降低初始噪声
                min_noise = 0.001

                # 累积变量
                accumulated_step = 0
                running_metrics = {
                    "recon_loss": 0.0,
                    "adv_loss": 0.0,
                    "disc_loss": 0.0,
                    "real_acc": 0.0,
                    "fake_acc": 0.0,
                    "count": 0,
                }

                # ==================== 主训练循环 ====================
                while step < steps_3:
                    for batch in train_loader:
                        try:
                            # 减少内存检查频率
                            if step % 50 == 0:
                                check_memory_and_cleanup(threshold_gb=12.0)

                            # 准备数据
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

                            batch_size = batch_local_targets.size(0)
                            noise_level = max(
                                min_noise, start_noise * (1.0 - step / steps_3)
                            )

                            # ==================== 训练判别器 ====================
                            for param in model_cn.parameters():
                                param.requires_grad = False
                            for param in model_bqd.parameters():
                                param.requires_grad = False
                            for param in model_cd.parameters():
                                param.requires_grad = True

                            """with autocast(
                                device_type=device.type,
                                dtype=torch.float32,
                                enabled=True,
                            ):"""
                            # 生成假样本
                            with torch.no_grad():
                                fake_outputs = safe_tensor_operation(
                                    model_cn,
                                    batch_local_inputs,
                                    batch_local_masks,
                                    batch_global_inputs,
                                    batch_global_masks,
                                )

                            # 应用适度噪声
                            batch_local_targets_noisy = (
                                batch_local_targets
                                + torch.randn_like(batch_local_targets) * noise_level
                            )
                            fake_outputs_noisy = (
                                fake_outputs
                                + torch.randn_like(fake_outputs) * noise_level
                            )

                            # 嵌入到全局
                            fake_global_embedded, fake_global_mask_embedded, _ = (
                                merge_local_to_global(
                                    fake_outputs_noisy,
                                    batch_global_inputs,
                                    batch_global_masks,
                                    metadata,
                                )
                            )
                            real_global_embedded, real_global_mask_embedded, _ = (
                                merge_local_to_global(
                                    batch_local_targets_noisy,
                                    batch_global_inputs,
                                    batch_global_masks,
                                    metadata,
                                )
                            )

                            # 标签平滑
                            real_labels = torch.ones(batch_size, 1, device=device) * 0.9
                            fake_labels = (
                                torch.zeros(batch_size, 1, device=device) + 0.1
                            )

                            # 判别器前向传播
                            fake_predictions, fake_lf, fake_gf = safe_tensor_operation(
                                model_cd,
                                fake_outputs_noisy,
                                batch_local_masks,
                                fake_global_embedded,
                                fake_global_mask_embedded,
                            )
                            real_predictions, real_lf, real_gf = safe_tensor_operation(
                                model_cd,
                                batch_local_targets_noisy,
                                batch_local_masks,
                                real_global_embedded,
                                real_global_mask_embedded,
                            )

                            # 计算准确率
                            with torch.no_grad():
                                real_probs = torch.sigmoid(real_predictions)
                                fake_probs = torch.sigmoid(fake_predictions)
                                real_acc = (real_probs >= 0.5).float().mean().item()
                                fake_acc = (fake_probs < 0.5).float().mean().item()

                            # 获取动态权重

                            real_weight, fake_weight = (
                                stability_tracker.get_loss_weights()
                            )

                            # 计算损失
                            loss_cd_real = (
                                F.binary_cross_entropy_with_logits(
                                    real_predictions, real_labels
                                )
                                * real_weight
                            )
                            loss_cd_fake = (
                                F.binary_cross_entropy_with_logits(
                                    fake_predictions, fake_labels
                                )
                                * fake_weight
                            )

                            # 特征对比损失（减少权重）
                            fm_loss_d = (
                                feature_contrastive_loss(
                                    real_lf, real_gf, fake_lf, fake_gf
                                )
                                * fm_weight
                            )

                            # 组合判别器损失并标准化
                            loss_cd = (
                                (loss_cd_real + loss_cd_fake + fm_loss_d)
                                * alpha_d
                                / accumulation_steps
                            )

                            # 判别器反向传播
                            scaler_cd.scale(loss_cd).backward()
                            # loss_cd.backward()

                            # =================== 训练BQD =======================
                            for param in model_bqd.parameters():
                                param.requires_grad = True
                            for param in model_cn.parameters():
                                param.requires_grad = False
                            for param in model_cd.parameters():
                                param.requires_grad = False

                            # BQD学习识别生成输出的边界
                            with torch.no_grad():
                                fake_outputs_for_bqd = safe_tensor_operation(
                                    model_cn,
                                    batch_local_inputs,
                                    batch_local_masks,
                                    batch_global_inputs,
                                    batch_global_masks,
                                )

                            # 第一次backward
                            nld_fo = normalization(
                                fake_outputs_for_bqd, batch_global_inputs
                            )

                            _, bqd_loss_out, _ = model_bqd(
                                nld_fo,
                                batch_local_masks,  # detach阻止梯度传播到补全网络
                            )
                            # 确保loss是标量
                            bqd_loss_out = ensure_scalar_loss(bqd_loss_out)
                            bqd_loss_out_scaled = bqd_loss_out / accumulation_steps * 2
                            scaler_bqd_p3.scale(bqd_loss_out_scaled).backward(
                                retain_graph=True
                            )

                            # 第二次backward
                            nld_li = normalization(
                                batch_local_inputs, batch_global_inputs
                            )
                            bqd_outputs_in, bqd_loss_in, _ = model_bqd(
                                nld_li, batch_local_masks
                            )
                            bqd_loss_in = ensure_scalar_loss(bqd_loss_in)
                            bqd_loss_in_scaled = bqd_loss_in / accumulation_steps * 2

                            bqd_loss_ta_scaled = (
                                bqd_loss_out_scaled + bqd_loss_in_scaled
                            )
                            scaler_bqd_p3.scale(bqd_loss_ta_scaled).backward()
                            # bqd_loss_ta_scaled.backward()

                            """# 第3次backward
                            bqd_outputs_ta, bqd_loss_ta, _ = model_bqd(
                                batch_local_targets, torch.zeros_like(batch_local_targets)
                            )
                            bqd_loss_ta = ensure_scalar_loss(bqd_loss_ta)
                            bqd_loss_ta_scaled = (
                                bqd_loss_ta / accumulation_steps * 2
                            )
                            scaler_bqd_p3.scale(bqd_loss_ta_scaled).backward() """

                            av_bqd_loss = (
                                bqd_loss_in_scaled
                                + bqd_loss_out_scaled
                                + bqd_loss_ta_scaled
                            ) / 3

                            # - 补全网络的对抗损失（让生成结果欺骗BQD） ####

                            """ # 冻结BQD参数，避免对抗损失影响BQD训练
                            for param in model_bqd.parameters():
                                param.requires_grad = False
                            for param in model_cn.parameters():
                                param.requires_grad = True

                            # 重新计算BQD输出，但不detach，让梯度流向补全网络
                            # !!!!注意，此处只返回mask部分的loss，否则会使mask外的异常边缘提取被判为优
                            _, _, bqd_loss_mask_adversarial = model_bqd(
                                fake_outputs, batch_local_masks
                            )
                            # 对抗损失：让补全网络生成的结果能欺骗BQD网络
                            adversarial_loss = bqd_loss_mask_adversarial/ accumulation_steps * 2  # 降低权重避免过强
                            scaler_cn.scale(adversarial_loss).backward()  """

                            # ==================== 训练生成器 ====================
                            for param in model_cd.parameters():
                                param.requires_grad = False
                            for param in model_bqd.parameters():
                                param.requires_grad = False
                            for param in model_cn.parameters():
                                param.requires_grad = True

                            """with autocast(
                                device_type=device.type,
                                dtype=torch.float32,
                                enabled=True,
                            ):"""
                            # 重新生成（需要梯度）
                            fake_outputs = safe_tensor_operation(
                                model_cn,
                                batch_local_inputs,
                                batch_local_masks,
                                batch_global_inputs,
                                batch_global_masks,
                            )

                            # 重建损失
                            fake_completed, fake_completed_mask, pos = (
                                merge_local_to_global(
                                    fake_outputs,
                                    batch_global_inputs,
                                    batch_global_masks,
                                    metadata,
                                )
                            )
                            loss_cn_recon = completion_network_loss(
                                fake_outputs,
                                batch_local_targets,
                                batch_local_masks,
                                batch_global_targets,
                                fake_completed,
                                fake_completed_mask,
                                pos,
                            )

                            # 对抗损失
                            fake_global_embedded, fake_global_mask_embedded, _ = (
                                merge_local_to_global(
                                    fake_outputs,
                                    batch_global_inputs,
                                    batch_global_masks,
                                    metadata,
                                )
                            )
                            fake_predictions, fake_lf, fake_gf = safe_tensor_operation(
                                model_cd,
                                fake_outputs,
                                batch_local_masks,
                                fake_global_embedded,
                                fake_global_mask_embedded,
                            )
                            loss_cn_adv = F.binary_cross_entropy_with_logits(
                                fake_predictions, real_labels
                            )
                            # BQD损失
                            nld_fo = normalization(fake_outputs, batch_global_inputs)
                            _, _, bloss = model_bqd(nld_fo, batch_local_masks)
                            # 确保loss是标量 - 新增这行
                            bloss = ensure_scalar_loss(bloss)
                            """if step > steps_3 * 3 / 5:
                                bloss *= 1.5
                                loss_cn_adv *= 2"""

                            # 组合生成器损失并标准化
                            loss_cn = (
                                loss_cn_recon + alpha_g * loss_cn_adv + bloss * 0.1
                            ) / accumulation_steps

                            # 生成器反向传播
                            # scaler_cn.scale(loss_cn).backward(retain_graph=False)
                            loss_cn.backward(retain_graph=False)

                            # ==================== 累积梯度更新 ====================
                            accumulated_step += 1

                            # 更新运行指标
                            running_metrics["recon_loss"] += loss_cn_recon.item()
                            running_metrics["adv_loss"] += loss_cn_adv.item()
                            running_metrics["disc_loss"] += (
                                loss_cd_real + loss_cd_fake
                            ).item()
                            running_metrics["real_acc"] += real_acc
                            running_metrics["fake_acc"] += fake_acc
                            running_metrics["count"] += 1

                            # 每accumulation_steps或到达epoch末尾时更新参数
                            if (
                                accumulated_step % accumulation_steps == 0
                                or accumulated_step == len(train_loader)
                            ):
                                # 更新判别器
                                scaler_cd.unscale_(opt_cd_p3)
                                torch.nn.utils.clip_grad_norm_(
                                    model_cd.parameters(), max_norm=1.0
                                )
                                scaler_cd.step(opt_cd_p3)
                                # opt_cd_p3.step()
                                scaler_cd.update()
                                opt_cd_p3.zero_grad(set_to_none=True)

                                # 更新BQD
                                scaler_bqd_p3.unscale_(opt_bqd)
                                torch.nn.utils.clip_grad_norm_(
                                    model_bqd.parameters(), max_norm=1.0
                                )
                                scaler_bqd_p3.step(opt_bqd)
                                scaler_bqd_p3.update()
                                opt_bqd.zero_grad(set_to_none=True)

                                # 更新生成器
                                # scaler_cn.unscale_(opt_cn_p3)
                                torch.nn.utils.clip_grad_norm_(
                                    model_cn.parameters(), max_norm=1.0
                                )
                                # scaler_cn.step(opt_cn_p3)
                                opt_cn_p3.step()
                                # scaler_cn.update()
                                opt_cn_p3.zero_grad(set_to_none=True)

                                # 更新稳定性追踪器
                                avg_recon = (
                                    running_metrics["recon_loss"]
                                    / running_metrics["count"]
                                )
                                avg_real_acc = (
                                    running_metrics["real_acc"]
                                    / running_metrics["count"]
                                )
                                avg_fake_acc = (
                                    running_metrics["fake_acc"]
                                    / running_metrics["count"]
                                )
                                stability_tracker.update(
                                    avg_real_acc, avg_fake_acc, avg_recon
                                )

                                # 重置运行指标
                                for key in running_metrics:
                                    running_metrics[key] = 0.0

                            # 解除参数冻结
                            for param in model_cd.parameters():
                                param.requires_grad = True

                            # ==================== 可视化和进度更新 ====================
                            if viz is not None and step % 100 == 0:
                                try:
                                    if "phase3_disc" in loss_windows:
                                        viz.line(
                                            Y=torch.tensor(
                                                [loss_cd.item() * accumulation_steps]
                                            ),
                                            X=torch.tensor([step]),
                                            win=loss_windows["phase3_disc"],
                                            update="append",
                                        )
                                    if "phase3_gen" in loss_windows:
                                        viz.line(
                                            Y=torch.tensor(
                                                [
                                                    [
                                                        loss_cn.item()
                                                        * accumulation_steps,
                                                        av_bqd_loss.item()
                                                        * accumulation_steps,
                                                        bloss.item()
                                                        * accumulation_steps,
                                                    ]
                                                ]
                                            ),
                                            X=torch.tensor([step]),
                                            win=loss_windows["phase3_gen"],
                                            update="append",
                                        )
                                    if "phase3_fake_acc" in loss_windows:
                                        viz.line(
                                            Y=torch.tensor([[fake_acc, real_acc]]),
                                            X=torch.tensor([step]),
                                            win=loss_windows["phase3_fake_acc"],
                                            update="append",
                                        )
                                except:
                                    pass

                            step += 1
                            pbar.set_description(
                                f"Phase 3 | D: {(loss_cd.item() * accumulation_steps):.4f}, "
                                f"G: {(loss_cn.item() * accumulation_steps):.4f}, "
                                f"R_acc: {real_acc:.3f}, F_acc: {fake_acc:.3f}, "
                                f"Smooth E: {bloss.item() * accumulation_steps:.4f}, "
                            )
                            pbar.update(1)

                            # 清理临时变量
                            del fake_outputs, fake_completed, fake_completed_mask, pos
                            del (
                                loss_cn_recon,
                                loss_cn_adv,
                                loss_cn,
                                loss_cd,
                                loss_cd_real,
                                loss_cd_fake,
                            )
                            del fake_global_embedded, fake_global_mask_embedded
                            del real_global_embedded, real_global_mask_embedded
                            del (
                                fake_predictions,
                                fake_lf,
                                fake_gf,
                                real_predictions,
                                real_lf,
                                real_gf,
                            )
                            del batch_local_targets_noisy, fake_outputs_noisy
                            del batch_data

                            # ==================== 验证和保存 ====================
                            if step % snaperiod_3 == 0:
                                try:
                                    val_results = validateAll(
                                        model_cn,
                                        model_cd,
                                        model_bqd,
                                        val_subset_loader,
                                        device,
                                    )

                                    print(f"\n=== 验证结果 (Step {step}) ===")
                                    print(f"重建损失: {val_results['recon_loss']:.4f}")
                                    print(f"对抗损失: {val_results['adv_loss']:.4f}")
                                    print(f"真实准确率: {val_results['real_acc']:.4f}")
                                    print(
                                        f"假样本准确率: {val_results['fake_acc']:.4f}"
                                    )
                                    print(
                                        f"判别器平均准确率: {val_results['disc_acc']:.4f}"
                                    )
                                    print(
                                        f"稳定性EMA - 真实: {stability_tracker.real_acc_ema:.3f}, 假样本: {stability_tracker.fake_acc_ema:.3f}"
                                    )

                                    # 更新学习率调度器
                                    scheduler_cn.step(val_results["recon_loss"])
                                    scheduler_cd.step(val_results["disc_acc"])

                                    # 检测判别器崩溃
                                    if stability_tracker.is_collapsed():
                                        print("检测到判别器不稳定，调整训练参数...")
                                        # 降低学习率
                                        for param_group in opt_cd_p3.param_groups:
                                            param_group["lr"] *= 0.5
                                            print(
                                                f"判别器学习率降低到: {param_group['lr']:.2e}"
                                            )

                                    # 可视化验证结果
                                    if viz is not None and "phase3_val" in loss_windows:
                                        try:
                                            viz.line(
                                                Y=torch.tensor(
                                                    [
                                                        [
                                                            val_results["recon_loss"],
                                                            val_results["adv_loss"],
                                                            val_results["real_acc"],
                                                            val_results["fake_acc"],
                                                            val_results["disc_acc"],
                                                            val_results["bqd_loss"],
                                                        ]
                                                    ]
                                                ),
                                                X=torch.tensor([step]),
                                                win=loss_windows["phase3_val"],
                                                update="append",
                                            )
                                        except:
                                            pass

                                    # 保存最佳模型
                                    if val_results["recon_loss"] < best_val_loss_joint:
                                        best_val_loss_joint = val_results["recon_loss"]
                                        cn_best_path = os.path.join(
                                            result_dir,
                                            "phase_3",
                                            "model_cn_best" + interrupt,
                                        )
                                        cd_best_path = os.path.join(
                                            result_dir,
                                            "phase_3",
                                            "model_cd_best" + interrupt,
                                        )
                                        torch.save(model_cn.state_dict(), cn_best_path)
                                        torch.save(model_cd.state_dict(), cd_best_path)
                                        print(
                                            f"保存新的最佳模型 (重建损失: {val_results['recon_loss']:.4f})"
                                        )

                                    # 可视化测试结果
                                    with torch.no_grad():
                                        try:
                                            test_batch = next(iter(val_subset_loader))
                                            test_data = prepare_batch_data(
                                                test_batch, device
                                            )
                                            (
                                                test_local_inputs,
                                                test_local_masks,
                                                test_local_targets,
                                                test_global_inputs,
                                                test_global_masks,
                                                test_global_targets,
                                                metadata,
                                            ) = test_data

                                            test_output = safe_tensor_operation(
                                                model_cn,
                                                test_local_inputs,
                                                test_local_masks,
                                                test_global_inputs,
                                                test_global_masks,
                                            )
                                            completed, completed_mask, _ = (
                                                merge_local_to_global(
                                                    test_output,
                                                    test_global_inputs,
                                                    test_global_masks,
                                                    metadata,
                                                )
                                            )
                                            nld_tli = normalization(
                                                test_local_inputs, test_global_inputs
                                            )
                                            nld_to = normalization(
                                                test_output, test_global_inputs
                                            )
                                            nld_tlt = normalization(
                                                test_local_targets, test_global_inputs
                                            )
                                            bqd_in, _, _ = model_bqd(
                                                nld_tli, test_local_masks
                                            )
                                            bqd_out, _, _ = model_bqd(
                                                nld_to, test_local_masks
                                            )
                                            nld_tlt = nld_tlt.unsqueeze(1)
                                            bqd_target, _, _ = model_bqd(
                                                nld_tlt, test_local_masks
                                            )

                                            visualize_results(
                                                test_local_targets,
                                                test_local_inputs,
                                                test_output,
                                                bqd_in,
                                                bqd_out,
                                                bqd_target,
                                                completed,
                                                step,
                                                "phase_3",
                                            )

                                            del (
                                                test_output,
                                                completed,
                                                completed_mask,
                                                test_data,
                                                bqd_in,
                                                bqd_out,
                                            )
                                        except Exception as e:
                                            print(f"可视化失败: {e}")

                                    folder = os.path.join(result_dir, "phase_3")

                                    # 处理两类模型文件
                                    for model_type in ["cn", "cd"]:
                                        pattern = os.path.join(
                                            folder, f"model_{model_type}_step*"
                                        )
                                        old_files = glob.glob(pattern)

                                        # 删除step小于当前step-1000的文件
                                        for file in old_files:
                                            try:
                                                # 从文件名中提取step数字
                                                basename = os.path.basename(file)
                                                match = re.search(
                                                    rf"model_{model_type}_step(\d+)(?:\.pth)?$",
                                                    basename,
                                                )
                                                if match:
                                                    file_step = int(match.group(1))
                                                    # 只删除step小于当前step-1000的文件
                                                    if file_step < step - 1000:
                                                        os.remove(file)
                                                        print(
                                                            f"删除旧文件: {basename} (step={file_step})"
                                                        )
                                                else:
                                                    print(f"无法解析文件名: {basename}")
                                            except Exception as e:
                                                print(f"删除文件失败: {e}")

                                    # 保存当前模型
                                    cn_path = os.path.join(
                                        result_dir, "phase_3", f"model_cn_step{step}"
                                    )
                                    cd_path = os.path.join(
                                        result_dir, "phase_3", f"model_cd_step{step}"
                                    )
                                    torch.save(model_cn.state_dict(), cn_path)
                                    torch.save(model_cd.state_dict(), cd_path)
                                    print(
                                        f"保存模型: model_cn_step{step} 和 model_cd_step{step}"
                                    )

                                    """save_checkpoint(
                                        model_cn,
                                        model_cd,
                                        model_bqd,
                                        opt_cn_p3,
                                        opt_cd_p3,
                                        opt_bqd,
                                        step,
                                        3,
                                        result_dir,
                                        best_val_loss=best_val_loss_joint,
                                        cn_lr=opt_cn_p3.param_groups[0]["lr"],
                                        cd_lr=opt_cd_p3.param_groups[0]["lr"],
                                        bqd_lr=opt_bqd.param_groups[0]["lr"],
                                        schedulers=(
                                            {"cn": scheduler_cn, "cd": scheduler_cd}
                                            if "scheduler_cn" in locals()
                                            else None
                                        ),
                                        scalers=(
                                            {
                                                "cn": scaler_cn,
                                                "cd": scaler_cd,
                                                "bqd": scaler_bqd_p3,
                                            }
                                            if "scaler_cn" in locals()
                                            else None
                                        ),
                                        stability_tracker=(
                                            stability_tracker
                                            if "stability_tracker" in locals()
                                            else None
                                        ),
                                    )"""

                                except Exception as e:
                                    print(f"验证步骤失败: {e}")
                                    cleanup_memory()

                            if step >= steps_3:
                                break

                        except Exception as e:
                            print(f"训练批次处理失败: {e}")
                            traceback.print_exc()  # 打印详细的错误堆栈信息
                            cleanup_memory()
                            continue

                pbar.close()
                print(f"🎉 Phase 3 训练完成! 最佳重建损失: {best_val_loss_joint:.4f}")

            except Exception as e:
                print(f"Phase 3训练过程中发生严重错误: {e}")
                cleanup_memory()
                raise

        # Final visualization: Compare best models from each phase
        try:
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
            p1_path = os.path.join(result_dir, "phase_1", "model_cn_best")
            if os.path.exists(p1_path):
                best_model_p1.load_state_dict(
                    torch.load(p1_path, weights_only=True, map_location=device)
                )
                best_model_p1.eval()
            # Best model from phase 3 (joint training)
            best_model_p3 = CompletionNetwork(input_channels=1).to(device)
            p3_path = os.path.join(result_dir, "phase_3", "model_cn_best")
            if os.path.exists(p3_path):
                best_model_p3.load_state_dict(
                    torch.load(p3_path, weights_only=True, map_location=device)
                )
                best_model_p3.eval()
            with torch.no_grad():
                # Generate completions with best models
                if os.path.exists(p1_path):
                    output_p1 = safe_tensor_operation(
                        best_model_p1,
                        test_local_inputs,
                        test_local_masks,
                        test_global_inputs,
                        test_global_masks,
                    )
                    # Merge outputs to global for visualization
                    completed_p1, _, _ = merge_local_to_global(
                        output_p1, test_global_inputs, test_global_masks, metadata
                    )
                else:
                    output_p1 = None
                    completed_p1 = None
                if os.path.exists(p3_path):
                    output_p3 = safe_tensor_operation(
                        best_model_p3,
                        test_local_inputs,
                        test_local_masks,
                        test_global_inputs,
                        test_global_masks,
                    )
                    completed_p3, _, _ = merge_local_to_global(
                        output_p3, test_global_inputs, test_global_masks, metadata
                    )
                else:
                    output_p3 = None
                    completed_p3 = None
                # 清理比较变量
                if output_p1 is not None:
                    del output_p1, completed_p1
                if output_p3 is not None:
                    del output_p3, completed_p3
                del test_data

            # 清理最终测试模型
            if "best_model_p1" in locals():
                del best_model_p1
            if "best_model_p3" in locals():
                del best_model_p3
        except Exception as e:
            print(f"最终可视化失败: {e}")
            cleanup_memory()
        print("Training complete!")

        # 最终清理
        cleanup_memory()

    except Exception as e:
        print(f"训练过程中发生严重错误: {e}")
        print(f"错误详情: {traceback.format_exc()}")
        cleanup_memory()
        raise
    finally:
        # 确保最终清理
        cleanup_memory()
        # 关闭数据加载器
        if "train_loader" in locals():
            del train_loader
        if "val_loader" in locals():
            del val_loader
        if "val_subset_loader" in locals():
            del val_subset_loader
        print("所有资源已清理完毕")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DEM completion network training")
    parser.add_argument(
        "--resume",
        type=str,
        default=r"",  # e:\KingCrimson Dataset\Simulate\data0\results31\latest_checkpoint.pth
        help="resume from checkpoint path",
    )
    parser.add_argument(
        "--dir", type=str, default="results46", help="directory to save results"
    )
    parser.add_argument(
        "--envi", type=str, default="DEM46", help="visdom environment name"
    )
    parser.add_argument("--cuda", type=str, default="cuda:3", help="CUDA device to use")
    parser.add_argument(
        "--test", type=bool, default=False, help="whether to run in test mode"
    )
    parser.add_argument("--batch", type=int, default=8, help="batch size for training")
    parser.add_argument(
        "--phase", type=int, default=3, help="training phase to start from (1, 2, or 3)"
    )
    args = parser.parse_args()
    with visdom_server():
        viz = visdom.Visdom()
        if viz.check_connection():
            print("成功连接到Visdom服务器")
            # 进行你的可视化操作
        else:
            print("连接失败")
    try:
        train(
            dir=args.dir,
            envi=args.envi,
            cuda=args.cuda,
            test=args.test,
            batch=args.batch,
            resume_from=args.resume,
            Phase=args.phase,
        )
    except KeyboardInterrupt:
        print("训练被用户中断")
        cleanup_memory()
    except Exception as e:
        print(f"训练失败: {e}")
        cleanup_memory()
        raise
