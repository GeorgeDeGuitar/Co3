import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):
        # Focal loss for elevation data (focusing on areas with high error)
        loss = F.mse_loss(pred, target, reduction="none")
        # Apply focal weighting
        pt = torch.exp(-loss)  # Similar to how focal loss works for classification
        focal_weight = (1 - pt) ** self.gamma
        # Apply weight
        return (focal_weight * loss).mean()


def masked_tv_loss(completed, completed_mask):
    """
    计算只在补全区域 (completed_mask == 1) 的 TV 损失
    Args:
        completed: 补全输出，[B, C, H, W]
        completed_mask: 补全掩码，[B, 1, H, W]，1 表示补全区域
    Returns:
        loss: 掩码加权的 TV 损失
    """
    diff_h = torch.abs(completed[:, :, 1:, :] - completed[:, :, :-1, :])
    diff_w = torch.abs(completed[:, :, :, 1:] - completed[:, :, :, :-1])

    # 调整掩码尺寸以匹配差异图
    mask_h = completed_mask[:, :, 1:, :]
    mask_w = completed_mask[:, :, :, 1:]

    # 只在补全区域 (mask == 1) 计算损失
    loss_h = (diff_h * mask_h).sum()
    loss_w = (diff_w * mask_w).sum()

    return (loss_h + loss_w) / (mask_h.sum() + mask_w.sum() + 1e-8)


def edge_continuity_loss(completed, global_target, pos, edge_width, completed_mask):
    """
    计算局部补全区域与全局目标的连续性损失，包括0阶(像素)、1阶(梯度)和2阶(曲率)误差
    对整个区域应用权重，距离边缘越近权重越大
    使用向量化操作提高计算效率
    
    Args:
        completed: 局部补全输出，[B, C, H_local, W_local]
        global_target: 全局目标图像，[B, C, H_global, W_global]
        pos: 局部区域位置 [xmin, xmax, ymin, ymax]
        edge_width: 边缘区域的宽度（像素）
    
    Returns:
        loss: 加权边缘连续性损失
    """
    batch_size, channels, h_local, w_local = completed.shape
    zero_loss = 0.0
    one_loss = 0.0
    two_loss = 0.0

    for b in range(batch_size):
        # 提取位置信息
        xmin, xmax, ymin, ymax = pos[b]
        
        # 计算内部区域大小
        inner_h, inner_w = xmax - xmin, ymax - ymin
        
        # 提取包含内部区域和周围edge_width区域的完整区域
        extended_xmin, extended_xmax = xmin - edge_width, xmax + edge_width
        extended_ymin, extended_ymax = ymin - edge_width, ymax + edge_width
        
        # 提取全局目标和完成区域的扩展区域
        global_region = global_target[b:b+1, :, extended_xmin:extended_xmax, extended_ymin:extended_ymax]
        completed_region = completed[b:b+1, :, extended_xmin:extended_xmax, extended_ymin:extended_ymax]
        comtarget_region = completed_mask[b:b+1, :, extended_xmin:extended_xmax, extended_ymin:extended_ymax]
        
        # 创建权重掩码，使用向量化操作替代循环
        ext_h, ext_w = extended_xmax - extended_xmin, extended_ymax - extended_ymin

        
        # 生成坐标网格
        y_coords = torch.arange(ext_h, device=completed.device).view(-1, 1).repeat(1, ext_w)
        x_coords = torch.arange(ext_w, device=completed.device).view(1, -1).repeat(ext_h, 1)
        
        # 转换为相对内部区域的坐标
        rel_y = y_coords - edge_width
        rel_x = x_coords - edge_width
        
        # 计算到补全区域边界的距离
        dist_to_top = rel_y  # 到上边缘
        dist_to_bottom = inner_h - rel_y  # 到下边缘
        dist_to_left = rel_x  # 到左边缘
        dist_to_right = inner_w - rel_x  # 到右边缘
        
        # 计算到最近边界的距离（曼哈顿距离）
        dist_to_edge = abs(torch.min(
            torch.min(dist_to_top, dist_to_bottom),
            torch.min(dist_to_left, dist_to_right)
        ))
        
        
        # 指数衰减权重
        decay_factor = 0.4  # 可调衰减因子，替换原来的 10
        weight_mask = torch.exp(-dist_to_edge * decay_factor)
        
        # 扩展权重掩码的维度
        weight_mask = weight_mask.view(1, 1, ext_h, ext_w).expand(1, channels, ext_h, ext_w)
        
        # 1. 计算像素差异 (0阶导数误差)
        pixel_diff = torch.abs(completed_region - global_region) * weight_mask*comtarget_region
        zero_loss += pixel_diff.sum() / ((weight_mask*comtarget_region).sum() + 1e-8)

        # 2. 计算梯度差异 (1阶导数误差)
        # 水平梯度
        out_grad_h = completed_region[:, :, 1:, :] - completed_region[:, :, :-1, :]
        global_grad_h = global_region[:, :, 1:, :] - global_region[:, :, :-1, :]
        weight_mask_h = weight_mask[:, :, 1:, :]
        comtarget_region_h = comtarget_region[:, :, 1:, :]
        
        # 垂直梯度
        out_grad_w = completed_region[:, :, :, 1:] - completed_region[:, :, :, :-1]
        global_grad_w = global_region[:, :, :, 1:] - global_region[:, :, :, :-1]
        weight_mask_w = weight_mask[:, :, :, 1:]
        comtarget_region_w = comtarget_region[:, :, :, 1:]

        grad_diff_h = torch.abs(out_grad_h - global_grad_h) * weight_mask_h*comtarget_region_h
        grad_diff_w = torch.abs(out_grad_w - global_grad_w) * weight_mask_w*comtarget_region_w

        one_loss += (grad_diff_h.sum() + grad_diff_w.sum()) / (
            (weight_mask_h*comtarget_region_h).sum() + (weight_mask_w*comtarget_region_w).sum() + 1e-8
        )

        # 3. 计算二阶导数差异 (2阶导数误差/曲率误差) - 并行计算所有方向
        # 水平二阶导数
        out_grad2_h = (
            completed_region[:, :, 2:, :]
            - 2 * completed_region[:, :, 1:-1, :]
            + completed_region[:, :, :-2, :]
        )
        global_grad2_h = (
            global_region[:, :, 2:, :]
            - 2 * global_region[:, :, 1:-1, :]
            + global_region[:, :, :-2, :]
        )
        weight_mask_h2 = weight_mask[:, :, 2:, :]
        comtarget_region_h2 = comtarget_region[:, :, 2:, :]

        # 垂直二阶导数
        out_grad2_w = (
            completed_region[:, :, :, 2:]
            - 2 * completed_region[:, :, :, 1:-1]
            + completed_region[:, :, :, :-2]
        )
        global_grad2_w = (
            global_region[:, :, :, 2:]
            - 2 * global_region[:, :, :, 1:-1]
            + global_region[:, :, :, :-2]
        )
        weight_mask_w2 = weight_mask[:, :, :, 2:]
        comtarget_region_w2 = comtarget_region[:, :, :, 2:]

        # 对角线二阶导数
        out_grad2_diag = (
            completed_region[:, :, 2:, 2:]
            - 2 * completed_region[:, :, 1:-1, 1:-1]
            + completed_region[:, :, :-2, :-2]
        )
        global_grad2_diag = (
            global_region[:, :, 2:, 2:]
            - 2 * global_region[:, :, 1:-1, 1:-1]
            + global_region[:, :, :-2, :-2]
        )
        weight_mask_diag = weight_mask[:, :, 2:, 2:]
        comtarget_region_diag = comtarget_region[:, :, 2:, 2:]

        # 并行计算所有二阶导数差异
        grad2_diff_h = torch.abs(out_grad2_h - global_grad2_h) * weight_mask_h2 * comtarget_region_h2
        grad2_diff_w = torch.abs(out_grad2_w - global_grad2_w) * weight_mask_w2 * comtarget_region_w2
        grad2_diff_diag = torch.abs(out_grad2_diag - global_grad2_diag) * weight_mask_diag * comtarget_region_diag

        two_loss += (
            grad2_diff_h.sum() + grad2_diff_w.sum() + grad2_diff_diag.sum()
        ) / ((weight_mask_h2*comtarget_region_h2).sum() + (weight_mask_w2*comtarget_region_w2).sum() + (weight_mask_diag*comtarget_region_diag).sum() + 1e-8)

    # 加权合并损失项
    f1 = 0.6  # 像素差异权重
    f2 = 0.2  # 一阶导数权重
    f3 = 0.2  # 二阶导数权重
    total_loss = f1 * zero_loss + f2 * one_loss + f3 * two_loss

    return total_loss / batch_size


def completion_network_loss(
    output, local_target, mask, global_target, completed, completed_mask, pos
):
    """Calculate the completion network loss."""
    nw = 0.3
    vw = 0.1
    sw = 0 #completed，大部分在无法改变的global
    ew = 0.6
    criterion = nn.L1Loss()
    mse_loss = nn.MSELoss(reduction="mean")
    null_loss = mse_loss(local_target * (1-mask), output * (1-mask))  # target 是目标补全图像
    '''valid_loss = mse_loss(
        local_target * mask, output * mask
    )  # target 是目标补全图像'''
    # smooth_loss = masked_tv_loss(completed, completed_mask) * 100
    edge_loss = edge_continuity_loss(completed, global_target, pos, 2, completed_mask)*10
    print(
        f"Null loss: {null_loss.item()*nw}, Edge loss: {edge_loss.item()*ew}"
    )
    return null_loss * nw + edge_loss * ew

    # 增设沿global local边缘平滑的函数！
