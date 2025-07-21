import json
import os
import argparse
from torch.utils.data import DataLoader
from torch.optim import Adadelta, Adam
from torch.nn import BCELoss, DataParallel, MSELoss
from torchvision.utils import save_image
import torch
import numpy as np
from tqdm import tqdm
from modelcn4 import CompletionNetwork
from modelsM import ContextDiscriminator
from DemDataset3 import DemDataset, fast_collate_fn
import torch.nn.functional as F


def dem_completion_loss(gt, pred, mask):
    """DEM补全损失函数 - mask: 1表示有效，0表示无效/需要填充"""
    # 计算掩码区域的损失（需要填充的区域）
    hole_mask = 1 - mask  # 0->1, 1->0, 将掩码反转
    hole_loss = F.mse_loss(pred * hole_mask, gt * hole_mask)

    return hole_loss


def crop_local_from_global(global_tensor, metadata):
    """从全局数据中裁剪出局部区域"""
    # metadata: [centerx, centery, xbegain, ybegain, xnum, ynum, id_val]
    batch_size = global_tensor.shape[0]
    kernel_size = 33

    local_tensors = []

    for i in range(batch_size):
        centerx, centery, xbegain, ybegain = metadata[i][:4].astype(int)

        # 计算局部区域在全局数组中的位置
        minx = int(centerx - xbegain) - kernel_size // 2
        maxx = int(centerx - xbegain) + kernel_size // 2 + 1
        miny = int(centery - ybegain) - kernel_size // 2
        maxy = int(centery - ybegain) + kernel_size // 2 + 1

        # 提取局部区域
        local_region = global_tensor[i, :, minx:maxx, miny:maxy]
        local_tensors.append(local_region)

    return torch.stack(local_tensors, dim=0)


def save_dem_results(dem_tensor, save_path, nrow):
    """保存DEM结果，转换为可视化图像"""
    # 归一化到[0,1]用于可视化
    dem_min = dem_tensor.min()
    dem_max = dem_tensor.max()

    if dem_max > dem_min:
        normalized = (dem_tensor - dem_min) / (dem_max - dem_min)
    else:
        normalized = torch.zeros_like(dem_tensor)

    # 复制到3通道用于保存
    rgb_tensor = normalized.repeat(1, 3, 1, 1)
    save_image(rgb_tensor, save_path, nrow=nrow)

def merge_local_to_global(
            local_data, local_mask, global_data, global_mask, metadata, kernel_size=33
        ):
            """
            Merge local patch data into global data based on metadata.
            优化：增加错误处理和内存管理
            """
            try:
                merged_data = global_data.clone()
                merged_mask = global_mask.clone()

                pos = []
                for i in range(len(metadata)):
                    centerx = metadata[i][0]
                    centery = metadata[i][1]
                    xbegain = metadata[i][2]
                    ybegain = metadata[i][3]

                    # Calculate the top-left corner of the kernel_size x kernel_size window
                    minx = int(centerx - xbegain) - kernel_size // 2
                    maxx = int(centerx - xbegain) + kernel_size // 2 + 1
                    miny = int(centery - ybegain) - kernel_size // 2
                    maxy = int(centery - ybegain) + kernel_size // 2 + 1

                    # Embed the local data into the global data
                    merged_data[i, :, minx:maxx, miny:maxy] = local_data[i]
                    merged_mask[i, :, minx:maxx, miny:maxy] = 1 # local_mask[i]
                    pos.append([minx, maxx, miny, maxy])

                return merged_data, merged_mask, pos
            except Exception as e:
                print(f"合并本地到全局数据时发生错误: {e}")
                # 返回原始数据作为fallback
                return global_data.clone(), global_mask.clone(), []


parser = argparse.ArgumentParser()
parser.add_argument(
    "--json_dir",
    type=str,
    required=False,
    help="JSON文件目录",
    default=r"E:\KingCrimson Dataset\Simulate\data0\json",
)
parser.add_argument(
    "--array_dir",
    type=str,
    required=False,
    help="Array文件目录",
    default=r"E:\KingCrimson Dataset\Simulate\data0\arraynmask_a\array",
)
parser.add_argument(
    "--mask_dir",
    type=str,
    required=False,
    help="Mask文件目录",
    default=r"E:\KingCrimson Dataset\Simulate\data0\arraynmask_a\mask",
)
parser.add_argument(
    "--target_dir",
    type=str,
    required=False,
    help="Target文件目录",
    default=r"E:\KingCrimson Dataset\Simulate\data0\groundtruthstatus\statusarray",
)
parser.add_argument(
    "--result_dir",
    type=str,
    required=False,
    help="结果保存目录",
    default=r"E:\KingCrimson Dataset\Comparation\result0",
)
parser.add_argument("--data_parallel", action="store_true")
parser.add_argument("--init_model_cn", type=str, default=None)
parser.add_argument("--init_model_cd", type=str, default=None)
parser.add_argument("--steps_1", type=int, default=80000)
parser.add_argument("--steps_2", type=int, default=80000)
parser.add_argument("--steps_3", type=int, default=200000)
parser.add_argument("--snaperiod_1", type=int, default=500)
parser.add_argument("--snaperiod_2", type=int, default=500)
parser.add_argument("--snaperiod_3", type=int, default=500)
parser.add_argument("--cn_input_size", type=int, default=600)
parser.add_argument("--ld_input_size", type=int, default=33)
parser.add_argument("--bsize", type=int, default=8)
parser.add_argument("--bdivs", type=int, default=1)
parser.add_argument("--num_test_completions", type=int, default=8)
parser.add_argument("--alpha", type=float, default=4e-4)
parser.add_argument("--min_valid_pixels", type=int, default=20)
parser.add_argument("--max_valid_pixels", type=int, default=800)
parser.add_argument("--enable_synthetic_masks", action="store_true", default=True)
parser.add_argument("--synthetic_ratio", type=float, default=1.0)


def main(args):
    # ================================================
    # 准备工作
    # ================================================
    if not torch.cuda.is_available():
        raise Exception("需要至少一个GPU")
    gpu = torch.device("cuda:1")

    # 创建结果目录
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    for phase in ["phase_1", "phase_2", "phase_3"]:
        if not os.path.exists(os.path.join(args.result_dir, phase)):
            os.makedirs(os.path.join(args.result_dir, phase))

    # 加载DEM数据集
    print("加载DEM数据集...")
    train_dset = DemDataset(
        jsonDir=args.json_dir,
        arrayDir=args.array_dir,
        maskDir=args.mask_dir,
        targetDir=args.target_dir,
        min_valid_pixels=args.min_valid_pixels,
        max_valid_pixels=args.max_valid_pixels,
        enable_synthetic_masks=args.enable_synthetic_masks,
        synthetic_ratio=args.synthetic_ratio,
    )

    train_loader = DataLoader(
        train_dset,
        batch_size=args.bsize,
        shuffle=True,
        collate_fn=fast_collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # 保存训练配置
    args_dict = vars(args)
    with open(os.path.join(args.result_dir, "config.json"), mode="w") as f:
        json.dump(args_dict, f)

    # 损失函数
    alpha = torch.tensor(args.alpha, dtype=torch.float32).to(gpu)
    phase=2

    # ================================================
    # 训练阶段1: 预训练生成器
    # ================================================
    print("开始阶段1: 预训练CompletionNetwork...")

    model_cn = CompletionNetwork()
    if args.init_model_cn is not None:
        model_cn.load_state_dict(torch.load(args.init_model_cn, map_location="cpu"))
    if args.data_parallel:
        model_cn = DataParallel(model_cn)
    model_cn = model_cn.to(gpu)
    opt_cn = Adadelta(model_cn.parameters())

    cnt_bdivs = 0
    pbar = tqdm(total=args.steps_1)
    while pbar.n < args.steps_1 and phase==1:
        for batch_data in train_loader:
            # 解包数据
            (
                local_inputs,  # [B, C, 33, 33] - 局部输入（带掩码的数据）
                local_masks,  # [B, C, 33, 33] - 局部掩码（1=有效，0=无效）
                local_targets,  # [B, 1, 33, 33] - 局部目标
                global_inputs,  # [B, 1, 600, 600] - 全局输入
                global_masks,  # [B, 1, 600, 600] - 全局掩码
                global_targets,  # [B, 1, 600, 600] - 全局目标
                metadata,  # 元数据列表
            ) = batch_data

            # 移动到GPU
            global_inputs, global_masks, _ = merge_local_to_global(local_inputs, local_masks, global_inputs, global_masks, metadata)
            global_inputs = global_inputs.to(gpu)
            global_masks = global_masks.to(gpu)
            global_targets = global_targets.to(gpu)
            local_targets = local_targets.to(gpu)
            local_masks = local_masks.to(gpu)

            # 准备CompletionNetwork的输入：全局DEM + 全局mask
            input_cn = torch.cat((global_inputs, global_masks), dim=1)
            output = model_cn(input_cn)

            # 计算损失（在全局尺度上）
            loss = dem_completion_loss(global_targets, output, global_masks)

            # 反向传播
            loss.backward()
            cnt_bdivs += 1
            if cnt_bdivs >= args.bdivs:
                cnt_bdivs = 0
                opt_cn.step()
                opt_cn.zero_grad()
                pbar.set_description("phase 1 | train loss: %.5f" % loss.cpu())
                pbar.update()

                # 测试和保存
                if pbar.n % args.snaperiod_1 == 0:
                    model_cn.eval()
                    with torch.no_grad():
                        # 使用当前batch进行测试可视化
                        test_input = torch.cat((global_inputs, global_masks), dim=1)
                        test_output = model_cn(test_input)

                        # 创建完成结果：保留有效区域 + 填充无效区域
                        completed = global_inputs * global_masks + test_output * (
                            1 - global_masks
                        )

                        # 保存结果（只保存前几个样本）
                        num_save = min(
                            args.num_test_completions, global_inputs.shape[0]
                        )
                        imgs = torch.cat(
                            (
                                global_targets[:num_save].cpu(),
                                global_inputs[:num_save].cpu(),
                                completed[:num_save].cpu(),
                            ),
                            dim=0,
                        )

                        imgpath = os.path.join(
                            args.result_dir, "phase_1", "step%d.png" % pbar.n
                        )
                        model_cn_path = os.path.join(
                            args.result_dir, "phase_1", "model_cn_step%d" % pbar.n
                        )

                        save_dem_results(imgs, imgpath, nrow=num_save)

                        if args.data_parallel:
                            torch.save(model_cn.module.state_dict(), model_cn_path)
                        else:
                            torch.save(model_cn.state_dict(), model_cn_path)

                        # 只保留最新3个文件
                        def keep_latest_files(dir_path, prefix, ext, max_files=3):
                            files = [f for f in os.listdir(dir_path) if f.startswith(prefix) and f.endswith(ext)]
                            files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else -1)
                            for f in files[:-max_files]:
                                try:
                                    os.remove(os.path.join(dir_path, f))
                                except Exception:
                                    pass

                        phase_dir = os.path.join(args.result_dir, "phase_1")
                        keep_latest_files(phase_dir, "step", ".png", 3)
                        keep_latest_files(phase_dir, "model_cn_step", "", 3)
                    model_cn.train()
                if pbar.n >= args.steps_1:
                    break
    pbar.close()

    # ================================================
    # 训练阶段2: 预训练判别器
    # ================================================
    print("开始阶段2: 预训练ContextDiscriminator...")

    model_cd = ContextDiscriminator(
        local_input_shape=(1, args.ld_input_size, args.ld_input_size),
        global_input_shape=(1, args.cn_input_size, args.cn_input_size),
    )
    if args.init_model_cd is not None:
        model_cd.load_state_dict(torch.load(args.init_model_cd, map_location="cpu"))
    if args.data_parallel:
        model_cd = DataParallel(model_cd)
    model_cd = model_cd.to(gpu)
    opt_cd = Adadelta(model_cd.parameters())
    bceloss = BCELoss()

    model_cn.load_state_dict(torch.load(r"e:\KingCrimson Dataset\Simulate\data0\results27\phase_1\model_cn_step16000", map_location="cpu"))

    cnt_bdivs = 0
    pbar = tqdm(total=args.steps_2)
    while pbar.n < args.steps_2 and phase==2:
        for batch_data in train_loader:
            # 解包数据
            (
                local_inputs,
                local_masks,
                local_targets,
                global_inputs,
                global_masks,
                global_targets,
                metadata,
            ) = batch_data

            # 移动到GPU
            # global_inputs, global_masks, _ = merge_local_to_global(local_inputs, local_masks, global_inputs, global_masks, metadata)
            local_targets = local_targets.to(gpu)
            global_inputs = global_inputs.to(gpu)
            global_masks = global_masks.to(gpu)
            global_targets = global_targets.to(gpu)
            local_inputs = local_inputs.to(gpu)      # 新增
            local_masks = local_masks.to(gpu)        # 新增

            # === 假样本前向传播 ===
            fake_labels = torch.zeros((global_inputs.shape[0], 1)).to(gpu)

            # 准备生成器输入
            input_cn = torch.cat((global_inputs, global_masks), dim=1)

            with torch.no_grad():  # 不更新生成器
                output_cn = model_cn(local_inputs, local_masks, global_inputs, global_masks)

            global_inputs, _, _ = merge_local_to_global(output_cn, local_masks, global_inputs, global_masks, metadata)
            # 从生成的全局结果中提取局部区域
            fake_local = output_cn # crop_local_from_global(output_cn, metadata)
            fake_global = global_inputs # output_cn.detach()

            output_fake = model_cd((fake_local, fake_global))
            loss_fake = bceloss(output_fake, fake_labels)

            # === 真样本前向传播 ===
            real_labels = torch.ones((global_targets.shape[0], 1)).to(gpu)
            real_local = local_targets


            global_targets, _, _ = merge_local_to_global(local_targets, local_masks, global_inputs, global_masks, metadata)
            real_global = global_targets

            output_real = model_cd((real_local, real_global))
            loss_real = bceloss(output_real, real_labels)

            # 总损失
            loss = (loss_fake + loss_real) / 2.0

            # 反向传播
            loss.backward()
            cnt_bdivs += 1
            if cnt_bdivs >= args.bdivs:
                cnt_bdivs = 0
                opt_cd.step()
                opt_cd.zero_grad()
                real_acc = (output_real > 0.5).float().mean().item()
                fake_acc = (output_fake < 0.5).float().mean().item()
                pbar.set_description(f"Phase 2 | train loss: {loss.item():.5f}, real_acc: {real_acc:.3f}, fake_acc: {fake_acc:.3f}")
                pbar.update()

                # 测试和保存
                if pbar.n % args.snaperiod_2 == 0:
                    model_cn.eval()
                    model_cd.eval()
                    with torch.no_grad():
                        # 使用当前batch进行测试
                        test_input = torch.cat((global_inputs, global_masks), dim=1)
                        test_output = model_cn(local_inputs, local_masks, global_inputs, global_masks)

                        completed = global_inputs * global_masks + test_output * (
                            1 - global_masks
                        )

                        num_save = min(
                            args.num_test_completions, global_inputs.shape[0]
                        )
                        imgs = torch.cat(
                            (
                                global_targets[:num_save].cpu(),
                                global_inputs[:num_save].cpu(),
                                completed[:num_save].cpu(),
                            ),
                            dim=0,
                        )

                        imgpath = os.path.join(
                            args.result_dir, "phase_2", "step%d.png" % pbar.n
                        )
                        model_cd_path = os.path.join(
                            args.result_dir, "phase_2", "model_cd_step%d" % pbar.n
                        )

                        save_dem_results(imgs, imgpath, nrow=num_save)

                        if args.data_parallel:
                            torch.save(model_cd.module.state_dict(), model_cd_path)
                        else:
                            torch.save(model_cd.state_dict(), model_cd_path)

                        # 只保留最新3个文件
                        def keep_latest_files(dir_path, prefix, ext, max_files=3):
                            files = [f for f in os.listdir(dir_path) if f.startswith(prefix) and f.endswith(ext)]
                            files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else -1)
                            for f in files[:-max_files]:
                                try:
                                    os.remove(os.path.join(dir_path, f))
                                except Exception:
                                    pass

                        phase_dir = os.path.join(args.result_dir, "phase_2")
                        keep_latest_files(phase_dir, "step", ".png", 3)
                        keep_latest_files(phase_dir, "model_cd_step", "", 3)
                    model_cn.train()
                    model_cd.train()
                if pbar.n >= args.steps_2:
                    break
    pbar.close()

    # ================================================
    # 训练阶段3: 联合对抗训练
    # ================================================
    print("开始阶段3: 联合对抗训练...")

    cnt_bdivs = 0
    pbar = tqdm(total=args.steps_3)
    while pbar.n < args.steps_3 and phase==3:
        for batch_data in train_loader:
            # 解包数据
            (
                local_inputs,
                local_masks,
                local_targets,
                global_inputs,
                global_masks,
                global_targets,
                metadata,
            ) = batch_data

            # 移动到GPU
            local_targets = local_targets.to(gpu)
            global_inputs = global_inputs.to(gpu)
            global_masks = global_masks.to(gpu)
            global_targets = global_targets.to(gpu)

            # === 训练判别器 ===
            # 假样本
            fake_labels = torch.zeros((global_inputs.shape[0], 1)).to(gpu)
            input_cn = torch.cat((global_inputs, global_masks), dim=1)

            with torch.no_grad():
                output_cn = model_cn(input_cn)

            fake_local = crop_local_from_global(output_cn, metadata)
            fake_global = output_cn.detach()
            output_fake = model_cd((fake_local, fake_global))
            loss_cd_fake = bceloss(output_fake, fake_labels)

            # 真样本
            real_labels = torch.ones((global_targets.shape[0], 1)).to(gpu)
            real_local = local_targets
            real_global = global_targets
            output_real = model_cd((real_local, real_global))
            loss_cd_real = bceloss(output_real, real_labels)

            # 判别器总损失
            loss_cd = (loss_cd_fake + loss_cd_real) * alpha / 2.0

            # 反向传播判别器
            loss_cd.backward()
            cnt_bdivs += 1
            if cnt_bdivs >= args.bdivs:
                opt_cd.step()
                opt_cd.zero_grad()

            # === 训练生成器 ===
            # 重新计算output_cn以便梯度流动
            output_cn = model_cn(input_cn)

            # 补全损失
            loss_cn_1 = dem_completion_loss(global_targets, output_cn, global_masks)

            # 对抗损失
            fake_local = crop_local_from_global(output_cn, metadata)
            fake_global = output_cn
            output_fake = model_cd((fake_local, fake_global))
            loss_cn_2 = bceloss(output_fake, real_labels)  # 生成器希望骗过判别器

            # 生成器总损失
            loss_cn = (loss_cn_1 + alpha * loss_cn_2) / 2.0

            # 反向传播生成器
            loss_cn.backward()
            if cnt_bdivs >= args.bdivs:
                cnt_bdivs = 0
                opt_cn.step()
                opt_cn.zero_grad()

                pbar.set_description(
                    "phase 3 | train loss (cd): %.5f (cn): %.5f"
                    % (loss_cd.cpu(), loss_cn.cpu())
                )
                pbar.update()

                # 测试和保存
                if pbar.n % args.snaperiod_3 == 0:
                    model_cn.eval()
                    model_cd.eval()
                    with torch.no_grad():
                        test_input = torch.cat((global_inputs, global_masks), dim=1)
                        test_output = model_cn(test_input)

                        completed = global_inputs * global_masks + test_output * (
                            1 - global_masks
                        )

                        num_save = min(
                            args.num_test_completions, global_inputs.shape[0]
                        )
                        imgs = torch.cat(
                            (
                                global_targets[:num_save].cpu(),
                                global_inputs[:num_save].cpu(),
                                completed[:num_save].cpu(),
                            ),
                            dim=0,
                        )

                        # 保存结果文件名
                        imgpath = os.path.join(
                            args.result_dir, "phase_3", "step%d.png" % pbar.n
                        )
                        model_cn_path = os.path.join(
                            args.result_dir, "phase_3", "model_cn_step%d" % pbar.n
                        )
                        model_cd_path = os.path.join(
                            args.result_dir, "phase_3", "model_cd_step%d" % pbar.n
                        )

                        save_dem_results(imgs, imgpath, nrow=num_save)

                        # 保存模型
                        if args.data_parallel:
                            torch.save(model_cn.module.state_dict(), model_cn_path)
                            torch.save(model_cd.module.state_dict(), model_cd_path)
                        else:
                            torch.save(model_cn.state_dict(), model_cn_path)
                            torch.save(model_cd.state_dict(), model_cd_path)

                        # 只保留最新3个文件
                        def keep_latest_files(dir_path, prefix, ext, max_files=3):
                            files = [f for f in os.listdir(dir_path) if f.startswith(prefix) and f.endswith(ext)]
                            files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else -1)
                            for f in files[:-max_files]:
                                try:
                                    os.remove(os.path.join(dir_path, f))
                                except Exception:
                                    pass

                        phase_dir = os.path.join(args.result_dir, "phase_3")
                        keep_latest_files(phase_dir, "step", ".png", 3)
                        keep_latest_files(phase_dir, "model_cn_step", "", 3)
                        keep_latest_files(phase_dir, "model_cd_step", "", 3)
                    model_cn.train()
                    model_cd.train()
                if pbar.n >= args.steps_3:
                    break
    pbar.close()

    print("训练完成!")


if __name__ == "__main__":
    args = parser.parse_args()

    # 展开路径
    for path_arg in ["json_dir", "array_dir", "mask_dir", "target_dir", "result_dir"]:
        if hasattr(args, path_arg) and getattr(args, path_arg):
            setattr(args, path_arg, os.path.expanduser(getattr(args, path_arg)))

    if args.init_model_cn is not None:
        args.init_model_cn = os.path.expanduser(args.init_model_cn)
    if args.init_model_cd is not None:
        args.init_model_cd = os.path.expanduser(args.init_model_cd)

    main(args)
