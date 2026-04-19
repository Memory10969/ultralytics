import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from ultralytics.nn.modules import BackgroundSuppression


def visualize_background_suppression(trainer):
    """
    在每个 epoch 结束后，保存 BackgroundSuppression 模块的：
      - 背景掩码 (mask)
      - 前景权重 (fg_weight)
      - 抑制后的输出特征图 (out)
    同时打印当前阈值。
    """
    # 只在主进程保存（多GPU时）
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() != 0:
            return

    # 查找所有的 BackgroundSuppression 模块
    modules = [m for m in trainer.model.model.modules() if isinstance(m, BackgroundSuppression)]
    if not modules:
        return

    save_dir = Path('./bg_suppression_viz')
    save_dir.mkdir(parents=True, exist_ok=True)

    epoch = trainer.epoch + 1
    for idx, module in enumerate(modules):
        # 1. 获取需要可视化的张量（需确保模块中存储了这些中间结果）
        #    注意：模块 forward 中需要保存 mask 或 fg_weight 等属性
        if hasattr(module, 'last_mask'):
            mask = module.last_mask          # (B,1,H,W) 硬掩码（背景为1）
            fg_weight = 1 - mask
        elif hasattr(module, 'last_fg_weight'):
            fg_weight = module.last_fg_weight
            mask = 1 - fg_weight
        else:
            # 如果模块没有存储，可以跳过
            continue

        # 获取输出特征图（模块的输出）
        # 需要模块存储 last_output，或者从 trainer 中获取（较复杂）
        # 这里假设模块已存储 last_output
        if hasattr(module, 'last_output'):
            out = module.last_output
        else:
            continue

        # 取第一个 batch 的第一个样本
        mask_np = mask[0, 0].detach().cpu().numpy()  # (H, W)
        fg_np = fg_weight[0, 0].detach().cpu().numpy()
        out_np = out[0].detach().cpu().numpy()       # (C, H, W)
        # 输出特征图取通道均值
        out_mean = np.mean(out_np, axis=0)            # (H, W)

        # 归一化到 0-255
        def norm(arr):
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
            return (arr * 255).astype(np.uint8)

        mask_img = norm(mask_np)
        fg_img = norm(fg_np)
        out_img = norm(out_mean)

        # 转为伪彩色（可选）
        mask_color = cv2.applyColorMap(mask_img, cv2.COLORMAP_JET)
        fg_color = cv2.applyColorMap(fg_img, cv2.COLORMAP_JET)
        out_color = cv2.applyColorMap(out_img, cv2.COLORMAP_JET)

        # 保存图像
        prefix = f"epoch{epoch}_module{idx}"
        cv2.imwrite(str(save_dir / f"{prefix}_mask.png"), mask_color)
        cv2.imwrite(str(save_dir / f"{prefix}_fg_weight.png"), fg_color)
        cv2.imwrite(str(save_dir / f"{prefix}_output.png"), out_color)

        # 打印阈值
        if hasattr(module, 'threshold'):
            print(f"[Epoch {epoch}] BackgroundSuppression module {idx} threshold = {module.threshold.item():.4f}")

def on_train_epoch_end(trainer):
    visualize_background_suppression(trainer)