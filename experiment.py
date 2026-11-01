# Added/adapted for this course project, 2026-11-01.
# License: GPL-3.0; upstream attribution and modification details in SOURCES.md/CHANGES.md.
"""Shared loss, foreground metrics and offline plots; derived Dice retains upstream license."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from upstream_utils.dice_score import dice_loss
OUT = Path('runs')

def loss_fn(logits, target):
    one_hot = F.one_hot(target, 2).permute(0, 3, 1, 2).float()
    return F.cross_entropy(logits, target) + dice_loss(F.softmax(logits, dim=1), one_hot, multiclass=True)

@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    n = tp = fp = fn = 0
    for batch in loader:
        image = batch['image'].contiguous(memory_format=torch.channels_last)
        target = batch['mask']
        output = model(image)
        total_loss += float(loss_fn(output, target)) * len(image)
        n += len(image)
        pred = output.argmax(1).bool()
        foreground = target.bool()
        tp += int((pred & foreground).sum())
        fp += int((pred & ~foreground).sum())
        fn += int((~pred & foreground).sum())
    return dict(loss=total_loss / n, foreground_iou=tp / max(tp + fp + fn, 1),
                foreground_dice=2 * tp / max(2 * tp + fp + fn, 1), images=n,
                true_positive_pixels=tp, false_positive_pixels=fp, false_negative_pixels=fn)

def plot_results(model, dataset, history):
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    indices = np.linspace(0, len(dataset) - 1, 6, dtype=int)
    fig, axes = plt.subplots(6, 3, figsize=(9, 15), layout='constrained')
    model.eval()
    with torch.inference_mode():
        for row, index in enumerate(indices):
            sample = dataset[int(index)]
            prediction = model(sample['image'].unsqueeze(0).contiguous(memory_format=torch.channels_last)).argmax(1)[0].numpy()
            axes[row, 0].imshow(sample['image'].permute(1, 2, 0).numpy())
            axes[row, 1].imshow(sample['mask'].numpy(), cmap='gray', vmin=0, vmax=1)
            axes[row, 2].imshow(prediction, cmap='gray', vmin=0, vmax=1)
            axes[row, 0].set_ylabel(sample['name'], fontsize=8)
            for column, title in enumerate(['原图', '真实标注（宠物与边界为白色）', '最佳验证模型的预测']):
                if row == 0:
                    axes[row, column].set_title(title, fontsize=10)
                axes[row, column].set_xticks([])
                axes[row, column].set_yticks([])
    fig.suptitle('验证图片：原图、真实标注与最佳模型预测', fontsize=14)
    fig.savefig(OUT / 'predictions.png', dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), layout='constrained')
    ax.plot([r['epoch'] for r in history], [r['train_loss'] for r in history], 'o-', label='训练损失')
    ax.plot([r['epoch'] for r in history], [r['validation']['loss'] for r in history], 'o-', label='验证损失')
    ax.set(xlabel='训练轮次', ylabel='交叉熵 + Dice 损失', title='训练与验证损失')
    ax.set_xticks([r['epoch'] for r in history])
    ax.legend()
    fig.savefig(OUT / 'training_curve.png', dpi=150)
    plt.close(fig)
