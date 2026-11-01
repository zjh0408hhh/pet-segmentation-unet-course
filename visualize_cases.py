# Added/adapted for this course project, 2026-11-01.
# License: GPL-3.0; upstream attribution and modification details in SOURCES.md/CHANGES.md.
"""Visualize both the strongest improvement and strongest regression, without tuning."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from models.factory import build_model
from pet_data import PetDataset

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--checkpoint-dir', type=Path, default=ROOT/'checkpoints')
    parser.add_argument('--comparison-dir', type=Path, default=ROOT/'runs/comparison')
    args = parser.parse_args()
    out = args.comparison_dir
    result = json.loads((out/'test_comparison.json').read_text(encoding='utf-8'))
    with (out/'per_image_test.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = [dict(image=r['image'], baseline_iou=float(r['baseline_iou']), improved_iou=float(r['improved_iou'])) for r in csv.DictReader(stream)]
    assert len(rows) == 3669 and len({r['image'] for r in rows}) == 3669
    for row in rows:
        row['delta'] = row['improved_iou']-row['baseline_iou']
    ranked = sorted(rows, key=lambda r: r['delta'])
    torch.set_num_threads(6)
    models = {}
    for name, run in result['runs'].items():
        state = torch.load(args.checkpoint_dir/f'{name}_best.pth', map_location='cpu', weights_only=True)
        model = build_model('upstream_unet' if name=='baseline' else 'efficient_unet').to(memory_format=torch.channels_last)
        model.load_state_dict(state['model_state'])
        model.eval()
        models[name] = model
    dataset = PetDataset(args.data_root, ROOT/'splits/test.txt')
    index = {name: i for i, name in enumerate(dataset.names)}
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    cases = []
    with torch.inference_mode():
        for kind, row in [('最大改善', ranked[-1]), ('最大退步', ranked[0])]:
            sample = dataset[index[row['image']]]
            image = sample['image'].unsqueeze(0).contiguous(memory_format=torch.channels_last)
            target = sample['mask'].numpy().astype(bool)
            pictures = [sample['image'].permute(1, 2, 0).numpy(), target]
            details = {}
            for name, model in models.items():
                pred = model(image).argmax(1)[0].numpy().astype(bool)
                pictures.append(pred)
                tp = int((pred & target).sum())
                fp = int((pred & ~target).sum())
                fn = int((~pred & target).sum())
                details[name] = dict(tp=tp, fp=fp, fn=fn, single_image_iou=tp/max(tp+fp+fn, 1))
            fig, axes = plt.subplots(1, 4, figsize=(12, 3.8), layout='constrained')
            titles = ['原图', '真实标注', f"原版 IoU {row['baseline_iou']:.3f}", f"改进版 IoU {row['improved_iou']:.3f}"]
            for i, picture in enumerate(pictures):
                axes[i].imshow(picture, **({} if i==0 else dict(cmap='gray', vmin=0, vmax=1)))
                axes[i].set_title(titles[i])
                axes[i].axis('off')
            fig.suptitle(f"{kind}案例：{row['image']}（从全部测试图片中按 IoU 差值选取）", fontsize=12)
            fig.savefig(out/f'{kind}案例.png', dpi=150)
            plt.close(fig)
            cases.append(dict(kind=kind, **row, pixel_errors=details))
    changes = dict(images_improved=sum(r['delta']>1e-8 for r in rows),
                   images_regressed=sum(r['delta']<-1e-8 for r in rows),
                   images_tied=sum(abs(r['delta'])<=1e-8 for r in rows),
                   median_per_image_iou_delta=statistics.median(r['delta'] for r in rows),
                   mean_per_image_iou_baseline=statistics.mean(r['baseline_iou'] for r in rows),
                   mean_per_image_iou_improved=statistics.mean(r['improved_iou'] for r in rows))
    (out/'case_analysis.json').write_text(json.dumps(dict(selection='Extrema across all test images; these are deliberately selected extreme examples, not typical-performance estimates.',
                  scope='No model or threshold changes after observing test results.', counts=changes, cases=cases),ensure_ascii=False,indent=2), encoding='utf-8')
    print(json.dumps(dict(counts=changes,cases=cases), ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
