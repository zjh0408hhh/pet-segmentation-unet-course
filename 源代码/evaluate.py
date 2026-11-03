# Added/adapted for this course project, 2026-11-01.
# License: GPL-3.0; see 部署文档/许可与来源.txt at the project root.
# Project paths reorganized, 2026-11-03.
"""Locked best-validation checkpoints: shared final test and CPU timing protocol."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.factory import build_model
from pet_data import PetDataset

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / '部署文档' / '数据划分'
RUN_NAMES = {'baseline': '原版模型', 'improved': '改进模型'}


def metrics(tp, fp, fn):
    return dict(foreground_iou=tp/max(tp+fp+fn, 1), foreground_dice=2*tp/max(2*tp+fp+fn, 1),
                true_positive_pixels=tp, false_positive_pixels=fp, false_negative_pixels=fn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--checkpoint-dir', type=Path, default=ROOT/'结果/新实验/模型权重')
    parser.add_argument('--experiments-dir', type=Path, default=ROOT/'结果/新实验')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'结果/新实验/测试对比')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out / '测试指标.json').exists():
        print('Final evaluation already exists; refusing to overwrite it automatically.', flush=True)
        return
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    summaries = {name: json.loads((args.experiments_dir/RUN_NAMES[name]/'训练汇总.json').read_text(encoding='utf-8')) for name in ('baseline', 'improved')}
    configs = [{k: v for k, v in s['configuration'].items() if k != 'model'} for s in summaries.values()]
    assert configs[0] == configs[1], 'Experiment conditions differ'
    assert all(s['completed_epochs'] == 10 and s['checkpoint_reload_metrics_match'] for s in summaries.values())
    dataset = PetDataset(args.data_root, SPLITS/'test.txt')
    assert len(dataset) == 3669
    models, records = {}, {}
    for name, summary in summaries.items():
        checkpoint = args.checkpoint_dir/f'{name}_best.pth'
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        assert state['epoch'] == summary['best_epoch']
        net = build_model(summary['configuration']['model']).to(memory_format=torch.channels_last)
        net.load_state_dict(state['model_state'])
        net.eval()
        models[name] = net
        records[name] = dict(best_epoch=state['epoch'], parameters=sum(p.numel() for p in net.parameters()),
                             checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                             validation=summary['best_validation'],
                             recorded_training_seconds=sum(r['train_seconds'] for r in summary['history']),
                             inference_seconds=[])
    # Identical real input, single-image latency, interleaved order and warmup.
    probe = dataset[0]['image'].unsqueeze(0).contiguous(memory_format=torch.channels_last)
    if args.check_only:
        archived_rows = args.experiments_dir/'测试对比/逐图指标.csv'
        first_row = None
        if archived_rows.exists():
            with archived_rows.open(encoding='utf-8-sig', newline='') as stream:
                first_row = next(csv.DictReader(stream))
            assert first_row['image'] == dataset.names[0]
        with torch.inference_mode():
            for name, model in models.items():
                output = model(probe)
                assert output.shape == (1, 2, 128, 128) and torch.isfinite(output).all()
                sample = dataset[0]
                pred = output.argmax(1)[0].bool()
                target = sample['mask'].bool()
                tp = int((pred & target).sum())
                fp = int((pred & ~target).sum())
                fn = int((~pred & target).sum())
                if first_row:
                    assert abs(tp/max(tp+fp+fn, 1)-float(first_row[f'{name}_iou'])) < 1e-6
        print('Checkpoint selection, shared configuration, test IDs and real-input predictions verified; no test sweep performed.')
        return
    with torch.inference_mode():
        for model in models.values():
            for _ in range(5):
                model(probe)
        for repeat in range(30):
            order = list(models) if repeat % 2 == 0 else list(reversed(models))
            for name in order:
                start = time.perf_counter()
                models[name](probe)
                records[name]['inference_seconds'].append(time.perf_counter()-start)
    for record in records.values():
        record['inference_ms_median'] = statistics.median(record['inference_seconds'])*1000
        record['inference_ms_p90'] = float(np.percentile(record['inference_seconds'], 90))*1000
    (out/'预测计时.json').write_text(json.dumps(records, indent=2), encoding='utf-8')

    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, drop_last=False)
    counts = {name: [0, 0, 0] for name in models}
    case_rows = []
    started = time.perf_counter()
    processed = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            image = batch['image'].contiguous(memory_format=torch.channels_last)
            target = batch['mask'].bool()
            per_model = {}
            order = list(models) if batch_index % 2 == 0 else list(reversed(models))
            for name in order:
                pred = models[name](image).argmax(1).bool()
                tp = (pred & target).sum((1, 2))
                fp = (pred & ~target).sum((1, 2))
                fn = (~pred & target).sum((1, 2))
                for k, value in enumerate((tp, fp, fn)):
                    counts[name][k] += int(value.sum())
                per_model[name] = [metrics(int(tp[i]), int(fp[i]), int(fn[i])) for i in range(len(image))]
            for i, image_name in enumerate(batch['name']):
                case_rows.append(dict(image=image_name,
                    baseline_iou=per_model['baseline'][i]['foreground_iou'],
                    improved_iou=per_model['improved'][i]['foreground_iou']))
            processed += len(image)
            if processed % 100 == 0 or processed == len(dataset):
                progress = dict(status='evaluating_test', processed_images=processed, total_images=len(dataset),
                                elapsed_seconds=time.perf_counter()-started)
                (out/'测试进度.json').write_text(json.dumps(progress, indent=2), encoding='utf-8')
                print(json.dumps(progress), flush=True)
    for name in models:
        records[name]['test'] = dict(images=processed, **metrics(*counts[name]))
    with (out/'逐图指标.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=['image', 'baseline_iou', 'improved_iou'])
        writer.writeheader()
        writer.writerows(case_rows)
    result = dict(protocol='Checkpoints selected only by validation Dice; final architecture fixed before testing. No threshold tuning on test.',
                  input_size=128, cpu_threads=6, test_batch_size=2, inference_timing_batch_size=1,
                  inference_timing='FP32 channels_last; same preloaded real image; 5 warmups/model, 30 alternating-order repeats; excludes loading and output saving.',
                  metric='Foreground TP/FP/FN aggregated over all test pixels; border merged into foreground.',
                  test_split_sha256=hashlib.sha256((SPLITS/'test.txt').read_bytes()).hexdigest(),
                  runs=records, test_evaluation_seconds=time.perf_counter()-started,
                  limitation='One seed and one training run per architecture; not evidence of statistical significance.')
    result['differences'] = dict(
        test_iou_percentage_points=100*(records['improved']['test']['foreground_iou']-records['baseline']['test']['foreground_iou']),
        test_dice_percentage_points=100*(records['improved']['test']['foreground_dice']-records['baseline']['test']['foreground_dice']),
        parameter_reduction=1-records['improved']['parameters']/records['baseline']['parameters'],
        inference_speedup=records['baseline']['inference_ms_median']/records['improved']['inference_ms_median'])
    (out/'测试指标.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (out/'测试进度.json').write_text(json.dumps(dict(status='complete', processed_images=processed), indent=2), encoding='utf-8')

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    by_name = {name: i for i, name in enumerate(dataset.names)}
    ranked = sorted(case_rows, key=lambda r: r['improved_iou']-r['baseline_iou'])
    chosen = [dataset.names[int(i)] for i in np.linspace(0, len(dataset)-1, 4, dtype=int)]
    chosen += [ranked[-1]['image'], ranked[0]['image']]
    fig, axes = plt.subplots(6, 4, figsize=(12, 16), layout='constrained')
    with torch.inference_mode():
        for row, name in enumerate(chosen):
            sample = dataset[by_name[name]]
            values = [sample['image'].permute(1, 2, 0).numpy(), sample['mask'].numpy()]
            for model in models.values():
                values.append(model(sample['image'].unsqueeze(0).contiguous(memory_format=torch.channels_last)).argmax(1)[0].numpy())
            for column, value in enumerate(values):
                axes[row, column].imshow(value, **({} if column==0 else dict(cmap='gray',vmin=0,vmax=1)))
                axes[row, column].set_xticks([])
                axes[row, column].set_yticks([])
                if row == 0:
                    axes[row, column].set_title(['原图', '真实标注', '原版预测', '改进版预测'][column])
            axes[row, 0].set_ylabel(name + ('\n最大改善例' if row==4 else '\n最大退步例' if row==5 else '\n固定间隔样本'), fontsize=8)
    fig.suptitle('独立测试集对比：固定间隔样本 + 最大改善 / 最大退步案例', fontsize=13)
    fig.savefig(out/'预测对比.png', dpi=130)
    plt.close(fig)
    print('TEST_COMPARISON_COMPLETE', json.dumps(result['differences']), flush=True)


if __name__ == '__main__':
    main()
