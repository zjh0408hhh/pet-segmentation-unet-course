# Added/adapted for this course project, 2026-11-01.
# License: GPL-3.0; upstream attribution and modification details in SOURCES.md/CHANGES.md.
"""Full-data CPU training shared by the baseline and modified architectures."""
import argparse
import gc
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.factory import build_model
from pet_data import PetDataset
import experiment as reporting
from utils.runtime import acquire_lock, keep_awake

PROJECT = Path(__file__).resolve().parent




def atomic_save(value, path):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--model', choices=['upstream_unet', 'efficient_unet'], default='upstream_unet')
    parser.add_argument('--checkpoint-dir', type=Path, default=PROJECT/'checkpoints')
    parser.add_argument('--check-only', action='store_true', help='Validate model, dataset and optional resume state without training.')
    args = parser.parse_args()
    keep_awake()
    if args.epochs < 1:
        raise ValueError('--epochs must be positive')
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    torch.manual_seed(2026)
    np.random.seed(2026)
    random.seed(2026)
    run_name = 'baseline'
    if args.model == 'efficient_unet':
        run_name = 'improved'
    out = PROJECT / 'runs' / run_name
    out.mkdir(parents=True, exist_ok=True)
    run_lock = acquire_lock(out / '.training.lock')
    reporting.OUT = out
    best_path = args.checkpoint_dir / f'{run_name}_best.pth'
    last_path = args.checkpoint_dir / f'{run_name}_resume.pth'
    best_path.parent.mkdir(exist_ok=True)
    if (last_path.exists() or (out / 'summary.json').exists()) and not args.resume:
        raise RuntimeError('Results or checkpoints already exist. Use --resume to continue safely.')

    train_names = (PROJECT / 'splits' / 'train.txt').read_text().splitlines()
    val_names = (PROJECT / 'splits' / 'val.txt').read_text().splitlines()
    assert (len(train_names), len(val_names)) == (2944, 736)
    assert not set(train_names) & set(val_names)
    assert not (set(train_names) | set(val_names)) & set((PROJECT / 'splits' / 'test.txt').read_text().splitlines())
    train_split, val_split = 'train.txt', 'val.txt'
    train_set = PetDataset(args.data_root, PROJECT / 'splits' / train_split)
    val_set = PetDataset(args.data_root, PROJECT / 'splits' / val_split)
    generator = torch.Generator().manual_seed(2026)
    train_loader = DataLoader(train_set, batch_size=2, shuffle=True, num_workers=0,
                              generator=generator, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=2, shuffle=False, num_workers=0, drop_last=False)
    model = build_model(args.model).to(memory_format=torch.channels_last)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-5, weight_decay=1e-8, momentum=0.999, foreach=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    config = dict(model=args.model, upstream_commit='21d7850f2af30a9695bbeea75f3136aa538cfc4a',
                  image_size=128, batch_size=2, cpu_threads=6, seed=2026,
                  train_images=len(train_set), validation_images=len(val_set), breeds=37,
                  train_split=train_split, validation_split=val_split,
                  optimizer='RMSprop', learning_rate=1e-5, weight_decay=1e-8, momentum=0.999,
                  loss='CrossEntropy + original Dice', gradient_clip=1.0,
                  augmentation=False, dtype='float32', memory_format='channels_last',
                  validation_frequency='once per epoch', device='cpu', torch=str(torch.__version__),
                  label_policy='Original trimap 1 and 3 -> foreground; 2 -> background')
    config['scheduler'] = 'ReduceLROnPlateau, mode=max, patience=5, once per epoch'
    history = []
    best_dice = -1.0
    first_epoch = 1
    elapsed_before = 0.0
    initial = None
    if args.resume:
        state = torch.load(last_path, map_location='cpu', weights_only=True)
        assert state['configuration'] == config, 'Resume configuration differs'
        model.load_state_dict(state['model_state'])
        optimizer.load_state_dict(state['optimizer_state'])
        if scheduler is not None:
            scheduler.load_state_dict(state['scheduler_state'])
        generator.set_state(state['loader_rng_state'])
        torch.set_rng_state(state['torch_rng_state'])
        history = state['history']
        initial = state['initial_validation']
        best_dice = state['best_dice']
        first_epoch = state['epoch'] + 1
        elapsed_before = state['elapsed_seconds']
    if args.check_only:
        model.eval()
        with torch.inference_mode():
            sample = train_set[0]
            prediction = model(sample['image'].unsqueeze(0).contiguous(memory_format=torch.channels_last))
        assert prediction.shape == (1, 2, 128, 128) and torch.isfinite(prediction).all()
        print(json.dumps(dict(check='passed', model=args.model, resume=args.resume,
                              next_epoch=first_epoch, parameters=sum(p.numel() for p in model.parameters()))))
        return
    started = time.perf_counter()
    print(f'RUN_START run={run_name} train={len(train_set)} val={len(val_set)} epochs={args.epochs} resume_epoch={first_epoch}', flush=True)
    if initial is None:
        initial = reporting.evaluate(model, val_loader)

    for epoch in range(first_epoch, args.epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        total = 0.0
        n = 0
        for step, batch in enumerate(train_loader, 1):
            image = batch['image'].contiguous(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            output = model(image)
            loss = reporting.loss_fn(output, batch['mask'])
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(image)
            n += len(image)
            if step == 1 or step % 10 == 0 or n == len(train_set):
                elapsed = time.perf_counter() - started + elapsed_before
                print(f'epoch={epoch} images={n}/{len(train_set)} train_loss={total/n:.5f} elapsed_min={elapsed/60:.2f}', flush=True)
                progress = dict(status='training', epoch=epoch, planned_epochs=args.epochs,
                                processed_images=n, train_images=len(train_set), history=history,
                                elapsed_seconds=elapsed, configuration=config)
                (out / 'progress.json').write_text(json.dumps(progress, indent=2), encoding='utf-8')
        training_seconds = time.perf_counter() - epoch_start
        validation_start = time.perf_counter()
        validation = reporting.evaluate(model, val_loader)
        pixels = len(val_set) * 128 * 128
        validation['predicted_foreground_fraction'] = (validation['true_positive_pixels'] + validation['false_positive_pixels']) / pixels
        validation['true_foreground_fraction'] = (validation['true_positive_pixels'] + validation['false_negative_pixels']) / pixels
        row = dict(epoch=epoch, train_loss=total/n, train_seconds=training_seconds,
                   validation_seconds=time.perf_counter()-validation_start, validation=validation)
        history.append(row)
        if scheduler is not None:
            scheduler.step(validation['foreground_dice'])
        if validation['foreground_dice'] > best_dice:
            best_dice = validation['foreground_dice']
            atomic_save(dict(model_state=model.state_dict(), epoch=epoch,
                             validation=validation, configuration=config), best_path)
        elapsed = time.perf_counter() - started + elapsed_before
        atomic_save(dict(model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
                         scheduler_state=scheduler.state_dict() if scheduler is not None else None,
                         epoch=epoch, configuration=config, history=history,
                         initial_validation=initial, best_dice=best_dice, elapsed_seconds=elapsed,
                         torch_rng_state=torch.get_rng_state(), loader_rng_state=generator.get_state()), last_path)
        (out / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        print('EPOCH_COMPLETE', json.dumps(row), flush=True)

    del optimizer, model
    if first_epoch <= args.epochs:
        del image, output, loss
    gc.collect()
    best = torch.load(best_path, map_location='cpu', weights_only=True)
    loaded = build_model(args.model).to(memory_format=torch.channels_last)
    loaded.load_state_dict(best['model_state'])
    reloaded_metrics = reporting.evaluate(loaded, val_loader)
    for key in ['foreground_iou', 'foreground_dice', 'loss']:
        assert abs(reloaded_metrics[key] - best['validation'][key]) < 1e-6, 'Checkpoint evaluation changed'
    reporting.plot_results(loaded, val_set, history)
    true_fraction = best['validation']['true_foreground_fraction']
    reference = dict(foreground_iou=true_fraction, foreground_dice=2*true_fraction/(1+true_fraction))
    summary = dict(purpose=f'Full-data experiment: {args.model}; official test not evaluated.',
                   configuration=config, planned_epochs=args.epochs, completed_epochs=len(history),
                   initial_validation=initial, history=history, best_epoch=best['epoch'],
                   best_validation=best['validation'], all_foreground_reference=reference,
                   checkpoint_reload_metrics_match=True,
                   elapsed_seconds=time.perf_counter()-started+elapsed_before,
                   test_set_evaluated=False, parameters=sum(p.numel() for p in loaded.parameters()))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (out / 'progress.json').write_text(json.dumps(dict(status='complete', **summary), indent=2), encoding='utf-8')
    print('TRAINING_COMPLETE', json.dumps(summary['best_validation']), flush=True)


if __name__ == '__main__':
    main()
