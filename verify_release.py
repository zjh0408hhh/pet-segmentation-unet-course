# Added/adapted for this course project, 2026-11-01.
# License: GPL-3.0; upstream attribution and modification details in SOURCES.md/CHANGES.md.
"""Standard-library verification of sources, fixed splits and archived evidence."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8'))


def main():
    manifest = read('MANIFEST.json')
    for name, expected in manifest['files'].items():
        actual = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'File differs from release manifest: {name}')
    for row in read('results/source_audit.json'):
        assert hashlib.sha256((ROOT/row['file']).read_bytes()).hexdigest() == row['sha256'], row['file']
    splits = {name: (ROOT/'splits'/f'{name}.txt').read_text().splitlines() for name in ('train','val','test')}
    assert [len(splits[n]) for n in ('train','val','test')] == [2944,736,3669]
    assert all(len(set(values))==len(values) for values in splits.values())
    assert not (set(splits['train']) & set(splits['val']) or set(splits['train']) & set(splits['test']) or set(splits['val']) & set(splits['test']))
    runs={name:read(f'results/{name}/summary.json') for name in ('baseline','improved')}
    assert [{k:v for k,v in s['configuration'].items() if k!='model'} for s in runs.values()][0] == [{k:v for k,v in s['configuration'].items() if k!='model'} for s in runs.values()][1]
    for summary in runs.values():
        assert summary['completed_epochs']==10 and len(summary['history'])==10
        assert max(summary['history'],key=lambda r:r['validation']['foreground_dice'])['epoch']==summary['best_epoch']
    final=read('results/comparison/test_comparison.json')
    assert final['test_split_sha256']==hashlib.sha256((ROOT/'splits/test.txt').read_bytes()).hexdigest()
    weights=read('checkpoints/manifest.json')
    for name, row in final['runs'].items():
        assert row['checkpoint_sha256']==weights[name]['sha256']
        assert row['best_epoch']==runs[name]['best_epoch']==weights[name]['best_epoch']
        t=row['test']
        assert t['images']==3669
        tp,fp,fn=[t[k] for k in ('true_positive_pixels','false_positive_pixels','false_negative_pixels')]
        assert abs(t['foreground_iou']-tp/(tp+fp+fn))<1e-12
        assert abs(t['foreground_dice']-2*tp/(2*tp+fp+fn))<1e-12
    with (ROOT/'results/comparison/per_image_test.csv').open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    assert [r['image'] for r in rows]==splits['test']
    delta=[float(r['improved_iou'])-float(r['baseline_iou']) for r in rows]
    cases=read('results/comparison/case_analysis.json')
    assert sum(d>1e-8 for d in delta)==cases['counts']['images_improved']
    assert sum(d<-1e-8 for d in delta)==cases['counts']['images_regressed']
    assert sum(abs(d)<=1e-8 for d in delta)==cases['counts']['images_tied']
    print(json.dumps(dict(status='passed', verified_files=len(manifest['files']),
                          upstream_files_unchanged=True, split_counts={n:len(v) for n,v in splits.items()},
                          checkpoints_traceable=True, test_metrics_verified=True),indent=2))


if __name__=='__main__':
    main()
