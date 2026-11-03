# Added/adapted for this course project, 2026-11-01.
# License: GPL-3.0; see 部署文档/许可与来源.txt at the project root.
# Project paths reorganized, 2026-11-03.
"""Verify official archives, safely unpack files and fix disjoint dataset splits."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import shutil
import tarfile

from PIL import Image

EXPECTED = {'images.tar.gz': '5c4f3ee8e5d25df40f4fd59a7f44e54c',
            'annotations.tar.gz': '95a8c909bbe2e81eed6a22bccdf3f68f'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archives', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2026, choices=[2026])
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    root = args.data_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, expected in EXPECTED.items():
        archive = args.archives / name
        digest = hashlib.md5()
        with archive.open('rb') as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                digest.update(block)
        hashes[name] = digest.hexdigest()
        if hashes[name] != expected:
            raise RuntimeError(f'Official MD5 mismatch: {name}')
        print(f'Official MD5 verified: {name}', flush=True)
        marker = root / f'.{name}.verified'
        if marker.exists() and marker.read_text().strip() == expected:
            print(f'Extraction already completed: {name}', flush=True)
            continue
        count = 0
        with tarfile.open(archive, 'r:gz') as tar:
            for member in tar:
                if not member.isfile():
                    continue
                relative = Path(member.name)
                destination = (root / relative).resolve()
                if relative.is_absolute() or not destination.is_relative_to(root):
                    raise RuntimeError(f'Unsafe archive path: {member.name}')
                if not relative.parts or relative.parts[0] not in {'images', 'annotations'}:
                    raise RuntimeError(f'Unexpected archive path: {member.name}')
                destination.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, destination.open('wb') as output:
                    shutil.copyfileobj(source, output)
                count += 1
                if count % 2000 == 0:
                    print(f'Extracted {count} files from {name}', flush=True)
        marker.write_text(expected, encoding='utf-8')
        print(f'Extracted {name}: {count} files', flush=True)

    def read_official(split):
        return [line.split() for line in (root / 'annotations' / f'{split}.txt').read_text().splitlines() if line.strip()]

    trainval_rows = read_official('trainval')
    test_rows = read_official('test')
    assert len(trainval_rows) == 3680 and len(test_rows) == 3669
    official_names = [row[0] for row in trainval_rows + test_rows]
    assert len(set(official_names)) == len(official_names), 'Duplicate or overlapping official split IDs'
    for i, name in enumerate(official_names, 1):
        image_path = root / 'images' / f'{name}.jpg'
        mask_path = root / 'annotations' / 'trimaps' / f'{name}.png'
        with Image.open(image_path) as image:
            size = image.size
            image.verify()
        with Image.open(mask_path) as mask:
            assert mask.size == size, f'Size mismatch: {name}'
            lo, hi = mask.getextrema()
            assert 1 <= lo <= hi <= 3, f'Unexpected mask values: {name}'
        if i % 2000 == 0:
            print(f'Checked {i}/{len(official_names)} image/mask pairs', flush=True)

    groups = defaultdict(list)
    for row in trainval_rows:
        groups[int(row[1])].append(row[0])
    train, val = [], []
    for label, names in sorted(groups.items()):
        names = sorted(names)
        random.Random(args.seed + label).shuffle(names)
        n_val = max(1, round(len(names) * 0.2))
        val.extend(names[:n_val])
        train.extend(names[n_val:])
    splits = dict(train=sorted(train), val=sorted(val),
                  test=sorted(row[0] for row in test_rows))
    assert not (set(train) & set(val) or set(train) & set(splits['test']) or set(val) & set(splits['test']))
    split_dir = project / '部署文档' / '数据划分'
    split_dir.mkdir(parents=True, exist_ok=True)
    for split, names in splits.items():
        (split_dir / f'{split}.txt').write_text('\n'.join(names) + '\n', encoding='utf-8')
    official_name_set = set(official_names)
    manifest = dict(dataset='Oxford-IIIT Pet', official_home='https://www.robots.ox.ac.uk/~vgg/data/pets/',
                    archives_md5=hashes, seed=args.seed, validation_fraction=0.2,
                    validation_method='Stratified by breed within official trainval; official test unchanged.',
                    counts={k: len(v) for k, v in splits.items()}, checked_pairs=len(official_names),
                    breeds=len(groups), label_mapping={'1': 'pet -> 1', '2': 'background -> 0', '3': 'border -> 1'},
                    mask_policy='Binary pet-vs-background, border merged into foreground; fixed for all variants.',
                    excluded_from_official_protocol=sorted(p.stem for p in (root / 'images').glob('*.jpg') if p.stem not in official_name_set))
    result_dir = project / '结果' / '新实验' / '数据检查'
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / '数据检查.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest['counts']), flush=True)
    print('Data prepared and fixed splits saved.', flush=True)


if __name__ == '__main__':
    main()
