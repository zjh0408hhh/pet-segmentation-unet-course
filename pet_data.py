# Added/adapted for this course project, 2026-11-01.
# License: GPL-3.0; upstream attribution and modification details in SOURCES.md/CHANGES.md.
"""Oxford trimaps adapted to binary pet segmentation; no augmentation in baseline."""
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class PetDataset(Dataset):
    def __init__(self, root, split_file, size=128):
        self.root = Path(root)
        self.names = Path(split_file).read_text(encoding='utf-8').splitlines()
        self.size = size

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        with Image.open(self.root / 'images' / f'{name}.jpg') as source:
            image = source.convert('RGB').resize((self.size, self.size), Image.Resampling.BICUBIC)
            image = np.asarray(image, dtype=np.float32) / 255.0
        with Image.open(self.root / 'annotations' / 'trimaps' / f'{name}.png') as source:
            mask = np.asarray(source.resize((self.size, self.size), Image.Resampling.NEAREST))
        assert set(np.unique(mask)).issubset({1, 2, 3})
        target = ((mask == 1) | (mask == 3)).astype(np.int64)
        return dict(image=torch.from_numpy(image.transpose(2, 0, 1).copy()),
                    mask=torch.from_numpy(target.copy()), name=name)
