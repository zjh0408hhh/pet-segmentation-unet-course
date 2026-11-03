# Added/adapted for this course project, 2026-11-01.
# License: GPL-3.0; see 部署文档/许可与来源.txt at the project root.
# Project paths reorganized, 2026-11-03.
from .unet import UNet
from .efficient_unet import EfficientUNet


def build_model(name):
    if name == 'upstream_unet':
        return UNet(3, 2, bilinear=False)
    if name == 'efficient_unet':
        return EfficientUNet(3, 2, bilinear=False)
    raise ValueError(f'Unknown model: {name}')
