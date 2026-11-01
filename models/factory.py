# Added/adapted for this course project, 2026-11-01.
# License: GPL-3.0; upstream attribution and modification details in SOURCES.md/CHANGES.md.
from .unet import UNet
from .efficient_unet import EfficientUNet


def build_model(name):
    if name == 'upstream_unet':
        return UNet(3, 2, bilinear=False)
    if name == 'efficient_unet':
        return EfficientUNet(3, 2, bilinear=False)
    raise ValueError(f'Unknown model: {name}')
