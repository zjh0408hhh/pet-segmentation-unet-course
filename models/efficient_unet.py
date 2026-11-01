# Added/adapted for this course project, 2026-11-01.
# License: GPL-3.0; upstream attribution and modification details in SOURCES.md/CHANGES.md.
"""Course modification: factorize deep/decoder convolutions, keep shallow features.

Derived from milesial/Pytorch-UNet, commit 21d7850f2af30a9695bbeea75f3136aa538cfc4a.
GPL v3; see LICENSE and SOURCES.md. This is a conventional technique, not a new
algorithm claim. Quality and actual CPU speed must be measured.
"""
from torch import nn
from .unet import UNet
from .unet.unet_parts import DoubleConv


class DepthwiseSeparableConv(nn.Sequential):
    """3x3 spatial convolution per input channel, then 1x1 channel mixing."""
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        )


class EfficientUNet(UNet):
    """Keep inc/down1, skip connections, widths, BN/ReLU and upsampling unchanged."""
    def __init__(self, n_channels=3, n_classes=2, bilinear=False):
        super().__init__(n_channels, n_classes, bilinear)
        self.modified_blocks = []
        for name, block in self.named_modules():
            if not isinstance(block, DoubleConv):
                continue
            if name == 'inc' or name.startswith('down1.'):
                continue
            for position in (0, 3):
                original = block.double_conv[position]
                block.double_conv[position] = DepthwiseSeparableConv(original.in_channels, original.out_channels)
            self.modified_blocks.append(name)
