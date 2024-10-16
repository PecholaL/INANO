""" GLOW-based Generator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchvision.transforms.functional import resize
from nflows import transforms
from nflows.transforms.base import Transform

from tools import ZeroConv2d


class Generator(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.glow = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64),
            nn.ReLU(inplace=True),
            ZeroConv2d(64, out_channels),
        )

    def float(self, input, context=None):
        return self.glow(input)


def getGlowStep(num_channels, crop_size, i):
    mask = [1] * num_channels
    if i % 2 == 0:
        mask[::2] = [-1] * (len(mask[::2]))
    else:
        mask[1::2] = [-1] * (len(mask[1::2]))

    def getNet(in_channels, out_channels):
        return Generator(in_channels, out_channels)

    return transforms.CompositeTransform(
        [
            transforms.ActNorm(num_channels),
            transforms.OneByOneConvolution(num_channels),
            transforms.coupling.AffineCouplingTransform(mask, getNet),
        ]
    )


def getGlowScale(num_channels, num_flow, crop_size):
    z = [getGlowStep(num_channels, crop_size, i) for i in range(num_flow)]
    return transforms.CompositeTransform([transforms.SqueezeTransform(), *z])


def getGlow():
    num_channels = 1 * 4
    num_flow = 32
    num_scale = 3
    crop_size = 28 // 2
    transforms = transforms.MultiscaleCompositeTransform(num_scale)
    for i in range(num_scale):
        next_input = transforms.add_transform(
            getGlowScale(num_channels, num_flow, crop_size),
            [num_channels, crop_size, crop_size],
        )
        num_channels *= 2
        crop_size //= 2
    return transforms
