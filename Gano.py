""" Anonymized Speaker Generator (Gano)
"""

import torch
import torch.nn as nn
from nflows import transforms
from tools.tools import ZeroConv2d

# actnorm + conv + affineCoupling -> step
# squeeze + step * num_flow -> scale
# scale * num_scale -> Glow

class Net(nn.Module):

    def __init__(self, in_channel, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channel, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 1),
            nn.ReLU(inplace=True),
            ZeroConv2d(64, out_channels),
        )

    def forward(self, inp, context=None):
        return self.net(inp)


def getGlowStep(num_channels, crop_size, i):
    mask = [1] * num_channels
    
    if i % 2 == 0:
        mask[::2] = [-1] * (len(mask[::2]))
    else:
        mask[1::2] = [-1] * (len(mask[1::2]))

    def getNet(in_channel, out_channels):
        return Net(in_channel, out_channels)

    return transforms.CompositeTransform([
        transforms.ActNorm(num_channels),
        transforms.OneByOneConvolution(num_channels),
        transforms.coupling.AffineCouplingTransform(mask, getNet)
    ])



def getGlowScale(num_channels, num_flow, crop_size):
    z = [getGlowStep(num_channels, crop_size, i) for i in range(num_flow)]
    return transforms.CompositeTransform([
        transforms.SqueezeTransform(),
        *z
    ])


def getGLOW():
    num_channels = 1 * 4
    num_flow = 32
    num_scale = 2
    crop_size = 8 // 2
    transform = transforms.MultiscaleCompositeTransform(num_scale)
    for i in range(num_scale):
        _ = transform.add_transform(getGlowScale(num_channels, num_flow, crop_size),
                                             [num_channels, crop_size, crop_size])
        # print(f'Scale {i} input shape: {num_channels}, {crop_size}, {crop_size}')
        num_channels *= 2
        crop_size //= 2

    return transform

"""test
"""
# Gano = getGLOW()
# image = torch.randn(3,1,8,8)
# output, logabsdet = Gano(image)
# print(output.size())