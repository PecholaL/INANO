""" GLOW-based Generator
"""

import torch.nn as nn
from nflows import transforms
from tools import ZeroConv2d


# actnorm + conv + affineCoupling -> step
# squeeze + step * num_flow -> scale
# scale * num_scale -> glow


class Subnet(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=64):
        super(Subnet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 1),
            nn.ReLU(inplace=True),
            ZeroConv2d(mid_channels, out_channels),
        )

    def float(self, input, context=None):
        return self.net(input)


# actnorm+conv+affineCoupling
def getGlowStep(channels, i):
    mask = [1] * channels  # for affineCoupling
    if i % 2 == 0:
        mask[::2] = [-1] * (len(mask[::2]))  # mask = [-1, 1, ..., -1, 1]
    else:
        mask[1::2] = [-1] * (len(mask[1::2]))  # mask = [1, -1, ..., 1, -1]

    def getSubnet(in_channels, out_channels):
        return Subnet(in_channels, out_channels)

    return transforms.CompositeTransform(
        [
            transforms.ActNorm(channels),
            transforms.OneByOneConvolution(channels),
            transforms.coupling.AffineCouplingTransform(mask, getSubnet),
        ]
    )


def getGlowScale(channels, num_flow):
    z = [getGlowStep(channels, i) for i in range(num_flow)]
    return transforms.CompositeTransform([transforms.SqueezeTransform(), *z])


def getFlow(num_channels, num_flow, num_scale, crop_size):
    # num_channels = 1 * 4, num_flow = 32, num_scale = 3, crop_size = 28 // 2
    transforms = transforms.MultiscaleCompositeTransform(num_scale)
    for i in range(num_scale):
        next_input = transforms.add_transform(
            getGlowScale(num_channels, num_flow),
            [num_channels, crop_size, crop_size],
        )
        num_channels *= 2
        crop_size //= 2
    return transforms
