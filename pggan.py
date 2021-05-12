import torch
import torch.nn as nn
import torch.nn.functional as F

import math


upscale = lambda x: F.interpolate(x, scale_factor=2, mode='nearest')
downscale = lambda x: F.avg_pool2d(x, 2, 2)

class Generator(nn.Module):

    def __init__(self, blocks, to_rgbs):
        super().__init__()

        self.blocks = nn.ModuleList(blocks)
        self.to_rgbs = nn.ModuleList(to_rgbs)

    def forward(self, latents, labels, n, alpha):
        x = torch.cat((latents, labels), -1)
        xs = []
        for i in range(n + 1):
            x = self.blocks[i](x)
            xs.append(x)

        image = self.to_rgbs[n](xs[-1])
        if n == 0 or alpha >= 1.0:
            return image
        else:
            return upscale(self.to_rgbs[n-1](xs[-2])) * (1. - alpha) + image * alpha

class Discriminator(nn.Module):

    def __init__(self, blocks, from_rgbs):
        super().__init__()

        self.blocks = nn.ModuleList(blocks)
        self.from_rgbs = nn.ModuleList(from_rgbs)

    def forward(self, image, n, alpha):
        num_block = len(self.blocks)
        block_index = num_block - n - 1

        x = self.from_rgbs[block_index](image)
        if not (n == 0 or alpha >= 1.0):
            x = self.from_rgbs[block_index + 1](downscale(image)) * (1.0 - alpha) + self.blocks[block_index](x) * alpha
            block_index += 1
        for i in range(block_index, num_block):
            x = self.blocks[i](x)
        return x

class DenseToImage(nn.Module):

    def __init__(self, h, w):
        super().__init__()

        self.h, self.w = h, w

    def forward(self, x):
        n, _ = x.shape
        return x.reshape(n, -1, self.h, self.w)


class ImageToDense(nn.Module):

    def forward(self, x):
        return x.reshape(x.size(0), -1)


class PixelNormalization(nn.Module):

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x, epsilon=1e-8):
        return x * (((x**2).mean(dim=1, keepdim=True) + self.eps).rsqrt())


class MiniBatchStdDev(nn.Module):

    def __init__(self, group_size=4):
        super().__init__()

        self.group_size = group_size

    def forward(self, x):
        n, c, h, w = x.shape
        group_size = min(self.group_size, n)
        assert n % group_size == 0, 'batch size must be divisible by {}'.format(group_size)

        y = x.reshape(group_size, -1, c, h, w) # GMCHW
        y -= torch.mean(y, axis=0, keepdims=True) #GMCHW
        y = torch.mean(torch.square(y), axis=0) # MCHW
        y = torch.sqrt(y + 1e-8) # MCHW
        y = torch.mean(y, axis=(1, 2, 3), keepdims=True) # M111
        y = y.repeat(group_size, 1, h, w)
        return torch.cat([x, y], 1)


class EqualizedWeight(object):

    def __init__(self, name, value):
        self.name = name
        self.value = value

    def compute_weight(self, module):
        w = getattr(module, self.name + '_eq')
        return w * self.value

    @staticmethod
    def apply(module, name, value):
        fn = EqualizedWeight(name, value)
        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_eq', nn.Parameter(weight.data))
        setattr(module, name, fn.compute_weight(module))
        module.register_forward_pre_hook(fn)
        return fn

    def __call__(self, module, inputs):
        setattr(module, self.name, self.compute_weight(module))

def equalize_weight(module, name='weight', value=1.0):
    EqualizedWeight.apply(module, name, value)
    return module

def initialize(module, gain=math.sqrt(2.0), use_wscale=False):
    # init bias
    module.bias.data.fill_(0)

    # init weight
    fan_in = module.weight.shape[1:].numel()
    std = gain / math.sqrt(fan_in)
    if use_wscale:
        nn.init.normal_(module.weight)
        module = equalize_weight(module, value=std)
    else:
        nn.init.normal_(module.weight, std=std)
    return module

def build_generator(
    latent_size,
    label_size,
    resolution,
    num_channels=3,
    fmap_base=8192,
    fmap_decay=1.0,
    fmap_max=512,
    normalize_latents=True,
    use_wscale=True,
    use_pixelnorm=True,
    pixelnorm_epsilon=1e-8,
    use_leakyrelu=True):

    num_scale = int(math.log2(resolution))
    assert resolution >= 4 and 2 ** num_scale == resolution

    Act = lambda inplace=False: nn.LeakyReLU(negative_slope=0.2, inplace=inplace)\
        if use_leakyrelu else nn.ReLU(inplace=inplace)

    def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
    if latent_size is None: latent_size = nf(0)

    def build_block(res):
        if res == 2: # 4x4
            nin, nh, nout = latent_size + label_size, nf(res - 1), nf(res - 1)
            block = [
                PixelNormalization(pixelnorm_epsilon) if normalize_latents else nn.Identity(),
                initialize(nn.Linear(nin, nh * 16), math.sqrt(2) / 4, use_wscale),
                Act(True),
                DenseToImage(4, 4),
                PixelNormalization(pixelnorm_epsilon) if use_pixelnorm else nn.Identity(),
                initialize(nn.Conv2d(nh, nout, 3, 1, 1), use_wscale=use_wscale),
                Act(True),
                PixelNormalization(pixelnorm_epsilon) if use_pixelnorm else nn.Identity(),
            ]
        else:
            nin, nh, nout = nf(res - 2), nf(res - 1), nf(res - 1)
            block = [
                nn.Upsample(scale_factor=2, mode='nearest'),
                initialize(nn.Conv2d(nin, nh, 3, 1, 1), use_wscale=use_wscale),
                Act(True),
                PixelNormalization(pixelnorm_epsilon) if use_pixelnorm else nn.Identity(),
                initialize(nn.Conv2d(nh, nout, 3, 1, 1), use_wscale=use_wscale),
                Act(True),
                PixelNormalization(pixelnorm_epsilon) if use_pixelnorm else nn.Identity(),
            ]
        return nn.Sequential(*list(filter(lambda m: not isinstance(m, nn.Identity), block)))


    def build_to_rgb(res):
        nin, nout = nf(res - 1), num_channels
        return initialize(nn.Conv2d(nin, nout, 1, 1, 0), 1.0, use_wscale)

    blocks = [build_block(i) for i in range(2, num_scale + 1)]
    to_rgbs = [build_to_rgb(i) for i in range(2, num_scale + 1)]
    return Generator(blocks, to_rgbs)



def build_discriminator(
    resolution,
    num_channels,
    label_size,
    fmap_base=8192,
    fmap_decay=1.0,
    fmap_max=512,
    use_wscale=True,
    mbstd_group_size=4):

    num_scale = int(math.log2(resolution))
    assert resolution >= 4 and 2 ** num_scale == resolution

    def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

    Act = lambda inplace=False: nn.LeakyReLU(negative_slope=0.2, inplace=inplace)

    def build_block(res):
        if res >= 3:
            nin, nh, nout = nf(res - 1), nf(res - 1), nf(res - 2)
            block = [
                initialize(nn.Conv2d(nin, nh, 3, 1, 1), use_wscale=use_wscale),
                Act(True),
                initialize(nn.Conv2d(nh, nout, 3, 1, 1), use_wscale=use_wscale),
                Act(True),
                nn.AvgPool2d(2, 2),
            ]
        else:
            nin = nf(res - 3) + 1 if mbstd_group_size > 1 else nf(res - 3)
            nh, nout = nf(res - 1), nf(res - 2)
            block = [
                MiniBatchStdDev(mbstd_group_size) if mbstd_group_size > 1 else nn.Identity(),
                initialize(nn.Conv2d(nin, nh, 3, 1, 1), use_wscale=use_wscale),
                Act(True),
                ImageToDense(),
                initialize(nn.Linear(nh * 16, nout), use_wscale=use_wscale),
                Act(True),
                initialize(nn.Linear(nout, 1 + label_size), gain=1, use_wscale=use_wscale)
            ]
        return nn.Sequential(*list(filter(lambda m: not isinstance(m, nn.Identity), block)))

    def build_from_rgbs(res):
        return initialize(nn.Conv2d(num_channels, nf(res - 1), 1, 1, 0), use_wscale=use_wscale)

    blocks = [build_block(i) for i in range(num_scale, 1, -1)]
    from_rgbs = [build_from_rgbs(i) for i in range(num_scale, 1, -1)]
    return Discriminator(blocks, from_rgbs)
