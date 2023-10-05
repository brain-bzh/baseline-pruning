# From https://github.com/facebookresearch/ConvNeXt


import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import math


def norm_cdf(x):
    return (1. + math.erf(x / math.sqrt(2.))) / 2.


def trunc_normal(tensor, mean=0., std=1., a=-2., b=2.):
    # From https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/weight_init.py
    with torch.no_grad():
        if (mean < a - 2 * std) or (mean > b + 2 * std):
            warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                          "The distribution of values may be incorrect.",
                          stacklevel=2)
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


class Block(nn.Module):
    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + x
        return x


class ConvNeXt(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 large_input=True
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        if large_input:
            stem = nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
            )
        else:  # No initial reduction for CIFAR-10, as for ResNets
            stem = nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=3, stride=1),
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
            )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


def convnext_tiny(num_classes=1000, large_input=True, width=96):
    return ConvNeXt(depths=[3, 3, 9, 3],
                    dims=[width, width * 2, width * 4, width * 8],
                    num_classes=num_classes,
                    large_input=large_input)


def convnext_small(num_classes=1000, large_input=True, width=96):
    return ConvNeXt(depths=[3, 3, 27, 3],
                    dims=[width, width * 2, width * 4, width * 8],
                    num_classes=num_classes,
                    large_input=large_input)


def convnext_base(num_classes=1000, large_input=True, width=128):
    return ConvNeXt(depths=[3, 3, 27, 3],
                    dims=[width, width * 2, width * 4, width * 8],
                    num_classes=num_classes,
                    large_input=large_input)


def convnext_large(num_classes=1000, large_input=True, width=192):
    return ConvNeXt(depths=[3, 3, 27, 3],
                    dims=[width, width * 2, width * 4, width * 8],
                    num_classes=num_classes,
                    large_input=large_input)


def convnext_xlarge(num_classes=1000, large_input=True, width=256):
    return ConvNeXt(depths=[3, 3, 27, 3],
                    dims=[width, width * 2, width * 4, width * 8],
                    num_classes=num_classes,
                    large_input=large_input)
