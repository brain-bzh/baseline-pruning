import copy
import math
from functools import partial

import torch
from torch import nn
from torchvision.ops import StochasticDepth


def adjust_channels(channels, width_mult, min_value=None):
    return make_divisible(channels * width_mult, 8, min_value)


class SqueezeExcitation(torch.nn.Module):
    def __init__(
            self,
            input_channels,
            squeeze_channels,
            activation=torch.nn.ReLU,
            scale_activation=torch.nn.Sigmoid,
    ):
        super().__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = torch.nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, x):
        scale = self.avgpool(x)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, x):
        scale = self._scale(x)
        return scale * x


def make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class MBConv(nn.Module):
    def __init__(
            self,
            cnf,
            stochastic_depth_prob,
            norm_layer,
            se_layer=SqueezeExcitation,
    ):
        super().__init__()

        if not (1 <= cnf['stride'] <= 2):
            print(cnf['stride'])
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf['stride'] == 1 and cnf['input_channels'] == cnf['out_channels']

        layers = []
        activation_layer = nn.SiLU

        # expand
        expanded_channels = adjust_channels(cnf['input_channels'], cnf['expand_ratio'])
        if expanded_channels != cnf['input_channels']:
            layers.extend(
                [
                    nn.Conv2d(cnf['input_channels'], expanded_channels, kernel_size=1, bias=False),
                    norm_layer(expanded_channels),
                    activation_layer()
                ]
            )

        # depthwise
        layers.extend(
            [
                nn.Conv2d(expanded_channels,
                          expanded_channels,
                          kernel_size=cnf['kernel'],
                          stride=cnf['stride'],
                          padding=int((cnf['kernel'] - 1) / 2),
                          groups=expanded_channels,
                          bias=False),
                norm_layer(expanded_channels),
                activation_layer()
            ]
        )

        # squeeze and excitation
        squeeze_channels = max(1, cnf['input_channels'] // 4)
        layers.append(se_layer(expanded_channels, squeeze_channels, activation=partial(nn.SiLU, inplace=True)))

        # project
        layers.extend(
            [
                nn.Conv2d(expanded_channels, cnf['out_channels'], kernel_size=1, bias=False),
                norm_layer(cnf['out_channels'])
            ]
        )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf['out_channels']

    def forward(self, x):
        result = self.block(x)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += x
        return result


class FusedMBConv(nn.Module):
    def __init__(
            self,
            cnf,
            stochastic_depth_prob,
            norm_layer,
    ):
        super().__init__()

        if not (1 <= cnf['stride'] <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf['stride'] == 1 and cnf['input_channels'] == cnf['out_channels']

        layers = []
        activation_layer = nn.SiLU

        expanded_channels = adjust_channels(cnf['input_channels'], cnf['expand_ratio'])
        if expanded_channels != cnf['input_channels']:
            # fused expand
            layers.extend(
                [
                    nn.Conv2d(cnf['input_channels'],
                              expanded_channels,
                              kernel_size=cnf['kernel'],
                              padding=int((cnf['kernel'] - 1) / 2),
                              stride=cnf['stride'],
                              bias=False),
                    norm_layer(expanded_channels),
                    activation_layer()
                ]
            )

            # project
            layers.extend(
                [
                    nn.Conv2d(expanded_channels, cnf['out_channels'], kernel_size=1, bias=False),
                    norm_layer(cnf['out_channels'])
                ]
            )
        else:
            layers.extend(
                [
                    nn.Conv2d(cnf['input_channels'],
                              cnf['out_channels'],
                              kernel_size=cnf['kernel'],
                              padding=int((cnf['kernel'] - 1) / 2),
                              stride=cnf['stride'],
                              bias=False),
                    norm_layer(expanded_channels),
                    activation_layer()
                ]
            )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf['out_channels']

    def forward(self, x):
        result = self.block(x)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += x
        return result


class EfficientNet(nn.Module):
    def __init__(
            self,
            inverted_residual_setting,
            dropout,
            stochastic_depth_prob=0.2,
            num_classes=1000,
            norm_layer=None,
            last_channel=None,
            large_input=True
    ):
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0]['input_channels']
        if large_input:
            layers.extend(
                [
                    nn.Conv2d(3, firstconv_output_channels, kernel_size=3, stride=2, padding=1, bias=False),
                    norm_layer(firstconv_output_channels),
                    nn.SiLU()
                ]
            )
        else:
            layers.extend(
                [
                    nn.Conv2d(3, firstconv_output_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    norm_layer(firstconv_output_channels),
                    nn.SiLU()
                ]
            )

        # building inverted residual blocks
        total_stage_blocks = sum(cnf['num_layers'] for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage = []
            for _ in range(cnf['num_layers']):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf['input_channels'] = block_cnf['out_channels']
                    block_cnf['stride'] = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                stage.append(block_cnf['block'](block_cnf, sd_prob, norm_layer))
                stage_block_id += 1

            layers.append(nn.Sequential(*stage))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1]['out_channels']
        lastconv_output_channels = last_channel if last_channel is not None else 4 * lastconv_input_channels
        layers.extend(
            [
                nn.Conv2d(lastconv_input_channels, lastconv_output_channels, kernel_size=1, bias=False),
                norm_layer(lastconv_output_channels),
                nn.SiLU()
            ]
        )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(lastconv_output_channels, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def efficientnet_b0(num_classes=1000, large_input=True, width=8, dropout=0.2):
    config = [
        {
            'expand_ratio': 1, 'kernel': 3, 'stride': 1, 'input_channels': width * 4, 'out_channels': width * 2,
            'num_layers': 1, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 3, 'stride': 2, 'input_channels': width * 2, 'out_channels': width * 3,
            'num_layers': 2, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 5, 'stride': 2, 'input_channels': width * 3, 'out_channels': width * 5,
            'num_layers': 2, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 3, 'stride': 2, 'input_channels': width * 5, 'out_channels': width * 10,
            'num_layers': 3, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 5, 'stride': 1, 'input_channels': width * 10, 'out_channels': width * 14,
            'num_layers': 3, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 5, 'stride': 2, 'input_channels': width * 14, 'out_channels': width * 24,
            'num_layers': 4, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 3, 'stride': 1, 'input_channels': width * 24, 'out_channels': width * 40,
            'num_layers': 1, 'block': MBConv},
    ]
    return EfficientNet(config, dropout=dropout, last_channel=None, num_classes=num_classes, large_input=large_input)


def efficientnet_b1(num_classes=1000, large_input=True, width=8, dropout=0.2):
    config = [
        {
            'expand_ratio': 1, 'kernel': 3, 'stride': 1, 'input_channels': width * 4, 'out_channels': width * 2,
            'num_layers': 2, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 3, 'stride': 2, 'input_channels': width * 2, 'out_channels': width * 3,
            'num_layers': 3, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 5, 'stride': 2, 'input_channels': width * 3, 'out_channels': width * 5,
            'num_layers': 3, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 3, 'stride': 2, 'input_channels': width * 5, 'out_channels': width * 10,
            'num_layers': 4, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 5, 'stride': 1, 'input_channels': width * 10, 'out_channels': width * 14,
            'num_layers': 4, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 5, 'stride': 2, 'input_channels': width * 14, 'out_channels': width * 24,
            'num_layers': 5, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 3, 'stride': 1, 'input_channels': width * 24, 'out_channels': width * 40,
            'num_layers': 2, 'block': MBConv},
    ]

    return EfficientNet(config, dropout=dropout, last_channel=None, num_classes=num_classes, large_input=large_input)


def efficientnet_b2(num_classes=1000, large_input=True, width=8, dropout=0.3):
    config = [
        {
            'expand_ratio': 1, 'kernel': 3, 'stride': 1, 'input_channels': width * 4, 'out_channels': width * 2,
            'num_layers': 2, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 3, 'stride': 2, 'input_channels': width * 2, 'out_channels': width * 3,
            'num_layers': 3, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 5, 'stride': 2, 'input_channels': width * 3, 'out_channels': width * 6,
            'num_layers': 3, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 3, 'stride': 2, 'input_channels': width * 6, 'out_channels': width * 11,
            'num_layers': 4, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 5, 'stride': 1, 'input_channels': width * 11, 'out_channels': width * 15,
            'num_layers': 4, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 5, 'stride': 2, 'input_channels': width * 15, 'out_channels': width * 26,
            'num_layers': 5, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 3, 'stride': 1, 'input_channels': width * 26, 'out_channels': width * 44,
            'num_layers': 2, 'block': MBConv},
    ]
    return EfficientNet(config, dropout=dropout, last_channel=None, num_classes=num_classes, large_input=large_input)


def efficientnet_b3(num_classes=1000, large_input=True, width=8, dropout=0.3):
    config = [
        {
            'expand_ratio': 1, 'kernel': 3, 'stride': 1, 'input_channels': width * 5, 'out_channels': width * 3,
            'num_layers': 2, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 3, 'stride': 2, 'input_channels': width * 3, 'out_channels': width * 4,
            'num_layers': 3, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 5, 'stride': 2, 'input_channels': width * 4, 'out_channels': width * 6,
            'num_layers': 3, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 3, 'stride': 2, 'input_channels': width * 6, 'out_channels': width * 12,
            'num_layers': 5, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 5, 'stride': 1, 'input_channels': width * 12, 'out_channels': width * 17,
            'num_layers': 5, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 5, 'stride': 2, 'input_channels': width * 17, 'out_channels': width * 29,
            'num_layers': 6, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 3, 'stride': 1, 'input_channels': width * 29, 'out_channels': width * 48,
            'num_layers': 2, 'block': MBConv},
    ]
    return EfficientNet(config, dropout=dropout, last_channel=None, num_classes=num_classes, large_input=large_input)


def efficientnet_b4(num_classes=1000, large_input=True, width=8, dropout=0.4):
    config = [
        {
            'expand_ratio': 1, 'kernel': 3, 'stride': 1, 'input_channels': width * 6, 'out_channels': width * 3,
            'num_layers': 2, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 3, 'stride': 2, 'input_channels': width * 3, 'out_channels': width * 4,
            'num_layers': 4, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 5, 'stride': 2, 'input_channels': width * 4, 'out_channels': width * 7,
            'num_layers': 4, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 3, 'stride': 2, 'input_channels': width * 7, 'out_channels': width * 14,
            'num_layers': 6, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 5, 'stride': 1, 'input_channels': width * 14, 'out_channels': width * 20,
            'num_layers': 6, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 5, 'stride': 2, 'input_channels': width * 20, 'out_channels': width * 34,
            'num_layers': 8, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 3, 'stride': 1, 'input_channels': width * 34, 'out_channels': width * 56,
            'num_layers': 2, 'block': MBConv},
    ]
    return EfficientNet(config, dropout=dropout, last_channel=None, num_classes=num_classes, large_input=large_input)


def efficientnet_b5(num_classes=1000, large_input=True, width=8, dropout=0.4):
    config = [
        {
            'expand_ratio': 1, 'kernel': 3, 'stride': 1, 'input_channels': width * 6, 'out_channels': width * 3,
            'num_layers': 3, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 3, 'stride': 2, 'input_channels': width * 3, 'out_channels': width * 5,
            'num_layers': 5, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 5, 'stride': 2, 'input_channels': width * 5, 'out_channels': width * 8,
            'num_layers': 5, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 3, 'stride': 2, 'input_channels': width * 8, 'out_channels': width * 16,
            'num_layers': 7, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 5, 'stride': 1, 'input_channels': width * 16, 'out_channels': width * 22,
            'num_layers': 7, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 5, 'stride': 2, 'input_channels': width * 22, 'out_channels': width * 38,
            'num_layers': 9, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 3, 'stride': 1, 'input_channels': width * 38, 'out_channels': width * 64,
            'num_layers': 3, 'block': MBConv},
    ]

    return EfficientNet(config, dropout=dropout, last_channel=None,
                        norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01), num_classes=num_classes,
                        large_input=large_input)


def efficientnet_b6(num_classes=1000, large_input=True, width=8, dropout=0.5):
    config = [
        {
            'expand_ratio': 1, 'kernel': 3, 'stride': 1, 'input_channels': width * 7, 'out_channels': width * 4,
            'num_layers': 3, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 3, 'stride': 2, 'input_channels': width * 4, 'out_channels': width * 5,
            'num_layers': 6, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 5, 'stride': 2, 'input_channels': width * 5, 'out_channels': width * 9,
            'num_layers': 6, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 3, 'stride': 2, 'input_channels': width * 9, 'out_channels': width * 18,
            'num_layers': 8, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 5, 'stride': 1, 'input_channels': width * 18, 'out_channels': width * 25,
            'num_layers': 8, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 5, 'stride': 2, 'input_channels': width * 25, 'out_channels': width * 43,
            'num_layers': 11, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 3, 'stride': 1, 'input_channels': width * 43, 'out_channels': width * 72,
            'num_layers': 3, 'block': MBConv},
    ]
    return EfficientNet(config, dropout=dropout, last_channel=None,
                        norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01), num_classes=num_classes,
                        large_input=large_input)


def efficientnet_b7(num_classes=1000, large_input=True, width=8, dropout=0.5):
    config = [
        {
            'expand_ratio': 1, 'kernel': 3, 'stride': 1, 'input_channels': width * 8, 'out_channels': width * 4,
            'num_layers': 4, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 3, 'stride': 2, 'input_channels': width * 4, 'out_channels': width * 6,
            'num_layers': 7, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 5, 'stride': 2, 'input_channels': width * 6, 'out_channels': width * 10,
            'num_layers': 7, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 3, 'stride': 2, 'input_channels': width * 10, 'out_channels': width * 20,
            'num_layers': 10, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 5, 'stride': 1, 'input_channels': width * 20, 'out_channels': width * 28,
            'num_layers': 10, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 5, 'stride': 2, 'input_channels': width * 28, 'out_channels': width * 48,
            'num_layers': 13, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 3, 'stride': 1, 'input_channels': width * 48, 'out_channels': width * 80,
            'num_layers': 4, 'block': MBConv},
    ]
    return EfficientNet(config, dropout=dropout, last_channel=None,
                        norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01), num_classes=num_classes,
                        large_input=large_input)


def efficientnet_v2_s(num_classes=1000, large_input=True, width=8, dropout=0.2):
    config = [
        {
            'expand_ratio': 1, 'kernel': 3, 'stride': 1, 'input_channels': width * 3, 'out_channels': width * 3,
            'num_layers': 2, 'block': FusedMBConv},
        {
            'expand_ratio': 4, 'kernel': 3, 'stride': 2, 'input_channels': width * 3, 'out_channels': width * 6,
            'num_layers': 4, 'block': FusedMBConv},
        {
            'expand_ratio': 4, 'kernel': 3, 'stride': 2, 'input_channels': width * 6, 'out_channels': width * 8,
            'num_layers': 4, 'block': FusedMBConv},
        {
            'expand_ratio': 4, 'kernel': 3, 'stride': 2, 'input_channels': width * 8, 'out_channels': width * 16,
            'num_layers': 6, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 3, 'stride': 1, 'input_channels': width * 16, 'out_channels': width * 20,
            'num_layers': 9, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 3, 'stride': 2, 'input_channels': width * 20, 'out_channels': width * 32,
            'num_layers': 15, 'block': MBConv},
    ]

    return EfficientNet(config, dropout=dropout, last_channel=width * 160,
                        norm_layer=partial(nn.BatchNorm2d, eps=1e-03), num_classes=num_classes, large_input=large_input)


def efficientnet_v2_m(num_classes=1000, large_input=True, width=8, dropout=0.3):
    config = [
        {
            'expand_ratio': 1, 'kernel': 3, 'stride': 1, 'input_channels': width * 3, 'out_channels': width * 3,
            'num_layers': 3, 'block': FusedMBConv},
        {
            'expand_ratio': 4, 'kernel': 3, 'stride': 2, 'input_channels': width * 3, 'out_channels': width * 6,
            'num_layers': 5, 'block': FusedMBConv},
        {
            'expand_ratio': 4, 'kernel': 3, 'stride': 2, 'input_channels': width * 6, 'out_channels': width * 10,
            'num_layers': 5, 'block': FusedMBConv},
        {
            'expand_ratio': 4, 'kernel': 3, 'stride': 2, 'input_channels': width * 10, 'out_channels': width * 20,
            'num_layers': 7, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 3, 'stride': 1, 'input_channels': width * 20, 'out_channels': width * 22,
            'num_layers': 14, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 3, 'stride': 2, 'input_channels': width * 22, 'out_channels': width * 38,
            'num_layers': 18, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 3, 'stride': 1, 'input_channels': width * 38, 'out_channels': width * 64,
            'num_layers': 5, 'block': MBConv},
    ]

    return EfficientNet(config, dropout=dropout, last_channel=width * 160,
                        norm_layer=partial(nn.BatchNorm2d, eps=1e-03), num_classes=num_classes, large_input=large_input)


def efficientnet_v2_l(num_classes=1000, large_input=True, width=8, dropout=0.4):
    config = [
        {
            'expand_ratio': 1, 'kernel': 3, 'stride': 1, 'input_channels': width * 4, 'out_channels': width * 4,
            'num_layers': 4, 'block': FusedMBConv},
        {
            'expand_ratio': 4, 'kernel': 3, 'stride': 2, 'input_channels': width * 4, 'out_channels': width * 8,
            'num_layers': 7, 'block': FusedMBConv},
        {
            'expand_ratio': 4, 'kernel': 3, 'stride': 2, 'input_channels': width * 8, 'out_channels': width * 12,
            'num_layers': 7, 'block': FusedMBConv},
        {
            'expand_ratio': 4, 'kernel': 3, 'stride': 2, 'input_channels': width * 12, 'out_channels': width * 24,
            'num_layers': 10, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 3, 'stride': 1, 'input_channels': width * 24, 'out_channels': width * 28,
            'num_layers': 19, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 3, 'stride': 2, 'input_channels': width * 28, 'out_channels': width * 48,
            'num_layers': 25, 'block': MBConv},
        {
            'expand_ratio': 6, 'kernel': 3, 'stride': 1, 'input_channels': width * 48, 'out_channels': width * 80,
            'num_layers': 7, 'block': MBConv},
    ]

    return EfficientNet(config, dropout=dropout, last_channel=width * 160,
                        norm_layer=partial(nn.BatchNorm2d, eps=1e-03), num_classes=num_classes, large_input=large_input)
