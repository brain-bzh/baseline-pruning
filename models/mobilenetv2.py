# From https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py
import torch
from torch import nn

__all__ = ["MobileNetV2", "mobilenet_v2"]


# necessary for backwards compatibility
class InvertedResidual(nn.Module):
    def __init__(
            self, inp: int, oup: int, stride: int, expand_ratio: int):
        super().__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.extend(
                [
                    nn.Conv2d(inp, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6()
                ]
            )
        layers.extend(
            [
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim,
                          bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(
            self,
            num_classes: int = 1000,
            base_width=8,
            dropout: float = 0.2,
            large_input=True
    ):
        super().__init__()

        input_channel = base_width * 4
        last_channel = base_width * 160

        # t = expand ratio
        # c = channels
        # n = number of blocks
        # s = stride
        inverted_residual_setting = [
            # t, c, n, s
            [1, base_width * 2, 1, 1],
            [6, base_width * 3, 2, 2],
            [6, base_width * 4, 3, 2],
            [6, base_width * 8, 4, 2],
            [6, base_width * 12, 3, 1],
            [6, base_width * 20, 3, 2],
            [6, base_width * 40, 1, 1],
        ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError(
                f"inverted_residual_setting should be non-empty or a 4-element list, got {inverted_residual_setting}"
            )

        # building first layer
        self.last_channel = last_channel
        if large_input:
            features = [
                nn.Conv2d(3, input_channel, stride=2, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(input_channel),
                nn.ReLU6()
            ]
        else:
            features = [
                nn.Conv2d(3, input_channel, stride=1, kernel_size=3, padding=0, bias=False),
                nn.BatchNorm2d(input_channel),
                nn.ReLU6()
            ]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = c
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.extend(
            [
                nn.Conv2d(input_channel, self.last_channel, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.last_channel),
                nn.ReLU6()
            ]
        )
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def mobilenet_v2(num_classes=1000, large_input=True, width=8, dropout=0.2):
    return MobileNetV2(num_classes=num_classes, base_width=width, dropout=dropout, large_input=large_input)
