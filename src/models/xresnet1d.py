"""
Self-contained 1D XResNet for 12-lead ECG (Decision D5).

A faithful 1D adaptation of the XResNet ("bag-of-tricks" ResNet) used by the
PTB-XL benchmark: a 3-conv stem, bottleneck blocks, and ResNet-D style
downsampling (AvgPool + 1x1 conv in the shortcut). Implemented here rather than
imported so the frozen model is fully reproducible from this repo alone.

`xresnet1d50` and `xresnet1d101` differ only in block depth, matching the 2D
ResNet-50/101 layer configs. The final conv stage (`.features[-1]`) is the
Grad-CAM target layer (Phase 3).
"""

from __future__ import annotations

import torch
import torch.nn as nn

_LAYERS = {
    "xresnet1d50": (3, 4, 6, 3),
    "xresnet1d101": (3, 4, 23, 3),
}


def _conv(ni, nf, ks=3, stride=1):
    return nn.Conv1d(ni, nf, ks, stride=stride, padding=ks // 2, bias=False)


def _conv_bn(ni, nf, ks=3, stride=1, act=True, zero_bn=False):
    bn = nn.BatchNorm1d(nf)
    nn.init.constant_(bn.weight, 0.0 if zero_bn else 1.0)   # zero-init last BN in a block
    layers = [_conv(ni, nf, ks, stride), bn]
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, ni, nf, stride=1):
        super().__init__()
        # nf is the "bottleneck width"; output channels = nf * expansion.
        self.convs = nn.Sequential(
            _conv_bn(ni, nf, ks=1),
            _conv_bn(nf, nf, ks=3, stride=stride),
            _conv_bn(nf, nf * self.expansion, ks=1, act=False, zero_bn=True),
        )
        # ResNet-D shortcut: AvgPool for stride, then 1x1 conv to match channels.
        idlayers = []
        if stride != 1:
            idlayers.append(nn.AvgPool1d(2, stride=stride, ceil_mode=True))
        if ni != nf * self.expansion:
            idlayers.append(_conv_bn(ni, nf * self.expansion, ks=1, act=False))
        self.idconv = nn.Sequential(*idlayers) if idlayers else None
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x if self.idconv is None else self.idconv(x)
        return self.act(self.convs(x) + identity)


class XResNet1d(nn.Module):
    def __init__(self, layers, n_in=12, n_classes=1, stem_width=64):
        super().__init__()
        # 3-conv stem (the XResNet trick vs. a single 7x7 stem).
        stem = nn.Sequential(
            _conv_bn(n_in, stem_width // 2, ks=7, stride=2),
            _conv_bn(stem_width // 2, stem_width // 2, ks=3),
            _conv_bn(stem_width // 2, stem_width, ks=3),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        blocks, ni, widths = [], stem_width, (64, 128, 256, 512)
        for i, (n_blocks, width) in enumerate(zip(layers, widths)):
            for b in range(n_blocks):
                stride = 2 if (b == 0 and i > 0) else 1
                blocks.append(Bottleneck(ni, width, stride))
                ni = width * Bottleneck.expansion
        # `features` is the full conv trunk; its last module is the Grad-CAM layer.
        self.features = nn.Sequential(stem, *blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(ni, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))

    @property
    def gradcam_layer(self) -> nn.Module:
        """Final conv stage — the layer Grad-CAM/Grad-CAM++ hook onto."""
        return self.features[-1]


def get_model(name: str = "xresnet1d101", n_in: int = 12, n_classes: int = 1) -> XResNet1d:
    if name not in _LAYERS:
        raise ValueError(f"unknown model {name!r}; choose from {list(_LAYERS)}")
    return XResNet1d(_LAYERS[name], n_in=n_in, n_classes=n_classes)
