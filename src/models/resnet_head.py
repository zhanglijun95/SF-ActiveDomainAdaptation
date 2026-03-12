"""ResNet + bottleneck + classifier model."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet101, resnet50


class SafeBatchNorm1d(nn.BatchNorm1d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # BatchNorm1d cannot compute batch statistics with batch size 1 in training.
        # Fall back to running statistics for that edge case.
        if self.training and input.dim() >= 2 and input.size(0) == 1:
            return F.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                False,
                self.momentum,
                self.eps,
            )
        return super().forward(input)


class ResNetBottleneckClassifier(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        bottleneck_dim: int = 256,
        pretrained: bool = True,
        use_relu: bool = True,
    ) -> None:
        super().__init__()
        if backbone_name == "resnet50":
            backbone = resnet50(weights="IMAGENET1K_V2" if pretrained else None)
        elif backbone_name == "resnet101":
            backbone = resnet101(weights="IMAGENET1K_V2" if pretrained else None)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        layers = [nn.Linear(2048, bottleneck_dim), SafeBatchNorm1d(bottleneck_dim)]
        if use_relu:
            layers.append(nn.ReLU(inplace=True))
        self.bottleneck = nn.Sequential(*layers)
        self.classifier = nn.Linear(bottleneck_dim, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        feat_2048 = torch.flatten(self.avgpool(x), 1)
        feat_256 = self.bottleneck(feat_2048)
        logits = self.classifier(feat_256)

        if not return_features:
            return logits
        return {"logits": logits, "feat_2048": feat_2048, "feat_256": feat_256}
