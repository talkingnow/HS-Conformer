import torch.nn as nn
import torch.nn.functional as F
import torch

class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(64)

        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        self.layer1 = self._make_layer(64, 64, num_block=2, stride=2)
        self.layer2 = self._make_layer(64, 128, num_block=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_block=2, stride=2)
        self.layer4 = self._make_layer(256, 512, num_block=2, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # fst conv
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        # resblock
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x

    def _make_layer(self, in_channels, out_channels, num_block, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(BasicBlock(2, in_channels, out_channels, stride, downsample))
        for i in range(1, num_block):
            layers.append(BasicBlock(2, out_channels, out_channels))

        return nn.Sequential(*layers)