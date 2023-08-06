import math
import torch
import torch.nn as nn

class ECAPA_TDNN(nn.Module):
    def __init__(self, input_size, embedding_size, channel_size = 1024):
        super(ECAPA_TDNN, self).__init__()

        C = channel_size
        self.conv1 = nn.Conv1d(input_size, C, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(C)

        self.layer1 = Bottle2neck(C, C, 3, 2, 8)
        self.layer2 = Bottle2neck(C, C, 3, 3, 8)
        self.layer3 = Bottle2neck(C, C, 3, 4, 8)
        self.layer4 = nn.Conv1d(3 * C, (3 * C) // 2, kernel_size=1)

    def forward(self, x):
        # ecapa
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)

        x = torch.cat((x1, x2, x3), dim=1)

        x = self.layer4(x)
        x = self.relu(x)

        return x
    
class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.se(input)
        return input * x

class Bottle2neck(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, dilation, scale):
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1

        bns = []
        convs = []
        num_pad = math.floor(kernel_size / 2) * dilation
        for _ in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.se = SEModule(planes)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x_split = torch.split(x, self.width, 1)
        for i in range(self.nums):
            sp = x_split[i] if i == 0 else sp + x_split[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            x = sp if i == 0 else torch.cat((x, sp), 1)
        x = torch.cat((x, x_split[self.nums]), 1)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.bn3(x)
        
        x = self.se(x)
        x += identity
        return x 