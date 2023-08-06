import torch
import torch.nn as nn

"""
Code from https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/tree/master
"""

class LCNN(nn.Module):
    def __init__(self, channels=[64,64,96,96,128,128,64,64,64], dropout=0.7):
        super(LCNN, self).__init__()
        
        self.m_transform = nn.Sequential(
            nn.Conv2d(1, channels[0], [5, 5], 1, padding=[2, 2]),
            MaxFeatureMap2D(),
            torch.nn.MaxPool2d([2, 2], [2, 2]),

            nn.Conv2d(channels[0] // 2, channels[1], [1, 1], 1, padding=[0, 0]),
            MaxFeatureMap2D(),
            nn.BatchNorm2d(channels[1] // 2, affine=False),
            nn.Conv2d(channels[1] // 2, channels[2], [3, 3], 1, padding=[1, 1]),
            MaxFeatureMap2D(),

            torch.nn.MaxPool2d([2, 2], [2, 2]),
            nn.BatchNorm2d(channels[2] // 2, affine=False),

            nn.Conv2d(channels[2] // 2, channels[3], [1, 1], 1, padding=[0, 0]),
            MaxFeatureMap2D(),
            nn.BatchNorm2d(channels[3] // 2, affine=False),
            nn.Conv2d(channels[3] // 2, channels[4], [3, 3], 1, padding=[1, 1]),
            MaxFeatureMap2D(),

            torch.nn.MaxPool2d([2, 2], [2, 2]),

            nn.Conv2d(channels[4] // 2, channels[5], [1, 1], 1, padding=[0, 0]),
            MaxFeatureMap2D(),
            nn.BatchNorm2d(channels[5] // 2, affine=False),
            nn.Conv2d(channels[5] // 2, channels[6], [3, 3], 1, padding=[1, 1]),
            MaxFeatureMap2D(),
            nn.BatchNorm2d(channels[6] // 2, affine=False),

            nn.Conv2d(channels[6] // 2, channels[7], [1, 1], 1, padding=[0, 0]),
            MaxFeatureMap2D(),
            nn.BatchNorm2d(channels[7] // 2, affine=False),
            nn.Conv2d(channels[7] // 2, channels[8], [3, 3], 1, padding=[1, 1]),
            MaxFeatureMap2D(),
            nn.MaxPool2d([2, 2], [2, 2]),
            
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # (batchsize, length, feature_dim)
        assert len(x.size()) == 3, f'Input size error in LCNN. Need 3, but get {len(x.size())}'
        
        # (batchsize, 1, length, feature_dim) -> (batch, channel, frame//N, feat_dim//N)
        batch_size = x.size(0)
        x = x.unsqueeze(1)
        hidden_features = self.m_transform(x)
        
        # (batch, channel, frame//N, feat_dim//N) -> (batch, frame//N, channel * feat_dim//N)
        hidden_features = hidden_features.permute(0, 2, 1, 3).contiguous()
        frame_num = hidden_features.shape[1]
        hidden_features = hidden_features.reshape(batch_size, frame_num, -1)

        return hidden_features


class MaxFeatureMap2D(nn.Module):
    def __init__(self, max_dim = 1):
        super().__init__()
        self.max_dim = max_dim
        
    def forward(self, inputs):
        # suppose inputs (batchsize, channel, length, dim)
        
        shape = list(inputs.size())
        
        if self.max_dim >= len(shape):
            print("MaxFeatureMap: maximize on %d dim" % (self.max_dim))
            print("But input has %d dimensions" % (len(shape)))
            sys.exit(1)
        if shape[self.max_dim] // 2 * 2 != shape[self.max_dim]:
            print("MaxFeatureMap: maximize on %d dim" % (self.max_dim))
            print("But this dimension has an odd number of data")
            sys.exit(1)
        shape[self.max_dim] = shape[self.max_dim]//2
        shape.insert(self.max_dim, 2)
        
        # view to (batchsize, 2, channel//2, ...)
        # maximize on the 2nd dim
        m, i = inputs.view(*shape).max(self.max_dim)
        return m

if __name__ == '__main__':
    from torchsummary import summary
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    
    from backend.attention import Attention
    
    model = torch.nn.Sequential(
        LCNN(),
        Attention('LCNN', 120, 64, input_mean_std=False)
    ).cuda()
    summary(model, input_size=(401, 120), batch_size=2)