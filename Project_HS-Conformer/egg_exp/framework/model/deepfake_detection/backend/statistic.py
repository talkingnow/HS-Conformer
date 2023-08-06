import torch
import torch.nn as nn

# FIXME: feature_processing_name, in_dim 정리하기 (불필요한 변수)
class StatisticsPooling(nn.Module):
    def __init__(self, feature_processing_name, in_dim, hidden_dim):
        super(StatisticsPooling, self).__init__()
        
        if feature_processing_name == 'LCNN':
            in_dim = (in_dim // 16) * 32
        elif feature_processing_name == 'ECAPA_TDNN':
            # CAUTION 
            # There is no BatchNorm1d before Linear layer
            in_dim = (in_dim * 3) // 2
            
        # mean, std
        self.m_output_act = nn.Linear(in_dim * 2, hidden_dim)
        self.m_bn = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, x):
        # (batchsize, length, feature_dim)
        assert len(x.size()) == 3, f'Input size error in pooling. Need 3, but get {len(x.size())}'

        temp1 = torch.mean(x, dim=1, keepdim=False)
        temp2 = torch.sqrt(torch.var(x, dim=1, keepdim=False).clamp(min=1e-9))
        x = torch.cat((temp1, temp2), dim=1)
            
        output = self.m_output_act(x)
        
        return output