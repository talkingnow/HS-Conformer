import torch
import torch.nn as nn

from .. import Criterion

class LastHiddenMSE(Criterion):
    '''KD loss simply comparing the last outputs of teachers and students using MSE. 
    '''
    def __init__(self):
        super(Criterion, self).__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, x, label):
        x = x[:, -1, :, :]
        if label.size(1) != 1:
            label = label[:, -1, :, :]
        
        loss = self.mse(x, label)
        return loss

class DistilHubertKDLoss(Criterion):
    '''KD loss function used in papaer 'Distilhubert: Speech representation learning by layer-wise distillation of hidden-unit bert'.
    This compute (l1_loss - cos_sim) in the hidden layers. 
    '''
    def __init__(self):
        super(DistilHubertKDLoss, self).__init__()

        self.log_sigmoid = nn.LogSigmoid()
        self.cos_sim = nn.CosineSimilarity(dim=1)

    def forward(self, x, label):
        bs, l, s, h = x.size()  # bs, num_layer, sequence_length, hidden_size

        # calculate L1-loss
        if label.size(1) != 3:
            label = label[:, [4, 8, 12], :, :]
        
        l1_loss = torch.abs(x.reshape(bs, l, s * h) - label.reshape(bs, l, s * h))
        l1_loss = torch.mean(l1_loss, dim=-1)

        # calculate cosine score
        cos_sim_loss = self.cos_sim(x.reshape(bs * l * s, h), label.reshape(bs * l * s, h))
        cos_sim_loss = self.log_sigmoid(cos_sim_loss).view(bs, l, s)
        cos_sim_loss = cos_sim_loss.sum(dim=2)

        loss = l1_loss - cos_sim_loss
        loss = loss.sum(dim=1)
        loss = loss.mean()

        return loss