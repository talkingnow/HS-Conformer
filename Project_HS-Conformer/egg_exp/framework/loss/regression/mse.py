import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import Criterion

class MSE(Criterion):
    def __init__(self):
        super(Criterion, self).__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        loss = self.mse(x, label)
        return loss
