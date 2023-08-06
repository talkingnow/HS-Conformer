import torch
import torch.nn as nn
import torch.nn.functional as F
from .interface import Framework

class Loss_w(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.Tensor([1]*5), requires_grad=True)
        
    def forward(self, loss):
        w = F.softmax(F.tanh(self.w), dim=0)
        final_loss = torch.mul(loss, w).sum(0)
        return final_loss, w

class DeepfakeDetectionFramework_DA_learn_multiloss(Framework):
    def __init__(self, augmentation, preprocessing, frontend, backend, loss):
        super(DeepfakeDetectionFramework_DA_learn_multiloss, self).__init__()
        
        self.num_loss = 5
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.trainable_modules['frontend'] = frontend
        for i in range(self.num_loss):
            self.trainable_modules[f'backend{i}'] = backend[i]
            self.trainable_modules[f'loss{i}'] = loss[i]
          
        self.trainable_modules['w'] = Loss_w()

    def __call__(self, x, label=None):
        # pre_processing
        
        with torch.set_grad_enabled(False):
            x = self.augmentation(x)
        x = self.preprocessing(x)

        # feed forward
        x, embedding = self.trainable_modules['frontend'](x)
        embed_list = []
        for i in range(self.num_loss-1):
            embed_list.append(self.trainable_modules[f'backend{i}'](embedding[:, i, :].unsqueeze(1)))
        embed_list.append(self.trainable_modules[f'backend{self.num_loss-1}'](x))
            
        # loss 
        if label is not None:
            loss_embs = []
            for i in range(self.num_loss):
                loss_emb = self.trainable_modules[f'loss{i}'](embed_list[i], label)
                loss_embs.append(loss_emb)
            
            loss_embs = torch.stack(loss_embs, dim=0)
            
            final_loss, w = self.trainable_modules['w'](loss_embs)
            return x, final_loss, loss_embs, w
        else:
            x = self.trainable_modules[f'loss{self.num_loss-1}'](embed_list[-1])
            return x
        
