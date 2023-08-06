import torch
from .interface import Framework

class DeepfakeDetectionFramework_DA_multiloss_attn(Framework):
    def __init__(self, augmentation, preprocessing, frontend, backend, loss, loss_weight):
        super(DeepfakeDetectionFramework_DA_multiloss_attn, self).__init__()

        self.num_loss = len(loss_weight)
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.trainable_modules['frontend'] = frontend
        for i in range(self.num_loss):
            self.trainable_modules[f'backend{i}'] = backend[i]
            self.trainable_modules[f'loss{i}'] = loss[i]
        self.loss_weight = loss_weight

    def __call__(self, x, label=None):
        # pre_processing
        
        with torch.set_grad_enabled(False):
            x = self.augmentation(x)
        x = self.preprocessing(x)

        # feed forward
        x, embedding, attn_w = self.trainable_modules['frontend'](x)
        x = self.trainable_modules[f'backend{self.num_loss-1}'](x)
        embed_list = []
        for i in range(self.num_loss-1):
            embed_list.append(self.trainable_modules[f'backend{i}'](embedding[:, i, :].unsqueeze(1)))
            
        # loss 
        if label is not None:
            loss_embs = []
            final_loss = []
            for i in range(self.num_loss-1):
                loss_emb = self.trainable_modules[f'loss{i}'](embed_list[i], label)
                loss_embs.append(loss_emb)
                final_loss.append(loss_emb * self.loss_weight[i])
            loss_emb = self.trainable_modules[f'loss{self.num_loss-1}'](x, label)
            loss_embs.append(loss_emb)
            final_loss.append(loss_emb * self.loss_weight[-1])
            
            final_loss = sum(final_loss)
            return x, final_loss, loss_embs, attn_w
        else:
            x = self.trainable_modules[f'loss{self.num_loss-1}'](x)
            return x