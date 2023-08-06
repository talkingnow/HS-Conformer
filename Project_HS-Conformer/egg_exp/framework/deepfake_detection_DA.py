import torch
from .interface import Framework

class DeepfakeDetectionFramework_DA(Framework):
    def __init__(self, augmentation, preprocessing, frontend, backend, loss):
        super(DeepfakeDetectionFramework_DA, self).__init__()

        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.trainable_modules['frontend'] = frontend
        self.trainable_modules['backend'] = backend
        self.trainable_modules['loss'] = loss

    def __call__(self, x, label=None):
        # pre_processing
        
        with torch.set_grad_enabled(False):
            x = self.augmentation(x)
        x = self.preprocessing(x)

        # feed forward
        x = self.trainable_modules['frontend'](x)
        x = self.trainable_modules['backend'](x)

        # loss 
        if label is not None:
            loss = self.trainable_modules['loss'](x, label)
            return x, loss
        else:
            x = self.trainable_modules['loss'](x)
            return x