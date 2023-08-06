import torch
from .interface import Framework

class SSL_KD_Framework(Framework):
    def __init__(self, teacher, student, kd_loss):
        super(SSL_KD_Framework, self).__init__()
        self.teacher = teacher
        self.trainable_modules['student'] = student
        self.trainable_modules['kd_loss'] = kd_loss
        
    def __call__(self, x, x_teacher=None, kd_label=None):
        # student
        x = self.trainable_modules['student'](x)
        
        # loss (KD)
        kd_loss = None
        if x_teacher is not None:
            with torch.set_grad_enabled(False):
                kd_label = self.teacher(x_teacher, output_hidden_states=True).hidden_states
                kd_label = torch.stack(kd_label, dim=1)
        if kd_label is not None:
            kd_loss = self.trainable_modules['kd_loss'](x, kd_label)
            return x, kd_loss
        else:
            return x

class SSL_TAKD_Framework(Framework):
    def __init__(self, teacher, student, backend, kd_loss, ft_loss):
        super(SSL_TAKD_Framework, self).__init__()
        self.teacher = teacher
        self.trainable_modules['student'] = student
        self.trainable_modules['backend'] = backend
        self.trainable_modules['kd_loss'] = kd_loss
        self.trainable_modules['ft_loss'] = ft_loss
        
    def __call__(self, x, x_teacher=None, kd_label=None, ft_label=None):
        student = self.trainable_modules['student']
        backend = self.trainable_modules['backend']
        
        # student
        x = student(x)
        
        # loss (KD)
        kd_loss = None
        if x_teacher is not None or kd_label is not None:
            if x_teacher is not None:
                with torch.set_grad_enabled(False):
                    kd_label = self.teacher(x_teacher, output_hidden_states=True).hidden_states
                    kd_label = torch.stack(kd_label, dim=1)
            kd_loss = self.trainable_modules['kd_loss'](x, kd_label)
        
        # backend
        x = x[:, -1, :, :]
        x = backend(x)
        
        # loss (FT)
        ft_loss = None
        if ft_label is not None:
            ft_loss = self.trainable_modules['ft_loss'](x, ft_label)
            
        if kd_loss is None and ft_loss is None:
            return x
        else:
            return x, kd_loss, ft_loss