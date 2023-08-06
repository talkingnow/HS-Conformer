import math
import torch
import torch.nn as nn
from transformers import AutoConfig, HubertModel

BASE_960 = 'facebook/hubert-base-ls960'

class StudentHubert(nn.Module):
    def __init__(self, num_hidden_layer, hidden_size, return_all_hiddens=False, init_teacher_param=None):
        super(StudentHubert, self).__init__()
        self.return_all_hiddens = return_all_hiddens

        # set transformer encoder
        config = AutoConfig.from_pretrained(BASE_960)
        config.num_hidden_layers = num_hidden_layer
        config.hidden_size = hidden_size
        self.hubert = HubertModel(config=config)
        
        # weight initialization
        teacher = HubertModel.from_pretrained(
            BASE_960,
            from_tf=bool(".ckpt" in BASE_960),
            config=AutoConfig.from_pretrained(BASE_960),
            revision="main",
            ignore_mismatched_sizes=False,
        )
        self.hubert.feature_extractor.load_state_dict(teacher.feature_extractor.state_dict(), strict=False)
        self.hubert.feature_projection.load_state_dict(teacher.feature_projection.state_dict(), strict=False)
        if init_teacher_param is not None:
            for i in range(num_hidden_layer):
                self.hubert.encoder.layers[i].load_state_dict(teacher.encoder.layers[init_teacher_param[i]].state_dict(), strict=False)
        
    def forward(self, x):
        x = self.hubert(x, output_hidden_states=self.return_all_hiddens)
        
        if self.return_all_hiddens:
            return torch.stack(x.hidden_states, dim=1)
        else:
            return x.last_hidden_state