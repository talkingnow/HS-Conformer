import math
import torch
import torch.nn as nn
from transformers import AutoConfig, WavLMModel

BASE_PLUS = 'microsoft/wavlm-base-plus'

class StudentWavLMPlus(nn.Module):
    def __init__(self, num_hidden_layer, hidden_size, return_all_hiddens=False, init_teacher_param=None):
        super(StudentWavLMPlus, self).__init__()
        self.return_all_hiddens = return_all_hiddens

        # set transformer encoder
        config = AutoConfig.from_pretrained(BASE_PLUS)
        config.num_hidden_layers = num_hidden_layer
        config.hidden_size = hidden_size
        self.wavlm = WavLMModel(config=config)
        
        # weight initialization
        teacher = WavLMModel.from_pretrained(
            BASE_PLUS,
            from_tf=bool(".ckpt" in BASE_PLUS),
            config=AutoConfig.from_pretrained(BASE_PLUS),
            revision="main",
            ignore_mismatched_sizes=False,
        )
        self.wavlm.feature_extractor.load_state_dict(teacher.feature_extractor.state_dict(), strict=False)
        self.wavlm.feature_projection.load_state_dict(teacher.feature_projection.state_dict(), strict=False)
        if init_teacher_param is not None:
            for i in range(num_hidden_layer):
                self.wavlm.encoder.layers[i].load_state_dict(teacher.encoder.layers[init_teacher_param[i]].state_dict(), strict=False)
        
    def forward(self, x):
        x = self.wavlm(x, output_hidden_states=self.return_all_hiddens)
        
        if self.return_all_hiddens:
            return torch.stack(x.hidden_states, dim=1)
        else:
            return x.last_hidden_state