import torch
import torch.nn as nn
import torch.nn.functional as F

from .wenet.transformer.encoder import TransformerEncoder

class Transformer(nn.Module):
    def __init__(self, bin_size=60, num_blocks=6, output_size=128, input_layer="no_sampling", 
            pos_enc_layer_type="abs_pos", use_ssl=False, ssl_layers=12, linear_units = 256):

        super(Transformer, self).__init__()
        
        self.use_ssl = use_ssl
        if self.use_ssl:
            self.w = nn.Parameter(torch.rand(1, ssl_layers + 1, 1, 1))

        # self.conformer = ConformerEncoder(input_size=bin_size, linear_units=linear_units, num_blocks=num_blocks, 
        #         output_size=output_size, input_layer=input_layer, pos_enc_layer_type=pos_enc_layer_type, cnn_module_kernel=cnn_module_kernel)
        self.transformer = TransformerEncoder(input_size=bin_size, linear_units=linear_units, num_blocks=num_blocks,
                            output_size=output_size, input_layer=input_layer, pos_enc_layer_type=pos_enc_layer_type)
           
    def forward(self, x):
        # (batchsize, length, feature_dim)
        assert len(x.size()) == 3, f'Input size error in Conformer. Need 3, but get {len(x.size())}'
        
        if self.use_ssl:
            # (B, L, T, H)
            # weighted-sum
            x = x * self.w.repeat(x.size(0), 1, 1, 1)
            x = x.sum(dim=1)
        
        lens = torch.ones(x.shape[0]).to(x.device)
        lens = torch.round(lens * x.shape[1]).int()
        x, masks = self.transformer(x, lens)
        
        # (batchsize, length, hidden_dim)
        return x