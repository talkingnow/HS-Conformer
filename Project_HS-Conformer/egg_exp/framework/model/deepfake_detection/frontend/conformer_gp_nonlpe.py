import torch
import torch.nn as nn
import torch.nn.functional as F

# from .wenet.transformer.encoder_gp_nonlpe import ConformerEncoder
from wenet.transformer.encoder_gp_nonlpe import ConformerEncoder

class ConformerGP_NonLPE(nn.Module):
    def __init__(self, bin_size=120, num_blocks=6, output_size=128, input_layer="conv2d2", 
            pos_enc_layer_type="rel_pos", use_ssl=False, ssl_layers=12, linear_units = 256, cnn_module_kernel=15,
            downsample_layer=[1,3], pooling_size=0.5, input_seq_len=200, use_cls=False, dropout=0):

        super(ConformerGP_NonLPE, self).__init__()
        
        self.use_ssl = use_ssl
        if self.use_ssl:
            self.w = nn.Parameter(torch.rand(1, ssl_layers + 1, 1, 1))
            
        self.conformer_mp = ConformerEncoder(input_size=bin_size, linear_units=linear_units, num_blocks=num_blocks, 
                output_size=output_size, input_layer=input_layer, pos_enc_layer_type=pos_enc_layer_type, cnn_module_kernel=cnn_module_kernel,
                downsample_layer=downsample_layer, pooling_size=pooling_size, input_seq_len=input_seq_len, use_cls=use_cls)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
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
        x = self.conformer_mp(x, lens)
        x = self.dropout(x)
        # print(x.size())
        # (batchsize, length, hidden_dim)
        return x

if __name__ == '__main__': 
    from torchsummary import summary
    
    model = ConformerGP_NonLPE().cuda()
    summary(model, input_size=(401, 120), batch_size=2)