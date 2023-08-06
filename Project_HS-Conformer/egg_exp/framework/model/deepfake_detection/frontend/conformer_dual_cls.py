import torch
import torch.nn as nn
import torch.nn.functional as F

from wenet.conformer_cls.encoder_dual_cls import ConformerEncoder
# from .wenet.conformer_cls.encoder_dual_cls import ConformerEncoder

class ConformerDualCLS(nn.Module):
    def __init__(self, bin_size=(60,200), num_blocks=6, output_size=128, input_layer="linear", 
            pos_enc_layer_type="rel_pos", use_ssl=False, ssl_layers=12, linear_units = 256, cnn_module_kernel=15, use_cls=True, strides=(2, 2)):

        super(ConformerDualCLS, self).__init__()
        
        self.use_ssl = use_ssl
        if self.use_ssl:
            self.w = nn.Parameter(torch.rand(1, ssl_layers + 1, 1, 1))
            
        self.conformer_dual_cls = ConformerEncoder(input_size=bin_size, linear_units=linear_units, num_blocks=num_blocks, 
                output_size=output_size, input_layer=input_layer, pos_enc_layer_type=pos_enc_layer_type, cnn_module_kernel=cnn_module_kernel, 
                cls_token=use_cls, strides=strides)

    
    def forward(self, x):
        # (batchsize, length, feature_dim)
        assert len(x.size()) == 3, f'Input size error in Conformer. Need 3, but get {len(x.size())}'
        
        if self.use_ssl:
            # (B, L, T, H)
            # weighted-sum
            x = x * self.w.repeat(x.size(0), 1, 1, 1)
            x = x.sum(dim=1)
        
        x = x.unsqueeze(1)
        x = self.conformer_dual_cls(x)

        # (batchsize, length, hidden_dim)
        return x
    
if __name__ == '__main__':
    from torchsummary import summary
    model = ConformerDualCLS(bin_size=(27, 50), input_layer='conv2d6', strides=(2,(2,1), (2,1))).cuda()
    summary(model, input_size=(401, 60))
    x = torch.rand(2, 401, 60).cuda()
    x = model(x)