import torch
import torch.nn as nn
import torch.nn.functional as F

from .wenet.transformer.encoder_cls import ConformerEncoder

class ConformerDualCLS2(nn.Module):
    def __init__(self, bin_size=60, frame_size=401, num_blocks=6, output_size=128, input_layer_T="conv2d2", input_layer_F="linear",
            pos_enc_layer_type="rel_pos", use_ssl=False, ssl_layers=12, linear_units = 256, cnn_module_kernel=15, use_cls=True):

        super(ConformerDualCLS2, self).__init__()
        
        self.use_ssl = use_ssl
        if self.use_ssl:
            self.w = nn.Parameter(torch.rand(1, ssl_layers + 1, 1, 1))
            
        self.conformer_cls_T = ConformerEncoder(input_size=bin_size, linear_units=linear_units, num_blocks=num_blocks, 
                output_size=output_size, input_layer=input_layer_T, pos_enc_layer_type=pos_enc_layer_type, cnn_module_kernel=cnn_module_kernel, 
                cls_token=use_cls)
        
        self.conformer_cls_F = ConformerEncoder(input_size=frame_size, linear_units=linear_units, num_blocks=num_blocks, 
                output_size=output_size, input_layer=input_layer_F, pos_enc_layer_type=pos_enc_layer_type, cnn_module_kernel=cnn_module_kernel, 
                cls_token=use_cls)
    
    def forward(self, x):
        # (batchsize, length, feature_dim)
        assert len(x.size()) == 3, f'Input size error in Conformer. Need 3, but get {len(x.size())}'
        
        if self.use_ssl:
            # (B, L, T, H)
            # weighted-sum
            x = x * self.w.repeat(x.size(0), 1, 1, 1)
            x = x.sum(dim=1)
        
        lens_batch = torch.ones(x.shape[0]).to(x.device)
        lens_T = torch.round(lens_batch * x.shape[1]).int()
        lens_F = torch.round(lens_batch * x.shape[2]).int()
        
        # length(time) axis conformer
        x_T, _ = self.conformer_cls_T(x, lens_T)
        cls_T = x_T[ :, 0, :]

        # feature_dim(spatial) axis conformer
        x = x.permute(0, 2, 1)
        x_F, _ = self.conformer_cls_F(x, lens_F)
        cls_F = x_F[ :, 0, :]
        
        output = torch.cat((cls_T, cls_F), dim=1)
        
        return output
    
if __name__ == '__main__':
    model = ConformerDualCLS().cuda()
    x = torch.rand(2, 60, 401).cuda()
    x = model(x)