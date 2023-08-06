import torch
import torch.nn as nn
import torch.nn.functional as F

from .wenet.transformer.encoder_mp_nonlpe_hieracls22 import ConformerEncoder
from ..backend.attention import SelfWeightedPooling

# from wenet.transformer.encoder_mp_nonlpe_hieracls22 import ConformerEncoder

class ConformerMP_NonLPE_HieraCLS2(nn.Module):
    def __init__(self, bin_size=120, num_blocks=6, output_size=128, input_layer="conv2d2", 
            pos_enc_layer_type="rel_pos", use_ssl=False, ssl_layers=12, linear_units = 256, cnn_module_kernel=15,
            downsample_layer=[1,3], pooling_size=2, input_seq_len=200, layer_cls=True, dropout=0, emb_dropout=0, multiloss=False,
            use_emb=[0,1,2,3,4]):

        super(ConformerMP_NonLPE_HieraCLS2, self).__init__()
        
        self.use_ssl = use_ssl
        if self.use_ssl:
            self.w = nn.Parameter(torch.rand(1, ssl_layers + 1, 1, 1))
        
        self.conformer_mp = ConformerEncoder(input_size=bin_size, linear_units=linear_units, num_blocks=num_blocks, 
                output_size=output_size, input_layer=input_layer, pos_enc_layer_type=pos_enc_layer_type, cnn_module_kernel=cnn_module_kernel,
                downsample_layer=downsample_layer, pooling_size=pooling_size, input_seq_len=input_seq_len, layer_cls=layer_cls)
        
        self.use_emb = use_emb
        
        if 0 in use_emb:
            self.fc0 = nn.Linear(output_size, output_size)
            self.bn0 = nn.BatchNorm1d(output_size)
        if 1 in use_emb:
            self.fc1 = nn.Linear(output_size, output_size)
            self.bn1 = nn.BatchNorm1d(output_size)
        if 2 in use_emb:
            self.fc2 = nn.Linear(output_size, output_size)
            self.bn2 = nn.BatchNorm1d(output_size)
        if 3 in use_emb:
            self.attn = SelfWeightedPooling(output_size, num_head=1, mean_only=True)

        self.fc3 = nn.Linear(int(output_size * (len(use_emb)-1)), output_size)
        self.bn3 = nn.BatchNorm1d(output_size)
        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.emb_dropout = nn.Dropout(emb_dropout) if emb_dropout > 0 else nn.Identity()
        
        self.multiloss = multiloss
        
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
        x, cls = self.conformer_mp(x, lens)
        B, T, H = x.size()
        
        emb = []
        if 0 in self.use_emb:
            emb.append(self.emb_dropout(self.bn0(self.silu(self.fc0(cls[:, 0, :])))))
        if 1 in self.use_emb:
            emb.append(self.emb_dropout(self.bn1(self.silu(self.fc1(cls[:, 1, :])))))
        if 2 in self.use_emb:
            emb.append(self.emb_dropout(self.bn2(self.silu(self.fc2(cls[:, 2, :])))))
        if 3 in self.use_emb:
            emb.append(self.emb_dropout(self.attn(x)))
        
        embedding = torch.cat(emb, dim=1)
        output = self.dropout(embedding)
        output = self.bn3(self.silu(self.fc3(output))).unsqueeze(1)
        
        if self.multiloss:
            embedding = embedding.reshape(B, len(self.use_emb)-1, H)
            return output, embedding
        return output

# class SelfWeightedPooling(nn.Module):
#     def __init__(self, feature_dim, num_head=1, mean_only=False):
#         super(SelfWeightedPooling, self).__init__()

#         self.feature_dim = feature_dim
#         self.mean_only = mean_only
#         self.noise_std = 1e-5
#         self.num_head = num_head

#         # transformation matrix (num_head, feature_dim)
#         self.mm_weights = nn.Parameter(
#             torch.Tensor(num_head, feature_dim), requires_grad=True)
#         torch.nn.init.kaiming_uniform_(self.mm_weights)
        
#     def forward(self, inputs):
#         # batch size
#         batch_size = inputs.size(0)
#         # feature dimension
#         feat_dim = inputs.size(2)
        
#         # input is (batch, legth, feature_dim)
#         # change mm_weights to (batchsize, feature_dim, num_head)
#         # weights will be in shape (batchsize, length, num_head)
#         weights = torch.bmm(inputs, 
#                             self.mm_weights.permute(1, 0).contiguous()\
#                             .unsqueeze(0).repeat(batch_size, 1, 1))
        
#         # attention (batchsize, length, num_head)
#         attentions = nn.functional.softmax(torch.tanh(weights), dim=1)        
        
#         # apply attention weight to input vectors
#         if self.num_head == 1:
#             # We can use the mode below to compute self.num_head too
#             # But there is numerical difference.
#             #  original implementation in github
            
#             # elmentwise multiplication
#             # weighted input vector: (batchsize, length, feature_dim)
#             weighted = torch.mul(inputs, attentions.expand_as(inputs))
#         else:
#             # weights_mat = (batch * length, feat_dim, num_head)
#             weighted = torch.bmm(
#                 inputs.view(-1, feat_dim, 1), 
#                 attentions.view(-1, 1, self.num_head))
            
#             # weights_mat = (batch, length, feat_dim * num_head)
#             weighted = weighted.view(batch_size, -1, feat_dim * self.num_head)
            
#         # pooling
#         if self.mean_only:
#             # only output the mean vector
#             representations = weighted.sum(1)
#         else:
#             # output the mean and std vector
#             noise = self.noise_std * torch.randn(
#                 weighted.size(), dtype=weighted.dtype, device=weighted.device)

#             avg_repr, std_repr = weighted.sum(1), (weighted+noise).std(1)
#             # concatenate mean and std
#             representations = torch.cat((avg_repr,std_repr),1)

#         # done
#         return representations
    
# if __name__ == '__main__':
#     from torchsummary import summary
    
#     model = ConformerMP_NonLPE_HieraCLS2().cuda()
#     summary(model, input_size=(401, 120), batch_size=2)

