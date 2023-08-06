from typing import Tuple

import torch
import torch.nn as nn

class BaseSubsampling(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.right_context = 0
        self.subsampling_rate = 1

    def position_encoding(self, offset: int, size: int) -> torch.Tensor:
        return self.pos_enc.position_encoding(offset, size)

class Conv2dSubsampling2(BaseSubsampling):
    def __init__(self, idim_F: int, idim_T: int, strides: tuple, odim: int, dropout_rate: float,
                 pos_enc_class: torch.nn.Module):   
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, strides[0]),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim, odim),    
        )
        self.pos_enc = pos_enc_class
        
        # cls token
        self.cls_token_F = nn.Parameter(torch.rand(1, 1, idim_T, odim))
        self.cls_token_T = nn.Parameter(torch.rand(1, idim_F+1, 1, odim))

    def forward(
            self,
            x: torch.Tensor,
            offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.conv(x) # (B, odim, T, F)
        x = x.permute(0, 3, 2, 1)   # (B, F, T, odim)
        x = self.out(x) # (B, F, T, odim)
        x = torch.cat((self.cls_token_F.repeat(x.size(0), 1, 1, 1), x), dim=1)
        x = torch.cat((self.cls_token_T.repeat(x.size(0), 1, 1, 1), x), dim=2)
        # (B, F+1, T+1, odim)

        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb

class Conv2dSubsampling4(BaseSubsampling):
    def __init__(self, idim_F: int, idim_T: int, strides: tuple, odim: int, dropout_rate: float,
                 pos_enc_class: torch.nn.Module):   
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, strides[0]),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, strides[1]),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim, odim),    
        )
        self.pos_enc = pos_enc_class
        
        # cls token
        self.cls_token_F = nn.Parameter(torch.rand(1, 1, idim_T, odim))
        self.cls_token_T = nn.Parameter(torch.rand(1, idim_F+1, 1, odim))

    def forward(
            self,
            x: torch.Tensor,
            offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.conv(x) # (B, odim, T, F)
        x = x.permute(0, 3, 2, 1)   # (B, F, T, odim)
        x = self.out(x) # (B, F, T, odim)
        # print(x.size())
        x = torch.cat((self.cls_token_F.repeat(x.size(0), 1, 1, 1), x), dim=1)
        x = torch.cat((self.cls_token_T.repeat(x.size(0), 1, 1, 1), x), dim=2)
        # (B, F+1, T+1, odim)

        x, pos_emb = self.pos_enc(x, offset)

        return x, pos_emb
    
class Conv2dSubsampling6(BaseSubsampling):
    def __init__(self, idim_F: int, idim_T: int, strides: tuple, odim: int, dropout_rate: float,
                 pos_enc_class: torch.nn.Module):   
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, strides[0]),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, strides[1]),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, strides[2]),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim, odim),    
        )
        self.pos_enc = pos_enc_class
        
        # cls token
        self.cls_token_F = nn.Parameter(torch.rand(1, 1, idim_T, odim))
        self.cls_token_T = nn.Parameter(torch.rand(1, idim_F+1, 1, odim))

    def forward(
            self,
            x: torch.Tensor,
            offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.conv(x) # (B, odim, T, F)
        x = x.permute(0, 3, 2, 1)   # (B, F, T, odim)
        x = self.out(x) # (B, F, T, odim)
        # print(x.size()) 
        x = torch.cat((self.cls_token_F.repeat(x.size(0), 1, 1, 1), x), dim=1)
        x = torch.cat((self.cls_token_T.repeat(x.size(0), 1, 1, 1), x), dim=2)
        # (B, F+1, T+1, odim)

        x, pos_emb = self.pos_enc(x, offset)

        return x, pos_emb
    
