import torch
import torch.nn as nn

from . import _processing
from .lfb import LFB
from .lfcc import LFCC
from .mfcc import MFCC
from .spectrogram import Spectrogram

class LFCCLFB(nn.Module):
    def __init__(self, mix_type, sample_rate, n_bins, coef, n_fft, win_length, hop, 
                        with_delta=True, with_emphasis=True, with_energy=True,
                        frq_mask=False, p=0, max=0):
        super(LFCCLFB, self).__init__()
        # type: str = "conv", "concat", "linear"
        self.mix_type = mix_type
        self.LFCC = LFCC(sample_rate=sample_rate, 
                        n_lfcc=n_bins,
                        coef=coef, 
                        n_fft=n_fft, 
                        win_length=win_length, 
                        hop=hop,
                        frq_mask=frq_mask)

        self.LFB = LFB(sample_rate=sample_rate, 
                        n_lbf=n_bins, 
                        coef=coef, 
                        n_fft=n_fft, 
                        win_length=win_length, 
                        hop=hop,
                        with_delta=with_delta,
                        with_emphasis=with_emphasis,
                        frq_mask=frq_mask)
        if mix_type == "conv":
            self.conv = nn.Conv2d(2, 1, 1, padding=0)
            self.swish = nn.SiLU()
        elif mix_type == "concat":
            pass
        elif mix_type == "linear":
            in_bins = n_bins * 6 if with_delta else n_bins * 2
            out_bins = n_bins * 3 if with_delta else n_bins
            self.fc = nn.Linear(in_bins, out_bins)
            self.bn = nn.LayerNorm(out_bins)
            self.swish = nn.SiLU()
            
        self.device = "cpu"
        
    def forward(self, x):
        assert len(x.size()) == 2, f'Input size error in MFCC. Need 2, but get {len(x.size())}'
        
        # device sync
        if x.device != self.device:
            self.device = x.device
            self.to(x.device)
        
        with torch.no_grad():
            lfcc = self.LFCC(x) # B, L, F
            lfb = self.LFB(x) # B, L, F
        
        if self.mix_type == "conv":
            assert ValueError(f"There is no module named: conv in pre-processing")
        elif self.mix_type == "concat":
            x = torch.cat((lfcc, lfb), dim=2) # B, L, F * 2
        elif self.mix_type == "linear":
            x = torch.cat((lfcc, lfb), dim=2)
            x = self.fc(x)
            x = self.bn(x)
            x = self.swish(x)
    
        return x
    