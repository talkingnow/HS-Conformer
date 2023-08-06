import torch
import torch.nn as nn

from . import _processing
from .lfb import LFB
from .lfcc import LFCC
from .mfcc import MFCC
from .spectrogram import Spectrogram

class LFCCSpectrogram(nn.Module):
    def __init__(self, mix_type, sample_rate, n_bins, coef, n_fft, win_length, hop, 
                        with_delta=True, with_emphasis=True, with_energy=True,
                        frq_mask=False, p=0, max=0):
        super(LFCCSpectrogram, self).__init__()
        # type: str = "conv", "concat", "linear"
        self.mix_type = mix_type
        self.LFCC = LFCC(sample_rate=sample_rate, 
                        n_lfcc=n_bins,
                        coef=coef, 
                        n_fft=n_fft, 
                        win_length=win_length, 
                        hop=hop,
                        frq_mask=frq_mask)

        self.Spectrogram = Spectrogram(sample_rate=sample_rate,  
                            coef=coef, 
                            n_fft=n_fft, 
                            win_length=win_length, 
                            hop=hop,
                            frq_mask=frq_mask)
        if mix_type == "conv":
            self.conv = nn.Conv2d(2, 1, 1, padding=0)
            self.relu = nn.SiLU()
        elif mix_type == "concat":
            pass
        elif mix_type == "linear":
            self.with_delta = with_delta
            self.n_bins = n_bins
            out_bin = self.n_bins * 3 if self.with_delta else self.n_bins
            # self.fc = None
            self.fc = nn.Linear(377, out_bin)
            self.bn = nn.LayerNorm(out_bin)
            self.relu = nn.SiLU()
            
        self.device = "cpu"
        
    def forward(self, x):
        assert len(x.size()) == 2, f'Input size error in MFCC. Need 2, but get {len(x.size())}'
        
        # device sync
        if x.device != self.device:
            self.device = x.device
            self.to(x.device)
        
        with torch.no_grad():
            lfcc = self.LFCC(x) # B, L, F
            spectrogram = self.Spectrogram(x) # B, L, F
        
        if self.mix_type == "conv":
            x = torch.cat((lfcc.unsqueeze(1), spectrogram.unsqueeze(1)), dim=1) # B, 2, L, F
            x = self.conv(x)
            x = self.relu(x)
            x = x.squeeze(1)
        elif self.mix_type == "concat":
            x = torch.cat((lfcc, mfcc), dim=2) # B, L, F * 2
        elif self.mix_type == "linear":
            x = torch.cat((lfcc, spectrogram), dim=2)
            # Batchnorm2d로 실험해볼 것
            x = self.fc(x)
            x = self.bn(x)
            x = self.relu(x)
    
        return x
    