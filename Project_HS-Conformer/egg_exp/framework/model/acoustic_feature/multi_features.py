import torch
import torch.nn as nn

from . import _processing
from .lfb import LFB
from .lfcc import LFCC
from .mfcc import MFCC
from .spectrogram import Spectrogram

class MultiFeatures(nn.Module):
    def __init__(self, f_types, sample_rate, n_bins, coef, n_fft, win_length, hop, 
                        with_delta=True, with_emphasis=True, with_energy=True,
                        frq_mask=False, p=0, max=0):
        super(MultiFeatures, self).__init__()
        # f_types = ['LFCC', 'MFCC', 'Spec', 'LFB']
        self.f_types = f_types
        if 'LFCC' in f_types:
            self.LFCC = LFCC(sample_rate=sample_rate, 
                            n_lfcc=n_bins,
                            coef=coef, 
                            n_fft=n_fft, 
                            win_length=win_length, 
                            hop=hop,
                            frq_mask=frq_mask)
        if 'MFCC' in f_types:
            self.MFCC = MFCC(sample_rate=sample_rate, 
                             n_mfcc=n_bins, 
                             coef=coef, 
                             n_fft=n_fft, 
                             win_length=win_length, 
                             hop=hop,
                             frq_mask=frq_mask)
        if 'Spec' in f_types:
            self.Spec = Spectrogram(sample_rate=sample_rate, 
                                    coef=coef, 
                                    n_fft=n_fft, 
                                    win_length=win_length, 
                                    hop=hop,
                                    frq_mask=frq_mask)
        if 'LFB' in f_types:
            self.LFB = LFB(sample_rate=sample_rate, 
                            n_lbf=n_bins,
                            coef=coef, 
                            n_fft=n_fft, 
                            win_length=win_length, 
                            hop=hop,
                            frq_mask=frq_mask)
        
    def forward(self, x):
        assert len(x.size()) == 2, f'Input size error in MFCC. Need 2, but get {len(x.size())}'
        
        # device sync
        if x.device != self.device:
            self.device = x.device
            self.to(x.device)
        
        with torch.no_grad():
            if 'LFCC' in self.f_types:
                lfcc = self.LFCC(x)
            if 'MFCC' in self.f_types:
                mfcc = self.MFCC(x)
            if 'Spec' in self.f_types:
                spec = self.Spec(x)
            if 'LFB' in self.f_types:
                lfb = self.LFB(x)
            
            
                
            return x