import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import get_window


class STFT(torch.nn.Module):
    def __init__(self, filter_length: int = 800, hop_length: int = 200, win_length: int = 800, window_name: str = 'hann'):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length

        window = torch.from_numpy(get_window(window_name, win_length, fftbins=True))
        self.register_buffer('window', window.float())
        self.window: torch.Tensor

    def forward(self, x: torch.Tensor):
        return torch.stft(x, self.filter_length, self.hop_length, self.win_length, window=self.window, pad_mode='reflect', return_complex=True)


class ISTFT(torch.nn.Module):
    def __init__(self, filter_length: int = 800, hop_length: int = 200, win_length: int = 800, window_name: str = 'hann'):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window: torch.Tensor

        window = torch.from_numpy(get_window(window_name, win_length, fftbins=True))
        self.register_buffer('window', window.float())

    def forward(self, x_spec_tr: torch.Tensor):
        return torch.istft(x_spec_tr, self.filter_length, self.hop_length, self.win_length, window=self.window)


def dct1(x: torch.Tensor, n: int | None = None):  # DCT type I
    if n is not None:
        if n > x.shape[-1]:
            x = F.pad(x, (0, n - x.shape[-1]))
        elif n < x.shape[-1]:
            x = x[..., :n]
    return torch.real(torch.fft.rfft(torch.cat([x, x.flip(-1)[..., 1:-1]], dim=-1)))

def dct_resample(input: torch.Tensor, inDim: int, outDim: int):
    if inDim == 0:
        inDim = int(outDim * 128 / 442)
    output = dct1(dct1(input)[..., :inDim], n=outDim)
    output /= input.shape[-1] * 2
    return output

def getSpn(xnfft, resampleInDim, resampleOutDim=None):
    resampleOutDim = resampleOutDim if resampleOutDim is not None else xnfft.shape[-2]
    #xnfft = xnfft + 1e-8
    xnfft = torch.clamp(xnfft, min=1e-8)
    logXnfft = torch.log(xnfft)
    logSpnfft = dct_resample(logXnfft.transpose(-1, -2), resampleInDim, resampleOutDim).transpose(-1, -2)
    spnfft = torch.exp(logSpnfft)
    return spnfft

def spnfftPowerCorrection(xnfft, spnfft):
    powerRatio = (
        torch.sum(torch.square(xnfft), dim=-1, keepdim=True) /
        torch.sum(torch.square(spnfft), dim=-1, keepdim=True))
    return spnfft * torch.sqrt(powerRatio)


