import typing
import collections.abc
from collections.abc import Iterable, Iterator, Callable, Sequence as Seq, Mapping as Map
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize
from torch.nn.utils.parametrizations import weight_norm, spectral_norm

from utils import u_fourier

if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=4, linewidth=64)
    torch.set_printoptions(sci_mode=False, precision=4, linewidth=64)


def remove_parametrizations(model: nn.Module):  # recursive
    for child in model.children():
        if torch.nn.utils.parametrize.is_parametrized(child, 'weight'):
            torch.nn.utils.parametrize.remove_parametrizations(child, 'weight')
        else:
            remove_parametrizations(child)


class Generator2(nn.Module):
    def __init__(self, channel_sizes: Seq[int], fft_size: int, hop_length: int):
        super().__init__()

        self.stft  = u_fourier.STFT (fft_size, hop_length, fft_size)
        self.istft = u_fourier.ISTFT(fft_size, hop_length, fft_size)

        self.pre_conv = weight_norm(nn.Conv2d(3, channel_sizes[0], (3, 3), padding='same', padding_mode='reflect'))
        self.convs = nn.ModuleList([
            nn.Sequential(
                weight_norm(nn.Conv2d(channel_size, channel_size, (5, 1), groups=channel_size, padding='same', padding_mode='reflect')),
                weight_norm(nn.Conv2d(channel_size, channel_size, (1, 5), groups=channel_size, padding='same', padding_mode='reflect')),
                weight_norm(nn.Conv2d(channel_size, next_channel_size, (1, 1), padding='same', padding_mode='reflect')))
            for channel_size, next_channel_size in zip(channel_sizes, [*channel_sizes[1:], *channel_sizes[-1:]])])
        self.post_conv = weight_norm(nn.Conv2d(channel_sizes[-1], 2, (3, 3), padding='same', padding_mode='reflect'))

    def forward(self, spnfft: torch.Tensor, yhfftc: torch.Tensor, z: torch.Tensor | None = None):
        yhfft = torch.abs(yhfftc)

        if z is None:
            z = torch.randn_like(spnfft)

        x: torch.Tensor = torch.stack([spnfft, yhfft, z], dim=1)
        x = self.pre_conv(x)

        for conv in self.convs:
            x = F.leaky_relu(x, 0.2)
            x = conv(x)

            x = torch.complex(x[:, ::2], x[:, 1::2])
            batchSize, channelSize, nMels, frameCount = x.shape
            x = self.istft(x.view(-1, nMels, frameCount))
            x = self.stft(x).view(batchSize, channelSize, nMels, frameCount)
            x = torch.concat([x.real, x.imag], dim=1)

        x = self.post_conv(x)

        x = torch.complex(x[:, 0], x[:, 1])
        x = torch.exp(1j * torch.angle(x))
        ynfftp = x

        return ynfftp

class DiscriminatorP(nn.Module):
    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3):
        super().__init__()

        self.period = period

        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(2,   64,   (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            weight_norm(nn.Conv2d(64,  128,  (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            weight_norm(nn.Conv2d(128, 256,  (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            weight_norm(nn.Conv2d(256, 512,  (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            weight_norm(nn.Conv2d(512, 1024, (kernel_size, 1), 1,           padding=(kernel_size // 2, 0))),])

        self.conv_post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor, y: torch.Tensor, xn):
        fmap = []

        # 1d to 2d
        batch_size, sample_count = x.shape
        if sample_count % self.period != 0: # pad first
            n_pad = self.period - (sample_count % self.period)
            x = F.pad(x, (0, n_pad), 'reflect')
            sample_count = sample_count + n_pad
            y = F.pad(y, (0, n_pad), 'reflect')
        x = x.view(batch_size, 1, sample_count // self.period, self.period)
        y = y.view(batch_size, 1, sample_count // self.period, self.period)

        x = torch.cat((x, y), 1)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.2)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.discriminators = nn.ModuleList(
            [DiscriminatorP(period=period) for period in [2, 3, 5, 7, 11]])

    def forward(self, x, y, xn):
        ret = []
        for disc in self.discriminators:
            ret.append(disc(x, y, xn))

        return ret  # [(feat, score), (feat, score), (feat, score), (feat, score), (feat, score)]


class DiscriminatorR(torch.nn.Module):
    def __init__(self, fft_size: int, hop_length: int, win_length: int):
        super().__init__()

        self.fourier = u_fourier.STFT(filter_length=fft_size, hop_length=hop_length, win_length=win_length)

        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(5, 32, (3, 9), padding=(1, 4))),
            weight_norm(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(32, 32, (3, 3), padding=(1, 1))),
        ])
        self.conv_post = weight_norm(nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))

    def forward(self, x: torch.Tensor, y: torch.Tensor, xn: torch.Tensor):
        fmap = []

        x = self.fourier(x)
        y = self.fourier(y)
        xn = self.fourier(xn)
        # ins

        Spn = u_fourier.getSpn(torch.abs(xn), int(xn.shape[-2] * 128 / 442))
        Spn = Spn.detach()
        y = y.detach()

        x = torch.stack([torch.abs(x), torch.angle(x), torch.abs(y), torch.angle(y), Spn], dim=1)
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.2)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)

        x = torch.flatten(x, 1, -1)
        return x


class MultiResolutionDiscriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.discriminators = nn.ModuleList([
            DiscriminatorR(1764, 270, 1080),
            DiscriminatorR(3528, 420, 1680),
            DiscriminatorR(882,  80,  480 )])

    def forward(self, x, y, xn):
        ret = list()
        for disc in self.discriminators:
            ret.append(disc(x, y, xn))

        return ret  # [(feat, score), (feat, score), (feat, score)]


class MultiResolutionDiscriminator1D(nn.Module):
    def __init__(self):
        super().__init__()

        fft_size, hop_length, win_length = 1764, 270, 1080
        self.fourier = u_fourier.STFT(fft_size, hop_length, win_length)

        self.convs = nn.ModuleList([
            weight_norm(nn.Conv1d(5 * int(fft_size / 2 + 1), 256, 3, padding=1)),
            weight_norm(nn.Conv1d(256, 256, 3, padding=1)),
            weight_norm(nn.Conv1d(256, 128, 3, padding=1)),
            weight_norm(nn.Conv1d(128, 128, 3, padding=1)),
            weight_norm(nn.Conv1d(128, 64, 3, padding=1)),])
        self.conv_post = weight_norm(nn.Conv1d(64, 1, 3, padding=1))

    def forward(self, x: torch.Tensor, y: torch.Tensor, xn: torch.Tensor):
        x = self.fourier(x)
        y = self.fourier(y)
        xn = self.fourier(xn)

        Spn = u_fourier.getSpn(torch.abs(xn), int(xn.shape[-2] * 128 / 442))
        Spn = Spn.detach()
        y = y.detach()

        x = torch.cat([torch.abs(x), torch.angle(x), torch.abs(y), torch.angle(y), Spn], dim=1)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.2)
        x = self.conv_post(x)
        x = torch.flatten(x, 1, -1)

        ret = []
        ret.append(x)
        return ret


class Discriminator(nn.Module):
    def __init__(self, filter_length: int, hop_length: int, win_length: int):
        super().__init__()
        self.istft = u_fourier.ISTFT(filter_length, hop_length, win_length)
        self.MRD = MultiResolutionDiscriminator()
        self.MPD = MultiPeriodDiscriminator()
        self.MRD1 = MultiResolutionDiscriminator1D()

    def forward(self, x: torch.Tensor, y: torch.Tensor, xn: torch.Tensor):
        x  = self.istft(x)
        y  = self.istft(y)
        xn = self.istft(xn)

        return self.MPD(x, y, xn), self.MRD(x, y, xn), self.MRD1(x, y, xn)



class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(self, filter_length: int, hop_length: int, win_length: int,):
        """Initialize Multi resolution STFT loss module.
        Args:
            resolutions (list): List of (FFT size, hop size, window length).
        """
        super().__init__()

        self.istft = u_fourier.ISTFT(filter_length=filter_length, hop_length=hop_length, win_length=win_length)

        self.mrd_fouriers = nn.ModuleList([
            u_fourier.STFT(1764, 270, 1080),
            u_fourier.STFT(3528, 420, 1680),
            u_fourier.STFT(882,  80,  480)])

    def forward(self, ynfftp, xnfftp):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = 0.
        mag_loss = 0.
        for i in range(len(self.mrd_fouriers)):
            ynp = self.istft(ynfftp)
            xnp = self.istft(xnfftp)
            ynpfftc: torch.Tensor = self.mrd_fouriers[i](ynp)
            xnpfftc: torch.Tensor = self.mrd_fouriers[i](xnp)

            ynpfft = ynpfftc.abs().clamp_min(1e-9)
            xnpfft = xnpfftc.abs().clamp_min(1e-9)

            sc_loss += torch.norm(xnpfft - ynpfft, p='fro') / torch.norm(xnpfft, p='fro')   # Spectral Convergenge Loss
            mag_loss += F.l1_loss(torch.log(xnpfft), torch.log(ynpfft))                     # Log STFT Magnitude Loss

        sc_loss /= len(self.mrd_fouriers)
        mag_loss /= len(self.mrd_fouriers)

        return sc_loss, mag_loss


