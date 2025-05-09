import sys
import os
import pathlib
import shutil
import itertools
import functools
import typing
import collections.abc
from collections.abc import Iterable, Iterator, Callable, Sequence as Seq, Mapping as Map
import argparse
import time
import logging
import tqdm
import random
import numpy as np
import soundfile as sf
import torch
import torch.distributed
import torch.jit
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.utils.data
import torch.utils.tensorboard.writer
import matplotlib.pyplot as plt

from models import m_v01br
from utils import u_fourier


# Changing these parameters requires updating related code elsewhere
fs = 44100
fft_size = 882
hop_size = 441


class MelFromDisk(torch.utils.data.Dataset):
    def __init__(self, data_dir: str | pathlib.Path, segment_length: int, hop_length: int, seed: int = 1234):
        random.seed(seed)
        self.split = True
        self.data_dir = pathlib.Path(data_dir).expanduser()
        self.meta = self.load_metadata()

        self.segment_size = segment_length
        self.hop_size = hop_length

    def __len__(self):
        return len(self.meta)

    def slice_or_pad(self, *xs: np.ndarray):
        inputSegmentSize = xs[0].shape[-1]
        if inputSegmentSize >= self.segment_size:
            frame_start = random.randint(0, (inputSegmentSize - self.segment_size) // self.hop_size)
            audio_start = frame_start * self.hop_size
            xs = tuple(map(lambda x: x[..., audio_start:audio_start + self.segment_size], xs))
        else:
            xs = tuple(map(lambda x: np.pad(x, [(0, self.segment_size - inputSegmentSize)], 'constant'), xs))

        return xs

    def __getitem__(self, idx: int):
        file_sub_path = self.meta[idx]
        yh_path = pathlib.Path(self.data_dir, 'yh4', file_sub_path)
        xn_path = pathlib.Path(self.data_dir, 'xn', file_sub_path)

        yh, _fs = sf.read(yh_path, dtype='float32')
        assert fs == _fs, f'{yh_path}: fs mismatch: {fs} != {_fs}'
        xn, _fs = sf.read(xn_path, dtype='float32')
        assert fs == _fs, f'{xn_path}: fs mismatch: {fs} != {_fs}'

        if self.split:
            yh, xn = self.slice_or_pad(yh, xn)

        xn = torch.Tensor(xn)
        yh = torch.Tensor(yh)

        return xn, yh, str(file_sub_path)

    def load_metadata(self) -> list[pathlib.Path]:
        data_dir = pathlib.Path(self.data_dir).expanduser()
        xnBaseDir = pathlib.Path(data_dir, 'xn')
        wavSubPaths = sorted(path.relative_to(xnBaseDir) for path in xnBaseDir.rglob('*.wav'))

        return wavSubPaths


class TrainConfig(typing.NamedTuple):
    num_gpus: int
    port: int
    num_workers: int  # 16

    channel_sizes: list[int] | Seq[int]  # [16, 12]

    lr: float     # 0.0001
    lam_gloss: float  # 1
    lam_stft: float   # 1 or 2.5

    data_dir: str        # '@data/OpenSinger'
    batch_size: int      # 12
    segment_length: int  # 22050

    ckpt_dir: str  # '@ckpt'
    log_dir: str   # '@logs'
    session_name: str

def train(rank: int, h: TrainConfig):
    if h.num_gpus > 1:
        torch.distributed.init_process_group(
            backend='nccl', init_method=f'tcp://localhost:{h.port}', world_size=1 * h.num_gpus, rank=rank)

    seed = 1234 + rank
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device(f'cuda:{rank:d}')
    torch.cuda.set_device(device)

    def _create_generator():
        return m_v01br.Generator2(h.channel_sizes, fft_size=fft_size, hop_length=hop_size)

    model_g_inner = _create_generator().to(device)
    model_d_inner = m_v01br.Discriminator(filter_length=fft_size, hop_length=hop_size, win_length=fft_size).to(device)
    stft  = u_fourier.STFT (filter_length=fft_size, hop_length=hop_size, win_length=fft_size).to(device)
    istft = u_fourier.ISTFT(filter_length=fft_size, hop_length=hop_size, win_length=fft_size).to(device)

    optim_g = torch.optim.AdamW(model_g_inner.parameters(), lr=h.lr, betas=(0.5, 0.9))
    optim_d = torch.optim.AdamW(model_d_inner.parameters(), lr=h.lr, betas=(0.5, 0.9))

    init_epoch = -1
    step = 0

    # define logger, writer, valloader, stft at rank_zero
    logger = None
    writer = None
    pt_dir = os.path.join(h.ckpt_dir, h.session_name)
    log_dir = os.path.join(h.log_dir, h.session_name)

    if rank == 0:
        os.makedirs(pt_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, '%s-%d.log' % (h.session_name, time.time()))),
                logging.StreamHandler()])

        logger = logging.getLogger()
        writer = torch.utils.tensorboard.writer.SummaryWriter(log_dir)

        writer.add_text('config', str(h))

    if h.num_gpus > 1:
        model_g = torch.nn.parallel.DistributedDataParallel(model_g_inner, device_ids=[rank]).to(device)
        model_d = torch.nn.parallel.DistributedDataParallel(model_d_inner, device_ids=[rank]).to(device)
    else:
        model_g = model_g_inner
        model_d = model_d_inner

    # this accelerates training when the size of minibatch is always consistent.
    # if not consistent, it'll horribly slow down.
    torch.backends.cudnn.benchmark = True

    dataset = MelFromDisk(data_dir=h.data_dir, segment_length=h.segment_length, hop_length=hop_size, seed=seed)
    trainloader = torch.utils.data.DataLoader(dataset=dataset,
        batch_size=h.batch_size, shuffle=True, num_workers=h.num_workers, persistent_workers=True, pin_memory=True, drop_last=True)

    model_g.train()
    model_d.train()
    stft_criterion = m_v01br.MultiResolutionSTFTLoss(filter_length=fft_size, hop_length=hop_size, win_length=fft_size).to(device)

    for epoch in itertools.count(init_epoch + 1):
        if rank == 0:
            loader = tqdm.tqdm(trainloader, desc='Loading train data')
        else:
            loader = trainloader

        g_loss_items: list[float] = []
        d_loss_items: list[float] = []
        stft_loss_items: list[float] = []
        for xn, yh, file_sub_path in loader:
            yh: torch.Tensor = yh.to(device)
            xn: torch.Tensor = xn.to(device)
            with torch.no_grad():
                yhfftc: torch.Tensor = stft(yh)
                xnfftc: torch.Tensor = stft(xn)
                xnfftp = torch.exp(torch.angle(xnfftc) * 1j)

                xnfft = torch.abs(xnfftc)
                spnfft = u_fourier.getSpn(xnfft, 128)
                spnfft = u_fourier.spnfftPowerCorrection(xnfft, spnfft)

            # generator
            optim_g.zero_grad()
            ynfftp: torch.Tensor = model_g(spnfft, yhfftc)

            # STFT Loss
            sc_loss, mag_loss = stft_criterion(ynfftp, xnfftp)
            stft_loss: torch.Tensor = sc_loss + mag_loss

            # phaseMagLoss
            res_fake, period_fake, C1_fake = model_d(ynfftp, yhfftc, xnfftc)

            g_losses: list[torch.Tensor] = []
            for score_fake in (res_fake + period_fake + C1_fake):
                g_losses.append(torch.mean(torch.square(score_fake - 1.)))

            g_loss: torch.Tensor = torch.mean(torch.stack(g_losses))

            g_loss *= h.lam_gloss
            g_loss += stft_loss * h.lam_stft
            assert not torch.isnan(g_loss), f'g_loss is NaN'
            g_loss.backward()
            optim_g.step()

            optim_d.zero_grad()

            res_fake, period_fake, C1_fake = model_d(ynfftp.detach(), yhfftc, xnfftc)
            res_real, period_real, C1_real = model_d(xnfftp.detach(), yhfftc, xnfftc)

            d_losses: list[torch.Tensor] = []
            for score_fake, score_real in zip(res_fake + period_fake + C1_fake, res_real + period_real + C1_real):
                d_losses.append(torch.mean(torch.square(score_real - 1.)))
                d_losses.append(torch.mean(torch.square(score_fake)))

            d_loss: torch.Tensor = torch.sum(torch.stack(d_losses)) / len(res_fake + period_fake + C1_fake)

            assert not torch.isnan(d_loss), f'd_loss is NaN'
            d_loss.backward()
            optim_d.step()

            step += 1

            # logging
            g_loss_items.append(g_loss.item())
            d_loss_items.append(d_loss.item())
            stft_loss_items.append(stft_loss.item())

            if rank == 0:
                assert isinstance(loader, tqdm.tqdm), f'[{rank}] loader type mismatch: {type(loader).__name__}'
                loader.set_description(
                    f'g {np.mean(g_loss_items):.04f} '
                    f'd {np.mean(d_loss_items):.04f} '
                    f'stft {np.mean(stft_loss_items):.04f} '
                    f'| step {step}')

        if rank == 0 and epoch % 1 == 0:
            assert writer is not None, f'[{rank}] writer is None'
            writer.add_scalar('train/g_loss', np.mean(g_loss_items), epoch)
            writer.add_scalar('train/d_loss', np.mean(d_loss_items), epoch)
            writer.add_scalar('train/stft_loss', np.mean(stft_loss_items), epoch)

            with torch.no_grad():
                yn0 = istft(ynfftp[0] * spnfft[0])  # type: ignore[unbound-local-variable]
                y0 = yn0 + yh[0]                    # type: ignore[unbound-local-variable]
                writer.add_audio('train samples/yh', yh[0], epoch, fs)  # type: ignore[unbound-local-variable]
                writer.add_audio('train samples/xn', xn[0], epoch, fs)  # type: ignore[unbound-local-variable]
                writer.add_audio('train samples/yn', yn0,   epoch, fs)
                writer.add_audio('train samples/y',  y0,    epoch, fs)

        if rank == 0 and epoch % 1 == 0:
            save_path = os.path.join(pt_dir, '%s_%04d.jit.pt' % (h.session_name, epoch))
            print(f'Saving model to {save_path} ...')

            with torch.no_grad():
                model_g_detached = _create_generator()
                model_g_detached.load_state_dict(model_g_inner.state_dict())
                m_v01br.remove_parametrizations(model_g_detached)
                for param in model_g_detached.parameters():
                    param.requires_grad = False

                model_g_detached = torch.jit.script(model_g_detached)
                torch.jit.save(model_g_detached, save_path)

            assert logger is not None, f'[{rank}] logger is None'
            logger.info('Saved checkpoint to: %s' % save_path)


if __name__ == '__main__':
    np.set_printoptions(precision=4, linewidth=160, edgeitems=5, suppress=True)
    torch.set_printoptions(precision=4, linewidth=160, edgeitems=5, sci_mode=False)

    num_gpus = torch.cuda.device_count()
    print(f'Found {num_gpus} GPU(s), CUDA available: {torch.cuda.is_available()}')

    # for debugging purposes only
    rank = 0
    device = torch.device(f'cpu')

    h = TrainConfig(
        num_gpus=num_gpus,
        port=1234,
        num_workers=16,

        channel_sizes=[16, 12],

        lr=0.0001,
        lam_gloss=1.,
        lam_stft=1.,

        data_dir='@data/OpenSinger',
        batch_size=12,
        segment_length=22050,

        ckpt_dir='@ckpt',
        log_dir='@logs',
        session_name='0702',)

    # parse arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-p', '--port',          type=int, default=55192, help='Port number for distributed training')
    argparser.add_argument('-j', '--num_workers',   type=int, default=16,    help='Number of workers for data loading')

    argparser.add_argument('-c', '--channel_sizes', type=str, default='16,12', help='Channel sizes for the model, comma separated (e.g., 16,12)')

    argparser.add_argument('--lr',        type=float, default=0.0001, help='Learning rate for the optimizer')
    argparser.add_argument('--lam_gloss', type=float, default=1., help='Weight for the generator loss')
    argparser.add_argument('--lam_stft',  type=float, default=1., help='Weight for the STFT loss')

    argparser.add_argument('-i', '--data_dir',     type=str, default='@data/OpenSinger', help='Directory for the training data')
    argparser.add_argument('-b', '--batch_size',   type=int, default=12, help='Batch size for training')
    argparser.add_argument('--seg_len',            type=int, default=22050, help='Segment length for training')

    argparser.add_argument('--ckpt_dir',           type=str, default='@ckpt', help='Directory for saving checkpoints')
    argparser.add_argument('--log_dir',            type=str, default='@logs', help='Directory for saving logs')
    argparser.add_argument('-n', '--session_name', type=str, required=True, help='Name of the session for logging and saving checkpoints')
    args = argparser.parse_args()

    h = TrainConfig(
        num_gpus=num_gpus,
        port=args.port,
        num_workers=args.num_workers,

        channel_sizes=[int(x) for x in args.channel_sizes.split(',')],

        lr=args.lr,
        lam_gloss=args.lam_gloss,
        lam_stft=args.lam_stft,

        data_dir=args.data_dir,
        batch_size=args.batch_size,
        segment_length=args.seg_len,

        ckpt_dir=args.ckpt_dir,
        log_dir=args.log_dir,
        session_name=args.session_name,)

    print(str(h))

    # train
    if num_gpus > 1:
        mp.spawn(train, nprocs=num_gpus, args=(h,))
    else:
        train(0, h)


