import sys
import os
import pathlib
import shutil
import itertools
import argparse
import tqdm
import multiprocessing as mp
import concurrent.futures
import numpy as np
import soundfile as sf
import torch
from utils import u_exciter_np, u_hb_decomp, u_rmvpe

if __name__ == '__main__':
    mp.set_start_method('spawn')
    np.set_printoptions(suppress=True)


hb_decomp = None
rmvpe = None
pulse_train = None

def _proc_init(gpu_indices_queue: mp.Queue):
    global hb_decomp, rmvpe, pulse_train

    gpu_index = gpu_indices_queue.get(timeout=1)
    ort_providers = [('CUDAExecutionProvider', {'device_id': gpu_index,})]
    hb_decomp = u_hb_decomp.HBDecomposer(ort_providers=ort_providers)
    rmvpe = u_rmvpe.RMVPE(hop_ms=u_hb_decomp.hop_ms, ort_providers=ort_providers)
    pulse_train = u_exciter_np.PulseTrain(u_hb_decomp.hop_ms, u_hb_decomp.fs44k, normAmp=False)

def _proc_utt(args):
    assert hb_decomp is not None, f'{hb_decomp=} has not been initialized'
    assert rmvpe is not None, f'{rmvpe=} has not been initialized'
    assert pulse_train is not None, f'{pulse_train=} has not been initialized'

    in_data_dir, out_data_dir, in_wav_path_rel = args

    try:
        x, fs = sf.read(pathlib.Path(in_data_dir, in_wav_path_rel), dtype='float32')
        assert len(x.shape) == 1, f'Input is not mono (shape is {x.shape})'
        if len(x) / fs >= 20:
            raise Exception(f'Duration too long: {len(x) / fs:.4f} sec')
        if len(x) / fs < 1:
            raise Exception(f'Duration too short: {len(x) / fs:.4f} sec')

        # volume down to avoid clipping during resampling
        x *= 0.8
        x, fs44k = u_hb_decomp.ensure_fs(x, fs, target_fs=u_hb_decomp.fs44k)

        # harmonics-breathiness decomposition
        xh, xn = hb_decomp(x, fs44k)

        # f0 estimation
        f0, pd = rmvpe(x, fs44k)

        _hop_samples = int(round(fs44k * u_hb_decomp.hop_ms / 1000))
        # sometimes rmvpe returns one more frame under fs=16k because of rounding-up on sample count during resampling
        xh = xh[:len(xh) // _hop_samples * _hop_samples]
        xn = xn[:len(xn) // _hop_samples * _hop_samples]
        f0 = f0[:len(x) // _hop_samples + 1]
        pd = pd[:len(x) // _hop_samples + 1]

        f0 = u_rmvpe.threshold(f0, pd, threshold=0.03)
        assert np.all(np.isfinite(f0)), f'F0 non finite'
        f0 = u_rmvpe.stonemask(x, fs44k, f0, hop_ms=u_hb_decomp.hop_ms)

        # synthesize harmonics (periodic part) using pulse train
        p = pulse_train(f0).astype(np.float32) / (u_exciter_np.fftSize / 2)   # area of a size-2048 hann window
        p = p[:len(xh)]

        t = np.arange(len(f0)) * u_hb_decomp.hop_ms / 1000
        sph4fft = u_hb_decomp.cheaptrick(xh, f0, t, fs44k, normAmp=False).astype(np.float32)
        pfftc = u_hb_decomp.stft(p, u_hb_decomp.fft_size, u_hb_decomp.hop_samples)[..., :sph4fft.shape[-2], :]
        yh4 = u_hb_decomp.istft(sph4fft * pfftc, u_hb_decomp.hop_samples)

        # save results
        out_xn_path = pathlib.Path(out_data_dir, 'xn', in_wav_path_rel)
        out_xn_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_xn_path, xn, fs44k, subtype='PCM_16')

        out_yh4_path = pathlib.Path(out_data_dir, 'yh4', in_wav_path_rel)
        out_yh4_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_yh4_path, yh4, fs44k, subtype='PCM_16')

    except Exception as ex:
        return RuntimeError(f'[{in_wav_path_rel}] {ex}')


if __name__ == '__main__':
    # parse arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-i', '--in_dir', type=str, required=True, help=r'Original OpenSinger dataset path with subfolders "ManRaw" and "WomanRaw" (e.g. E:\Datasets\OpenSinger)')
    argparser.add_argument('-o', '--out_dir', type=str, default='@data/OpenSinger', help=r'Output processed dataset path (default: @data/OpenSinger)')
    argparser.add_argument('-d', '--gpus', type=str, default=None, help=r'GPU indices to use, comma-separated (e.g. 0,1,2) (default: all available GPUs)')
    args = argparser.parse_args()

    # in_data_dir = pathlib.Path('~/datasets/OpenSinger').expanduser()
    # out_data_dir = pathlib.Path('@data/OpenSinger')
    in_data_dir  = pathlib.Path(args.in_dir).expanduser()
    out_data_dir = pathlib.Path(args.out_dir).expanduser()
    if not in_data_dir.is_dir():
        raise Exception(f'Input directory does not exist: {in_data_dir}')

    # check GPU info
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        raise Exception(
            'This script uses a combination of PyTorch and ONNX Runtime with CUDA backend. '
            'PyTorch gets affected by the environment variable CUDA_VISIBLE_DEVICES, but ONNX Runtime does not. '
            'It is recommended to use command line arguments "--gpus" to specify GPU device indices instead. ')

    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        raise Exception('No GPU available')

    gpu_indices = [int(gpu_index_str) for gpu_index_str in args.gpus.split(',')] if args.gpus else range(torch.cuda.device_count())
    gpu_names: dict[int, str] = {}
    gpu_vrams: dict[int, float] = {}
    for gpu_index in gpu_indices:
        gpu_names[gpu_index] = torch.cuda.get_device_name(gpu_index)
        gpu_vrams[gpu_index] = torch.cuda.get_device_properties(gpu_index).total_memory / 1073741824
    for gpu_index in gpu_indices:
        print(f'> GPU {gpu_index}: {gpu_names[gpu_index]} ({gpu_vrams[gpu_index]:.2f} GB)')

    # clear existing data
    if out_data_dir.is_dir():
        print(f'Clearing existing data in {out_data_dir} ...')
        for path in out_data_dir.iterdir():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

    # traverse input directory
    print(f'Traversing input directory ...')
    in_wav_paths_rel = [path.relative_to(in_data_dir) for path in in_data_dir.rglob('*.wav')]
    in_wav_paths_rel = sorted(in_wav_paths_rel, key=lambda in_wav_paths_rel: -pathlib.Path(in_data_dir, in_wav_paths_rel).stat().st_size)
    print(f'Found {len(in_wav_paths_rel)} audio files')

    # spawn child worker processes
    # gpu_indices_content = gpu_indices
    gpu_indices_content = [gpu_index for gpu_index in gpu_indices for _ in range(int(gpu_vrams[gpu_index] / 14.2))]  # assuming each child process uses at most 14.2 GB of VRAM
    gpu_indices_queue = mp.Queue(len(gpu_indices_content))
    _ = [gpu_indices_queue.put(i) for i in gpu_indices_content]

    with concurrent.futures.ProcessPoolExecutor(max_workers=len(gpu_indices_content), initializer=_proc_init, initargs=(gpu_indices_queue,)) as executor:
        args_list = [(in_data_dir, out_data_dir, in_wav_path_rel) for in_wav_path_rel in in_wav_paths_rel]
        for result in tqdm.tqdm(executor.map(_proc_utt, args_list), total=len(in_wav_paths_rel)):
            if isinstance(result, Exception):
                tqdm.tqdm.write(f'{result}', file=sys.stderr)
            else:
                pass


