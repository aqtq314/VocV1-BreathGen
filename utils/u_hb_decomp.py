import pathlib
import numpy as np
import librosa
import pyworld as pw


fs44k = 44100
fft_size = 2048
hop_samples = 441
hop_ms = hop_samples * 1000 / fs44k


def ensure_fs(x: np.ndarray, fs: int, target_fs: int, res_type='fft'):
    if len(x.shape) != 1:
        raise Exception(f'x is not mono (shape is {x.shape})')
    if fs != target_fs:
        x = librosa.resample(x, orig_sr=fs, target_sr=target_fs, res_type=res_type)
        fs = target_fs
    return x, fs

def stft(xs, fft_size, hop_samples):
    return librosa.stft(xs, n_fft=fft_size, hop_length=hop_samples, window='hann', center=True, pad_mode='reflect').swapaxes(-1, -2)

def istft(ffts, hop_samples):
    return librosa.istft(ffts.swapaxes(-1, -2), hop_length=hop_samples, window='hann', center=True)

def cheaptrick(x, f0, t, fs, pwFftSize=None, normAmp=False):
    pwFftSize = pwFftSize or pw.get_cheaptrick_fft_size(fs)  # type: ignore
    windowArea = pwFftSize / 2
    spxfft = pw.cheaptrick(x.astype(np.float64), f0, t, fs, fft_size=pwFftSize)  # type: ignore
    if normAmp:
        spxfft = np.sqrt(spxfft * (fs / f0[:, None]))
    else:
        spxfft = np.sqrt(spxfft * (f0[:, None] / fs)) * windowArea
    spxfft = np.where((f0 > 0)[:, None], spxfft, 1e-12)
    return spxfft.astype(x.dtype)

class HBDecomposer:
    def __init__(self, ort_providers=['CPUExecutionProvider']):
        import onnxruntime as ort
        model_path = pathlib.Path('assets/[vocv1-hbdecomp][220228-001211][img73]modelv5c/model.onnx')
        self.session = ort.InferenceSession(str(model_path), providers=ort_providers)

    def __call__(self, x: np.ndarray, fs: int):
        x, fs = ensure_fs(x, fs, fs44k)
        xfftc = stft(x, fft_size, hop_samples)

        onnx_in = {'X': np.stack([xfftc.real, xfftc.imag], axis=-1)[None].astype(np.float32)}
        onnx_out, = self.session.run(None, onnx_in)
        xhfftc = (onnx_out[0, ..., 0] + onnx_out[0, ..., 1] * 1j)
        xnfftc = (onnx_out[0, ..., 2] + onnx_out[0, ..., 3] * 1j)

        xh = istft(xhfftc, hop_samples)
        xn = istft(xnfftc, hop_samples)

        return xh, xn


