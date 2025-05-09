import pathlib
import numpy as np
import librosa
import librosa.filters
import onnxruntime as ort

if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=4)


class MelSpectrogram:
    def __init__(self, n_mel_channels, sampling_rate, win_length, hop_length, n_fft=None, mel_fmin=0, mel_fmax=None, clamp=1e-5,):
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        self.mel_basis = librosa.filters.mel(sr=sampling_rate, n_fft=n_fft, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax, htk=True,).astype(np.float32)
        self.n_fft = win_length if n_fft is None else n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.clamp = clamp

    def __call__(self, audio):
        n_fft = self.n_fft
        win_length = self.win_length
        hop_length = self.hop_length
        fft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window='hann',
            center=True, pad_mode='reflect',)
        magnitude = np.abs(fft)

        mel_output = np.matmul(self.mel_basis, magnitude)
        log_mel_spec = np.log(np.clip(mel_output, a_min=self.clamp, a_max=None))
        return log_mel_spec


fs16k = 16000

class RMVPE:
    def __init__(self, hop_ms=20., ort_providers=['CPUExecutionProvider']):
        model_path = pathlib.Path('assets/[rmvpe]/rmvpe.onnx')
        self.session = ort.InferenceSession(model_path, providers=ort_providers)

        hop_16k = int(round(fs16k * hop_ms / 1000))
        self.mel_extractor = MelSpectrogram(n_mel_channels=128, sampling_rate=fs16k,
            win_length=1024, hop_length=hop_16k, n_fft=None, mel_fmin=30, mel_fmax=8000)

    def __call__(self, x: np.ndarray, fs: int):
        # resample if needed
        x = x.astype(np.float32)

        if fs != fs16k:
            import librosa
            x = librosa.resample(x, orig_sr=fs, target_sr=fs16k, res_type='fft')
            fs = fs16k

        x0 = x[None]
        xmel = self.mel_extractor(x0)

        n_frames = xmel.shape[-1]
        cents_map = 20 * np.arange(360) + 1997.3794084376191

        xmel = np.pad(xmel, [(0, 0), (0, 0), (0, 32 * ((n_frames - 1) // 32 + 1) - n_frames)], mode='reflect')
        logits, = self.session.run(None, {'input': xmel})
        logits = logits.squeeze(0)[:n_frames]

        center = np.argmax(logits, axis=1)  # 帧长#index

        logits_mask = np.abs(np.arange(360) - center[..., None]) <= 4    # 帧长,360
        logits_masked = logits_mask * logits
        cents_pred = np.sum(logits_masked * cents_map, axis=-1).astype(np.float32) / np.sum(logits_masked, axis=-1).clip(min=1e-9)  # 帧长

        confidence = np.max(logits, axis=-1)

        f0 = 10 * (2 ** (cents_pred / 1200))

        return f0, confidence


def threshold(f0, confidence, threshold=0.03):
    f0[confidence <= threshold] = 0
    return f0

def stonemask(x, fs, f0, hop_ms=20.):
    import pyworld as pw
    return pw.stonemask(x.astype(float), f0.astype(float), np.arange(len(f0)) * hop_ms / 1000, fs) # type: ignore


