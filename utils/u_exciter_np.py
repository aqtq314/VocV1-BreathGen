import numpy as np
import scipy.fft
import librosa
import onnxruntime as ort
import pyworld as pw


fftSize = 2048

class Kaiser():
    def __init__(self, windowRadius, zeroCrossings, rolloff, kaiserBeta):
        self.windowRadius = windowRadius
        self.zeroCrossings = zeroCrossings
        self.rolloff = rolloff
        self.kaiserBeta = kaiserBeta

        kaiserWindow = np.kaiser(windowRadius * 2 + 1, kaiserBeta)[windowRadius:]
        self.outWindow = kaiserWindow * np.sinc(np.linspace(0, zeroCrossings * rolloff, num=windowRadius + 1)) * rolloff
        self.xs = np.linspace(0., zeroCrossings, num=windowRadius + 1)

    def __call__(self, x):
        return np.interp(np.abs(x), self.xs, self.outWindow, right=0.)

class PulseTrain():
    def __init__(self, hopSize: float, fs: int, initUnitPhase=0.5, normKaiser=True, normAmp=True):
        self.hopSize = hopSize      # usually 10.
        self.fs = fs
        self.initUnitPhase = initUnitPhase
        self.normKaiser = normKaiser
        self.normAmp = normAmp

        self.hopSamplesF = self.hopSize * fs / 1000.
        self.hopSamples = int(round(self.hopSamplesF))
        assert self.hopSamples == self.hopSamplesF, f'HopSamples must be integer, but received {self.hopSamplesF}'

        # params stolen from https://github.com/bmcfee/resampy/blob/819621f1555742848826d7cf448c446aa0ccc08f/resampy/filters.py
        self.kaiser = Kaiser(32768, zeroCrossings=64, rolloff=0.9475937167399596, kaiserBeta=14.769656459379492)
        self.kaiserCompensation = np.abs(np.fft.rfft(self.kaiser(np.arange(-fftSize // 2, fftSize // 2, dtype=float))))

    def __call__(self, f0: np.ndarray):
        hopSize = self.hopSize
        fs = self.fs
        initUnitPhase = self.initUnitPhase
        hopSamplesF = self.hopSamplesF
        hopSamples = self.hopSamples
        kaiser = self.kaiser
        h = self.hopSize / 1000.
        pt = np.zeros((hopSamples * len(f0),),)

        pulseTime = []
        pulseValue = []
        y0 = initUnitPhase
        for i, (currF0, nextF0) in enumerate(zip(f0, np.array([*f0[1:], 0.]))):
            currSi = i * hopSamples
            unitSampleXs = np.arange(hopSamples, dtype=float) / hopSamples

            if currF0 > 0. and nextF0 > 0.:
                v0, v1 = currF0, nextF0

                y1 = y0 + h * (v0 + v1) / 2.
                k = np.arange(1, int(y1) + 1, dtype=float)
                pts = (k - y0) / v0 if v0 == v1 else (
                    (h * v0 - np.sqrt(h * (h * v0 * v0 - 2. * (v0 - v1) * (k - y0)))) / (v0 - v1))

                pss = np.ones_like(pts)
                pt[currSi:(currSi + hopSamples)] = -(v0 + (v1 - v0) * unitSampleXs) / fs

            elif currF0 > 0. or nextF0 > 0.:
                v = currF0 if currF0 > 0. else nextF0

                y1 = y0 + h * v
                k = np.arange(1, int(y1) + 1, dtype=float)
                pts = (k - y0) / v

                if currF0 <= 0.:    # fade in
                    pss = np.sin(pts / h * np.pi / 2.) ** 2.
                    pt[currSi:(currSi + hopSamples)] = -(v / fs * np.sin(unitSampleXs * np.pi / 2.) ** 2.)

                else:  # fade out
                    pss = np.cos(pts / h * np.pi / 2.) ** 2.
                    pt[currSi:(currSi + hopSamples)] = -(v / fs * np.cos(unitSampleXs * np.pi / 2.) ** 2.)

            else:
                continue

            pulseTime.extend(pts + i * h)
            pulseValue.extend(pss)

            y0 = y1 % 1.

        pulseTime = np.array(pulseTime)
        pulseValue = np.array(pulseValue)

        pulseSampleTime = pulseTime * fs
        ptWriteIndices2D = np.floor(pulseSampleTime).astype(np.int64) + np.arange(kaiser.zeroCrossings, -kaiser.zeroCrossings, step=-1)[:, None]
        ptWriteValues2D = kaiser(ptWriteIndices2D - pulseSampleTime) * pulseValue
        ptWriteIndexValid = (ptWriteIndices2D >= 0) & (ptWriteIndices2D < pt.shape[0])
        for i in range(pulseTime.shape[0]):
            ptWriteIndices = np.extract(ptWriteIndexValid[:, i], ptWriteIndices2D[:, i])
            ptWriteValues = np.extract(ptWriteIndexValid[:, i], ptWriteValues2D[:, i])
            pt[ptWriteIndices] += ptWriteValues

        # optional high-frequency amplitude compensation for kaiser window
        if self.normKaiser:
            pfftc = librosa.stft(pt, n_fft=fftSize, hop_length=hopSamples, pad_mode='reflect').T
            pfftc /= self.kaiserCompensation
            pt = librosa.istft(pfftc.T, hop_length=hopSamples)

        # amplitude normalization
        if not self.normAmp:
            for i, (currF0, nextF0) in enumerate(zip(f0, np.r_[f0[1:], [0.]])):
                currSi = i * hopSamples

                if currF0 > 0. and nextF0 > 0.:
                    v0, v1 = currF0, nextF0
                    pt[currSi:(currSi + hopSamples)] /= np.linspace(v0, v1, hopSamples, endpoint=False) / fs

                elif currF0 > 0. or nextF0 > 0.:
                    v = currF0 if currF0 > 0. else nextF0
                    pt[currSi:(currSi + hopSamples)] /= v / fs

        return pt


