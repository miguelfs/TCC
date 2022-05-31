import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from modulation_toolbox_py.filterbank.filterbanksynth import filterbanksynth
from modulation_toolbox_py.filterbank.filtersubbands import filtersubbands
from modulation_toolbox_py.index import index


def filterbankfreqz(filterbankparams: dict, nfft: int, fs: float = 2):
    plt.figure(figsize=(12,10))
    nfft = max(512, int(np.max(filterbankparams['afilterorders'])) + 1) if nfft is None else nfft
    for k in range(filterbankparams['numbands']):
        h = index(filterbankparams['afilters'], k)
        center = index(filterbankparams['centers'], k)
        h = vmult(h, np.exp(2j * np.pi / 2 * center * np.arange(len(h))))
        _, H = signal.freqz(h, 1, nfft, whole=True)

        H = 20 * np.log10(np.abs(H[0:int(np.floor(nfft / 2))]))
        plt.subplot(2, 1, 1, )
        frequency_axis = np.arange(int(np.floor(nfft / 2))) / nfft * fs
        plt.plot(frequency_axis, H)
        plt.xlim([0, fs // 2])
        plt.ylim([max(H) - 60, max(H) + 10])
    plt.title('Subband filter frequency')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('dB magnitude')
    # impulse response plot
    leftPad = int(np.floor((nfft - 1) / 2))
    rightPad = int(np.ceil((nfft - 1) / 2))
    impulse = np.vstack((np.zeros((leftPad, 1)), np.array([[1]]), np.zeros((rightPad, 1))))
    impSubbands = filtersubbands(impulse, filterbankparams)
    impResponse = filterbanksynth(impSubbands, filterbankparams)
    IR = 20 * np.log10(np.abs(np.fft.fft(impResponse, axis=-1 if impResponse.shape[0] == 1 else 0)))
    # IR = 20 * np.log10(np.abs(np.fft.fft(impResponse)))
    minMag = np.min(IR)
    maxMag = np.max(IR)
    minMag, maxMag = (minMag - 5, maxMag + 5) if maxMag - minMag < 10 else (minMag, maxMag)
    plt.subplot(2, 1, 2)
    # f = np.arange(np.shape(IR)[1]) * fs / len(IR)
    f = np.arange(np.shape(IR)[1]) * fs / np.shape(IR)[1]
    plt.plot(f, IR[0, :])
    plt.xlim([0, fs // 2])
    plt.ylim([minMag, maxMag])
    plt.title('Filterbank impulse response (analysis + reconstruction)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('dB magnitude')
    plt.show()


def vmult(x1, x2):  # multiplies two vectors element-wise, regardless of orientation. Output shape matches x1.
    return x1 * x2.reshape(x1.shape)
