import numpy as np
import matplotlib.pyplot as plt

from modulation_toolbox_py.index import index


def filterbankfreqz(filterbankparams: dict, nfft: int, fs: float = 2):
    nfft = max(512, max(filterbankparams['afilterorders']) + 1) if nfft is None else nfft
    for k in range(len(filterbankparams['numbands'])):
        h = index(filterbankparams['afilters'], k)
        center = index(filterbankparams['centers'], k)

        h = vmult(h, np.exp(2j * np.pi / 2 * center * list(range(0, h))))
        H = 20 * np.log10(np.abs(H[0:np.floor(nfft/2)]))
        plt.subplot(2,1,1)
        plt.plot(range(0,np.floor(nfft/2)/nfft * fs, H))
        plt.axis([0, fs/2, max(H)-60, max(H)+10])
    plt.title('Subband filter frequency')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('dB magnitude')
    leftPad = np.floor((nfft - 1)/2)
    rightPad = np.ceil((nfft-1)/2)
    impSubbands = filter


def vmult(x1, x2):  # multiplies two vectors element-wise, regardless of orientation. Output shape matches x1.
    return x1 * x2.reshape(x1.shape)