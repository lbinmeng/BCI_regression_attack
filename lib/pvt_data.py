import numpy as np
import scipy.io as sp
from glob import glob
import numpy as np
from scipy.signal import welch, get_window
from tqdm import tqdm


def data_loader_pvt(root, verbose=0, max_sub=16, npz=True):
    """
    load pvt data
    :param root: root path of pvt data
    :param verbose: if verbose > 0, show loading progress
    :param max_sub: total subject number, default=16
    :return:
    """
    if root[-1] != '/':
        root += '/'
    file_format = 'PVT_Sub{}_Sess{}.'
    if npz:
        file_format += 'npz'
    else:
        file_format += 'mat'
    # files = glob(root + file_format.format('*', '*'))
    # print('Tol %d file detected' % len(files))

    x, y, psd = [], [], []
    print("loading data", end='')
    for s in range(max_sub):
        print('.', end='', flush=True)
        files = glob(root + file_format.format(s + 1, '*'))
        sub_x = []
        sub_y = []
        sub_psd = []
        if verbose > 0:
            print('Subject %d, Sess: %d' % (s + 1, len(files)))
        for fn in files:
            if npz:
                f = np.load(fn)
                EEG = f['x']
                label = f['y']
                PSD = f['psd']
            else:
                f = sp.loadmat(fn)
                EEG = f['subSessEEG'].transpose([2, 0, 1])
                label = f['subSessRTs'].transpose()
                PSD = f['subSessPSD']
            sub_x.append(EEG)
            sub_y.append(label)
            sub_psd.append(PSD)
        x.append(np.concatenate(sub_x, axis=0))
        y.append(np.concatenate(sub_y, axis=0))
        psd.append(np.concatenate(sub_psd, axis=0))
    print('', end='\n')
    return np.array(x), np.array(y), np.array(psd)


def extract_PSD_welch(x, bands=None, fs=256):
    """

    :param x: n_Epochs * n_Channs * n_Samples
    :param bands: list type bands
    :return:
    """
    print('[Welch] bands: {}'.format(bands))
    N, C = x.shape[:2]
    n_b = len(bands)
    x_fea = np.zeros([N, C * n_b])
    for i in tqdm(range(N)):
        for b in range(n_b):
            psd = []
            lower = bands[b][0]
            upper = bands[b][1]
            for c in range(C):
                fpos, pxx = welch(x[i, c, :], fs=256, window=get_window('hamming', fs * 1))
                mean_psd = np.mean(pxx[(fpos >= lower) & (fpos <= upper)])
                psd.append(mean_psd)
            x_fea[i, b::n_b] = psd
    return x_fea


if __name__ == '__main__':
    pvt_data = data_loader_pvt('/mnt/disk1/yqcui/attack/pvt_raw_data/data')
    eeg = pvt_data[0][0]
    label = pvt_data[1][0]
    psd = pvt_data[2][0]
    pvt_fft = np.abs(np.fft.fftn(eeg[0, :, :], axes=[1]))
    pvt_psd = pvt_fft[:, :int(pvt_fft.shape[1] / 2)] ** 2 / pvt_fft.shape[1]
    pvt_feature = np.mean(pvt_psd[:, 20:40], axis=1, keepdims=True)
    pvt_feature = np.concatenate([pvt_feature, np.mean(pvt_psd[:, 40:65], axis=1, keepdims=True)], axis=1)
    pvt_feature = np.reshape(pvt_feature, [62 * 2])
    fpos, pxx = welch(eeg[0, 0, :], fs=256, window=get_window('hamming', 256 * 1))
    print('a')
