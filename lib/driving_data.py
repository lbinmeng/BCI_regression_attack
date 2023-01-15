import scipy.io as scio
import numpy as np
from lib import utils
import h5py as h5
import matplotlib.pyplot as plt

maxEEG = 20


def load(is_normal):
    f = scio.loadmat('/mnt/disk2/mlb/dataset/drivingdata.mat')
    x = f['X_data'][0]
    y = f['Y_data'][0]
    nEvents = f['nEvents'][0]

    for ids in range(len(nEvents)):
        badEpoch = np.min(x[ids], axis=1) >= 10
        x[ids] = x[ids][~badEpoch]
        y[ids] = y[ids][~badEpoch]

    all = np.concatenate(x, axis=0)

    badchannels = np.max(all, axis=0) >= maxEEG
    print(badchannels)

    for ids in range(len(nEvents)):
        x[ids] = x[ids][:, ~badchannels]
        if is_normal:
            max = np.max(x[ids])
            min = np.min(x[ids])
            x[ids] = (x[ids] - min) / (max - min)

    return x, y, len(nEvents)

def driving_load(source_number, split=True, is_normal=True):
    x, y, _ = load(is_normal)
    source_data = x[source_number]
    source_label = y[source_number]

    if split:
        return utils.data_split(source_data, source_label, [9, 1])
    else:
        return source_data, source_label


def driving_cross_subject(source_number, is_normal=True):
    dataset = {}
    x, y, _ = load(is_normal)

    temp_x = np.concatenate([x[:source_number], x[source_number + 1:]], axis=0)
    temp_y = np.concatenate([y[:source_number], y[source_number + 1:]], axis=0)
    train_x = temp_x[0]
    train_y = temp_y[0]
    for i in range(1, len(temp_x)):
        train_x = np.concatenate([train_x, temp_x[i]], axis=0)
        train_y = np.concatenate([train_y, temp_y[i]], axis=0)

    test_x = x[source_number]
    test_y = y[source_number]

    dataset['train'] = [train_x, train_y]
    dataset['test'] = [test_x, test_y]

    return dataset


def data_loader_driving(root):
    """ import raw driving data """
    f = h5.File(root, 'r')
    EEGs = np.array(f.get('EEGs'))
    RTs = np.array(f.get('resTime'))
    nEvents = np.array(f.get('length'))

    x = []
    y = []
    idx_l = 0
    idx_r = 0
    for num in nEvents:
        idx_r += int(num)
        x.append(EEGs[idx_l:idx_r])
        y.append(RTs[idx_l:idx_r])
        idx_l = idx_r

    return np.asarray(x), np.asarray(y)


if __name__ == '__main__':
    # x, y = data_loader_driving('/mnt/disk1/yqcui/dataset/driving/2017_DL_Driving_removeBadEpoch.hdf5')
    x, y= data_loader_driving('/mnt/disk1/yqcui/dataset/driving/2017_DL_Driving_removeBadEpoch.hdf5')
    clean = x[0][0]
    channels = clean.shape[0]
    x = np.arange(clean.shape[1])
    plt.plot(x, clean[0], linewidth=1, color='dodgerblue', label='Original sample')  # plot clean data
    for i in range(1, channels):
        plt.plot(x, clean[i] + i, linewidth=1, color='dodgerblue')  # plot clean data
    plt.ylabel('channels')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.savefig('plt1.eps')
    plt.show()
    print('a')
