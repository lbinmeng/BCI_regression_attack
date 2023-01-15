"""
Last updated: 2018.09.14
This file is used to provide some methods for visualization.
Please email to xiao_zhang@hust.edu.cn if you have any questions.
"""
import matplotlib.pyplot as plt
from scipy.misc import imshow
from sklearn.metrics import confusion_matrix
import numpy as np


def plot_data(data):
    data = data.reshape(-1)
    plt.hist(data, bins=500)
    plt.show()


def plot_loss(train_loss, val_loss, title=None):
    if not title:
        title = 'Train Loss and Val Loss'
    if not len(train_loss) == len(val_loss):
        raise TypeError('The lengths of train_loss and val_loss should be the same!')

    x = range(len(train_loss))
    plt.plot(x, train_loss, label='train')
    plt.plot(x, val_loss, label='val')

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title(title)


def show_x_and_adversarial_x(clean_x, adversarial_x, file_name):
    """
    This function is used to plot x, adversarial x and the normalized (adv_x-x).
    :param x_list: the list of the normal images.
    :param adv_x_list: the list of the adversarial images.
    """

    dif = adversarial_x - clean_x
    dif_heat = (dif - np.min(dif)) / (np.max(dif) - np.min(dif))

    x = np.arange(len(clean_x))
    plt.plot(x, clean_x, color='red', linewidth=1.5, label='clean')
    plt.plot(x, adversarial_x, color='dodgerblue', linewidth=1.5, label='adversarial')
    # plt.plot(x, dif_heat, label='noise')
    plt.legend()
    plt.savefig(file_name + '.eps')
    plt.show()


# def show_predict(pre_clean, pre_fgsm, pre_bim, file_name):
#     x = np.arange(len(pre_clean))
#     plt.plot(x, pre_clean, label='pre_clean')
#     plt.plot(x, pre_fgsm, label='pre_L1')
#     plt.plot(x, pre_bim, label='pre_L2')
#     plt.legend()
#     plt.savefig(file_name + '.png')
#     plt.show()

def show_predict(data, labels, file_name):
    x = np.arange(len(data[0]))
    l = []
    for i, (y, label) in enumerate(zip(data, labels)):
        l_temp, = plt.plot(x, y, label=label)
        l.append(l_temp)
    plt.ylabel('Output', fontsize=16)
    plt.xlabel('Sample', fontsize=16)
    plt.tick_params(labelsize=13)
    plt.ylim([0.35, 1.25])
    temp_y = np.arange(8) * 0.1 + 0.4
    y_names = ['0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0', '1.1']
    plt.yticks(temp_y, y_names, fontsize=13)
    plt.legend(handles=[l[6], l[3], l[0], l[4], l[1], l[5], l[2]],
               labels=[labels[6], labels[3], labels[0], labels[4], labels[1], labels[5], labels[2]], loc='upper center',
               ncol=4,
               fontsize=10)
    plt.savefig(file_name + '.eps')


def plot_raw(clean, adv, file_name):
    plt.figure()
    channels = clean.shape[0]
    x = np.arange(clean.shape[1]) * 1.0 / 256
    l1, = plt.plot(x, adv[0] - np.mean(adv[0]), linewidth=1.0, color='red', label='Adversarial sample')  # plot adv data
    l2, = plt.plot(x, clean[0] - np.mean(adv[0]), linewidth=1.0, color='dodgerblue',
                   label='Original sample')  # plot clean data
    for i in range(1, 10):
        plt.plot(x, adv[i] + i - np.mean(adv[i]), linewidth=1.0, color='red')  # plot adv data
        plt.plot(x, clean[i] + i - np.mean(adv[i]), linewidth=1.0, color='dodgerblue')  # plot clean data

    plt.xlabel('Time (s)', fontsize=12)
    plt.ylim([-1, 10.5])
    temp_y = np.arange(10)
    y_names = ['Channel {}'.format(int(y_id)) for y_id in temp_y]
    plt.yticks(temp_y, y_names, fontsize=10)
    plt.legend(handles=[l2, l1], labels=['Original sample', 'Adversarial sample'], loc='upper right', ncol=2,
               fontsize=10)
    plt.savefig(file_name + '.eps')
