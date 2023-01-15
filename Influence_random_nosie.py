import numpy as np
import tensorflow as tf
import os
from lib import data_loader_pvt, data_split, plot_raw
from sklearn.linear_model import Ridge
from PVT_models import DensNet, TFRidge
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
e_setting = 'within'  # within or cross
model_name = 'ridge'  # ridge or dnn
ckpt_dir = 'runs/influence_of_parameters/'


def success_rate(y, a_y, target):
    rate = 0
    for i in range(len(y)):
        if a_y[i] > y[i] + target:
            rate += 1
    return rate / len(y)


# ------------------------------
pvt_data = data_loader_pvt('/mnt/disk2/mlb/dataset/PVT', npz=False)
clean = []
cw = []
llc = []
noise = []
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

source_number = 1
# within-subject
eeg = pvt_data[0][source_number]
label = pvt_data[1][source_number]

# normalize
for i in range(eeg.shape[0]):
    for j in range(eeg.shape[1]):
        max = np.max(eeg[i][j])
        min = np.min(eeg[i][j])
        eeg[i][j] = (eeg[i][j] - min) / (max - min)

dataset = data_split(eeg, label)
train_data = dataset['train']
test_data = dataset['test']

sess = tf.Session()

if model_name == 'ridge':
    # ———————————————————————— using ridge model ———————————————————————— #
    # compute the PSD features
    # theta band 4-7.5 Hz; Alpha band 7.5-12 Hz
    # Sampling rate 256Hz
    # get w and b
    pvt_fft = np.fft.fftn(train_data[0], axes=[2])
    pvt_psd = np.real(pvt_fft * np.conj(pvt_fft)) / 1280
    pvt_psd = pvt_psd[:, :, :640]
    pvt_feature = np.mean(pvt_psd[:, :, 20:40], axis=2, keepdims=True)
    pvt_feature = np.concatenate([pvt_feature, np.mean(pvt_psd[:, :, 40:65], axis=2, keepdims=True)], axis=2)
    pvt_feature = np.reshape(pvt_feature, [-1, 62 * 2])

    ridge = Ridge(alpha=0.1)
    ridge.fit(pvt_feature, train_data[1])
    w, b = ridge.coef_.T, ridge.intercept_
    model = TFRidge(sess)
    model.set_weight(w, b)
    # eval
    c_y, c_loss, c_avg, c_cc = model.eval(test_data)
else:
    # ———————————————————————— using DNN model ———————————————————————— #
    model = DensNet(sess=sess, input_shape=[train_data[0].shape[1], train_data[0].shape[2]],
                    model_path=ckpt_dir + '/model_pvt_' + e_setting + '/subject_' + str(source_number))

    # train DNN
    print('train dnn')
    model.train(train_data)
    print('dnn eval')
    c_y, c_loss, c_avg, c_cc = model.eval(test_data)

# ———————————————————————— Gaussian noise ———————————————————————— #
# loss = []
# mo = []
# distortion = []
# asr = []
# amplitude = 0.0
# for i in range(200):
#     amplitude = i * 0.0001
#     mu = 0
#     sigma = 1
#     random_noise = amplitude * np.random.normal(mu, sigma, size=train_data[0].shape[2])
#     l2_noise = np.sqrt(np.sum(np.square(random_noise))) * train_data[0].shape[1]
#     print('amplitude = {}'.format(amplitude))
#     noise_data = test_data[0] + random_noise
#     n_y, n_loss, n_avg, n_cc = model.eval([noise_data, test_data[1]])
#     asr_temp = success_rate(test_data[1], n_y, 0.2)
#
#     loss.append(n_loss)
#     mo.append(n_avg)
#     distortion.append(l2_noise)
#     asr.append(asr_temp)
#
# np.savez(ckpt_dir + 'random_noise', mse=loss, mo=mo, distortion=distortion, asr=asr)
#
# x = np.arange(len(loss)) * 0.0001 + 0.0001
# # plt.figure(figsize=(8, 6))
# # plt.plot(x, asr, 'b', linewidth=3.0)
# # plt.ylabel('ASR', fontsize=24)
# # plt.xlabel(r'$\sigma$', fontsize=24)
# # # plt.legend(loc='lower right', fontsize=14)
# # plt.tick_params(labelsize=20)
# # plt.tight_layout()
# # plt.savefig('runs/influence_of_parameters' + '/random_noise_asr' + '.eps')
#
# plt.figure(figsize=(8, 6))
# plt.plot(x, distortion, 'r', linewidth=3.0)
# plt.ylabel('Mean Distortion', fontsize=24)
# plt.xlabel(r'$\sigma$', fontsize=24)
# # plt.legend(loc='lower right', fontsize=14)
# plt.tick_params(labelsize=20)
# # plt.tight_layout()
# plt.savefig('runs/influence_of_parameters' + '/random_noise_distortion' + '.eps')
#
# plt.figure(figsize=(8, 6))
# plt.plot(x, mo, 'b', linewidth=3.0)
# plt.ylabel('Mean output', fontsize=24)
# plt.xlabel(r'$\sigma$', fontsize=24)
# # plt.legend(loc='lower right', fontsize=14)
# plt.tick_params(labelsize=20)
# # plt.tight_layout()
# plt.savefig('runs/influence_of_parameters' + '/random_noise_MO' + '.eps')

mu = 0
sigma = 1
random_noise = 0.02 * np.random.normal(mu, sigma, size=train_data[0].shape[2])
l2_noise = np.sqrt(np.sum(np.square(random_noise))) * train_data[0].shape[1]
noise_data = test_data[0] + random_noise
plot_raw(test_data[0][0], noise_data[0],'noise_example')