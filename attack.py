import numpy as np
import tensorflow as tf
import os
from lib import plot_raw, data_loader_pvt, data_loader_driving, data_split, show_predict
from sklearn.linear_model import Ridge
from Driving_models import DensNet, TFRidge
# from PVT_models import DensNet, TFRidge
import methods
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '6'
ckpt_dir = 'runs/attack'


def success_rate(y, a_y, target):
    rate = 0
    for i in range(len(y)):
        if a_y[i] > y[i] + target:
            rate += 1
    return rate / len(y)


# ************ generate the example of adversarial examples in PVT ************ #
source_number = 0
pvt_data = data_loader_pvt('/mnt/disk2/mlb/dataset/PVT', npz=False)
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

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

# ———————————————————————— using ridge model ———————————————————————— #
# compute the PSD features
# theta band 4-7.5 Hz; Alpha band 7.5-12 Hz
# Sampling rate 256 Hz
pvt_fft = np.fft.fftn(train_data[0], axes=[2])
pvt_psd = np.real(pvt_fft * np.conj(pvt_fft)) / 1280
pvt_psd = pvt_psd[:, :, :640]
pvt_feature = np.mean(pvt_psd[:, :, 20:40], axis=2, keepdims=True)
pvt_feature = np.concatenate([pvt_feature, np.mean(pvt_psd[:, :, 40:65], axis=2, keepdims=True)], axis=2)
pvt_feature = np.reshape(pvt_feature, [-1, 62 * 2])

# get w and b
ridge = Ridge(alpha=0.1)
ridge.fit(pvt_feature, train_data[1])
w, b = ridge.coef_.T, ridge.intercept_
model = TFRidge(sess)
model.set_weight(w, b)
# eval
c_y, c_loss, c_avg, c_cc = model.eval(test_data)

# ———————————————————————— C&W attack ———————————————————————— #
cw_attack = methods.CWL2(sess=sess,
                         model=model,
                         initial_c=0.001,
                         batch_size=20,
                         learning_rate=5e-4,
                         target=0.2,
                         binary_search_step=9,
                         input_shape=[train_data[0].shape[1], train_data[0].shape[2]],
                         max_iteration=1000)

adv_cw, l2_cw = cw_attack.attack(test_data[0])
cw_y, cw_loss, cw_avg, cw_cc = model.eval([adv_cw, test_data[1]])

# ———————————————————————— IFGSM ———————————————————————— #
asr = 0
for i in range(1, 30):
    eps = i * 0.001
    least_like_class = methods.LeastLikeClass(sess=sess,
                                              model=model,
                                              target=0.2,
                                              eps=eps,
                                              alpha=0.001,
                                              iteration=25)
    adv_l, l2_l = least_like_class.attack([test_data[0], c_y.reshape(test_data[1].shape)])
    l_y, l_loss, l_avg, l_cc = model.eval([adv_l, test_data[1]])

    if success_rate(c_y, l_y, 0.2) - asr < 0.001 and success_rate(c_y, l_y, 0.2) > 0.9:
        break
    else:
        asr = success_rate(c_y, l_y, 0.2)

ASR_cw = success_rate(c_y, cw_y, 0.2)
ASR_ifgsm = success_rate(c_y, l_y, 0.2)
print('ASR_C&W = {}, ASR_IFGSM = {}'.format(ASR_cw, ASR_ifgsm))

# visualization the adversarial example
plot_raw(test_data[0][1], adv_cw[1], ckpt_dir + '/diff_cw_pvt')
plot_raw(test_data[0][1], adv_l[1], ckpt_dir + '/diff_ifgsm_pvt')
np.savez(ckpt_dir + '/adv_dnn_pvt.npz', adv_cw=adv_cw, adv_ifgsm=adv_l, clean=test_data[0], y_cw=cw_y, y_ifgsm=l_y,
         y_clean=c_y)
#
# # # ************ generate the example of adversarial examples in Driving ************ #
# x, y = data_loader_driving('/mnt/disk1/Dataset_Driving/2017_DL_Driving_250Hz.hdf5')
# # ------------------------------
# if not os.path.exists(ckpt_dir):
#     os.makedirs(ckpt_dir)
# # within-subject
# eeg = x[0]
# label = y[0]
#
# # normalize
# for i in range(eeg.shape[0]):
#     for j in range(eeg.shape[1]):
#         max = np.max(eeg[i][j])
#         min = np.min(eeg[i][j])
#         eeg[i][j] = (eeg[i][j] - min) / (max - min)
#
# dataset = data_split(eeg, label)
# train_data = dataset['train']
# test_data = dataset['test']
#
# sess = tf.Session()
#
# # ———————————————————————— using ridge model ———————————————————————— #
# # compute the PSD features
# # theta band 4-7.5 Hz; Alpha band 7.5-12 Hz
# # Sampling rate 250 Hz
# channels = [0, 1, 4, 5, 6, 7, 10, 11, 16, 17, 19, 20, 21, 25, 32, 33, 36, 37, 38, 39, 42, 43, 44, 45, 46,
#             47, 48, 49,
#             52, 53, 54, 55, 58, 59]
# pvt_fft = np.fft.fftn(train_data[0], axes=[2])
# pvt_psd = np.real(pvt_fft * np.conj(pvt_fft)) / 7500
# pvt_psd = pvt_psd[:, :, :3750]
# pvt_feature = np.mean(pvt_psd[:, :, 60:115], axis=2, keepdims=True)
# pvt_feature = np.concatenate([pvt_feature, np.mean(pvt_psd[:, :, 116:180], axis=2, keepdims=True)], axis=2)
# pvt_feature = np.reshape(pvt_feature, [-1, 30 * 2])
# pvt_feature = pvt_feature[:, channels]
#
# # get w and b
# ridge = Ridge(alpha=0.1)
# ridge.fit(pvt_feature, train_data[1])
# w, b = ridge.coef_.T, ridge.intercept_
# model = TFRidge(sess)
# model.set_weight(w, b)
# # eval
# c_y, c_loss, c_avg, c_cc = model.eval(test_data)
#
# # ———————————————————————— C&W attack ———————————————————————— #
# cw_attack = methods.CWL2(sess=sess,
#                          model=model,
#                          initial_c=0.01,
#                          batch_size=20,
#                          learning_rate=1e-3,
#                          target=0.2,
#                          binary_search_step=9,
#                          input_shape=[train_data[0].shape[1], train_data[0].shape[2]],
#                          max_iteration=1000)
#
# adv_cw, l2_cw = cw_attack.attack(test_data[0])
# cw_y, cw_loss, cw_avg, cw_cc = model.eval([adv_cw, test_data[1]])
#
# # ———————————————————————— IFGSM attack ———————————————————————— #
# asr = 0
# for i in range(1, 30):
#     eps = i * 0.001
#     least_like_class = methods.LeastLikeClass(sess=sess,
#                                               model=model,
#                                               target=0.2,
#                                               eps=eps,
#                                               alpha=0.001,
#                                               iteration=25)
#     adv_l, l2_l = least_like_class.attack([test_data[0], c_y.reshape(test_data[1].shape)])
#     l_y, l_loss, l_avg, l_cc = model.eval([adv_l, test_data[1]])
#     if success_rate(c_y, l_y, 0.2) - asr < 0.001 and success_rate(c_y, l_y, 0.2) > 0.9:
#         break
#     else:
#         asr = success_rate(c_y, l_y, 0.2)
#
# ASR_cw = success_rate(c_y, cw_y, 0.2)
# ASR_ifgsm = success_rate(c_y, l_y, 0.2)
# print('ASR_C&W = {}, ASR_IFGSM = {}'.format(ASR_cw, ASR_ifgsm))
#
# # visualization the adversarial example
# plot_raw(test_data[0][10][:, :1250], adv_l[10][:, :1250], ckpt_dir + '/diff_fgsm_driving')
# plot_raw(test_data[0][10][:, :1250], adv_cw[10][:, :1250], ckpt_dir + '/diff_cw_driving')
# np.savez(ckpt_dir + '/adv_dnn_driving.npz', adv_cw=adv_cw, adv_ifgsm=adv_l, clean=test_data[0], y_cw=cw_y, y_ifgsm=l_y,
#          y_clean=c_y)
