import numpy as np
import tensorflow as tf
import os
from lib import data_loader_pvt, data_split
from sklearn.linear_model import Ridge
from PVT_models import TFRidge
import methods

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
ckpt_dir = 'runs/influence_of_parameters'
source_number = 0


def success_rate(y, a_y, target):
    rate = 0
    for i in range(len(y)):
        if a_y[i] > y[i] + target:
            rate += 1
    return rate / len(y)


# ------------------------------
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

# PVT dataset
pvt_data = data_loader_pvt('/mnt/disk2/mlb/dataset/PVT', npz=False)
dist = []
pre = []
pre_adv = []
success = []
cw = []
ifgsm = []
cy = []
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

for source_number in range(len(pvt_data[0])):
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
    test_data[0] = test_data[0][:45]
    test_data[1] = test_data[1][:45]

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

    ridge = Ridge(alpha=0.1)
    ridge.fit(pvt_feature, train_data[1])
    w, b = ridge.coef_.T, ridge.intercept_
    model = TFRidge(sess)
    model.set_weight(w, b)
    # eval
    c_y, c_loss, c_avg, c_cc = model.eval(test_data)

    # ——————————————————————————————————— different target —————————————————————————————————————————— #
    t = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    cw_temp = []
    ifgsm_temp = []
    for target in t:
        # ———————————————————————— C&W attack ———————————————————————— #
        cw_attack = methods.CWL2(sess=sess,
                                 model=model,
                                 initial_c=0.01,
                                 batch_size=20,
                                 learning_rate=5e-4,
                                 target=target,
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
                                                      target=target,
                                                      eps=eps,
                                                      alpha=0.001,
                                                      iteration=25)
            adv_l, l2_l = least_like_class.attack([test_data[0], c_y.reshape(test_data[1].shape)])
            l_y, l_loss, l_avg, l_cc = model.eval([adv_l, test_data[1]])
            if success_rate(c_y, l_y, 0.2) - asr < 0.001 and success_rate(c_y, l_y, 0.2) > 0.9:
                break
            else:
                asr = success_rate(c_y, l_y, 0.2)

        cw_temp.append(cw_y)
        ifgsm_temp.append(l_y)

    tf.reset_default_graph()
    cw.append(cw_temp)
    ifgsm.append(ifgsm_temp)
    cy.append(c_y)

cw = np.mean(np.asarray(cw), axis=0, keepdims=0)
ifgsm = np.mean(np.asarray(ifgsm), axis=0, keepdims=0)
cy = np.mean(np.asarray(cy), axis=0, keepdims=0)
# save c&w result
r = np.argsort(cy.reshape([-1]))
y = [a[r] for a in cw]
y.append(cy[r])
np.savez(ckpt_dir + '/predict_ridge_cw_diff_target.npz', y=y)

# save ifgm result
y = [a[r] for a in ifgsm]
labels = ['t = ' + str(i) for i in t]
y.append(cy[r])
np.savez(ckpt_dir + '/predict_ridge_ifgsm_diff_target.npz', y=y)
