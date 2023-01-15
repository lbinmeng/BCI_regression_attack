import numpy as np
import tensorflow as tf
import os
from lib import data_loader_pvt, data_split
from sklearn.linear_model import Ridge
from PVT_models import DensNet, TFRidge
import methods

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
ckpt_dir = 'runs/transferability'
transfer = 'dnn2ridge'  # dnn2ridge or ridge2dnn

if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)


def success_rate(y, a_y, target):
    rate = 0
    for i in range(len(y)):
        if a_y[i] > y[i] + target:
            rate += 1
    return rate / len(y)


c_y_all = []
cw_source_all = []
ifgsm_source_all = []
cw_target_all = []
ifgsm_target_all = []

# ------------------------------
pvt_data = data_loader_pvt('/mnt/disk2/mlb/dataset/PVT', npz=False)

for source_number in range(1):
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

    # get w and b
    pvt_fft = np.fft.fftn(train_data[0], axes=[2])
    pvt_psd = np.real(pvt_fft * np.conj(pvt_fft)) / 1280
    pvt_psd = pvt_psd[:, :, :640]
    pvt_feature = np.mean(pvt_psd[:, :, 20:40], axis=2, keepdims=True)
    pvt_feature = np.concatenate([pvt_feature, np.mean(pvt_psd[:, :, 40:65], axis=2, keepdims=True)], axis=2)
    pvt_feature = np.reshape(pvt_feature, [-1, 62 * 2])

    r = Ridge(alpha=0.1)
    r.fit(pvt_feature, train_data[1])
    w, b = r.coef_.T, r.intercept_

    if transfer == 'dnn2ridge':
        source_model = DensNet(sess=sess, input_shape=[train_data[0].shape[1], train_data[0].shape[2]],
                               model_path=ckpt_dir + '/model_pvt_' + transfer)
        source_model.train(train_data)
        target_model = TFRidge(sess)
        target_model.set_weight(w, b)
    else:
        source_model = TFRidge(sess)
        source_model.set_weight(w, b)
        target_model = DensNet(sess=sess, input_shape=[train_data[0].shape[1], train_data[0].shape[2]],
                               model_path=ckpt_dir + '/model_pvt_' + transfer)
        target_model.train(train_data)

    c_y, c_loss, c_avg, c_cc = source_model.eval(test_data)

    cw_attack = methods.CWL2(sess=sess,
                             model=source_model,
                             initial_c=0.01,
                             batch_size=20,
                             learning_rate=5e-4,
                             target=0.2,
                             binary_search_step=9,
                             input_shape=[train_data[0].shape[1], train_data[0].shape[2]],
                             max_iteration=1000)

    adv_cw, l2_cw = cw_attack.attack(test_data[0])

    asr = 0
    for i in range(1, 30):
        eps = i * 0.001
        least_like_class = methods.LeastLikeClass(sess=sess,
                                                  model=source_model,
                                                  target=0.2,
                                                  eps=eps,
                                                  alpha=0.001,
                                                  iteration=25)

        adv_l, l2_l = least_like_class.attack([test_data[0], c_y.reshape(test_data[1].shape)])
        l_y, l_loss, l_avg, l_cc = source_model.eval([adv_l, test_data[1]])
        if success_rate(c_y, l_y, 0.2) - asr < 0.001 and success_rate(c_y, l_y, 0.2) > 0.9:
            break
        else:
            asr = success_rate(c_y, l_y, 0.2)

    # source
    print('cw adversarial examples eval in source model')
    cw_source, _, _, _ = source_model.eval([adv_cw, test_data[1]])
    print('ifgsm adversarial examples eval in source model')
    ifgsm_source, _, _, _ = source_model.eval([adv_l, test_data[1]])
    # target
    print('cw adversarial examples eval in target model')
    cw_target, _, _, _ = target_model.eval([adv_cw, test_data[1]])
    print('ifgsm adversarial examples eval in target model')
    ifgsm_target, _, _, _ = target_model.eval([adv_l, test_data[1]])

    c_y_all.append(c_y)
    cw_source_all.append(cw_source)
    ifgsm_source_all.append(ifgsm_source)
    cw_target_all.append(cw_target)
    ifgsm_target_all.append(ifgsm_target)
    tf.reset_default_graph()

c_y_all = np.mean(np.asarray(c_y_all), axis=0, keepdims=0)
cw_source_all = np.mean(np.asarray(cw_source_all), axis=0, keepdims=0)
ifgsm_source_all = np.mean(np.asarray(ifgsm_source_all), axis=0, keepdims=0)
cw_target_all = np.mean(np.asarray(cw_target_all), axis=0, keepdims=0)
ifgsm_target_all = np.mean(np.asarray(ifgsm_target_all), axis=0, keepdims=0)

r = np.argsort(c_y_all.reshape([-1]))
y = [c_y_all[r], cw_source_all[r], ifgsm_source_all[r], cw_target_all[r], ifgsm_target_all[r]]
if transfer == 'dnn2ridge':
    np.savez(ckpt_dir + '/dnn2ridge.npz', y=y)
else:
    np.savez(ckpt_dir + '/ridge2dnn.npz', y=y)
