import numpy as np
import tensorflow as tf
import os
from lib import data_loader_driving, data_split
from sklearn.linear_model import Ridge
from Driving_models import DensNet, TFRidge
import methods

os.environ['CUDA_VISIBLE_DEVICES'] = '6'
e_setting = 'cross' # within or cross
model_name = 'ridge' # ridge or dnn
ckpt_dir = 'runs/' + model_name + '_attack'

clean = []
cw = []
llc = []
noise = []


def success_rate(y, a_y, target):
    rate = 0
    for i in range(len(y)):
        if a_y[i] > y[i] + target:
            rate += 1
    return rate / len(y)


x, y = data_loader_driving('/mnt/disk1/Dataset_Driving/2017_DL_Driving_250Hz.hdf5')
# ------------------------------
for source_number in range(len(x)):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    if e_setting == 'within':
        # within-subject
        # eeg = x[source_number]
        # label = y[source_number]
        eeg = x[source_number]
        label = y[source_number]

        # normalize
        for i in range(eeg.shape[0]):
            for j in range(eeg.shape[1]):
                max = np.max(eeg[i][j])
                min = np.min(eeg[i][j])
                eeg[i][j] = (eeg[i][j] - min) / (max - min)

        dataset = data_split(eeg, label)
        train_data = dataset['train']
        test_data = dataset['test']

    else:
        # cross-subject
        temp_x = np.concatenate([x[:source_number], x[source_number + 1:]], axis=0)
        temp_y = np.concatenate([y[:source_number], y[source_number + 1:]], axis=0)
        train_x = temp_x[0][:200]
        train_y = temp_y[0][:200]
        for i in range(1, len(temp_x)):
            train_x = np.concatenate([train_x, temp_x[i][:200]], axis=0)
            train_y = np.concatenate([train_y, temp_y[i][:200]], axis=0)

        test_x = x[source_number]
        test_y = y[source_number]

        # normalize
        for i in range(train_x.shape[0]):
            for j in range(train_x.shape[1]):
                max = np.max(train_x[i][j])
                min = np.min(train_x[i][j])
                train_x[i][j] = (train_x[i][j] - min) / (max - min)
        for i in range(test_x.shape[0]):
            for j in range(test_x.shape[1]):
                max = np.max(test_x[i][j])
                min = np.min(test_x[i][j])
                test_x[i][j] = (test_x[i][j] - min) / (max - min)

        train_data = [train_x, train_y]
        test_data = [test_x[:200], test_y[:200]]

    sess = tf.Session()

    if model_name == 'ridge':
        # ———————————————————————— using ridge model ———————————————————————— #
        # compute the PSD features
        # theta band 4-7.5 Hz; Alpha band 7.5-12 Hz
        # Sampling rate 250 Hz
        channels = [0, 1, 4, 5, 6, 7, 10, 11, 16, 17, 19, 20, 21, 25, 32, 33, 36, 37, 38, 39, 42, 43, 44, 45, 46,
                    47, 48, 49,
                    52, 53, 54, 55, 58, 59]
        pvt_fft = np.fft.fftn(train_data[0], axes=[2])
        pvt_psd = np.real(pvt_fft * np.conj(pvt_fft)) / 7500
        pvt_psd = pvt_psd[:, :, :3750]
        pvt_feature = np.mean(pvt_psd[:, :, 60:115], axis=2, keepdims=True)
        pvt_feature = np.concatenate([pvt_feature, np.mean(pvt_psd[:, :, 116:180], axis=2, keepdims=True)], axis=2)
        pvt_feature = np.reshape(pvt_feature, [-1, 30 * 2])
        pvt_feature = pvt_feature[:, channels]

        # get w and b
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
                        model_path=ckpt_dir + '/model_driving_' + e_setting + '/subject_' + str(source_number))

        # train DNN
        print('train dnn')
        model.train(train_data)
        print('dnn eval')
        c_y, c_loss, c_avg, c_cc = model.eval(test_data)

    # ———————————————————————— C&W attack ———————————————————————— #
    cw_attack = methods.CWL2(sess=sess,
                             model=model,
                             initial_c=0.01,
                             batch_size=20,
                             learning_rate=1e-3,
                             target=0.2,
                             binary_search_step=9,
                             input_shape=[train_data[0].shape[1], train_data[0].shape[2]],
                             max_iteration=1000)

    adv_cw, l2_cw = cw_attack.attack(test_data[0])
    print('cw attack success')
    cw_y, cw_loss, cw_avg, cw_cc = model.eval([adv_cw, test_data[1]])
    print('eval success')

    # ———————————————————————— IFGSM attack ———————————————————————— #
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

    # ———————————————————————— Gaussian noise ———————————————————————— #
    amplitude = 0.0
    for i in range(1000):
        amplitude = i * 0.0001
        mu = 0
        sigma = 1
        random_noise = amplitude * np.random.normal(mu, sigma, size=train_data[0].shape[2])
        l2_noise = np.sqrt(np.sum(np.square(random_noise))) * train_data[0].shape[1]
        if l2_noise >= l2_l: break

    print('amplitude = {}'.format(amplitude))
    noise_data = test_data[0] + random_noise
    n_y, n_loss, n_avg, n_cc = model.eval([noise_data, test_data[1]])

    clean.append([c_loss, c_avg, c_cc[0][1], np.mean(test_data[1])])
    noise.append([n_loss, n_avg, n_cc[0][1], l2_noise, amplitude])
    cw.append([cw_loss, cw_avg, cw_cc[0][1], l2_cw, success_rate(c_y, cw_y, target=0.2)])
    llc.append([l_loss, l_avg, l_cc[0][1], l2_l, success_rate(c_y, l_y, target=0.2)])
    tf.reset_default_graph()

if e_setting == 'within':
    np.savez(ckpt_dir + '/driving_attack_result', clean=np.asarray(clean), noise=noise, cw=np.asarray(cw),
             llc=np.asarray(llc))
else:
    np.savez(ckpt_dir + '/driving_attack_result_cross', clean=np.asarray(clean), noise=noise, cw=np.asarray(cw),
             llc=np.asarray(llc))
