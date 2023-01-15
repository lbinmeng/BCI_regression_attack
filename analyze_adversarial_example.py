import os
import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy.io import loadmat
from tqdm import tqdm

data_dir = 'runs/attack/'
# data_name = 'adv_dnn_pvt.npz'
data_name = 'adv_dnn_pvt.npz'
save_dir = 'runs/analyze_adv_examples/'
dataset = 'pvt'

wave_name = 'cgau8'
sampling_rate = 256

fc = pywt.central_frequency(wave_name)
cparam = 2 * fc * sampling_rate
scales = cparam / np.arange(sampling_rate, 1, -1)

# analyze_data = np.load(data_dir + data_name)
# adv_cw_data = analyze_data['adv_cw']
# adv_ifgsm_data = analyze_data['adv_ifgsm']
# clean_data = analyze_data['clean']
# adv_cw_label = analyze_data['y_cw']
# adv_ifgsm_label = analyze_data['y_ifgsm']
# clean_label = analyze_data['y_clean']
#
# grad_cw_all = adv_cw_data - clean_data
# grad_ifgsm_all = adv_ifgsm_data - clean_data
#
# success_id_cw = np.argwhere(adv_cw_label >= clean_label + 0.2).squeeze()
# success_id_ifgsm = np.argwhere(adv_ifgsm_label >= clean_label + 0.2).squeeze()
#
# channel = 27
#
# # clean data analyze
# clean_f = []
# for i in tqdm(range(len(clean_data))):
#     coef, freqs = pywt.cwt(clean_data[i, channel, :], scales, wave_name, 1.0 / sampling_rate)
#     clean_f.append(coef)
#
# # CW method analyze
# adv_cw = adv_cw_data[success_id_cw]
# grad_cw = grad_cw_all[success_id_cw]
# adv_cw_f = []
# grad_cw_f = []
#
# for i in tqdm(range(len(adv_cw))):
#     coef, freqs = pywt.cwt(adv_cw[i, channel, :], scales, wave_name, 1.0 / sampling_rate)
#     coef1, freqs1 = pywt.cwt(grad_cw[i, channel, :], scales, wave_name, 1.0 / sampling_rate)
#     adv_cw_f.append(coef)
#     grad_cw_f.append(coef1)
#     if i == 0: np.savez(save_dir + dataset+'_freqs.npz', freqs=freqs)
#
# # ifgsm method analyze
# adv_ifgsm = adv_ifgsm_data[success_id_ifgsm]
# grad_ifgsm = grad_ifgsm_all[success_id_ifgsm]
# adv_ifgsm_f = []
# grad_ifgsm_f = []
#
# for i in tqdm(range(len(adv_ifgsm))):
#     coef, freqs = pywt.cwt(adv_ifgsm[i, channel, :], scales, wave_name, 1.0 / sampling_rate)
#     coef1, freqs1 = pywt.cwt(grad_ifgsm[i, channel, :], scales, wave_name, 1.0 / sampling_rate)
#     adv_ifgsm_f.append(coef)
#     grad_ifgsm_f.append(coef1)
#
# np.savez(save_dir + dataset+'_analyze.npz',
#          adv_cw_f=adv_cw_f,
#          grad_cw_f=grad_cw_f,
#          adv_ifgsm_f=adv_ifgsm_f,
#          grad_ifgsm_f=grad_ifgsm_f,
#          clean_f=clean_f
#          )


# plot the Time frequency analysis
data = np.load(save_dir + dataset + '_analyze.npz')
freqs = np.load(save_dir + dataset + '_freqs.npz')['freqs']

adv_cw_f = data['adv_cw_f']
grad_cw_f = data['grad_cw_f']
adv_ifgsm_f = data['adv_ifgsm_f']
grad_ifgsm_f = data['grad_ifgsm_f']
clean_f = data['clean_f']

t = np.arange(0, np.asarray(adv_cw_f).shape[2]) * 1.0 / sampling_rate

adv_cw_f = np.abs(adv_cw_f)
grad_cw_f = np.abs(grad_cw_f)
adv_ifgsm_f = np.abs(adv_ifgsm_f)
grad_ifgsm_f = np.abs(grad_ifgsm_f)
clean_f = np.abs(clean_f)

adv_cw_f = np.mean(adv_cw_f, axis=0).squeeze()
grad_cw_f = np.mean(grad_cw_f, axis=0).squeeze()
adv_ifgsm_f = np.mean(adv_ifgsm_f, axis=0).squeeze()
grad_ifgsm_f = np.mean(grad_ifgsm_f, axis=0).squeeze()
clean_f = np.mean(clean_f, axis=0).squeeze()

t_fontsize = 12
l_fontsize = 11
labelsize = 10

plt.figure(figsize=(12, 3))
plt.subplot(1, 3, 1)
plt.pcolormesh(t, freqs[-50:], clean_f[-50:, :], shading='gouraud')
plt.ylabel('Frequency (Hz)', fontsize=l_fontsize)
plt.xlabel('Time (s)', fontsize=l_fontsize)
# plt.clim(-0.2, 0.7)
plt.tick_params(labelsize=labelsize)
cb2=plt.colorbar()
cb2.ax.tick_params(labelsize=labelsize)
plt.title('Original EEG trial', fontsize=t_fontsize)

plt.subplot(1, 3, 2)
plt.pcolormesh(t, freqs[-50:], adv_cw_f[-50:, :], shading='gouraud')
plt.ylabel('Frequency (Hz)', fontsize=l_fontsize)
plt.xlabel('Time (s)', fontsize=l_fontsize)
# plt.clim(-0.2, 0.7)
plt.tick_params(labelsize=labelsize)
cb2=plt.colorbar()
cb2.ax.tick_params(labelsize=labelsize)
plt.title('Adversarial trial from CW-P', fontsize=t_fontsize)

plt.subplot(1, 3, 3)
plt.pcolormesh(t, freqs[-50:], grad_cw_f[-50:, :], shading='gouraud')
plt.ylabel('Frequency (Hz)', fontsize=l_fontsize)
plt.xlabel('Time (s)', fontsize=l_fontsize)
# plt.clim(-0.2, 0.7)
plt.tick_params(labelsize=labelsize)
cb2=plt.colorbar()
cb2.ax.tick_params(labelsize=labelsize)
plt.title('Perturbation from CW-P', fontsize=t_fontsize)

fig = plt.gcf()
fig.tight_layout()
plt.subplots_adjust(wspace=0.25, hspace=0)  # 调整子图间距
fig.savefig(save_dir + '{}_Spec_cw.jpg'.format(dataset), format='jpg')

# plot ifgsm
plt.figure(figsize=(12, 3))
plt.subplot(1, 3, 1)
plt.pcolormesh(t, freqs[-50:], clean_f[-50:, :], shading='gouraud')
plt.ylabel('Frequency (Hz)', fontsize=l_fontsize)
plt.xlabel('Time (s)', fontsize=l_fontsize)
plt.tick_params(labelsize=labelsize)
# plt.clim(-0.2, 0.7)
cb2 = plt.colorbar()
cb2.ax.tick_params(labelsize=labelsize)
plt.title('Original EEG trial', fontsize=t_fontsize)

plt.subplot(1, 3, 2)
plt.pcolormesh(t, freqs[-50:], adv_ifgsm_f[-50:, :], shading='gouraud')
plt.ylabel('Frequency (Hz)', fontsize=l_fontsize)
plt.xlabel('Time (s)', fontsize=l_fontsize)
plt.tick_params(labelsize=labelsize)
# plt.clim(-0.2, 0.7)
cb2 = plt.colorbar()
cb2.ax.tick_params(labelsize=labelsize)
plt.title('Adversarial trial from IFGSM-P', fontsize=t_fontsize)

plt.subplot(1, 3, 3)
plt.pcolormesh(t, freqs[-50:], grad_ifgsm_f[-50:, :], shading='gouraud')
plt.ylabel('Frequency (Hz)', fontsize=l_fontsize)
plt.xlabel('Time (s)', fontsize=l_fontsize)
plt.tick_params(labelsize=labelsize)
# plt.clim(-0.2, 0.7)
cb2 = plt.colorbar()
cb2.ax.tick_params(labelsize=labelsize)
plt.title('Perturbation from IFGSM-P', fontsize=t_fontsize)

fig = plt.gcf()
fig.tight_layout()
plt.subplots_adjust(wspace=0.22, hspace=0)  # 调整子图间距
fig.savefig(save_dir + '{}_Spec_ifgsm.jpg'.format(dataset), format='jpg')
plt.show()
