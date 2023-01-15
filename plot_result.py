import numpy as np
from lib import plot_raw
import matplotlib.pyplot as plt
from lib import show_predict

# # visualization pvt_attack
# y = np.load('runs/attack' + '/adv_dnn_pvt.npz')
# plot_raw(y['clean'][1], y['adv_ifgsm'][1], 'runs/attack' + '/diff_fgsm_pvt')
# plot_raw(y['clean'][1], y['adv_cw'][1], 'runs/attack' + '/diff_cw_pvt')
#
# # visualization driving_attack
# y = np.load('runs/attack' + '/adv_dnn_driving.npz')
# plot_raw(y['clean'][4][:, :800], y['adv_ifgsm'][4][:, :800], 'runs/attack' + '/diff_fgsm_driving')
# plot_raw(y['clean'][4][:, :800], y['adv_cw'][4][:, :800], 'runs/attack' + '/diff_cw_driving')
#
# # different c for cw
# y = np.load('runs/influence_of_parameters' + '/cw_constant.npz')['y']
# x = np.arange(len(y[0])) * 0.1 + 0.5
#
# plt.figure(figsize=(8, 6))
# plt.plot(x, y[3], 'r', linewidth=3.0, label='distortion')
# plt.ylabel('Mean Distortion', fontsize=24)
# plt.xlabel(r'$c$', fontsize=24)
# # plt.legend(loc='lower right', fontsize=14)
# plt.tick_params(labelsize=20)
# plt.tight_layout()
# plt.savefig('runs/influence_of_parameters' + '/cw_constant_distortion' + '.eps')
#
# plt.figure(figsize=(8, 6))
# plt.plot(x, y[0], 'b', linewidth=3.0, label='ASR')
# plt.ylabel('ASR', fontsize=24)
# plt.xlabel(r'$c$', fontsize=24)
# # plt.legend(loc='lower right', fontsize=14)
# plt.tick_params(labelsize=20)
# plt.tight_layout()
# plt.savefig('runs/influence_of_parameters' + '/cw_constant_ASR' + '.eps')
#
# # different eps for ifgsm
# y = np.load('runs/influence_of_parameters' + '/ifgsm_eps.npz')['y']
# x = np.arange(len(y[0])) * 0.001
#
# plt.figure(figsize=(8, 6))
# plt.plot(x, y[0], 'b', linewidth=3.0, label='ASR')
# plt.ylabel('ASR', fontsize=24)
# plt.xlabel(r'$\epsilon$', fontsize=24)
# # plt.legend(loc='lower right', fontsize=14)
# plt.tick_params(labelsize=20)
# plt.tight_layout()
# plt.savefig('runs/influence_of_parameters' + '/ifgsm_eps_ASR' + '.eps')
#
# plt.figure(figsize=(8, 6))
# plt.plot(x, y[3], 'r', linewidth=3.0, label='distortion')
# plt.ylabel('Mean Distortion', fontsize=24)
# plt.xlabel(r'$\epsilon$', fontsize=24)
# # plt.legend(loc='lower right', fontsize=14)
# plt.tick_params(labelsize=20)
# plt.tight_layout()
# plt.savefig('runs/influence_of_parameters' + '/ifgsm_eps_distortion' + '.eps')
#
# # different iteration for ifgsm
# y = np.load('runs/influence_of_parameters' + '/ifgsm_iteration.npz')['y']
# x = np.arange(len(y[0]))
#
# plt.figure(figsize=(8, 6))
# plt.plot(x, y[3], 'r', linewidth=3.0, label='distortion')
# plt.ylabel('Mean Distortion', fontsize=24)
# plt.xlabel(r'$M$', fontsize=24)
# # plt.legend(loc='lower right', fontsize=14)
# plt.tick_params(labelsize=20)
# plt.tight_layout()
# plt.savefig('runs/influence_of_parameters' + '/ifgsm_iteration_distortion' + '.eps')
#
# plt.figure(figsize=(8, 6))
# plt.plot(x, y[0], 'b', linewidth=3.0, label='ASR')
# plt.ylabel('ASR', fontsize=24)
# plt.xlabel(r'$M$', fontsize=24)
# # plt.legend(loc='lower right', fontsize=14)
# plt.tick_params(labelsize=20)
# plt.tight_layout()
# plt.savefig('runs/influence_of_parameters' + '/ifgsm_iteration_ASR' + '.eps')
#
# # different target value
# t = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
# labels = ['t = ' + str(i) for i in t]
# labels.append('clean')
# y = np.load('runs/influence_of_parameters' + '/predict_ridge_cw_diff_target.npz')['y']
# show_predict(y, labels, '../predict_diff_target')

# transferability
# MLP to Ridge
y = np.load('runs/transferability' + '/dnn2ridge.npz')['y']
c_y = y[0]
cw_dnn = y[1]
ifgsm_dnn = y[2]
cw_ridge = y[3]
ifgsm_ridge = y[4]

x = np.arange(len(c_y))
plt.figure()
plt.plot(x, c_y, label='Clean')
plt.plot(x, cw_dnn, label='CW-P,MLP')
plt.plot(x, ifgsm_dnn, label='IFGSM-P,MLP')
plt.plot(x, cw_ridge, ':', label='CW-P,RR')
plt.plot(x, ifgsm_ridge, '--', label='IFGSM-P,RR')
plt.ylabel('Output', fontsize=16)
plt.xlabel('Sample', fontsize=16)
plt.legend(loc='lower right', ncol=2, fontsize=10)
plt.tick_params(labelsize=13)
plt.savefig('runs/transferability' + '/dnn2ridge.eps')
plt.show()

# MLP to Ridge
y = np.load('runs/transferability' + '/ridge2dnn.npz')['y']

c_y = y[0]
cw_ridge = y[1]
ifgsm_ridge = y[2]
cw_dnn = y[3]
ifgsm_dnn = y[4]

x = np.arange(len(c_y))
plt.figure()
plt.plot(x, c_y, label='Clean')
plt.plot(x, cw_ridge, label='CW-P,RR')
plt.plot(x, ifgsm_ridge, label='IFGSM-P,RR')
plt.plot(x, cw_dnn, ':', label='CW-P,MLP')
plt.plot(x, ifgsm_dnn, '--', label='IFGSM-P,MLP')
plt.ylabel('Output', fontsize=16)
plt.xlabel('Sample', fontsize=16)
plt.legend(loc='lower right', ncol=2, fontsize=10)
plt.tick_params(labelsize=13)
plt.savefig('runs/transferability' + '/ridge2dnn.eps')
plt.show()
