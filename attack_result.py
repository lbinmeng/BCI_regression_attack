import numpy as np

# show the result of the attack on driving and PVT
models = ['ridge', 'dnn']
e_setting = ['within', 'cross']
datasets = ['driving', 'PVT']

for setting in e_setting:
    for dataset in datasets:
        for model in models:
            if setting == 'within':
                result = np.load('runs/' + model + '_attack/' + dataset + '_attack_result.npz')
            else:
                result = np.load('runs/' + model + '_attack/' + dataset + '_attack_result_cross.npz')

            clean = result['clean']
            cw = result['cw']
            llc = result['llc']
            noise = result['noise']

            clean = np.mean(clean, axis=0)
            cw = np.mean(cw, axis=0)
            llc = np.mean(llc, axis=0)
            noise = np.mean(noise, axis=0)

            print(
                '+++++++++++ attack ' + model + ' model trained on ' + dataset + ' in ' + setting
                + '-sbuject setting +++++++++++')
            print('clean rmse = {}  pre_avg = {} cc = {}'.format(clean[0], clean[1], clean[2]))
            print('noise rmse = {}  pre_avg = {} cc = {} l2 = {}, am = {}'.format(noise[0], noise[1], noise[2], noise[3],
                                                                                  noise[4]))
            print('cw rmse = {}  pre_avg = {} cc = {} l2 = {} success_rate = {}'.format(cw[0], cw[1], cw[2], cw[3], cw[4]))
            print('llc rmse = {} pre_avg = {} cc = {} l2 = {} success_rate = {}'.format(llc[0], llc[1], llc[2], llc[3],
                                                                                        llc[4]))
            print('\n\n')
