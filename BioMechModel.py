import numpy as np
import pandas as pd
from biomechanical_algorithms import *
from bilateral_filters import *
import scipy.stats as stats


tests = ['YY/test5', 'LRL/test1', 'LFC/test5']
preds = ['YY/test5//test5', 'LRL/test1/test1', 'LFC/test5/test5']

for ind, item in enumerate(tests):
    recons = './IsometricData/' + item + '/recons.xlsx'
    pred_  = './IsometricData/' + preds[ind] + 'result_with_Fpe.xlsx'
    df = pd.read_excel(recons, skiprows=range(0, 1))
    bi_emg = df['BIClong-EMG']
    bi_pre_df = pd.read_excel(pred_)
    bi_pre = list(bi_pre_df['BIClong'])
    bi_pre = prediction_emg_filter(bi_pre)
    bi_pre = normalization(bi_pre)
    bi_emg = normalization(bi_emg)
    bi_emg = emg_muscle_activation_interpretion(bi_emg, A=-2.5)
    res = stats.pearsonr(bi_pre, bi_emg)
    print('Pearson: ', res)
    fig_num = 311 + ind
    plt.subplot(fig_num)
    plt.plot(bi_emg, label=' ')
    plt.plot(bi_pre, label=' ', linestyle='--')

plt.legend(loc='upper left')
plt.show()









