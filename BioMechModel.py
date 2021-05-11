import numpy as np
import pandas as pd
from biomechanical_algorithms import *
from bilateral_filters import *
import scipy.stats as stats
import time
from sklearn.metrics import mean_squared_error, r2_score


tests = ['YY/test5', 'LRL/test1', 'LFC/test5', 'YY_L/test5', 'FCL/test1']
preds = ['YY/test5/test5', 'LRL/test1/test1', 'LFC/test5/test5', 'YY_L/test5/test5', 'FCL/test1/test1']


for ind, item in enumerate(tests):

    recons = './IsometricData/' + item + '/recons.xlsx'
    pred_ = './IsometricData/' + preds[ind] + 'result_with_Fpe_cost_01_modified_muscledata.xlsx'
    df = pd.read_excel(recons)
    try:
        bi_emg = df['BIClong-EMG']
    except:
        print(recons, '\n', df.keys())
        time.sleep(10)
    bi_pre_df = pd.read_excel(pred_)
    bi_pre = list(bi_pre_df['BIClong'])
    bi_pre = prediction_emg_filter(bi_pre)
    bi_pre = normalization(bi_pre)
    bi_emg = normalization(bi_emg)
    bi_emg = emg_muscle_activation_interpretion(bi_emg, A=-2.5)
    res = stats.pearsonr(bi_pre, bi_emg)
    print('Pearson: ', res)
    r2 = r2_score(bi_pre, bi_emg)
    print('R-squared: %f\n\n' % r2)
    fig_num = 511 + ind
    plt.subplot(fig_num)
    plt.plot(bi_emg, label=' ')
    plt.plot(bi_pre, label=' ', linestyle='--')

plt.xlabel('Frame number')
plt.ylabel('Normalized muscle activation level')
plt.legend(loc='upper left')
plt.show()









